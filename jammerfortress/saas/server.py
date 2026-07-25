"""
Production HTTP server: landing site, TRI-SKIN web console, JSON API, live
WebSocket event stream, Prometheus metrics, Stripe billing. Pure stdlib.

Hardening: bearer-token auth (no cookies => no CSRF surface), tokens hashed at
rest, per-IP token-bucket rate limiting, 64KB body cap, strict security
headers + CSP, explicit route map (no path traversal possible), atomic state
persistence, graceful shutdown, per-user session isolation inside the kernel.
"""
import argparse
import json
import os
import secrets
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .. import wire
from ..connectors import status as connector_status
from ..kernel import Kernel
from . import auth, billing
from .db import DB

MAX_BODY = 65536
RATE_LIMIT_PER_MIN = 120
USER_COMMANDS = {"record", "recall", "query", "route", "audit", "trust",
                 "intent", "stats", "menu", "persist",
                 "sh0w", "s3m4", "p1ck", "m1rr", "l0ck", "st4y", "m3nu", "st4t", "1nt3nt"}
WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")


class FortressServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, data_dir):
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        self.db = DB(os.path.join(data_dir, "fortress.db"))
        self.kernel = Kernel(secret=self._secret(data_dir),
                             state_path=os.path.join(data_dir, "fortress_state.json"))
        self.kernel.boot(role="system_server", user="saas")
        self.rate = {}
        self.rate_lock = threading.Lock()
        self._stop_persist = threading.Event()
        self._persist_thread = threading.Thread(target=self._persist_loop, daemon=True)
        super().__init__(addr, Handler)
        self._persist_thread.start()

    @staticmethod
    def _secret(data_dir):
        env = os.environ.get("JF_SECRET")
        if env:
            return env
        keyfile = os.path.join(data_dir, "secret.key")
        if os.path.exists(keyfile):
            with open(keyfile) as f:
                return f.read().strip()
        secret = secrets.token_urlsafe(48)
        fd = os.open(keyfile, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secret)
        return secret

    def _persist_loop(self):
        while not self._stop_persist.wait(60):
            try:
                self.kernel.persist()
            except OSError:
                pass

    def rate_ok(self, ip):
        now = time.time()
        with self.rate_lock:
            count, start = self.rate.get(ip, (0, now))
            if now - start >= 60:
                count, start = 0, now
            count += 1
            self.rate[ip] = (count, start)
            if len(self.rate) > 10000:
                self.rate = {k: v for k, v in self.rate.items() if now - v[1] < 60}
            return count <= RATE_LIMIT_PER_MIN

    def shutdown_clean(self):
        self._stop_persist.set()
        try:
            self.kernel.persist()
        except OSError:
            pass
        self.shutdown()


class Handler(BaseHTTPRequestHandler):
    server_version = "JammerFortress/2.0"
    protocol_version = "HTTP/1.1"

    # ---- plumbing ----------------------------------------------------------
    def log_message(self, fmt, *args):  # keep stdout clean; audit lives in db
        pass

    def _headers_common(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                         "script-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                         "connect-src 'self' ws: wss:; frame-ancestors 'none'")

    def _send(self, code, body, ctype):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self._headers_common()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, default=str), "application/json")

    def _body(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None
        if length < 0 or length > MAX_BODY:
            return None
        return self.rfile.read(length) if length else b""

    def _json_body(self):
        raw = self._body()
        if raw is None:
            return None
        if not raw:
            return {}
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None

    def _bearer(self):
        h = self.headers.get("Authorization", "")
        if h.startswith("Bearer "):
            return h[7:].strip()
        return None

    def _user(self, token=None):
        raw = token or self._bearer()
        if not raw:
            return None
        th = auth.token_hash(raw)
        return (self.server.db.session_user(th)
                or self.server.db.user_by_api_key(th))

    def _base_url(self):
        proto = self.headers.get("X-Forwarded-Proto", "http")
        host = self.headers.get("Host", "localhost")
        return f"{proto}://{host}"

    def _static(self, name, ctype="text/html; charset=utf-8"):
        path = os.path.join(WEB_DIR, name)  # explicit names only; never from URL
        try:
            with open(path, "rb") as f:
                self._send(200, f.read(), ctype)
        except OSError:
            self._json(404, {"error": "asset missing"})

    def _gate(self):
        """Auth + rate limit + usage metering. Returns user dict or None (and
        has already written the error response)."""
        if not self.server.rate_ok(self.client_address[0]):
            self._json(429, {"error": "rate limit exceeded (120 req/min per IP)"})
            return None
        user = self._user()
        if not user:
            self._json(401, {"error": "authentication required"})
            return None
        plan, _ = billing.plan_for(self.server.db, user["id"])
        count, allowed = self.server.db.incr_usage(user["id"], billing.limit_for(plan))
        if not allowed:
            self._json(402, {"error": f"daily limit reached on the {plan} plan",
                             "used_today": count, "limit": billing.limit_for(plan),
                             "upgrade": "/api/plans"})
            return None
        user["plan"] = plan
        return user

    # ---- GET ---------------------------------------------------------------
    def do_GET(self):  # noqa: N802
        path, _, query = self.path.partition("?")
        qs = urllib.parse.parse_qs(query)
        if path == "/":
            return self._static("index.html")
        if path == "/console":
            return self._static("console.html")
        if path == "/healthz":
            return self._json(200, {"ok": True, "version": "FUSEPACK-2.0"})
        if path == "/readyz":
            ok = self.server.db.health()
            return self._json(200 if ok else 503, {"ready": ok})
        if path == "/metrics":
            return self._send(200, self.server.kernel.exo.telemetry.prometheus(),
                              "text/plain; version=0.0.4")
        if path == "/api/plans":
            return self._json(200, {"plans": billing.public_plans(),
                                    "billing_configured": connector_status()["stripe"]})
        if path == "/api/me":
            if not self.server.rate_ok(self.client_address[0]):
                return self._json(429, {"error": "rate limit exceeded"})
            user = self._user()
            if not user:
                return self._json(401, {"error": "authentication required"})
            plan, sub = billing.plan_for(self.server.db, user["id"])
            return self._json(200, {
                "email": user["email"], "plan": plan,
                "status": sub.get("status", "active"),
                "used_today": self.server.db.usage_today(user["id"]),
                "limit": billing.limit_for(plan),
                "connectors": connector_status(),
            })
        if path == "/api/qr":
            user = self._gate()
            if not user:
                return None
            text = (qs.get("text", [""])[0]).strip()
            if not text or len(text) > 1000:
                return self._json(400, {"error": "text must be 1-1000 characters"})
            dark = qs.get("dark", ["#000000"])[0]
            light = qs.get("light", ["#ffffff"])[0]
            shape = qs.get("shape", ["square"])[0]
            logo = qs.get("logo", ["0"])[0] == "1"
            if not all(_safe_color(c) for c in (dark, light)) or shape not in ("square", "dot"):
                return self._json(400, {"error": "invalid color or shape"})
            try:
                result = self.server.kernel.qr(text, dark=dark, light=light,
                                               shape=shape, logo_slot=0.18 if logo else 0.0)
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            return self._send(200, result["svg"], "image/svg+xml")
        if path == "/ws/events":
            return self._ws_events(qs)
        return self._json(404, {"error": "not found"})

    # ---- POST ----------------------------------------------------------------
    def do_POST(self):  # noqa: N802
        path = self.path.partition("?")[0]
        if path == "/api/billing/webhook":
            return self._webhook()
        if not self.server.rate_ok(self.client_address[0]):
            return self._json(429, {"error": "rate limit exceeded (120 req/min per IP)"})
        if path == "/api/signup":
            return self._signup()
        if path == "/api/login":
            return self._login()
        if path == "/api/logout":
            token = self._bearer()
            if token:
                self.server.db.delete_session(auth.token_hash(token))
            return self._json(200, {"ok": True})
        if path == "/api/command":
            return self._command()
        if path == "/api/intent":
            return self._intent()
        if path == "/api/billing/checkout":
            return self._checkout()
        if path == "/api/billing/portal":
            return self._portal()
        return self._json(404, {"error": "not found"})

    # ---- auth endpoints ---------------------------------------------------
    def _signup(self):
        body = self._json_body()
        if body is None:
            return self._json(400, {"error": "invalid JSON body (max 64KB)"})
        email = (body.get("email") or "").strip()
        pw = body.get("password") or ""
        if not auth.valid_email(email):
            return self._json(400, {"error": "enter a valid email address"})
        problem = auth.password_problem(pw)
        if problem:
            return self._json(400, {"error": problem})
        if self.server.db.user_by_email(email):
            return self._json(409, {"error": "an account with this email already exists"})
        uid = self.server.db.create_user(email, auth.hash_password(pw))
        token = auth.new_session_token()
        self.server.db.create_session(auth.token_hash(token), uid)
        self.server.db.log("signup", {"user": uid})
        return self._json(201, {"token": token, "email": email, "plan": "free"})

    def _login(self):
        body = self._json_body()
        if body is None:
            return self._json(400, {"error": "invalid JSON body (max 64KB)"})
        email = (body.get("email") or "").strip()
        pw = body.get("password") or ""
        user = self.server.db.user_by_email(email) if auth.valid_email(email) else None
        if not user or not auth.verify_password(pw, user["pw"]):
            time.sleep(0.3)  # flat cost for wrong email and wrong password alike
            return self._json(401, {"error": "invalid email or password"})
        token = auth.new_session_token()
        self.server.db.create_session(auth.token_hash(token), user["id"])
        self.server.db.log("login", {"user": user["id"]})
        plan, _ = billing.plan_for(self.server.db, user["id"])
        return self._json(200, {"token": token, "email": user["email"], "plan": plan})

    # ---- kernel endpoints ----------------------------------------------------
    def _command(self):
        user = self._gate()
        if not user:
            return None
        body = self._json_body()
        if body is None:
            return self._json(400, {"error": "invalid JSON body (max 64KB)"})
        cmd = (body.get("command") or "").strip()
        arg = body.get("arg")
        if arg is not None and not isinstance(arg, str):
            return self._json(400, {"error": "arg must be a string"})
        if cmd not in USER_COMMANDS:
            return self._json(400, {"error": f"unknown command '{cmd}'",
                                    "available": sorted(USER_COMMANDS)})
        try:
            result = self.server.kernel.command(cmd, arg, session_id=user["id"])
        except (ValueError, RuntimeError) as e:
            return self._json(400, {"error": str(e)})
        self.server.db.log("command", {"user": user["id"], "command": cmd})
        return self._json(200, {"command": cmd, "result": result})

    def _intent(self):
        user = self._gate()
        if not user:
            return None
        body = self._json_body()
        if body is None:
            return self._json(400, {"error": "invalid JSON body (max 64KB)"})
        text = (body.get("text") or "").strip()
        if not text or len(text) > 4000:
            return self._json(400, {"error": "text must be 1-4000 characters"})
        try:
            result = self.server.kernel.intent(text, session_id=user["id"])
        except (ValueError, RuntimeError) as e:
            return self._json(400, {"error": str(e)})
        self.server.db.log("intent", {"user": user["id"]})
        return self._json(200, result)

    # ---- billing endpoints -----------------------------------------------------
    def _checkout(self):
        user = self._user()
        if not user:
            return self._json(401, {"error": "authentication required"})
        body = self._json_body()
        if body is None:
            return self._json(400, {"error": "invalid JSON body"})
        plan = (body.get("plan") or "").strip()
        try:
            session = billing.create_checkout(self.server.db, user, plan, self._base_url())
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        except billing.BillingNotConfigured as e:
            return self._json(503, {"error": str(e)})
        except Exception as e:  # ConnectorError -> surface honestly
            return self._json(502, {"error": f"Stripe request failed: {e}"})
        return self._json(200, {"url": session.get("url"), "id": session.get("id")})

    def _portal(self):
        user = self._user()
        if not user:
            return self._json(401, {"error": "authentication required"})
        try:
            session = billing.create_portal(self.server.db, user, self._base_url())
        except billing.BillingNotConfigured as e:
            return self._json(503, {"error": str(e)})
        except Exception as e:
            return self._json(502, {"error": f"Stripe request failed: {e}"})
        return self._json(200, {"url": session.get("url")})

    def _webhook(self):
        secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
        raw = self._body()
        if raw is None:
            return self._json(400, {"error": "body too large or invalid"})
        if not secret:
            return self._json(503, {"error": "webhooks not configured: set STRIPE_WEBHOOK_SECRET"})
        sig = self.headers.get("Stripe-Signature", "")
        if not billing.verify_stripe_signature(raw, sig, secret):
            self.server.db.log("webhook_rejected", {"reason": "bad signature"})
            return self._json(400, {"error": "signature verification failed"})
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid JSON"})
        action = billing.handle_event(self.server.db, event)
        self.server.db.log("webhook", {"type": event.get("type"), "action": action})
        return self._json(200, {"received": True, "action": action})

    # ---- websocket event stream -------------------------------------------------
    def _ws_events(self, qs):
        token = qs.get("token", [""])[0]
        user = self._user(token=token)
        if not user:
            return self._json(401, {"error": "authentication required (pass ?token=)"})
        key = self.headers.get("Sec-WebSocket-Key")
        if not key or self.headers.get("Upgrade", "").lower() != "websocket":
            return self._json(400, {"error": "websocket upgrade required"})
        self.close_connection = True
        self.connection.sendall(wire.handshake_response(key))
        ws = wire.WebSocket(self.connection)
        events = self.server.kernel.exo.events
        cursor = len(events)
        ws.send_text(json.dumps({"channel": "events", "status": "connected"}))
        deadline = time.time() + 3600  # hard cap per connection
        while ws.open and time.time() < deadline:
            snapshot = list(events)
            while cursor < len(snapshot):
                try:
                    ws.send_text(json.dumps(snapshot[cursor], default=str))
                except RuntimeError:
                    return None
                cursor += 1
            cursor = min(cursor, len(snapshot))
            op, _payload = ws.recv(timeout=1.0)
            if op is None:
                return None
        ws.close()
        return None


def _safe_color(c):
    if not c or len(c) not in (4, 7) or c[0] != "#":
        return False
    try:
        int(c[1:], 16)
        return True
    except ValueError:
        return False


def create_server(host="127.0.0.1", port=8788, data_dir="fortress_data"):
    return FortressServer((host, port), data_dir)


def main(argv=None):
    # Platform-friendly defaults (Railway/Render/Fly/Heroku-style):
    # - PORT env var wins for the port; when set, bind 0.0.0.0 so the
    #   platform's edge proxy can reach the process.
    # - RAILWAY_VOLUME_MOUNT_PATH is auto-set when a Railway volume is
    #   attached; use it for the data dir so state survives redeploys.
    env_port = os.environ.get("PORT", "").strip()
    default_port = int(env_port) if env_port.isdigit() else 8788
    default_host = os.environ.get("JF_HOST") or ("0.0.0.0" if env_port else "127.0.0.1")
    default_data = (os.environ.get("JF_DATA_DIR")
                    or os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
                    or "fortress_data")
    p = argparse.ArgumentParser(prog="jammerfortress serve")
    p.add_argument("--host", default=default_host)
    p.add_argument("--port", type=int, default=default_port)
    p.add_argument("--data-dir", default=default_data)
    args = p.parse_args(argv)
    from .. import embeddings as _emb
    if os.environ.get("JF_EMBED_AUTO_FETCH") == "1" and _emb.find_model_dir() is None:
        from ..model_fetch import fetch
        target = os.path.join(args.data_dir, "models", "embedder")
        try:
            fetch(target)
            os.environ["JF_EMBED_MODEL_DIR"] = target
            _emb.reload()
            print("[embeddings] model fetched to %s" % target)
        except Exception as e:  # noqa: BLE001 - degrade loudly, keep serving
            print("[embeddings] auto-fetch failed (%s); continuing on the "
                  "hash backend" % e)
    server = create_server(args.host, args.port, args.data_dir)
    ei = _emb.info()
    print("[embeddings] backend=%s dim=%s" % (ei["backend"], ei.get("dim")))
    if ei.get("fallback_reason"):
        print("[embeddings] note: %s" % ei["fallback_reason"])
    print("[JAMMER-FORTRESS] data dir:   %s" % os.path.abspath(args.data_dir))
    if not os.environ.get("JF_DATA_DIR") and not os.environ.get("RAILWAY_VOLUME_MOUNT_PATH") and env_port:
        print("[storage] WARNING: no volume detected. Attach a volume (Railway: "
              "service -> Attach Volume) or set JF_DATA_DIR, or accounts/memory "
              "reset on every redeploy.")
    host, port = server.server_address[:2]
    print("[JAMMER-FORTRESS] serving on http://%s:%s" % (host, port))
    print("[JAMMER-FORTRESS] console:    http://%s:%s/console" % (host, port))
    if not os.environ.get("STRIPE_SECRET_KEY"):
        print("[billing] Stripe not configured (set STRIPE_SECRET_KEY, "
              "STRIPE_WEBHOOK_SECRET, STRIPE_PRICE_PRO, STRIPE_PRICE_FORTRESS "
              "to enable paid plans). Free plan is fully functional.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[JAMMER-FORTRESS] shutting down, persisting state...")
        server.shutdown_clean()
    return 0


if __name__ == "__main__":
    sys.exit(main())
