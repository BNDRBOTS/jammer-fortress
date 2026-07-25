"""
LAYER 1 - EXO-LATTICE: resilient runtime. Pure stdlib.
  * HMAC session tokens          * Rate limiter (per client)
  * Circuit breaker + retry      * Feature flags
  * Sanitized event log queue    * Prometheus-style telemetry
"""
import hashlib
import hmac
import json
import random
import threading
import time
from collections import Counter, deque

SENSITIVE = {"token", "password", "secret", "api_key", "system_key"}


class CircuitBreaker:
    def __init__(self, fail_max=5, reset_timeout=30):
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.opened_at = 0.0
        self.state = "closed"
        self._lock = threading.Lock()

    def call(self, fn, *a, **k):
        with self._lock:
            if self.state == "open":
                if time.time() - self.opened_at > self.reset_timeout:
                    self.state = "half-open"
                else:
                    raise RuntimeError("circuit_open")
        try:
            r = fn(*a, **k)
        except Exception:
            with self._lock:
                self.failures += 1
                if self.failures >= self.fail_max:
                    self.state = "open"
                    self.opened_at = time.time()
            raise
        with self._lock:
            self.failures = 0
            self.state = "closed"
        return r


def retry(fn, attempts=5, base=0.05, cap=1.0):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(cap, base * (2 ** i)) + random.random() * base)
    raise last


class Telemetry:
    def __init__(self):
        self._lock = threading.Lock()
        self.counters = Counter()
        self.hist = {}

    def inc(self, name, n=1):
        with self._lock:
            self.counters[name] += n

    def observe(self, name, value):
        with self._lock:
            self.hist.setdefault(name, []).append(value)

    def render(self):
        with self._lock:
            out = {"counters": dict(self.counters)}
            out["histograms"] = {k: {"count": len(v), "avg": round(sum(v) / len(v), 6)}
                                 for k, v in self.hist.items() if v}
            return out

    def prometheus(self):
        """Prometheus text exposition format."""
        with self._lock:
            lines = []
            for k, v in sorted(self.counters.items()):
                name = "fortress_" + k
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {v}")
            for k, vals in sorted(self.hist.items()):
                if not vals:
                    continue
                name = "fortress_" + k
                lines.append(f"# TYPE {name} summary")
                lines.append(f"{name}_count {len(vals)}")
                lines.append(f"{name}_sum {round(sum(vals), 6)}")
            return "\n".join(lines) + "\n"


class ExoLattice:
    def __init__(self, secret="FUSEPACK-DEFAULT-KEY", rate_limit=100):
        self.secret = secret.encode()
        self.rate_limit = rate_limit
        self.sessions = {}
        self.rate_map = {}
        self.events = deque(maxlen=100_000)
        self.telemetry = Telemetry()
        self.flags = {"token": True, "record": True, "recall": True, "query": True,
                      "route": True, "audit": True, "metrics": True, "qr": True,
                      "intent": True, "trust": True}
        self.breaker = CircuitBreaker()
        self._lock = threading.Lock()

    # ---- tokens ----------------------------------------------------------
    def issue_token(self, role, user, hours=24):
        rnd = hashlib.sha256(f"{time.time_ns()}{random.random()}".encode()).hexdigest()[:32]
        expiry = int(time.time()) + hours * 3600
        data = f"{rnd}.{role}.{user}.{expiry}"
        sig = hmac.new(self.secret, data.encode(), hashlib.sha256).hexdigest()
        token = f"{data}.{sig}"
        with self._lock:
            self.sessions[token] = {"role": role, "user": user, "expires": expiry}
        self.log("token_issued", {"role": role, "user": user})
        return token

    def verify_token(self, token):
        try:
            rnd, role, user, expiry, sig = token.split(".")
            data = f"{rnd}.{role}.{user}.{expiry}"
            expected = hmac.new(self.secret, data.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, sig):
                return False, None, None
            if time.time() > int(expiry):
                with self._lock:
                    self.sessions.pop(token, None)
                return False, None, None
            return True, role, user
        except Exception:  # noqa: BLE001
            return False, None, None

    # ---- rate limit ------------------------------------------------------
    def allow(self, client="local"):
        with self._lock:
            now = time.time()
            cnt, ts = self.rate_map.get(client, (0, now))
            if now - ts > 60:
                cnt, ts = 0, now
            cnt += 1
            self.rate_map[client] = (cnt, ts)
            return cnt <= self.rate_limit

    # ---- feature flags ---------------------------------------------------
    def enabled(self, name):
        return self.flags.get(name, False)

    # ---- event log -------------------------------------------------------
    @staticmethod
    def _sanitize(d):
        return {k: ("***" if k.lower() in SENSITIVE else v) for k, v in (d or {}).items()}

    def log(self, cat, payload):
        self.events.append({"ts": time.time(), "cat": cat, "payload": self._sanitize(payload)})
        self.telemetry.inc(f"event_{cat}")

    def flush(self, path):
        with self._lock:
            batch = list(self.events)
            self.events.clear()
        with open(path, "a") as f:
            for e in batch:
                f.write(json.dumps(e) + "\n")
        return len(batch)

    def get_statistics(self):
        return {"sessions": len(self.sessions), "events_buffered": len(self.events),
                "breaker": self.breaker.state, "telemetry": self.telemetry.render()}
