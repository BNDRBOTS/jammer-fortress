# 🔐 JAMMER-FORTRESS — FUSEPACK v2.0

Five layers, one runtime, zero external Python dependencies. A semantic memory
mesh that audits itself for contradictions and blind spots, a swarm agent
router, an enforcement engine for behavior contracts, a QR studio — productized
with accounts, plans, Stripe billing, and a glassmorphism web console.

## Quick start (no coding required)

```bash
python3 -m jammerfortress serve
```

Then open http://127.0.0.1:8788 — the landing page — and click **Open Console**.
Create an account and type plain English into the command deck
("remember the hearing moved to Thursday", "find anything about the lease",
"run a contradiction audit").

Other entry points:

```bash
python3 -m jammerfortress            # terminal console (TRI-SKIN skin 2)
python3 -m jammerfortress selftest   # forensic self-test, exit 0 = all pass
python3 -m jammerfortress serve --host 0.0.0.0 --port 8788 --data-dir /var/lib/fortress
```

## The five layers

| # | Layer | What it actually is |
|---|-------|---------------------|
| 1 | **EXO-LATTICE** | Resilient runtime: circuit breaker, exponential retry, HMAC-signed session tokens, rate limiter, sanitized audit event queue, Prometheus telemetry (`/metrics`) |
| 2 | **TRI-SKIN UI** | Web console (dark/light, dual accent pickers, glass panels, QR studio) + terminal console + JSON API |
| 3 | **OBLIVION MIRROR** | Memory auditor: contradiction detection, blind-spot index, timeline trail, Trust Vector Anchoring™, drift-loop detection |
| 4 | **BOSSWRAP-88** | Swarm router (researcher / legal / builder / validator) with shared context, race-free async task pool, connectors to Stripe, Notion, Gumroad, Neon |
| 5 | **TRUST-LOCK** | Behavior contracts enforced in code: overconfidence/hedging/echo scans, soft-language stripping, explicit admission: *“Not a truth anchor. Use only as boxed tool.”* |

## Deploy on Railway (no database pods, no extra services)

JAMMER-FORTRESS is a **single service**. It has no external database — the
schema (users, sessions, api_keys, subscriptions, usage, audit_log + indexes)
is created automatically inside its own SQLite file on first boot. Do **not**
add a Postgres/Redis/MySQL plugin; there is nothing to connect it to.

1. Push this folder to a GitHub repo (or `railway up` from inside it with the
   Railway CLI). `railway.json`, `Procfile`, and `requirements.txt` are already
   included — Railway detects Python and runs the right start command on its
   assigned `PORT` with zero configuration.
2. **Attach a volume** (the only “storage bin” you need): service → right-click
   or Settings → **Attach Volume**. Any mount path works — the server reads
   Railway’s `RAILWAY_VOLUME_MOUNT_PATH` automatically and keeps its database,
   state, and secret key there so nothing resets on redeploys. If you skip
   this, the app still runs and prints a clear warning that data won’t survive
   a redeploy.
3. Settings → Networking → **Generate Domain**. Railway terminates HTTPS and
   forwards `X-Forwarded-Proto`, which the server already honors.
4. (Optional, for paid plans) add the four Stripe variables below in the
   service’s **Variables** tab and point a Stripe webhook at
   `https://<your-domain>/api/billing/webhook`.

The health check is prewired to `/readyz`, so Railway only routes traffic once
the database and kernel report healthy.

## Semantic engine (embedding backends)

Three interchangeable backends power the memory mesh and OBLIVION-MIRROR:

| Backend | Cost | Honest quality note |
|---------|------|--------------------|
| `syn` (default) | Zero deps — a ~145k-entry thesaurus ships inside the package | Paraphrase-aware: “the lawyer kept the file” and “the attorney retained the file” land close together, so reworded contradictions get caught. Limits it admits to: irregular word forms (kept/keep) and sense-shifted rewrites can still slip. |
| `hash` | Zero deps, zero data | Fallback floor only. Paraphrases with **no shared vocabulary** score low and can slip past contradiction detection. |
| `onnx` | One-time ~90 MB download + `pip install onnxruntime` | Real sentence-transformer (all-MiniLM-L6-v2). The strongest option — catches vocabulary-free paraphrases. **Inference is fully offline** — network is used only by the one-time download. |

Selection is automatic: `onnx` when installed, else `syn` (bundled, so this is
what you get out of the box), else `hash`. Every downgrade is recorded and
shown in `stats` — the engine never silently pretends to be smarter than it is.

Upgrade (two commands, then restart — it auto-activates):

```bash
pip install onnxruntime
python3 -m jammerfortress get-model
```

On Railway: uncomment `onnxruntime` in `requirements.txt` and set
`JF_EMBED_AUTO_FETCH=1` — the model downloads into the attached volume on
first boot and is reused after that. Existing memories migrate automatically:
snapshots store text (never vectors), so everything re-embeds under the new
backend on load. The active backend is always visible in `stats` and the boot
log; forcing `JF_EMBED_BACKEND=onnx` hard-fails with the exact missing piece
rather than silently degrading.

## Configuration (environment variables)

Everything is optional. Missing keys degrade **explicitly** (clear 503 with the
variable name), never silently and never with fake data.

| Variable | Purpose |
|----------|---------|
| `JF_SECRET` | Token-signing secret. If unset, one is generated and stored at `<data-dir>/secret.key` (0600). |
| `JF_EMBED_BACKEND` | `auto` (default), `syn`, `hash`, or `onnx`. |
| `JF_EMBED_MODEL_DIR` | Where the ONNX model lives (default `./models/embedder`). |
| `JF_EMBED_AUTO_FETCH` | `1` = download the model into the data dir on boot if missing. |
| `JF_THESAURUS` | Override path to a MyThes-format `.dat` (bundled English data used by default). |
| `JF_PUBLIC_URL` | Your public URL (e.g. `https://app.example.com`). **Set this in production** — it pins Stripe redirect links so a spoofed Host header can never steer them. |
| `JF_MESH_CAPACITY` | Total kernel memory entries (default 100,000). |
| `JF_MESH_PER_USER` | Per-user memory quota (default 1,000) — one user can never evict another’s memories. |
| `JF_LOGIN_FAIL_MAX` | Failed logins per ip+email before a 15-minute lockout (default 5). |
| `STRIPE_SECRET_KEY` | Enables live Stripe checkout + billing portal. |
| `STRIPE_WEBHOOK_SECRET` | Enables `/api/billing/webhook` signature verification. |
| `STRIPE_PRICE_PRO` / `STRIPE_PRICE_FORTRESS` | Stripe Price IDs for the paid plans. |
| `NOTION_TOKEN` | Notion connector (search, append). |
| `GUMROAD_PRODUCT_ID` | Gumroad license verification connector. |
| `NEON_API_KEY` | Neon project connector. |

### Stripe setup

1. Create two recurring Prices ($19 Pro, $49 Fortress) and export their IDs as
   `STRIPE_PRICE_PRO` / `STRIPE_PRICE_FORTRESS` with `STRIPE_SECRET_KEY`.
2. Point a webhook at `https://your-host/api/billing/webhook` with events
   `checkout.session.completed`, `customer.subscription.updated`,
   `customer.subscription.deleted`; export its signing secret as
   `STRIPE_WEBHOOK_SECRET`.
3. Plans: **Free** 200 commands/day · **Pro** 5,000/day · **Fortress** 50,000/day.

## Security model

- Passwords: PBKDF2-HMAC-SHA256, 210k iterations, per-user salt.
- Sessions/API keys: opaque bearer tokens, stored **only** as SHA-256 hashes.
- No cookies → no CSRF surface. Strict CSP, `X-Frame-Options: DENY`, nosniff.
- Per-IP rate limit (120 req/min), 64 KB body cap, parameterized SQL only.
- Failed-login throttle: 5 misses per ip+email = 15-minute lockout (anti password-spray).
- Stripe webhook replay guard: every event id is recorded once; duplicates are ignored.
- Per-user memory isolation **and** per-user memory quotas inside the kernel
  (one user can’t read or evict another’s memories) — verified by the self-test.
- Live event stream (`/ws/events`) is per-user: you see your own activity and
  system boot events only, never other users’.
- Event log auto-redacts sensitive fields; secrets never appear in telemetry.
- State saves atomically (`os.replace`) — a crash can’t corrupt it.

## Operations

- `GET /healthz` liveness · `GET /readyz` readiness (DB check) · `GET /metrics` Prometheus.
- Live activity: `ws://host/ws/events?token=…` (RFC 6455, stdlib implementation).
- State persists to `<data-dir>/fortress_state.json` every 60 s, on shutdown, and on `persist`/`st4y`.
- Deploy: any box with Python 3.10+. Put a TLS terminator (Caddy/nginx) in front and
  pass `X-Forwarded-Proto: https` so checkout return-URLs are correct.

## Forensic self-test

`python3 -m jammerfortress selftest` runs 35 checks: every layer, the QR
encoder verified by OpenCV round-trip decode (when OpenCV is installed), the
WebSocket codec against the RFC 6455 test vector, Stripe signature
verify/tamper/stale cases, and a **live** HTTP+WS+billing end-to-end pass
against a real server instance. Exit code 0 only on 100% pass.
