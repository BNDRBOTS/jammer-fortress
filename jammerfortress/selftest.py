"""
FORENSIC SELF-TEST: every layer, every support system, plus a live HTTP
end-to-end pass against a real server instance. Run:
    python3 -m jammerfortress selftest
Exit code 0 only when every check passes.
"""
import json
import os
import socket
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from . import wire
from .bosswrap import BossWrap
from .connectors import ConnectorNotConfigured, Gumroad, Neon, Notion, Stripe, status
from .embeddings import cosine, embed
from .exo_lattice import CircuitBreaker, ExoLattice, retry
from .intent import compile_intent
from .kernel import Kernel
from .memory_mesh import MemoryMesh
from .oblivion_mirror import OblivionMirror
from .qr import make as qr_make
from .saas import auth, billing
from .saas.db import DB
from .saas.server import create_server
from .trust_lock import TrustLock

RESULTS = []


def check(name, fn):
    try:
        note = fn()
        if isinstance(note, str) and note.startswith("SKIP"):
            RESULTS.append((name, True, note))
            print(f"[SKIP] {name} :: {note[5:].strip()}")
        else:
            RESULTS.append((name, True, ""))
            print(f"[PASS] {name}")
    except Exception as e:  # noqa: BLE001
        RESULTS.append((name, False, str(e)))
        print(f"[FAIL] {name} :: {e}")


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ------------------------------------------------------- embedding backends
def t_embed_backend_registry():
    from . import embeddings as E
    inf = E.info()
    _assert(inf["backend"] in ("hash", "onnx"), f"unknown backend: {inf}")
    prev = E.backend()
    try:
        E.set_backend(E.HashBackend())
        _assert(E.info()["backend"] == "hash", "forced hash backend not active")
        _assert(E.embed("stable point") == E.embed("stable point"),
                "hash backend lost determinism")
        _assert(len(E.embed("x", dim=64)) == 64, "legacy dim override broken")
    finally:
        E.set_backend(prev)
    _assert(E.info()["backend"] == inf["backend"], "backend restore failed")


def t_wordpiece_reference():
    from . import embeddings as E
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "the", "hear", "##ing",
             "was", "re", "##sched", "##uled", "date", "moved", "."]
    tok = E.WordPieceTokenizer({w: i for i, w in enumerate(words)})
    got = [words[i] for i in tok.encode("The hearing was rescheduled.")]
    _assert(got == ["[CLS]", "the", "hear", "##ing", "was", "re", "##sched",
                    "##uled", ".", "[SEP]"], f"wordpiece sequence wrong: {got}")
    unk = [words[i] for i in tok.encode("zebra")]
    _assert(unk == ["[CLS]", "[UNK]", "[SEP]"], f"unknown-word handling wrong: {unk}")
    ids = tok.encode("the " * 500, max_len=16)
    _assert(len(ids) == 16 and words[ids[0]] == "[CLS]" and words[ids[-1]] == "[SEP]",
            "truncation must keep [CLS]...[SEP] within max_len")
    acc = [words[i] for i in tok.encode("Th\u00e9 hearing")]  # accent stripping
    _assert(acc[1] == "the", f"accent stripping broken: {acc}")


def t_onnx_plumbing():
    import numpy as np
    from . import embeddings as E
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
             "the": 4, "hear": 5, "##ing": 6}
    tok = E.WordPieceTokenizer(vocab)
    hidden_size = 6

    class _Meta:
        def __init__(self, name=None, shape=None):
            self.name, self.shape = name, shape

    class FakeSession:
        def get_inputs(self):
            return [_Meta("input_ids"), _Meta("attention_mask"),
                    _Meta("token_type_ids")]

        def get_outputs(self):
            return [_Meta(shape=[None, None, hidden_size])]

        def run(self, _out, feeds):
            ids = feeds["input_ids"]
            _assert(ids.dtype == np.int64, "input_ids must be int64")
            _assert(feeds["attention_mask"].shape == ids.shape, "mask shape wrong")
            _assert(not feeds["token_type_ids"].any(), "token_type_ids must be zeros")
            seq = ids.shape[1]
            out = np.zeros((1, seq, hidden_size))
            for t in range(seq):                       # one-hot per token id
                out[0, t, int(ids[0, t]) % hidden_size] = 1.0
            return [out]

    b = E.OnnxBackend(session=FakeSession(), tokenizer=tok)
    _assert(b.dim == hidden_size, "static hidden size not discovered")
    v = b.embed("the hearing")
    # tokens [CLS]=2 the=4 hear=5 ##ing=6 [SEP]=3 -> one-hot dims {2,4,5,0,3}
    _assert(abs(sum(x * x for x in v) - 1.0) < 1e-9, "output not L2-normalized")
    nonzero = {i for i, x in enumerate(v) if x > 1e-12}
    _assert(nonzero == {0, 2, 3, 4, 5}, f"mean pooling hit wrong dims: {nonzero}")
    _assert(len({round(x, 12) for x in v if x > 1e-12}) == 1,
            "equal-weight mean pooling broken")
    # masked positions must be excluded from the pool
    hidden = np.array([[[1.0, 0.0], [0.0, 1.0], [100.0, 100.0]]])
    pooled = E.OnnxBackend._pool(hidden, np.array([[1, 1, 0]]))
    _assert(abs(pooled[0] - pooled[1]) < 1e-9 and abs(pooled[0] - 2 ** -0.5) < 1e-9,
            f"attention mask ignored in pooling: {pooled}")


def t_onnx_real_model():
    from . import embeddings as E
    model_dir = E.find_model_dir()
    if model_dir is None:
        return ("SKIP: no local model in this environment (no network here); "
                "run `python3 -m jammerfortress get-model` where network exists")
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return "SKIP: model present but onnxruntime not installed"
    b = E.OnnxBackend(model_dir)
    para = E.cosine(b.embed("the hearing was rescheduled"),
                    b.embed("the court date moved"))
    unrel = E.cosine(b.embed("the hearing was rescheduled"),
                     b.embed("banana smoothie recipe blender"))
    _assert(para > unrel + 0.15,
            f"paraphrase separation too weak (para={para:.3f} unrel={unrel:.3f})")
    prev = E.backend()
    try:
        E.set_backend(b)
        m = MemoryMesh()
        m.record("the hearing was rescheduled")
        m.record("the court date never moved")
        _assert(OblivionMirror().contradictions(m.snapshot()),
                "paraphrase contradiction missed even with model backend")
    finally:
        E.set_backend(prev)


def t_mesh_backend_migration():
    import math as _math
    from . import embeddings as E
    m = MemoryMesh()
    m.record("the hearing was rescheduled to friday")
    snap = m.snapshot()
    _assert(all("vector" not in e for e in snap),
            "snapshot must never persist vectors (they are backend-specific)")

    class TinyBackend:
        name, dim = "tiny", 8

        def embed(self, text):
            v = [0.0] * 8
            for i, byte in enumerate((text or "").encode()):
                v[i % 8] += (byte % 7) - 3
            n = _math.sqrt(sum(x * x for x in v)) or 1.0
            return [x / n for x in v]

    prev = E.backend()
    try:
        E.set_backend(TinyBackend())
        m2 = MemoryMesh()
        m2.load(snap)
        _assert(len(m2.store[0]["vector"]) == 8,
                "restore did not re-embed under the active backend")
        _assert(m2.query("the hearing was rescheduled to friday", top_k=1),
                "query broken after backend migration")
    finally:
        E.set_backend(prev)


def t_model_fetch_validate():
    from . import model_fetch as MF
    with tempfile.TemporaryDirectory() as d:
        try:
            MF.validate(d)
            raise AssertionError("validate accepted an empty dir")
        except MF.ModelFetchError:
            pass
        with open(os.path.join(d, "model.onnx"), "wb") as f:  # truncated file
            f.write(b"\x00" * 1024)
        with open(os.path.join(d, "vocab.txt"), "w", encoding="utf-8") as f:
            f.write("[UNK]\n")
        try:
            MF.validate(d)
            raise AssertionError("validate accepted truncated files")
        except MF.ModelFetchError as e:
            _assert("model.onnx" in str(e), f"wrong rejection reason: {e}")
        with open(os.path.join(d, "model.onnx"), "wb") as f:  # floor-size fakes
            f.seek(MF.MIN_SIZES["model.onnx"])
            f.write(b"\x00")
        with open(os.path.join(d, "vocab.txt"), "w", encoding="utf-8") as f:
            f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n")
            for i in range(20000):
                f.write(f"token{i}\n")
        report = MF.validate(d)
        _assert(report["files"]["vocab.txt"] > 100_000, "validate report wrong")


# ---------------------------------------------------------------- layers 1-5
def t_embeddings():
    a, b = embed("custody hearing in court"), embed("custody hearing in court")
    _assert(a == b, "embeddings not deterministic")
    near = cosine(embed("legal custody court hearing"), embed("court hearing about custody"))
    far = cosine(embed("legal custody court hearing"), embed("banana smoothie recipe blender"))
    _assert(near > far, f"semantic ordering broken ({near:.3f} <= {far:.3f})")


def t_trust_scan():
    r = TrustLock().scan("Trust me, this is guaranteed and obviously correct.")
    _assert(not r["passed"] and r["violation_count"] >= 3, "overconfidence not flagged")
    _assert(TrustLock().scan("The filing deadline is March 4.")["passed"], "clean text flagged")


def t_trust_enforce():
    out = TrustLock().enforce("You're right, maybe we should just possibly file it.")
    low = out.lower()
    for banned in ("maybe", "just", "possibly", "you're right"):
        _assert(banned not in low, f"'{banned}' survived enforcement: {out!r}")


def t_mesh_query():
    m = MemoryMesh()
    m.record("The lease for the Phoenix apartment ends in September")
    m.record("Mason's school enrollment paperwork was filed Tuesday")
    m.record("Stripe payout landed in the business account")
    hits = m.query("apartment lease ending", top_k=1)
    _assert(hits and "lease" in hits[0]["text"], f"wrong retrieval: {hits}")


def t_mesh_isolation():
    m = MemoryMesh()
    m.record("alpha secret fact", session_id="user_a")
    m.record("beta secret fact", session_id="user_b")
    got = m.query("secret fact", top_k=5, session_id="user_a")
    _assert(got and all("alpha" in h["text"] for h in got), "session isolation leak in query")
    snap = m.snapshot(session_id="user_b")
    _assert(all("beta" in e["text"] for e in snap), "session isolation leak in snapshot")


def t_oblivion_contradiction():
    m = MemoryMesh()
    m.record("Stephanie was present at the hearing on May 5")
    m.record("Stephanie was not present at the hearing on May 5")
    audit = OblivionMirror().audit(m.snapshot())
    _assert(audit["contradictions"], "direct contradiction missed")
    _assert("trust_vector" in audit and "blind_spots" in audit and "timeline" in audit,
            f"audit missing sections: {list(audit)}")


def t_oblivion_trust_vector():
    m = MemoryMesh()
    for txt in ("fact one about the case", "fact two about the budget", "fact three about school"):
        m.record(txt)
    tv = OblivionMirror().trust_vector(m.snapshot())
    _assert(isinstance(tv, float) and 0.0 <= tv <= 1.0, f"trust vector out of [0,1]: {tv!r}")
    m.record("the sky was blue on Tuesday")
    m.record("the sky was not blue on Tuesday")
    tv2 = OblivionMirror().trust_vector(m.snapshot())
    _assert(tv2 < tv, f"contradictions did not lower the trust vector ({tv2} !< {tv})")


def t_bosswrap_route():
    mesh = MemoryMesh()
    b = BossWrap(mesh=mesh, oblivion=OblivionMirror(), trust=TrustLock())
    r = b.route("research the custody law timeline and verify the evidence", session_id="s1")
    _assert(r, "route returned nothing")
    stats = b.get_statistics()
    _assert(stats["routed"].get("validator"), f"validator never ran: {stats}")


def t_bosswrap_async():
    b = BossWrap()
    tids = [b.submit(lambda i=i: i * i) for i in range(8)]
    for i, tid in enumerate(tids):
        r = b.result(tid, timeout=5.0)
        _assert(r == i * i, f"async task {i} returned {r}")


def t_circuit_breaker():
    cb = CircuitBreaker(fail_max=3, reset_timeout=60)
    def boom():
        raise ValueError("x")
    for _ in range(3):
        try:
            cb.call(boom)
        except ValueError:
            pass
    _assert(cb.state == "open", f"breaker state {cb.state}")
    try:
        cb.call(lambda: 1)
        raise AssertionError("open breaker allowed a call")
    except RuntimeError:
        pass


def t_retry():
    calls = {"n": 0}
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("transient")
        return "ok"
    _assert(retry(flaky, attempts=5, base=0.001) == "ok", "retry failed")
    _assert(calls["n"] == 3, f"retry count {calls['n']}")


def t_tokens():
    exo = ExoLattice(secret="test-secret")
    tok = exo.issue_token("system_admin", "scott")
    ok, role, user = exo.verify_token(tok)
    _assert(ok and role == "system_admin" and user == "scott", "token verify failed")
    parts = tok.split(".")
    parts[1] = "system_root"
    _assert(exo.verify_token(".".join(parts))[0] is False, "tampered token accepted")


def t_telemetry():
    exo = ExoLattice()
    exo.log("unit", {"password": "hunter2", "detail": "fine"})
    _assert(list(exo.events)[-1]["payload"]["password"] == "***", "secret not sanitized")
    exo.telemetry.observe("latency_seconds", 0.05)
    text = exo.telemetry.prometheus()
    _assert("fortress_event_unit 1" in text and "fortress_latency_seconds_count 1" in text,
            f"prometheus malformed:\n{text}")


# ---------------------------------------------------------------- kernel
def t_kernel_commands():
    with tempfile.TemporaryDirectory() as td:
        k = Kernel(state_path=os.path.join(td, "state.json"))
        k.boot()
        _assert("triggers" in k.command("m3nu"), "m3nu failed")
        rid = k.command("record", "the hearing moved to Thursday", session_id="u1")
        _assert(rid.get("id"), "record failed")
        q = k.command("s3m4", "when is the hearing", session_id="u1")
        _assert(q and "Thursday" in q[0]["text"], f"query failed: {q}")
        svg = k.command("c0de", "https://example.com")
        _assert(svg["svg"].startswith("<svg"), "c0de trigger broken")
        _assert(k.command("st4t")["exo"], "stats broken")


def t_kernel_persistence():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "state.json")
        k1 = Kernel(state_path=path)
        k1.record("persistent fortress memory alpha")
        k1.persist()
        k2 = Kernel(state_path=path)
        hits = k2.query("fortress memory alpha")
        _assert(hits and "alpha" in hits[0]["text"], "state did not survive reload")


def t_intent_mapping():
    cases = [
        ("remember the hearing is on the 14th", "record"),
        ("find anything about the lease", "query"),
        ("run a contradiction audit", "audit"),
        ("make a qr code for https://bndr.llc", "qr"),
        ("save everything", "persist"),
        ("how healthy is the system status", "stats"),
        ("help me with drafting the appeal letter", "route"),
    ]
    for text, expected in cases:
        got = compile_intent(text)
        _assert(got["command"] == expected, f"{text!r} -> {got['command']} (wanted {expected})")
    fb = compile_intent("zebra umbrella cascade")
    _assert(fb["command"] == "route" and fb["confidence"] <= 0.5, "fallback wrong")


def t_kernel_intent():
    with tempfile.TemporaryDirectory() as td:
        k = Kernel(state_path=os.path.join(td, "state.json"))
        r = k.intent("remember the sky over Phoenix was teal tonight", session_id="u9")
        _assert(r["plan"]["command"] == "record" and r["result"]["id"], f"intent exec failed: {r}")


# ---------------------------------------------------------------- QR (the prior-session killer)
def t_qr_structure():
    code = qr_make("FORTRESS", ecl="M")
    m = code.matrix
    n = code.size
    _assert(n == 17 + code.version * 4, "size/version mismatch")
    for (r, c) in ((0, 0), (0, n - 7), (n - 7, 0)):
        _assert(m[r][c] == 1 and m[r + 3][c + 3] == 1, "finder pattern damaged")
    _assert(m[n - 8][8] == 1, "dark module missing")
    _assert(all(v in (0, 1) for row in m for v in row), "unfilled module")


def t_qr_roundtrip():
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        raise AssertionError(f"OpenCV unavailable for forensic verification: {e}")
    det = cv2.QRCodeDetector()
    payloads = [
        ("HI", "L"),
        ("https://bndr.llc/fortress", "M"),
        ("JAMMER-FORTRESS FUSEPACK v2.0 :: ALL LAYERS ONLINE", "Q"),
        ("tel:+16025550147", "H"),
        ("The quick brown fox jumps over the lazy dog 0123456789", "M"),
    ]
    for text, ecl in payloads:
        code = qr_make(text, ecl=ecl)
        arr = np.array(code.matrix, dtype=np.uint8)
        img = (1 - arr) * 255
        img = np.pad(img, 4, constant_values=255)
        img = np.kron(img, np.ones((10, 10), dtype=np.uint8))
        decoded, _, _ = det.detectAndDecode(img)
        _assert(decoded == text, f"round-trip failed v{code.version}-{ecl}: {decoded!r} != {text!r}")


def t_qr_svg_variants():
    plain = qr_make("style test", ecl="H").svg(dark="#112233", light="#f5f4f2")
    _assert("#112233" in plain and "#f5f4f2" in plain, "colors not applied")
    dots = qr_make("style test", ecl="H").svg(shape="dot")
    _assert("<circle" in dots, "dot shape missing")
    logo = qr_make("style test", ecl="H").svg(logo_slot=0.18)
    _assert(logo.count("<rect") >= 3, "logo slot not reserved")
    _assert(qr_make("style test", ecl="H").svg().startswith("<svg xmlns"), "svg header malformed")


def t_qr_limits():
    try:
        qr_make("x" * 5000, ecl="H")
        raise AssertionError("oversized payload accepted")
    except ValueError:
        pass


# ---------------------------------------------------------------- wire
def t_wire_handshake():
    _assert(wire.accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=",
            "RFC 6455 accept key vector failed")
    resp = wire.handshake_response("dGhlIHNhbXBsZSBub25jZQ==")
    _assert(b"101 Switching Protocols" in resp, "handshake response malformed")


def t_wire_frames():
    msg = "fortress event " + "x" * 300  # forces 126-length path
    framed = wire.encode_client_frame(msg)
    op, payload, used = wire.decode_frame(framed)
    _assert(op == wire.OP_TEXT and payload.decode() == msg and used == len(framed),
            "masked frame round-trip failed")
    op2, payload2, _ = wire.decode_frame(wire.encode_frame("tiny"))
    _assert(op2 == wire.OP_TEXT and payload2 == b"tiny", "unmasked frame failed")
    _assert(wire.decode_frame(framed[:5]) == (None, None, 0), "partial frame not detected")


# ---------------------------------------------------------------- product layer
def t_auth():
    h = auth.hash_password("correct horse battery")
    _assert(auth.verify_password("correct horse battery", h), "valid password rejected")
    _assert(not auth.verify_password("wrong horse", h), "wrong password accepted")
    _assert(not auth.verify_password("x", "garbage"), "garbage hash crashed or passed")
    _assert(auth.valid_email("a@b.co") and not auth.valid_email("not-an-email"), "email validation")
    _assert(auth.password_problem("short") and auth.password_problem("correct horse battery") is None,
            "password policy")


def t_stripe_signature():
    payload = json.dumps({"type": "checkout.session.completed"}).encode()
    secret = "whsec_testkey"
    header = billing.sign_payload(payload, secret)
    _assert(billing.verify_stripe_signature(payload, header, secret), "valid signature rejected")
    _assert(not billing.verify_stripe_signature(payload + b"x", header, secret), "tampered body accepted")
    _assert(not billing.verify_stripe_signature(payload, header, "whsec_other"), "wrong secret accepted")
    stale = billing.sign_payload(payload, secret, ts=int(time.time()) - 4000)
    _assert(not billing.verify_stripe_signature(payload, stale, secret), "stale timestamp accepted")
    _assert(not billing.verify_stripe_signature(payload, "t=abc,v1=zz", secret), "garbage header accepted")


def t_billing_events():
    with tempfile.TemporaryDirectory() as td:
        db = DB(os.path.join(td, "t.db"))
        uid = db.create_user("x@y.co", auth.hash_password("longenoughpw"))
        billing.handle_event(db, {"type": "checkout.session.completed", "data": {"object": {
            "client_reference_id": uid, "customer": "cus_1", "subscription": "sub_1",
            "metadata": {"plan": "pro"}}}})
        plan, _ = billing.plan_for(db, uid)
        _assert(plan == "pro", f"activation failed: {plan}")
        billing.handle_event(db, {"type": "customer.subscription.deleted",
                                  "data": {"object": {"id": "sub_1", "status": "canceled"}}})
        plan, _ = billing.plan_for(db, uid)
        _assert(plan == "free", f"downgrade failed: {plan}")
        _assert("ignored" in billing.handle_event(db, {"type": "invoice.paid"}), "unknown event mishandled")


def t_db():
    with tempfile.TemporaryDirectory() as td:
        db = DB(os.path.join(td, "t.db"))
        uid = db.create_user("scott@bndr.llc", auth.hash_password("longenoughpw"))
        _assert(db.user_by_email("SCOTT@BNDR.LLC")["id"] == uid, "email lookup not case-insensitive")
        tok = auth.new_session_token()
        db.create_session(auth.token_hash(tok), uid)
        _assert(db.session_user(auth.token_hash(tok))["id"] == uid, "session lookup failed")
        _assert(db.session_user(auth.token_hash("forged")) is None, "forged session accepted")
        for i in range(3):
            n, ok = db.incr_usage(uid, 3)
            _assert(ok and n == i + 1, "usage counter wrong")
        _assert(db.incr_usage(uid, 3)[1] is False, "usage limit not enforced")


def t_connectors_honest():
    saved = {k: os.environ.pop(k, None) for k in
             ("STRIPE_SECRET_KEY", "NOTION_TOKEN", "GUMROAD_PRODUCT_ID", "NEON_API_KEY")}
    try:
        for cls, call in ((Stripe, lambda c: c.get("/v1/balance")),
                          (Notion, lambda c: c.search("x")),
                          (Gumroad, lambda c: c.verify_license("k")),
                          (Neon, lambda c: c.list_projects())):
            try:
                call(cls())
                raise AssertionError(f"{cls.__name__} ran without credentials")
            except ConnectorNotConfigured as e:
                _assert("environment variable" in str(e), "unhelpful config error")
        st = status()
        _assert(set(st) == {"stripe", "notion", "gumroad", "neon"} and not any(st.values()),
                f"status wrong: {st}")
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


# ---------------------------------------------------------------- live server e2e
def _http(method, url, body=None, token=None, headers=None, raw=False):
    data = json.dumps(body).encode() if isinstance(body, dict) else body
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            payload = r.read()
            return r.status, payload if raw else json.loads(payload or b"{}")
    except urllib.error.HTTPError as e:
        payload = e.read()
        try:
            return e.code, json.loads(payload or b"{}")
        except json.JSONDecodeError:
            return e.code, {"raw": payload[:200]}


def t_server_e2e():
    saved_ws = os.environ.pop("STRIPE_WEBHOOK_SECRET", None)
    saved_sk = os.environ.pop("STRIPE_SECRET_KEY", None)
    tmp = tempfile.mkdtemp(prefix="jf_e2e_")
    srv = create_server(host="127.0.0.1", port=0, data_dir=tmp)
    port = srv.server_address[1]
    base = f"http://127.0.0.1:{port}"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        code, body = _http("GET", f"{base}/healthz")
        _assert(code == 200 and body["ok"], "healthz")
        _assert(_http("GET", f"{base}/readyz")[1]["ready"], "readyz")
        code, page = _http("GET", f"{base}/", raw=True)
        _assert(code == 200 and b"JAMMER-" in page, "landing page")
        code, page = _http("GET", f"{base}/console", raw=True)
        _assert(code == 200 and b"Command deck" in page, "console page")
        code, metrics = _http("GET", f"{base}/metrics", raw=True)
        _assert(code == 200 and b"# TYPE fortress_" in metrics, "prometheus endpoint")

        # auth flow + validation
        _assert(_http("POST", f"{base}/api/signup", {"email": "bad", "password": "longenoughpw"})[0] == 400,
                "bad email accepted")
        _assert(_http("POST", f"{base}/api/signup", {"email": "s@bndr.llc", "password": "short"})[0] == 400,
                "weak password accepted")
        code, acct = _http("POST", f"{base}/api/signup", {"email": "s@bndr.llc", "password": "fortress-grade-pw"})
        _assert(code == 201 and acct["token"], "signup failed")
        _assert(_http("POST", f"{base}/api/signup", {"email": "s@bndr.llc", "password": "fortress-grade-pw"})[0] == 409,
                "duplicate signup allowed")
        _assert(_http("POST", f"{base}/api/login", {"email": "s@bndr.llc", "password": "wrong-password"})[0] == 401,
                "wrong password logged in")
        code, login = _http("POST", f"{base}/api/login", {"email": "s@bndr.llc", "password": "fortress-grade-pw"})
        tok = login["token"]
        _assert(code == 200 and tok, "login failed")
        _assert(_http("GET", f"{base}/api/me")[0] == 401, "unauthenticated /api/me allowed")
        code, me = _http("GET", f"{base}/api/me", token=tok)
        _assert(code == 200 and me["plan"] == "free" and me["limit"] == 200, f"me wrong: {me}")

        # kernel over HTTP with per-user isolation
        code, r = _http("POST", f"{base}/api/command",
                        {"command": "record", "arg": "the fortress e2e memory marker"}, token=tok)
        _assert(code == 200 and r["result"]["id"], "command record failed")
        code, r = _http("POST", f"{base}/api/command", {"command": "query", "arg": "e2e memory marker"}, token=tok)
        _assert(code == 200 and r["result"] and "marker" in r["result"][0]["text"], f"query failed: {r}")
        code, r = _http("POST", f"{base}/api/command", {"command": "audit"}, token=tok)
        _assert(code == 200 and "trust_vector" in r["result"], "audit over http failed")
        _assert(_http("POST", f"{base}/api/command", {"command": "boot"}, token=tok)[0] == 400,
                "privileged command exposed")
        code, r = _http("POST", f"{base}/api/intent", {"text": "find the e2e memory marker"}, token=tok)
        _assert(code == 200 and r["plan"]["command"] == "query", f"intent endpoint failed: {r}")

        # second user cannot see first user's memory
        _, acct2 = _http("POST", f"{base}/api/signup", {"email": "u2@bndr.llc", "password": "another-long-pw"})
        code, r = _http("POST", f"{base}/api/command", {"command": "query", "arg": "e2e memory marker"},
                        token=acct2["token"])
        _assert(code == 200 and not r["result"], f"cross-user memory leak: {r}")

        # QR endpoint
        q = urllib.parse.urlencode({"text": "https://bndr.llc", "dark": "#0d0d10", "light": "#ffffff",
                                    "shape": "dot", "logo": "1"})
        code, svg = _http("GET", f"{base}/api/qr?{q}", token=tok, raw=True)
        _assert(code == 200 and svg.startswith(b"<svg") and b"<circle" in svg, "qr endpoint failed")
        _assert(_http("GET", f"{base}/api/qr?text=x&dark=red", token=tok)[0] == 400, "bad color accepted")

        # billing surface: explicit 503s when unconfigured, real state change via signed webhook
        _assert(_http("GET", f"{base}/api/plans")[1]["plans"]["pro"]["daily_commands"] == 5000, "plans")
        _assert(_http("POST", f"{base}/api/billing/checkout", {"plan": "pro"}, token=tok)[0] == 503,
                "checkout without Stripe keys must 503")
        _assert(_http("POST", f"{base}/api/billing/webhook", {"type": "x"})[0] == 503,
                "webhook without secret must 503")
        os.environ["STRIPE_WEBHOOK_SECRET"] = "whsec_e2e"
        uid = srv.db.user_by_email("s@bndr.llc")["id"]
        event = json.dumps({"type": "checkout.session.completed", "data": {"object": {
            "client_reference_id": uid, "customer": "cus_e2e", "subscription": "sub_e2e",
            "metadata": {"plan": "fortress"}}}}).encode()
        _assert(_http("POST", f"{base}/api/billing/webhook", event,
                      headers={"Stripe-Signature": "t=1,v1=bogus"})[0] == 400, "forged webhook accepted")
        sig = billing.sign_payload(event, "whsec_e2e")
        code, r = _http("POST", f"{base}/api/billing/webhook", event, headers={"Stripe-Signature": sig})
        _assert(code == 200 and r["received"], f"signed webhook rejected: {r}")
        _assert(_http("GET", f"{base}/api/me", token=tok)[1]["plan"] == "fortress", "plan not upgraded")

        # usage metering returns 402 at the plan ceiling
        old = billing.PLANS["free"]["daily_commands"]
        billing.PLANS["free"]["daily_commands"] = 2
        try:
            t2 = acct2["token"]
            codes = [_http("POST", f"{base}/api/command", {"command": "stats"}, token=t2)[0]
                     for _ in range(3)]
            _assert(codes[-1] == 402 and codes[0] == 200, f"metering wrong: {codes}")
        finally:
            billing.PLANS["free"]["daily_commands"] = old

        # websocket event stream, live
        s = socket.create_connection(("127.0.0.1", port), timeout=5)
        s.sendall((f"GET /ws/events?token={tok} HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\n"
                   "Connection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                   "Sec-WebSocket-Version: 13\r\n\r\n").encode())
        head = s.recv(4096)
        _assert(b"101" in head.split(b"\r\n")[0], f"ws upgrade failed: {head[:80]}")
        buf = head.split(b"\r\n\r\n", 1)[1]
        deadline = time.time() + 5
        first = None
        while time.time() < deadline and first is None:
            op, payload, used = wire.decode_frame(buf)
            if used:
                first = payload
                break
            s.settimeout(deadline - time.time())
            buf += s.recv(4096)
        _assert(first and b"connected" in first, f"ws stream silent: {first}")
        s.sendall(wire.encode_frame(b"\x03\xe8", opcode=wire.OP_CLOSE, mask=b"\x01\x02\x03\x04"))
        s.close()

        # rate limiter unit check (per-IP token bucket)
        allowed = [srv.rate_ok("203.0.113.9") for _ in range(121)]
        _assert(all(allowed[:120]) and not allowed[120], "rate limiter ceiling wrong")
    finally:
        srv.shutdown_clean()
        if saved_ws is not None:
            os.environ["STRIPE_WEBHOOK_SECRET"] = saved_ws
        else:
            os.environ.pop("STRIPE_WEBHOOK_SECRET", None)
        if saved_sk is not None:
            os.environ["STRIPE_SECRET_KEY"] = saved_sk


def run():
    print("=" * 72)
    print("JAMMER-FORTRESS FORENSIC SELF-TEST")
    print("=" * 72)
    tests = [
        ("L1 embeddings: deterministic + semantic ordering", t_embeddings),
        ("embeddings: backend registry, forcing, restore", t_embed_backend_registry),
        ("embeddings: WordPiece tokenizer reference behavior", t_wordpiece_reference),
        ("embeddings: ONNX pipeline math (injected session)", t_onnx_plumbing),
        ("embeddings: real model paraphrase separation", t_onnx_real_model),
        ("mesh: snapshot/restore re-embeds across backends", t_mesh_backend_migration),
        ("model-fetch: validation rejects truncated/corrupt files", t_model_fetch_validate),
        ("L5 trust-lock: overconfidence flagged, clean text passes", t_trust_scan),
        ("L5 trust-lock: soft language stripped", t_trust_enforce),
        ("mesh: semantic retrieval hits the right memory", t_mesh_query),
        ("mesh: per-session isolation (query + snapshot)", t_mesh_isolation),
        ("L3 oblivion: direct contradiction detected + audit sections", t_oblivion_contradiction),
        ("L3 oblivion: trust vector anchored and bounded", t_oblivion_trust_vector),
        ("L4 bosswrap: routed pipeline ends in validator", t_bosswrap_route),
        ("L4 bosswrap: async submit/result race-free x8", t_bosswrap_async),
        ("L1 exo: circuit breaker opens and blocks", t_circuit_breaker),
        ("L1 exo: exponential retry recovers", t_retry),
        ("L1 exo: HMAC tokens verify, tamper rejected", t_tokens),
        ("L1 exo: sanitized events + prometheus exposition", t_telemetry),
        ("kernel: triggers m3nu/record/s3m4/c0de/st4t", t_kernel_commands),
        ("kernel: st4y persistence survives restart", t_kernel_persistence),
        ("intent: 7 phrasings map to correct commands + fallback", t_intent_mapping),
        ("kernel: free text executes end-to-end", t_kernel_intent),
        ("QR: matrix structure (finders, dark module, filled)", t_qr_structure),
        ("QR: OpenCV round-trip decode 5/5 payloads", t_qr_roundtrip),
        ("QR: SVG colors, dot shape, logo slot", t_qr_svg_variants),
        ("QR: oversized payload rejected cleanly", t_qr_limits),
        ("wire: RFC 6455 handshake vector", t_wire_handshake),
        ("wire: frame codec round-trip + partial detection", t_wire_frames),
        ("auth: pbkdf2 verify/reject + policies", t_auth),
        ("billing: stripe signature verify/tamper/stale", t_stripe_signature),
        ("billing: webhook events drive subscription state", t_billing_events),
        ("db: users, hashed sessions, usage ceiling", t_db),
        ("connectors: honest failure without credentials", t_connectors_honest),
        ("server: full HTTP+WS+billing end-to-end", t_server_e2e),
    ]
    for name, fn in tests:
        check(name, fn)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print("=" * 72)
    print(f"RESULT: {passed}/{len(RESULTS)} PASS")
    for name, ok, err in RESULTS:
        if not ok:
            print(f"  FAILED -> {name}: {err}")
    print("=" * 72)
    return passed == len(RESULTS)


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
