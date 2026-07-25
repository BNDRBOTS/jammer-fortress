"""
LAYER 2 - TRI-SKIN (terminal skin): real stdin console over the fused Kernel.
Supports encoded triggers and plain verbs plus free-text intent.

Run:  python -m jammerfortress             (interactive console)
      python -m jammerfortress selftest    (full-stack self-test)
      python -m jammerfortress serve       (SaaS server + web console)
"""
import json
import sys

from .kernel import Kernel, TRIGGERS

BANNER = r"""
  ____   _   _  ____  ____    _____ _   _ ____  _____ ____   _____ ____  ____
 | __ ) | \ | ||  _ \|  _ \  |  ___| | | / ___|| ____|  _ \ | ____/ ___||  _ \
 |  _ \ |  \| || | | | |_) | | |_  | | | \___ \|  _| | |_) ||  _| \___ \| |_) |
 | |_) || |\  || |_| |  _ <  |  _| | |_| |___) | |___|  __/ | |___ ___) |  __/
 |____/ |_| \_||____/|_| \_\ |_|    \___/|____/|_____|_|    |_____|____/|_|
        J A M M E R - F O R T R E S S   ::   FUSEPACK LOCK-IN   ::   v2.0
"""

HELP = """commands:
  boot                       bring all five layers online
  record <text>              store into the semantic memory mesh
  recall [n]                 show last n memory entries
  query <text>               semantic search (real embeddings)
  route <task>               run the swarm agent pipeline
  audit                      OBLIVION-MIRROR full logic audit
  trust <text>               TRUST-LOCK scan + enforce
  qr <text>                  generate a QR code (prints SVG)
  intent <plain english>     compile free text into a command and run it
  persist                    save cross-session state
  stats                      live telemetry across all layers
  triggers                   show encoded command triggers
  help | quit
encoded triggers: """ + ", ".join(sorted(TRIGGERS)) + "\n"

VERBS = ("boot", "record", "recall", "query", "route", "audit", "trust",
         "qr", "intent", "persist", "stats")


def _emit(obj):
    print(json.dumps(obj, indent=2, default=str))


def run_console(kernel=None):
    k = kernel or Kernel()
    print(BANNER)
    print(HELP)
    while True:
        try:
            line = input("fortress> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[TRI-SKIN] session closed")
            return
        if not line:
            continue
        parts = line.split(" ", 1)
        cmd, arg = parts[0], (parts[1] if len(parts) > 1 else None)
        if cmd in ("quit", "exit"):
            k.persist()
            print("[st4y] state saved. session closed")
            return
        if cmd == "help":
            print(HELP)
            continue
        if cmd == "triggers":
            _emit(TRIGGERS)
            continue
        try:
            if cmd == "boot":
                _emit(k.boot())
            elif cmd in VERBS or cmd in TRIGGERS:
                _emit(k.command(cmd, arg))
            else:
                _emit(k.intent(line))  # free text -> intent compiler, never a dead end
        except Exception as e:  # noqa: BLE001
            print(f"[error] {e}")


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if argv and argv[0] == "selftest":
        from .selftest import run
        raise SystemExit(0 if run() else 1)
    if argv and argv[0] == "serve":
        from .saas.server import main as serve_main
        raise SystemExit(serve_main(argv[1:]))
    if argv and argv[0] == "get-model":
        from .model_fetch import ModelFetchError, fetch
        dest = argv[1] if len(argv) > 1 else "models/embedder"
        print(f"[get-model] downloading embedding model into {dest} "
              "(~90 MB, one time; inference stays fully offline)")
        try:
            report = fetch(dest)
        except ModelFetchError as e:
            print(f"[get-model] FAILED: {e}")
            raise SystemExit(1)
        _emit(report)
        try:
            import onnxruntime  # noqa: F401
            print("[get-model] onnxruntime present -> onnx backend will "
                  "auto-activate on next start (JF_EMBED_BACKEND=auto).")
        except ImportError:
            print("[get-model] model ready. Now install the runtime: "
                  "pip install onnxruntime")
        raise SystemExit(0)
    run_console()
