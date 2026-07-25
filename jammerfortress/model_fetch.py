"""
One-time embedding-model download. This is the ONLY place network is used,
and only when you run it explicitly (`python3 -m jammerfortress get-model`)
or opt in with JF_EMBED_AUTO_FETCH=1. Inference is always offline.

Default model: sentence-transformers/all-MiniLM-L6-v2 (ONNX export published
in the official repo; ~90 MB). Override with JF_EMBED_MODEL_URL to pin a
mirror or a different repo with the same file layout.
"""
import os
import time
import urllib.request

DEFAULT_BASE = ("https://huggingface.co/sentence-transformers/"
                "all-MiniLM-L6-v2/resolve/main")
FILES = {
    "model.onnx": "onnx/model.onnx",
    "vocab.txt": "vocab.txt",
}
# Honest floor sizes: reject truncated downloads outright.
MIN_SIZES = {"model.onnx": 5_000_000, "vocab.txt": 100_000}
REQUIRED_VOCAB_TOKENS = ("[UNK]", "[CLS]", "[SEP]")


class ModelFetchError(RuntimeError):
    pass


def validate(dest_dir):
    """Verify a model dir is structurally sound. Raises ModelFetchError."""
    report = {"dir": os.path.abspath(dest_dir), "files": {}}
    for fname, floor in MIN_SIZES.items():
        path = os.path.join(dest_dir, fname)
        if not os.path.exists(path):
            raise ModelFetchError(f"missing {fname} in {dest_dir}")
        size = os.path.getsize(path)
        if size < floor:
            raise ModelFetchError(
                f"{fname} is {size} bytes (< {floor}); truncated or corrupt")
        report["files"][fname] = size
    with open(os.path.join(dest_dir, "vocab.txt"), encoding="utf-8") as f:
        head = f.read(2_000_000)
    for tok in REQUIRED_VOCAB_TOKENS:
        if tok not in head:
            raise ModelFetchError(f"vocab.txt missing required token {tok}")
    return report


def _download(url, dest, attempts=3):
    last = None
    for i in range(attempts):
        tmp = dest + ".part"
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "jammer-fortress/2.0"})
            with urllib.request.urlopen(req, timeout=180) as r, \
                    open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, dest)   # atomic: never leaves a half-file in place
            return
        except Exception as e:  # noqa: BLE001
            last = e
            try:
                os.unlink(tmp)
            except OSError:
                pass
            time.sleep(2 ** i)
    raise ModelFetchError(f"download failed after {attempts} attempts: "
                          f"{url} ({last})")


def fetch(dest_dir="models/embedder", base_url=None):
    """Download any missing model files into dest_dir, then validate."""
    base = (base_url or os.environ.get("JF_EMBED_MODEL_URL")
            or DEFAULT_BASE).rstrip("/")
    os.makedirs(dest_dir, exist_ok=True)
    for fname, rel in FILES.items():
        dest = os.path.join(dest_dir, fname)
        if os.path.exists(dest) and os.path.getsize(dest) >= MIN_SIZES[fname]:
            continue  # idempotent: keep good files, re-fetch bad/missing ones
        _download(base + "/" + rel, dest)
    return validate(dest_dir)
