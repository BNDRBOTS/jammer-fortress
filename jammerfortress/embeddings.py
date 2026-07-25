"""
Pluggable LOCAL embedding backends. Inference never touches the network.

backend "hash" : signed random projection over word tokens + char trigrams.
                 Zero dependencies, deterministic. Honest quality ceiling:
                 paraphrases that share no vocabulary score low, so
                 OBLIVION-MIRROR can miss contradictions like
                 "the hearing was rescheduled" vs "the date moved".
backend "onnx" : real sentence-transformer (default: all-MiniLM-L6-v2) run
                 fully offline via onnxruntime. Network is used ONLY by the
                 one-time `python3 -m jammerfortress get-model` download,
                 never at embed time.

Selection (env JF_EMBED_BACKEND = auto | hash | onnx, default auto):
  auto -> onnx when onnxruntime + model files are present, else hash
          (fallback reason is recorded and surfaced in stats/boot logs).
  onnx -> hard error naming the exact missing piece. No silent fallback.
Model dir search order: $JF_EMBED_MODEL_DIR, ./models/embedder,
<package>/models/embedder.
"""
import hashlib
import json
import math
import os
import re
import threading
import unicodedata

DIM = 256  # hash-backend dimension (kept for API compatibility)
_word_re = re.compile(r"[a-z0-9']+")


class EmbeddingsUnavailable(RuntimeError):
    """Raised when an explicitly requested backend cannot be built."""


# --------------------------------------------------------------- hash backend
def _hash_embed(text, dim=DIM):
    v = [0.0] * dim
    toks = _tokens(text)
    if not toks:
        return v
    for t in toks:
        h = hashlib.sha1(t.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "big") % dim
        sign = 1.0 if (h[4] & 1) else -1.0
        v[idx] += sign
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _tokens(text):
    text = (text or "").lower()
    words = _word_re.findall(text)
    toks = list(words)
    joined = " ".join(words)
    for i in range(len(joined) - 2):
        tri = joined[i:i + 3]
        if tri.strip():
            toks.append("@" + tri)
    return toks


class HashBackend:
    name = "hash"

    def __init__(self, dim=DIM):
        self.dim = dim

    def embed(self, text):
        return _hash_embed(text, self.dim)


# ------------------------------------------------------- WordPiece tokenizer
def _is_punct(ch):
    cp = ord(ch)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(ch).startswith("P")


class WordPieceTokenizer:
    """BERT-style WordPiece: basic tokenize (lowercase, strip accents, split
    punctuation) + greedy longest-match subwords with '##' continuations."""

    def __init__(self, vocab, unk="[UNK]", cls_tok="[CLS]", sep_tok="[SEP]",
                 lower=True, max_word_chars=100):
        self.vocab = dict(vocab)
        for required in (unk, cls_tok, sep_tok):
            if required not in self.vocab:
                raise ValueError(f"vocab missing required token {required}")
        self.unk, self.cls_tok, self.sep_tok = unk, cls_tok, sep_tok
        self.lower = lower
        self.max_word_chars = max_word_chars

    @classmethod
    def from_dir(cls, model_dir):
        vocab_txt = os.path.join(model_dir, "vocab.txt")
        tok_json = os.path.join(model_dir, "tokenizer.json")
        if os.path.exists(vocab_txt):
            with open(vocab_txt, encoding="utf-8") as f:
                vocab = {line.rstrip("\n"): i for i, line in enumerate(f)}
            return cls(vocab)
        if os.path.exists(tok_json):
            with open(tok_json, encoding="utf-8") as f:
                data = json.load(f)
            vocab = data.get("model", {}).get("vocab")
            if isinstance(vocab, dict):
                return cls(vocab)
        raise EmbeddingsUnavailable(
            f"no vocab.txt or tokenizer.json in {model_dir}")

    def _basic(self, text):
        text = unicodedata.normalize("NFD", text or "")
        out = []
        word = []
        for ch in text:
            if unicodedata.category(ch) == "Mn":      # strip accents
                continue
            if self.lower:
                ch = ch.lower()
            if ch.isspace():
                if word:
                    out.append("".join(word))
                    word = []
            elif _is_punct(ch):
                if word:
                    out.append("".join(word))
                    word = []
                out.append(ch)
            else:
                word.append(ch)
        if word:
            out.append("".join(word))
        return out

    def _wordpiece(self, word):
        if len(word) > self.max_word_chars:
            return [self.unk]
        pieces, start = [], 0
        while start < len(word):
            end = len(word)
            piece = None
            while start < end:
                cand = word[start:end]
                if start > 0:
                    cand = "##" + cand
                if cand in self.vocab:
                    piece = cand
                    break
                end -= 1
            if piece is None:
                return [self.unk]
            pieces.append(piece)
            start = end
        return pieces

    def tokenize(self, text):
        toks = []
        for word in self._basic(text):
            toks.extend(self._wordpiece(word))
        return toks

    def encode(self, text, max_len=256):
        """[CLS] tokens... [SEP], truncated to max_len. Returns id list."""
        body = self.tokenize(text)[:max(0, max_len - 2)]
        toks = [self.cls_tok] + body + [self.sep_tok]
        return [self.vocab[t] for t in toks]


# --------------------------------------------------------------- onnx backend
class OnnxBackend:
    """Sentence-transformer inference: tokenize -> transformer -> masked mean
    pooling -> L2 normalize. `session`/`tokenizer` are injectable for tests."""

    name = "onnx"

    def __init__(self, model_dir=None, session=None, tokenizer=None, max_len=256):
        if session is None:
            import onnxruntime  # hard dep only when actually used
            model_path = os.path.join(model_dir, "model.onnx")
            if not os.path.exists(model_path):
                raise EmbeddingsUnavailable(f"missing {model_path}")
            session = onnxruntime.InferenceSession(
                model_path, providers=["CPUExecutionProvider"])
        self.session = session
        self.tokenizer = tokenizer or WordPieceTokenizer.from_dir(model_dir)
        self.model_dir = model_dir
        self.max_len = max_len
        self._input_names = [i.name for i in session.get_inputs()]
        self.dim = None
        try:  # static hidden size when the graph declares it
            shape = session.get_outputs()[0].shape
            if shape and isinstance(shape[-1], int):
                self.dim = shape[-1]
        except Exception:  # noqa: BLE001 - discovered on first embed instead
            pass

    def embed(self, text):
        import numpy as np
        ids = self.tokenizer.encode(text, self.max_len)
        arr = np.array([ids], dtype=np.int64)
        mask = np.ones_like(arr)
        feeds = {}
        for name in self._input_names:
            if "input_ids" in name:
                feeds[name] = arr
            elif "attention_mask" in name:
                feeds[name] = mask
            elif "token_type" in name:
                feeds[name] = np.zeros_like(arr)
        hidden = self.session.run(None, feeds)[0]   # (1, seq, hidden)
        vec = self._pool(hidden, mask)
        self.dim = len(vec)
        return vec

    @staticmethod
    def _pool(hidden, mask):
        """Attention-masked mean pooling + L2 normalization."""
        import numpy as np
        hidden = np.asarray(hidden, dtype=np.float64)
        m = np.asarray(mask, dtype=np.float64)[..., None]
        summed = (hidden * m).sum(axis=1)
        counts = np.clip(m.sum(axis=1), 1e-9, None)
        v = (summed / counts)[0]
        norm = float(np.sqrt((v * v).sum())) or 1.0
        return [float(x) for x in (v / norm)]


# ----------------------------------------------------------- backend selection
_lock = threading.RLock()
_active = None
_fallback_reason = None


def _model_dir_candidates():
    env = os.environ.get("JF_EMBED_MODEL_DIR")
    cands = [env] if env else []
    cands.append(os.path.join(os.getcwd(), "models", "embedder"))
    cands.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "models", "embedder"))
    return cands


def find_model_dir():
    """First candidate dir containing model.onnx + a vocab, else None."""
    for d in _model_dir_candidates():
        if d and os.path.exists(os.path.join(d, "model.onnx")) and (
                os.path.exists(os.path.join(d, "vocab.txt"))
                or os.path.exists(os.path.join(d, "tokenizer.json"))):
            return d
    return None


def _build():
    mode = os.environ.get("JF_EMBED_BACKEND", "auto").strip().lower() or "auto"
    if mode == "hash":
        return HashBackend(), None
    model_dir = find_model_dir()
    if mode == "onnx":
        if model_dir is None:
            raise EmbeddingsUnavailable(
                "JF_EMBED_BACKEND=onnx but no model found. Run: "
                "python3 -m jammerfortress get-model  (one-time download), "
                "or set JF_EMBED_MODEL_DIR.")
        try:
            return OnnxBackend(model_dir), None
        except ImportError as e:
            raise EmbeddingsUnavailable(
                "JF_EMBED_BACKEND=onnx but onnxruntime is not installed. "
                "Run: pip install onnxruntime") from e
    # auto
    if model_dir is not None:
        try:
            return OnnxBackend(model_dir), None
        except ImportError:
            return HashBackend(), (f"model present at {model_dir} but "
                                   "onnxruntime not installed (pip install onnxruntime)")
        except Exception as e:  # noqa: BLE001 - corrupt model must not kill boot
            return HashBackend(), f"onnx backend failed to load: {e}"
    return HashBackend(), ("no local model; hash backend active "
                           "(upgrade: python3 -m jammerfortress get-model)")


def backend():
    global _active, _fallback_reason
    with _lock:
        if _active is None:
            _active, _fallback_reason = _build()
        return _active


def set_backend(b):
    """Force a backend instance (tests), or None to re-select lazily."""
    global _active, _fallback_reason
    with _lock:
        _active, _fallback_reason = b, None


def reload():
    set_backend(None)


def info():
    b = backend()
    out = {"backend": b.name, "dim": b.dim}
    if getattr(b, "model_dir", None):
        out["model_dir"] = b.model_dir
    if _fallback_reason:
        out["fallback_reason"] = _fallback_reason
    return out


# ------------------------------------------------------------------ public API
def embed(text, dim=DIM):
    """Embed via the active backend. `dim` is honored by the hash backend only
    (model backends have a fixed hidden size)."""
    b = backend()
    if isinstance(b, HashBackend) and dim != b.dim:
        return _hash_embed(text, dim)
    return b.embed(text)


def cosine(a, b):
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
