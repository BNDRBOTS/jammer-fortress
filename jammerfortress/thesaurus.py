"""
Offline thesaurus engine (MyThes format, ~145k entries, bundled in
jammerfortress/data/). Powers the paraphrase-aware "syn" embedding backend
and synonym-aware contradiction anchors in OBLIVION-MIRROR.

Deterministic, zero network, zero third-party dependencies. Override the
bundled data with JF_THESAURUS=/path/to/th_xx.dat (matching .idx required).
"""
import os
import threading

_DEFAULT_DAT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "th_en_US_v2.dat")


def lemma_candidates(word):
    """Cheap deterministic lemma guesses, most specific first."""
    w = (word or "").lower()
    cands = [w]
    if len(w) > 3 and w.endswith("ies"):
        cands.append(w[:-3] + "y")
    if len(w) > 2 and w.endswith("es"):
        cands.append(w[:-2])
    if len(w) > 2 and w.endswith("s") and not w.endswith("ss"):
        cands.append(w[:-1])
    if len(w) > 3 and w.endswith("ed"):
        cands.append(w[:-2])
        cands.append(w[:-1])          # moved -> move
    if len(w) > 4 and w.endswith("ing"):
        cands.append(w[:-3])
        cands.append(w[:-3] + "e")    # filing -> file
    seen, out = set(), []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


class Thesaurus:
    """Lazy MyThes reader: index in memory, senses read from .dat on demand."""

    def __init__(self, dat_path=None):
        self.dat_path = dat_path or os.environ.get("JF_THESAURUS", "").strip() or _DEFAULT_DAT
        self.idx_path = os.path.splitext(self.dat_path)[0] + ".idx"
        self._offsets = None
        self._cache = {}
        self._lock = threading.Lock()

    @property
    def ok(self):
        return os.path.exists(self.dat_path) and os.path.exists(self.idx_path)

    def offsets(self):
        with self._lock:
            if self._offsets is None:
                self._offsets = self._load_index() if self.ok else {}
            return self._offsets

    def entry_count(self):
        return len(self.offsets())

    def _load_index(self):
        offsets = {}
        with open(self.idx_path, encoding="utf-8", errors="replace") as f:
            first = True
            for line in f:
                if first:                      # encoding declaration line
                    first = False
                    continue
                line = line.rstrip("\n")
                word, sep, off = line.rpartition("|")
                if not sep or not off.isdigit():
                    continue                   # entry-count line / malformed
                offsets[word.lower()] = int(off)
        return offsets

    def synonyms(self, word):
        """One-hop single-word synonyms across all senses (lowercase set)."""
        w = (word or "").lower()
        with self._lock:
            if w in self._cache:
                return self._cache[w]
        off = self.offsets().get(w)
        out = set()
        if off is not None:
            try:
                with open(self.dat_path, "rb") as f:
                    f.seek(off)
                    head = f.readline().decode("utf-8", "replace").strip()
                    parts = head.split("|")
                    n = int(parts[-1]) if len(parts) >= 2 and parts[-1].isdigit() else 0
                    for _ in range(n):
                        sense = f.readline().decode("utf-8", "replace").strip()
                        for syn in sense.split("|")[1:]:
                            syn = syn.split("(")[0].strip().lower()
                            if syn and syn != w and " " not in syn and "-" not in syn:
                                out.add(syn)
            except OSError:
                out = set()
        with self._lock:
            if len(self._cache) > 50_000:
                self._cache.clear()
            self._cache[w] = out
        return out

    def expand(self, word):
        """(base_form, synonyms) - first lemma candidate found in the index."""
        offs = self.offsets()
        for cand in lemma_candidates(word):
            if cand in offs:
                return cand, self.synonyms(cand)
        return (word or "").lower(), set()

    def bridges(self, words_a, words_b):
        """Synonym links between two word sets, e.g. ['kept~retained'].
        A bridge exists when two different surface words share a base form
        or one appears in the other's one-hop synonym set."""
        ex_a = {w: self.expand(w) for w in sorted(set(words_a))}
        ex_b = {w: self.expand(w) for w in sorted(set(words_b))}
        out = []
        for wa, (ba, sa) in ex_a.items():
            for wb, (bb, sb) in ex_b.items():
                if wa == wb:
                    continue               # direct overlap is the caller's job
                if ba == bb or bb in sa or ba in sb:
                    out.append(f"{wa}~{wb}")
        return out


_instance = None
_ilock = threading.Lock()


def get():
    global _instance
    with _ilock:
        if _instance is None:
            _instance = Thesaurus()
        return _instance


def available():
    return get().ok
