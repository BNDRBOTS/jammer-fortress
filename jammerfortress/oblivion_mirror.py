"""
LAYER 3 - OBLIVION-MIRROR: stack-aware runtime logic monitor.
  * Contradiction detection : semantically-close statement pairs, opposite polarity
  * Blind-spot index        : thinly-covered vs densely-covered terms
  * Drift loops             : near-duplicate entries = epistemic instability
  * Trust-vector anchoring  : stability score in [0,1]
  * Timeline audit          : ordered state trail
Operates over MemoryMesh entries. Pure python. Vectors computed once per audit.
"""
import re

from .embeddings import embed, cosine

_NEG = re.compile(r"\b(not|no|never|isn'?t|aren'?t|won'?t|can'?t|cannot|didn'?t|"
                  r"doesn'?t|don'?t|false|fail(?:ed|s)?|without|denies?|denied)\b", re.I)
_word_re = re.compile(r"[a-z0-9']+")
_STOP = set("the a an of to and or in on at is are was were be been being for with "
            "this that these those it its as by from i you he she they we my our "
            "your their but if then so do does did have has had not no will can".split())


def _polarity(text):
    return len(_NEG.findall(text or "")) % 2  # 0 = positive, 1 = negated


def _content_words(text):
    return [w for w in _word_re.findall((text or "").lower()) if w not in _STOP and len(w) > 2]


class OblivionMirror:
    def __init__(self, sim_threshold=0.55, drift_threshold=0.92):
        self.sim_threshold = sim_threshold
        self.drift_threshold = drift_threshold

    @staticmethod
    def _vecs(entries):
        return [(e, embed(e["text"])) for e in entries]

    def contradictions(self, entries, vecs=None):
        """Statement pairs that are topically close but opposite polarity."""
        out = []
        vecs = vecs if vecs is not None else self._vecs(entries)
        for i in range(len(vecs)):
            ei, vi = vecs[i]
            for j in range(i + 1, len(vecs)):
                ej, vj = vecs[j]
                sim = cosine(vi, vj)
                if sim >= self.sim_threshold and _polarity(ei["text"]) != _polarity(ej["text"]):
                    shared = set(_content_words(ei["text"])) & set(_content_words(ej["text"]))
                    if shared:
                        out.append({
                            "a": ei["text"], "b": ej["text"],
                            "similarity": round(sim, 4),
                            "shared_anchor": sorted(shared)[:6],
                        })
        return out

    def blind_spots(self, entries):
        """Content words that appear exactly once = thinly-covered blind spots."""
        freq = {}
        for e in entries:
            for w in set(_content_words(e["text"])):
                freq[w] = freq.get(w, 0) + 1
        singletons = sorted([w for w, c in freq.items() if c == 1])
        dense = sorted([(c, w) for w, c in freq.items() if c >= 3], reverse=True)
        return {
            "blind_spot_terms": singletons[:40],
            "blind_spot_count": len(singletons),
            "dense_terms": [w for _, w in dense[:15]],
        }

    def drift_loops(self, entries, vecs=None):
        """Near-duplicate entries = epistemic instability / trust-vector wobble."""
        loops = []
        vecs = vecs if vecs is not None else self._vecs(entries)
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sim = cosine(vecs[i][1], vecs[j][1])
                if sim >= self.drift_threshold:
                    loops.append({"a": vecs[i][0]["text"], "b": vecs[j][0]["text"],
                                  "similarity": round(sim, 4)})
        return loops

    def _tv(self, entries, contra_n, loop_n):
        n = max(1, len(entries))
        emo_heavy = sum(1 for e in entries if e.get("emotions") and len(e["text"].split()) < 6)
        penalty = (contra_n * 2 + loop_n + emo_heavy) / (n * 2)
        return round(max(0.0, 1.0 - penalty), 4)

    def trust_vector(self, entries):
        """Epistemic stability score in [0,1]; 1 = stable, inspectable state."""
        vecs = self._vecs(entries)
        return self._tv(entries, len(self.contradictions(entries, vecs)),
                        len(self.drift_loops(entries, vecs)))

    def timeline(self, entries):
        return [{"ts": e["ts"], "text": e["text"], "emotions": e.get("emotions", [])}
                for e in sorted(entries, key=lambda x: x["ts"])]

    def audit(self, entries):
        vecs = self._vecs(entries)
        contra = self.contradictions(entries, vecs)
        loops = self.drift_loops(entries, vecs)
        return {
            "trust_vector": self._tv(entries, len(contra), len(loops)),
            "contradictions": contra,
            "drift_loops": loops,
            "blind_spots": self.blind_spots(entries),
            "timeline": self.timeline(entries),
            "entries_examined": len(entries),
        }
