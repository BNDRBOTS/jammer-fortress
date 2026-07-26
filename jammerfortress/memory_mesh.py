"""
SUPPORT - MEMORY-MESH: unified semantic memory.
O(1) LRU, category index, session isolation, emotion tagging, drift detection,
real embeddings. Thread-safe. Deterministic. No network.
"""
import threading
import time
import uuid
from collections import OrderedDict, defaultdict

from .embeddings import embed, cosine

EMO_LEX = {
    "frustration": ["sad", "tired", "angry", "bullshit", "ignored", "pissed", "broken", "fuck"],
    "drive": ["hope", "goal", "focus", "future", "ready", "build", "win", "execute"],
    "fear": ["worried", "scared", "afraid", "risk", "lose", "threat"],
}


class MemoryMesh:
    def __init__(self, max_capacity=1000, max_per_session=None):
        self.max_capacity = max_capacity
        self.max_per_session = max_per_session  # per-user quota; None = off
        self.store = []                      # list of dict entries
        self.index = {}                      # id -> position
        self.recency = OrderedDict()         # id -> ts  (O(1) LRU)
        self.category_index = defaultdict(set)
        self.session_index = defaultdict(list)
        self.lock = threading.RLock()
        self.stats = {"insertions": 0, "searches": 0, "retrievals": 0, "drift": 0}

    # ---- analysis --------------------------------------------------------
    @staticmethod
    def detect_emotions(text):
        low = (text or "").lower()
        found = []
        for emo, words in EMO_LEX.items():
            if any(w in low for w in words):
                found.append(emo)
        return found

    def _detect_drift(self, text, session_id=None):
        for e in self.store:
            if e["text"] == text and (session_id is None or e["session_id"] == session_id):
                return True
        return False

    # ---- write -----------------------------------------------------------
    def record(self, text, categories=None, session_id=None, source="live"):
        with self.lock:
            eid = str(uuid.uuid4())
            drifted = self._detect_drift(text, session_id)
            entry = {
                "id": eid,
                "text": text,
                "vector": embed(text),
                "emotions": self.detect_emotions(text),
                "drift": drifted,
                "categories": list(categories or []),
                "session_id": session_id,
                "source": source,
                "ts": time.time(),
            }
            if drifted:
                self.stats["drift"] += 1
            sess_ids = self.session_index.get(session_id) if session_id else None
            if self.max_per_session and sess_ids and len(sess_ids) >= self.max_per_session:
                # quota hit: this user's oldest entry makes room, nobody else's
                old_id = sess_ids.pop(0)
                self.recency.pop(old_id, None)
                pos = self.index.pop(old_id)
                old = self.store[pos]
                for c in old["categories"]:
                    self.category_index[c].discard(old_id)
                self.store[pos] = entry
                self.index[eid] = pos
            elif len(self.store) >= self.max_capacity:
                old_id, _ = self.recency.popitem(last=False)
                pos = self.index.pop(old_id)
                old = self.store[pos]
                for c in old["categories"]:
                    self.category_index[c].discard(old_id)
                if old.get("session_id"):
                    try:
                        self.session_index[old["session_id"]].remove(old_id)
                    except ValueError:
                        pass
                self.store[pos] = entry
                self.index[eid] = pos
            else:
                self.store.append(entry)
                self.index[eid] = len(self.store) - 1
            self.recency[eid] = entry["ts"]
            self.recency.move_to_end(eid)
            for c in entry["categories"]:
                self.category_index[c].add(eid)
            if session_id:
                self.session_index[session_id].append(eid)
            self.stats["insertions"] += 1
            return eid

    def record_bulk(self, texts, **kw):
        return [self.record(t, **kw) for t in texts]

    # ---- read ------------------------------------------------------------
    def recall(self, limit=5, session_id=None):
        with self.lock:
            entries = self.store
            if session_id:
                ids = set(self.session_index.get(session_id, []))
                entries = [e for e in self.store if e["id"] in ids]
            return [self._public(e) for e in entries[-limit:]]

    def query(self, text, top_k=3, min_similarity=0.05, category_filter=None, session_id=None):
        with self.lock:
            self.stats["searches"] += 1
            qv = embed(text)
            cand = self.store
            if session_id:
                ids = set(self.session_index.get(session_id, []))
                cand = [e for e in cand if e["id"] in ids]
            if category_filter:
                ids = set()
                for c in category_filter:
                    ids |= self.category_index.get(c, set())
                cand = [e for e in cand if e["id"] in ids]
            scored = []
            for e in cand:
                sim = cosine(qv, e["vector"])
                if sim >= min_similarity:
                    scored.append((sim, e))
            scored.sort(key=lambda x: x[0], reverse=True)
            out = []
            for sim, e in scored[:top_k]:
                self.recency[e["id"]] = time.time()
                self.recency.move_to_end(e["id"])
                self.stats["retrievals"] += 1
                pub = self._public(e)
                pub["similarity"] = round(sim, 4)
                out.append(pub)
            return out

    @staticmethod
    def _public(e):
        return {k: e[k] for k in ("id", "text", "emotions", "drift", "categories", "ts")}

    def snapshot(self, session_id=None):
        with self.lock:
            out = []
            for e in self.store:
                if session_id is not None and e["session_id"] != session_id:
                    continue
                out.append({k: e[k] for k in ("id", "text", "emotions", "drift",
                                              "categories", "session_id", "ts")})
            return out

    def load(self, entries):
        with self.lock:
            for e in entries:
                self.record(e["text"], categories=e.get("categories"),
                            session_id=e.get("session_id"), source="restore")

    def get_statistics(self):
        with self.lock:
            return {**self.stats, "used": len(self.store), "capacity": self.max_capacity,
                    "per_session_quota": self.max_per_session}
