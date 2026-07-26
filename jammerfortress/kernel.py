"""
FUSION - KERNEL: locks all five layers into one runtime.
EXO-LATTICE + MEMORY-MESH + OBLIVION-MIRROR + BOSSWRAP-88 + TRUST-LOCK +
QR + INTENT behind a single command surface, with cross-session JSON
persistence ('st4y' hotload) and encoded command triggers.
"""
import json
import os
import time

from . import qr as qr_mod
from .bosswrap import BossWrap
from .exo_lattice import ExoLattice
from .intent import compile_intent
from .memory_mesh import MemoryMesh
from .oblivion_mirror import OblivionMirror
from .trust_lock import TrustLock

TRIGGERS = {
    "st4rt": "init", "t0k3n": "token", "sh0w": "recall", "s3m4": "query",
    "p1ck": "route", "m1rr": "audit", "l0ck": "trust", "st4y": "persist",
    "m3nu": "menu", "st4t": "stats", "c0de": "qr", "1nt3nt": "intent",
}


class Kernel:
    def __init__(self, secret="FUSEPACK-DEFAULT-KEY", state_path="fortress_state.json",
                 mesh_capacity=1000, mesh_per_session=None):
        self.state_path = state_path
        self.exo = ExoLattice(secret=secret)
        self.mesh = MemoryMesh(max_capacity=mesh_capacity,
                               max_per_session=mesh_per_session)
        self.oblivion = OblivionMirror()
        self.trust = TrustLock()
        self.boss = BossWrap(mesh=self.mesh, oblivion=self.oblivion, trust=self.trust)
        self.booted_at = time.time()
        self._session = None
        if os.path.exists(state_path):
            self.load()

    # ---- lifecycle -------------------------------------------------------
    def boot(self, role="system_admin", user="BNDR"):
        self._session = self.exo.issue_token(role, user)
        self.exo.log("boot", {"role": role, "user": user, "version": "FUSEPACK"})
        return {"session": self._session, "layers": self.layers()}

    def layers(self):
        return {
            "1_exo_lattice": "online",
            "2_tri_skin": "cli+web ready",
            "3_oblivion_mirror": "armed",
            "4_bosswrap_88": f"{len(self.boss.agents)} agents",
            "5_trust_lock": "enforcing",
        }

    # ---- capabilities (feature-flag gated) -------------------------------
    def record(self, text, session_id=None, **kw):
        if not self.exo.enabled("record"):
            raise RuntimeError("record disabled")
        if not (text or "").strip():
            raise ValueError("nothing to record")
        eid = self.mesh.record(text, session_id=session_id, **kw)
        self.exo.log("record", {"id": eid}, session=session_id)
        return eid

    def recall(self, limit=5, session_id=None):
        if not self.exo.enabled("recall"):
            raise RuntimeError("recall disabled")
        return self.mesh.recall(limit=limit, session_id=session_id)

    def query(self, text, top_k=3, session_id=None):
        if not self.exo.enabled("query"):
            raise RuntimeError("query disabled")
        t0 = time.time()
        r = self.mesh.query(text, top_k=top_k, session_id=session_id)
        self.exo.telemetry.observe("query_seconds", time.time() - t0)
        return r

    def route(self, task, session_id=None):
        if not self.exo.enabled("route"):
            raise RuntimeError("route disabled")
        self.exo.log("route", {"task": (task or "")[:80]}, session=session_id)
        return self.boss.route(task or "", session_id=session_id)

    def audit(self, session_id=None):
        if not self.exo.enabled("audit"):
            raise RuntimeError("audit disabled")
        return self.oblivion.audit(self.mesh.snapshot(session_id=session_id))

    def enforce(self, text):
        if not self.exo.enabled("trust"):
            raise RuntimeError("trust disabled")
        return {"scan": self.trust.scan(text), "enforced": self.trust.enforce(text)}

    def qr(self, text, ecl="M", dark="#000000", light="#ffffff",
           shape="square", logo_slot=0.0):
        if not self.exo.enabled("qr"):
            raise RuntimeError("qr disabled")
        if not (text or "").strip():
            raise ValueError("nothing to encode")
        if logo_slot and logo_slot > 0:
            ecl = "H"  # reserve center safely only at max error correction
        code = qr_mod.make(text, ecl=ecl)
        self.exo.log("qr", {"len": len(text), "version": code.version, "ecl": ecl})
        return {"svg": code.svg(dark=dark, light=light, shape=shape, logo_slot=logo_slot),
                "version": code.version, "ecl": ecl, "size": code.size}

    def intent(self, text, session_id=None):
        if not self.exo.enabled("intent"):
            raise RuntimeError("intent disabled")
        plan = compile_intent(text)
        result = self.command(plan["command"], plan.get("arg"), session_id=session_id)
        return {"plan": plan, "result": result}

    # ---- persistence (st4y hotload) ---------------------------------------
    def persist(self):
        state = {
            "version": "FUSEPACK",
            "saved_at": time.time(),
            "memory": self.mesh.snapshot(),
            "telemetry": self.exo.telemetry.render(),
        }
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, self.state_path)  # atomic: never a half-written state file
        return {"saved": len(state["memory"]), "path": self.state_path}

    def load(self):
        with open(self.state_path) as f:
            state = json.load(f)
        self.mesh.load(state.get("memory", []))
        return {"restored": len(state.get("memory", []))}

    # ---- command dispatch (TRI-SKIN + web console use this) ----------------
    def command(self, trigger, arg=None, session_id=None):
        action = TRIGGERS.get(trigger, trigger)
        if action == "init":
            return self.boot()
        if action == "token":
            return {"token": self.exo.issue_token("system_operator", arg or "guest")}
        if action == "record":
            return {"id": self.record(arg or "", session_id=session_id)}
        if action == "recall":
            limit = 5
            if arg:
                try:
                    limit = max(1, min(50, int(arg)))
                except ValueError:
                    limit = 5
            return self.recall(limit=limit, session_id=session_id)
        if action == "query":
            return self.query(arg or "", session_id=session_id)
        if action == "route":
            return self.route(arg or "", session_id=session_id)
        if action == "audit":
            return self.audit(session_id=session_id)
        if action == "trust":
            return self.enforce(arg or "")
        if action == "qr":
            return self.qr(arg or "")
        if action == "intent":
            return self.intent(arg or "", session_id=session_id)
        if action == "persist":
            return self.persist()
        if action == "stats":
            return self.stats()
        if action == "menu":
            return {"triggers": TRIGGERS}
        return {"error": f"unknown trigger '{trigger}'"}

    def stats(self):
        from . import embeddings as _emb
        return {
            "uptime_s": round(time.time() - self.booted_at, 3),
            "exo": self.exo.get_statistics(),
            "mesh": self.mesh.get_statistics(),
            "boss": self.boss.get_statistics(),
            "embeddings": _emb.info(),
            "trust_vector": self.oblivion.trust_vector(self.mesh.snapshot()),
        }
