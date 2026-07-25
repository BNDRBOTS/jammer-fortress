"""
LAYER 4 - BOSSWRAP-88: swarm stack agent builder.
Routes a task through an ordered pipeline of role-agents (researcher / legal /
builder / validator). One shared blackboard threads through all roles; context
never switches. Async task processor with race-free registration.
"""
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor


class SwarmAgent:
    def __init__(self, name, keywords, handler):
        self.name = name
        self.keywords = set(keywords)
        self.handler = handler

    def affinity(self, task):
        low = task.lower()
        return sum(1 for k in self.keywords if k in low)


class BossWrap:
    def __init__(self, mesh=None, oblivion=None, trust=None):
        self.mesh = mesh
        self.oblivion = oblivion
        self.trust = trust
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.tasks = {}
        self.lock = threading.Lock()
        self.exec_counter = Counter()
        self.agents = [
            SwarmAgent("researcher", ["research", "find", "gather", "analyze", "why", "how", "data"], self._researcher),
            SwarmAgent("legal", ["law", "legal", "court", "custody", "jurisdiction", "statute", "evidence", "due process"], self._legal),
            SwarmAgent("builder", ["build", "code", "make", "create", "implement", "deploy", "draft"], self._builder),
            SwarmAgent("validator", ["check", "verify", "test", "validate", "audit"], self._validator),
        ]

    # ---- role handlers ---------------------------------------------------
    def _researcher(self, task, ctx):
        session_id = ctx.get("session_id")
        hits = self.mesh.query(task, top_k=3, session_id=session_id) if self.mesh else []
        ctx["research"] = hits
        return f"pulled {len(hits)} related memory entr" + ("ies" if len(hits) != 1 else "y")

    def _legal(self, task, ctx):
        anchors = []
        for kw in ("jurisdiction", "due process", "evidence", "custody", "venue", "statute"):
            if kw in task.lower():
                anchors.append(kw)
        ctx["legal_anchors"] = anchors
        return f"flagged {len(anchors)} legal anchor(s): {', '.join(anchors) or 'none'}"

    def _builder(self, task, ctx):
        plan = [
            "decompose request into atomic deliverables",
            "select fused modules required",
            "produce artifact + wire into kernel",
        ]
        ctx["build_plan"] = plan
        if self.mesh:
            ctx["build_record_id"] = self.mesh.record(f"BUILD PLAN :: {task}", categories=["build"],
                                                      session_id=ctx.get("session_id"))
        return f"{len(plan)}-step build plan staged"

    def _validator(self, task, ctx):
        report = {}
        if self.trust:
            report["trust_lock"] = self.trust.scan(task)
        if self.oblivion and self.mesh:
            report["trust_vector"] = self.oblivion.trust_vector(
                self.mesh.snapshot(session_id=ctx.get("session_id")))
        ctx["validation"] = report
        ok = report.get("trust_lock", {}).get("passed", True)
        return "validation PASS" if ok else "validation FLAGGED"

    # ---- routing ---------------------------------------------------------
    def route(self, task, session_id=None):
        """Rank agents by affinity, always end on validator, run the pipeline."""
        ranked = sorted(self.agents, key=lambda a: a.affinity(task), reverse=True)
        pipeline = [a for a in ranked if a.name != "validator" and a.affinity(task) > 0]
        if not pipeline:
            pipeline = [self._by_name("researcher"), self._by_name("builder")]
        pipeline.append(self._by_name("validator"))
        ctx = {"task": task, "trail": [], "session_id": session_id}
        for agent in pipeline:
            out = agent.handler(task, ctx)
            self.exec_counter[agent.name] += 1
            ctx["trail"].append({"agent": agent.name, "output": out})
        ctx.pop("session_id", None)
        return ctx

    def _by_name(self, name):
        return next(a for a in self.agents if a.name == name)

    # ---- async task processor (race-free) --------------------------------
    def submit(self, fn, *a, **k):
        tid = str(uuid.uuid4())
        with self.lock:
            self.tasks[tid] = {"status": "processing", "submitted": time.time()}
        fut = self.executor.submit(self._wrap, tid, fn, *a, **k)
        with self.lock:
            self.tasks[tid]["future"] = fut
        return tid

    def _wrap(self, tid, fn, *a, **k):
        try:
            r = fn(*a, **k)
            with self.lock:
                self.tasks[tid].update(status="completed", result=r, completed=time.time())
            return r
        except Exception as e:  # noqa: BLE001
            with self.lock:
                self.tasks[tid].update(status="failed", error=str(e))
            return None

    def result(self, tid, timeout=5.0):
        """Block on a submitted task until it finishes (or timeout). Deterministic."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                t = self.tasks.get(tid)
                fut = t.get("future") if t else None
            if fut is not None:
                try:
                    return fut.result(timeout=max(0.01, deadline - time.time()))
                except Exception:  # noqa: BLE001
                    return None
            time.sleep(0.005)
        return None

    def status(self, tid):
        with self.lock:
            t = self.tasks.get(tid, {"status": "unknown"})
            return {k: v for k, v in t.items() if k != "future"}

    def get_statistics(self):
        with self.lock:
            return {"routed": dict(self.exec_counter), "async_tasks": len(self.tasks)}
