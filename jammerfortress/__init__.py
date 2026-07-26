"""
JAMMER-FORTRESS / FUSEPACK LOCK-IN
Fusion of the source stacks (NeuroFusion Ultra, BOSS_MODE_XL, Mesh+) into one
runnable, dependency-light system, productized with auth, billing, and a web
console.

Layers:
  1 EXO-LATTICE      exo_lattice.py    tokens, rate limit, breaker, flags,
                                        event log, telemetry
  2 TRI-SKIN         cli.py + web/     terminal console + web command mesh
  3 OBLIVION-MIRROR  oblivion_mirror.py contradiction / blind-spot / drift /
                                        trust-vector anchoring / timeline
  4 BOSSWRAP-88      bosswrap.py       swarm agent router + async processor
  5 TRUST-LOCK       trust_lock.py     explicit behavior contracts
  support            memory_mesh.py    LRU semantic memory, real embeddings
  support            qr.py             pure-stdlib QR encoder (SVG, colors, logo slot)
  support            wire.py           stdlib WebSocket framing + upgrade
  support            intent.py         plain-English -> command compiler
  support            connectors.py     real Stripe / Notion / Gumroad / Neon HTTP clients
  fusion             kernel.py         orchestrator + persistence
  product            saas/             auth, billing, hardened HTTP server, web UI

No external packages required at runtime. Pure standard library.
"""

__version__ = "2.0.0-FUSEPACK"

from .kernel import Kernel  # noqa: F401
