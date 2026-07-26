"""
SUPPORT - INTENT: plain-English -> command compiler.
No developer jargon required to drive the fortress: free text is compiled to a
kernel command with an inspectable confidence. Never guesses silently; the
compiled plan is always returned alongside the result.
"""
import re

# Ordered: earlier rules win. (command, pattern, has_arg, confidence)
RULES = [
    ("audit", re.compile(r"\b(audit|contradiction|contradict|blind\s*spot|inconsistenc\w*|drift|mirror|cross[- ]?exam)\b", re.I), False, 0.9),
    ("record", re.compile(r"^(?:remember|record|store|note|log)\b[:,]?\s*(.+)$", re.I | re.S), True, 0.95),
    ("query", re.compile(r"^(?:search|find|query|look\s*up|what\s+do\s+(?:i|we)\s+know\s+about)\b\s*(?:for\s+)?(.+)$", re.I | re.S), True, 0.9),
    ("recall", re.compile(r"^(?:recall|show\s+(?:me\s+)?(?:my\s+)?(?:memory|memories|recent|last))\b", re.I), False, 0.85),
    ("qr", re.compile(r"\bqr(?:\s*code)?\b(?:\s*(?:for|of|from))?\s*[:,]?\s*(.*)$", re.I | re.S), True, 0.9),
    ("trust", re.compile(r"^(?:trust|scan|vet|check)\b[:,]?\s*(.+)$", re.I | re.S), True, 0.8),
    ("persist", re.compile(r"\b(save|persist|keep\s+this|don'?t\s+lose)\b", re.I), False, 0.85),
    ("stats", re.compile(r"\b(stats|status|telemetry|health|uptime)\b", re.I), False, 0.85),
    ("route", re.compile(r"^(?:route|run|do|handle|execute|help\s+me\s+with|work\s+on)\b\s*(.+)$", re.I | re.S), True, 0.8),
]


def compile_intent(text):
    """Return {command, arg, confidence, matched}. Fallback: route the raw text
    through the swarm at low confidence (inspectable, never a dead end)."""
    raw = (text or "").strip()
    if not raw:
        return {"command": "stats", "arg": None, "confidence": 0.5,
                "matched": "empty input -> stats"}
    for cmd, pat, has_arg, conf in RULES:
        m = pat.search(raw)
        if not m:
            continue
        arg = None
        if has_arg:
            arg = (m.group(1) or "").strip() or None
            if cmd == "qr" and not arg:
                continue  # 'qr' with no payload: keep looking / fall through
        return {"command": cmd, "arg": arg, "confidence": conf,
                "matched": pat.pattern[:60]}
    return {"command": "route", "arg": raw, "confidence": 0.3,
            "matched": "fallback -> swarm route"}
