"""
LAYER 5 - TRUST-LOCK: explicit behavior contracts.
Enforcement engine, not a slogan: scans generated text against hard contracts,
strips soft language, returns an inspectable report.
Contract: never final authority; inspectable component; quality > speed.
"""
import re

BANNED = {
    "false_confidence": [r"\btrust me\b", r"\bguaranteed\b", r"\b100% (?:sure|certain)\b",
                         r"\bwithout a doubt\b", r"\bobviously\b"],
    "hedging": [r"\bit depends\b", r"\bhard to say\b", r"\bi'?m not sure but\b",
                r"\bcould be wrong\b"],
    "soft_language": [r"\bmaybe\b", r"\bjust\b", r"\bi think\b", r"\bpossibly\b",
                      r"\bperhaps\b", r"\bsort of\b", r"\bkind of\b", r"\bactually\b"],
    "ai_phrasing": [r"\bas an ai\b", r"\bi cannot\b", r"\bi'?m unable to\b",
                    r"\blanguage model\b"],
    "echo": [r"\byou'?re right\b", r"\bgreat question\b", r"\bgood point\b"],
}


class TrustLock:
    def __init__(self, max_words=400):
        self.max_words = max_words
        self._compiled = {k: [re.compile(p, re.I) for p in pats] for k, pats in BANNED.items()}

    def scan(self, text):
        """Return an inspectable violation report for a candidate output."""
        violations = []
        for cat, pats in self._compiled.items():
            for pat in pats:
                for m in pat.finditer(text or ""):
                    violations.append({"category": cat, "match": m.group(0), "span": list(m.span())})
        words = len((text or "").split())
        if words > self.max_words:
            violations.append({"category": "verbosity", "match": f"{words} words", "span": [0, 0]})
        return {
            "passed": len(violations) == 0,
            "violation_count": len(violations),
            "violations": violations,
            "contract": "never final authority; inspectable component; quality>speed",
        }

    def enforce(self, text):
        """Strip soft language and echoes; collapse whitespace."""
        out = text or ""
        for pat in self._compiled["soft_language"]:
            out = pat.sub("", out)
        for pat in self._compiled["echo"]:
            out = pat.sub("", out)
        out = re.sub(r"\s+", " ", out).strip()
        return out
