"""
Authentication primitives. PBKDF2-HMAC-SHA256 (210k iterations, per-user salt),
constant-time comparisons, opaque bearer tokens hashed at rest.
"""
import hashlib
import hmac
import re
import secrets

ITERATIONS = 210_000
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD = 10


def valid_email(email):
    return bool(email) and len(email) <= 254 and bool(EMAIL_RE.match(email))


def password_problem(pw):
    """Return a human-readable problem string, or None if the password is fine."""
    if not pw or len(pw) < MIN_PASSWORD:
        return f"password must be at least {MIN_PASSWORD} characters"
    if len(pw) > 256:
        return "password must be 256 characters or fewer"
    if pw.lower() in ("password12", "1234567890", "qwertyuiop"):
        return "password is too common"
    return None


def hash_password(pw):
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, ITERATIONS)
    return f"pbkdf2${ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(pw, stored):
    try:
        _, iters, salt_hex, dk_hex = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(dk.hex(), dk_hex)
    except Exception:  # noqa: BLE001
        return False


def new_session_token():
    return "jfs_" + secrets.token_urlsafe(32)


def new_api_key():
    return "jf_live_" + secrets.token_urlsafe(32)


def token_hash(raw):
    return hashlib.sha256((raw or "").encode()).hexdigest()
