"""
SQLite persistence for the product layer. WAL mode, thread-local connections,
parameterized statements only. Session and API-key tokens are stored as
SHA-256 hashes - a leaked database never leaks a usable credential.
"""
import json
import sqlite3
import threading
import time
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, pw TEXT NOT NULL,
  created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
  token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, created REAL NOT NULL,
  expires REAL NOT NULL);
CREATE TABLE IF NOT EXISTS api_keys (
  key_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, label TEXT,
  created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS subscriptions (
  user_id TEXT PRIMARY KEY, plan TEXT NOT NULL, status TEXT NOT NULL,
  stripe_customer TEXT, stripe_subscription TEXT, period_end REAL,
  updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS usage (
  user_id TEXT NOT NULL, day TEXT NOT NULL, count INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (user_id, day));
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, cat TEXT NOT NULL,
  payload TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);
"""


class DB:
    def __init__(self, path):
        self.path = path
        self._local = threading.local()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=15)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    # ---- users ------------------------------------------------------------
    def create_user(self, email, pw_hash):
        uid = str(uuid.uuid4())
        with self._conn() as c:
            c.execute("INSERT INTO users (id, email, pw, created) VALUES (?,?,?,?)",
                      (uid, email.lower(), pw_hash, time.time()))
            c.execute("INSERT INTO subscriptions (user_id, plan, status, updated) "
                      "VALUES (?,?,?,?)", (uid, "free", "active", time.time()))
        return uid

    def user_by_email(self, email):
        r = self._conn().execute("SELECT * FROM users WHERE email=?",
                                 (email.lower(),)).fetchone()
        return dict(r) if r else None

    def user_by_id(self, uid):
        r = self._conn().execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        return dict(r) if r else None

    # ---- sessions ----------------------------------------------------------
    def create_session(self, token_hash, user_id, ttl_hours=72):
        now = time.time()
        with self._conn() as c:
            c.execute("INSERT INTO sessions (token_hash, user_id, created, expires) "
                      "VALUES (?,?,?,?)", (token_hash, user_id, now, now + ttl_hours * 3600))

    def session_user(self, token_hash):
        c = self._conn()
        r = c.execute("SELECT * FROM sessions WHERE token_hash=?", (token_hash,)).fetchone()
        if not r:
            return None
        if r["expires"] < time.time():
            with c:
                c.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash,))
            return None
        return self.user_by_id(r["user_id"])

    def delete_session(self, token_hash):
        with self._conn() as c:
            c.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash,))

    # ---- api keys ----------------------------------------------------------
    def create_api_key(self, key_hash, user_id, label="default"):
        with self._conn() as c:
            c.execute("INSERT INTO api_keys (key_hash, user_id, label, created) "
                      "VALUES (?,?,?,?)", (key_hash, user_id, label, time.time()))

    def user_by_api_key(self, key_hash):
        r = self._conn().execute("SELECT user_id FROM api_keys WHERE key_hash=?",
                                 (key_hash,)).fetchone()
        return self.user_by_id(r["user_id"]) if r else None

    # ---- subscriptions -------------------------------------------------------
    def get_subscription(self, user_id):
        r = self._conn().execute("SELECT * FROM subscriptions WHERE user_id=?",
                                 (user_id,)).fetchone()
        return dict(r) if r else {"user_id": user_id, "plan": "free", "status": "active"}

    def set_subscription(self, user_id, plan, status, stripe_customer=None,
                         stripe_subscription=None, period_end=None):
        with self._conn() as c:
            c.execute(
                "INSERT INTO subscriptions (user_id, plan, status, stripe_customer, "
                "stripe_subscription, period_end, updated) VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET plan=excluded.plan, "
                "status=excluded.status, "
                "stripe_customer=COALESCE(excluded.stripe_customer, subscriptions.stripe_customer), "
                "stripe_subscription=COALESCE(excluded.stripe_subscription, subscriptions.stripe_subscription), "
                "period_end=COALESCE(excluded.period_end, subscriptions.period_end), "
                "updated=excluded.updated",
                (user_id, plan, status, stripe_customer, stripe_subscription,
                 period_end, time.time()))

    def user_by_stripe_subscription(self, sub_id):
        r = self._conn().execute(
            "SELECT user_id FROM subscriptions WHERE stripe_subscription=?",
            (sub_id,)).fetchone()
        return r["user_id"] if r else None

    # ---- usage ---------------------------------------------------------------
    @staticmethod
    def _day():
        return time.strftime("%Y-%m-%d", time.gmtime())

    def incr_usage(self, user_id, limit):
        """Atomically increment today's usage. Returns (count, allowed)."""
        day = self._day()
        with self._conn() as c:
            c.execute("INSERT INTO usage (user_id, day, count) VALUES (?,?,0) "
                      "ON CONFLICT(user_id, day) DO NOTHING", (user_id, day))
            cur = c.execute("UPDATE usage SET count = count + 1 "
                            "WHERE user_id=? AND day=? AND count < ? "
                            "RETURNING count", (user_id, day, limit)).fetchone()
        if cur is None:
            r = self._conn().execute("SELECT count FROM usage WHERE user_id=? AND day=?",
                                     (user_id, day)).fetchone()
            return (r["count"] if r else 0), False
        return cur["count"], True

    def usage_today(self, user_id):
        r = self._conn().execute("SELECT count FROM usage WHERE user_id=? AND day=?",
                                 (user_id, self._day())).fetchone()
        return r["count"] if r else 0

    # ---- audit -----------------------------------------------------------------
    def log(self, cat, payload):
        with self._conn() as c:
            c.execute("INSERT INTO audit_log (ts, cat, payload) VALUES (?,?,?)",
                      (time.time(), cat, json.dumps(payload, default=str)))

    def health(self):
        return self._conn().execute("SELECT 1").fetchone() is not None
