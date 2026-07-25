"""
Billing: plans, Stripe Checkout, and webhook signature verification.
Real Stripe API via connectors.Stripe - zero mock data. Without keys, billing
endpoints fail explicitly (503 + the exact env var to set); the core product
keeps running on the free plan. Webhook signatures are verified with
HMAC-SHA256 over 't.payload' exactly per Stripe's signing scheme.
"""
import hashlib
import hmac
import os
import time

from ..connectors import Stripe

PLANS = {
    "free": {"label": "Free", "price_usd": 0, "daily_commands": 200,
             "memory_capacity": 1000, "price_env": None},
    "pro": {"label": "Pro", "price_usd": 19, "daily_commands": 5000,
            "memory_capacity": 1000, "price_env": "STRIPE_PRICE_PRO"},
    "fortress": {"label": "Fortress", "price_usd": 49, "daily_commands": 50000,
                 "memory_capacity": 1000, "price_env": "STRIPE_PRICE_FORTRESS"},
}


def public_plans():
    return {name: {"label": p["label"], "price_usd": p["price_usd"],
                   "daily_commands": p["daily_commands"]}
            for name, p in PLANS.items()}


def plan_for(db, user_id):
    sub = db.get_subscription(user_id)
    plan = sub.get("plan", "free")
    if plan not in PLANS or sub.get("status") not in ("active", "trialing"):
        plan = "free"
    return plan, sub


def limit_for(plan):
    return PLANS.get(plan, PLANS["free"])["daily_commands"]


class BillingNotConfigured(RuntimeError):
    pass


def create_checkout(db, user, plan, base_url):
    if plan not in ("pro", "fortress"):
        raise ValueError("plan must be 'pro' or 'fortress'")
    stripe = Stripe()
    if not stripe.configured:
        raise BillingNotConfigured("Stripe is not configured. Set STRIPE_SECRET_KEY.")
    price_env = PLANS[plan]["price_env"]
    price_id = os.environ.get(price_env or "", "")
    if not price_id:
        raise BillingNotConfigured(f"No Stripe price for plan '{plan}'. Set {price_env}.")
    session = stripe.create_checkout_session(
        price_id=price_id,
        success_url=f"{base_url}/console?upgraded=1",
        cancel_url=f"{base_url}/console?canceled=1",
        customer_email=user["email"],
        client_reference_id=user["id"],
        plan=plan,
    )
    db.log("checkout_created", {"user": user["id"], "plan": plan,
                                "session": session.get("id")})
    return session


def create_portal(db, user, base_url):
    sub = db.get_subscription(user["id"])
    customer = sub.get("stripe_customer")
    if not customer:
        raise BillingNotConfigured("No Stripe customer on file; upgrade first.")
    stripe = Stripe()
    if not stripe.configured:
        raise BillingNotConfigured("Stripe is not configured. Set STRIPE_SECRET_KEY.")
    return stripe.billing_portal(customer, f"{base_url}/console")


def verify_stripe_signature(payload, sig_header, secret, tolerance=300, now=None):
    """Verify a Stripe-Signature header against the raw request body."""
    if not sig_header or not secret:
        return False
    t = None
    sigs = []
    for item in sig_header.split(","):
        k, _, v = item.strip().partition("=")
        if k == "t":
            t = v
        elif k == "v1":
            sigs.append(v)
    if not t or not sigs:
        return False
    try:
        ts = int(t)
    except ValueError:
        return False
    now = time.time() if now is None else now
    if abs(now - ts) > tolerance:
        return False
    expected = hmac.new(secret.encode(), f"{t}.".encode() + payload,
                        hashlib.sha256).hexdigest()
    return any(hmac.compare_digest(expected, s) for s in sigs)


def sign_payload(payload, secret, ts=None):
    """Produce a Stripe-Signature header (used by the self-test to prove the
    verifier against real crypto, and usable with the Stripe CLI)."""
    ts = int(time.time()) if ts is None else int(ts)
    sig = hmac.new(secret.encode(), f"{ts}.".encode() + payload,
                   hashlib.sha256).hexdigest()
    return f"t={ts},v1={sig}"


def handle_event(db, event):
    """Apply a verified Stripe event to local subscription state. Returns a
    short action string for the audit log."""
    etype = event.get("type", "")
    obj = (event.get("data") or {}).get("object") or {}
    if etype == "checkout.session.completed":
        user_id = obj.get("client_reference_id")
        plan = (obj.get("metadata") or {}).get("plan", "pro")
        if not user_id or plan not in PLANS:
            return "ignored: missing user/plan"
        db.set_subscription(user_id, plan, "active",
                            stripe_customer=obj.get("customer"),
                            stripe_subscription=obj.get("subscription"))
        db.log("subscription_activated", {"user": user_id, "plan": plan})
        return f"activated {plan} for {user_id}"
    if etype in ("customer.subscription.updated", "customer.subscription.deleted"):
        sub_id = obj.get("id")
        user_id = db.user_by_stripe_subscription(sub_id) if sub_id else None
        if not user_id:
            return "ignored: unknown subscription"
        status = obj.get("status", "canceled")
        plan = (obj.get("metadata") or {}).get("plan")
        if etype.endswith("deleted") or status in ("canceled", "unpaid", "incomplete_expired"):
            db.set_subscription(user_id, "free", "active")
            db.log("subscription_downgraded", {"user": user_id, "reason": status})
            return f"downgraded {user_id} to free ({status})"
        current = db.get_subscription(user_id)
        db.set_subscription(user_id, plan or current.get("plan", "pro"), status,
                            period_end=obj.get("current_period_end"))
        db.log("subscription_updated", {"user": user_id, "status": status})
        return f"updated {user_id}: {status}"
    return f"ignored: {etype or 'unknown event'}"
