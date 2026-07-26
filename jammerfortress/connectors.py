"""
SUPPORT - CONNECTORS: real HTTP clients for Stripe, Notion, Gumroad, Neon.
Pure stdlib (urllib). No mock data anywhere: every method hits the real API.
When a credential is absent the connector raises ConnectorNotConfigured with
the exact environment variable to set - an explicit, graceful failure mode,
never a simulation.
"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request


class ConnectorError(RuntimeError):
    def __init__(self, service, status, detail):
        self.service = service
        self.status = status
        self.detail = detail
        super().__init__(f"{service} error (HTTP {status}): {detail}")


class ConnectorNotConfigured(RuntimeError):
    def __init__(self, service, env_var):
        self.service = service
        self.env_var = env_var
        super().__init__(f"{service} is not configured. Set the {env_var} "
                         f"environment variable to enable it.")


def _request(service, url, method="GET", headers=None, body=None, timeout=20):
    req = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise ConnectorError(service, e.code, e.read().decode("utf-8", "replace")[:500])
    except urllib.error.URLError as e:
        raise ConnectorError(service, 0, str(e.reason))


def _form(fields):
    return urllib.parse.urlencode({k: v for k, v in fields.items() if v is not None}).encode()


class Stripe:
    BASE = "https://api.stripe.com"

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("STRIPE_SECRET_KEY")

    @property
    def configured(self):
        return bool(self.api_key)

    def _require(self):
        if not self.api_key:
            raise ConnectorNotConfigured("Stripe", "STRIPE_SECRET_KEY")

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/x-www-form-urlencoded"}

    def create_checkout_session(self, price_id, success_url, cancel_url,
                                customer_email, client_reference_id, plan):
        self._require()
        fields = {
            "mode": "subscription",
            "line_items[0][price]": price_id,
            "line_items[0][quantity]": "1",
            "success_url": success_url,
            "cancel_url": cancel_url,
            "customer_email": customer_email,
            "client_reference_id": client_reference_id,
            "metadata[plan]": plan,
            "subscription_data[metadata][plan]": plan,
        }
        return _request("Stripe", f"{self.BASE}/v1/checkout/sessions", "POST",
                        self._headers(), _form(fields))

    def billing_portal(self, customer, return_url):
        self._require()
        return _request("Stripe", f"{self.BASE}/v1/billing_portal/sessions", "POST",
                        self._headers(), _form({"customer": customer,
                                                "return_url": return_url}))

    def get(self, path):
        self._require()
        return _request("Stripe", f"{self.BASE}{path}", "GET", self._headers())


class Notion:
    BASE = "https://api.notion.com/v1"
    VERSION = "2022-06-28"

    def __init__(self, token=None):
        self.token = token or os.environ.get("NOTION_TOKEN")

    @property
    def configured(self):
        return bool(self.token)

    def _require(self):
        if not self.token:
            raise ConnectorNotConfigured("Notion", "NOTION_TOKEN")

    def _headers(self):
        return {"Authorization": f"Bearer {self.token}",
                "Notion-Version": self.VERSION,
                "Content-Type": "application/json"}

    def search(self, query, page_size=10):
        self._require()
        body = json.dumps({"query": query, "page_size": page_size}).encode()
        return _request("Notion", f"{self.BASE}/search", "POST", self._headers(), body)

    def append_paragraph(self, block_id, text):
        self._require()
        body = json.dumps({"children": [{"object": "block", "type": "paragraph",
                                         "paragraph": {"rich_text": [{"type": "text",
                                                                      "text": {"content": text[:2000]}}]}}]}).encode()
        return _request("Notion", f"{self.BASE}/blocks/{block_id}/children", "PATCH",
                        self._headers(), body)


class Gumroad:
    BASE = "https://api.gumroad.com/v2"

    def __init__(self, product_id=None):
        self.product_id = product_id or os.environ.get("GUMROAD_PRODUCT_ID")

    @property
    def configured(self):
        return bool(self.product_id)

    def verify_license(self, license_key, increment_uses=False):
        if not self.product_id:
            raise ConnectorNotConfigured("Gumroad", "GUMROAD_PRODUCT_ID")
        body = _form({"product_id": self.product_id, "license_key": license_key,
                      "increment_uses_count": "true" if increment_uses else "false"})
        return _request("Gumroad", f"{self.BASE}/licenses/verify", "POST",
                        {"Content-Type": "application/x-www-form-urlencoded"}, body)


class Neon:
    BASE = "https://console.neon.tech/api/v2"

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("NEON_API_KEY")

    @property
    def configured(self):
        return bool(self.api_key)

    def _require(self):
        if not self.api_key:
            raise ConnectorNotConfigured("Neon", "NEON_API_KEY")

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}

    def list_projects(self):
        self._require()
        return _request("Neon", f"{self.BASE}/projects", "GET", self._headers())


def status():
    """Inspectable connector configuration state (no secrets echoed)."""
    return {
        "stripe": Stripe().configured,
        "notion": Notion().configured,
        "gumroad": Gumroad().configured,
        "neon": Neon().configured,
    }
