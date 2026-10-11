"""Minimal REST client for Kalshi market data.

Market data endpoints (markets, order book, trades) are public, so no key is
needed to use this module. Only the standard library is used.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

PROD = "https://external-api.kalshi.com"
DEMO = "https://external-api.demo.kalshi.co"
API_PREFIX = "/trade-api/v2"


class KalshiREST:
    def __init__(self, base_url: str = PROD, signer=None, timeout: float = 10.0, max_retries: int = 4) -> None:
        self.base_url = base_url.rstrip("/")
        self.signer = signer
        self.timeout = timeout
        self.max_retries = max_retries

    def get(self, path: str, **params) -> dict:
        full_path = API_PREFIX + path
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v not in (None, "")})
        url = self.base_url + full_path + ("?" + query if query else "")
        delay = 0.5
        for attempt in range(self.max_retries + 1):
            headers = {"Accept": "application/json"}
            if self.signer is not None:
                headers.update(self.signer.headers("GET", full_path))
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                # Back off on rate limits and server errors; anything else is ours to fix.
                if e.code not in (429, 500, 502, 503, 504) or attempt == self.max_retries:
                    raise
            except urllib.error.URLError:
                if attempt == self.max_retries:
                    raise
            time.sleep(delay)
            delay *= 2
        raise AssertionError("unreachable")

    def market(self, ticker: str) -> dict:
        return self.get(f"/markets/{ticker}")["market"]

    def markets(self, **filters) -> list[dict]:
        return self._paged("/markets", "markets", **filters)

    def orderbook(self, ticker: str, depth: int = 0) -> dict:
        """Returns {"yes_dollars": [[price, size], ...], "no_dollars": [...]}."""
        return self.get(f"/markets/{ticker}/orderbook", depth=depth)["orderbook_fp"]

    def trades(self, ticker: str, limit: int = 1000, max_pages: int = 1) -> list[dict]:
        return self._paged("/markets/trades", "trades", max_pages=max_pages, ticker=ticker, limit=limit)

    def _paged(self, path: str, key: str, max_pages: int = 50, **params) -> list[dict]:
        out: list[dict] = []
        cursor = ""
        for _ in range(max_pages):
            page = self.get(path, cursor=cursor, **params)
            out.extend(page.get(key, []))
            cursor = page.get("cursor") or ""
            if not cursor:
                break
        return out


def tick_size(market: dict, price_ticks: int) -> int:
    """Tick size (in 1e-4 dollars) at a price, from the market's ``price_ranges``.

    Kalshi markets do not all trade in whole cents, and some use a finer grid
    near 0 and 1 than in the middle. ``price_ranges`` is the source of truth.
    """
    from ..fixedpoint import price_to_ticks

    for band in market.get("price_ranges", []):
        if price_to_ticks(band["start"]) <= price_ticks <= price_to_ticks(band["end"]):
            return price_to_ticks(band["step"])
    return 100
