"""``/usage-stats``: weekly unique-visitor counts of the hosted GUI, as JSON.

The visitor database lives on the server, where GitHub cannot reach it. This
endpoint publishes the one aggregate the usage graph needs, so that
``.github/workflows/usage-stats.yml`` can fetch it and redraw the graph
(``scripts/update_usage_graph.py``). It returns counts per week and nothing
else: no visitor IDs, IP addresses or user agents leave the server.

``entrypoint.sh`` loads this module through ``panel serve --plugins``, which
adds ``ROUTES`` to the server. Like ``/liveness``, a request builds no Bokeh
session and records no visit.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from typing import Any

from tornado.ioloop import IOLoop
from tornado.web import RequestHandler

from pixelmap.analytics import get_weekly_unique_visitors

ENDPOINT = "/usage-stats"

# The endpoint is public, so the query runs at most once per TTL however
# often it is requested.
CACHE_TTL_SECONDS = 300

_cache_lock = threading.Lock()
_cached_payload: dict[str, Any] | None = None
_cache_expires_at = 0.0


def build_payload() -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "week_starts_on": "monday",
        "timezone": "UTC",
        "weeks": get_weekly_unique_visitors(),
    }


def get_cached_payload() -> dict[str, Any]:
    global _cached_payload, _cache_expires_at
    with _cache_lock:
        if _cached_payload is None or time.monotonic() >= _cache_expires_at:
            _cached_payload = build_payload()
            _cache_expires_at = time.monotonic() + CACHE_TTL_SECONDS
        return _cached_payload


class UsageStatsHandler(RequestHandler):
    async def get(self) -> None:
        # Off the event loop: it also serves every session and /liveness.
        payload = await IOLoop.current().run_in_executor(None, get_cached_payload)
        self.set_header("Content-Type", "application/json; charset=UTF-8")
        self.set_header("Cache-Control", f"public, max-age={CACHE_TTL_SECONDS}")
        self.write(payload)


ROUTES = [(ENDPOINT, UsageStatsHandler)]
