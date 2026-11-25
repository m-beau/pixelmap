from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import panel as pn
import param
from panel.reactive import ReactiveHTML

DEFAULT_ANALYTICS_DB_PATH = Path("output") / "visitor_analytics.sqlite3"
DEFAULT_COOKIE_NAME = "pixelmap_visitor_id"
DEFAULT_COOKIE_MAX_AGE_DAYS = 365
DEFAULT_COOKIE_SAMESITE = "Lax"

_DB_INIT_LOCK = threading.Lock()


@dataclass(frozen=True)
class AnalyticsConfig:
    db_path: Path
    cookie_name: str = DEFAULT_COOKIE_NAME
    cookie_max_age_days: int = DEFAULT_COOKIE_MAX_AGE_DAYS
    cookie_secure: bool = False
    cookie_samesite: str = DEFAULT_COOKIE_SAMESITE
    trust_proxy_headers: bool = False


@dataclass(frozen=True)
class RequestContext:
    session_id: str | None
    ip_address: str
    forwarded_for: str | None
    user_agent: str | None
    path: str | None


class VisitorCookieBridge(ReactiveHTML):
    visitor_id = param.String(default="")
    cookie_name = param.String(default=DEFAULT_COOKIE_NAME)
    cookie_max_age_days = param.Integer(default=DEFAULT_COOKIE_MAX_AGE_DAYS)
    cookie_samesite = param.String(default=DEFAULT_COOKIE_SAMESITE)
    cookie_secure = param.Boolean(default=False)

    _template = '<div id="visitor-cookie-bridge" style="display: none;"></div>'

    _scripts = {
        "render": """
        const cookiePrefix = `${data.cookie_name}=`;
        const existingCookie = document.cookie
          .split('; ')
          .find((entry) => entry.startsWith(cookiePrefix));

        let visitorId = existingCookie
          ? decodeURIComponent(existingCookie.slice(cookiePrefix.length))
          : '';

        if (!visitorId) {
          visitorId = (window.crypto && window.crypto.randomUUID)
            ? window.crypto.randomUUID()
            : `${Date.now()}-${Math.random().toString(16).slice(2)}`;

          const maxAgeSeconds = data.cookie_max_age_days * 24 * 60 * 60;
          const secureFlag = data.cookie_secure ? '; Secure' : '';
          document.cookie = `${data.cookie_name}=${encodeURIComponent(visitorId)}; path=/; max-age=${maxAgeSeconds}; SameSite=${data.cookie_samesite}${secureFlag}`;
        }

        data.visitor_id = visitorId;
        """
    }


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_analytics_db_path() -> Path:
    configured_path = os.getenv("ANALYTICS_DB_PATH")
    if configured_path:
        return Path(configured_path).expanduser()
    return Path.cwd() / DEFAULT_ANALYTICS_DB_PATH


def load_analytics_config() -> AnalyticsConfig:
    return AnalyticsConfig(
        db_path=resolve_analytics_db_path(),
        cookie_name=os.getenv("ANALYTICS_COOKIE_NAME", DEFAULT_COOKIE_NAME),
        cookie_max_age_days=int(os.getenv("ANALYTICS_COOKIE_MAX_AGE_DAYS", str(DEFAULT_COOKIE_MAX_AGE_DAYS))),
        cookie_secure=_env_flag("ANALYTICS_COOKIE_SECURE", default=False),
        cookie_samesite=os.getenv("ANALYTICS_COOKIE_SAMESITE", DEFAULT_COOKIE_SAMESITE),
        trust_proxy_headers=_env_flag("ANALYTICS_TRUST_PROXY_HEADERS", default=False),
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA foreign_keys=ON;")
    return connection


def initialize_database(db_path: Path | None = None) -> Path:
    target_path = db_path or resolve_analytics_db_path()
    with _DB_INIT_LOCK:
        with _connect(target_path) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS visits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    anonymous_visitor_id TEXT,
                    session_id TEXT,
                    ip_address TEXT NOT NULL,
                    forwarded_for TEXT,
                    user_agent TEXT,
                    path TEXT,
                    visited_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS visitor_summary (
                    visitor_key TEXT PRIMARY KEY,
                    anonymous_visitor_id TEXT,
                    ip_address TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    visit_count INTEGER NOT NULL DEFAULT 0
                );
                """
            )
    return target_path


def _header_value(headers: Any, key: str) -> str | None:
    if not headers:
        return None
    value = headers.get(key)
    if value is not None:
        return value
    lowered_key = key.lower()
    for header_name, header_value in headers.items():
        if header_name.lower() == lowered_key:
            return header_value
    return None


def build_request_context(
    request: Any,
    session_id: str | None = None,
    trust_proxy_headers: bool = False,
) -> RequestContext:
    headers = getattr(request, "headers", {}) or {}
    forwarded_for = _header_value(headers, "X-Forwarded-For")
    ip_address = getattr(request, "remote_ip", None) or "unknown"

    if trust_proxy_headers and forwarded_for:
        forwarded_ips = [ip.strip() for ip in forwarded_for.split(",") if ip.strip()]
        if forwarded_ips:
            ip_address = forwarded_ips[0]

    return RequestContext(
        session_id=session_id,
        ip_address=ip_address,
        forwarded_for=forwarded_for,
        user_agent=_header_value(headers, "User-Agent"),
        path=getattr(request, "path", None),
    )


def build_visitor_key(anonymous_visitor_id: str | None, ip_address: str) -> str:
    if anonymous_visitor_id:
        return f"cookie:{anonymous_visitor_id}"
    return f"ip:{ip_address}"


def record_visit(
    anonymous_visitor_id: str | None,
    request_context: RequestContext,
    db_path: Path | None = None,
    visited_at: str | None = None,
) -> None:
    timestamp = visited_at or _utc_now()
    target_path = initialize_database(db_path)
    visitor_key = build_visitor_key(anonymous_visitor_id, request_context.ip_address)

    with _connect(target_path) as connection:
        connection.execute(
            """
            INSERT INTO visits (
                anonymous_visitor_id,
                session_id,
                ip_address,
                forwarded_for,
                user_agent,
                path,
                visited_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                anonymous_visitor_id,
                request_context.session_id,
                request_context.ip_address,
                request_context.forwarded_for,
                request_context.user_agent,
                request_context.path,
                timestamp,
            ),
        )
        connection.execute(
            """
            INSERT INTO visitor_summary (
                visitor_key,
                anonymous_visitor_id,
                ip_address,
                first_seen_at,
                last_seen_at,
                visit_count
            ) VALUES (?, ?, ?, ?, ?, 1)
            ON CONFLICT(visitor_key) DO UPDATE SET
                anonymous_visitor_id = excluded.anonymous_visitor_id,
                ip_address = excluded.ip_address,
                last_seen_at = excluded.last_seen_at,
                visit_count = visitor_summary.visit_count + 1
            """,
            (
                visitor_key,
                anonymous_visitor_id,
                request_context.ip_address,
                timestamp,
                timestamp,
            ),
        )
        connection.commit()


def record_session_end(
    anonymous_visitor_id: str | None,
    ip_address: str,
    db_path: Path | None = None,
    ended_at: str | None = None,
) -> None:
    timestamp = ended_at or _utc_now()
    target_path = initialize_database(db_path)
    visitor_key = build_visitor_key(anonymous_visitor_id, ip_address)

    with _connect(target_path) as connection:
        connection.execute(
            """
            UPDATE visitor_summary
            SET last_seen_at = ?
            WHERE visitor_key = ?
            """,
            (timestamp, visitor_key),
        )
        connection.commit()


def get_total_visits(db_path: Path | None = None) -> int:
    target_path = initialize_database(db_path)
    with _connect(target_path) as connection:
        row = connection.execute("SELECT COUNT(*) AS visit_count FROM visits").fetchone()
    return int(row["visit_count"])


def get_unique_visitors(db_path: Path | None = None) -> int:
    target_path = initialize_database(db_path)
    with _connect(target_path) as connection:
        row = connection.execute("SELECT COUNT(*) AS visitor_count FROM visitor_summary").fetchone()
    return int(row["visitor_count"])


def get_visitor_summary(visitor_key: str, db_path: Path | None = None) -> sqlite3.Row | None:
    target_path = initialize_database(db_path)
    with _connect(target_path) as connection:
        row = connection.execute(
            """
            SELECT visitor_key, anonymous_visitor_id, ip_address, first_seen_at, last_seen_at, visit_count
            FROM visitor_summary
            WHERE visitor_key = ?
            """,
            (visitor_key,),
        ).fetchone()
    return row


class AnalyticsSessionTracker:
    def __init__(self, config: AnalyticsConfig | None = None):
        self.config = config or load_analytics_config()
        self._recorded_visitor_id: str | None = None
        self._request_context: RequestContext | None = None

        initialize_database(self.config.db_path)
        self.cookie_bridge = VisitorCookieBridge(
            cookie_name=self.config.cookie_name,
            cookie_max_age_days=self.config.cookie_max_age_days,
            cookie_samesite=self.config.cookie_samesite,
            cookie_secure=self.config.cookie_secure,
            visible=False,
            width=0,
            height=0,
            margin=0,
        )
        self.cookie_bridge.param.watch(self._record_visit_from_cookie, "visitor_id")

    def _build_current_request_context(self) -> RequestContext:
        if pn.state.curdoc is None or pn.state.curdoc.session_context is None:
            raise RuntimeError("Analytics tracking requires an active Panel session context.")

        session_context = pn.state.curdoc.session_context
        return build_request_context(
            request=session_context.request,
            session_id=getattr(session_context, "id", None),
            trust_proxy_headers=self.config.trust_proxy_headers,
        )

    def _record_visit_from_cookie(self, event: param.parameterized.Event) -> None:
        visitor_id = event.new or None
        if visitor_id is None or visitor_id == self._recorded_visitor_id:
            return

        request_context = self._build_current_request_context()
        record_visit(visitor_id, request_context, db_path=self.config.db_path)
        self._recorded_visitor_id = visitor_id
        self._request_context = request_context

    def handle_session_destroyed(self, session_context: Any) -> None:
        if self._request_context is None:
            return
        record_session_end(
            anonymous_visitor_id=self._recorded_visitor_id,
            ip_address=self._request_context.ip_address,
            db_path=self.config.db_path,
        )
