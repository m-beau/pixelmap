"""Catch-all error reporting: toast for the user, webhook for the deployer.

An exception escaping a GUI callback must never kill the server or leave a
session silently broken. Every such error is routed through `report_error`,
which

* logs the full traceback (``pixelmap.errors`` logger),
* shows an error toast in the session that hit it (if there is one), and
* POSTs a JSON description to a webhook, if the deployer configured one.

Webhook configuration (environment variables, read on every report):

``PIXELMAP_ERROR_WEBHOOK_URL``
    http(s) endpoint to POST errors to. Unset or blank disables the webhook.
``PIXELMAP_ERROR_WEBHOOK_TOKEN``
    Optional; sent as ``Authorization: Bearer <token>``.
``PIXELMAP_ERROR_WEBHOOK_TIMEOUT``
    Seconds before a POST is abandoned (default 5).
``PIXELMAP_ERROR_WEBHOOK_DEDUPE_SECONDS``
    Identical errors (same type, message and location) are posted at most
    once per window (default 60; 0 posts every occurrence).

The payload carries ``text`` (Slack, Mattermost) and ``content`` (Discord)
summaries plus an ``event`` object with the structured details, so it can be
pointed straight at a chat webhook or at a custom collector.

Webhook delivery runs on a single daemon thread with a bounded queue: a slow
or unreachable endpoint can neither block the Bokeh event loop nor pile up
memory, and excess reports are dropped rather than queued forever.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import math
import os
import queue
import socket
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import panel as pn

from pixelmap import __version__

logger = logging.getLogger("pixelmap.errors")

DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_DEDUPE_SECONDS = 60.0
DEFAULT_MAX_PENDING = 20
# Discord rejects `content` over 2000 chars; keep the summary under that.
MAX_TEXT_CHARS = 1900
MAX_FIELD_CHARS = 8000
MAX_TOAST_CHARS = 300


def _safe_str(value: Any) -> str:
    try:
        return str(value)
    except Exception:  # noqa: BLE001 - str() of arbitrary objects may raise anything
        return f"<unprintable {type(value).__name__}>"


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _env_float(environ: Mapping[str, str], name: str, default: float, *, allow_zero: bool) -> float:
    raw = environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        return default
    return value


#######################
#### Configuration ####
#######################


@dataclass(frozen=True)
class WebhookConfig:
    url: str | None = None
    token: str | None = None
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    dedupe_seconds: float = DEFAULT_DEDUPE_SECONDS

    @property
    def enabled(self) -> bool:
        return self.url is not None

    def headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"pixelmap/{__version__}",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers


def _valid_webhook_url(raw: str | None) -> str | None:
    url = (raw or "").strip()
    if not url:
        return None
    parsed = urllib.parse.urlsplit(url)
    # Only http(s): urllib would happily "POST" to file:// or ftp:// URLs.
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        logger.warning("Ignoring PIXELMAP_ERROR_WEBHOOK_URL: not an http(s) URL")
        return None
    return url


def load_webhook_config(environ: Mapping[str, str] | None = None) -> WebhookConfig:
    environ = os.environ if environ is None else environ
    return WebhookConfig(
        url=_valid_webhook_url(environ.get("PIXELMAP_ERROR_WEBHOOK_URL")),
        token=(environ.get("PIXELMAP_ERROR_WEBHOOK_TOKEN") or "").strip() or None,
        timeout=_env_float(
            environ, "PIXELMAP_ERROR_WEBHOOK_TIMEOUT", DEFAULT_TIMEOUT_SECONDS, allow_zero=False
        ),
        dedupe_seconds=_env_float(
            environ, "PIXELMAP_ERROR_WEBHOOK_DEDUPE_SECONDS", DEFAULT_DEDUPE_SECONDS, allow_zero=True
        ),
    )


#################
#### Payload ####
#################


def _format_traceback(exc: BaseException) -> str:
    try:
        return "".join(traceback.format_exception(exc))
    except Exception:  # noqa: BLE001 - formatting must never fail the report
        return f"<traceback unavailable for {type(exc).__name__}>"


def _current_session_id() -> str | None:
    try:
        return pn.state.curdoc.session_context.id
    except Exception:  # noqa: BLE001 - no session outside a server context
        return None


def build_payload(
    exc: BaseException, context: str | None = None, session_id: str | None = None
) -> dict[str, Any]:
    error_type = type(exc).__name__
    message = _truncate(_safe_str(exc), MAX_FIELD_CHARS)
    host = socket.gethostname()
    summary = f"PixelMap {__version__} error on {host}"
    if context:
        summary += f" ({context})"
    summary = _truncate(f"{summary}: {error_type}: {message}", MAX_TEXT_CHARS)
    return {
        "text": summary,
        "content": summary,
        "event": {
            "app": "pixelmap",
            "version": __version__,
            "host": host,
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "context": _truncate(context, MAX_FIELD_CHARS) if context else None,
            "session_id": session_id,
            "error_type": error_type,
            "error_message": message,
            "traceback": _truncate(_format_traceback(exc), MAX_FIELD_CHARS),
        },
    }


def post_webhook(payload: dict[str, Any], config: WebhookConfig) -> bool:
    """POST ``payload`` to the webhook. Returns success; never raises."""
    if not config.enabled:
        return False
    try:
        request = urllib.request.Request(
            config.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=config.headers(),
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=config.timeout) as response:
            return 200 <= response.status < 300
    except Exception as exc:  # noqa: BLE001 - any delivery failure is only logged
        # Deliberately not report_error: a dead webhook must not recurse.
        logger.warning("Error webhook delivery failed: %s", _safe_str(exc))
        return False


##################
#### Reporter ####
##################


def _dedupe_key(exc: BaseException) -> tuple:
    frames = traceback.extract_tb(exc.__traceback__)
    location = (frames[-1].filename, frames[-1].lineno) if frames else None
    return (type(exc).__qualname__, _safe_str(exc)[:200], location)


def notify(level: str, message: str, **kwargs) -> None:
    """Emit a Panel notification, or do nothing if the session has no area.

    `pn.state.notifications` is None outside a served session (tests, scripts).
    Reaching through it unguarded raises inside a Bokeh callback, and Bokeh
    discards every pending document write when a locked callback raises -- so
    one missing notification silently loses all the plot updates the callback
    had already made. Swallowing the notification is always the lesser evil.
    """
    try:
        notifications = pn.state.notifications
        if notifications is not None:
            getattr(notifications, level)(message, **kwargs)
    except Exception:
        logger.debug("Could not show %s notification", level, exc_info=True)


class ErrorReporter:
    def __init__(
        self,
        config_loader: Callable[[], WebhookConfig] = load_webhook_config,
        sender: Callable[[dict[str, Any], WebhookConfig], bool] = post_webhook,
        clock: Callable[[], float] = time.monotonic,
        synchronous: bool = False,
        max_pending: int = DEFAULT_MAX_PENDING,
    ):
        self._config_loader = config_loader
        self._sender = sender
        self._clock = clock
        self._synchronous = synchronous
        self._queue: queue.Queue = queue.Queue(maxsize=max_pending)
        self._lock = threading.Lock()
        self._last_sent: dict[tuple, float] = {}
        self._worker: threading.Thread | None = None

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    def report(
        self, exc: BaseException, context: str | None = None, *, notify_user: bool = True
    ) -> None:
        """Log, toast and webhook ``exc``. Never raises."""
        try:
            self._report(exc, context, notify_user)
        except Exception:  # noqa: BLE001 - the reporter must never raise
            # Last line of defence: the reporter itself must not crash callers.
            try:
                logger.exception("Error reporter failed")
            except Exception:  # noqa: BLE001, S110 - logging failed too; nothing left to do
                pass

    def _report(self, exc: BaseException, context: str | None, notify_user: bool) -> None:
        logger.error(
            "Unhandled error%s: %s",
            f" in {context}" if context else "",
            _safe_str(exc),
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        try:
            config = self._config_loader()
        except Exception:
            logger.exception("Could not load error webhook config")
            config = WebhookConfig()

        if notify_user:
            message = f"Something went wrong: {type(exc).__name__}: {_safe_str(exc)}"
            message = _truncate(message, MAX_TOAST_CHARS)
            if config.enabled:
                message += " (the error has been reported)"
            notify("error", message, duration=10_000)

        if config.enabled and self._should_send(exc, config):
            self._dispatch(build_payload(exc, context, _current_session_id()), config)

    def _should_send(self, exc: BaseException, config: WebhookConfig) -> bool:
        if config.dedupe_seconds <= 0:
            return True
        key = _dedupe_key(exc)
        now = self._clock()
        with self._lock:
            last = self._last_sent.get(key)
            if last is not None and now - last < config.dedupe_seconds:
                return False
            self._last_sent[key] = now
            # Keep the table from growing without bound under varied errors.
            if len(self._last_sent) > 1000:
                cutoff = now - config.dedupe_seconds
                self._last_sent = {k: t for k, t in self._last_sent.items() if t >= cutoff}
        return True

    def _dispatch(self, payload: dict[str, Any], config: WebhookConfig) -> None:
        if self._synchronous:
            self._sender(payload, config)
            return
        self._ensure_worker()
        try:
            self._queue.put_nowait((payload, config))
        except queue.Full:
            logger.warning("Error webhook backlog full; dropping report")

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._drain, name="pixelmap-error-webhook", daemon=True
                )
                self._worker.start()

    def _drain(self) -> None:
        while True:
            payload, config = self._queue.get()
            try:
                self._sender(payload, config)
            except Exception:
                logger.warning("Error webhook sender raised", exc_info=True)


_REPORTER = ErrorReporter()


def report_error(
    exc: BaseException, context: str | None = None, *, notify_user: bool = True
) -> None:
    """Report ``exc`` via the process-wide reporter. Never raises."""
    _REPORTER.report(exc, context, notify_user=notify_user)


###############
#### Hooks ####
###############


def session_exception_handler(exc: BaseException) -> None:
    """Panel ``config.exception_handler``: every widget/param callback error lands here.

    Returning normally (rather than re-raising, Panel's default) also keeps
    Bokeh from discarding the document writes the callback already made.
    """
    try:
        report_error(exc, context="GUI callback")
    except Exception:  # noqa: BLE001, S110 - the exception handler must never raise
        pass


def guarded(context: str):
    """Decorate a raw Bokeh callback so its errors are reported, not raised.

    Panel's ``exception_handler`` only covers Panel-managed callbacks; Bokeh
    ``on_change`` callbacks bypass it, and an exception there makes Bokeh drop
    the pending document patch for the whole callback.
    """

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - report any callback error instead of raising
                report_error(exc, context=context)
                return None

        return wrapper

    return decorate


_GLOBAL_HANDLERS_INSTALLED = False
_LOOPS_WITH_HANDLER: weakref.WeakSet = weakref.WeakSet()


def install_global_handlers() -> None:
    """Report errors from background threads and anything reaching top level.

    No toast: these are not tied to a user session. Idempotent.
    """
    global _GLOBAL_HANDLERS_INSTALLED
    if _GLOBAL_HANDLERS_INSTALLED:
        return
    _GLOBAL_HANDLERS_INSTALLED = True

    previous_thread_hook = threading.excepthook

    def thread_hook(args):
        if args.exc_value is not None and not issubclass(args.exc_type, SystemExit):
            name = args.thread.name if args.thread is not None else "unknown"
            report_error(args.exc_value, context=f"thread {name}", notify_user=False)
        previous_thread_hook(args)

    previous_sys_hook = sys.excepthook

    def sys_hook(exc_type, exc_value, exc_tb):
        if exc_value is not None and not issubclass(exc_type, KeyboardInterrupt):
            report_error(exc_value, context="uncaught", notify_user=False)
        previous_sys_hook(exc_type, exc_value, exc_tb)

    threading.excepthook = thread_hook
    sys.excepthook = sys_hook


def install_loop_handler(loop: asyncio.AbstractEventLoop | None = None) -> None:
    """Report exceptions the event loop would otherwise only log (e.g. orphaned tasks)."""
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
    if loop in _LOOPS_WITH_HANDLER:
        return
    _LOOPS_WITH_HANDLER.add(loop)

    previous = loop.get_exception_handler()

    def handler(loop, context):
        exc = context.get("exception")
        message = _safe_str(context.get("message", "event loop error"))
        if exc is None:
            exc = RuntimeError(message)
        report_error(exc, context=f"event loop: {message}", notify_user=False)
        if previous is not None:
            previous(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(handler)
