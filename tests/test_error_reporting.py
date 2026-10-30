"""Tests for centralised error reporting (toast + webhook).

An exception escaping a GUI callback must never take down the server or a
session: it is shown to the user as a toast, logged, and -- if the deployer
configured one -- posted to a webhook.
"""

import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import panel as pn
import pytest

from pixelmap import error_reporting
from pixelmap.error_reporting import (
    ErrorReporter,
    WebhookConfig,
    build_payload,
    guarded,
    load_webhook_config,
    post_webhook,
)


def _raise(exc):
    try:
        raise exc
    except BaseException as caught:
        return caught


class _Recorder:
    """Stands in for ``post_webhook``; records what would have been sent."""

    def __init__(self, result=True):
        self.calls = []
        self.result = result

    def __call__(self, payload, config):
        self.calls.append((payload, config))
        return self.result


def _reporter(url="https://hooks.example/abc", dedupe=60.0, sender=None, clock=None):
    sender = sender or _Recorder()
    config = WebhookConfig(url=url, dedupe_seconds=dedupe)
    reporter = ErrorReporter(
        config_loader=lambda: config,
        sender=sender,
        clock=clock or (lambda: 0.0),
        synchronous=True,
    )
    return reporter, sender


@pytest.fixture
def toasts(monkeypatch):
    seen = []

    class FakeArea:
        def __getattr__(self, level):
            return lambda message, **kwargs: seen.append((level, message, kwargs))

    monkeypatch.setattr(type(pn.state), "notifications", property(lambda self: FakeArea()))
    return seen


@pytest.fixture
def no_toasts(monkeypatch):
    monkeypatch.setattr(type(pn.state), "notifications", property(lambda self: None))


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_webhook_disabled_when_env_unset():
    config = load_webhook_config({})
    assert config.url is None
    assert not config.enabled


@pytest.mark.parametrize("raw", ["", "   ", "\n"])
def test_blank_webhook_url_means_disabled(raw):
    assert not load_webhook_config({"PIXELMAP_ERROR_WEBHOOK_URL": raw}).enabled


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "ftp://example.com/x", "not a url", "//example.com"]
)
def test_non_http_webhook_urls_are_rejected(url):
    assert not load_webhook_config({"PIXELMAP_ERROR_WEBHOOK_URL": url}).enabled


def test_webhook_config_reads_all_settings():
    config = load_webhook_config(
        {
            "PIXELMAP_ERROR_WEBHOOK_URL": " https://hooks.example/abc ",
            "PIXELMAP_ERROR_WEBHOOK_TIMEOUT": "2.5",
            "PIXELMAP_ERROR_WEBHOOK_DEDUPE_SECONDS": "0",
            "PIXELMAP_ERROR_WEBHOOK_TOKEN": "s3cret",
        }
    )
    assert config.url == "https://hooks.example/abc"
    assert config.timeout == 2.5
    assert config.dedupe_seconds == 0
    assert config.headers()["Authorization"] == "Bearer s3cret"


@pytest.mark.parametrize("raw", ["abc", "0", "-1", "", "nan", "inf"])
def test_bad_timeout_falls_back_to_default(raw):
    config = load_webhook_config(
        {"PIXELMAP_ERROR_WEBHOOK_URL": "https://h.example", "PIXELMAP_ERROR_WEBHOOK_TIMEOUT": raw}
    )
    assert config.timeout == error_reporting.DEFAULT_TIMEOUT_SECONDS


@pytest.mark.parametrize("raw", ["abc", "-5", "", "nan"])
def test_bad_dedupe_falls_back_to_default(raw):
    config = load_webhook_config(
        {
            "PIXELMAP_ERROR_WEBHOOK_URL": "https://h.example",
            "PIXELMAP_ERROR_WEBHOOK_DEDUPE_SECONDS": raw,
        }
    )
    assert config.dedupe_seconds == error_reporting.DEFAULT_DEDUPE_SECONDS


def test_no_auth_header_without_token():
    config = load_webhook_config({"PIXELMAP_ERROR_WEBHOOK_URL": "https://h.example"})
    assert "Authorization" not in config.headers()
    assert config.headers()["Content-Type"] == "application/json"


# --------------------------------------------------------------------------
# Payload
# --------------------------------------------------------------------------


def test_payload_carries_error_details():
    exc = _raise(ValueError("boom"))
    payload = build_payload(exc, context="apply preset", session_id="sess-1")

    event = payload["event"]
    assert event["error_type"] == "ValueError"
    assert event["error_message"] == "boom"
    assert event["context"] == "apply preset"
    assert event["session_id"] == "sess-1"
    assert "Traceback" in event["traceback"]
    assert event["app"] == "pixelmap"
    # Chat-style webhooks (Slack/Mattermost use `text`, Discord `content`).
    assert "ValueError: boom" in payload["text"]
    assert payload["content"] == payload["text"]
    json.dumps(payload)  # must be serialisable


def test_payload_without_context_or_session():
    payload = build_payload(_raise(RuntimeError("x")), context=None, session_id=None)
    assert payload["event"]["context"] is None
    assert payload["event"]["session_id"] is None


def test_payload_survives_an_exception_whose_str_raises():
    class Nasty(Exception):
        def __str__(self):
            raise RuntimeError("str() exploded")

    payload = build_payload(_raise(Nasty()), context="x")
    assert payload["event"]["error_type"] == "Nasty"
    json.dumps(payload)


def test_payload_for_exception_without_traceback():
    payload = build_payload(KeyError("never raised"), context=None)
    assert payload["event"]["error_type"] == "KeyError"
    json.dumps(payload)


def test_payload_text_is_bounded():
    payload = build_payload(_raise(ValueError("x" * 50_000)), context="y" * 50_000)
    assert len(payload["text"]) <= error_reporting.MAX_TEXT_CHARS
    assert len(payload["event"]["error_message"]) <= error_reporting.MAX_FIELD_CHARS
    assert len(payload["event"]["traceback"]) <= error_reporting.MAX_FIELD_CHARS


# --------------------------------------------------------------------------
# Webhook transport
# --------------------------------------------------------------------------


@pytest.fixture
def webhook_server():
    received = []

    class Handler(BaseHTTPRequestHandler):
        status = 204

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            received.append(
                {"headers": dict(self.headers), "body": json.loads(self.rfile.read(length))}
            )
            self.send_response(Handler.status)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/hook", received, Handler
    finally:
        server.shutdown()
        server.server_close()


def test_post_webhook_delivers_json(webhook_server):
    url, received, _ = webhook_server
    config = WebhookConfig(url=url, token="tok")

    assert post_webhook({"text": "hello"}, config) is True
    assert received[0]["body"] == {"text": "hello"}
    assert received[0]["headers"]["Authorization"] == "Bearer tok"
    assert received[0]["headers"]["Content-Type"] == "application/json"


def test_post_webhook_reports_http_errors_without_raising(webhook_server):
    url, _, handler = webhook_server
    handler.status = 500
    assert post_webhook({"text": "x"}, WebhookConfig(url=url)) is False


def test_post_webhook_unreachable_host_does_not_raise():
    # Port 9 (discard) on localhost is essentially never listening.
    assert post_webhook({"text": "x"}, WebhookConfig(url="http://127.0.0.1:9/", timeout=1)) is False


def test_post_webhook_disabled_is_a_noop():
    assert post_webhook({"text": "x"}, WebhookConfig(url=None)) is False


# --------------------------------------------------------------------------
# Reporter
# --------------------------------------------------------------------------


def test_report_toasts_and_posts(toasts):
    reporter, sender = _reporter()
    reporter.report(_raise(ValueError("boom")), context="apply preset")

    assert len(sender.calls) == 1
    assert sender.calls[0][0]["event"]["context"] == "apply preset"
    assert len(toasts) == 1
    level, message, _ = toasts[0]
    assert level == "error"
    assert "boom" in message
    assert "reported" in message.lower()


def test_toast_does_not_claim_reporting_without_webhook(toasts):
    reporter, sender = _reporter(url=None)
    reporter.report(_raise(ValueError("boom")))

    assert sender.calls == []
    assert "reported" not in toasts[0][1].lower()


def test_report_without_notification_area_still_posts(no_toasts):
    reporter, sender = _reporter()
    reporter.report(_raise(ValueError("boom")))
    assert len(sender.calls) == 1


def test_report_can_skip_the_toast(toasts):
    reporter, sender = _reporter()
    reporter.report(_raise(ValueError("boom")), notify_user=False)
    assert toasts == []
    assert len(sender.calls) == 1


def test_report_never_raises_when_toast_fails(monkeypatch):
    class Broken:
        def error(self, *args, **kwargs):
            raise RuntimeError("notification area is broken")

    monkeypatch.setattr(type(pn.state), "notifications", property(lambda self: Broken()))
    reporter, sender = _reporter()
    reporter.report(_raise(ValueError("boom")))
    assert len(sender.calls) == 1


def test_report_never_raises_when_sender_fails(no_toasts):
    def sender(payload, config):
        raise RuntimeError("network stack on fire")

    reporter, _ = _reporter(sender=sender)
    reporter.report(_raise(ValueError("boom")))  # must not raise


def test_report_never_raises_when_config_loader_fails(no_toasts):
    def loader():
        raise RuntimeError("bad env")

    reporter = ErrorReporter(config_loader=loader, sender=_Recorder(), synchronous=True)
    reporter.report(_raise(ValueError("boom")))  # must not raise


def test_duplicate_errors_are_deduplicated_within_window(no_toasts):
    now = [0.0]
    reporter, sender = _reporter(dedupe=60.0, clock=lambda: now[0])

    def fail():
        raise ValueError("same")

    for _ in range(3):
        reporter.report(_raise_from(fail))
    assert len(sender.calls) == 1

    now[0] = 61.0
    reporter.report(_raise_from(fail))
    assert len(sender.calls) == 2


def test_distinct_errors_are_not_deduplicated(no_toasts):
    reporter, sender = _reporter(dedupe=60.0)
    reporter.report(_raise(ValueError("a")))
    reporter.report(_raise(TypeError("a")))
    reporter.report(_raise(ValueError("b")))
    assert len(sender.calls) == 3


def test_zero_dedupe_window_sends_every_error(no_toasts):
    reporter, sender = _reporter(dedupe=0)
    for _ in range(3):
        reporter.report(_raise(ValueError("same")))
    assert len(sender.calls) == 3


def test_toast_is_shown_even_for_deduplicated_errors(toasts):
    reporter, sender = _reporter(dedupe=60.0)
    reporter.report(_raise(ValueError("same")))
    reporter.report(_raise(ValueError("same")))
    assert len(toasts) == 2
    assert len(sender.calls) == 1


def test_asynchronous_dispatch_delivers(no_toasts):
    sender = _Recorder()
    done = threading.Event()

    def wrapped(payload, config):
        sender(payload, config)
        done.set()
        return True

    reporter = ErrorReporter(
        config_loader=lambda: WebhookConfig(url="https://h.example"), sender=wrapped
    )
    reporter.report(_raise(ValueError("boom")))
    assert done.wait(5)
    assert len(sender.calls) == 1


def test_asynchronous_dispatch_drops_when_backlogged(no_toasts):
    release = threading.Event()
    calls = []

    def slow(payload, config):
        calls.append(payload)
        release.wait(5)
        return True

    reporter = ErrorReporter(
        config_loader=lambda: WebhookConfig(url="https://h.example", dedupe_seconds=0),
        sender=slow,
        max_pending=2,
    )
    try:
        for i in range(10):
            reporter.report(_raise(ValueError(str(i))))
        assert reporter.pending <= 2
    finally:
        release.set()


def _raise_from(fn):
    try:
        fn()
    except Exception as exc:
        return exc


# --------------------------------------------------------------------------
# Hooks
# --------------------------------------------------------------------------


def test_guarded_swallows_and_reports(monkeypatch):
    reported = []
    monkeypatch.setattr(error_reporting, "report_error", lambda exc, **kw: reported.append((exc, kw)))

    @guarded("electrode selection")
    def callback(attr, old, new):
        raise ValueError("bad selection")

    assert callback("indices", [], [1]) is None
    assert isinstance(reported[0][0], ValueError)
    assert reported[0][1]["context"] == "electrode selection"


def test_guarded_passes_through_return_values():
    @guarded("x")
    def callback(a, b=2):
        return a + b

    assert callback(1) == 3
    assert callback(1, b=5) == 6


def test_guarded_preserves_signature_for_bokeh():
    """Bokeh's ``on_change`` inspects the callback's arity."""
    import inspect

    class Owner:
        def on_change(self, attr, old, new):
            pass

    wrapped = guarded("x")(Owner().on_change)
    assert list(inspect.signature(wrapped).parameters) == ["attr", "old", "new"]


def test_guarded_lets_keyboard_interrupt_through():
    @guarded("x")
    def callback():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        callback()


def test_session_exception_handler_reports(monkeypatch):
    reported = []
    monkeypatch.setattr(error_reporting, "report_error", lambda exc, **kw: reported.append(exc))
    exc = ValueError("x")
    error_reporting.session_exception_handler(exc)
    assert reported == [exc]


def test_session_exception_handler_never_raises(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("reporter broke")

    monkeypatch.setattr(error_reporting, "report_error", explode)
    error_reporting.session_exception_handler(ValueError("x"))  # must not raise


def test_install_global_handlers_is_idempotent(monkeypatch):
    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)
    monkeypatch.setattr(threading, "excepthook", threading.__excepthook__)
    monkeypatch.setattr(error_reporting, "_GLOBAL_HANDLERS_INSTALLED", False)

    error_reporting.install_global_handlers()
    first_sys, first_thread = sys.excepthook, threading.excepthook
    error_reporting.install_global_handlers()

    assert sys.excepthook is first_sys
    assert threading.excepthook is first_thread
    assert sys.excepthook is not sys.__excepthook__


def test_thread_exceptions_are_reported(monkeypatch):
    reported = []
    monkeypatch.setattr(error_reporting, "report_error", lambda exc, **kw: reported.append((exc, kw)))
    monkeypatch.setattr(threading, "excepthook", threading.__excepthook__)
    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)
    monkeypatch.setattr(error_reporting, "_GLOBAL_HANDLERS_INSTALLED", False)
    error_reporting.install_global_handlers()

    def boom():
        raise ValueError("in a thread")

    t = threading.Thread(target=boom, name="pixelmap-test")
    t.start()
    t.join()

    assert isinstance(reported[0][0], ValueError)
    assert reported[0][1]["notify_user"] is False
    assert "pixelmap-test" in reported[0][1]["context"]


def test_asyncio_loop_exceptions_are_reported(monkeypatch):
    reported = []
    monkeypatch.setattr(error_reporting, "report_error", lambda exc, **kw: reported.append(exc))

    async def scenario():
        loop = asyncio.get_running_loop()
        error_reporting.install_loop_handler(loop)
        error_reporting.install_loop_handler(loop)  # idempotent
        loop.call_exception_handler({"message": "task died", "exception": ValueError("x")})
        loop.call_exception_handler({"message": "no exception attached"})

    asyncio.run(scenario())
    assert isinstance(reported[0], ValueError)
    assert isinstance(reported[1], RuntimeError)
    assert len(reported) == 2


# --------------------------------------------------------------------------
# GUI wiring
# --------------------------------------------------------------------------


def test_create_app_installs_session_exception_handler():
    import inspect

    from pixelmap.gui import gui

    source = inspect.getsource(gui.create_app)
    assert "exception_handler=" in source


def test_create_app_survives_a_failing_gui_build(monkeypatch, no_toasts):
    from pixelmap.gui import gui

    reported = []
    monkeypatch.setattr(error_reporting, "report_error", lambda exc, **kw: reported.append(exc))

    def broken(*args, **kwargs):
        raise RuntimeError("wiring maps missing")

    monkeypatch.setattr(gui, "ChannelmapGUI", broken)
    app = gui.create_app()

    assert app is not None
    assert isinstance(reported[0], RuntimeError)


def test_raw_bokeh_callbacks_are_guarded():
    """Panel's exception_handler doesn't see raw Bokeh ``on_change`` callbacks."""
    import inspect
    import re

    from pixelmap.gui import gui

    source = inspect.getsource(gui)
    raw = re.findall(r"\.on_change\(\s*\"\w+\",\s*([^)]+)\)", source)
    assert raw, "expected Bokeh on_change registrations in gui.py"
    for callback in raw:
        assert callback.strip().startswith("guarded("), callback
