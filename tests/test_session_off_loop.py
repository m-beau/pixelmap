"""A session's GUI is built off the Bokeh event loop, without racing the document.

``panel serve`` runs the app per session on the loop that serves every session
and ``/liveness``.  The session gets a placeholder at once; the GUI is built on
one worker thread and swapped in on the loop.  These tests use a real server,
and fire the ``document_ready`` event a browser would send: ``pull_session``
never sends it.
"""

from __future__ import annotations

import gc
import socket
import threading
import time
import weakref

import panel as pn
import pytest
from bokeh.client import pull_session
from bokeh.core.property.bases import validation_on
from bokeh.document import Document
from bokeh.events import DocumentReady
from bokeh.io.doc import patch_curdoc
from panel.io.state import set_curdoc, state

from pixelmap import error_reporting
from pixelmap.gui import gui as gui_module
from pixelmap.utils import url_share

RealGUI = gui_module.ChannelmapGUI
LOADING = "Loading PixelMap"


def _wait_for(cond, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


def _live_guis() -> int:
    gc.collect()
    return sum(isinstance(o, RealGUI) for o in gc.get_objects())


class Served:
    """A real threaded server whose app factory remembers what it returned."""

    def __init__(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.apps = []  # weakrefs, so these tests can see sessions die
        self.factory_threads = []
        self.server = pn.serve(
            {"app": self._factory},
            port=self.port,
            address="127.0.0.1",
            show=False,
            threaded=True,
            check_unused_sessions_milliseconds=100,
            unused_session_lifetime_milliseconds=100,
            allow_websocket_origin=[f"127.0.0.1:{self.port}"],
        )
        _wait_for(lambda: self.server.server_id in state._servers, 30, "server start")
        self.bokeh = state._servers[self.server.server_id][0]
        self.url = f"http://127.0.0.1:{self.port}/app"

    def _factory(self):
        self.factory_threads.append(threading.current_thread())
        app = gui_module.create_app()
        self.apps.append(weakref.ref(app))
        return app

    def sessions(self):
        return [s for ctx in self.bokeh._tornado._applications.values() for s in ctx.sessions]

    def open(self, **arguments):
        """Open a session; returns (client session, server session, app)."""
        known = {s.id for s in self.sessions()}
        client = pull_session(url=self.url, arguments=arguments or None)
        server_session = next(s for s in self.sessions() if s.id not in known)
        return client, server_session, self.apps[-1]()

    def on_loop(self, func):
        """Run ``func`` on the server's event loop and wait for its result."""
        done = threading.Event()
        box = {}

        def call():
            try:
                box["value"] = func()
            except BaseException as exc:  # noqa: BLE001 - re-raised in the test thread
                box["error"] = exc
            done.set()

        self.bokeh.io_loop.add_callback(call)
        assert done.wait(30), "the event loop did not respond"
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def fire_ready(self, session):
        """Do what the browser's ``document_ready`` does."""
        doc = session.document
        self.bokeh.io_loop.add_callback(
            session.with_document_locked,
            lambda: doc.callbacks.trigger_event(DocumentReady()),
        )

    def stop(self):
        self.server.stop()


@pytest.fixture
def served():
    srv = Served()
    yield srv
    srv.stop()


@pytest.fixture
def reports(monkeypatch):
    seen = []
    monkeypatch.setattr(
        error_reporting, "report_error", lambda exc, **kw: seen.append((exc, kw))
    )
    return seen


class Gate:
    """Replaces the GUI class: records its thread and blocks until released."""

    def __init__(self, monkeypatch, *, block=True):
        self.started = threading.Event()
        self.release = threading.Event()
        if not block:
            self.release.set()
        self.threads = []
        self.curdocs = []
        self.running = 0
        self.max_running = 0
        self.guis = []
        monkeypatch.setattr(gui_module, "ChannelmapGUI", self)

    def __call__(self, **params):
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        self.threads.append(threading.current_thread())
        self.curdocs.append(weakref.ref(pn.state.curdoc))  # weak: leak tests watch the document
        self.started.set()
        try:
            assert self.release.wait(30)
            gui = RealGUI(**params)
            self.guis.append(weakref.ref(gui))
            return gui
        finally:
            self.running -= 1


def _shows(app, cls) -> bool:
    return any(isinstance(o, cls) for o in app.objects)


def _is_placeholder(app) -> bool:
    return len(app.objects) == 1 and LOADING in str(app.objects[0].object)


def _is_built(app) -> bool:
    return not _is_placeholder(app)


def _gui_of(session) -> RealGUI:
    return next(
        cb.__self__
        for cb in session.document.session_destroyed_callbacks
        if getattr(cb, "__name__", "") == "_cleanup_session"
    )


# ---------------------------------------------------------------------------
# Where the work runs
# ---------------------------------------------------------------------------


def test_session_starts_with_a_placeholder_and_no_gui(served, monkeypatch):
    gate = Gate(monkeypatch)
    client, _, app = served.open()
    assert _is_placeholder(app)
    assert gate.threads == []
    client.close()


def test_gui_is_built_on_the_session_worker_while_the_loop_stays_free(served, monkeypatch):
    gate = Gate(monkeypatch)
    client, session, app = served.open()
    served.fire_ready(session)
    assert gate.started.wait(30)

    # The build is blocked mid-way, and the loop still answers.
    assert served.on_loop(lambda: "alive") == "alive"
    assert _is_placeholder(app)
    assert gate.threads[0].name.startswith("pixelmap-session-build")
    assert gate.threads[0] is not served.factory_threads[0]

    gate.release.set()
    _wait_for(lambda: _is_built(app), 30, "the GUI to be swapped in")
    client.close()


def test_worker_sees_its_own_document_not_the_one_the_loop_is_serving(served, monkeypatch):
    """Bokeh's curdoc patch stack is process-global: only the context is per-session."""
    gate = Gate(monkeypatch)
    client, session, app = served.open()
    other = Document()
    other._session_context = lambda: None  # state.curdoc requires a session context
    with patch_curdoc(other):
        served.fire_ready(session)
        assert gate.started.wait(30)
        gate.release.set()
        _wait_for(lambda: gate.curdocs, 30, "the build to read curdoc")
    assert gate.curdocs[0]() is session.document
    _wait_for(lambda: _is_built(app), 30, "swap")
    client.close()


def test_session_builds_never_overlap_and_run_in_order(served, monkeypatch):
    gate = Gate(monkeypatch, block=False)
    opened = [served.open() for _ in range(3)]
    for _, session, _ in opened:
        served.fire_ready(session)
    _wait_for(lambda: all(_is_built(app) for *_, app in opened), 90, "all sessions to build")

    assert gate.max_running == 1
    assert len({t.ident for t in gate.threads}) == 1
    # FIFO: each build saw the session it belongs to.
    assert [ref() for ref in gate.curdocs] == [session.document for _, session, _ in opened]
    for client, *_ in opened:
        client.close()


def test_worker_does_not_touch_any_document(served, monkeypatch):
    """Everything that mutates a Document must happen on the loop."""
    touched = []
    for method in ("on_session_destroyed", "add_root", "remove_root", "on_event", "hold", "unhold"):
        original = getattr(Document, method)

        def spy(self, *args, _original=original, _method=method, **kwargs):
            touched.append((_method, threading.current_thread().name))
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(Document, method, spy)
    client, session, app = served.open()
    served.fire_ready(session)
    _wait_for(lambda: _is_built(app), 60, "swap")

    assert ("on_session_destroyed", served.factory_threads[0].name) in touched
    assert [t for t in touched if t[1].startswith("pixelmap-session-build")] == []
    client.close()


def test_constructor_registers_nothing_on_the_current_document():
    """The worker has a document in context; the constructor must leave it alone."""
    doc = Document()
    pn.state.curdoc = doc
    try:
        RealGUI()
    finally:
        pn.state.curdoc = None
    assert doc.session_destroyed_callbacks == set()


# ---------------------------------------------------------------------------
# validate(False) toggled a process-global flag
# ---------------------------------------------------------------------------


def test_electrode_source_is_built_with_validation_on(monkeypatch):
    gui = RealGUI()
    seen = []
    real_source = gui_module.ColumnDataSource

    def spy(*args, **kwargs):
        seen.append(validation_on())
        return real_source(*args, **kwargs)

    monkeypatch.setattr(gui_module, "ColumnDataSource", spy)
    gui.create_electrode_data()
    assert seen == [True]


def test_concurrent_electrode_sources_leave_validation_on():
    """Overlapping save/restore of the global flag used to leave it off for good."""
    guis = [RealGUI(), RealGUI()]
    barrier = threading.Barrier(2)

    def hammer(gui):
        barrier.wait()
        for _ in range(15):
            gui.create_electrode_data()

    workers = [threading.Thread(target=hammer, args=(g,)) for g in guis]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert validation_on()


# ---------------------------------------------------------------------------
# Tab closed before the swap
# ---------------------------------------------------------------------------


def test_tab_closed_mid_build_releases_the_gui_and_the_document(served, monkeypatch, reports):
    baseline = _live_guis()
    gate = Gate(monkeypatch)
    cleanups = []
    real_cleanup = RealGUI._cleanup_session
    monkeypatch.setattr(
        RealGUI, "_cleanup_session",
        lambda self, ctx: (cleanups.append(ctx.id), real_cleanup(self, ctx))[1],
    )
    client, session, app = served.open()
    session_id, app_ref = session.id, weakref.ref(app)
    doc_ref = weakref.ref(session.document)
    served.fire_ready(session)
    assert gate.started.wait(30)

    del app
    client.close()
    _wait_for(lambda: not served.sessions(), 30, "the session to be discarded")
    assert session.destroyed

    gate.release.set()
    _wait_for(lambda: cleanups == [session_id], 30, "the orphaned GUI's cleanup")
    del session
    _wait_for(lambda: _live_guis() <= baseline, 15, "the orphaned GUI to be collected")
    _wait_for(lambda: gc.collect() is not None and app_ref() is None, 15, "the app to be collected")
    _wait_for(lambda: gc.collect() is not None and doc_ref() is None, 15, "the document to be collected")
    assert reports == []


def test_tab_closed_while_queued_never_builds(served, monkeypatch, reports):
    gate = Gate(monkeypatch)
    first_client, first, first_app = served.open()
    served.fire_ready(first)
    assert gate.started.wait(30)  # occupies the single worker

    client, session, _ = served.open()
    served.fire_ready(session)
    time.sleep(0.2)  # let its onload queue the build
    client.close()
    _wait_for(lambda: session.destroyed, 30, "the queued session to be discarded")

    gate.release.set()
    _wait_for(lambda: _is_built(first_app), 30, "the first session's swap")
    assert len(gate.threads) == 1
    assert reports == []
    first_client.close()


def test_session_that_never_loads_leaks_nothing_and_never_builds(served, monkeypatch):
    """A client that never sends document_ready (a bot, a stray curl) costs a placeholder."""
    gate = Gate(monkeypatch)
    client, session, app = served.open()
    doc_ref = weakref.ref(session.document)
    client.close()
    del client, app, session
    _wait_for(lambda: not served.sessions(), 30, "the session to be discarded")
    served.apps.clear()

    _wait_for(lambda: gc.collect() is not None and doc_ref() is None, 15, "the document to be collected")
    assert gate.threads == []


def test_session_closed_after_swap_runs_every_hook(served, monkeypatch):
    calls = []
    real_cleanup = RealGUI._cleanup_session
    monkeypatch.setattr(
        RealGUI, "_cleanup_session",
        lambda self, ctx: (calls.append("gui"), real_cleanup(self, ctx))[1],
    )
    real_destroyed = gui_module.AnalyticsSessionTracker.handle_session_destroyed
    monkeypatch.setattr(
        gui_module.AnalyticsSessionTracker, "handle_session_destroyed",
        lambda self, ctx: (calls.append("tracker"), real_destroyed(self, ctx))[1],
    )
    real_release = gui_module._release_dead_session_views
    monkeypatch.setattr(
        gui_module, "_release_dead_session_views",
        lambda ctx: (calls.append("views"), real_release(ctx))[1],
    )
    client, session, app = served.open()
    served.fire_ready(session)
    _wait_for(lambda: _is_built(app), 60, "swap")
    client.close()
    _wait_for(lambda: len(calls) >= 3, 30, "the destroy hooks")
    assert sorted(calls) == ["gui", "tracker", "views"]


# ---------------------------------------------------------------------------
# Build failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("broken", ["ChannelmapGUI", "AnalyticsSessionTracker"])
def test_build_failure_replaces_the_placeholder_with_the_alert(served, monkeypatch, reports, broken):
    def boom(*args, **kwargs):
        raise RuntimeError("wiring maps missing")

    monkeypatch.setattr(gui_module, broken, boom)
    client, session, app = served.open()
    served.fire_ready(session)
    _wait_for(lambda: _shows(app, pn.pane.Alert), 30, "the failure alert")

    assert "PixelMap failed to start" in app.objects[0].object
    assert not _is_placeholder(app)
    exc, kwargs = reports[0]
    assert isinstance(exc, RuntimeError)
    assert kwargs == {"context": "session startup", "notify_user": False}
    client.close()


def test_build_failure_after_the_tab_closed_is_reported_without_a_document(served, monkeypatch, reports):
    gate = Gate(monkeypatch)
    real_call = gate.__call__

    def boom(**params):
        real_call(**params)
        raise RuntimeError("late failure")

    monkeypatch.setattr(gui_module, "ChannelmapGUI", boom)
    client, session, _ = served.open()
    served.fire_ready(session)
    assert gate.started.wait(30)
    client.close()
    _wait_for(lambda: session.destroyed, 30, "the session to be discarded")
    gate.release.set()
    _wait_for(lambda: reports, 30, "the failure report")
    assert isinstance(reports[0][0], RuntimeError)
    assert reports[0][1]["context"] == "session startup"


def test_swap_failure_shows_the_alert_and_reports(served, monkeypatch, reports):
    """A child that can't render fails on the loop, after the worker succeeded."""
    def unrenderable(self, *args, **kwargs):
        raise RuntimeError("cannot render")

    real_layout = RealGUI.create_layout

    def poisoned_layout(self):
        layout = real_layout(self)
        monkeypatch.setattr(type(layout), "_get_model", unrenderable)
        return layout

    monkeypatch.setattr(RealGUI, "create_layout", poisoned_layout)
    client, session, app = served.open()
    served.fire_ready(session)
    _wait_for(lambda: _shows(app, pn.pane.Alert), 30, "the failure alert")
    assert reports[0][1]["context"] == "session startup"
    client.close()


# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------


def _share_link(electrodes, probe_type="2.0-4shanks") -> str:
    gui = RealGUI()
    if probe_type != gui.probe_type:
        gui.probe_type = probe_type
    for electrode in electrodes:
        gui.electrodes.select(gui_module.Electrode(*electrode))
    return gui.build_share_link().split(f"{url_share.QUERY_PARAM}=", 1)[1]


def test_share_links_reach_the_right_sessions(served):
    """?cfg= is read on the loop, in the session's own request."""
    link_a = _share_link([(0, 0), (0, 1)])
    link_b = _share_link([(0, 5)], probe_type="1.0")
    client_a, session_a, app_a = served.open(cfg=link_a)
    client_b, session_b, app_b = served.open(cfg=link_b)
    client_c, session_c, app_c = served.open()
    # Latest first, so a build reading the request late would get another session's.
    for session in (session_c, session_b, session_a):
        served.fire_ready(session)
    for app in (app_a, app_b, app_c):
        _wait_for(lambda app=app: _is_built(app), 90, "swap")

    gui_a, gui_b, gui_c = (_gui_of(s) for s in (session_a, session_b, session_c))
    assert len(gui_a.electrodes.selected) == 2
    assert (gui_b.probe_type, len(gui_b.electrodes.selected)) == ("1.0", 1)
    assert (gui_c.probe_type, len(gui_c.electrodes.selected)) == ("2.0-4shanks", 0)
    # The counter is rendered before the swap, from the restored selection.
    assert "Selected Electrodes: 2/" in gui_a.electrode_counter.object
    for client in (client_a, client_b, client_c):
        client.close()


@pytest.mark.parametrize("cfg", ["", "not-a-share-link", "0"])
def test_bad_share_links_still_build(served, cfg, reports):
    client, session, app = served.open(cfg=cfg)
    served.fire_ready(session)
    _wait_for(lambda: _is_built(app), 60, "swap")
    assert not _shows(app, pn.pane.Alert)
    assert len(_gui_of(session).electrodes.selected) == 0
    assert reports == []
    client.close()


def test_share_link_failure_is_not_a_startup_failure(served, monkeypatch, reports):
    def boom(self, encoded):
        raise ValueError("corrupt")

    monkeypatch.setattr(RealGUI, "apply_url_state", boom)
    client, session, app = served.open(cfg="anything")
    served.fire_ready(session)
    _wait_for(lambda: _is_built(app), 60, "swap")
    assert not _shows(app, pn.pane.Alert)
    assert reports == []
    client.close()


def test_built_session_has_its_notifications_tracker_and_handler(served):
    # Not the first session: that is what used to lose its notification area.
    first_client, first, first_app = served.open()
    served.fire_ready(first)
    _wait_for(lambda: _is_built(first_app), 60, "first swap")
    client, session, app = served.open()
    served.fire_ready(session)
    _wait_for(lambda: _is_built(app), 60, "second swap")

    assert app._analytics_tracker.cookie_bridge is app.objects[0]

    def in_session():
        with set_curdoc(session.document):
            area = pn.state.notifications
            return area, pn.config.exception_handler

    area, handler = served.on_loop(in_session)
    assert area is not None
    assert handler is error_reporting.session_exception_handler
    client.close()
    first_client.close()


# ---------------------------------------------------------------------------
# No document: scripts, tests, the `pn.serve` factory's callers
# ---------------------------------------------------------------------------


def test_create_app_without_a_document_builds_inline():
    app = gui_module.create_app()
    assert isinstance(app, pn.Column)
    assert app._analytics_tracker.cookie_bridge is app.objects[0]
    assert len(app.objects) == 2
    assert not _is_placeholder(app)


def test_create_app_with_a_sessionless_document_builds_inline_and_registers_hooks():
    doc = Document()
    pn.state.curdoc = doc
    try:
        app = gui_module.create_app()
    finally:
        pn.state.curdoc = None
    assert not _is_placeholder(app)
    names = {getattr(cb, "__name__", "") for cb in doc.session_destroyed_callbacks}
    assert {"_cleanup_session", "handle_session_destroyed", "_release_dead_session_views"} <= names


def test_create_app_inline_failure_returns_the_alert(monkeypatch, reports):
    def boom(*args, **kwargs):
        raise RuntimeError("no wiring")

    monkeypatch.setattr(gui_module, "ChannelmapGUI", boom)
    app = gui_module.create_app()
    assert isinstance(app, pn.pane.Alert)
    assert reports[0][1] == {"context": "session startup", "notify_user": False}
