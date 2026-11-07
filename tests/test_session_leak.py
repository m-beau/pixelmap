"""Regression tests: a closed browser tab must not keep its session in memory.

Production RSS grew without bound (~3.4 GB) and the container was OOM-killed.
A load test against the production image showed every closed session leaving
its whole ``ChannelmapGUI`` (plus ~90k ``Electrode`` objects of wiring maps)
alive.  The anchor was ``panel.state._views``: Panel registers the per-session
``NotificationArea`` there, and its teardown hook is only reached through
``state._notifications[doc]`` -- which Panel's own ``Document.destroy``
override deletes *before* Bokeh runs the session-destroyed callbacks.  So the
view entry, and through it the dead Document and everything on it, is never
released (Panel 1.7.5 through at least 1.9.4).
"""

from __future__ import annotations

import gc
import socket
import time
import types
import weakref

import panel as pn
import pytest
from bokeh.client import pull_session
from bokeh.document import Document
from bokeh.events import DocumentReady
from panel.io.state import state

from pixelmap.gui import gui as gui_module
from pixelmap.gui.gui import _release_dead_session_views


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for(cond, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


def _live_guis() -> int:
    gc.collect()
    return sum(isinstance(o, gui_module.ChannelmapGUI) for o in gc.get_objects())


def _dead_doc_views() -> list:
    """``state._views`` entries whose Document has already been destroyed."""
    return [ref for ref, (_v, _r, doc, _c) in state._views.items()
            if doc is not None and not hasattr(doc, "_roots")]


# --------------------------------------------------------------------------
# End to end: real Panel server, real session, real teardown
# --------------------------------------------------------------------------

@pytest.fixture
def server():
    port = _free_port()
    srv = pn.serve(
        {"app": gui_module.create_app},
        port=port,
        address="127.0.0.1",
        show=False,
        threaded=True,
        check_unused_sessions_milliseconds=100,
        unused_session_lifetime_milliseconds=100,
        allow_websocket_origin=[f"127.0.0.1:{port}"],
    )
    _wait_for(lambda: srv.server_id in state._servers, 30, "server start")
    yield f"http://127.0.0.1:{port}/app", state._servers[srv.server_id][0]
    srv.stop()


def _sessions(bokeh_server):
    return [s for ctx in bokeh_server._tornado._applications.values()
            for s in ctx.sessions]


@pytest.mark.parametrize("n_sessions", [1, 3])
def test_closed_sessions_release_their_gui(server, n_sessions):
    url, bokeh_server = server
    baseline = _live_guis()

    for _ in range(n_sessions):
        session = pull_session(url=url)
        session.close()

    _wait_for(lambda: not _sessions(bokeh_server), 30, "server sessions to be discarded")
    # Teardown runs on the server's loop thread; give it a moment to finish.
    _wait_for(lambda: _live_guis() <= baseline, 15, "dead sessions' GUIs to be collected")

    assert _dead_doc_views() == []


def _fire_document_ready(bokeh_server, session) -> None:
    """Sessions build once a browser reports ``document_ready``; ``pull_session`` never does."""
    doc = session.document
    bokeh_server.io_loop.add_callback(
        session.with_document_locked,
        lambda: doc.callbacks.trigger_event(DocumentReady()),
    )


@pytest.mark.parametrize("n_sessions", [1, 3])
def test_closed_sessions_that_built_a_gui_release_it(server, n_sessions):
    url, bokeh_server = server
    baseline = _live_guis()

    clients = [pull_session(url=url) for _ in range(n_sessions)]
    for session in _sessions(bokeh_server):
        _fire_document_ready(bokeh_server, session)
    _wait_for(lambda: _live_guis() == baseline + n_sessions, 60, "every session to build its GUI")
    # The swap follows the build on the loop; a close racing it is covered too.
    for client in clients:
        client.close()

    _wait_for(lambda: not _sessions(bokeh_server), 30, "server sessions to be discarded")
    _wait_for(lambda: _live_guis() <= baseline, 15, "dead sessions' GUIs to be collected")
    assert _dead_doc_views() == []


def test_closed_sessions_leave_no_queued_load_behind(server):
    """A queued onload reaches its document, so Panel's weak mapping would pin it forever."""
    url, bokeh_server = server
    client = pull_session(url=url)
    doc = _sessions(bokeh_server)[0].document
    assert doc in state._onload  # never loaded: no browser sent document_ready

    client.close()
    _wait_for(lambda: not _sessions(bokeh_server), 30, "server session to be discarded")
    _wait_for(lambda: doc not in state._onload, 15, "the queued load to be dropped")


# --------------------------------------------------------------------------
# The hook itself
# --------------------------------------------------------------------------

class _FakeView:
    def __init__(self):
        self.cleaned = []

    def _cleanup(self, root):
        self.cleaned.append(root)


class _ExplodingView:
    def _cleanup(self, root):
        raise RuntimeError("boom")


@pytest.fixture
def views(monkeypatch):
    monkeypatch.setattr(state, "_views", {})
    return state._views


def _ctx(doc):
    return types.SimpleNamespace(_document=doc, id="test-session")


def test_hook_drops_only_the_dead_documents_views(views):
    dead, alive = Document(), Document()
    v1, v2, v3 = _FakeView(), _FakeView(), _FakeView()
    views["a"] = (v1, "root-a", dead, None)
    views["b"] = (v2, "root-b", dead, None)
    views["c"] = (v3, "root-c", alive, None)

    _release_dead_session_views(_ctx(dead))

    assert set(views) == {"c"}
    assert v1.cleaned == ["root-a"] and v2.cleaned == ["root-b"]
    assert v3.cleaned == []


def test_hook_with_no_views_for_doc_is_noop(views):
    other = Document()
    views["c"] = (_FakeView(), "root-c", other, None)
    _release_dead_session_views(_ctx(Document()))
    assert set(views) == {"c"}


def test_hook_drops_only_the_dead_documents_queued_load(views, monkeypatch):
    dead, alive = Document(), Document()
    onload = weakref.WeakKeyDictionary({dead: [("load", False)], alive: [("load", False)]})
    monkeypatch.setattr(state, "_onload", onload)
    _release_dead_session_views(_ctx(dead))
    assert list(onload) == [alive]


def test_hook_with_empty_views(views):
    _release_dead_session_views(_ctx(Document()))
    assert views == {}


@pytest.mark.parametrize("ctx", [None, types.SimpleNamespace(_document=None, id="x")])
def test_hook_tolerates_missing_document(views, ctx):
    views["c"] = (_FakeView(), "root-c", Document(), None)
    _release_dead_session_views(ctx)
    assert set(views) == {"c"}


def test_hook_still_drops_view_when_cleanup_raises(views):
    """A failing ``_cleanup`` must not keep the session pinned."""
    doc = Document()
    views["a"] = (_ExplodingView(), "root-a", doc, None)
    _release_dead_session_views(_ctx(doc))
    assert views == {}


# --------------------------------------------------------------------------
# Canary: is the Panel bug still there?
# --------------------------------------------------------------------------

def _plain_notifications_app():
    pn.extension(notifications=True)
    return pn.Column(pn.widgets.Button(name="x"))


@pytest.mark.xfail(
    strict=True,
    reason="Panel bug (1.7.5..1.9.4): NotificationArea views outlive destroyed "
           "sessions. When this XPASSes, Panel is fixed and "
           "_release_dead_session_views can be removed.",
)
def test_panel_still_leaks_notification_views_without_workaround():
    port = _free_port()
    srv = pn.serve(
        {"app": _plain_notifications_app},
        port=port,
        address="127.0.0.1",
        show=False,
        threaded=True,
        check_unused_sessions_milliseconds=100,
        unused_session_lifetime_milliseconds=100,
        allow_websocket_origin=[f"127.0.0.1:{port}"],
    )
    try:
        _wait_for(lambda: srv.server_id in state._servers, 30, "server start")
        bokeh_server = state._servers[srv.server_id][0]
        pull_session(url=f"http://127.0.0.1:{port}/app").close()
        _wait_for(lambda: not _sessions(bokeh_server), 30, "session discard")
        time.sleep(0.5)
        assert _dead_doc_views() == []
    finally:
        srv.stop()


# --------------------------------------------------------------------------
# Session teardown must not stall the shared event loop
# --------------------------------------------------------------------------

def test_session_cleanup_does_not_force_gc(monkeypatch):
    """A full ``gc.collect()`` per closed tab froze every other session.

    The production watchdog measured ~6 s per full collection.  Teardown runs
    on the event loop for every session that ends, so it must leave cycle
    collection to the regular generational GC.
    """
    gui = gui_module.ChannelmapGUI()
    calls = []
    monkeypatch.setattr(gc, "collect", lambda *a, **k: calls.append(a) or 0)

    gui._cleanup_session(types.SimpleNamespace(id="test-session"))

    assert calls == []
    assert not hasattr(gui, "electrode_source")
    assert not hasattr(gui, "plot")


def test_session_cleanup_is_idempotent():
    """Cleanup may run after attributes are already gone (double destroy)."""
    gui = gui_module.ChannelmapGUI()
    ctx = types.SimpleNamespace(id="test-session")
    gui._cleanup_session(ctx)
    gui._cleanup_session(ctx)
