"""Tests for notification handling in the served (multi-session) app.

Regression cover for a server-only bug: ``pn.config`` is stored per session
(keyed on ``pn.state.curdoc``), so the module-level ``pn.extension`` in
``gui.py`` only configured whichever session happened to trigger the import.
Under ``panel serve`` that is session #1; every later visitor got
``pn.state.notifications is None``.

That was doubly bad. Reaching through the None raised inside a Bokeh callback,
and Bokeh drops *every* pending document write when a locked callback raises
(``ServerSession._pending_writes`` is discarded before it is awaited) — so a
missing notification silently threw away the plot updates the same callback
had already made, leaving e.g. the survey overlay unrendered until some later
interaction pushed a fresh patch.
"""

import inspect
import re

import panel as pn
import pytest

from pixelmap.gui import gui


def test_notify_is_a_noop_without_a_notification_area(monkeypatch):
    """The guard, not the caller, absorbs a missing notification area."""
    monkeypatch.setattr(type(pn.state), "notifications", property(lambda self: None))
    gui._notify("success", "should not raise")


def test_notify_forwards_to_the_notification_area(monkeypatch):
    calls = []

    class FakeArea:
        def success(self, message, **kwargs):
            calls.append(("success", message, kwargs))

    monkeypatch.setattr(type(pn.state), "notifications", property(lambda self: FakeArea()))
    gui._notify("success", "loaded", duration=5_000)

    assert calls == [("success", "loaded", {"duration": 5_000})]


def test_no_unguarded_notification_access_in_gui():
    """Every call site must go through ``_notify``.

    A single unguarded ``pn.state.notifications.foo(...)`` is enough to
    reintroduce the dropped-document-writes bug for every session but the
    first, so this is enforced rather than left to review.
    """
    source = inspect.getsource(gui)
    unguarded = re.findall(r"pn\.state\.notifications\.\w+\(", source)

    assert not unguarded, f"unguarded notification calls found: {unguarded}"


def test_create_app_re_enables_extensions_per_session():
    """``create_app`` must call ``pn.extension`` itself.

    The module-level call in ``gui.py`` is not enough under ``panel serve``:
    the module is imported once, inside the first session's document context,
    so the per-session config never reaches later sessions.
    """
    source = inspect.getsource(gui.create_app)

    assert "pn.extension(" in source
    assert "notifications=True" in source


@pytest.mark.parametrize("level", ["info", "success", "warning", "error"])
def test_notify_supports_every_level_used_by_the_gui(level, monkeypatch):
    seen = []

    class FakeArea:
        def __getattr__(self, name):
            return lambda message, **kwargs: seen.append(name)

    monkeypatch.setattr(type(pn.state), "notifications", property(lambda self: FakeArea()))
    gui._notify(level, "msg")

    assert seen == [level]
