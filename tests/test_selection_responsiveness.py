"""Regression tests: an electrode click must not stall the server.

Production logs showed the single-threaded Bokeh event loop freezing for
20-90 s after nearly every ``Selection changed: [] -> [N]``.  The handler ran a
full ``gc.collect()`` on every tap, which walks every live object in a process
holding several GB and hundreds of thousands of Bokeh models.  While frozen,
the healthcheck failed and the reverse proxy dropped users' websockets.
"""

from __future__ import annotations

import gc

import pytest

from pixelmap.gui.gui import ChannelmapGUI
from pixelmap.types import Electrode


@pytest.fixture(scope="module")
def gui():
    """One GUI instance for the module — construction loads all probe data."""
    return ChannelmapGUI()


@pytest.fixture(autouse=True)
def clean_selection(gui):
    gui.electrodes.clear_selection()
    gui.select_mode = "select"
    yield
    gui.electrodes.clear_selection()


@pytest.fixture
def gc_calls(monkeypatch):
    """Count full collections triggered anywhere while the handler runs."""
    calls = []
    monkeypatch.setattr(gc, "collect", lambda *a, **k: calls.append(a) or 0)
    return calls


def _electrode_at(gui, idx):
    data = gui.electrode_source.data
    return Electrode(data["shank_id"][idx], data["electrode_id"][idx])


def test_single_tap_does_not_force_gc(gui, gc_calls):
    gui.on_electrode_selection("indices", [], [0])
    assert gc_calls == []


def test_box_select_does_not_force_gc(gui, gc_calls):
    gui.on_electrode_selection("indices", [], [0, 1, 2, 3])
    assert gc_calls == []


def test_many_taps_do_not_force_gc(gui, gc_calls):
    for _ in range(20):
        gui.on_electrode_selection("indices", [], [0])
    assert gc_calls == []


def test_single_tap_still_toggles(gui):
    target = _electrode_at(gui, 0)
    gui.on_electrode_selection("indices", [], [0])
    assert target in gui.electrodes.selected
    gui.on_electrode_selection("indices", [], [0])
    assert target not in gui.electrodes.selected


def test_tap_clears_bokeh_selection(gui):
    gui.on_electrode_selection("indices", [], [0])
    assert gui.electrode_source.selected.indices == []


@pytest.mark.parametrize("empty", [[], None])
def test_empty_selection_is_noop(gui, empty):
    """The handler's own reset (``indices = []``) re-fires it with an empty list."""
    gui.on_electrode_selection("indices", [0], empty)
    assert gui.electrodes.selected == set()
