"""Regression tests: a PDF download must not stall the server.

Production watchdog logs showed the single Bokeh event loop blocked for 5-6 s
inside ``generate_pdf_content``: once in the legend's ``loc="best"`` search
(which scores candidate spots against every electrode patch on each draw) and
once in a full ``gc.collect()`` after saving.  Every session and the
healthcheck freeze while that runs.
"""

from __future__ import annotations

import gc

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.collections import PatchCollection

from pixelmap import backend
from pixelmap.gui.gui import ChannelmapGUI

PROBE_TYPES = ["1.0", "2.0-1shank", "2.0-4shanks", "QuadBase"]


@pytest.fixture(scope="module")
def gui():
    """One GUI instance for the module — construction loads all probe data."""
    return ChannelmapGUI()


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _select_preset(gui, probe_type):
    gui.probe_type = probe_type
    gui.preset = next(p for p in gui.param.preset.objects if p)
    gui.apply_preset()


def _plot(gui, **kwargs):
    gui.update_filename()
    gui.generate_imro()
    backend.plot_probe_layout(
        gui.probe_type, gui.imro_list, gui.positions_file, gui.wiring_file, "t", **kwargs
    )
    return plt.gcf().axes[0]


def test_pdf_download_does_not_force_gc(gui, monkeypatch):
    _select_preset(gui, "2.0-4shanks")
    calls = []
    monkeypatch.setattr(gc, "collect", lambda *a, **k: calls.append(a) or 0)

    buf = gui.generate_pdf_content()

    assert buf is not None and buf.getvalue().startswith(b"%PDF")
    assert calls == []


def test_pdf_download_not_ready_returns_none(gui, monkeypatch):
    monkeypatch.setattr(gui, "ready_to_download", lambda: False)
    assert gui.generate_pdf_content() is None


@pytest.mark.parametrize("probe_type", PROBE_TYPES)
def test_legend_location_is_pinned(gui, probe_type):
    """loc="best" re-runs an O(n_artists) search on every draw; it must be fixed."""
    _select_preset(gui, probe_type)
    legend = _plot(gui).get_legend()
    assert legend is not None
    assert legend._loc != 0  # 0 == "best"


@pytest.mark.parametrize("probe_type", PROBE_TYPES)
def test_legend_matches_previous_best_position(gui, probe_type):
    """The pinned spot is where loc="best" used to put it (output unchanged)."""
    _select_preset(gui, probe_type)
    ax = _plot(gui)
    fig = ax.figure
    legend = ax.get_legend()
    fig.canvas.draw()
    pinned = legend.get_window_extent().bounds

    legend.set_loc("best")
    fig.canvas.draw()
    assert legend.get_window_extent().bounds == pytest.approx(pinned)


@pytest.mark.parametrize("probe_type", PROBE_TYPES)
@pytest.mark.parametrize("with_survey", [False, True])
def test_electrodes_drawn_as_collections(gui, probe_type, with_survey):
    """Thousands of individual patches made each PDF take seconds to build/draw."""
    _select_preset(gui, probe_type)
    n_shanks = 1 if probe_type in ("1.0", "2.0-1shank") else 4
    kwargs = {}
    if with_survey:
        # Includes an electrode with no survey value -> falls back to grey.
        kwargs = {"survey_colors": {(0, 0): "#ff0000"}, "survey_range": (0.0, 1.0)}
    ax = _plot(gui, **kwargs)

    collections = [c for c in ax.collections if isinstance(c, PatchCollection)]
    per_shank = 2 if with_survey else 1
    assert len(collections) == n_shanks * per_shank
    # Only shank outlines / anatomy bands remain as standalone patches.
    assert len(ax.patches) < 50
    # Every electrode is still drawn, exactly once.
    drawn = sum(len(c.get_paths()) for c in collections if c.get_zorder() != 50)
    assert drawn == len(gui.positions_df)


def test_selected_electrodes_coloured(gui):
    """Selected electrodes keep the opaque red facecolour inside the collection."""
    _select_preset(gui, "2.0-1shank")
    ax = _plot(gui)
    (electrodes,) = [c for c in ax.collections if isinstance(c, PatchCollection)]

    def rgba(c):
        return tuple(round(float(v), 6) for v in c)

    colours = [rgba(c) for c in electrodes.get_facecolor()]
    selected = rgba(matplotlib.colors.to_rgba("red", 1.0))
    unselected = rgba(matplotlib.colors.to_rgba("lightgrey", 0.7))
    forbidden = rgba(matplotlib.colors.to_rgba("black", 1.0))
    assert set(colours) <= {selected, unselected, forbidden}
    assert colours.count(selected) == len(gui.imro_list) - 1  # minus header row


def test_add_rects_empty_is_noop():
    _fig, ax = plt.subplots()
    backend._add_rects(ax, [], [])
    assert len(ax.collections) == 0
