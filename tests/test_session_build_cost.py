"""Keep a session's Bokeh document small: every session pays for it on the event loop.

``panel serve`` builds, validates, serialises and sends each session's document
on the loop that serves every other session and ``/liveness``, and all of that
scales with the number of models and the size of their data.
"""

from __future__ import annotations

import numpy as np
import pytest
from bokeh.models import BoxSelectTool, GlyphRenderer, MultiLine, Rect, TapTool, Text

from pixelmap.constants import WIRING_FILE_MAP
from pixelmap.gui.gui import ChannelmapGUI, shank_outline_geometry
from pixelmap.types import Electrode

NUMERIC_COLUMNS = (
    "x", "y", "width", "height", "shank_id", "electrode_id", "alpha", "line_width",
    "val", "bar_x", "bar_height", "bar_alpha",
)
SINGLE_SHANK = np.array([[0, e, 0, y] for e, y in enumerate(range(0, 7680, 10))], dtype=float)


@pytest.fixture(scope="module", params=sorted(WIRING_FILE_MAP))
def probe_gui(request):
    gui = ChannelmapGUI()
    gui.probe_type = request.param
    gui.load_probe_data()
    gui.setup_bokeh_plot()
    return gui


def _glyph_counts(gui):
    renderers = gui.plot.select({"type": GlyphRenderer})
    return {
        kind: sum(isinstance(r.glyph, kind) for r in renderers)
        for kind in (Rect, MultiLine, Text)
    }


def test_outlines_use_one_renderer_per_style_on_every_probe(probe_gui):
    counts = _glyph_counts(probe_gui)
    assert counts[MultiLine] == 2  # outlines, bank lines
    # Bank labels, the anatomy region labels and, on multi-shank probes, shank labels.
    assert counts[Text] <= 3


@pytest.mark.parametrize("tool_type", [TapTool, BoxSelectTool])
def test_selection_tools_only_hit_electrodes_never_decoration(probe_gui, tool_type):
    # A shared outline renderer is hit-tested per item, so a tap on one outline
    # would "select" it and Bokeh would dim every other outline.
    tools = probe_gui.plot.select({"type": tool_type})
    assert tools
    for tool in tools:
        assert set(tool.renderers) == {probe_gui.electrode_renderer, probe_gui.survey_bar_renderer}


def test_numeric_electrode_columns_are_arrays_for_binary_encoding(probe_gui):
    for column in NUMERIC_COLUMNS:
        assert isinstance(probe_gui.electrode_source.data[column], np.ndarray), column


def test_electrode_columns_keep_one_value_per_electrode(probe_gui):
    lengths = {len(v) for v in probe_gui.electrode_source.data.values()}
    assert lengths == {len(probe_gui.positions_df)}


def test_tool_icons_reuse_the_cached_data_uri(probe_gui):
    icons = [probe_gui.box_select_tool.icon, probe_gui.box_deselect_tool.icon]
    assert all(icon.startswith("data:image/png;base64,") for icon in icons)


@pytest.fixture
def default_gui():
    gui = ChannelmapGUI()
    yield gui
    gui.electrodes.clear_selection()


def test_rebuilding_keeps_the_current_selection_and_survey(default_gui):
    gui = default_gui
    first = Electrode(*gui.positions_df[["shank", "electrode"]].iloc[0].tolist())
    gui.electrodes.selected.add(first)
    gui.survey_values = {first: 0.0}  # zero is a real value, not "missing"

    gui.create_electrode_data()

    data = gui.electrode_source.data
    assert (data["status"][0], data["color"][0], data["alpha"][0]) == ("Selected", "red", 1.0)
    assert data["val"][0] == 0.0
    assert data["status"][1] == "Unselected"
    assert np.isnan(data["val"][1])  # no survey value for this electrode
    gui.survey_values = None


def test_updating_colors_matches_a_full_rebuild(default_gui):
    gui = default_gui
    gui.electrodes.selected.add(Electrode(*gui.positions_df[["shank", "electrode"]].iloc[5].tolist()))

    gui.update_electrode_colors()
    updated = dict(gui.electrode_source.data)
    gui.create_electrode_data()

    assert updated["status"][5] == "Selected"
    for column, values in gui.electrode_source.data.items():
        np.testing.assert_array_equal(updated[column], values, err_msg=column)


def test_single_shank_geometry_matches_the_drawn_probe():
    geometry = shank_outline_geometry(SINGLE_SHANK, single_shank=True)

    assert geometry.outline_xs == [[-50, 50, 50, 0, -50, -50]]
    assert geometry.outline_ys[0][:3] == [7770.0, 7770.0, 0.0]
    assert geometry.outline_ys[0][3] == pytest.approx(-767.0)
    assert geometry.shank_labels == ([], [], [])
    assert geometry.bank_labels[2] == ["Bank 0", "Bank 1"]  # banks start at 0 and 384
    assert geometry.bank_lines_ys == [[0.0, 0.0], [3840.0, 3840.0]]


def test_multi_shank_geometry_draws_every_shank_and_labels_banks_on_the_last():
    positions = np.vstack([SINGLE_SHANK + [shank, 0, 0, 0] for shank in range(4)])

    geometry = shank_outline_geometry(positions, single_shank=False)

    assert len(geometry.outline_xs) == 4
    assert geometry.outline_xs[3][3] == 750  # tip of the rightmost shank
    assert geometry.shank_labels[2] == ["Shank 0", "Shank 1", "Shank 2", "Shank 3"]
    assert len(geometry.bank_lines_xs) == 8  # two banks on each shank
    assert set(geometry.bank_labels[0]) == {817}  # right of the rightmost shank


@pytest.mark.parametrize("single_shank", [True, False])
def test_geometry_of_a_probe_shorter_than_one_bank_has_one_bank_line(single_shank):
    positions = np.array([[s, e, 0, e * 20.0] for s in range(4) for e in range(3)])
    if single_shank:
        positions = positions[positions[:, 0] == 0]

    geometry = shank_outline_geometry(positions, single_shank=single_shank)

    assert len(geometry.bank_lines_xs) == (1 if single_shank else 4)
    assert geometry.bank_labels[2] == ["Bank 0"]


def test_geometry_of_a_single_electrode_has_a_flat_tip():
    geometry = shank_outline_geometry(np.array([[0, 0, 0, 0.0]]), single_shank=True)
    assert geometry.outline_ys[0][2:4] == [0.0, 0.0]
