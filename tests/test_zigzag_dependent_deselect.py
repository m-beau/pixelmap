"""Regression coverage for freeing only zigzag-pattern wiring dependents."""

import pytest
from bokeh.models import BoxSelectTool

from pixelmap.gui.gui import ChannelmapGUI
from pixelmap.types import Electrode


@pytest.fixture(scope="module")
def gui():
    return ChannelmapGUI()


@pytest.fixture(autouse=True)
def clean_selection(gui):
    gui.probe_type = "1.0"
    gui.electrodes.clear_selection()
    gui.electrodes._undo_stack.clear()
    gui.electrodes._redo_stack.clear()
    gui.select_mode = "zigzag_dependent_deselect"
    yield
    gui.electrodes.clear_selection()


def _sites(gui):
    data = gui.electrode_source.data
    return [Electrode(s, e) for s, e in zip(data["shank_id"], data["electrode_id"])]


def _indices(gui, electrodes):
    return [i for i, e in enumerate(_sites(gui)) if e in electrodes]


def _dense_source(gui):
    """Select bank 0 and box the matching bank-1 range on shank 0."""
    source = {Electrode(0, e) for e in range(384)}
    chunk = {Electrode(0, e) for e in range(384, 768)}
    for e in sorted(source, key=lambda e: e.electrode_id):
        gui.electrodes.select(e)
    assert gui.electrodes.selected == source
    assert chunk <= gui.electrodes.unavailable
    return source, chunk


@pytest.mark.parametrize("probe_type", ["1.0", "2.0-1shank", "2.0-4shanks", "QuadBase"])
def test_zigzag_wiring_closure(gui, probe_type):
    gui.probe_type = probe_type
    subset = set(gui.get_zigzag_subset())
    zigzag = {e for e in gui.electrodes.wiring_map if e.electrode_id in subset}
    assert zigzag
    for e in zigzag:
        assert gui.electrodes.wiring_map[e] <= zigzag, (probe_type, e)


@pytest.mark.parametrize("probe_type", ["1.0", "2.0-1shank"])
def test_dense_source_keeps_complement_and_frees_only_zigzag(gui, probe_type):
    gui.probe_type = probe_type
    gui.select_mode = "zigzag_dependent_deselect"
    source, chunk = _dense_source(gui)
    subset = set(gui.get_zigzag_subset())
    zigzag_chunk = {e for e in chunk if e.electrode_id in subset}
    assert len(zigzag_chunk) == len(chunk) // 2

    gui.on_electrode_selection(None, [], _indices(gui, chunk))

    assert gui.electrodes.selected == {e for e in source if e.electrode_id not in subset}
    assert zigzag_chunk <= gui.electrodes.available
    assert chunk - zigzag_chunk <= gui.electrodes.unavailable
    assert not chunk & gui.electrodes.selected


def test_no_unavailable_electrodes_is_noop(gui):
    gui.electrodes.select(Electrode(0, 0))
    before = gui.electrodes.selected.copy()
    gui.on_electrode_selection(None, [], _indices(gui, {Electrode(0, 2), Electrode(0, 4)}))
    assert gui.electrodes.selected == before


def test_only_non_zigzag_unavailable_is_noop(gui):
    source, chunk = _dense_source(gui)
    subset = set(gui.get_zigzag_subset())
    targets = {e for e in chunk if e.electrode_id not in subset}
    gui.on_electrode_selection(None, [], _indices(gui, targets))
    assert gui.electrodes.selected == source
    assert chunk <= gui.electrodes.unavailable


def test_selected_electrodes_in_box_stay_selected(gui):
    source, chunk = _dense_source(gui)
    # These red sites have different wiring from the blocked sites in the box.
    red = {Electrode(0, 0), Electrode(0, 1)}
    targets = red | {Electrode(0, 386), Electrode(0, 387)}
    gui.on_electrode_selection(None, [], _indices(gui, targets))
    assert red <= gui.electrodes.selected
    assert Electrode(0, 2) not in gui.electrodes.selected
    assert Electrode(0, 3) in gui.electrodes.selected
    assert gui.electrodes.selected <= source
    assert not chunk & gui.electrodes.selected


def test_red_blocker_inside_box_is_deselected(gui):
    """Red sites in the box are not targets, but they are still deselected
    when they block a targeted black zigzag site in the same box."""
    _dense_source(gui)
    # 1.0: 2 blocks 386 (both zigzag); 3 blocks 387 (both non-zigzag).
    targets = {Electrode(0, 2), Electrode(0, 3), Electrode(0, 386), Electrode(0, 387)}
    gui.on_electrode_selection(None, [], _indices(gui, targets))
    assert Electrode(0, 2) not in gui.electrodes.selected
    assert Electrode(0, 386) in gui.electrodes.available
    assert Electrode(0, 3) in gui.electrodes.selected
    assert Electrode(0, 387) in gui.electrodes.unavailable


def test_empty_selection_is_noop(gui):
    gui.on_electrode_selection(None, [], [])
    assert gui.electrodes.selected == set()
    assert not gui.electrodes.can_undo()


def test_single_electrode_still_toggles(gui):
    target = _sites(gui)[0]
    gui.on_electrode_selection(None, [], [0])
    assert gui.electrodes.selected == {target}
    gui.on_electrode_selection(None, [], [0])
    assert gui.electrodes.selected == set()


def test_drag_is_one_undo_step(gui):
    source, chunk = _dense_source(gui)
    unavailable = gui.electrodes.unavailable.copy()
    gui.on_electrode_selection(None, [], _indices(gui, chunk))
    assert gui.electrodes.selected != source
    assert gui.electrodes.undo() is True
    assert gui.electrodes.selected == source
    assert gui.electrodes.unavailable == unavailable
    assert gui.electrodes.undo() is False


def test_existing_dependent_deselect_frees_whole_chunk(gui):
    _, chunk = _dense_source(gui)
    gui.select_mode = "dependent_deselect"
    gui.on_electrode_selection(None, [], _indices(gui, chunk))
    assert gui.electrodes.selected == set()
    assert chunk <= gui.electrodes.available


def test_tool_state_maps_new_mode(gui):
    gui.select_mode = "select"
    gui.on_tool_state_change(None, {}, {"active_tool": ["zigzag_dependent_deselect"]})
    assert gui.select_mode == "zigzag_dependent_deselect"


def test_toolbar_and_monitor_include_sixth_tool(gui):
    boxes = [tool for tool in gui.plot.toolbar.tools if isinstance(tool, BoxSelectTool)]
    assert len(boxes) == 6
    assert boxes[-1].description == "Zigzag-deselect Dependents"
    monitor = gui.plot.js_event_callbacks["selectiongeometry"][0]
    assert "Zigzag-deselect Dependents" in monitor.args.values()
    assert "zigzag_dependent_deselect" in monitor.code


def test_side_panel_help_documents_every_box_tool(gui):
    """Every toolbar box tool has an entry with an embedded icon in the help."""
    boxes = [tool for tool in gui.plot.toolbar.tools if isinstance(tool, BoxSelectTool)]
    assert len(gui._SELECTION_TOOL_DOCS) == len(boxes)
    html = gui._selection_tools_help_html()
    assert html.count('src="data:image/png;base64,') == len(boxes)
    assert "Zigzag-deselect dependents" in html

