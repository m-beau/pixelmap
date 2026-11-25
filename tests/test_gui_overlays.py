"""End-to-end tests for the two GUI overlays: activity survey ⚡ and anatomy 🧠.

The other overlay test modules cover the pure computation layers
(``utils.survey``, ``anatomy.*``). These drive the ``ChannelmapGUI`` handlers
themselves — the code path a user actually triggers by clicking "Load survey
overlay" or "Compute anatomical overlay" — and assert on what ends up in the
Bokeh data sources and widgets.

No atlas is downloaded: ``BrainGlobeAtlas`` is patched with a tiny fake volume.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time

import numpy as np
import pandas as pd
import panel as pn
import pytest

from pixelmap.anatomy import atlas as atlas_module
from pixelmap.gui import gui as gui_module
from pixelmap.gui.gui import ChannelmapGUI
from pixelmap.types import Electrode

FAKE_ATLAS = "fake_gui_atlas"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gui():
    """One GUI instance for the module — construction loads all probe data."""
    return ChannelmapGUI()


@pytest.fixture(autouse=True)
def clean_overlays(gui):
    """Leave the GUI overlay-free before and after every test."""
    gui.clear_survey_overlay()
    gui.clear_anatomy_overlay()
    yield
    gui.clear_survey_overlay()
    gui.clear_anatomy_overlay()


def _survey_content(positions_df: pd.DataFrame, val_of) -> str:
    """Render a SpikeGLX-style survey .txt for the given probe positions."""
    lines = ["Shank\tXum\tZum\tVal"]
    for shank, _elec, x, y in positions_df[["shank", "electrode", "x", "y"]].itertuples(index=False):
        lines.append(f"{int(shank)}\t{int(x)}\t{int(y)}\t{val_of(shank, x, y)}")
    return "\n".join(lines) + "\n"


def _load_survey(gui, content: str, filename: str = "survey.txt"):
    """Drive the GUI's file-loading path the way the upload widget does."""
    gui.survey_file_loader.filename = filename
    gui.survey_file_loader.value = content.encode("utf-8")
    gui.load_survey_file()


@pytest.fixture
def layered_atlas(monkeypatch):
    """A DV-layered fake brain wide enough to hold the whole probe.

    Canonical indexing is (AP, DV, ML) at 25 µm. DV voxels 0–39 are outside the
    brain, 40–199 are region 1 (DEEP) then region 2 (SUP) — so the surface sits
    at DV ≈ 987.5 µm and there are 4 mm of brain below it.
    """
    ann = np.zeros((200, 200, 200), dtype=np.int32)
    ann[:, 40:120, :] = 2   # upper half of the brain
    ann[:, 120:200, :] = 1  # lower half

    class _FakeAtlas:
        def __init__(self, name, **_kwargs):
            self.name = name
            self.orientation = "asr"
            self.resolution = (25.0, 25.0, 25.0)
            self.annotation = ann
            self.structures = {
                1: {"acronym": "DEEP", "name": "Deep region", "rgb_triplet": [10, 20, 30]},
                2: {"acronym": "SUP", "name": "Superficial region", "rgb_triplet": [200, 100, 50]},
            }

    atlas_module.release_atlas_memory()
    atlas_module.get_atlas.cache_clear()
    atlas_module.canonical_annotation.cache_clear()
    atlas_module.derive_origin_from_ac.cache_clear()
    atlas_module._region_info_from_id.cache_clear()
    monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _FakeAtlas)
    # Pretend it's cached so the GUI treats it as a local atlas throughout.
    monkeypatch.setattr(atlas_module, "is_downloaded", lambda name=FAKE_ATLAS: True)
    yield ann
    atlas_module.release_atlas_memory()
    atlas_module.get_atlas.cache_clear()
    atlas_module.canonical_annotation.cache_clear()
    atlas_module.derive_origin_from_ac.cache_clear()
    atlas_module._region_info_from_id.cache_clear()


@pytest.fixture
def posed_gui(gui, layered_atlas):
    """GUI pointed at the fake atlas, tip 3 mm below its surface, vertical."""
    # Tip entry first: in Insertion mode the tip fields are derived, and the
    # writes below would be overwritten by the mirror.
    gui.coord_mode_input.value = "Tip"
    gui.atlas_name_input.value = FAKE_ATLAS
    gui.bregma_relative_toggle.value = False
    # The atlas transformation applies in either coordinate mode: start neutral.
    for w in (gui.ap_squish_input, gui.ml_squish_input, gui.dv_squish_input):
        w.value = 1.0
    gui.tilt_input.value = 0.0
    for w in (gui.bregma_ap_input, gui.bregma_ml_input, gui.bregma_dv_input):
        w.value = 0.0
    gui.pitch_input.value = 0.0
    gui.yaw_input.value = 0.0
    gui.shank_orientation_input.value = 0.0
    gui.tip_ap_input.value = 2500.0
    gui.tip_ml_input.value = 2500.0
    gui.tip_dv_input.value = 4000.0
    yield gui


# ==========================================================================
# Activity survey overlay
# ==========================================================================

class TestSurveyOverlayLoading:
    def test_loading_a_survey_populates_values_and_enables_the_controls(self, gui):
        content = _survey_content(gui.positions_df, lambda s, x, y: y / 100.0)
        _load_survey(gui, content)

        assert gui.survey_values is not None
        assert len(gui.survey_values) == len(gui.positions_df)
        assert gui.survey_vmin_input.disabled is False
        assert gui.survey_vmax_input.disabled is False
        assert gui.survey_color_bar.visible is True
        assert gui.survey_color_bar_title.visible is True

    def test_survey_values_land_on_the_right_electrodes(self, gui):
        # Encode the electrode's own depth so a mismatch is detectable.
        content = _survey_content(gui.positions_df, lambda s, x, y: y)
        _load_survey(gui, content)

        for shank, elec, _x, y in gui.positions_df[["shank", "electrode", "x", "y"]].head(50).itertuples(index=False):
            assert gui.survey_values[Electrode(int(shank), int(elec))] == pytest.approx(float(y))

    def test_bars_are_drawn_and_colored_by_value(self, gui):
        content = _survey_content(gui.positions_df, lambda s, x, y: y / 100.0)
        _load_survey(gui, content)

        data = gui.electrode_source.data
        assert all(a > 0 for a in data["bar_alpha"])
        # A depth gradient must produce many distinct colors, not one flat fill.
        assert len(set(data["bar_color"])) > 50
        assert not any(np.isnan(v) for v in data["val"])

    def test_electrode_colors_are_untouched_by_the_overlay(self, gui):
        """Selection state must stay readable underneath the survey bars."""
        gui.electrodes.select(Electrode(0, 0))
        gui.update_electrode_colors()
        before = list(gui.electrode_source.data["color"])

        content = _survey_content(gui.positions_df, lambda s, x, y: y / 100.0)
        _load_survey(gui, content)

        assert list(gui.electrode_source.data["color"]) == before
        gui.clear_selection()


class TestSurveyColormapDefaults:
    """The overlay must be legible with its preset vmin/vmax, before any edit."""

    def test_defaults_clip_the_tails_instead_of_using_raw_min_max(self, gui):
        rng = np.random.default_rng(0)
        # Heavy-tailed like a real spike-rate survey: a few very active contacts.
        vals = rng.gamma(2.0, 1.0, size=len(gui.positions_df))
        vals[:5] = 5_000.0
        by_row = {i: float(v) for i, v in enumerate(vals)}
        counter = iter(range(len(gui.positions_df)))
        content = _survey_content(gui.positions_df, lambda s, x, y: by_row[next(counter)])
        _load_survey(gui, content)

        assert gui.survey_cmap.high < vals.max() / 10  # outliers clipped away
        assert gui.survey_cmap.low > vals.min()

    def test_heavy_tailed_survey_still_renders_visible_contrast(self, gui):
        """Regression: the overlay used to look blank until vmin/vmax was edited."""
        rng = np.random.default_rng(1)
        vals = rng.gamma(2.0, 1.0, size=len(gui.positions_df))
        vals[:5] = 5_000.0
        counter = iter(range(len(gui.positions_df)))
        by_row = {i: float(v) for i, v in enumerate(vals)}
        content = _survey_content(gui.positions_df, lambda s, x, y: by_row[next(counter)])
        _load_survey(gui, content)

        # With a raw min→max range every contact would collapse onto the first
        # few colors of Viridis; with clipped tails the spread is wide.
        assert len(set(gui.electrode_source.data["bar_color"])) > 100

    def test_widgets_agree_with_the_colormap_after_loading(self, gui):
        content = _survey_content(gui.positions_df, lambda s, x, y: y / 100.0)
        _load_survey(gui, content)

        assert gui.survey_vmin_input.value == pytest.approx(gui.survey_cmap.low)
        assert gui.survey_vmax_input.value == pytest.approx(gui.survey_cmap.high)

    def test_constant_survey_does_not_produce_a_degenerate_range(self, gui):
        content = _survey_content(gui.positions_df, lambda s, x, y: 7.0)
        _load_survey(gui, content)
        assert gui.survey_cmap.high > gui.survey_cmap.low


class TestSurveyRangeEditing:
    def test_editing_vmax_rescales_the_colormap_and_redraws(self, gui):
        content = _survey_content(gui.positions_df, lambda s, x, y: y / 100.0)
        _load_survey(gui, content)
        before = list(gui.electrode_source.data["bar_color"])

        gui.survey_vmax_input.value = gui.survey_cmap.low + 1.0

        assert gui.survey_cmap.high == pytest.approx(gui.survey_vmax_input.value)
        assert list(gui.electrode_source.data["bar_color"]) != before

    def test_inverted_range_is_refused_and_leaves_the_colormap_alone(self, gui):
        content = _survey_content(gui.positions_df, lambda s, x, y: y / 100.0)
        _load_survey(gui, content)
        low, high = gui.survey_cmap.low, gui.survey_cmap.high

        gui.survey_vmin_input.value = high + 10.0  # vmin above vmax

        assert (gui.survey_cmap.low, gui.survey_cmap.high) == (low, high)


class TestSurveyOverlayRejection:
    def test_wrong_extension_is_rejected(self, gui):
        content = _survey_content(gui.positions_df, lambda s, x, y: 1.0)
        _load_survey(gui, content, filename="survey.csv")
        assert gui.survey_values is None

    def test_unparseable_content_is_rejected(self, gui):
        _load_survey(gui, "not a survey file at all\n")
        assert gui.survey_values is None

    def test_survey_from_a_probe_with_other_shanks_is_rejected(self):
        """A 4-shank survey must not silently overlay onto a 1-shank probe."""
        one_shank_gui = ChannelmapGUI()
        one_shank_gui.probe_type = "1.0"
        four_shank = pd.read_csv(one_shank_gui.wiring_maps_dir / "2.0-4shanks_positions.csv")
        content = _survey_content(four_shank, lambda s, x, y: 1.0)
        _load_survey(one_shank_gui, content)
        assert one_shank_gui.survey_values is None

    def test_clicking_load_without_a_file_is_a_no_op(self, gui):
        gui.survey_file_loader.value = None
        gui.load_survey_file()
        assert gui.survey_values is None


class TestSurveyOverlayClearing:
    def test_clearing_resets_values_bars_and_controls(self, gui):
        content = _survey_content(gui.positions_df, lambda s, x, y: y / 100.0)
        _load_survey(gui, content)

        gui.clear_survey_overlay()

        assert gui.survey_values is None
        assert gui.survey_vmin_input.disabled is True
        assert gui.survey_vmax_input.disabled is True
        assert gui.survey_color_bar.visible is False
        assert all(a == 0 for a in gui.electrode_source.data["bar_alpha"])
        assert all(np.isnan(v) for v in gui.electrode_source.data["val"])

    def test_switching_probe_type_drops_the_overlay(self):
        """Contact positions differ across probes, so a stale overlay would lie."""
        gui = ChannelmapGUI()
        content = _survey_content(gui.positions_df, lambda s, x, y: y / 100.0)
        _load_survey(gui, content)
        assert gui.survey_values is not None

        gui.probe_type = "1.0"

        assert gui.survey_values is None


# ==========================================================================
# Anatomical overlay
# ==========================================================================

class TestAnatomyOverlayCompute:
    def test_computing_populates_bands_labels_and_boundaries(self, posed_gui):
        posed_gui.compute_anatomy_overlay()

        assert posed_gui._anatomy_overlay_active is True
        assert len(posed_gui.region_band_source.data["x"]) > 0
        assert len(posed_gui.region_label_source.data["text"]) > 0
        assert len(posed_gui.region_boundary_source.data["y0"]) > 0
        assert posed_gui.anatomy_locator_section.visible is True

    def test_bands_report_the_regions_the_shank_actually_crosses(self, posed_gui):
        posed_gui.compute_anatomy_overlay()
        acronyms = set(posed_gui.region_band_source.data["acronym"])
        # A 4 mm-deep tip in a brain split in half at DV = 2 mm sees both halves.
        assert acronyms == {"DEEP", "SUP"}

    def test_band_colors_come_from_the_atlas_palette(self, posed_gui):
        posed_gui.compute_anatomy_overlay()
        colors = dict(zip(
            posed_gui.region_band_source.data["acronym"],
            posed_gui.region_band_source.data["color"],
        ))
        assert colors["DEEP"] == "#0a141e"    # rgb (10, 20, 30)
        assert colors["SUP"] == "#c86432"     # rgb (200, 100, 50)

    def test_deep_region_bands_sit_below_superficial_ones(self, posed_gui):
        posed_gui.compute_anatomy_overlay()
        data = posed_gui.region_band_source.data
        deep_y = [y for y, a in zip(data["y"], data["acronym"]) if a == "DEEP"]
        sup_y = [y for y, a in zip(data["y"], data["acronym"]) if a == "SUP"]
        # Probe y grows upward from the tip, and the tip is in the DEEP half.
        assert max(deep_y) < min(sup_y)

    def test_every_shank_gets_bands_on_a_multishank_probe(self, posed_gui):
        assert posed_gui.probe_type == "2.0-4shanks"
        posed_gui.compute_anatomy_overlay()
        centers = set(posed_gui._shank_plot_centers().values())
        assert set(posed_gui.region_band_source.data["x"]) == centers

    def test_legend_lists_the_traversed_regions(self, posed_gui):
        posed_gui.compute_anatomy_overlay()
        legend = posed_gui.anatomy_legend.object
        assert "DEEP" in legend and "SUP" in legend
        assert "Deep region" in legend and "Superficial region" in legend

    def test_probe_outside_the_volume_reports_no_regions(self, posed_gui):
        posed_gui.tip_ap_input.value = 500_000.0  # far outside the fake volume
        posed_gui.compute_anatomy_overlay()
        assert len(posed_gui.region_band_source.data["x"]) == 0
        assert "No regions found" in posed_gui.anatomy_legend.object


class TestAnatomyOverlayLiveUpdate:
    def test_moving_the_tip_updates_the_bands_without_recomputing_manually(self, posed_gui):
        posed_gui.compute_anatomy_overlay()
        before = list(posed_gui.region_band_source.data["y"])

        posed_gui.tip_dv_input.value = 2200.0  # pull the probe up by 1.8 mm

        assert list(posed_gui.region_band_source.data["y"]) != before

    def test_pose_changes_are_ignored_until_an_overlay_exists(self, posed_gui):
        assert posed_gui._anatomy_overlay_active is False
        posed_gui.tip_dv_input.value = 3800.0
        assert len(posed_gui.region_band_source.data["x"]) == 0


class TestAnatomyOverlayClearing:
    def test_clearing_wipes_every_anatomy_layer(self, posed_gui):
        posed_gui.compute_anatomy_overlay()
        posed_gui.clear_anatomy_overlay()

        assert posed_gui._anatomy_overlay_active is False
        assert posed_gui.region_band_source.data["x"] == []
        assert posed_gui.region_label_source.data["text"] == []
        assert posed_gui.region_boundary_source.data["y0"] == []
        assert posed_gui.anatomy_locator.object == ""
        assert posed_gui.anatomy_locator_section.visible is False
        assert "No anatomical overlay computed yet" in posed_gui.anatomy_legend.object


class TestTipDepthAlongShank:
    """The greyed-out depth field: insertion point → tip, along the shank.

    It replaced both the Entry DV field and the old read-only depth box.
    The fake brain's surface is at DV ≈ 987.5 µm (see ``layered_atlas``).
    """

    def test_it_is_derived_in_tip_mode_and_typed_in_insertion_mode(self, posed_gui):
        assert posed_gui.tip_depth_input.name == "Probe insertion depth (µm)"
        assert posed_gui.tip_depth_input.disabled
        posed_gui.coord_mode_input.value = "Insertion"
        assert not posed_gui.tip_depth_input.disabled
        posed_gui.coord_mode_input.value = "Tip"
        assert posed_gui.tip_depth_input.disabled
        assert not hasattr(posed_gui, "tip_depth_readout")
        assert not hasattr(posed_gui, "entry_dv_input")

    def test_it_needs_no_overlay(self, posed_gui):
        assert posed_gui._anatomy_overlay_active is False
        # Surface at DV ≈ 987.5 µm, tip at DV = 4000 µm → ~3012 µm.
        assert posed_gui.tip_depth_input.value == pytest.approx(3012.5, abs=25.0)

    def test_it_follows_the_tip(self, posed_gui):
        posed_gui.tip_dv_input.value = 2000.0
        assert posed_gui.tip_depth_input.value == pytest.approx(1012.5, abs=25.0)

    def test_tilting_the_probe_lengthens_the_path(self, posed_gui):
        vertical = posed_gui.tip_depth_input.value
        posed_gui.pitch_input.value = 30.0
        tilted = posed_gui.tip_depth_input.value
        assert tilted == pytest.approx(vertical / np.cos(np.deg2rad(30.0)), abs=30.0)

    def test_typing_it_in_insertion_mode_places_the_tip(self, posed_gui):
        posed_gui.coord_mode_input.value = "Insertion"
        posed_gui.tip_depth_input.value = 1000.0
        # Vertical probe: straight down from the surface at DV ≈ 987.5 µm.
        np.testing.assert_allclose(
            _triplet(posed_gui._tip_inputs), (2500.0, 2500.0, SURFACE_DV + 1000.0), atol=25.0)
        assert posed_gui.tip_depth_input.value == 1000.0  # never rewritten

    def test_in_insertion_mode_it_is_kept_when_the_probe_is_tilted(self, posed_gui):
        """The depth is travel along the shank, so a tilt shortens the DV reached."""
        posed_gui.coord_mode_input.value = "Insertion"
        posed_gui.tip_depth_input.value = 1000.0
        posed_gui.pitch_input.value = 60.0
        assert posed_gui.tip_depth_input.value == 1000.0
        entry = np.array(posed_gui._entry_atlas_um())
        tip = np.array(posed_gui._tip_atlas_um())
        assert np.linalg.norm(tip - entry) == pytest.approx(1000.0, abs=0.2)
        assert tip[2] - entry[2] == pytest.approx(500.0, abs=0.2)  # cos 60°

    def test_a_horizontal_probe_is_placed_too(self, posed_gui):
        posed_gui.coord_mode_input.value = "Insertion"
        posed_gui.tip_depth_input.value = 1000.0
        posed_gui.pitch_input.value = 90.0
        entry = np.array(posed_gui._entry_atlas_um())
        tip = np.array(posed_gui._tip_atlas_um())
        assert tip[2] == pytest.approx(entry[2], abs=0.2)
        assert abs(tip[0] - entry[0]) == pytest.approx(1000.0, abs=0.2)

    def test_a_tip_above_the_brain_reads_negative(self, posed_gui):
        posed_gui.tip_dv_input.value = 100.0
        assert posed_gui.tip_depth_input.value == pytest.approx(-887.5, abs=25.0)

    def test_it_is_blank_without_an_insertion_point(self, posed_gui):
        posed_gui.tip_ml_input.value = 99_999.0  # the trajectory misses the brain
        assert posed_gui.tip_depth_input.value is None

    def test_it_is_the_distance_between_the_two_markers(self, posed_gui):
        posed_gui.pitch_input.value = 20.0
        posed_gui.yaw_input.value = -10.0
        tip = np.array(posed_gui._tip_atlas_um())
        entry = np.array(posed_gui._entry_atlas_um())
        assert posed_gui.tip_depth_input.value == pytest.approx(
            np.linalg.norm(tip - entry), abs=0.2)


class TestBothOverlaysTogether:
    def test_survey_and_anatomy_overlays_coexist(self, posed_gui):
        """The headline workflow: anatomy bands behind, survey bars alongside."""
        content = _survey_content(posed_gui.positions_df, lambda s, x, y: y / 100.0)
        _load_survey(posed_gui, content)
        posed_gui.compute_anatomy_overlay()

        assert posed_gui.survey_values is not None
        assert len(posed_gui.region_band_source.data["x"]) > 0
        assert all(a > 0 for a in posed_gui.electrode_source.data["bar_alpha"])

    def test_region_labels_shift_outward_to_clear_the_survey_bars(self, posed_gui):
        posed_gui.compute_anatomy_overlay()
        plain_offsets = [
            x - c for x, c in zip(
                posed_gui.region_label_source.data["x"],
                posed_gui.region_label_source.data["center"],
            )
        ]

        content = _survey_content(posed_gui.positions_df, lambda s, x, y: y / 100.0)
        _load_survey(posed_gui, content)
        posed_gui.compute_anatomy_overlay()
        with_survey = [
            x - c for x, c in zip(
                posed_gui.region_label_source.data["x"],
                posed_gui.region_label_source.data["center"],
            )
        ]

        assert max(with_survey) > max(plain_offsets)


# ==========================================================================
# Document locking around the async compute-overlay callback
# ==========================================================================

@pytest.fixture
def served(posed_gui):
    """Attach the GUI's data sources to a real Bokeh ``ServerSession``.

    This is what makes the served app's failure reproducible in-process: the
    ``_pending_writes`` guard lives on ``ServerSession``, so a bare Document
    never raises no matter how a callback is scheduled.

    Teardown detaches the sources again — the ``gui`` fixture is module-scoped,
    and a session left attached makes every later test's overlay bookkeeping
    raise.
    """
    from bokeh.document import Document
    from bokeh.server.session import ServerSession

    sources = (
        posed_gui.region_band_source,
        posed_gui.region_label_source,
        posed_gui.region_boundary_source,
    )
    doc = Document()
    for source in sources:
        doc.add_root(source)
    session = ServerSession("test-session", doc, io_loop=None, token="test-token")
    yield posed_gui, session
    # Unsubscribe the session *before* touching the document: removing a root
    # is itself a document change, so it would need the lock we are tearing
    # down. Detached, the sources go back to being plain models.
    doc.callbacks.remove_on_change(session)
    for source in sources:
        doc.remove_root(source)



def _run_through_panel_scheduler(gui, timeout: float = 5.0) -> list[BaseException]:
    """Click Compute the way a served app does; return exceptions Panel caught.

    Builds a Document with a session context and a real ``ServerSession`` on a
    live IOLoop, then hands the callback to ``panel.io.server.async_execute``
    — the function Panel routes every async widget callback through. Panel
    funnels failures into ``state._handle_exception``, which is intercepted
    here rather than left to surface as a Tornado log line.
    """
    from functools import partial

    from bokeh.document import Document
    from bokeh.server.session import ServerSession
    from panel.io.server import async_execute
    from panel.io.state import state
    from tornado.ioloop import IOLoop

    sources = (
        gui.region_band_source,
        gui.region_label_source,
        gui.region_boundary_source,
    )
    errors: list[BaseException] = []

    class _SessionContext:
        """The minimum ``async_execute`` checks for to take the served path."""

        destroyed = False

        def __init__(self, doc):
            self._document = doc

        @property
        def session(self):
            return None

    async def _drive():
        doc = Document()
        for source in sources:
            doc.add_root(source)
        doc._session_context = lambda: _SessionContext(doc)
        session = ServerSession("test-session", doc, io_loop=IOLoop.current(),
                                token="test-token")

        original_handler = state._handle_exception
        state._handle_exception = errors.append
        state.curdoc = doc
        try:
            async_execute(partial(gui._on_compute_anatomy_click, None))
            deadline = asyncio.get_running_loop().time() + timeout
            while asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
                if errors or gui.region_band_source.data["acronym"]:
                    break
            # Let any queued document writes drain before we tear the doc down.
            await asyncio.sleep(0.05)
        finally:
            state.curdoc = None
            state._handle_exception = original_handler
            doc.callbacks.remove_on_change(session)
            for source in sources:
                doc.remove_root(source)
        return errors

    return asyncio.run(_drive())


class TestComputeButtonRunsUnderTheDocumentLock:
    """Regression tests for a dead anatomy overlay on the served app.

    ``_on_compute_anatomy_click`` is ``async`` so the atlas download can go to
    a worker thread instead of stalling the event loop for every other user.
    But Panel schedules async callbacks with ``nolock`` set unless the
    function carries ``lock = True`` (``panel.io.server.async_execute``), and
    this handler assigns to Bokeh ``ColumnDataSource`` models directly. Run
    unlocked, every one of those assignments raises "_pending_writes should be
    non-None ...", which fails *quietly* from the user's point of view: the
    exception goes to the server log, the overlay never appears, and the
    button is left stuck on its "Downloading…" label.

    Panel's own widgets tolerate unlocked callbacks, so nothing else in the
    GUI notices a missing lock — hence these tests.
    """

    def test_handler_is_marked_to_run_with_the_document_lock(self):
        """The wiring that makes Panel hand the callback the lock.

        Cheap, and the only check that catches the mistake at its source: the
        behavioural tests below supply the lock themselves, so they pass
        either way.
        """
        handler = ChannelmapGUI._on_compute_anatomy_click
        assert getattr(handler, "lock", False) is True, (
            "_on_compute_anatomy_click must be decorated with @pn.io.with_lock "
            "or Panel will schedule it nolock and every Bokeh model write in it "
            "will raise"
        )

    def test_the_harness_reproduces_the_unlocked_failure(self, served):
        """Guards the guard: prove an unlocked write really does raise here."""
        gui, _session = served
        with pytest.raises(RuntimeError, match="_pending_writes"):
            gui.region_band_source.data = {
                "x": [], "y": [], "width": [], "height": [], "color": [], "acronym": [],
            }

    def test_computing_under_the_lock_populates_the_overlay(self, served):
        """The whole callback, on an already-downloaded atlas, end to end."""
        gui, session = served

        asyncio.run(session.with_document_locked(gui._on_compute_anatomy_click, None))

        assert len(gui.region_band_source.data["acronym"]) > 0
        assert gui.compute_anatomy_button.name == "Compute anatomical overlay 🧠"
        assert gui.compute_anatomy_button.disabled is False

    def test_clicking_compute_renders_through_panels_own_scheduler(self, posed_gui):
        """The definitive test: Panel schedules the callback, as it does live.

        The tests above hand the callback a lock themselves, so they pass with
        or without the decorator. This one goes through
        ``panel.io.server.async_execute`` against a document with a session
        context — the same route a real button click takes — so it is the one
        that actually fails if the lock wiring regresses.

        It asserts on the *exception*, not on the data source, because that is
        the shape of the bug: ``source.data = ...`` assigns the new value and
        only then raises while notifying the session. Server-side state
        therefore looks correct while the browser is never told anything, which
        is why the overlay silently failed to appear.
        """
        errors = _run_through_panel_scheduler(posed_gui)

        assert not errors, (
            "Panel-scheduled callback raised "
            f"{type(errors[0]).__name__}: {errors[0]} — the handler is not "
            "running under the document lock, so overlay updates never reach "
            "the browser"
        )
        assert len(posed_gui.region_band_source.data["acronym"]) > 0

    def test_a_cold_atlas_is_downloaded_off_the_event_loop(self, served, monkeypatch):
        """The download must happen in a worker thread, not on the loop.

        On the server the calling thread serves every other session, so a
        download that runs inline stalls the whole app — the behaviour this
        handler exists to prevent.
        """
        gui, session = served
        download_threads = []
        downloaded = []

        # Cold until ensure_downloaded has run, warm afterwards.
        monkeypatch.setattr(atlas_module, "is_downloaded", lambda name: bool(downloaded))

        def _fake_download(name):
            download_threads.append(threading.current_thread())
            downloaded.append(name)

        monkeypatch.setattr(atlas_module, "ensure_downloaded", _fake_download)

        asyncio.run(session.with_document_locked(gui._on_compute_anatomy_click, None))

        assert downloaded == [FAKE_ATLAS]
        assert download_threads[0] is not threading.main_thread(), (
            "the atlas download ran on the event loop thread"
        )
        assert len(gui.region_band_source.data["acronym"]) > 0

    def test_a_failed_download_leaves_the_button_usable(self, served, monkeypatch):
        """A network failure must not strand the button mid-"Downloading…"."""
        gui, session = served
        monkeypatch.setattr(atlas_module, "is_downloaded", lambda name: False)

        def _boom(name):
            raise ConnectionError("atlas host unreachable")

        monkeypatch.setattr(atlas_module, "ensure_downloaded", _boom)

        asyncio.run(session.with_document_locked(gui._on_compute_anatomy_click, None))

        assert gui.compute_anatomy_button.disabled is False
        assert gui.compute_anatomy_button.name == "Download & compute atlas 🧠 ⏳"


class TestAtlasSwitchNeverDownloadsOnTheEventLoop:
    """Regression tests for a server-wide freeze on atlas switch.

    With an overlay on screen, picking a not-yet-cached atlas from the dropdown
    live-recomputed the overlay *synchronously* — so the multi-second atlas
    download ran on the Bokeh event loop, freezing every session and the
    healthcheck (the watchdog logged 5–10 s stalls in
    ``_on_atlas_change → _recompute_anatomy_if_active → compute_anatomy_overlay``).
    The Compute button already pushed downloads to a worker thread; the live
    update path has to do the same.
    """

    COLD = "cold_gui_atlas"
    COLDER = "colder_gui_atlas"

    @pytest.fixture
    def traced(self, posed_gui, monkeypatch):
        """Record every atlas construction and download, with its thread.

        Atlases in ``cold`` count as not downloaded until ``ensure_downloaded``
        has fetched them; constructing one before then is exactly the inline
        network fetch that froze the server.
        """
        events: list[tuple[str, str, threading.Thread]] = []
        cold = {self.COLD, self.COLDER}
        fake_atlas = atlas_module.BrainGlobeAtlas

        class _TracingAtlas(fake_atlas):
            def __init__(self, name, **kwargs):
                events.append(("construct", name, threading.current_thread()))
                super().__init__(name, **kwargs)

        def _fake_download(name):
            events.append(("download", name, threading.current_thread()))
            cold.discard(name)

        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _TracingAtlas)
        monkeypatch.setattr(atlas_module, "is_downloaded", lambda name=FAKE_ATLAS: name not in cold)
        monkeypatch.setattr(atlas_module, "ensure_downloaded", _fake_download)
        return posed_gui, events, cold

    def _assert_downloaded_off_loop_before_use(self, events, name):
        mine = [(kind, thread) for kind, n, thread in events if n == name]
        assert mine, f"{name} was never downloaded or opened"
        kind, thread = mine[0]
        assert kind == "download", (
            f"{name} was opened before being downloaded — on a cold cache that "
            "is an inline network fetch"
        )
        assert thread is not threading.main_thread(), (
            f"{name} was downloaded on the event loop thread"
        )

    def test_switching_to_a_cold_atlas_downloads_off_the_event_loop(self, traced):
        gui, events, _cold = traced
        gui.compute_anatomy_overlay()
        assert gui._anatomy_overlay_active is True

        gui.atlas_name_input.value = self.COLD

        self._assert_downloaded_off_loop_before_use(events, self.COLD)
        # And the overlay is redrawn for the new atlas once it has arrived.
        assert gui._anatomy_overlay_active is True
        assert len(gui.region_band_source.data["acronym"]) > 0
        assert gui.compute_anatomy_button.disabled is False
        assert gui.compute_anatomy_button.name == "Compute anatomical overlay 🧠"

    def test_switching_again_mid_download_fetches_the_latest_off_the_loop(self, traced, monkeypatch):
        """The atlas can change while the first download is still running.

        The draw that follows a download reads the *current* dropdown value, so
        it must not assume that is the atlas it just fetched.
        """
        gui, events, cold = traced
        gui.compute_anatomy_overlay()

        def _download_then_switch(name):
            events.append(("download", name, threading.current_thread()))
            cold.discard(name)
            if name == self.COLD:
                gui.atlas_name_input.value = self.COLDER  # user picks again

        monkeypatch.setattr(atlas_module, "ensure_downloaded", _download_then_switch)

        gui.atlas_name_input.value = self.COLD

        downloads = [n for kind, n, _t in events if kind == "download"]
        assert downloads == [self.COLD, self.COLDER], (
            "each atlas should be fetched exactly once, the second by the "
            "in-flight download rather than a parallel or inline one"
        )
        self._assert_downloaded_off_loop_before_use(events, self.COLDER)
        assert len(gui.region_band_source.data["acronym"]) > 0
        assert gui.compute_anatomy_button.disabled is False

    def test_a_failed_download_on_switch_leaves_the_button_usable(self, traced, monkeypatch):
        gui, events, _cold = traced
        gui.compute_anatomy_overlay()

        def _boom(name):
            raise ConnectionError("atlas host unreachable")

        monkeypatch.setattr(atlas_module, "ensure_downloaded", _boom)

        gui.atlas_name_input.value = self.COLD

        assert not [e for e in events if e[1] == self.COLD and e[2] is threading.main_thread()]
        assert gui.compute_anatomy_button.disabled is False
        assert gui.compute_anatomy_button.name == "Download & compute atlas 🧠 ⏳"

    def test_switching_to_a_cold_atlas_without_an_overlay_downloads_nothing(self, traced):
        gui, events, _cold = traced
        assert gui._anatomy_overlay_active is False

        gui.atlas_name_input.value = self.COLD

        assert not [e for e in events if e[1] == self.COLD]
        assert gui.compute_anatomy_button.name == "Download & compute atlas 🧠 ⏳"

    def test_switching_to_a_warm_atlas_still_recomputes_immediately(self, traced):
        gui, events, cold = traced
        gui.compute_anatomy_overlay()
        cold.discard(self.COLD)  # already on disk
        gui.region_band_source.data = {k: [] for k in gui.region_band_source.data}

        gui.atlas_name_input.value = self.COLD

        assert not [e for e in events if e[0] == "download"]
        assert len(gui.region_band_source.data["acronym"]) > 0


def _assert_off_loop(threads, what: str):
    """``what`` ran, and never on the main (event loop) thread."""
    assert threads, f"{what} never ran"
    assert all(t is not threading.main_thread() for t in threads), (
        f"{what} ran on the event loop thread"
    )


class TestOverlayComputeRunsOffTheEventLoop:
    """Regression tests: even a *cached* atlas froze the server on every edit.

    Slicing the annotation and rendering the locator takes 0.7 s per overlay
    compute on allen_mouse_25um and 3.3 s on allen_mouse_10um (measured), and
    the overlay recomputes on every pose keystroke — all of it on the Bokeh
    event loop that serves every session.  Locating an anterior-commissure
    origin scans the whole volume (0.8–8 s), and ran on the loop on every
    switch to a human or fish atlas.  All of these must happen in a worker.
    """

    @pytest.fixture
    def threads(self, monkeypatch):
        """Record which thread each atlas-reading step of the overlay runs on."""
        seen: dict[str, list[threading.Thread]] = {}

        def _tracing(label, fn):
            def wrapper(*args, **kwargs):
                seen.setdefault(label, []).append(threading.current_thread())
                return fn(*args, **kwargs)
            return wrapper

        for label in ("compute_region_bands", "render_locator", "surface_point_um"):
            monkeypatch.setattr(gui_module, label, _tracing(label, getattr(gui_module, label)))
        return seen

    def test_a_live_pose_edit_computes_off_the_event_loop(self, posed_gui, threads):
        posed_gui.compute_anatomy_overlay()
        before = list(posed_gui.region_band_source.data["y"])
        threads.clear()

        posed_gui.tip_dv_input.value = 2200.0

        for label in ("compute_region_bands", "render_locator", "surface_point_um"):
            _assert_off_loop(threads.get(label), label)
        assert list(posed_gui.region_band_source.data["y"]) != before
        # 2200 µm tip, surface at ≈ 987.5 µm → ≈ 1212 µm below it.
        assert posed_gui.tip_depth_input.value == pytest.approx(1212.5, abs=25.0)

    def test_clicking_compute_on_a_cached_atlas_computes_off_the_loop(self, served, threads):
        gui, session = served

        asyncio.run(session.with_document_locked(gui._on_compute_anatomy_click, None))

        for label in ("compute_region_bands", "render_locator"):
            _assert_off_loop(threads.get(label), label)
        assert len(gui.region_band_source.data["acronym"]) > 0
        assert gui.compute_anatomy_button.disabled is False

    def test_edits_during_a_compute_are_coalesced_into_one_more(self, posed_gui, monkeypatch):
        """A burst of keystrokes must not queue one multi-second compute each."""
        posed_gui.compute_anatomy_overlay()
        real = gui_module.compute_region_bands
        tips = []

        def _slow_bands(**kwargs):
            tips.append(kwargs["tip_atlas"][2])
            if len(tips) == 1:  # the user keeps typing while this one runs
                for dv in (2100.0, 2200.0, 2300.0):
                    posed_gui.tip_dv_input.value = dv
            return real(**kwargs)

        monkeypatch.setattr(gui_module, "compute_region_bands", _slow_bands)

        posed_gui.tip_dv_input.value = 2000.0

        assert tips == [2000.0, 2300.0], "expected the first edit, then only the latest"
        assert posed_gui._anatomy_compute_running is False

    def test_clearing_mid_compute_discards_the_result(self, posed_gui, monkeypatch):
        posed_gui.compute_anatomy_overlay()
        real = gui_module.compute_region_bands

        def _cleared_meanwhile(**kwargs):
            bands = real(**kwargs)
            posed_gui.clear_anatomy_overlay()  # the user clicks Clear
            return bands

        monkeypatch.setattr(gui_module, "compute_region_bands", _cleared_meanwhile)

        posed_gui.tip_dv_input.value = 2200.0

        assert posed_gui._anatomy_overlay_active is False
        assert posed_gui.region_band_source.data["x"] == []
        assert posed_gui.anatomy_locator.object == ""

    def test_a_lookup_failure_leaves_the_button_usable(self, posed_gui, monkeypatch):
        posed_gui.compute_anatomy_overlay()

        def _boom(**kwargs):
            raise ValueError("bad pose")

        monkeypatch.setattr(gui_module, "compute_region_bands", _boom)

        posed_gui.tip_dv_input.value = 2200.0

        assert posed_gui._anatomy_compute_running is False
        assert posed_gui.compute_anatomy_button.disabled is False
        assert posed_gui.compute_anatomy_button.name == "Compute anatomical overlay 🧠"


class TestAnteriorCommissureOriginIsLocatedOffTheEventLoop:
    """Human and fish atlases use the AC as origin; finding it scans every voxel."""

    AC_ATLAS = "fake_zfish_atlas"  # "zfish" → anterior-commissure landmark
    AC_ORIGIN = (100.0, 200.0, 300.0)  # (AP, ML, DV) µm

    @pytest.fixture
    def ac_threads(self, posed_gui, monkeypatch):
        seen: list[threading.Thread] = []

        def _derive(name):
            seen.append(threading.current_thread())
            return self.AC_ORIGIN

        monkeypatch.setattr(atlas_module, "_derive_origin_from_ac", _derive)
        return posed_gui, seen

    def _assert_origin_filled(self, gui):
        assert (gui.bregma_ap_input.value, gui.bregma_ml_input.value,
                gui.bregma_dv_input.value) == self.AC_ORIGIN
        assert "anterior commissure (defined)" in gui.bregma_hint.value

    def test_switching_to_a_cached_ac_atlas_derives_it_off_the_loop(self, ac_threads):
        gui, seen = ac_threads
        assert gui._anatomy_overlay_active is False

        gui.atlas_name_input.value = self.AC_ATLAS

        _assert_off_loop(seen, "the whole-volume AC scan")
        self._assert_origin_filled(gui)

    def test_with_an_overlay_the_recompute_derives_it_off_the_loop(self, ac_threads):
        gui, seen = ac_threads
        gui.compute_anatomy_overlay()

        gui.atlas_name_input.value = self.AC_ATLAS

        _assert_off_loop(seen, "the whole-volume AC scan")
        self._assert_origin_filled(gui)
        assert len(gui.region_band_source.data["acronym"]) > 0

    def test_origin_spec_never_scans_on_its_own(self, ac_threads):
        """It runs on the loop several times per switch, so it may only peek."""
        gui, seen = ac_threads
        assert gui._origin_spec(self.AC_ATLAS)["status"] == "ac_pending"
        assert seen == []


def test_no_unlocked_async_gui_callbacks():
    """Every async GUI callback needs the same treatment.

    Enforced rather than left to review: an unlocked async callback fails
    only on the served app, only in the server log, and no ordinary test
    notices.
    """
    unlocked = [
        name
        for name, member in inspect.getmembers(ChannelmapGUI)
        if inspect.iscoroutinefunction(member) and not getattr(member, "lock", False)
    ]
    assert not unlocked, (
        f"async GUI callbacks missing @pn.io.with_lock: {unlocked}. Panel "
        "schedules these nolock, so any Bokeh model write inside them raises."
    )


class TestClearOverlayReleasesAtlasMemory:
    """"Clear overlay" must hand the annotation volume back, not just blank the plot.

    Blanking the Bokeh sources left the atlas resident — 308 MB for
    allen_mouse_25um, up to 4.8 GB for a 10 µm atlas — so a user who cleared the
    overlay saw RSS stay exactly where it was.
    """

    @staticmethod
    def _wait_for_release(timeout: float = 5.0) -> bool:
        """The release runs on a daemon thread; give it a moment to land."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if atlas_module._resident is None:
                return True
            time.sleep(0.01)
        return atlas_module._resident is None

    def test_clearing_the_overlay_releases_the_atlas(self, posed_gui):
        posed_gui.compute_anatomy_overlay()
        assert atlas_module._resident is not None, "overlay should have loaded an atlas"

        posed_gui.clear_anatomy_overlay()

        assert self._wait_for_release(), "clear overlay left the atlas resident in RAM"

    def test_recomputing_after_a_clear_still_works(self, posed_gui):
        """Releasing must be transparent — the next compute reloads from disk."""
        posed_gui.compute_anatomy_overlay()
        posed_gui.clear_anatomy_overlay()
        assert self._wait_for_release()

        posed_gui.compute_anatomy_overlay()
        assert atlas_module._resident is not None
        assert len(posed_gui.region_band_source.data["acronym"]) > 0, (
            "overlay did not come back after a release"
        )


class TestLocatorFigureIsReusedAcrossRenders:
    """The GUI must keep one locator figure per session, not one per render."""

    def test_recomputing_reuses_the_same_figure(self, posed_gui):
        posed_gui.compute_anatomy_overlay()
        first = posed_gui._locator_fig
        assert first is not None, "no locator figure was retained"

        for _ in range(4):
            posed_gui.compute_anatomy_overlay()

        assert posed_gui._locator_fig is first, (
            "a new figure was built per overlay compute — 3.1 MB of RSS each, "
            "on a path that re-runs for every pose edit"
        )

    def test_clearing_the_overlay_drops_the_figure(self, posed_gui):
        posed_gui.compute_anatomy_overlay()
        assert posed_gui._locator_fig is not None
        posed_gui.clear_anatomy_overlay()
        assert posed_gui._locator_fig is None, (
            "an idle session is still holding a matplotlib canvas"
        )


# ==========================================================================
# Tip / insertion-point coordinate entry
# ==========================================================================

SURFACE_DV = 987.5  # the fake brain's surface (see ``layered_atlas``)


def _entry(gui):
    """The insertion point as displayed: Entry AP, Entry ML and its (hidden) DV."""
    vals = (gui.entry_ap_input.value, gui.entry_ml_input.value, gui._entry_dv)
    return np.array([np.nan if v is None else float(v) for v in vals])


def _triplet(widgets):
    return np.array([np.nan if w.value is None else float(w.value) for w in widgets])


class TestCoordinateEntryModes:
    """Which fields are typed, which are derived, and what an angle pivots around."""

    def test_mode_switch_swaps_which_fields_are_editable(self, posed_gui):
        assert all(not w.disabled for w in posed_gui._tip_inputs)
        assert posed_gui.entry_ap_input.disabled and posed_gui.entry_ml_input.disabled
        assert posed_gui.tip_depth_input.disabled
        posed_gui.coord_mode_input.value = "Insertion"
        # The whole tip is derived; the craniotomy and the depth are typed.
        assert all(w.disabled for w in posed_gui._tip_inputs)
        assert not posed_gui.entry_ap_input.disabled
        assert not posed_gui.entry_ml_input.disabled
        assert not posed_gui.tip_depth_input.disabled

    def test_there_is_no_separate_depth_input(self, posed_gui):
        """One depth field does both jobs: readout in Tip mode, input in Insertion."""
        assert not hasattr(posed_gui, "insertion_depth_input")
        assert not hasattr(posed_gui, "snap_surface_button")

    @pytest.mark.parametrize("pitch, yaw", [(0.0, 0.0), (17.0, -11.0)])
    def test_repeated_toggling_does_not_move_the_probe(self, posed_gui, pitch, yaw):
        posed_gui.pitch_input.value = pitch
        posed_gui.yaw_input.value = yaw
        tip = _triplet(posed_gui._tip_inputs)
        entry = _entry(posed_gui)
        for _ in range(10):
            posed_gui.coord_mode_input.value = "Insertion"
            posed_gui.coord_mode_input.value = "Tip"
        np.testing.assert_allclose(_triplet(posed_gui._tip_inputs), tip, atol=0.11)
        np.testing.assert_array_equal(_entry(posed_gui), entry)

    def test_toggling_never_rewrites_tip_dv(self, posed_gui):
        """A value typed with more precision than the display rounding survives."""
        posed_gui.pitch_input.value = 17.0
        posed_gui.tip_dv_input.value = 4000.04
        posed_gui.coord_mode_input.value = "Insertion"
        posed_gui.coord_mode_input.value = "Tip"
        assert posed_gui.tip_dv_input.value == 4000.04

    def test_insertion_mode_puts_the_tip_down_the_shank_from_the_entry(self, posed_gui):
        from pixelmap.anatomy.transform import probe_axis_up

        posed_gui.coord_mode_input.value = "Insertion"
        posed_gui.pitch_input.value = 20.0
        posed_gui.yaw_input.value = -10.0
        posed_gui.entry_ap_input.value = 3000.0
        posed_gui.entry_ml_input.value = 2000.0
        posed_gui.tip_depth_input.value = 2500.0

        entry = np.array(posed_gui._entry_atlas_um())
        tip = np.array(posed_gui._tip_atlas_um())
        np.testing.assert_allclose(entry[:2], (3000.0, 2000.0))
        np.testing.assert_allclose(
            tip, entry - 2500.0 * probe_axis_up(20.0, -10.0), atol=0.1)

    def test_moving_the_craniotomy_keeps_the_depth(self, posed_gui):
        posed_gui.coord_mode_input.value = "Insertion"
        depth = posed_gui.tip_depth_input.value
        posed_gui.entry_ap_input.value = 3000.0
        assert posed_gui.tip_depth_input.value == depth
        assert posed_gui.tip_ap_input.value == pytest.approx(3000.0, abs=0.05)

    def test_insertion_mode_pivots_around_the_entry_point(self, posed_gui):
        posed_gui.coord_mode_input.value = "Insertion"
        entry = _entry(posed_gui)
        tip = _triplet(posed_gui._tip_inputs)
        depth = posed_gui.tip_depth_input.value
        posed_gui.pitch_input.value = 30.0
        np.testing.assert_array_equal(_entry(posed_gui), entry)
        assert posed_gui.tip_depth_input.value == depth
        assert posed_gui.tip_ap_input.value != tip[0]
        assert posed_gui.tip_dv_input.value < tip[2]  # same travel, less of it downward

    def test_tip_mode_pivots_around_the_tip(self, posed_gui):
        tip = _triplet(posed_gui._tip_inputs)
        entry = _entry(posed_gui)
        posed_gui.pitch_input.value = 30.0
        np.testing.assert_array_equal(_triplet(posed_gui._tip_inputs), tip)
        assert not np.allclose(_entry(posed_gui), entry)

    def test_a_cleared_field_does_not_raise(self, posed_gui):
        posed_gui.coord_mode_input.value = "Insertion"
        posed_gui.entry_ap_input.value = None  # what the browser sends on clear
        posed_gui.tip_depth_input.value = None
        posed_gui.coord_mode_input.value = "Tip"
        posed_gui.tip_ap_input.value = None

    def test_one_coordinate_edit_computes_once_with_the_derived_tip(
        self, posed_gui, monkeypatch
    ):
        """Deriving the tip must finish before the overlay snapshots it."""
        posed_gui.compute_anatomy_overlay()
        posed_gui.coord_mode_input.value = "Insertion"
        posed_gui.pitch_input.value = 20.0
        requests = []
        original = posed_gui._compute_anatomy

        def _spy(req):
            requests.append(req)
            return original(req)

        monkeypatch.setattr(posed_gui, "_compute_anatomy", _spy)
        posed_gui.entry_ap_input.value = 2800.0

        assert len(requests) == 1
        np.testing.assert_allclose(requests[0].tip, posed_gui._tip_atlas_um())
        np.testing.assert_allclose(requests[0].entry, posed_gui._entry_atlas_um())
        assert requests[0].entry[0] == 2800.0


class TestEntryFollowsSurface:
    """The insertion point is always on the brain surface, in both modes."""

    def _assert_on_surface(self, gui):
        assert gui._entry_atlas_um()[2] == pytest.approx(SURFACE_DV, abs=25.0)

    def test_entry_is_blank_until_the_surface_has_been_read(self):
        fresh = gui_module.ChannelmapGUI()  # the module's GUI has been posed already
        assert fresh.entry_ap_input.value is None and fresh.entry_ml_input.value is None
        assert fresh.tip_depth_input.value is None
        assert fresh._entry_atlas_um() is None

    def test_tip_mode_entry_sits_where_the_trajectory_crosses_the_surface(self, posed_gui):
        from pixelmap.anatomy.transform import probe_axis_up

        posed_gui.pitch_input.value = 20.0
        entry = np.array(posed_gui._entry_atlas_um())
        tip = np.array(posed_gui._tip_atlas_um())
        self._assert_on_surface(posed_gui)
        along = (entry - tip) / np.linalg.norm(entry - tip)
        np.testing.assert_allclose(along, probe_axis_up(20.0, 0.0), atol=1e-4)

    def test_tip_mode_entry_follows_every_tip_edit(self, posed_gui):
        """Regression: the entry kept the DV it had been given in Insertion mode."""
        posed_gui.coord_mode_input.value = "Insertion"
        posed_gui.coord_mode_input.value = "Tip"

        posed_gui.tip_ap_input.value = 3100.0
        posed_gui.tip_ml_input.value = 1900.0
        posed_gui.tip_dv_input.value = 2000.0
        np.testing.assert_allclose(
            _entry(posed_gui), (3100.0, 1900.0, SURFACE_DV), atol=25.0
        )
        # Moving the tip deeper must not drag the entry off the surface.
        posed_gui.tip_dv_input.value = 4500.0
        self._assert_on_surface(posed_gui)

    def test_tip_mode_entry_follows_the_surface_relief(self, posed_gui, layered_atlas):
        """A taller brain at another AP: the entry DV has to move with it."""
        layered_atlas[120:, 20:40, :] = 2  # surface 500 µm higher from AP 3000
        atlas_module.release_atlas_memory()
        posed_gui.tip_ap_input.value = 3500.0
        assert posed_gui._entry_atlas_um()[2] == pytest.approx(SURFACE_DV - 500.0, abs=25.0)
        posed_gui.tip_ap_input.value = 2500.0
        self._assert_on_surface(posed_gui)

    def test_a_trajectory_that_misses_the_brain_has_no_entry(self, posed_gui):
        posed_gui.tip_ml_input.value = 99_999.0
        assert posed_gui._entry_atlas_um() is None
        assert np.isnan(_entry(posed_gui)).all()
        posed_gui.compute_anatomy_overlay()  # and the locator copes without one

    def test_insertion_mode_entry_drops_onto_the_surface_whatever_the_tilt(self, posed_gui):
        posed_gui.pitch_input.value = 20.0
        posed_gui.yaw_input.value = -10.0
        posed_gui.coord_mode_input.value = "Insertion"

        posed_gui.entry_ap_input.value = 3000.0
        posed_gui.entry_ml_input.value = 2000.0

        assert posed_gui.entry_ap_input.value == 3000.0
        assert posed_gui.entry_ml_input.value == 2000.0
        self._assert_on_surface(posed_gui)

    def test_no_brain_below_the_entry_blanks_its_dv(self, posed_gui):
        posed_gui.coord_mode_input.value = "Insertion"
        posed_gui.entry_ml_input.value = 99_999.0  # far outside the volume
        assert posed_gui._entry_dv is None

    def test_without_a_local_atlas_compute_settles_the_entry(self, posed_gui, monkeypatch):
        monkeypatch.setattr(atlas_module, "is_downloaded", lambda name=None: False)
        posed_gui.tip_ap_input.value = 2600.0
        stale = posed_gui.entry_ap_input.value
        assert stale != 2600.0  # could not be read yet

        monkeypatch.setattr(atlas_module, "is_downloaded", lambda name=None: True)
        posed_gui.compute_anatomy_overlay()

        assert posed_gui.entry_ap_input.value == pytest.approx(2600.0, abs=0.05)
        self._assert_on_surface(posed_gui)

    def test_the_surface_is_read_off_the_event_loop(self, posed_gui, monkeypatch):
        threads = []
        original = gui_module.surface_point_um

        def _spy(*args, **kwargs):
            threads.append(threading.current_thread())
            return original(*args, **kwargs)

        monkeypatch.setattr(gui_module, "surface_point_um", _spy)
        posed_gui.tip_ap_input.value = 2700.0
        posed_gui.coord_mode_input.value = "Insertion"
        posed_gui.entry_ap_input.value = 2800.0
        assert threads and all(t is not threading.main_thread() for t in threads)

    def test_an_edit_during_the_lookup_discards_the_result(self, posed_gui, monkeypatch):
        """A surface found for the old pose must not be written over a new one."""
        original = gui_module.surface_point_um
        seen = []

        def _slow(point, **kwargs):
            seen.append(point)
            if len(seen) == 1:
                posed_gui.tip_ap_input.value = 3300.0  # the user keeps typing
            return original(point, **kwargs)

        monkeypatch.setattr(gui_module, "surface_point_um", _slow)
        posed_gui.tip_ap_input.value = 3200.0
        assert posed_gui.entry_ap_input.value == pytest.approx(3300.0, abs=0.05)

    @pytest.mark.parametrize("pitch", [0.0, 20.0])
    def test_computing_after_the_switch_does_not_nudge_the_tip(self, posed_gui, pitch):
        """Tip mode finds the surface along the shank, Insertion mode straight
        down; their sampling grids differ by up to a voxel, which must not
        move the probe."""
        posed_gui.pitch_input.value = pitch
        tip = _triplet(posed_gui._tip_inputs)
        posed_gui.coord_mode_input.value = "Insertion"
        posed_gui.compute_anatomy_overlay()
        np.testing.assert_allclose(_triplet(posed_gui._tip_inputs), tip, atol=0.11)


class TestEntryHonoursTheAtlasWarp:
    """Squish / tilt reshape the drawn brain; the entry must sit on *that* surface.

    Regression: the surface was read from the unwarped atlas, so in bregma
    mode the marker floated above (or sank into) the brain in the locator.
    """

    @pytest.fixture
    def bregma_gui(self, posed_gui):
        posed_gui.bregma_relative_toggle.value = True
        # Bregma well inside the fake volume, 1 mm below its surface.
        posed_gui.bregma_ap_input.value = 2500.0
        posed_gui.bregma_ml_input.value = 2500.0
        posed_gui.bregma_dv_input.value = SURFACE_DV + 1000.0
        for w in (posed_gui.ap_squish_input, posed_gui.ml_squish_input,
                  posed_gui.dv_squish_input):
            w.value = 1.0
        posed_gui.tilt_input.value = 0.0
        yield posed_gui
        posed_gui.bregma_relative_toggle.value = False

    def test_unwarped_entry_is_one_mm_above_bregma(self, bregma_gui):
        np.testing.assert_allclose(_entry(bregma_gui), (0, 0, -1000.0), atol=25.0)

    @pytest.mark.parametrize("mode", ["Tip", "Insertion"])
    def test_dv_squish_moves_the_surface_toward_bregma(self, bregma_gui, mode):
        bregma_gui.coord_mode_input.value = mode
        bregma_gui.dv_squish_input.value = 0.8
        # The brain is squashed about bregma: 1000 µm above it becomes 800.
        assert bregma_gui._entry_dv == pytest.approx(-800.0, abs=25.0)
        bregma_gui.dv_squish_input.value = 1.0
        assert bregma_gui._entry_dv == pytest.approx(-1000.0, abs=25.0)

    @pytest.mark.parametrize("mode", ["Tip", "Insertion"])
    def test_tilt_moves_the_surface(self, bregma_gui, mode):
        """A flat surface tilted about bregma is higher on one side of it."""
        bregma_gui.coord_mode_input.value = mode
        bregma_gui.tilt_input.value = 10.0
        field = bregma_gui.tip_ap_input if mode == "Tip" else bregma_gui.entry_ap_input
        field.value = 1000.0
        ahead = bregma_gui._entry_dv
        field.value = -1000.0
        behind = bregma_gui._entry_dv
        # Plane through (0, -1000/cos t) with slope tan t along AP.
        expected = 2 * 1000.0 * np.tan(np.deg2rad(10.0))
        assert abs(ahead - behind) == pytest.approx(expected, abs=40.0)

    def test_entry_lands_on_the_surface_the_locator_draws(self, bregma_gui):
        """The entry, mapped back into the atlas, is on the annotation's surface."""
        bregma_gui.dv_squish_input.value = 0.85
        bregma_gui.ap_squish_input.value = 0.95
        bregma_gui.tilt_input.value = 13.0
        bregma_gui.tip_ap_input.value = 600.0
        in_atlas = bregma_gui._atlas_warp().to_atlas(bregma_gui._entry_atlas_um())
        assert in_atlas[2] == pytest.approx(SURFACE_DV, abs=30.0)

    def test_a_landmark_edit_moves_the_entry(self, bregma_gui):
        bregma_gui.bregma_dv_input.value = SURFACE_DV + 1500.0
        assert bregma_gui._entry_dv == pytest.approx(-1500.0, abs=25.0)

    def test_the_depth_is_measured_to_the_warped_surface(self, bregma_gui):
        bregma_gui.dv_squish_input.value = 0.8
        bregma_gui.tip_dv_input.value = 1500.0
        # 1500 below bregma + the squashed 800 above it; unwarped would be 2500.
        assert bregma_gui.tip_depth_input.value == pytest.approx(2300.0, abs=25.0)


class TestAtlasTransformationWithoutBregmaMode:
    """Squish / tilt are editable and applied whichever frame coordinates use."""

    def test_the_fields_are_always_editable(self, posed_gui):
        fields = (posed_gui.dv_squish_input, posed_gui.ap_squish_input,
                  posed_gui.ml_squish_input, posed_gui.tilt_input)
        assert posed_gui.bregma_relative_toggle.value is False
        assert all(not w.disabled for w in fields)
        posed_gui.bregma_relative_toggle.value = True
        assert all(not w.disabled for w in fields)
        posed_gui.bregma_relative_toggle.value = False
        assert all(not w.disabled for w in fields)

    def test_section_is_titled_and_explained(self, posed_gui):
        assert "Atlas transformation" in posed_gui.atlas_transform_title.object
        assert "Julie Fabre" in posed_gui.atlas_transform_hint.value

    def test_neutral_values_mean_no_warp(self, posed_gui):
        assert posed_gui._atlas_warp() is None

    def test_dv_squish_moves_the_surface_in_absolute_coordinates(self, posed_gui):
        posed_gui.bregma_dv_input.value = SURFACE_DV + 1000.0
        try:
            posed_gui.dv_squish_input.value = 0.8
            # Squashed about the landmark: 1000 µm above it becomes 800.
            assert posed_gui._entry_dv == pytest.approx(SURFACE_DV + 200.0, abs=25.0)
            posed_gui.compute_anatomy_overlay()  # and the overlay draws with it
        finally:
            posed_gui.dv_squish_input.value = 1.0
            posed_gui.bregma_dv_input.value = 0.0
        assert posed_gui._entry_dv == pytest.approx(SURFACE_DV, abs=25.0)


class TestMoveProbeTipToRegion:
    """Search the atlas for a region and drop the tip in the middle of it.

    The fake brain has "SUP" (DV voxels 40–79) above "DEEP" (80 and below).
    """

    @staticmethod
    def _pick(gui, query):
        gui.region_search_input.value_input = query
        ids = list(gui.region_results.options.values())
        assert ids, f"no result for {query!r}"
        gui.region_results.value = [ids[0]]

    def _region_at_tip(self, gui):
        return atlas_module.lookup_regions(FAKE_ATLAS, [gui._tip_atlas_um()])[0].acronym

    def test_typing_lists_matches_by_acronym_or_name(self, posed_gui):
        posed_gui.region_search_input.value_input = "deep"
        assert list(posed_gui.region_results.options) == ["DEEP — Deep region"]
        posed_gui.region_search_input.value_input = "superficial"
        assert list(posed_gui.region_results.options) == ["SUP — Superficial region"]
        assert posed_gui.region_results.visible

    def test_a_typo_still_finds_the_region(self, posed_gui):
        posed_gui.region_search_input.value_input = "superfical regon"
        assert list(posed_gui.region_results.options) == ["SUP — Superficial region"]

    def test_typing_never_moves_the_tip(self, posed_gui):
        tip = _triplet(posed_gui._tip_inputs)
        for query in ("d", "de", "deep", "sup", ""):
            posed_gui.region_search_input.value_input = query
        np.testing.assert_array_equal(_triplet(posed_gui._tip_inputs), tip)

    def test_no_match_says_so(self, posed_gui):
        posed_gui.region_search_input.value_input = "zzzzzz"
        assert not posed_gui.region_results.visible
        assert "No region matches" in posed_gui.region_search_status.object

    @pytest.mark.parametrize("query, acronym", [("SUP", "SUP"), ("deep region", "DEEP")])
    def test_picking_a_region_moves_the_tip_into_it(self, posed_gui, query, acronym):
        self._pick(posed_gui, query)
        assert self._region_at_tip(posed_gui) == acronym
        assert acronym in posed_gui.region_search_status.object

    def test_the_same_region_can_be_picked_again(self, posed_gui):
        """Regression: it stayed selected, so a second click was not a change."""
        self._pick(posed_gui, "SUP")
        there = _triplet(posed_gui._tip_inputs)
        posed_gui.tip_dv_input.value = 4000.0  # wander off by hand
        self._pick(posed_gui, "SUP")
        np.testing.assert_array_equal(_triplet(posed_gui._tip_inputs), there)

    def test_the_angles_are_kept(self, posed_gui):
        posed_gui.pitch_input.value = 17.0
        posed_gui.yaw_input.value = -8.0
        posed_gui.shank_orientation_input.value = 45.0
        self._pick(posed_gui, "SUP")
        assert (posed_gui.pitch_input.value, posed_gui.yaw_input.value,
                posed_gui.shank_orientation_input.value) == (17.0, -8.0, 45.0)

    def test_the_tip_stays_in_its_hemisphere(self, posed_gui):
        midline = 199 / 2 * 25.0
        posed_gui.tip_ml_input.value = 1000.0
        self._pick(posed_gui, "SUP")
        assert posed_gui.tip_ml_input.value < midline
        posed_gui.tip_ml_input.value = 4000.0
        self._pick(posed_gui, "DEEP")
        assert posed_gui.tip_ml_input.value > midline

    def test_the_insertion_point_and_depth_follow(self, posed_gui):
        self._pick(posed_gui, "DEEP")
        assert posed_gui.entry_ap_input.value == posed_gui.tip_ap_input.value
        assert posed_gui._entry_atlas_um()[2] == pytest.approx(SURFACE_DV, abs=25.0)
        assert posed_gui.tip_depth_input.value == pytest.approx(
            posed_gui.tip_dv_input.value - SURFACE_DV, abs=25.0)

    def test_from_insertion_mode_it_switches_to_tip_entry(self, posed_gui):
        posed_gui.coord_mode_input.value = "Insertion"
        self._pick(posed_gui, "SUP")
        assert posed_gui.coord_mode_input.value == "Tip"
        assert self._region_at_tip(posed_gui) == "SUP"

    def test_it_lands_in_the_region_under_an_atlas_transformation(self, posed_gui):
        posed_gui.bregma_relative_toggle.value = True
        try:
            posed_gui.bregma_ap_input.value = 2500.0
            posed_gui.bregma_ml_input.value = 2500.0
            posed_gui.bregma_dv_input.value = 1500.0
            posed_gui.dv_squish_input.value = 0.8
            posed_gui.tilt_input.value = 10.0
            self._pick(posed_gui, "SUP")
            in_atlas = posed_gui._atlas_warp().to_atlas(posed_gui._tip_atlas_um())
            assert atlas_module.lookup_regions(FAKE_ATLAS, [in_atlas])[0].acronym == "SUP"
        finally:
            posed_gui.bregma_relative_toggle.value = False

    def test_an_active_overlay_is_redrawn_at_the_new_tip(self, posed_gui, monkeypatch):
        posed_gui.compute_anatomy_overlay()
        requests = []
        original = posed_gui._compute_anatomy

        def _spy(req):
            requests.append(req)
            return original(req)

        monkeypatch.setattr(posed_gui, "_compute_anatomy", _spy)
        self._pick(posed_gui, "SUP")
        assert requests
        np.testing.assert_allclose(requests[-1].tip, posed_gui._tip_atlas_um())

    def test_the_volume_is_scanned_off_the_event_loop(self, posed_gui, monkeypatch):
        threads = []
        original = gui_module.region_target_um

        def _spy(*args, **kwargs):
            threads.append(threading.current_thread())
            return original(*args, **kwargs)

        monkeypatch.setattr(gui_module, "region_target_um", _spy)
        self._pick(posed_gui, "DEEP")
        assert threads and threads[0] is not threading.main_thread()

    def test_search_waits_for_the_atlas_to_be_local(self, posed_gui, monkeypatch):
        monkeypatch.setattr(atlas_module, "is_downloaded", lambda name=None: False)
        posed_gui._refresh_compute_button()
        assert posed_gui.region_search_input.disabled
        assert "Download the atlas" in posed_gui.region_search_input.placeholder
        monkeypatch.setattr(atlas_module, "is_downloaded", lambda name=None: True)
        posed_gui._refresh_compute_button()
        assert not posed_gui.region_search_input.disabled


class TestAnatomyPanelCollapsesWithoutAnOverlay:
    """No overlay: just the atlas dropdown and Compute. The rest appears with it."""

    def _extras(self, gui):
        return (gui.anatomy_pose_section, gui.clear_anatomy_button, gui.region_search_section)

    def test_a_fresh_gui_shows_only_the_atlas_and_compute(self):
        fresh = gui_module.ChannelmapGUI()
        fresh.create_layout()
        assert all(not w.visible for w in self._extras(fresh))
        assert fresh.atlas_name_input.visible and fresh.compute_anatomy_button.visible
        # ... which are not inside the collapsed section.
        assert fresh.atlas_name_input not in fresh.anatomy_pose_section
        assert fresh.tip_ap_input in list(fresh.anatomy_pose_section.select(pn.widgets.FloatInput))

    def test_computing_reveals_the_controls_and_the_region_search(self, posed_gui):
        posed_gui.create_layout()
        assert all(not w.visible for w in self._extras(posed_gui))
        posed_gui.compute_anatomy_overlay()
        assert all(w.visible for w in self._extras(posed_gui))

    def test_clearing_collapses_it_again(self, posed_gui):
        posed_gui.create_layout()
        posed_gui.compute_anatomy_overlay()
        posed_gui.clear_anatomy_overlay()
        assert all(not w.visible for w in self._extras(posed_gui))

    def test_the_region_search_sits_under_the_locator_on_the_right(self, posed_gui):
        layout = posed_gui.create_layout()
        columns = [c for c in layout.select(pn.Column)
                   if posed_gui.anatomy_locator_section in c.objects]
        assert len(columns) == 1
        objects = columns[0].objects
        assert (objects.index(posed_gui.region_search_section)
                > objects.index(posed_gui.anatomy_legend)
                > objects.index(posed_gui.anatomy_locator_section))


class TestLocatorTitlesSpeakTheTypedCoordinates:
    """Regression: the panel titles quoted atlas µm even in bregma-relative mode."""

    @staticmethod
    def _titles(gui):
        return [ax.get_title() for ax in gui._locator_fig.axes[:3]]

    def test_absolute_mode_quotes_atlas_coordinates(self, posed_gui):
        posed_gui.compute_anatomy_overlay()
        assert self._titles(posed_gui) == [
            "Sagittal · AP 2500 µm", "Coronal · ML 2500 µm", "Horizontal · DV 4000 µm"]

    def test_bregma_mode_quotes_coordinates_relative_to_bregma(self, posed_gui):
        posed_gui.bregma_relative_toggle.value = True
        try:
            posed_gui.bregma_ap_input.value = 2000.0
            posed_gui.bregma_ml_input.value = 2400.0
            posed_gui.bregma_dv_input.value = 1000.0
            posed_gui.tip_ap_input.value = -700.0
            posed_gui.tip_ml_input.value = 300.0
            posed_gui.tip_dv_input.value = 1500.0
            posed_gui.compute_anatomy_overlay()
            assert self._titles(posed_gui) == [
                "Sagittal · AP -700 µm", "Coronal · ML 300 µm", "Horizontal · DV 1500 µm"]
        finally:
            posed_gui.bregma_relative_toggle.value = False


class TestBregmaRelativeByDefault:
    """Coordinates start relative to the landmark whenever the atlas has one."""

    def test_an_atlas_with_a_bregma_estimate_starts_ticked(self):
        fresh = gui_module.ChannelmapGUI()
        assert fresh._selected_atlas_name().startswith("allen_mouse")
        assert fresh.bregma_relative_toggle.value is True
        # ... with the bregma-relative default tip: at bregma, 4 mm deep.
        assert [w.value for w in fresh._tip_inputs] == [0.0, 0.0, 4000.0]

    def test_an_atlas_without_a_landmark_is_left_in_atlas_coordinates(self, posed_gui):
        assert posed_gui._selected_atlas_name() == FAKE_ATLAS
        assert posed_gui._landmark_known(FAKE_ATLAS) is False
        posed_gui.bregma_relative_toggle.value = True
        posed_gui.atlas_name_input.value = "allen_mouse_25um"
        posed_gui.atlas_name_input.value = FAKE_ATLAS
        assert posed_gui.bregma_relative_toggle.value is False

    def test_switching_to_an_atlas_with_bregma_ticks_it(self, posed_gui, monkeypatch):
        assert posed_gui.bregma_relative_toggle.value is False
        calls = []
        monkeypatch.setattr(posed_gui, "_refresh_probe_placement", lambda: calls.append(1))
        posed_gui.atlas_name_input.value = "whs_sd_rat_39um"
        try:
            assert posed_gui.bregma_relative_toggle.value is True
            assert [w.value for w in posed_gui._tip_inputs] == [0.0, 0.0, 4000.0]
            assert calls == []  # one refresh for the switch, by the atlas handler
        finally:
            posed_gui.atlas_name_input.value = FAKE_ATLAS


class TestLocatorLegendQuotesTheFields:
    """The legend writes the tip and insertion point exactly as the fields show them."""

    @staticmethod
    def _labels(gui):
        return [t.get_text() for t in gui._locator_fig.axes[3].get_legend().get_texts()]

    def test_tip_and_entry_coordinates_match_the_fields(self, posed_gui):
        posed_gui.pitch_input.value = 20.0
        posed_gui.compute_anatomy_overlay()
        tip, entry = self._labels(posed_gui)[:2]
        fields = [w.value for w in posed_gui._tip_inputs]
        assert tip.endswith(", ".join(f"{v:.0f}" for v in fields))
        shown = (posed_gui.entry_ap_input.value, posed_gui.entry_ml_input.value,
                 posed_gui.tip_depth_input.value)
        assert "depth along shank" in entry
        assert entry.endswith(", ".join(f"{v:.0f}" for v in shown))

    def test_coordinate_fields_carry_their_unit(self, posed_gui):
        for w in (posed_gui.entry_ap_input, posed_gui.entry_ml_input, posed_gui.tip_depth_input):
            assert w.name.endswith("(µm)")


class TestExplanationsSitBehindHoverIcons:
    """Help text is on "?" hover icons, not printed in the panel."""

    def test_the_inline_notes_are_gone(self, posed_gui):
        for attr in ("anatomy_coord_note", "anatomy_help", "bregma_estimate_header",
                     "atlas_transform_note"):
            assert not hasattr(posed_gui, attr)

    def test_atlas_dropdown_carries_the_coordinate_note(self, posed_gui):
        text = posed_gui.atlas_name_input.description
        assert "fixed CCF / atlas frame" in text
        assert f"<b>{FAKE_ATLAS}</b> native voxel orientation" in text

    def test_the_coordinate_note_follows_the_atlas(self, posed_gui):
        posed_gui.atlas_name_input.value = "whs_sd_rat_39um"
        try:
            assert "<b>whs_sd_rat_39um</b>" in posed_gui.atlas_name_input.description
        finally:
            posed_gui.atlas_name_input.value = FAKE_ATLAS

    def test_each_angle_explains_itself(self, posed_gui):
        assert posed_gui.pitch_input.name == "Pitch (AP tilt; °)"
        assert posed_gui.yaw_input.name == "Yaw (ML tilt; °)"
        assert posed_gui.shank_orientation_input.name == "Roll (rotation; °)"
        assert "forward / backward" in posed_gui.pitch_input.description
        assert "left / right" in posed_gui.yaw_input.description
        assert "shanks 0→3" in posed_gui.shank_orientation_input.description

    def test_the_bregma_note_sits_beside_the_toggle_and_follows_the_atlas(self, posed_gui):
        layout = posed_gui.create_layout()
        rows = [r for r in layout.select(pn.Row) if posed_gui.bregma_hint in r.objects]
        assert len(rows) == 1 and rows[0].objects[0] is posed_gui.bregma_relative_toggle
        posed_gui.atlas_name_input.value = "allen_mouse_25um"
        try:
            assert "Bregma estimate (rough)" in posed_gui.bregma_hint.value
            assert "Julie Fabre" in posed_gui.bregma_hint.value
        finally:
            posed_gui.atlas_name_input.value = FAKE_ATLAS

    def test_the_transformation_note_sits_beside_its_title(self, posed_gui):
        layout = posed_gui.create_layout()
        rows = [r for r in layout.select(pn.Row) if posed_gui.atlas_transform_hint in r.objects]
        assert len(rows) == 1 and rows[0].objects[0] is posed_gui.atlas_transform_title
