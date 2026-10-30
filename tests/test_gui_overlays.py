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
import pytest

from pixelmap.anatomy import atlas as atlas_module
from pixelmap.anatomy import regions as regions_module
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
    gui.atlas_name_input.value = FAKE_ATLAS
    gui.bregma_relative_toggle.value = False
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


def _measured_depth(gui):
    """The depth the overlay compute would measure for ``gui``'s current pose."""
    return gui_module._measure_tip_depth(
        gui._tip_atlas_um(),
        float(gui.pitch_input.value),
        float(gui.yaw_input.value),
        gui._selected_atlas_name(),
    )


class TestTipDepthReadout:
    """The readout is tied to the overlay: blank until Compute, blank after Clear."""

    def test_readout_is_blank_before_the_overlay_is_computed(self, posed_gui):
        assert posed_gui._anatomy_overlay_active is False
        assert "—" in posed_gui.tip_depth_readout.object
        assert "compute the overlay" in posed_gui.tip_depth_readout.object
        assert "µm" not in posed_gui.tip_depth_readout.object

    def test_moving_the_tip_before_computing_does_not_fill_the_readout(self, posed_gui):
        posed_gui.tip_dv_input.value = 3500.0
        assert "compute the overlay" in posed_gui.tip_depth_readout.object

    def test_computing_the_overlay_fills_the_readout(self, posed_gui):
        posed_gui.compute_anatomy_overlay()
        # Surface at DV ≈ 987.5 µm, tip at DV = 4000 µm → ~3012 µm.
        assert "3,0" in posed_gui.tip_depth_readout.object
        assert "3.01 mm" in posed_gui.tip_depth_readout.object

    def test_clearing_the_overlay_blanks_the_readout_again(self, posed_gui):
        posed_gui.compute_anatomy_overlay()
        assert "µm" in posed_gui.tip_depth_readout.object

        posed_gui.clear_anatomy_overlay()

        assert "compute the overlay" in posed_gui.tip_depth_readout.object
        assert "µm" not in posed_gui.tip_depth_readout.object

    def test_readout_follows_the_tip_once_an_overlay_exists(self, posed_gui):
        posed_gui.compute_anatomy_overlay()
        posed_gui.tip_dv_input.value = 2000.0  # 1 mm below the fake surface
        assert "1,0" in posed_gui.tip_depth_readout.object
        assert _measured_depth(posed_gui) == pytest.approx(1012.5, abs=25.0)

    def test_tilting_the_probe_lengthens_the_measured_path(self, posed_gui):
        posed_gui.compute_anatomy_overlay()
        vertical = _measured_depth(posed_gui)
        posed_gui.pitch_input.value = 30.0
        tilted = _measured_depth(posed_gui)
        assert tilted > vertical
        assert tilted == pytest.approx(vertical / np.cos(np.deg2rad(30.0)), abs=30.0)

    def test_tip_above_the_brain_is_reported_as_outside(self, posed_gui):
        posed_gui.compute_anatomy_overlay()
        posed_gui.tip_dv_input.value = 100.0
        assert _measured_depth(posed_gui) is None
        assert "outside the brain" in posed_gui.tip_depth_readout.object

    @pytest.mark.parametrize("depth", [None, 0.0, 1012.5])
    def test_refreshing_the_readout_never_reads_the_atlas(
        self, posed_gui, monkeypatch, depth
    ):
        """The readout runs on the event loop, so it must only format a depth
        the off-loop compute already measured — never touch the atlas, which
        could mean a tens-of-MB download."""
        posed_gui.compute_anatomy_overlay()

        def _boom(*args, **kwargs):
            raise AssertionError("the readout read the atlas")

        monkeypatch.setattr(atlas_module, "is_downloaded", _boom)
        # regions.py imports the symbol directly, so patch it where it's used.
        monkeypatch.setattr(regions_module, "canonical_annotation", _boom)

        posed_gui._update_tip_depth_readout(depth)

        expected = "outside the brain" if depth is None else f"{depth:,.0f} µm"
        assert expected in posed_gui.tip_depth_readout.object


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

        for label in ("compute_region_bands", "render_locator", "tip_depth_below_surface_um"):
            monkeypatch.setattr(gui_module, label, _tracing(label, getattr(gui_module, label)))
        return seen

    def test_a_live_pose_edit_computes_off_the_event_loop(self, posed_gui, threads):
        posed_gui.compute_anatomy_overlay()
        before = list(posed_gui.region_band_source.data["y"])
        threads.clear()

        posed_gui.tip_dv_input.value = 2200.0

        for label in ("compute_region_bands", "render_locator", "tip_depth_below_surface_um"):
            _assert_off_loop(threads.get(label), label)
        assert list(posed_gui.region_band_source.data["y"]) != before
        # 2200 µm tip, surface at ≈ 987.5 µm → ≈ 1212 µm below it.
        assert "1,2" in posed_gui.tip_depth_readout.object

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
        assert "anterior commissure (defined)" in gui.bregma_estimate_header.object

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
