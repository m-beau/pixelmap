"""Tests for the locator schematic (brain slice + colored regions + probe).

As elsewhere in the anatomy tests, we patch BrainGlobeAtlas with a tiny fake
atlas so nothing is downloaded.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from pixelmap.anatomy import atlas as atlas_module
from pixelmap.anatomy import schematic as schematic_module
from pixelmap.anatomy.schematic import render_locator


class _FakeAtlas:
    """Fake atlas: left half (ML 1-4) region 1, right half (ML 5-8) region 2."""

    def __init__(self, name: str, **_kwargs):
        self.name = name
        self.orientation = "asr"
        self.resolution = (25.0, 25.0, 25.0)  # (AP, DV, ML) µm/voxel
        ann = np.zeros((8, 6, 10), dtype=np.int32)  # (AP, DV, ML)
        ann[1:7, 1:5, 1:5] = 1
        ann[1:7, 1:5, 5:9] = 2
        self.annotation = ann
        self.structures = {
            1: {"acronym": "L", "name": "Left", "rgb_triplet": [200, 0, 0]},
            2: {"acronym": "R", "name": "Right", "rgb_triplet": [0, 0, 200]},
        }


@pytest.fixture(autouse=True)
def _reset_caches():
    # ``schematic._atlas_data`` is deliberately uncached now (it used to pin a
    # fourth set of annotation volumes), so there is nothing to clear for it.
    atlas_module.release_atlas_memory()
    atlas_module.get_atlas.cache_clear()
    atlas_module.canonical_annotation.cache_clear()
    yield
    atlas_module.release_atlas_memory()
    atlas_module.get_atlas.cache_clear()
    atlas_module.canonical_annotation.cache_clear()


@pytest.fixture
def fake_brainglobe(monkeypatch):
    monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _FakeAtlas)
    yield


def _render(shank_positions=None):
    return render_locator(
        "fake",
        tip_atlas=(100.0, 100.0, 75.0),  # ap_idx=4, ml_idx=4 → inside the blob
        pitch_deg=0.0,
        yaw_deg=0.0,
        shank_orientation_deg=0.0,
        shank_positions=shank_positions or {0: 0.0, 1: 50.0},
        y_range=(0.0, 80.0),
    )


def test_render_has_three_slices_plus_legend_with_a_probe_per_shank(fake_brainglobe):
    fig = _render()
    assert len(fig.axes) == 4                    # 3 slices + 1 legend panel
    for ax in fig.axes[:3]:                       # sagittal, coronal, horizontal
        assert len(ax.get_lines()) >= 2          # one Line2D per shank
        assert len(ax.images) >= 1               # region fill (+ outline) layer


def test_bregma_dot_adds_one_marker_per_slice(fake_brainglobe):
    common = dict(tip_atlas=(100.0, 100.0, 75.0), pitch_deg=0.0, yaw_deg=0.0,
                  shank_orientation_deg=0.0, shank_positions={0: 0.0}, y_range=(0.0, 80.0))
    without = render_locator("fake", **common)
    with_bregma = render_locator("fake", bregma_um=(100.0, 100.0, 50.0), **common)
    for a_no, a_yes in zip(without.axes[:3], with_bregma.axes[:3]):
        assert len(a_yes.get_lines()) == len(a_no.get_lines()) + 1


def test_region_fill_uses_atlas_colors(fake_brainglobe):
    atlas = _FakeAtlas("fake")
    label_img = np.array([[0, 1, 2]], dtype=np.int32)
    rgba = schematic_module._region_fill_rgba(label_img, atlas)
    assert tuple(rgba[0, 0]) == (0.0, 0.0, 0.0, 0.0)         # outside → transparent
    np.testing.assert_allclose(rgba[0, 1, :3], (200 / 255, 0, 0))  # region 1 red
    np.testing.assert_allclose(rgba[0, 2, :3], (0, 0, 200 / 255))  # region 2 blue
    assert rgba[0, 1, 3] > 0 and rgba[0, 2, 3] > 0


def test_outline_marks_region_boundaries(fake_brainglobe):
    label_img = np.array([[1, 1, 2, 2]], dtype=np.int32)
    outline = schematic_module._outline_rgba(label_img)
    # The 1↔2 transition (column 1) should be marked; uniform interior not.
    assert outline[0, 1, 3] > 0
    assert outline[0, 0, 3] == 0


def test_falls_back_to_silhouette_when_tip_outside_volume(fake_brainglobe):
    fig = render_locator(
        "fake",
        tip_atlas=(1e6, 1e6, 1e6),  # clamps to an edge slice with no brain
        pitch_deg=0.0,
        yaw_deg=0.0,
        shank_orientation_deg=0.0,
        shank_positions={0: 0.0},
        y_range=(0.0, 80.0),
    )
    assert len(fig.axes) == 4  # 3 slices + legend; renders without crashing



class _LoadTrackingAtlas:
    """Fake atlas whose annotation arrays are individually weak-referenced.

    Liveness is tracked on the *array*, not on the atlas object: a cache that
    stores the array directly keeps it alive even after the owning atlas has
    dropped its own reference, and that is exactly the failure mode being
    guarded against.
    """

    alive: list = []  # weakrefs to every annotation array ever materialised

    def __init__(self, name: str, **_kwargs):
        self.name = name
        self.orientation = "asr"
        self.resolution = (25.0, 25.0, 25.0)
        self.shape = (8, 6, 10)
        self.structures = {
            1: {"acronym": "L", "name": "Left", "rgb_triplet": [200, 0, 0]},
            2: {"acronym": "R", "name": "Right", "rgb_triplet": [0, 0, 200]},
        }
        self._annotation = None

    @property
    def annotation(self):
        if self._annotation is None:
            ann = np.zeros((8, 6, 10), dtype=np.int32)
            ann[1:7, 1:5, 1:5] = 1
            ann[1:7, 1:5, 5:9] = 2
            type(self).alive.append(weakref.ref(ann))
            self._annotation = ann
        return self._annotation

    @classmethod
    def live_annotations(cls) -> int:
        gc.collect()
        return sum(ref() is not None for ref in cls.alive)


def test_rendering_many_atlases_keeps_one_annotation_resident(monkeypatch):
    """The locator must not build an atlas cache of its own.

    ``_atlas_data`` used to be an ``lru_cache(maxsize=4)`` holding
    ``(atlas, annotation, resolution)``. Keyed and evicted independently of the
    caches in :mod:`pixelmap.anatomy.atlas`, it pinned up to four more
    annotation volumes — hundreds of MB apiece — which is part of why the
    server's RSS climbed monotonically as a user browsed atlases. Note it held
    the *array*, so releasing the atlas object would not have freed it.
    """
    monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _LoadTrackingAtlas)
    monkeypatch.setattr(_LoadTrackingAtlas, "alive", [])

    for i in range(12):  # three times the old cache bound
        render_locator(
            f"atlas_{i}",
            tip_atlas=(100.0, 100.0, 75.0),
            pitch_deg=0.0,
            yaw_deg=0.0,
            shank_orientation_deg=0.0,
            shank_positions={0: 0.0},
            y_range=(0.0, 80.0),
        )
        live = _LoadTrackingAtlas.live_annotations()
        assert live == 1, (
            f"{live} annotation volumes still in RAM after rendering atlas_{i}; "
            "exactly 1 is allowed"
        )


class TestSilhouetteIsOnlyBuiltWhenNeeded:
    """The whole-brain silhouette is the one thing here that reads every voxel.

    It is the fallback for a tip outside the volume, so on the normal path —
    tip in the brain, three real slices to draw — it must never be built. It
    used to be computed unconditionally (``inside = ann > 0``), which cost a
    full-size bool array on every single render: 1.2 GB for a 10 µm atlas.
    """

    @staticmethod
    def _count_projections(monkeypatch) -> list[int]:
        from pixelmap.anatomy.volume import CanonicalVolume

        calls = [0]
        original = CanonicalVolume.projections

        def counted(self):
            calls[0] += 1
            return original(self)

        monkeypatch.setattr(CanonicalVolume, "projections", counted)
        return calls

    def test_tip_inside_the_volume_never_builds_it(self, fake_brainglobe, monkeypatch):
        calls = self._count_projections(monkeypatch)
        _render()
        assert calls[0] == 0, "silhouette was built even though every view had a slice"

    def test_tip_outside_the_volume_falls_back_to_it(self, fake_brainglobe, monkeypatch):
        calls = self._count_projections(monkeypatch)
        # Far outside the 8x6x10-voxel fake brain.
        render_locator(
            "fake",
            tip_atlas=(50_000.0, 50_000.0, 50_000.0),
            pitch_deg=0.0,
            yaw_deg=0.0,
            shank_orientation_deg=0.0,
            shank_positions={0: 0.0},
            y_range=(0.0, 80.0),
        )
        assert calls[0] > 0, "no slice to draw, but the silhouette was never built"


class TestFigureReuse:
    """``render_locator`` must be able to draw into a caller-owned figure.

    Building a ``Figure`` per render — with its Agg canvas and renderer — cost
    3.1 MB of unreturned RSS per call in the container, more than the atlas read
    it wraps, and the overlay re-renders on every pose edit.
    """

    def test_passing_a_figure_draws_into_it(self, fake_brainglobe):
        first = _render()
        second = render_locator(
            "fake",
            tip_atlas=(100.0, 100.0, 75.0),
            pitch_deg=0.0,
            yaw_deg=0.0,
            shank_orientation_deg=0.0,
            shank_positions={0: 0.0, 1: 50.0},
            y_range=(0.0, 80.0),
            fig=first,
        )
        assert second is first, "render_locator built a new figure instead of reusing"

    def test_reuse_does_not_accumulate_axes(self, fake_brainglobe):
        """Without a clear, every render would stack another set of axes on."""
        fig = _render()
        n_axes = len(fig.axes)
        for _ in range(5):
            fig = render_locator(
                "fake",
                tip_atlas=(100.0, 100.0, 75.0),
                pitch_deg=0.0,
                yaw_deg=0.0,
                shank_orientation_deg=0.0,
                shank_positions={0: 0.0, 1: 50.0},
                y_range=(0.0, 80.0),
                fig=fig,
            )
        assert len(fig.axes) == n_axes, (
            f"{len(fig.axes)} axes after 6 renders, expected {n_axes} — the "
            "figure is not being cleared between renders"
        )

    def test_omitting_the_figure_still_builds_a_fresh_one(self, fake_brainglobe):
        assert _render() is not _render()
