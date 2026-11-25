"""Tests for sampling the atlas along a shank and collapsing into bands."""

from __future__ import annotations

import itertools
from typing import ClassVar

import numpy as np
import pytest

from pixelmap.anatomy import atlas as atlas_module
from pixelmap.anatomy.transform import AtlasWarp, ProbePlacement
from pixelmap.anatomy.visualization import RegionBand, band_acronyms_at, compute_region_bands


class _LayeredAtlas:
    """Layered along the DV axis: bottom half = region 1, top half = region 2."""

    def __init__(self, name: str, **_kwargs):
        self.name = name
        self.orientation = "asr"
        self.resolution = (25.0, 25.0, 25.0)
        # 8 DV voxels = 200 µm of depth. First 4 → region 1, next 4 → region 2.
        ann = np.zeros((4, 8, 4), dtype=np.int32)
        ann[:, :4, :] = 1
        ann[:, 4:, :] = 2
        self.annotation = ann
        self.structures = {
            1: {"acronym": "DEEP", "name": "Deep layer", "rgb_triplet": [10, 20, 30]},
            2: {"acronym": "SUP", "name": "Superficial layer", "rgb_triplet": [40, 50, 60]},
        }


@pytest.fixture(autouse=True)
def _reset_cache():
    atlas_module.release_atlas_memory()
    atlas_module.get_atlas.cache_clear()
    atlas_module.canonical_annotation.cache_clear()
    atlas_module._region_info_from_id.cache_clear()
    yield
    atlas_module.release_atlas_memory()
    atlas_module.get_atlas.cache_clear()
    atlas_module.canonical_annotation.cache_clear()
    atlas_module._region_info_from_id.cache_clear()


@pytest.fixture
def fake_brainglobe(monkeypatch):
    monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _LayeredAtlas)
    yield


def test_uniform_region_collapses_to_one_band(fake_brainglobe):
    """A probe fully inside the deep layer should produce a single band."""
    # Tip at DV=50 µm = voxel index 2 = region 1 ("DEEP"). Going up the
    # shank decreases DV further into the deep half (voxels 1, 0). Beyond
    # that we exit the volume — those samples are dropped, so we expect
    # exactly one contiguous DEEP band.
    bands = compute_region_bands(
        shank_positions={0: 0.0},
        y_range=(0.0, 175.0),
        placement=ProbePlacement((0.0, 0.0, 50.0)),
        atlas_name="fake",
        step_um=25.0,
    )

    assert len(bands) == 1
    assert bands[0].acronym == "DEEP"
    assert bands[0].shank_id == 0


def test_probe_spanning_both_layers_produces_two_bands(fake_brainglobe):
    # Place the tip at DV=175 (deep layer, voxel 7); going up reaches the
    # superficial half at DV=75 (voxel 3 → region 1, voxel 2 → region 1, etc.).
    # Wait: layered atlas has DV index 0..3 = region 1, 4..7 = region 2.
    # So tip at DV=175 (index 7) → region 2 ("SUP"). Going up to DV=0 → index 0
    # → region 1 ("DEEP"). One transition expected.
    bands = compute_region_bands(
        shank_positions={0: 0.0},
        y_range=(0.0, 175.0),
        placement=ProbePlacement((0.0, 0.0, 175.0)),
        atlas_name="fake",
        step_um=25.0,
    )
    acronyms = [b.acronym for b in bands]
    assert acronyms == ["SUP", "DEEP"], f"unexpected: {acronyms}"


def test_multi_shank_produces_bands_per_shank(fake_brainglobe):
    bands = compute_region_bands(
        shank_positions={0: 0.0, 1: 50.0, 2: 75.0},
        y_range=(0.0, 50.0),
        placement=ProbePlacement((0.0, 0.0, 50.0)),
        atlas_name="fake",
        step_um=25.0,
    )
    seen_shanks = {b.shank_id for b in bands}
    assert seen_shanks == {0, 1, 2}


def test_band_y_ranges_are_monotonic(fake_brainglobe):
    bands = compute_region_bands(
        shank_positions={0: 0.0},
        y_range=(0.0, 175.0),
        placement=ProbePlacement((0.0, 0.0, 175.0)),
        atlas_name="fake",
        step_um=25.0,
    )
    for band in bands:
        assert band.y_max > band.y_min

    # Consecutive bands on the same shank should not overlap.
    same_shank = [b for b in bands if b.shank_id == 0]
    for a, b in itertools.pairwise(same_shank):
        assert b.y_min >= a.y_max - 1e-6  # allow tiny float slack


def test_empty_y_range_returns_no_bands(fake_brainglobe):
    bands = compute_region_bands(
        shank_positions={0: 0.0},
        y_range=(100.0, 100.0),
        placement=ProbePlacement((0.0, 0.0, 0.0)),
        atlas_name="fake",
    )
    assert bands == []


def test_bands_are_read_through_the_atlas_warp(fake_brainglobe):
    """Regression: a squish / tilt reshaped the drawn brain but not the bands.

    The probe lives in the stereotaxic frame (the warped atlas). Squashing the
    brain to half its DV height about DV 0 leaves a tip at 87.5 µm sitting in
    the atlas's voxel at 175 µm, and every 12.5 µm up the shank crosses 25 µm
    of atlas: the SUP → DEEP boundary (atlas DV 87.5 µm) is 43.75 µm up.
    """
    common = {"shank_positions": {0: 0.0}, "y_range": (0.0, 75.0),
              "atlas_name": "fake", "step_um": 12.5}
    tip = (0.0, 0.0, 87.5)
    squashed = AtlasWarp((0.0, 0.0, 0.0), dv_squish=0.5)

    bands = compute_region_bands(placement=ProbePlacement(tip, warp=squashed), **common)

    assert [b.acronym for b in bands] == ["SUP", "DEEP"]
    assert bands[0].y_max == pytest.approx(43.75)
    assert bands[1].y_min == pytest.approx(43.75)
    # Without the warp the same tip is already at the top of SUP.
    plain = compute_region_bands(placement=ProbePlacement(tip), **common)
    assert plain[0].y_max == pytest.approx(6.25)


def test_an_identity_warp_changes_nothing(fake_brainglobe):
    common = {"shank_positions": {0: 0.0, 1: 50.0}, "y_range": (0.0, 175.0), "atlas_name": "fake"}
    tip = (0.0, 0.0, 175.0)
    plain = compute_region_bands(placement=ProbePlacement(tip), **common)
    neutral = compute_region_bands(
        placement=ProbePlacement(tip, warp=AtlasWarp((50.0, 50.0, 100.0))), **common)
    assert neutral == plain


def _band(shank_id, y_min, y_max, acronym):
    return RegionBand(shank_id=shank_id, probe_xp=0.0, y_min=y_min, y_max=y_max,
                      atlas_id=1, acronym=acronym, name=acronym, rgb=(0, 0, 0))


class TestBandAcronymsAt:
    """An electrode's region is the band drawn behind it."""

    BANDS: ClassVar[list] = [
        _band(0, -12.5, 87.5, "A"), _band(0, 87.5, 187.5, "B"),
        _band(0, 300.0, 400.0, "C"),          # a gap (no brain) between B and C
        _band(1, -12.5, 500.0, "D"),
    ]

    def test_each_electrode_gets_the_band_it_sits_in(self):
        out = band_acronyms_at(self.BANDS, [0, 0, 0, 1, 1], [0.0, 60.0, 150.0, 0.0, 495.0])
        assert out == ["A", "A", "B", "D", "D"]

    def test_a_boundary_belongs_to_the_band_above_it(self):
        assert band_acronyms_at(self.BANDS, [0, 0], [87.5, 300.0]) == ["B", "C"]

    def test_outside_every_band_is_none(self):
        out = band_acronyms_at(self.BANDS, [0, 0, 0, 0, 1], [-50.0, 200.0, 400.0, 1e4, 500.0])
        assert out == [None] * 5

    def test_a_shank_without_bands_is_none(self):
        assert band_acronyms_at(self.BANDS, [2, 3], [10.0, 10.0]) == [None, None]
        assert band_acronyms_at([], [0, 1], [10.0, 10.0]) == [None, None]

    def test_band_order_does_not_matter(self):
        ys = [0.0, 100.0, 350.0]
        assert band_acronyms_at(self.BANDS[::-1], [0, 0, 0], ys) == ["A", "B", "C"]

    def test_it_agrees_with_the_bands_it_is_read_from(self, fake_brainglobe):
        """End to end on the layered atlas: SUP below, DEEP above, then nothing."""
        bands = compute_region_bands(
            shank_positions={0: 0.0}, y_range=(0.0, 300.0),
            placement=ProbePlacement((0.0, 0.0, 175.0)), atlas_name="fake", step_um=25.0)
        ys = np.arange(0.0, 301.0, 15.0)
        out = band_acronyms_at(bands, np.zeros(len(ys)), ys)
        for y, acronym in zip(ys, out):
            covering = [b.acronym for b in bands if b.y_min <= y < b.y_max]
            assert [acronym] == (covering or [None])
        assert {"SUP", "DEEP", None} == set(out)

