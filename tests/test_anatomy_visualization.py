"""Tests for sampling the atlas along a shank and collapsing into bands."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pixelmap.anatomy import atlas as atlas_module
from pixelmap.anatomy.visualization import compute_region_bands


class _LayeredAtlas:
    """Layered along the DV axis: bottom half = region 1, top half = region 2."""

    def __init__(self, name: str):
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
    atlas_module.get_atlas.cache_clear()
    atlas_module.canonical_annotation.cache_clear()
    atlas_module._region_info_from_id.cache_clear()
    yield
    atlas_module.get_atlas.cache_clear()
    atlas_module.canonical_annotation.cache_clear()
    atlas_module._region_info_from_id.cache_clear()


@pytest.fixture
def fake_brainglobe(monkeypatch):
    fake_module = SimpleNamespace(BrainGlobeAtlas=_LayeredAtlas)
    monkeypatch.setitem(sys.modules, "brainglobe_atlasapi", fake_module)
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
        tip_atlas=(0.0, 0.0, 50.0),
        pitch_deg=0.0,
        yaw_deg=0.0,
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
        tip_atlas=(0.0, 0.0, 175.0),
        pitch_deg=0.0,
        yaw_deg=0.0,
        atlas_name="fake",
        step_um=25.0,
    )
    acronyms = [b.acronym for b in bands]
    assert acronyms == ["SUP", "DEEP"], f"unexpected: {acronyms}"


def test_multi_shank_produces_bands_per_shank(fake_brainglobe):
    bands = compute_region_bands(
        shank_positions={0: 0.0, 1: 50.0, 2: 75.0},
        y_range=(0.0, 50.0),
        tip_atlas=(0.0, 0.0, 50.0),
        pitch_deg=0.0,
        yaw_deg=0.0,
        atlas_name="fake",
        step_um=25.0,
    )
    seen_shanks = {b.shank_id for b in bands}
    assert seen_shanks == {0, 1, 2}


def test_band_y_ranges_are_monotonic(fake_brainglobe):
    bands = compute_region_bands(
        shank_positions={0: 0.0},
        y_range=(0.0, 175.0),
        tip_atlas=(0.0, 0.0, 175.0),
        pitch_deg=0.0,
        yaw_deg=0.0,
        atlas_name="fake",
        step_um=25.0,
    )
    for band in bands:
        assert band.y_max > band.y_min

    # Consecutive bands on the same shank should not overlap.
    same_shank = [b for b in bands if b.shank_id == 0]
    for a, b in zip(same_shank, same_shank[1:]):
        assert b.y_min >= a.y_max - 1e-6  # allow tiny float slack


def test_empty_y_range_returns_no_bands(fake_brainglobe):
    bands = compute_region_bands(
        shank_positions={0: 0.0},
        y_range=(100.0, 100.0),
        tip_atlas=(0.0, 0.0, 0.0),
        pitch_deg=0.0,
        yaw_deg=0.0,
        atlas_name="fake",
    )
    assert bands == []
