"""Tests for the locator schematic (brain slice + colored regions + probe).

As elsewhere in the anatomy tests, we patch BrainGlobeAtlas with a tiny fake
atlas so nothing is downloaded.
"""

from __future__ import annotations

import numpy as np
import pytest

from pixelmap.anatomy import atlas as atlas_module
from pixelmap.anatomy import schematic as schematic_module
from pixelmap.anatomy.schematic import render_locator


class _FakeAtlas:
    """Fake atlas: left half (ML 1-4) region 1, right half (ML 5-8) region 2."""

    def __init__(self, name: str):
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
    atlas_module.get_atlas.cache_clear()
    atlas_module.canonical_annotation.cache_clear()
    schematic_module._atlas_data.cache_clear()
    yield
    atlas_module.get_atlas.cache_clear()
    atlas_module.canonical_annotation.cache_clear()
    schematic_module._atlas_data.cache_clear()


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
