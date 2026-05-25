"""Tests for the locator schematic (brain silhouette + probe trajectory).

As elsewhere in the anatomy tests, we patch brainglobe with a tiny fake
atlas so nothing is downloaded.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pixelmap.anatomy import atlas as atlas_module
from pixelmap.anatomy import schematic as schematic_module
from pixelmap.anatomy.schematic import render_locator


class _FakeAtlas:
    """A small fake atlas: an inside blob (region 1) ringed by outside (0)."""

    def __init__(self, name: str):
        self.name = name
        self.resolution = (25.0, 25.0, 25.0)  # (AP, DV, ML) µm/voxel
        ann = np.zeros((8, 6, 10), dtype=np.int32)  # (AP, DV, ML)
        ann[1:7, 1:5, 1:9] = 1
        self.annotation = ann


@pytest.fixture(autouse=True)
def _reset_caches():
    atlas_module.get_atlas.cache_clear()
    schematic_module._silhouettes.cache_clear()
    yield
    atlas_module.get_atlas.cache_clear()
    schematic_module._silhouettes.cache_clear()


@pytest.fixture
def fake_brainglobe(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "brainglobe_atlasapi", SimpleNamespace(BrainGlobeAtlas=_FakeAtlas)
    )
    yield


def test_silhouettes_follow_contourf_shape_convention(fake_brainglobe):
    ap, dv, ml, sag, cor = schematic_module._silhouettes("fake")
    # contourf needs Z shaped (len(y), len(x)); y is DV in both views.
    assert sag.shape == (len(dv), len(ap))
    assert cor.shape == (len(dv), len(ml))
    # The bordering voxels are outside-brain; the interior blob is filled.
    assert sag.any() and cor.any()
    assert not sag.all() and not cor.all()


def test_render_returns_two_axis_figure_with_a_probe_per_shank(fake_brainglobe):
    fig = render_locator(
        "fake",
        tip_atlas=(100.0, 120.0, 100.0),
        pitch_deg=0.0,
        yaw_deg=0.0,
        shank_orientation_deg=0.0,
        shank_positions={0: 0.0, 1: 50.0},
        y_range=(0.0, 80.0),
    )
    assert len(fig.axes) == 2
    # Each view draws one Line2D per shank (plus the tip marker).
    for ax in fig.axes:
        assert len(ax.get_lines()) >= 2
