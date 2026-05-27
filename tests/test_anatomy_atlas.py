"""Tests for the brainglobe atlas wrapper.

We don't want the test suite to download a real atlas — that's tens of MB
and CI-hostile. We patch the brainglobe entry points with a tiny fake atlas
that exercises the indexing logic.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from pixelmap.anatomy import atlas as atlas_module
from pixelmap.anatomy import regions as regions_module


class _FakeAtlas:
    """A 3-voxel-cube fake atlas: half is region 1, half is region 2."""

    def __init__(self, name: str):
        self.name = name
        self.orientation = "asr"
        self.resolution = (25.0, 25.0, 25.0)  # µm per voxel
        # 4×4×4 volume; left ML half = region 1, right half = region 2; 0 = outside
        ann = np.zeros((4, 4, 4), dtype=np.int32)
        ann[:, :, :2] = 1
        ann[:, :, 2:] = 2
        self.annotation = ann
        self.structures = {
            1: {"acronym": "LEFT", "name": "Left hemisphere", "rgb_triplet": [200, 0, 0]},
            2: {"acronym": "RIGHT", "name": "Right hemisphere", "rgb_triplet": [0, 200, 0]},
        }


@pytest.fixture(autouse=True)
def _reset_atlas_cache():
    """Make sure no real atlas leaks across tests via the lru_cache."""
    atlas_module.get_atlas.cache_clear()
    atlas_module.canonical_annotation.cache_clear()
    atlas_module.derive_origin_from_ac.cache_clear()
    atlas_module._region_info_from_id.cache_clear()
    yield
    atlas_module.get_atlas.cache_clear()
    atlas_module.canonical_annotation.cache_clear()
    atlas_module.derive_origin_from_ac.cache_clear()
    atlas_module._region_info_from_id.cache_clear()


@pytest.fixture
def fake_brainglobe(monkeypatch):
    """Inject a fake brainglobe_atlasapi module so _import_brainglobe succeeds."""
    fake_module = SimpleNamespace(BrainGlobeAtlas=_FakeAtlas)
    monkeypatch.setitem(sys.modules, "brainglobe_atlasapi", fake_module)
    yield


class TestAvailability:
    def test_is_available_true_when_brainglobe_present(self, fake_brainglobe):
        assert atlas_module.is_available() is True

    def test_is_available_false_when_brainglobe_missing(self, monkeypatch):
        # Hide any installed copy so the import fails reliably.
        monkeypatch.setitem(sys.modules, "brainglobe_atlasapi", None)
        assert atlas_module.is_available() is False

    def test_lookup_raises_install_hint_when_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "brainglobe_atlasapi", None)
        with pytest.raises(ImportError, match="pixelmap\\[anatomy\\]"):
            atlas_module.get_atlas("allen_mouse_25um")


class TestLookup:
    def test_voxel_indexing_resolves_left_vs_right(self, fake_brainglobe):
        # ML=10 (voxel 0, in left half) vs ML=60 (voxel 2, in right half)
        coords = np.array([
            [0.0,  10.0, 0.0],   # left
            [0.0,  60.0, 0.0],   # right
        ])
        out = atlas_module.lookup_regions("fake", coords)
        assert out[0].acronym == "LEFT"
        assert out[1].acronym == "RIGHT"
        assert out[0].rgb == (200, 0, 0)

    def test_out_of_bounds_returns_none(self, fake_brainglobe):
        coords = np.array([
            [-100.0, 0.0, 0.0],
            [9999.0, 9999.0, 9999.0],
        ])
        out = atlas_module.lookup_regions("fake", coords)
        assert out == [None, None]

    def test_zero_label_returns_none(self, fake_brainglobe, monkeypatch):
        # Override the fake atlas to have all zeros (outside-brain everywhere).
        class Zeros(_FakeAtlas):
            def __init__(self, name):
                super().__init__(name)
                self.annotation = np.zeros((4, 4, 4), dtype=np.int32)

        monkeypatch.setitem(
            sys.modules,
            "brainglobe_atlasapi",
            SimpleNamespace(BrainGlobeAtlas=Zeros),
        )
        atlas_module.get_atlas.cache_clear()
        atlas_module.canonical_annotation.cache_clear()

        coords = np.array([[0.0, 0.0, 0.0]])
        assert atlas_module.lookup_regions("zeros", coords) == [None]


class TestRegionsForPositions:
    def test_end_to_end_lookup_with_pose(self, fake_brainglobe):
        # Place the tip at the very corner of the volume; both electrodes are
        # at probe (0, 0) and (xp=50, 0). With default pose (vertical, +xp=+ML),
        # the second electrode is 50 µm to the right — voxel 2 → "RIGHT".
        electrode_xy = np.array([[0.0, 0.0], [50.0, 0.0]])
        regions = regions_module.regions_for_positions(
            electrode_xy,
            tip_atlas=(0.0, 0.0, 0.0),
            atlas_name="fake",
        )
        assert regions[0].acronym == "LEFT"
        assert regions[1].acronym == "RIGHT"


def _atlas_cls(orientation, annotation, structures=None):
    """Build a fake BrainGlobeAtlas class with a given orientation/annotation."""
    structures = structures or {
        1: {"acronym": "ONE", "name": "One", "rgb_triplet": [1, 2, 3]},
        2: {"acronym": "TWO", "name": "Two", "rgb_triplet": [4, 5, 6]},
    }

    class _A:
        def __init__(self, name):
            self.name = name
            self.orientation = orientation
            self.resolution = (25.0, 25.0, 25.0)
            self.annotation = annotation
            self.structures = structures

    return _A


class TestOrientation:
    def test_anatomical_axes_asr_is_identity(self):
        atlas = _atlas_cls("asr", np.zeros((4, 5, 6), np.int32))("x")
        axes = atlas_module.anatomical_axes(atlas)
        assert (axes["AP"].array_axis, axes["AP"].flip) == (0, False)
        assert (axes["DV"].array_axis, axes["DV"].flip) == (1, False)
        assert (axes["ML"].array_axis, axes["ML"].flip) == (2, False)

    def test_anatomical_axes_permuted_and_flipped(self):
        # "sla": axis0=DV(s), axis1=ML(l → flipped vs canonical right), axis2=AP(a)
        axes = atlas_module.anatomical_axes(_atlas_cls("sla", np.zeros((4, 5, 6), np.int32))("x"))
        assert (axes["DV"].array_axis, axes["DV"].flip) == (0, False)
        assert (axes["ML"].array_axis, axes["ML"].flip) == (1, True)
        assert (axes["AP"].array_axis, axes["AP"].flip) == (2, False)
        assert axes["DV"].n == 4 and axes["ML"].n == 5 and axes["AP"].n == 6

    def test_canonical_annotation_reorients_to_asr(self, monkeypatch):
        # Native "sla" volume is indexed (DV, ML, AP); canonical must be (AP, DV, ML).
        native = np.arange(2 * 3 * 4, dtype=np.int32).reshape(2, 3, 4)
        monkeypatch.setitem(sys.modules, "brainglobe_atlasapi",
                            SimpleNamespace(BrainGlobeAtlas=_atlas_cls("sla", native)))
        atlas_module.get_atlas.cache_clear()
        atlas_module.canonical_annotation.cache_clear()
        arr, _ = atlas_module.canonical_annotation("x")
        expected = np.flip(np.transpose(native, (2, 0, 1)), axis=2)  # AP from axis2, flip ML
        assert arr.shape == (4, 2, 3)
        np.testing.assert_array_equal(arr, expected)

    def test_derive_origin_from_ac_finds_midline_crossing(self, monkeypatch):
        # AC at AP voxel 2, DV voxel 2, ML voxels 3-5 (midline 4) in an asr volume.
        ann = np.zeros((6, 4, 8), dtype=np.int32)
        ann[2, 2, 3:6] = 5
        structs = {5: {"id": 5, "acronym": "ac", "name": "anterior commissure",
                       "rgb_triplet": [1, 2, 3]}}
        monkeypatch.setitem(sys.modules, "brainglobe_atlasapi",
                            SimpleNamespace(BrainGlobeAtlas=_atlas_cls("asr", ann, structs)))
        atlas_module.get_atlas.cache_clear()
        atlas_module.canonical_annotation.cache_clear()
        atlas_module.derive_origin_from_ac.cache_clear()
        # (AP, ML, DV) µm = (2*25, 4*25, 2*25)
        assert atlas_module.derive_origin_from_ac("x") == (50.0, 100.0, 50.0)

    def test_derive_origin_from_ac_none_without_ac(self, monkeypatch):
        ann = np.zeros((4, 4, 4), dtype=np.int32)
        ann[1, 1, 1] = 9
        structs = {9: {"id": 9, "acronym": "x", "name": "some nucleus",
                       "rgb_triplet": [0, 0, 0]}}
        monkeypatch.setitem(sys.modules, "brainglobe_atlasapi",
                            SimpleNamespace(BrainGlobeAtlas=_atlas_cls("asr", ann, structs)))
        atlas_module.get_atlas.cache_clear()
        atlas_module.canonical_annotation.cache_clear()
        atlas_module.derive_origin_from_ac.cache_clear()
        assert atlas_module.derive_origin_from_ac("y") is None

    def test_lookup_resolves_through_ap_flip(self, monkeypatch):
        # "psr" reverses AP: a marker at the native posterior pole (index 0) must
        # be read at large canonical AP, not at AP=0 (the anterior pole).
        native = np.zeros((4, 2, 2), dtype=np.int32)
        native[0, :, :] = 2     # native index 0 = posterior
        native[1:, :, :] = 1
        monkeypatch.setitem(sys.modules, "brainglobe_atlasapi",
                            SimpleNamespace(BrainGlobeAtlas=_atlas_cls("psr", native)))
        atlas_module.get_atlas.cache_clear()
        atlas_module.canonical_annotation.cache_clear()
        anterior = atlas_module.lookup_regions("x", np.array([[0.0, 0.0, 0.0]]))
        posterior = atlas_module.lookup_regions("x", np.array([[75.0, 0.0, 0.0]]))
        assert anterior[0].acronym == "ONE"
        assert posterior[0].acronym == "TWO"
