"""Tests for the brainglobe atlas wrapper.

We don't want the test suite to download a real atlas — that's tens of MB
and CI-hostile. We patch atlas_module.BrainGlobeAtlas with a tiny fake atlas
that exercises the indexing logic without touching the network.
"""

from __future__ import annotations

import numpy as np
import pytest

from pixelmap.anatomy.transform import probe_axis

from pixelmap.anatomy import atlas as atlas_module
from pixelmap.anatomy import regions as regions_module


class _FakeAtlas:
    """A 3-voxel-cube fake atlas: half is region 1, half is region 2."""

    def __init__(self, name: str, **_kwargs):
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
    """Patch BrainGlobeAtlas in the atlas module with a tiny fake."""
    monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _FakeAtlas)
    yield


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

    def test_zero_label_returns_none(self, monkeypatch):
        # Override the fake atlas to have all zeros (outside-brain everywhere).
        class Zeros(_FakeAtlas):
            def __init__(self, name, **kwargs):
                super().__init__(name, **kwargs)
                self.annotation = np.zeros((4, 4, 4), dtype=np.int32)

        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", Zeros)
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


def _atlas_cls(orientation, annotation, structures=None, resolution=(25.0, 25.0, 25.0)):
    """Build a fake BrainGlobeAtlas class with a given orientation/annotation.

    ``resolution`` is in the atlas's *native* axis order, matching brainglobe.
    Pass an anisotropic value to catch axis-transposition bugs, which are
    invisible when every voxel dimension is the same.
    """
    structures = structures or {
        1: {"acronym": "ONE", "name": "One", "rgb_triplet": [1, 2, 3]},
        2: {"acronym": "TWO", "name": "Two", "rgb_triplet": [4, 5, 6]},
    }

    class _A:
        def __init__(self, name, **_kwargs):
            self.name = name
            self.orientation = orientation
            self.resolution = resolution
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
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _atlas_cls("sla", native))
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
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _atlas_cls("asr", ann, structs))
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
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _atlas_cls("asr", ann, structs))
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
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _atlas_cls("psr", native))
        atlas_module.get_atlas.cache_clear()
        atlas_module.canonical_annotation.cache_clear()
        anterior = atlas_module.lookup_regions("x", np.array([[0.0, 0.0, 0.0]]))
        posterior = atlas_module.lookup_regions("x", np.array([[75.0, 0.0, 0.0]]))
        assert anterior[0].acronym == "ONE"
        assert posterior[0].acronym == "TWO"


class TestVoxelIndexHelper:
    """The shared (AP, ML, DV) µm → (ap, dv, ml) index mapping."""

    def test_matches_the_documented_mapping(self):
        # Anisotropic on purpose: with equal voxel dimensions a swapped DV/ML
        # axis produces identical numbers and the bug goes unnoticed.
        voxel_size = np.array([25.0, 10.0, 50.0])   # annotation order (AP, DV, ML)
        coords = np.array([                          # coordinate order (AP, ML, DV)
            [100.0, 500.0, 40.0],
            [75.0, 150.0, 25.0],
        ])
        out = atlas_module._voxel_indices(coords, voxel_size)
        expected = np.array([
            [round(100.0 / 25.0), round(40.0 / 10.0), round(500.0 / 50.0)],
            [round(75.0 / 25.0), round(25.0 / 10.0), round(150.0 / 50.0)],
        ])
        np.testing.assert_array_equal(out, expected)

    def test_rounds_to_nearest_voxel(self):
        out = atlas_module._voxel_indices(np.array([[37.0, 0.0, 0.0]]), np.array([25.0] * 3))
        assert out[0, 0] == 1


def _brain_block_atlas(orientation="asr", resolution=(25.0, 25.0, 25.0)):
    """Fake atlas with a labelled block floating inside empty space.

    Unlike the module-level ``_FakeAtlas`` (solid, no margin) this one has a
    genuine dorsal surface for the ray-cast to find.
    """
    ann = np.zeros((8, 6, 10), dtype=np.int32)      # (AP, DV, ML)
    ann[1:7, 1:5, 1:9] = 1
    return _atlas_cls(orientation, ann, resolution=resolution)


class TestSurfacePoint:
    """Locating where a trajectory enters the brain."""

    def test_vertical_trajectory_hits_the_dorsal_surface(self, monkeypatch):
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _brain_block_atlas())
        out = atlas_module.surface_point_um("x", (100.0, 100.0, 100.0))
        # First labelled DV voxel is index 1 → 25 µm; AP/ML are unchanged
        # because an untilted trajectory only moves in DV.
        assert out is not None
        assert out[0] == pytest.approx(100.0)
        assert out[1] == pytest.approx(100.0)
        assert out[2] == pytest.approx(25.0, abs=25.0)
        assert out[2] < 100.0, "entry must be dorsal of the starting point"

    def test_result_depends_only_on_the_line(self, monkeypatch):
        # Casting from the tip and from a point far up the shank — outside
        # the volume entirely — must agree. This is what lets the GUI's snap
        # button behave identically in tip-entry and insertion-entry mode.
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _brain_block_atlas())
        tip = np.array([100.0, 100.0, 100.0])
        for pitch, yaw in [(0, 0), (30, 25), (-20, 10)]:
            from_tip = atlas_module.surface_point_um(
                "x", tuple(tip), pitch_deg=pitch, yaw_deg=yaw)
            far = tuple(tip + 4000.0 * probe_axis(pitch, yaw))
            from_far = atlas_module.surface_point_um(
                "x", far, pitch_deg=pitch, yaw_deg=yaw)
            assert from_tip is not None and from_far is not None
            np.testing.assert_allclose(from_tip, from_far, atol=1e-6)

    def test_returns_none_when_the_trajectory_misses_the_brain(self, monkeypatch):
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _brain_block_atlas())
        assert atlas_module.surface_point_um("x", (100.0, 99999.0, 100.0)) is None

    def test_returns_none_for_unlabelled_volume(self, monkeypatch):
        empty = _atlas_cls("asr", np.zeros((8, 6, 10), dtype=np.int32))
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", empty)
        assert atlas_module.surface_point_um("x", (100.0, 100.0, 100.0)) is None

    def test_tilted_trajectory_shifts_ap(self, monkeypatch):
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _brain_block_atlas())
        straight = atlas_module.surface_point_um("x", (100.0, 100.0, 100.0))
        tilted = atlas_module.surface_point_um(
            "x", (100.0, 100.0, 100.0), pitch_deg=45)
        assert straight is not None and tilted is not None
        # Positive pitch puts the top of the probe at larger AP, so entering
        # from above means the entry point is posterior of the anchor.
        assert tilted[0] > straight[0]

    def test_works_for_non_asr_orientation(self, monkeypatch):
        # Same brain, described in a flipped native frame: the entry point
        # must come out the same in canonical coordinates.
        ann_asr = np.zeros((8, 6, 10), dtype=np.int32)
        ann_asr[1:7, 1:5, 1:9] = 1
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _atlas_cls("asr", ann_asr))
        expected = atlas_module.surface_point_um("x", (100.0, 100.0, 100.0))

        atlas_module.get_atlas.cache_clear()
        atlas_module.canonical_annotation.cache_clear()
        # "psr" reverses AP, so flip the volume to describe the same brain.
        monkeypatch.setattr(
            atlas_module, "BrainGlobeAtlas",
            _atlas_cls("psr", np.flip(ann_asr, axis=0).copy()))
        got = atlas_module.surface_point_um("y", (100.0, 100.0, 100.0))
        assert expected is not None and got is not None
        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_thin_dorsal_layer_is_not_stepped_over(self, monkeypatch):
        # One 25 µm-thick DV layer inside a coarse 100 µm AP grid. A march
        # step taken from the wrong axis (or from max(resolution)) walks
        # straight past it and reports the layer below instead.
        ann = np.zeros((8, 6, 10), dtype=np.int32)
        ann[1:7, 1, 1:9] = 1          # thin dorsal sheet
        ann[1:7, 3:5, 1:9] = 2        # bulk, well below it
        structs = {1: {"acronym": "SKIN", "name": "s", "rgb_triplet": [1, 1, 1]},
                   2: {"acronym": "BULK", "name": "b", "rgb_triplet": [2, 2, 2]}}
        monkeypatch.setattr(
            atlas_module, "BrainGlobeAtlas",
            _atlas_cls("asr", ann, structs, resolution=(100.0, 25.0, 25.0)))
        out = atlas_module.surface_point_um("x", (300.0, 100.0, 200.0))
        assert out is not None
        region = atlas_module.lookup_regions("x", np.array([list(out)]))[0]
        assert region is not None and region.acronym == "SKIN"

    def test_rejects_non_positive_step(self, monkeypatch):
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _brain_block_atlas())
        with pytest.raises(ValueError):
            atlas_module.surface_point_um("x", (100.0, 100.0, 100.0), step_um=0.0)
