"""Tests for the tip-depth-below-brain-surface readout.

Like the other anatomy tests, these run against a tiny fake atlas rather than
downloading a real one, so the geometry is known exactly and CI stays offline.
"""

from __future__ import annotations

import numpy as np
import pytest

from pixelmap.anatomy import atlas as atlas_module
from pixelmap.anatomy.regions import tip_depth_below_surface_um
from pixelmap.anatomy.transform import probe_axis_up

ATLAS = "fake_surface"
RES_UM = 25.0


def _make_atlas_cls(annotation, orientation="asr", resolution=RES_UM):
    """Fake BrainGlobeAtlas class serving a given annotation volume."""

    class _A:
        def __init__(self, name, **_kwargs):
            self.name = name
            self.orientation = orientation
            self.resolution = (resolution, resolution, resolution)
            self.annotation = annotation
            self.structures = {
                1: {"acronym": "BRAIN", "name": "Brain", "rgb_triplet": [1, 2, 3]},
            }

    return _A


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
def slab_atlas(monkeypatch):
    """A brain that is a DV slab: voxels 4..15 are brain, 0..3 are outside.

    Canonical indexing is (AP, DV, ML) with 25 µm voxels, so the surface sits
    between DV voxel 3 (outside) and DV voxel 4 (brain) — i.e. at DV ≈ 87.5 µm,
    the midpoint between voxel centres 75 and 100.
    """
    ann = np.zeros((20, 20, 20), dtype=np.int32)
    ann[:, 4:16, :] = 1
    monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _make_atlas_cls(ann))
    return ann


class TestVerticalProbe:
    def test_depth_is_distance_from_tip_up_to_the_surface(self, slab_atlas):
        # Tip at DV = 300 µm (voxel 12, inside). Surface at DV ≈ 87.5 µm.
        depth = tip_depth_below_surface_um((250.0, 250.0, 300.0), atlas_name=ATLAS)
        assert depth == pytest.approx(300.0 - 87.5, abs=RES_UM / 2)

    def test_depth_grows_one_for_one_with_tip_depth(self, slab_atlas):
        shallow = tip_depth_below_surface_um((250.0, 250.0, 200.0), atlas_name=ATLAS)
        deep = tip_depth_below_surface_um((250.0, 250.0, 350.0), atlas_name=ATLAS)
        assert deep - shallow == pytest.approx(150.0, abs=1e-6)

    def test_tip_just_below_the_surface_gives_a_small_positive_depth(self, slab_atlas):
        depth = tip_depth_below_surface_um((250.0, 250.0, 100.0), atlas_name=ATLAS)
        assert 0.0 < depth <= RES_UM


class TestUndefinedCases:
    def test_tip_above_the_brain_returns_none(self, slab_atlas):
        assert tip_depth_below_surface_um((250.0, 250.0, 25.0), atlas_name=ATLAS) is None

    def test_tip_below_the_brain_returns_none(self, slab_atlas):
        # DV = 450 µm is past the bottom of the slab (voxel 18).
        assert tip_depth_below_surface_um((250.0, 250.0, 450.0), atlas_name=ATLAS) is None

    def test_tip_outside_the_volume_returns_none(self, slab_atlas):
        assert tip_depth_below_surface_um((-500.0, -500.0, -500.0), atlas_name=ATLAS) is None

    def test_edge_of_the_volume_counts_as_the_surface(self, monkeypatch):
        """An all-brain volume has its surface at the volume boundary.

        Region lookups treat anything outside the array as outside the brain,
        so the depth readout must agree rather than reporting nothing.
        """
        ann = np.ones((20, 20, 20), dtype=np.int32)
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _make_atlas_cls(ann))
        # DV voxel 0 is centred at DV = 0, so a tip at DV = 250 is 250 µm below
        # it (plus half a sampling step for the boundary midpoint).
        depth = tip_depth_below_surface_um((250.0, 250.0, 250.0), atlas_name=ATLAS)
        assert depth == pytest.approx(250.0, abs=RES_UM)

    def test_search_range_is_respected(self, slab_atlas):
        # The surface is ~212 µm above a tip at DV=300; a 50 µm search can't reach it.
        assert tip_depth_below_surface_um(
            (250.0, 250.0, 300.0), atlas_name=ATLAS, max_search_um=50.0
        ) is None

    def test_non_positive_step_is_rejected(self, slab_atlas):
        with pytest.raises(ValueError):
            tip_depth_below_surface_um((250.0, 250.0, 300.0), atlas_name=ATLAS, step_um=-1.0)


class TestTiltedProbe:
    @pytest.mark.parametrize("pitch_deg", [0.0, 15.0, -30.0])
    def test_pitch_lengthens_the_path_by_one_over_cos(self, slab_atlas, pitch_deg):
        """A tilted shank travels further to cover the same vertical distance."""
        tip = (250.0, 250.0, 300.0)
        vertical = tip_depth_below_surface_um(tip, atlas_name=ATLAS)
        tilted = tip_depth_below_surface_um(tip, pitch_deg=pitch_deg, atlas_name=ATLAS)
        expected = vertical / np.cos(np.deg2rad(pitch_deg))
        # The slab surface is flat, so the geometry is exact up to sampling.
        assert tilted == pytest.approx(expected, abs=RES_UM)

    def test_yaw_lengthens_the_path_by_one_over_cos(self, slab_atlas):
        tip = (250.0, 250.0, 300.0)
        vertical = tip_depth_below_surface_um(tip, atlas_name=ATLAS)
        tilted = tip_depth_below_surface_um(tip, yaw_deg=20.0, atlas_name=ATLAS)
        assert tilted == pytest.approx(vertical / np.cos(np.deg2rad(20.0)), abs=RES_UM)


class TestOutermostCrossing:
    def test_an_internal_gap_is_not_mistaken_for_the_surface(self, monkeypatch):
        """A ventricle between the tip and the surface must be walked through."""
        ann = np.zeros((20, 20, 20), dtype=np.int32)
        ann[:, 4:16, :] = 1
        ann[:, 8:10, :] = 0  # a 50 µm unannotated gap mid-way up
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _make_atlas_cls(ann))

        depth = tip_depth_below_surface_um((250.0, 250.0, 300.0), atlas_name=ATLAS)
        # Still measured from the true (outermost) surface at DV ≈ 87.5 µm,
        # not from the bottom of the gap at DV ≈ 250 µm.
        assert depth == pytest.approx(300.0 - 87.5, abs=RES_UM / 2)


class TestProbeAxisUp:
    def test_untilted_axis_points_straight_up(self):
        np.testing.assert_allclose(probe_axis_up(), [0.0, 0.0, -1.0], atol=1e-12)

    def test_axis_is_a_unit_vector_under_any_tilt(self):
        for pitch, yaw in [(0, 0), (10, 0), (0, 25), (-40, 33)]:
            assert np.linalg.norm(probe_axis_up(pitch, yaw)) == pytest.approx(1.0)

    def test_pitch_tilts_the_axis_in_the_ap_plane_only(self):
        ap, ml, dv = probe_axis_up(pitch_deg=20.0)
        assert ap == pytest.approx(np.sin(np.deg2rad(20.0)))  # top swings toward +AP
        assert ml == pytest.approx(0.0)
        assert dv == pytest.approx(-np.cos(np.deg2rad(20.0)))

    def test_yaw_tilts_the_axis_in_the_ml_plane_only(self):
        ap, ml, dv = probe_axis_up(yaw_deg=20.0)
        assert ap == pytest.approx(0.0)
        assert ml == pytest.approx(np.sin(np.deg2rad(20.0)))  # top swings toward +ML
        assert dv == pytest.approx(-np.cos(np.deg2rad(20.0)))

    def test_pitch_and_yaw_signs_are_antisymmetric(self):
        np.testing.assert_allclose(
            probe_axis_up(-15.0, -25.0) * np.array([-1.0, -1.0, 1.0]),
            probe_axis_up(15.0, 25.0),
            atol=1e-12,
        )

    def test_axis_matches_the_shank_direction_used_for_region_lookup(self):
        """probe_axis_up must agree with how probe_to_atlas places electrodes."""
        from pixelmap.anatomy.transform import probe_to_atlas

        tip = (1000.0, 2000.0, 3000.0)
        # An electrode 500 µm up the shank from the tip.
        up_the_shank = probe_to_atlas(
            np.array([[0.0, 500.0]]), tip, pitch_deg=15.0, yaw_deg=-10.0
        )[0]
        expected = np.asarray(tip) + 500.0 * probe_axis_up(15.0, -10.0)
        np.testing.assert_allclose(up_the_shank, expected, atol=1e-9)
