"""Tests for the brainglobe atlas wrapper.

We don't want the test suite to download a real atlas — that's tens of MB
and CI-hostile. We patch atlas_module.BrainGlobeAtlas with a tiny fake atlas
that exercises the indexing logic without touching the network.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from pixelmap.anatomy import atlas as atlas_module
from pixelmap.anatomy import regions as regions_module
from pixelmap.anatomy._registry_snapshot import REGISTRY_SNAPSHOT


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


def _must_not_be_called(*args, **kwargs):
    raise AssertionError("this call must never happen on the session path")


@pytest.fixture(autouse=True)
def _reset_atlas_cache(monkeypatch, tmp_path):
    """Make sure no real atlas leaks across tests, and nothing writes to
    a real ``~/.brainglobe`` — compact caches land under a fresh tmp dir."""
    monkeypatch.setenv("PIXELMAP_ATLAS_CACHE_DIR", str(tmp_path))
    atlas_module.clear_caches()
    yield
    atlas_module.clear_caches()


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


def _atlas_cls(orientation, annotation, structures=None):
    """Build a fake BrainGlobeAtlas class with a given orientation/annotation."""
    structures = structures or {
        1: {"acronym": "ONE", "name": "One", "rgb_triplet": [1, 2, 3]},
        2: {"acronym": "TWO", "name": "Two", "rgb_triplet": [4, 5, 6]},
    }

    class _A:
        def __init__(self, name, **_kwargs):
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
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _atlas_cls("sla", native))
        atlas_module.get_atlas.cache_clear()
        atlas_module.canonical_annotation.cache_clear()
        arr, _ = atlas_module.canonical_annotation("x")
        lut = atlas_module.label_ids("x")
        expected = np.flip(np.transpose(native, (2, 0, 1)), axis=2)  # AP from axis2, flip ML
        assert arr.shape == (4, 2, 3)
        # arr holds compact indices, not raw ids — map through the LUT.
        np.testing.assert_array_equal(lut[arr], expected)

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


class TestMetadataOnlyQueries:
    """Shape/extent queries must not touch ``annotation``.

    brainglobe v3 fetches the annotation array lazily from S3, so reading it
    just to learn the volume's size would trigger the very download these
    queries let the GUI avoid.
    """

    @staticmethod
    def _metadata_atlas_cls(shape, orientation="asr"):
        class _A:
            def __init__(self, name, **_kwargs):
                self.name = name
                self.orientation = orientation
                self.resolution = (25.0, 25.0, 25.0)
                self.shape = shape
                self.structures = {}

            @property
            def annotation(self):
                raise AssertionError("annotation must not be loaded here")

        return _A

    def test_volume_center_reads_shape_not_the_array(self, monkeypatch):
        monkeypatch.setattr(
            atlas_module, "BrainGlobeAtlas", self._metadata_atlas_cls((4, 8, 16))
        )
        # asr: shape is (AP, DV, ML); result is (AP, ML, DV) µm half-extents.
        assert atlas_module.volume_center_um("x") == (50.0, 200.0, 100.0)

    def test_anatomical_axes_reads_shape_not_the_array(self, monkeypatch):
        atlas = self._metadata_atlas_cls((4, 8, 16), "sla")("x")
        axes = atlas_module.anatomical_axes(atlas)
        assert (axes["DV"].n, axes["ML"].n, axes["AP"].n) == (4, 8, 16)

    def test_falls_back_to_the_array_when_shape_is_absent(self, monkeypatch):
        # Older brainglobe objects and our other test doubles carry no `shape`.
        atlas = _atlas_cls("asr", np.zeros((3, 5, 7), np.int32))("x")
        axes = atlas_module.anatomical_axes(atlas)
        assert (axes["AP"].n, axes["DV"].n, axes["ML"].n) == (3, 5, 7)


class TestIsDownloaded:
    """``is_downloaded`` answers "is reading the annotation free?".

    Not "is the atlas present": the atlas directory appears as soon as the tiny
    manifest lands, while the annotation is still remote, so the two questions
    come apart.
    """

    @pytest.fixture
    def listed(self, monkeypatch):
        monkeypatch.setattr(atlas_module, "get_downloaded_atlases", lambda: ["x"])

    def test_absent_atlas_is_never_downloaded(self, monkeypatch):
        monkeypatch.setattr(atlas_module, "get_downloaded_atlases", list)
        assert atlas_module.is_downloaded("x") is False

    def test_manifest_without_chunks_is_not_downloaded(
        self, monkeypatch, listed, tmp_path
    ):
        monkeypatch.setattr(
            atlas_module, "get_atlas", lambda name: _v3_atlas(tmp_path, chunks=False)
        )
        assert atlas_module.is_downloaded("x") is False

    def test_cached_chunks_are_downloaded(self, monkeypatch, listed, tmp_path):
        monkeypatch.setattr(
            atlas_module, "get_atlas", lambda name: _v3_atlas(tmp_path, chunks=True)
        )
        assert atlas_module.is_downloaded("x") is True

    def test_unreadable_metadata_reports_not_downloaded(self, monkeypatch, listed):
        """A half-written manifest must send callers down the "this will cost
        you" path rather than crashing the GUI."""

        def _boom(name):
            raise RuntimeError("corrupt manifest")

        monkeypatch.setattr(atlas_module, "get_atlas", _boom)
        assert atlas_module.is_downloaded("x") is False


def _v3_atlas(root, *, chunks: bool):
    """A stand-in for a v3 atlas whose OME-Zarr chunks may or may not be local."""
    annotation_dir = root / "annotation-sets" / "some-annotation" / "1_0"
    # brainglobe names pyramid levels s0, s1, ...; the level a given atlas
    # pulls depends on its resolution, so the check accepts any of them.
    scale = annotation_dir / atlas_module.V3_ANNOTATION_NAME / "s1"
    (scale / "c" if chunks else scale).mkdir(parents=True)

    class _A:
        root_dir = root
        metadata = {
            # brainglobe stores locations with a leading "/" that it strips.
            "annotation_set": {"location": "/annotation-sets/some-annotation/1_0"}
        }

    return _A()


class TestRegistryCachePath:
    def test_points_at_the_v3_layout(self, tmp_path):
        assert atlas_module._registry_cache_path(tmp_path) == (
            tmp_path / "brainglobe-atlasapi" / "atlases" / "last_versions.conf"
        )


class TestListAtlasesNeverBlocks:
    """The regression that took the server down.

    ``list_atlases`` runs while a GUI session is being built — on the deployed
    server, on the Bokeh event loop. brainglobe's registry fetch is a
    ``requests.get`` with no timeout, so doing it inline turned an outage at
    the atlas host into a hung server for every connected user.
    """

    @pytest.fixture(autouse=True)
    def _clear(self):
        atlas_module.list_atlases.cache_clear()
        yield
        atlas_module.list_atlases.cache_clear()

    def test_prefers_the_on_disk_registry_cache(self, monkeypatch):
        monkeypatch.setattr(
            atlas_module, "_registry_from_disk", lambda: ["b_atlas", "a_atlas"]
        )
        monkeypatch.setattr(
            atlas_module, "_refresh_registry_in_background", _must_not_be_called
        )
        assert atlas_module.list_atlases() == ["b_atlas", "a_atlas"]

    def test_cold_cache_returns_the_snapshot_without_network(self, monkeypatch):
        monkeypatch.setattr(atlas_module, "_registry_from_disk", lambda: None)
        monkeypatch.setattr(
            atlas_module, "get_all_atlases_lastversions", _must_not_be_called
        )
        refreshed = []
        monkeypatch.setattr(
            atlas_module,
            "_refresh_registry_in_background",
            lambda: refreshed.append(True),
        )

        out = atlas_module.list_atlases()

        assert out == sorted(REGISTRY_SNAPSHOT)
        assert "allen_mouse_25um" in out
        assert refreshed == [True], "a cold cache should schedule a background refresh"

    def test_result_is_memoised(self, monkeypatch):
        calls = []

        def _once():
            calls.append(True)
            return ["only_atlas"]

        monkeypatch.setattr(atlas_module, "_registry_from_disk", _once)
        for _ in range(5):
            assert atlas_module.list_atlases() == ["only_atlas"]
        assert len(calls) == 1

    def test_unreadable_cache_falls_back_instead_of_raising(self, monkeypatch, tmp_path):
        """A truncated conf file must not take the dropdown down with it."""
        cache = atlas_module._registry_cache_path(tmp_path)
        cache.parent.mkdir(parents=True)
        cache.write_text("this is not a conf file")

        import brainglobe_atlasapi.config as bg_config

        monkeypatch.setattr(bg_config, "get_brainglobe_dir", lambda: tmp_path)
        assert atlas_module._registry_from_disk() is None


class TestEnsureDownloaded:
    def test_materialises_the_annotation(self, monkeypatch):
        reads = []

        class _A:
            def __init__(self, name, **_kwargs):
                self.name = name

            @property
            def annotation(self):
                reads.append(self.name)
                return np.zeros((2, 2, 2), np.int32)

        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _A)
        atlas_module.ensure_downloaded("some_atlas")
        assert reads == ["some_atlas"]


class TestAtlasesAreReleased:
    """Atlas objects must not outlive ``get_atlas``'s cache.

    Regression test for a production OOM: ``_region_info_from_id`` used to take
    the atlas *object* as its cache key. ``lru_cache`` holds keys alive, so
    every atlas ever looked up stayed resident with its annotation volume —
    hundreds of MB each — and eviction from ``get_atlas`` leaked a fresh copy
    per lookup instead of reusing one.
    """

    @staticmethod
    def _tracked_atlas_cls(built):
        """A fake atlas class that weakly records every instance it builds."""

        class _A:
            def __init__(self, name, **_kwargs):
                self.name = name
                self.orientation = "asr"
                self.resolution = (25.0, 25.0, 25.0)
                self.annotation = np.ones((2, 2, 2), dtype=np.int32)
                self.structures = {
                    1: {"id": 1, "acronym": "R", "name": "Region",
                        "rgb_triplet": [1, 2, 3]}
                }
                built.append(weakref.ref(self))

        return _A

    @staticmethod
    def _live(built):
        gc.collect()
        return sum(ref() is not None for ref in built)

    def test_atlases_beyond_the_cache_bound_are_freed(self, monkeypatch):
        built: list = []
        monkeypatch.setattr(
            atlas_module, "BrainGlobeAtlas", self._tracked_atlas_cls(built)
        )
        coords = np.array([[0.0, 0.0, 0.0]])

        bound = atlas_module.get_atlas.cache_info().maxsize
        for i in range(bound * 3):
            atlas_module.lookup_regions(f"atlas_{i}", coords)
            atlas_module.canonical_annotation.cache_clear()  # this cache holds views

        assert len(built) == bound * 3, "expected one atlas per distinct name"
        assert self._live(built) <= bound

    def test_rotating_past_the_bound_does_not_accumulate(self, monkeypatch):
        """The case that took the server down: one more atlas in rotation than
        ``get_atlas`` can hold, so every lookup rebuilds an evicted atlas."""
        built: list = []
        monkeypatch.setattr(
            atlas_module, "BrainGlobeAtlas", self._tracked_atlas_cls(built)
        )
        coords = np.array([[0.0, 0.0, 0.0]])

        bound = atlas_module.get_atlas.cache_info().maxsize
        names = [f"atlas_{i}" for i in range(bound + 1)]
        for i in range(bound * 8):
            atlas_module.lookup_regions(names[i % len(names)], coords)
            atlas_module.canonical_annotation.cache_clear()

        # Rebuilding on eviction is expected; *retaining* the rebuilds is not.
        assert len(built) > bound, "expected evictions to force rebuilds"
        assert self._live(built) <= bound


class TestCompactCache:
    """Tests for the compact uint16 label volume (PR C)."""

    def test_lut_round_trips_the_native_annotation(self, monkeypatch):
        # Native "sla" volume, deliberately not asr, so compaction and the
        # reorientation both have to hold at once.
        native = np.arange(2 * 3 * 4, dtype=np.int32).reshape(2, 3, 4)
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _atlas_cls("sla", native))
        arr, _ = atlas_module.canonical_annotation("lut_roundtrip")
        lut = atlas_module.label_ids("lut_roundtrip")
        expected = np.flip(np.transpose(native, (2, 0, 1)), axis=2)
        np.testing.assert_array_equal(lut[arr], expected)

    def test_index_zero_always_maps_to_atlas_id_zero(self, monkeypatch):
        # No voxel is actually 0 here, but the LUT must still reserve index 0
        # for id 0 (the "outside the brain" sentinel every `!= 0` check relies on).
        ann = np.full((4, 4, 4), 5, dtype=np.int32)
        structs = {5: {"id": 5, "acronym": "X", "name": "x region", "rgb_triplet": [1, 1, 1]}}
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _atlas_cls("asr", ann, structs))
        lut = atlas_module.label_ids("all_nonzero")
        assert lut[0] == 0

    def test_compact_build_is_reused_not_rewritten(self, monkeypatch):
        """A second ensure_compact call must not touch the source annotation
        again, and must not rewrite the files on disk."""

        class _CountingAtlas:
            def __init__(self, name, **_kwargs):
                self.name = name
                self.orientation = "asr"
                self.resolution = (25.0, 25.0, 25.0)
                self.structures = {}
                self.reads = 0
                self._data = np.zeros((4, 4, 4), dtype=np.int32)
                self._data[2:, :, :] = 3
                self.shape = self._data.shape  # so the size guard doesn't need `.annotation`

            @property
            def annotation(self):
                self.reads += 1
                return self._data

        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _CountingAtlas)

        out_dir1 = atlas_module.ensure_compact("reuse_me")
        atlas = atlas_module.get_atlas("reuse_me")
        assert atlas.reads == 1
        mtime1 = (out_dir1 / "labels_u16.npy").stat().st_mtime_ns

        out_dir2 = atlas_module.ensure_compact("reuse_me")
        mtime2 = (out_dir2 / "labels_u16.npy").stat().st_mtime_ns

        assert out_dir1 == out_dir2
        assert mtime1 == mtime2
        assert atlas.reads == 1, "second ensure_compact call must not re-read the annotation"

    def test_sparse_id_space_round_trips_through_compaction(self, monkeypatch):
        """Region ids far apart (0, 7, 100000) must still resolve correctly
        once compacted into small, contiguous uint16 indices."""
        ann = np.zeros((4, 4, 4), dtype=np.int64)
        ann[1, 1, 1] = 7
        ann[2, 2, 2] = 100_000
        structs = {
            7: {"id": 7, "acronym": "SEVEN", "name": "seven region", "rgb_triplet": [9, 9, 9]},
            100_000: {"id": 100_000, "acronym": "BIG", "name": "big id region",
                      "rgb_triplet": [8, 8, 8]},
        }
        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _atlas_cls("asr", ann, structs))

        coords = np.array([
            [25.0, 25.0, 25.0],   # voxel (1,1,1) -> id 7
            [50.0, 50.0, 50.0],   # voxel (2,2,2) -> id 100000
        ])
        out = atlas_module.lookup_regions("sparse_ids", coords)
        assert out[0].acronym == "SEVEN"
        assert out[1].acronym == "BIG"


class TestSizeGuard:
    @staticmethod
    def _shape_only_atlas_cls(shape):
        class _A:
            def __init__(self, name, **_kwargs):
                self.name = name
                self.orientation = "asr"
                self.resolution = (25.0, 25.0, 25.0)
                self.shape = shape
                self.structures = {}

            @property
            def annotation(self):
                raise AssertionError(
                    "annotation must not be read once the size guard rejects the atlas"
                )

        return _A

    def test_oversized_atlas_is_rejected_without_reading_annotation(self, monkeypatch):
        monkeypatch.setenv("PIXELMAP_ATLAS_MAX_BYTES", "1000")  # far smaller than any real atlas
        monkeypatch.setattr(
            atlas_module, "BrainGlobeAtlas", self._shape_only_atlas_cls((100, 100, 100))
        )
        with pytest.raises(atlas_module.AtlasTooLargeError):
            atlas_module.canonical_annotation("way_too_big")

    def test_default_limit_admits_the_prebaked_rat_atlas(self):
        # whs_sd_rat_39um is 1024×512×512 uint32 ≈ 1.07 GB — must fit under
        # the default guard (1.5 GB) with room to spare.
        rat_bytes = 1024 * 512 * 512 * 4
        assert rat_bytes < atlas_module.atlas_max_bytes()

    def test_env_override_changes_the_limit(self, monkeypatch):
        monkeypatch.setenv("PIXELMAP_ATLAS_MAX_BYTES", "123456")
        assert atlas_module.atlas_max_bytes() == 123456


class TestSingleResidentVolume:
    def test_switching_atlases_drops_the_previous_slot(self, monkeypatch):
        data = {
            "a": np.zeros((2, 2, 2), dtype=np.int32),
            "b": np.ones((2, 2, 2), dtype=np.int32),
        }

        class _A:
            def __init__(self, name, **_kwargs):
                self.name = name
                self.orientation = "asr"
                self.resolution = (25.0, 25.0, 25.0)
                self.annotation = data[name]
                self.structures = {
                    1: {"acronym": "R", "name": "Region", "rgb_triplet": [1, 2, 3]}
                }

        monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", _A)

        atlas_module.canonical_annotation("a")
        assert atlas_module.canonical_annotation.cache_info().currsize == 1

        atlas_module.canonical_annotation("b")
        info = atlas_module.canonical_annotation.cache_info()
        assert info.currsize == 1, "only one atlas's volume may be resident at a time"

        # Asking for "a" again must be a fresh miss, not a cache hit — its
        # slot was dropped when "b" was loaded, not kept alongside it.
        misses_before = info.misses
        atlas_module.canonical_annotation("a")
        assert atlas_module.canonical_annotation.cache_info().misses == misses_before + 1
