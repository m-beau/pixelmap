"""Tests for the lazy, canonically-oriented annotation reader.

Two things have to hold, and they pull in opposite directions:

* **Correctness** — a ``CanonicalVolume`` must be indistinguishable from the
  eagerly reoriented array it replaces, for every atlas orientation. The
  reference here is literally the expression the old ``canonical_annotation``
  used, so any divergence in the transpose/flip logic shows up immediately.
* **Laziness** — it must not read more of the store than the caller asked for.
  Correctness alone would be satisfied by materialising everything, which is
  precisely the behaviour being removed, so the reads are spied on.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from pixelmap.anatomy.atlas import anatomical_axes
from pixelmap.anatomy.volume import CanonicalVolume

# Every orientation brainglobe can express: one letter per anatomical axis,
# in any axis order — 3! orderings x 2^3 directions.
ALL_ORIENTATIONS = [
    "".join(p)
    for axes in itertools.permutations("ASL")
    for p in itertools.product(*[{"A": "ap", "S": "si", "L": "lr"}[c] for c in axes])
]


class _FakeAtlas:
    def __init__(self, orientation, arr):
        self.orientation = orientation
        self.annotation = arr
        self.shape = arr.shape
        self.resolution = (25.0, 25.0, 25.0)


class _SpyStore:
    """Wraps an ndarray and records every index tuple asked of it."""

    def __init__(self, arr):
        self._arr = arr
        self.shape = arr.shape
        self.dtype = arr.dtype
        self.chunks = (2, 2, 2)
        self.reads: list = []

    def __getitem__(self, key):
        self.reads.append(key)
        return self._arr[key]

    def read_volume(self) -> int:
        """Total voxels pulled out of the store across all reads."""
        total = 0
        for key in self.reads:
            shape = self._arr[key].shape
            total += int(np.prod(shape)) if shape else 1
        return total


def _build(orientation, arr):
    """A CanonicalVolume plus the eagerly-reoriented reference array."""
    atlas = _FakeAtlas(orientation, arr)
    axes = anatomical_axes(atlas)
    vol = CanonicalVolume(arr, axes, (25.0, 25.0, 25.0))

    order = (axes["AP"].array_axis, axes["DV"].array_axis, axes["ML"].array_axis)
    ref = np.transpose(arr, order)
    flips = tuple(i for i, k in enumerate(("AP", "DV", "ML")) if axes[k].flip)
    if flips:
        ref = np.flip(ref, axis=flips)
    return vol, ref


@pytest.mark.parametrize("orientation", ALL_ORIENTATIONS)
class TestMatchesTheEagerReorientation:
    """The lazy reader must agree with transpose+flip on the full array."""

    ARR = np.arange(4 * 5 * 6, dtype=np.int32).reshape(4, 5, 6)

    def test_shape_and_full_materialisation(self, orientation):
        vol, ref = _build(orientation, self.ARR)
        assert vol.shape == ref.shape
        np.testing.assert_array_equal(np.asarray(vol), ref)

    def test_every_orthogonal_slice(self, orientation):
        vol, ref = _build(orientation, self.ARR)
        n_ap, n_dv, n_ml = ref.shape
        for i in range(n_ap):
            np.testing.assert_array_equal(vol[i, :, :], ref[i, :, :])
        for j in range(n_dv):
            np.testing.assert_array_equal(vol[:, j, :], ref[:, j, :])
        for k in range(n_ml):
            np.testing.assert_array_equal(vol[:, :, k], ref[:, :, k])

    def test_sub_boxes_and_scalars(self, orientation):
        vol, ref = _build(orientation, self.ARR)
        rng = np.random.default_rng(0)
        n_ap, n_dv, n_ml = ref.shape
        for _ in range(15):
            a0, a1 = sorted(rng.integers(0, n_ap + 1, 2))
            d0, d1 = sorted(rng.integers(0, n_dv + 1, 2))
            m0, m1 = sorted(rng.integers(0, n_ml + 1, 2))
            np.testing.assert_array_equal(
                vol[a0:a1, d0:d1, m0:m1], ref[a0:a1, d0:d1, m0:m1]
            )
        for a, d, m in itertools.product(range(n_ap), range(n_dv), range(n_ml)):
            assert vol[a, d, m] == ref[a, d, m]

    def test_gather_matches_fancy_indexing(self, orientation):
        vol, ref = _build(orientation, self.ARR)
        rng = np.random.default_rng(1)
        ap = rng.integers(0, ref.shape[0], 50)
        dv = rng.integers(0, ref.shape[1], 50)
        ml = rng.integers(0, ref.shape[2], 50)
        np.testing.assert_array_equal(vol.gather(ap, dv, ml), ref[ap, dv, ml])

    def test_projections_match_the_eager_silhouette(self, orientation):
        arr = self.ARR.copy()
        arr[arr % 3 == 0] = 0  # carve holes so the projections aren't all-True
        vol, ref = _build(orientation, arr)
        inside = ref > 0
        over_ml, over_ap, over_dv = vol.projections()
        np.testing.assert_array_equal(over_ml, inside.any(axis=2))
        np.testing.assert_array_equal(over_ap, inside.any(axis=0))
        np.testing.assert_array_equal(over_dv, inside.any(axis=1))

    def test_find_label_voxels_matches_isin(self, orientation):
        vol, ref = _build(orientation, self.ARR)
        ids = [int(v) for v in np.unique(ref)[3:7]]
        got = vol.find_label_voxels(ids)
        expected = np.where(np.isin(ref, ids))
        for g, e in zip(got, expected):
            np.testing.assert_array_equal(np.sort(g), np.sort(e))


class TestReadsAreBounded:
    """The whole point: never pull more out of the store than was asked for."""

    ARR = np.arange(8 * 8 * 8, dtype=np.int32).reshape(8, 8, 8) % 5

    def test_a_2d_slice_does_not_read_the_volume(self):
        atlas = _FakeAtlas("asr", self.ARR)
        spy = _SpyStore(self.ARR)
        vol = CanonicalVolume(spy, anatomical_axes(atlas), (25.0, 25.0, 25.0))

        vol[4, :, :]

        assert spy.read_volume() == 8 * 8, (
            f"read {spy.read_volume()} voxels for a 64-voxel slice — the "
            "whole volume is being materialised"
        )

    def test_gather_reads_only_the_bounding_box(self):
        atlas = _FakeAtlas("asr", self.ARR)
        spy = _SpyStore(self.ARR)
        vol = CanonicalVolume(spy, anatomical_axes(atlas), (25.0, 25.0, 25.0))

        ap = np.array([1, 2, 1])
        dv = np.array([0, 1, 1])
        ml = np.array([5, 6, 5])
        out = vol.gather(ap, dv, ml)

        np.testing.assert_array_equal(out, self.ARR[ap, dv, ml])
        assert len(spy.reads) == 1, "gather must issue exactly one read"
        assert spy.read_volume() == 2 * 2 * 2, (
            f"read {spy.read_volume()} voxels for a 2x2x2 bounding box"
        )

    def test_whole_volume_scans_are_slabbed_not_materialised(self, monkeypatch):
        import pixelmap.anatomy.volume as volume_module

        # One slab per 2 AP planes, so a full scan needs several reads.
        monkeypatch.setattr(volume_module, "_SLAB_BYTES", 2 * 8 * 8 * 4)
        atlas = _FakeAtlas("asr", self.ARR)
        spy = _SpyStore(self.ARR)
        vol = CanonicalVolume(spy, anatomical_axes(atlas), (25.0, 25.0, 25.0))

        vol.projections()

        assert len(spy.reads) > 1, "a full scan must be slabbed, not one big read"
        biggest = max(int(np.prod(self.ARR[k].shape)) for k in spy.reads)
        assert biggest < self.ARR.size, (
            "a single slab pulled the whole volume — peak RAM is unbounded"
        )


class TestRejectsUnsupportedIndexing:
    """Fail loudly rather than silently returning the wrong voxels."""

    ARR = np.zeros((4, 4, 4), dtype=np.int32)

    def _vol(self):
        atlas = _FakeAtlas("asr", self.ARR)
        return CanonicalVolume(self.ARR, anatomical_axes(atlas), (25.0, 25.0, 25.0))

    def test_wrong_number_of_axes(self):
        with pytest.raises(IndexError):
            self._vol()[1, 2]

    def test_strided_slice(self):
        with pytest.raises(TypeError):
            self._vol()[::2, :, :]

    def test_fancy_indexing_points_at_gather(self):
        with pytest.raises(TypeError):
            self._vol()[np.array([0, 1]), :, :]
