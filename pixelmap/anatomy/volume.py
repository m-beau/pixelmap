"""Lazy, canonically-oriented access to a brainglobe annotation volume.

PixelMap never needs a whole annotation.  It reads a few hundred voxels along a
probe, three 2D slices for the locator, and — only when the tip falls outside
the brain — a whole-brain silhouette.  brainglobe's ``atlas.annotation``
nevertheless materialises the entire array, which is the single largest cost in
the app: 308 MB for allen_mouse_25um, 692 MB for danionella_cerebrum_mixed_2.5um,
4.8 GB for allen_mouse_10um.

Worse, materialising costs far more than the array itself.  The data is stored
as chunked OME-Zarr, and assembling it decompresses every chunk: measured 3.4x
the array size through brainglobe's dask path, 2.4x through a plain zarr read.
danionella's chunks are 322**3 — 134 MB decompressed apiece, sixteen of them for
a 692 MB array — so simply opening it peaked at 2.3 GB.

:class:`CanonicalVolume` reads from the zarr store instead, so only the chunks
covering the requested region are ever decompressed.  It also hides the
orientation remap: every atlas is presented as canonical ``(AP, DV, ML)``
(brainglobe ``"asr"``) regardless of its native axis order, which is what the
rest of :mod:`pixelmap.anatomy` assumes.

The backing store is anything supporting numpy basic indexing and ``.shape`` —
a ``zarr.Array`` in production, a plain ``ndarray`` for test doubles and for
atlases whose on-disk layout cannot be resolved.
"""

from __future__ import annotations

import numpy as np

#: Target bytes per slab for the whole-volume scans (silhouette, label search).
#: These are the only operations that must touch every voxel, and this is what
#: keeps them O(slab) in RAM instead of O(volume).
_SLAB_BYTES = 64 * 1024 * 1024


class CanonicalVolume:
    """An annotation volume indexed as canonical ``(AP, DV, ML)``, read lazily.

    Supports ``vol[i, j, k]`` with ints and step-1 slices, which is everything
    PixelMap asks for.  Each read is translated into the store's native axis
    order, fetched, then transposed and flipped back into canonical order — so
    only the requested region is ever decompressed and held.
    """

    def __init__(self, store, axes, resolution):
        self._store = store
        self._axes = axes              # {"AP"|"DV"|"ML": _AnatAxis}
        self._order = ("AP", "DV", "ML")
        self.resolution = tuple(float(r) for r in resolution)
        self.shape = tuple(int(axes[k].n) for k in self._order)
        self.dtype = np.dtype(getattr(store, "dtype", np.int32))
        self._projections = None

    # -- indexing ----------------------------------------------------------

    def __getitem__(self, key) -> np.ndarray:
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) != 3:
            raise IndexError(
                f"CanonicalVolume needs a 3-axis index (AP, DV, ML); got {len(key)}"
            )

        native: list = [slice(None)] * 3
        kept: list[tuple[str, bool]] = []   # canonical axes that survive, + flip
        for kind, item in zip(self._order, key):
            axis = self._axes[kind]
            if isinstance(item, (int, np.integer)):
                native[axis.array_axis] = _native_index(int(item), axis.n, axis.flip)
            elif isinstance(item, slice):
                native[axis.array_axis] = _native_slice(item, axis.n, axis.flip)
                kept.append((kind, axis.flip))
            else:
                raise TypeError(
                    "CanonicalVolume supports ints and step-1 slices only; "
                    f"got {type(item).__name__} on the {kind} axis"
                )

        out = np.asarray(self._store[tuple(native)])
        if not kept:
            return out

        # The read came back in *native* axis order; put the surviving axes
        # back into canonical (AP, DV, ML) order, then undo any flipped reads.
        by_native = sorted(kept, key=lambda kf: self._axes[kf[0]].array_axis)
        out = np.transpose(out, [by_native.index(kf) for kf in kept])
        flipped = tuple(i for i, (_, flip) in enumerate(kept) if flip)
        return np.flip(out, axis=flipped) if flipped else out

    def __array__(self, dtype=None, copy=None):
        """Materialise the whole volume — the thing this class exists to avoid.

        Present so ``np.asarray(vol)`` and ``np.testing.assert_array_equal``
        work on small test volumes.  Never call it on a real atlas.
        """
        out = self[:, :, :]
        return out.astype(dtype) if dtype is not None else out

    def gather(self, ap, dv, ml) -> np.ndarray:
        """Values at N canonical voxel indices, via a single bounding-box read.

        Reading the points one at a time would be far worse than reading the
        whole volume: every scalar access decompresses the entire chunk holding
        it, so 380 lookups cost 289 MB and 475 ms against 340 MB and 78 ms for
        the full array.  One read of their bounding box costs 15 MB and 5 ms —
        a probe trajectory is a thin box, which is the case that matters.

        Indices must already be inside the volume.
        """
        ap = np.asarray(ap, dtype=int)
        dv = np.asarray(dv, dtype=int)
        ml = np.asarray(ml, dtype=int)
        if ap.size == 0:
            return np.zeros(0, dtype=self.dtype)

        lo = (int(ap.min()), int(dv.min()), int(ml.min()))
        hi = (int(ap.max()) + 1, int(dv.max()) + 1, int(ml.max()) + 1)
        box = self[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        return box[ap - lo[0], dv - lo[1], ml - lo[2]]

    # -- whole-volume scans, done a slab at a time --------------------------

    def _slabs(self):
        """Yield ``(start, stop, block)`` over canonical AP, bounded in RAM.

        Slabs are aligned to the store's native chunking along whichever axis
        AP maps to, so a chunk is decompressed once rather than once per slab.
        """
        n_ap = self.shape[0]
        plane = int(np.prod(self.shape[1:])) * self.dtype.itemsize
        thickness = max(1, _SLAB_BYTES // max(plane, 1))

        chunks = getattr(self._store, "chunks", None)
        if chunks:
            native_chunk = int(chunks[self._axes["AP"].array_axis])
            if native_chunk > 0:
                thickness = max(native_chunk, thickness - thickness % native_chunk)

        for start in range(0, n_ap, thickness):
            stop = min(start + thickness, n_ap)
            yield start, stop, self[start:stop, :, :]

    def projections(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Whole-brain silhouette, as the three orthogonal max-projections.

        Returns ``(over_ML, over_AP, over_DV)`` with shapes ``(AP, DV)``,
        ``(DV, ML)`` and ``(AP, ML)`` — the masks the locator falls back to when
        the tip lies outside the volume.  Computed slab-wise, so peak RAM is one
        slab rather than the volume plus a full-size bool array, and cached
        because the result is a few hundred KB and the scan is the expensive
        part.
        """
        if self._projections is not None:
            return self._projections

        n_ap, n_dv, n_ml = self.shape
        over_ml = np.zeros((n_ap, n_dv), dtype=bool)
        over_ap = np.zeros((n_dv, n_ml), dtype=bool)
        over_dv = np.zeros((n_ap, n_ml), dtype=bool)
        for start, stop, block in self._slabs():
            inside = block > 0
            over_ml[start:stop] = inside.any(axis=2)
            over_ap |= inside.any(axis=0)
            over_dv[start:stop] = inside.any(axis=1)

        self._projections = (over_ml, over_ap, over_dv)
        return self._projections

    def find_label_voxels(self, ids) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Canonical ``(AP, DV, ML)`` indices of every voxel labelled in ``ids``.

        Slab-wise for the same reason as :meth:`projections`: ``np.isin`` over a
        whole volume allocates a bool array the size of the voxel count on top
        of the volume itself — 1.2 GB for a 10 µm atlas.
        """
        ids = np.asarray(list(ids))
        ap_out: list = []
        dv_out: list = []
        ml_out: list = []
        for start, _stop, block in self._slabs():
            ap, dv, ml = np.where(np.isin(block, ids))
            if ap.size:
                ap_out.append(ap + start)
                dv_out.append(dv)
                ml_out.append(ml)
        if not ap_out:
            empty = np.zeros(0, dtype=int)
            return empty, empty.copy(), empty.copy()
        return (
            np.concatenate(ap_out),
            np.concatenate(dv_out),
            np.concatenate(ml_out),
        )


def _native_index(i: int, n: int, flip: bool) -> int:
    """Canonical position ``i`` as an index into a possibly-reversed native axis."""
    return n - 1 - i if flip else i


def _native_slice(item: slice, n: int, flip: bool) -> slice:
    """Canonical slice as a *forward* native slice (the caller re-flips the read).

    Zarr has no negative-step indexing, so a flipped axis is read forward over
    the mirrored range and the resulting block reversed afterwards.
    """
    start, stop, step = item.indices(n)
    if step != 1:
        raise TypeError("CanonicalVolume supports step-1 slices only")
    return slice(n - stop, n - start) if flip else slice(start, stop)
