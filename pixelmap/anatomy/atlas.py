"""Thin wrapper around ``brainglobe-atlasapi`` for PixelMap.

Why a wrapper:

* Make the dependency optional — if ``brainglobe-atlasapi`` is not
  installed, importing :mod:`pixelmap` itself must still work; only the
  anatomy-specific entry points should fail, and they should fail with a
  one-line install hint rather than an opaque ``ModuleNotFoundError``.
* Cache atlas instances per-process so repeated lookups don't re-load the
  annotation volume (it's tens of MB).
* Expose a tiny ``RegionInfo`` record so the rest of PixelMap doesn't
  depend on brainglobe's object model.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import numpy as np

_DEFAULT_ATLAS = "allen_mouse_25um"

_BRAINGLOBE_INSTALL_HINT = (
    "Anatomical-overlay features require brainglobe-atlasapi. "
    "Install with: pip install 'pixelmap[anatomy]'"
)


@dataclass(frozen=True)
class RegionInfo:
    """A region's atlas-derived identity for one electrode."""

    atlas_id: int          # Atlas integer label at the lookup voxel
    acronym: str           # Short region tag, e.g. "VISp"
    name: str              # Full name, e.g. "Primary visual area"
    rgb: tuple[int, int, int]  # 0-255 color as defined by the atlas


def _import_brainglobe() -> Any:
    """Return the BrainGlobeAtlas class, raising a friendly error if missing."""
    try:
        from brainglobe_atlasapi import BrainGlobeAtlas
    except ImportError as exc:
        raise ImportError(_BRAINGLOBE_INSTALL_HINT) from exc
    return BrainGlobeAtlas


@functools.lru_cache(maxsize=4)
def get_atlas(name: str = _DEFAULT_ATLAS):
    """Return a cached :class:`BrainGlobeAtlas` instance.

    First call for a given atlas may download tens of MB. We delegate the
    download/caching to brainglobe — its on-disk cache is shared across
    processes.
    """
    BrainGlobeAtlas = _import_brainglobe()
    return BrainGlobeAtlas(name)


def is_available() -> bool:
    """Cheap check used by the GUI to decide whether to show the panel."""
    try:
        _import_brainglobe()
    except ImportError:
        return False
    return True


def list_atlases() -> list[str]:
    """List every atlas in the brainglobe registry, not just downloaded ones.

    ``get_all_atlases_lastversions`` returns the full set published to the
    brainglobe registry (Allen mouse at several resolutions, Waxholm rat,
    Kim Lab mouse, zebrafish, axolotl, etc.). It does *not* trigger any
    atlas download — only an HTTP HEAD/JSON fetch of the registry index,
    which is cached locally by brainglobe.
    """
    try:
        from brainglobe_atlasapi.list_atlases import get_all_atlases_lastversions
    except ImportError as exc:
        raise ImportError(_BRAINGLOBE_INSTALL_HINT) from exc
    return sorted(get_all_atlases_lastversions().keys())


# brainglobe orientation codes (e.g. "asr") spell the (0,0,0) origin corner:
# one letter per array axis, naming the anatomical side the axis starts from.
_ORIGIN_WORDS = {
    "a": "anterior", "p": "posterior",
    "s": "superior", "i": "inferior",
    "l": "left", "r": "right",
}


def is_downloaded(name: str = _DEFAULT_ATLAS) -> bool:
    """True if the atlas is already on disk, so reading it won't download.

    Lets the GUI fetch an atlas's origin only when that's free — picking an
    un-downloaded atlas from a dropdown should not kick off a tens-of-MB
    download just to label the coordinate space.
    """
    try:
        from brainglobe_atlasapi.list_atlases import get_downloaded_atlases
    except ImportError as exc:
        raise ImportError(_BRAINGLOBE_INSTALL_HINT) from exc
    return name in get_downloaded_atlases()


def origin_corner(name: str = _DEFAULT_ATLAS) -> str:
    """Return the atlas volume's (0,0,0) origin corner in words.

    e.g. ``"anterior-superior-right"`` for the Allen mouse atlas
    (orientation ``"asr"``). Coordinates increase away from this corner.
    The origin differs between atlases, so this is read from the atlas's own
    ``orientation`` metadata rather than assumed.

    Reading the orientation triggers a download if the atlas is not cached;
    gate on :func:`is_downloaded` when the caller must stay cheap.
    """
    orientation = str(get_atlas(name).orientation)  # e.g. "asr"
    return "-".join(_ORIGIN_WORDS.get(c, c) for c in orientation)


def lookup_regions(
    atlas_name: str,
    atlas_coords_um: np.ndarray,
) -> list[RegionInfo | None]:
    """Look up the region at each ``(AP, ML, DV)`` µm coordinate.

    Args:
        atlas_name: brainglobe atlas identifier (e.g. ``"allen_mouse_25um"``).
        atlas_coords_um: shape ``(N, 3)``, atlas-frame ``(AP, ML, DV)`` in µm.

    Returns:
        List of length ``N``. Entries are ``None`` if the corresponding
        coordinate falls outside the volume.
    """
    atlas = get_atlas(atlas_name)

    coords = np.asarray(atlas_coords_um, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"atlas_coords_um must be (N, 3); got {coords.shape}")

    # brainglobe annotation volumes are indexed (AP, DV, ML); convert µm → voxel.
    voxel_size = np.asarray(atlas.resolution, dtype=float)  # (AP, DV, ML) µm/voxel
    ap_idx = np.round(coords[:, 0] / voxel_size[0]).astype(int)
    dv_idx = np.round(coords[:, 2] / voxel_size[1]).astype(int)
    ml_idx = np.round(coords[:, 1] / voxel_size[2]).astype(int)

    annotation = atlas.annotation
    shape = annotation.shape

    results: list[RegionInfo | None] = []
    for ap, dv, ml in zip(ap_idx, dv_idx, ml_idx):
        if not (0 <= ap < shape[0] and 0 <= dv < shape[1] and 0 <= ml < shape[2]):
            results.append(None)
            continue
        region_id = int(annotation[ap, dv, ml])
        if region_id == 0:  # outside-brain or undefined
            results.append(None)
            continue
        results.append(_region_info_from_id(atlas, region_id))
    return results


@functools.lru_cache(maxsize=4096)
def _region_info_from_id(atlas, region_id: int) -> RegionInfo | None:
    """Resolve a region integer label to acronym/name/rgb. Cached per-atlas."""
    try:
        entry = atlas.structures[region_id]
    except KeyError:
        return None
    rgb = tuple(int(c) for c in entry.get("rgb_triplet", (128, 128, 128)))
    return RegionInfo(
        atlas_id=region_id,
        acronym=str(entry.get("acronym", f"id{region_id}")),
        name=str(entry.get("name", "")),
        rgb=rgb,  # type: ignore[arg-type]
    )
