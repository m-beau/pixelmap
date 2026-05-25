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


def orientation_code(name: str = _DEFAULT_ATLAS) -> str:
    """The atlas's native voxel orientation string (e.g. ``"asr"``)."""
    return str(get_atlas(name).orientation)


# Published bregma estimates + atlas-vs-stereotaxic corrections, per atlas family.
# These let the GUI offer a bregma-relative coordinate mode; they are rough,
# editable defaults (the Allen CCF has no true fiducial — see AtlasScaling.md).
#
# * allen_mouse_*: bregma, DV "squish" and nose-up tilt as baked into the
#   Neuropixels Trajectory Explorer (Peters), which encodes the cortex-lab/IBL
#   bregma estimate and the "Toronto MRI" scaling. bregma voxel [ML,AP,DV] =
#   [570.5, 520, 44] @10µm; DV squish 0.885 (AP/ML default to 1.0 — no trusted
#   estimate); AP tilt 15° (~total disagreement with Franklin & Paxinos per
#   AtlasScaling.md, vs the IBL ~5° estimate).
# * whs_sd_rat: bregma is *defined* by the Waxholm atlas (Papp et al. 2014,
#   source NIfTI voxel [coronal,sagittal,horizontal] = [653, 266, 440]),
#   mapped into brainglobe's reoriented frame (AP & DV flipped, ML not) and
#   validated against the measured anterior-commissure decussation (the WHS
#   origin) to <100 µm in AP/DV. No squish/tilt — WHS is stereotaxically aligned.
#
# bregma_um is (AP, ML, DV) µm in the canonical asr frame; atlas DV =
# real DV / dv_squish; tilt_deg is nose-up rotation about the ML axis.
_ATLAS_REFERENCE = {
    "allen_mouse": {
        "bregma_um": (5200.0, 5705.0, 440.0),
        "ap_squish": 1.0, "ml_squish": 1.0, "dv_squish": 0.885, "tilt_deg": 15.0,
        "source": ("the cortex-lab / IBL estimate + Toronto-MRI scaling "
                   "(Neuropixels Trajectory Explorer), modified empirically by "
                   "Julie Fabre. The Allen CCF has no true bregma, so this is "
                   "approximate"),
    },
    "whs_sd_rat": {
        "bregma_um": (14469.0, 10374.0, 2808.0),
        "ap_squish": 1.0, "ml_squish": 1.0, "dv_squish": 1.0, "tilt_deg": 0.0,
        "source": ("the Waxholm atlas (Papp et al. 2014), mapped into this "
                   "atlas's frame and checked against the anterior-commissure "
                   "decussation (AP/DV within ~0.1 mm)"),
    },
}


def reference_params(name: str = _DEFAULT_ATLAS) -> dict | None:
    """Bregma + DV-squish + tilt estimates for an atlas, or ``None`` if unknown.

    Matched by atlas-name prefix, so every ``allen_mouse_*`` resolution shares
    one entry. Returned values are editable defaults, not ground truth.
    """
    name = str(name)
    for prefix, params in _ATLAS_REFERENCE.items():
        if name.startswith(prefix):
            return dict(params)
    return None


# Which anatomical axis each orientation letter belongs to, and the letter that
# marks the canonical (AP, DV, ML) origin: AP from anterior, DV from the dorsal
# (superior) surface, ML from the right. pixelmap works in this fixed frame.
_AXIS_KIND = {"a": "AP", "p": "AP", "s": "DV", "i": "DV", "l": "ML", "r": "ML"}
_CANONICAL_ORIGIN = {"AP": "a", "DV": "s", "ML": "r"}


@dataclass(frozen=True)
class _AnatAxis:
    """Where one anatomical axis lives in the native annotation array."""

    array_axis: int    # which array axis (0/1/2) this anatomical axis occupies
    flip: bool         # True if the native axis runs opposite the canonical one
    n: int             # voxel count along the axis
    res_um: float      # µm per voxel along the axis


def anatomical_axes(atlas) -> dict[str, _AnatAxis]:
    """Map ``"AP"``/``"DV"``/``"ML"`` to their place in the native array.

    Read from ``atlas.orientation`` (e.g. ``"asr"``) so coordinate lookups
    work for any brainglobe orientation, not just Allen's. See
    :func:`canonical_annotation` for how this is applied.
    """
    orientation = str(atlas.orientation).lower()
    shape = atlas.annotation.shape
    res = np.asarray(atlas.resolution, dtype=float)
    axes: dict[str, _AnatAxis] = {}
    for axis, letter in enumerate(orientation):
        kind = _AXIS_KIND[letter]
        axes[kind] = _AnatAxis(
            array_axis=axis,
            flip=(letter != _CANONICAL_ORIGIN[kind]),
            n=int(shape[axis]),
            res_um=float(res[axis]),
        )
    if set(axes) != {"AP", "DV", "ML"}:
        raise ValueError(f"Unsupported atlas orientation: {orientation!r}")
    return axes


@functools.lru_cache(maxsize=4)
def canonical_annotation(atlas_name: str):
    """Return ``(annotation, resolution)`` reoriented to canonical ``(AP, DV, ML)``.

    The canonical layout is brainglobe ``"asr"``: axis 0 = AP (anterior→posterior),
    axis 1 = DV (dorsal→ventral), axis 2 = ML (right→left), with µm measured from
    the anterior-superior-right corner. The rest of :mod:`pixelmap.anatomy`
    assumes this layout, so funnelling every atlas through here is what lets
    non-Allen orientations work. For an already-``asr`` atlas this is a no-op
    (identity transpose, no flips), so Allen behavior is unchanged.
    """
    atlas = get_atlas(atlas_name)
    axes = anatomical_axes(atlas)
    order = (axes["AP"].array_axis, axes["DV"].array_axis, axes["ML"].array_axis)
    arr = np.transpose(atlas.annotation, order)
    flip_axes = tuple(i for i, kind in enumerate(("AP", "DV", "ML")) if axes[kind].flip)
    if flip_axes:
        arr = np.flip(arr, axis=flip_axes)
    res = np.array(
        [axes["AP"].res_um, axes["DV"].res_um, axes["ML"].res_um], dtype=float
    )
    return arr, res


def volume_center_um(name: str = _DEFAULT_ATLAS) -> tuple[float, float, float]:
    """Geometric center of the atlas volume as ``(AP, ML, DV)`` µm.

    Handy as a default insertion target — it lands mid-brain for any atlas.
    Reads the atlas (downloads if not cached), so gate on
    :func:`is_downloaded` where staying cheap matters.
    """
    ann, res = canonical_annotation(name)
    n_ap, n_dv, n_ml = ann.shape
    return (n_ap * res[0] / 2.0, n_ml * res[2] / 2.0, n_dv * res[1] / 2.0)


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
    # Reorient to canonical (AP, DV, ML) so the indexing below holds for any
    # atlas orientation, not just Allen's native "asr".
    annotation, voxel_size = canonical_annotation(atlas_name)

    coords = np.asarray(atlas_coords_um, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"atlas_coords_um must be (N, 3); got {coords.shape}")

    # canonical annotation is indexed (AP, DV, ML); convert µm → voxel.
    ap_idx = np.round(coords[:, 0] / voxel_size[0]).astype(int)
    dv_idx = np.round(coords[:, 2] / voxel_size[1]).astype(int)
    ml_idx = np.round(coords[:, 1] / voxel_size[2]).astype(int)

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
