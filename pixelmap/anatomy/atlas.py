"""Thin wrapper around ``brainglobe-atlasapi`` for PixelMap.

Why a wrapper:

* Cache atlas instances per-process so repeated lookups don't re-load the
  annotation volume (it's tens of MB).
* Expose a tiny ``RegionInfo`` record so the rest of PixelMap doesn't
  depend on brainglobe's object model.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_atlasapi.list_atlases import (
    get_all_atlases_lastversions,
    get_downloaded_atlases,
)

from pixelmap.anatomy.transform import probe_axis

_DEFAULT_ATLAS = "allen_mouse_25um"


@dataclass(frozen=True)
class RegionInfo:
    """A region's atlas-derived identity for one electrode."""

    atlas_id: int          # Atlas integer label at the lookup voxel
    acronym: str           # Short region tag, e.g. "VISp"
    name: str              # Full name, e.g. "Primary visual area"
    rgb: tuple[int, int, int]  # 0-255 color as defined by the atlas


@functools.lru_cache(maxsize=4)
def get_atlas(name: str = _DEFAULT_ATLAS):
    """Return a cached :class:`BrainGlobeAtlas` instance.

    First call for a given atlas may download tens of MB. We delegate the
    download/caching to brainglobe — its on-disk cache is shared across
    processes.  ``check_latest=False`` skips the remote version check so the
    app doesn't hang when the GIN server is unreachable.
    """
    return BrainGlobeAtlas(name, check_latest=False)


def list_atlases() -> list[str]:
    """List every atlas in the brainglobe registry, not just downloaded ones.

    Reads the locally cached registry index first so the app starts instantly
    even when the GIN server is unreachable.  Falls back to
    ``get_all_atlases_lastversions`` (which may hit the network) only if the
    cache file is missing.
    """
    try:
        from brainglobe_atlasapi import config, utils

        cache_path = config.get_brainglobe_dir() / "last_versions.conf"
        if cache_path.exists():
            data = utils.conf_from_file(cache_path)
            return sorted(data["atlases"].keys())
    except Exception:
        pass
    # Cache missing or unreadable — fall back to the (potentially slow)
    # network fetch so the full list is still available on first run.
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
#   estimate); AP tilt 13° (empirical, between the IBL ~5° estimate and the
#   ~15° total disagreement with Franklin & Paxinos per AtlasScaling.md).
# * kim_mouse, ccfv2_mouse, ccfv2_fiber, allen_mouse_bluebrain_barrels: these are
#   the *same* Allen adult average template as allen_mouse (verified identical
#   grid: asr, 13200×8000×11400 µm), so the Allen estimate transfers unchanged.
#   The Kim atlas (Chon et al. 2019) shares Allen's reference image; CCFv2 is the
#   same average-template grid as CCFv3 (and is where the IBL bregma was derived).
# * whs_sd_rat: bregma is *defined* by the Waxholm atlas (Papp et al. 2014,
#   source NIfTI voxel [coronal,sagittal,horizontal] = [653, 266, 440]),
#   mapped into brainglobe's reoriented frame (AP & DV flipped, ML not) and
#   validated against the measured anterior-commissure decussation (the WHS
#   origin) to <100 µm in AP/DV. No squish/tilt — WHS is stereotaxically aligned.
# * NOT included (independent / per-age spaces — user must define bregma): the
#   developmental atlases (demba_*, kim_dev_*, ccfv2_dev), the LSFM templates
#   (princeton_mouse, perens_lsfm/multimodal, osten_mouse) and the flat-skull
#   perens_stereotaxic (stereotaxic, but its bregma voxel isn't recoverable from
#   brainglobe's metadata).
#
# bregma_um is (AP, ML, DV) µm in the canonical asr frame; atlas DV =
# real DV / dv_squish; tilt_deg is nose-up rotation about the ML axis.
_ALLEN_BREGMA_UM = (5200.0, 5705.0, 440.0)
_ALLEN_CALIB = {"ap_squish": 1.0, "ml_squish": 1.0, "dv_squish": 0.885, "tilt_deg": 13.0}


def _ccf_ref(source: str) -> dict:
    """A reference entry sharing the Allen CCF bregma + squish/tilt estimate."""
    return {"bregma_um": _ALLEN_BREGMA_UM, **_ALLEN_CALIB, "source": source}


_ATLAS_REFERENCE = {
    "allen_mouse": _ccf_ref(
        "the cortex-lab / IBL estimate + Toronto-MRI scaling (Neuropixels "
        "Trajectory Explorer), modified empirically by Julie Fabre. The Allen "
        "CCF has no true bregma, so this is approximate"),
    "kim_mouse": _ccf_ref(
        "the Allen CCF estimate — the Kim atlas (Chon et al. 2019) shares Allen's "
        "reference image (verified identical grid), modified empirically by "
        "Julie Fabre. No true bregma, so this is approximate"),
    "ccfv2_mouse": _ccf_ref(
        "the Allen CCF estimate — CCFv2 uses the same average-template grid as "
        "CCFv3 (and is where the cortex-lab / IBL bregma was derived), modified "
        "empirically by Julie Fabre. No true bregma, so this is approximate"),
    "ccfv2_fiber": _ccf_ref(
        "the Allen CCF estimate — CCFv2 uses the same grid as CCFv3, modified "
        "empirically by Julie Fabre. No true bregma, so this is approximate"),
    # Both Waxholm-Space rats (whs_sd_rat and the SWC female rat registered into
    # WHS) share this frame, so "whs_sd" covers both.
    "whs_sd": {
        "bregma_um": (14469.0, 10374.0, 2808.0),
        "ap_squish": 1.0, "ml_squish": 1.0, "dv_squish": 1.0, "tilt_deg": 0.0,
        "defined": True,  # a real, atlas-defined bregma (not an estimate)
        "source": ("the Waxholm atlas (Papp et al. 2014), which defines bregma "
                   "explicitly. Recovered by mapping its published bregma voxel "
                   "into this atlas's frame, then validated against the measured "
                   "anterior-commissure decussation (AP/DV within ~0.1 mm)"),
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


def landmark_policy(name: str) -> str | None:
    """The species' conventional stereotaxic origin landmark, or ``None``.

    * ``"anterior commissure"`` — human (AC-PC) and fish (zebrafish / cavefish
      atlases use the AC as their zero point). Derivable from the annotation.
    * ``"bregma"`` — rodents (mouse / rat / vole / mole-rat). A *skull* point we
      can't locate from the annotation; absent a hardcoded value, user defines.
    * ``"interaural"`` — cat (Horsley-Clarke interaural zero, per Snider-Niemer /
      Reinoso-Suárez; bregma is an alternative). External point → user defines.
    * ``None`` — axolotl / cephalopod / spinal cord: no established stereotaxic
      landmark, so the user defines an origin (coordinates are atlas-defined).

    Cheap — just a name check.
    """
    n = str(name).lower()
    if n.startswith("allen_human") or any(k in n for k in ("zfish", "zebrafish", "cavefish")):
        return "anterior commissure"
    if n.startswith("csl_cat"):
        return "interaural"
    if any(k in n for k in ("mouse", "rat", "vole")):
        return "bregma"
    return None


@functools.lru_cache(maxsize=8)
def derive_origin_from_ac(name: str) -> tuple[float, float, float] | None:
    """Origin at the anterior-commissure decussation, from the annotation.

    Returns canonical ``(AP, ML, DV)`` µm, or ``None`` if the atlas delineates
    no anterior commissure. Same recipe used (and validated) for the WHS rat:
    the AC's midline-crossing centroid. Requires the atlas (downloads if absent).
    """
    atlas = get_atlas(name)
    ac_ids = [int(s["id"]) for s in atlas.structures.values()
              if "anterior" in s["name"].lower() and "commis" in s["name"].lower()]
    if not ac_ids:
        return None
    ann, res = canonical_annotation(name)  # (AP, DV, ML)
    ap, dv, ml = np.where(np.isin(ann, ac_ids))
    if ap.size == 0:
        return None
    midline = float(ml.mean())                 # AC ~symmetric → centroid ML = midline
    near = np.abs(ml - midline) < 4             # voxels near midline = decussation
    return (float(ap[near].mean() * res[0]),
            midline * res[2],
            float(dv[near].mean() * res[1]))


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


def _voxel_indices(coords_um: np.ndarray, voxel_size: np.ndarray) -> np.ndarray:
    """``(N, 3)`` atlas ``(AP, ML, DV)`` µm → integer ``(ap, dv, ml)`` indices.

    The canonical annotation is indexed ``(AP, DV, ML)`` while coordinates are
    passed around as ``(AP, ML, DV)``, so the last two swap. ``voxel_size`` is
    already in annotation order ``(AP, DV, ML)`` — see
    :func:`canonical_annotation`. Kept in one place because getting this
    transposition wrong is silent for isotropic atlases.
    """
    c = np.asarray(coords_um, dtype=float)
    return np.round(c[:, [0, 2, 1]] / np.asarray(voxel_size, dtype=float)).astype(int)


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
    idx = _voxel_indices(coords, voxel_size)

    shape = annotation.shape

    results: list[RegionInfo | None] = []
    for ap, dv, ml in idx:
        if not (0 <= ap < shape[0] and 0 <= dv < shape[1] and 0 <= ml < shape[2]):
            results.append(None)
            continue
        region_id = int(annotation[ap, dv, ml])
        if region_id == 0:  # outside-brain or undefined
            results.append(None)
            continue
        results.append(_region_info_from_id(atlas, region_id))
    return results


def surface_point_um(
    atlas_name: str,
    point_atlas: np.ndarray | tuple[float, float, float],
    *,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    step_um: float | None = None,
) -> tuple[float, float, float] | None:
    """Where a probe trajectory enters the brain, as ``(AP, ML, DV)`` µm.

    Walks *down* the trajectory that passes through ``point_atlas`` at the
    given pose and returns the first sample sitting on a labelled voxel —
    i.e. the point at which the shank crosses the brain surface.

    The result depends only on the *line*, not on which of its points is
    passed in: calling this with the tip or with the insertion point gives
    the same answer. That is what lets the GUI's snap button behave
    identically in tip-entry and insertion-entry mode.

    Args:
        atlas_name: brainglobe atlas identifier.
        point_atlas: any ``(AP, ML, DV)`` µm point on the trajectory.
        pitch_deg: AP tilt — see :mod:`pixelmap.anatomy.transform`.
        yaw_deg: ML tilt — see :mod:`pixelmap.anatomy.transform`.
        step_um: march step. Defaults to the finest voxel dimension, so a
            thin dorsal layer is never stepped over.

    Returns:
        The entry point, or ``None`` if the trajectory misses the brain
        entirely (it never crosses a labelled voxel).
    """
    annotation, voxel_size = canonical_annotation(atlas_name)
    axis = probe_axis(pitch_deg, yaw_deg)          # up the shank, (AP, ML, DV)
    p = np.asarray(point_atlas, dtype=float).reshape(3)

    n_ap, n_dv, n_ml = annotation.shape
    # Volume extent in coordinate order (AP, ML, DV); note voxel_size is in
    # annotation order (AP, DV, ML), hence the crossed indices.
    extent = np.array([n_ap * voxel_size[0],
                       n_ml * voxel_size[2],
                       n_dv * voxel_size[1]])

    # Clip the infinite trajectory to the volume's bounding box, rather than
    # marching a fixed window around `point_atlas`. The insertion point is by
    # construction at or above the dorsal surface and is routinely *outside*
    # the volume, so a fixed window is not guaranteed to reach the brain —
    # and would make the answer depend on which anchor was passed in.
    t_lo, t_hi = -np.inf, np.inf
    for i in range(3):
        if abs(axis[i]) < 1e-12:                   # trajectory parallel to this face
            if not (0.0 <= p[i] <= extent[i]):
                return None
            continue
        t0, t1 = (0.0 - p[i]) / axis[i], (extent[i] - p[i]) / axis[i]
        t_lo, t_hi = max(t_lo, min(t0, t1)), min(t_hi, max(t0, t1))
    if t_hi < t_lo:
        return None

    step = float(np.min(voxel_size)) if step_um is None else float(step_um)
    if step <= 0:
        raise ValueError(f"step_um must be positive; got {step_um!r}")

    # March from the top of the clipped segment downward, so the first hit is
    # the entry point rather than the exit wound.
    ts = np.arange(t_hi, t_lo - step, -step)
    pts = p[None, :] + ts[:, None] * axis[None, :]

    idx = _voxel_indices(pts, voxel_size)
    in_bounds = ((idx >= 0) & (idx < np.array([n_ap, n_dv, n_ml]))).all(axis=1)
    labels = np.zeros(len(ts), dtype=annotation.dtype)
    labels[in_bounds] = annotation[idx[in_bounds, 0], idx[in_bounds, 1], idx[in_bounds, 2]]

    hit = np.flatnonzero(labels > 0)
    if hit.size == 0:
        return None
    return tuple(float(v) for v in pts[hit[0]])


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
