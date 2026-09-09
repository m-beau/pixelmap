"""Thin wrapper around ``brainglobe-atlasapi`` for PixelMap.

Why a wrapper:

* Cache atlas instances per-process so repeated lookups don't re-load the
  annotation volume (it's tens of MB).
* Expose a tiny ``RegionInfo`` record so the rest of PixelMap doesn't
  depend on brainglobe's object model.
* Absorb the brainglobe v2 -> v3 storage rewrite in one place, so both
  versions work (see below).

Supporting brainglobe-atlasapi v2 *and* v3
------------------------------------------
v3 kept the parts we use of the Python API (``annotation``, ``resolution``,
``orientation``, ``structures``, ``shape``) but rewrote how atlases live on
disk: OME-Zarr on S3 instead of tiff bundles on GIN, components shared between
atlases, stored under ``~/.brainglobe/brainglobe-atlasapi/`` rather than
``~/.brainglobe/<atlas>/``.  Two consequences matter here:

* The ``last_versions.conf`` registry cache moved, so :func:`list_atlases`
  looks in both places.
* Array data is fetched **lazily**, on first attribute access rather than at
  construction.  So "the atlas is downloaded" and "reading the annotation is
  free" stopped being the same question, and :func:`is_downloaded` — which
  callers use to decide whether an action is cheap — has to answer the second
  one.  That also means metadata-only queries (shape, resolution) must avoid
  touching ``annotation``, or they'd trigger the very download they're meant
  to let the caller avoid.

Everything below feature-detects rather than testing a version number: v3 is
young (verified against 2.3.1 and 3.0.1), and the layout, not the version
string, is what the code actually depends on.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_atlasapi.list_atlases import (
    get_all_atlases_lastversions,
    get_downloaded_atlases,
)

try:  # brainglobe-atlasapi >= 3
    from brainglobe_atlasapi.descriptors import (
        V3_ANNOTATION_NAME,
        V3_ATLAS_ROOTDIR,
    )

    #: True where atlas arrays are fetched lazily and live in the v3 layout.
    IS_V3 = True
except ImportError:  # brainglobe-atlasapi 2.x
    # v2 doesn't define these. Spelling out the values it would have keeps the
    # v3 code paths readable and testable on a v2 install; they are only ever
    # *used* behind an ``IS_V3`` check.
    V3_ANNOTATION_NAME = "annotations_compressed.ome.zarr"
    V3_ATLAS_ROOTDIR = "atlases"
    IS_V3 = False

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

    We delegate the download/caching to brainglobe — its on-disk cache is
    shared across processes.  ``check_latest=False`` skips the remote version
    check so the app doesn't hang when the atlas host is unreachable.

    How much this costs depends on the brainglobe version.  On v2 the first
    call downloads the whole atlas bundle (tens to hundreds of MB).  On v3 it
    only fetches the manifest and metadata (a few hundred KB); the annotation
    array is pulled from S3 later, when something first reads
    ``atlas.annotation``.  Either way it is free once the data is local.
    """
    return BrainGlobeAtlas(name, check_latest=False)


def list_atlases() -> list[str]:
    """List every atlas in the brainglobe registry, not just downloaded ones.

    Reads the locally cached registry index first so the app starts instantly
    even when the atlas host is unreachable.  Falls back to
    ``get_all_atlases_lastversions`` (which may hit the network) only if the
    cache file is missing.

    v3 moved that cache down into the ``brainglobe-atlasapi/atlases/``
    subtree, so both locations are tried — a machine that has used both
    versions has both files, and the newest one wins.
    """
    try:
        from brainglobe_atlasapi import config, utils

        for cache_path in _registry_cache_paths(config.get_brainglobe_dir()):
            if cache_path.exists():
                data = utils.conf_from_file(cache_path)
                return sorted(data["atlases"].keys())
    except Exception:
        pass
    # Cache missing or unreadable — fall back to the (potentially slow)
    # network fetch so the full list is still available on first run.
    return sorted(get_all_atlases_lastversions().keys())


def _registry_cache_paths(brainglobe_dir: Path) -> list[Path]:
    """Candidate ``last_versions.conf`` locations, preferred version first."""
    v3 = brainglobe_dir / "brainglobe-atlasapi" / V3_ATLAS_ROOTDIR / "last_versions.conf"
    v2 = brainglobe_dir / "last_versions.conf"
    return [v3, v2] if IS_V3 else [v2, v3]


# brainglobe orientation codes (e.g. "asr") spell the (0,0,0) origin corner:
# one letter per array axis, naming the anatomical side the axis starts from.
_ORIGIN_WORDS = {
    "a": "anterior", "p": "posterior",
    "s": "superior", "i": "inferior",
    "l": "left", "r": "right",
}


def is_downloaded(name: str = _DEFAULT_ATLAS) -> bool:
    """True if the atlas's annotation is on disk, so reading it won't download.

    Lets the GUI fetch an atlas's origin only when that's free — picking an
    un-downloaded atlas from a dropdown should not kick off a tens-of-MB
    download just to label the coordinate space.

    On v2 an atlas is all-or-nothing, so the presence of its directory settles
    it.  On v3 the directory appears as soon as the (tiny) manifest lands,
    while the annotation is still remote — so we additionally check that the
    OME-Zarr chunks are cached, which is what the callers actually care about.
    """
    if name not in get_downloaded_atlases():
        return False
    if not IS_V3:
        return True
    try:
        return _v3_annotation_is_cached(get_atlas(name))
    except Exception:
        # Metadata unreadable, or the manifest is there but incomplete: treat
        # it as not-downloaded so callers take the "this will cost you" path.
        return False


def _v3_annotation_is_cached(atlas) -> bool:
    """True if v3's lazily-fetched annotation chunks are already local.

    Mirrors the check ``core.Atlas.annotation`` makes before hitting S3: the
    OME-Zarr pyramid holds one directory per scale level, and a level's voxels
    live under ``<level>/c``.  We accept *any* cached level rather than
    reaching into brainglobe's private ``_annotation_pyramid_level`` — the
    atlas name pins the resolution, so the only level this app ever pulls is
    the one it would read back.
    """
    location = atlas.metadata["annotation_set"]["location"][1:]
    root = Path(atlas.root_dir) / location / V3_ANNOTATION_NAME
    if not root.is_dir():
        return False
    return any(level.joinpath("c").is_dir() for level in root.iterdir() if level.is_dir())


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


def _atlas_shape(atlas) -> tuple[int, ...]:
    """The atlas's voxel shape, from metadata where the atlas exposes it.

    Both brainglobe versions publish ``shape`` from the atlas manifest, which
    is the cheap way to ask: on v3, reading ``annotation.shape`` instead would
    pull the entire array from S3.  Test doubles only carry ``annotation``, so
    fall back to that.
    """
    shape = getattr(atlas, "shape", None)
    if shape is not None:
        return tuple(int(s) for s in shape)
    return tuple(int(s) for s in atlas.annotation.shape)


def anatomical_axes(atlas) -> dict[str, _AnatAxis]:
    """Map ``"AP"``/``"DV"``/``"ML"`` to their place in the native array.

    Read from ``atlas.orientation`` (e.g. ``"asr"``) so coordinate lookups
    work for any brainglobe orientation, not just Allen's. See
    :func:`canonical_annotation` for how this is applied.

    Metadata only — deliberately never touches ``atlas.annotation``, so
    callers that just want the volume's extent don't pay for a v3 download.
    """
    orientation = str(atlas.orientation).lower()
    shape = _atlas_shape(atlas)
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
    Computed from shape and resolution alone, so it never materialises the
    annotation; it still needs the atlas's metadata, which on a cold cache
    means a download, so gate on :func:`is_downloaded` where that matters.
    """
    axes = anatomical_axes(get_atlas(name))
    return (
        axes["AP"].n * axes["AP"].res_um / 2.0,
        axes["ML"].n * axes["ML"].res_um / 2.0,
        axes["DV"].n * axes["DV"].res_um / 2.0,
    )


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
