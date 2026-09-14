"""Thin wrapper around ``brainglobe-atlasapi`` for PixelMap.

Why a wrapper:

* Cache atlas instances per-process so repeated lookups don't re-load the
  annotation volume (hundreds of MB once decompressed).
* Expose a tiny ``RegionInfo`` record so the rest of PixelMap doesn't
  depend on brainglobe's object model.
* Keep every network call off the caller's thread, because on the hosted
  server that thread is the one serving every other user (see below).

Requires brainglobe-atlasapi >= 3
---------------------------------
v3 rewrote how atlases live on disk: OME-Zarr on S3 rather than tiff bundles
on GIN, components shared between atlases, stored under
``~/.brainglobe/brainglobe-atlasapi/``.  PixelMap requires it, and v2 is not
supported.  Two properties of that layout shape the code below:

* Array data is fetched **lazily**, on first attribute access rather than at
  construction.  So "the atlas is downloaded" and "reading the annotation is
  free" are different questions, and :func:`is_downloaded` — which callers use
  to decide whether an action is cheap — answers the second one.  Metadata-only
  queries (shape, resolution) must therefore avoid touching ``annotation``, or
  they would trigger the very download they exist to let the caller skip.
* Atlases are small on disk: allen_mouse_25um and whs_sd_rat_39um together are
  ~9 MB of compressed chunks, against ~1.3 GB of tiffs under v2.  That is what
  makes baking them into the Docker image cheap and reliable.

Why the network never runs inline
---------------------------------
The deployed app is a single-process Panel/Bokeh server: a blocking call on
its event loop stalls *every* connected user and the container healthcheck.
brainglobe's registry fetch uses ``requests.get`` with no timeout, so a
hosting outage used to hang the server rather than degrade it.  Hence
:func:`list_atlases` answers from disk or from a bundled snapshot and refreshes
in a daemon thread, and :func:`ensure_downloaded` exists so the GUI can move
the one genuinely expensive call onto a worker thread.

Memory strategy
----------------
``BrainGlobeAtlas.annotation`` decodes the atlas's OME-Zarr chunks into a
full ``uint32`` numpy array and caches it on the instance forever
(``core.Atlas.annotation`` -> ``self._annotation = ...data.compute()``): 308
MB for ``allen_mouse_25um``, 1,074 MB for ``whs_sd_rat_39um``, and roughly
4.8 GB for any 10 µm atlas.  Multiple independent ``lru_cache`` instances
used to each keep their own volume alive, so evicting one cache freed nothing while
the others still held a reference — the production container (capped at
3000 MB) died with exit 137 once a few large atlases were viewed in one
session.

The fix has three parts:

* **Compact on-disk format.**  :func:`ensure_compact` builds, once per
  atlas+version, a ``uint16`` label-index volume (``labels_u16.npy``) plus a
  small index -> atlas-id lookup table (``lut_u32.npy``), written under
  :func:`atlas_cache_dir` (``$PIXELMAP_ATLAS_CACHE_DIR``, else
  ``<brainglobe dir>/pixelmap_cache``).  Index 0 always maps to atlas id 0,
  so every existing ``!= 0`` "inside the brain" check keeps working
  unchanged.  The build reads the source volume in axis-0 slabs — for a real
  atlas that means driving the underlying dask/zarr array directly, so the
  full ``uint32`` volume never exists in memory at all; brainglobe's own
  ``_annotation`` cache is dropped immediately afterwards if anything did
  populate it.  Once built, every later read is ``np.load(path,
  mmap_mode="r")``: the OS pages in only the voxels actually touched.
* **One resident volume.**  :func:`canonical_annotation` holds at most one
  atlas's memmap open at a time (``lru_cache(maxsize=1)``); switching atlases
  drops the previous one instead of accumulating.
* **A size guard.**  Before ever touching the source data,
  :func:`ensure_compact` estimates the ``uint32`` volume's size from atlas
  metadata alone and refuses atlases above :func:`atlas_max_bytes`
  (``$PIXELMAP_ATLAS_MAX_BYTES``, default 1.5 GB — enough for the pre-baked
  rat atlas, too small for a 10 µm atlas) with :class:`AtlasTooLargeError`.
  The guard costs nothing once the compact file already exists.
"""

from __future__ import annotations

import functools
import gc
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_atlasapi.config import get_brainglobe_dir
from brainglobe_atlasapi.descriptors import V3_ANNOTATION_NAME, V3_ATLAS_ROOTDIR
from brainglobe_atlasapi.list_atlases import (
    get_all_atlases_lastversions,
    get_downloaded_atlases,
)

from pixelmap.anatomy._registry_snapshot import REGISTRY_SNAPSHOT

_DEFAULT_ATLAS = "allen_mouse_25um"

# Slab size (planes along array axis 0) used everywhere we walk a full atlas
# volume instead of materialising it: big enough to amortise per-slab
# overhead, small enough that a slab of even a 10 µm atlas is a few MB.
_CHUNK_PLANES = 24

#: uint16 can only address 65536 distinct labels (indices 0..65535); no real
#: atlas is anywhere close (the Allen CCF has ~1,300 structures), so this is
#: purely a sanity bound.
_MAX_COMPACT_IDS = 65536

_DEFAULT_MAX_ATLAS_BYTES = 1_500_000_000


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

    Construction fetches only the manifest and metadata (a few hundred KB);
    the annotation array is pulled from S3 later, when something first reads
    ``atlas.annotation``.  Either step is free once the data is local, but
    neither is free on a cold cache — see :func:`ensure_downloaded` for
    getting that cost off the event loop.

    The cache is bounded so a server that sees many atlases doesn't hold every
    annotation volume it has ever decoded.  Nothing else may keep a strong
    reference to an atlas object, or that bound stops meaning anything (see
    :func:`_region_info_from_id`).
    """
    return BrainGlobeAtlas(name, check_latest=False)


def ensure_downloaded(name: str = _DEFAULT_ATLAS) -> None:
    """Materialise ``name``'s annotation, downloading it if necessary.

    The one call in this module that can block for seconds on a cold cache, in
    a single place so callers can push it onto a worker thread — which the GUI
    does, because on the server the calling thread also serves every other
    session.  A no-op once the atlas is local.
    """
    _ = get_atlas(name).annotation


def ensure_ready(name: str = _DEFAULT_ATLAS) -> None:
    """Make ``name`` fully ready to render: chunks downloaded, compact cache built.

    This is the call the GUI should push onto a worker thread before its
    first render of an atlas — it supersedes :func:`ensure_downloaded` for
    that purpose, because "downloaded" alone no longer implies "cheap to
    read": the one-time compact-cache build (see :func:`ensure_compact`) also
    has to happen somewhere, and it must not happen inline on the event loop.
    A no-op once the compact cache already exists.
    """
    ensure_compact(name)


def is_ready(name: str = _DEFAULT_ATLAS) -> bool:
    """True if ``name`` can be read via :func:`canonical_annotation` for free.

    Stronger than :func:`is_downloaded`: the annotation chunks being local is
    not enough on its own any more, because the first read of an atlas also
    has to build its compact cache (see :func:`ensure_compact`) — a one-time
    cost that must run off the event loop. Checks :func:`is_downloaded` first
    so an atlas that was never downloaded doesn't pay for a ``get_atlas``
    call just to answer "not ready".
    """
    if not is_downloaded(name):
        return False
    try:
        return _compact_paths(name).labels.exists()
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def list_atlases() -> list[str]:
    """Every atlas in the brainglobe registry, not just the downloaded ones.

    Answers without touching the network, in three steps: the on-disk registry
    cache brainglobe maintains, then a snapshot bundled with PixelMap, and only
    as a last resort a live fetch — which happens in a daemon thread whose
    result the *next* caller picks up.

    That ordering is deliberate.  This runs while a GUI session is being built,
    which on the deployed server is on the Bokeh event loop, and brainglobe's
    registry fetch is a ``requests.get`` with no timeout: an outage at the host
    would otherwise hang the process for every connected user rather than cost
    one stale dropdown.  Cached for the life of the process — the registry
    changes a few times a year, and the background refresh clears the cache
    when it actually lands something new.  Callers get the memoised list
    itself, so treat it as read-only.
    """
    cached = _registry_from_disk()
    if cached:
        return cached
    _refresh_registry_in_background()
    return sorted(REGISTRY_SNAPSHOT)


def _registry_from_disk() -> list[str] | None:
    """Atlas names from brainglobe's on-disk registry cache, if it has one."""
    try:
        from brainglobe_atlasapi import config, utils

        cache_path = _registry_cache_path(config.get_brainglobe_dir())
        if cache_path.exists():
            data = utils.conf_from_file(cache_path)
            return sorted(data["atlases"].keys())
    except Exception:
        pass
    return None


def _registry_cache_path(brainglobe_dir: Path) -> Path:
    """Where brainglobe caches ``last_versions.conf``."""
    return brainglobe_dir / "brainglobe-atlasapi" / V3_ATLAS_ROOTDIR / "last_versions.conf"


#: Guards against piling up refresh threads when the host is unreachable and
#: every new session takes the snapshot path.
_registry_refresh_lock = threading.Lock()
_registry_refresh_running = False


def _refresh_registry_in_background() -> None:
    """Fetch the registry off-thread, so a later session gets the live list.

    Fire-and-forget: the fetch writes brainglobe's on-disk cache as a side
    effect, and we drop :func:`list_atlases`'s memo so the next caller reads
    it.  Failures are silent by design — the snapshot already answered.
    """
    global _registry_refresh_running
    with _registry_refresh_lock:
        if _registry_refresh_running:
            return
        _registry_refresh_running = True

    def _run():
        global _registry_refresh_running
        try:
            get_all_atlases_lastversions()
            if _registry_from_disk():
                list_atlases.cache_clear()
        except Exception:
            pass
        finally:
            with _registry_refresh_lock:
                _registry_refresh_running = False

    threading.Thread(
        target=_run, name="pixelmap-atlas-registry-refresh", daemon=True
    ).start()


# brainglobe orientation codes (e.g. "asr") spell the (0,0,0) origin corner:
# one letter per array axis, naming the anatomical side the axis starts from.
_ORIGIN_WORDS = {
    "a": "anterior", "p": "posterior",
    "s": "superior", "i": "inferior",
    "l": "left", "r": "right",
}


def is_downloaded(name: str = _DEFAULT_ATLAS) -> bool:
    """True if the atlas's annotation is on disk, so reading it won't download.

    Lets the GUI fetch an atlas's origin, extent or region labels only when
    that's free — picking an un-downloaded atlas from a dropdown should not
    kick off a download just to label the coordinate space.

    The atlas directory appears as soon as the (tiny) manifest lands, while the
    annotation is still remote, so the directory alone doesn't settle it: we
    also check that the OME-Zarr chunks are cached, which is what the callers
    actually care about.  Purely local — no network, safe on the event loop.
    """
    if name not in get_downloaded_atlases():
        return False
    try:
        return _annotation_is_cached(get_atlas(name))
    except Exception:
        # Metadata unreadable, or the manifest is there but incomplete: treat
        # it as not-downloaded so callers take the "this will cost you" path.
        return False


def _annotation_is_cached(atlas) -> bool:
    """True if the lazily-fetched annotation chunks are already local.

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

    ``ann`` (from :func:`canonical_annotation`) holds compact label *indices*,
    not atlas ids, so ``ac_ids`` is translated through :func:`label_ids`
    first.  The volume is then scanned in axis-0 slabs rather than with one
    ``np.isin`` over the whole array, so this never materialises a
    full-volume boolean temporary (up to ~1.2 GB for a 10 µm atlas).
    """
    atlas = get_atlas(name)
    ac_ids = [int(s["id"]) for s in atlas.structures.values()
              if "anterior" in s["name"].lower() and "commis" in s["name"].lower()]
    if not ac_ids:
        return None
    ann, res = canonical_annotation(name)  # (AP, DV, ML), compact indices
    lut = label_ids(name)
    ac_idx = np.flatnonzero(np.isin(lut, ac_ids))
    if ac_idx.size == 0:
        return None

    ap_parts, dv_parts, ml_parts = [], [], []
    n_ap = ann.shape[0]
    for start in range(0, n_ap, _CHUNK_PLANES):
        end = min(start + _CHUNK_PLANES, n_ap)
        slab = np.asarray(ann[start:end])
        mask = np.isin(slab, ac_idx)
        if not mask.any():
            continue
        a, d, m = np.nonzero(mask)
        ap_parts.append(a + start)
        dv_parts.append(d)
        ml_parts.append(m)

    if not ap_parts:
        return None
    ap = np.concatenate(ap_parts)
    dv = np.concatenate(dv_parts)
    ml = np.concatenate(ml_parts)

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

    brainglobe publishes ``shape`` from the atlas manifest, which is the cheap
    way to ask: reading ``annotation.shape`` instead would pull the entire
    array from S3.  Test doubles only carry ``annotation``, so fall back to
    that.
    """
    shape = getattr(atlas, "shape", None)
    if shape is not None:
        return tuple(int(s) for s in shape)
    return tuple(int(s) for s in atlas.annotation.shape)


class AtlasTooLargeError(RuntimeError):
    """Raised when an atlas's annotation would exceed :func:`atlas_max_bytes`.

    Raised before anything expensive happens — the estimate comes from
    metadata alone (see :func:`_atlas_shape`) — so it is cheap to hit
    repeatedly (e.g. every time the GUI dropdown offers a 10 µm atlas) and
    safe to surface directly to the user.
    """


def atlas_cache_dir() -> Path:
    """Where PixelMap writes its compact per-atlas volumes.

    ``$PIXELMAP_ATLAS_CACHE_DIR`` if set (tests point this at a tmp dir so
    nothing ever lands under a real home directory); otherwise a
    ``pixelmap_cache`` subdirectory of brainglobe's own data directory, so
    the compact cache lives alongside the atlases it was built from and
    survives container restarts the same way they do.
    """
    override = os.environ.get("PIXELMAP_ATLAS_CACHE_DIR")
    if override:
        return Path(override)
    try:
        brainglobe_dir = get_brainglobe_dir()
    except Exception:
        brainglobe_dir = Path.home() / ".brainglobe"
    return Path(brainglobe_dir) / "pixelmap_cache"


def atlas_max_bytes() -> int:
    """Largest ``uint32`` annotation volume :func:`ensure_compact` will build.

    ``$PIXELMAP_ATLAS_MAX_BYTES`` if set (and a valid positive integer),
    else 1.5 GB — comfortably above the pre-baked ``whs_sd_rat_39um``
    (~1.07 GB) and well below any 10 µm atlas (~4.8 GB).
    """
    raw = os.environ.get("PIXELMAP_ATLAS_MAX_BYTES")
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return _DEFAULT_MAX_ATLAS_BYTES


@dataclass(frozen=True)
class _CompactPaths:
    """Where one atlas+version's compact files live."""

    dir: Path
    labels: Path
    lut: Path
    meta: Path


def _atlas_version(atlas) -> str:
    """The atlas's version string for cache namespacing, or ``"unknown"``."""
    metadata = getattr(atlas, "metadata", None)
    if isinstance(metadata, dict):
        version = metadata.get("version")
        if version:
            return str(version)
    return "unknown"


def _compact_paths(atlas_name: str) -> _CompactPaths:
    atlas = get_atlas(atlas_name)
    out_dir = atlas_cache_dir() / atlas_name / _atlas_version(atlas)
    return _CompactPaths(
        dir=out_dir,
        labels=out_dir / "labels_u16.npy",
        lut=out_dir / "lut_u32.npy",
        meta=out_dir / "meta.json",
    )


def _open_annotation_dask(atlas):
    """The atlas's annotation as a lazy dask array, chunk-downloaded but
    never computed as a whole.

    Mirrors ``brainglobe_atlasapi.core.Atlas.annotation``'s own
    "download the pyramid level's chunks if they aren't local yet" logic
    (see also :func:`_annotation_is_cached`), but stops short of that
    property's final ``.data.compute()`` — which is exactly the full
    ``uint32`` materialisation :func:`ensure_compact` exists to avoid.  The
    returned array can be sliced along axis 0 and only those slabs are ever
    computed (:func:`_annotation_source` does that).

    This reaches into brainglobe's private attributes (``_annotation_pyramid
    _level``, ``fs``, ``root_dir``) because there is no public "give me a
    lazy handle" API. If a future brainglobe release changes that shape,
    :func:`_annotation_source` falls back to the safe (but memory-heavy)
    ``atlas.annotation`` path instead of failing outright.
    """
    import ngff_zarr as nz
    from brainglobe_atlasapi.descriptors import remote_url_s3
    from fsspec.callbacks import TqdmCallback

    annotation_location = atlas.metadata["annotation_set"]["location"][1:]
    annotation_path = Path(atlas.root_dir) / annotation_location / V3_ANNOTATION_NAME
    multiscale = nz.from_ngff_zarr(annotation_path)
    level = atlas._annotation_pyramid_level
    dataset_path = multiscale.metadata.datasets[level].path
    resolution_path = annotation_path / dataset_path

    if not (resolution_path / "c").exists():
        remote_path = remote_url_s3.format(
            f"{annotation_location}/{V3_ANNOTATION_NAME}/{dataset_path}/"
        )
        atlas.fs.get(remote_path, resolution_path, recursive=True, callback=TqdmCallback())

    return multiscale.images[level].data


def _annotation_source(atlas):
    """An axis-0-sliceable source for ``atlas``'s annotation.

    For a real ``BrainGlobeAtlas`` this is a lazy dask array (via
    :func:`_open_annotation_dask`): slicing it and converting the slice with
    ``np.asarray`` computes only that slab, so the full volume is never
    resident.  Test doubles carry a plain numpy ``annotation`` array with
    none of brainglobe's private plumbing — detected by the absence of
    ``fs``/``_annotation_pyramid_level`` — and are returned as-is; slicing a
    numpy array is already a cheap view.
    """
    if hasattr(atlas, "fs") and hasattr(atlas, "_annotation_pyramid_level"):
        try:
            return _open_annotation_dask(atlas)
        except Exception:
            pass  # brainglobe internals moved; fall back below.
    return np.asarray(atlas.annotation)


def ensure_compact(atlas_name: str) -> Path:
    """Build (once per atlas+version) the compact ``uint16`` label volume.

    Returns the directory containing ``labels_u16.npy`` (compact indices,
    same shape/orientation as the atlas's native annotation),
    ``lut_u32.npy`` (index -> atlas id, with index 0 always mapping to id 0
    so every ``!= 0`` "inside the brain" check downstream keeps its
    meaning), and ``meta.json``. A no-op — just a file-existence check — once
    built, which is what makes it safe to call from
    :func:`canonical_annotation` on every render.

    The size guard (:func:`atlas_max_bytes`) runs before any source data is
    touched, from metadata alone.  The build itself reads the source in
    axis-0 slabs (see :func:`_annotation_source`) so at most one slab's worth
    of the native ``uint32`` volume is ever resident — for a real atlas nothing
    beyond the on-disk OME-Zarr chunks is ever fetched into RAM at once.
    """
    paths = _compact_paths(atlas_name)
    if paths.labels.exists() and paths.lut.exists() and paths.meta.exists():
        return paths.dir

    atlas = get_atlas(atlas_name)
    shape = _atlas_shape(atlas)
    estimated_bytes = int(np.prod(shape)) * 4  # native annotation is uint32
    limit = atlas_max_bytes()
    if estimated_bytes > limit:
        raise AtlasTooLargeError(
            f"Atlas {atlas_name!r} would need ~{estimated_bytes / 1e9:.2f} GB "
            f"to load as uint32 (limit {limit / 1e9:.2f} GB). Raise the "
            "limit with the PIXELMAP_ATLAS_MAX_BYTES environment variable "
            "if you really want to load it."
        )

    paths.dir.mkdir(parents=True, exist_ok=True)
    source = _annotation_source(atlas)
    n = int(source.shape[0])

    try:
        # Pass 1: the sorted set of ids present, with 0 forced to index 0
        # (present even if the volume happens to have no background voxels,
        # so the invariant holds unconditionally).
        ids = np.array([0], dtype=np.int64)
        for start in range(0, n, _CHUNK_PLANES):
            end = min(start + _CHUNK_PLANES, n)
            slab = np.asarray(source[start:end])
            ids = np.union1d(ids, np.unique(slab))

        if ids.size > _MAX_COMPACT_IDS:
            raise AtlasTooLargeError(
                f"Atlas {atlas_name!r} has {ids.size} distinct region ids, "
                f"more than uint16 can compactly address ({_MAX_COMPACT_IDS})."
            )

        # Pass 2: write compact indices, still slab by slab.
        tmp_labels = paths.labels.with_suffix(".npy.tmp")
        labels_out = np.lib.format.open_memmap(
            tmp_labels, mode="w+", dtype=np.uint16, shape=tuple(shape)
        )
        try:
            for start in range(0, n, _CHUNK_PLANES):
                end = min(start + _CHUNK_PLANES, n)
                slab = np.asarray(source[start:end])
                labels_out[start:end] = np.searchsorted(ids, slab).astype(np.uint16)
            labels_out.flush()
        finally:
            del labels_out
        os.replace(tmp_labels, paths.labels)

        tmp_lut = paths.lut.with_suffix(".npy.tmp")
        # np.save appends ".npy" to a *string* path that doesn't already end
        # in it (so "lut_u32.npy.tmp" would silently become
        # "lut_u32.npy.tmp.npy") — write through an open file object instead,
        # which np.save uses verbatim.
        with open(tmp_lut, "wb") as f:
            np.save(f, ids.astype(np.uint32))
        os.replace(tmp_lut, paths.lut)

        tmp_meta = paths.meta.with_suffix(".json.tmp")
        meta = {
            "shape": [int(s) for s in shape],
            "orientation": str(getattr(atlas, "orientation", "")),
            "resolution": [float(r) for r in getattr(atlas, "resolution", ())],
            "source_dtype": "uint32",
            "n_ids": int(ids.size),
        }
        tmp_meta.write_text(json.dumps(meta))
        os.replace(tmp_meta, paths.meta)
    except BaseException:
        # Never leave a half-built directory that a later call would trust.
        for stray in (paths.labels.with_suffix(".npy.tmp"), paths.lut.with_suffix(".npy.tmp"),
                      paths.meta.with_suffix(".json.tmp")):
            stray.unlink(missing_ok=True)
        raise
    finally:
        # Release brainglobe's own copy, if reading it (directly or via the
        # dask fallback) populated it — the compact memmap is the resident
        # copy from here on.
        if getattr(atlas, "_annotation", None) is not None:
            atlas._annotation = None
        del source
        gc.collect()

    return paths.dir


@functools.lru_cache(maxsize=4)
def label_ids(atlas_name: str) -> np.ndarray:
    """The compact-index -> atlas-id lookup table for ``atlas_name``.

    Small (one ``uint32`` per distinct region — a few KB even for the Allen
    CCF), so caching a handful of these costs nothing worth bounding tightly.
    """
    out_dir = ensure_compact(atlas_name)
    return np.load(out_dir / "lut_u32.npy")


def anatomical_axes(atlas) -> dict[str, _AnatAxis]:
    """Map ``"AP"``/``"DV"``/``"ML"`` to their place in the native array.

    Read from ``atlas.orientation`` (e.g. ``"asr"``) so coordinate lookups
    work for any brainglobe orientation, not just Allen's. See
    :func:`canonical_annotation` for how this is applied.

    Metadata only — deliberately never touches ``atlas.annotation``, so
    callers that just want the volume's extent don't pay for a download.
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


@functools.lru_cache(maxsize=1)
def canonical_annotation(atlas_name: str):
    """Return ``(labels, resolution)`` reoriented to canonical ``(AP, DV, ML)``.

    The canonical layout is brainglobe ``"asr"``: axis 0 = AP (anterior→posterior),
    axis 1 = DV (dorsal→ventral), axis 2 = ML (right→left), with µm measured from
    the anterior-superior-right corner. The rest of :mod:`pixelmap.anatomy`
    assumes this layout, so funnelling every atlas through here is what lets
    non-Allen orientations work. For an already-``asr`` atlas this is a no-op
    (identity transpose, no flips), so Allen behavior is unchanged.

    ``labels`` holds *compact indices* into :func:`label_ids`, not atlas ids
    — see :func:`ensure_compact`. Index 0 always means atlas id 0 (outside
    the brain / undefined), so any existing ``!= 0`` check keeps working
    unchanged; anything that needs the real atlas id must map through
    ``label_ids(atlas_name)`` explicitly (e.g. :func:`lookup_regions`).

    ``maxsize=1``: only one atlas's volume is ever resident at a time.
    Switching atlases drops the previous one (its backing memmap is closed
    once nothing references it) rather than accumulating — see the module
    docstring's "Memory strategy" section.
    """
    atlas = get_atlas(atlas_name)
    axes = anatomical_axes(atlas)
    out_dir = ensure_compact(atlas_name)
    labels = np.load(out_dir / "labels_u16.npy", mmap_mode="r")
    order = (axes["AP"].array_axis, axes["DV"].array_axis, axes["ML"].array_axis)
    arr = np.transpose(labels, order)
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
    lut = label_ids(atlas_name)  # compact index -> atlas id

    results: list[RegionInfo | None] = []
    for ap, dv, ml in zip(ap_idx, dv_idx, ml_idx):
        if not (0 <= ap < shape[0] and 0 <= dv < shape[1] and 0 <= ml < shape[2]):
            results.append(None)
            continue
        region_idx = int(annotation[ap, dv, ml])
        if region_idx == 0:  # outside-brain or undefined
            results.append(None)
            continue
        region_id = int(lut[region_idx])
        results.append(_region_info_from_id(atlas_name, region_id))
    return results


@functools.lru_cache(maxsize=4096)
def _region_info_from_id(atlas_name: str, region_id: int) -> RegionInfo | None:
    """Resolve a region integer label to acronym/name/rgb. Cached per-atlas.

    Keyed on the atlas *name*, never the atlas object.  An ``lru_cache`` holds
    its keys alive, so taking the object here pinned every atlas ever looked up
    for the life of the process — each one keeping its annotation volume
    resident (294 MB for allen_mouse_25um, 1.0 GB for whs_sd_rat_39um).  The
    4096-entry cap was no protection: one probe traverses ~23 regions, so the
    whole registry fits underneath it and nothing was ever evicted.

    Worse, it turned :func:`get_atlas`'s eviction into a leak.  With a 5th
    atlas in rotation every lookup rebuilt an evicted atlas, loaded a fresh
    copy of its annotation, and pinned that too — so memory grew with traffic
    rather than with the number of atlases.  Going through the name means the
    only strong references live in ``get_atlas``'s own bounded cache.
    """
    try:
        entry = get_atlas(atlas_name).structures[region_id]
    except KeyError:
        return None
    rgb = tuple(int(c) for c in entry.get("rgb_triplet", (128, 128, 128)))
    return RegionInfo(
        atlas_id=region_id,
        acronym=str(entry.get("acronym", f"id{region_id}")),
        name=str(entry.get("name", "")),
        rgb=rgb,  # type: ignore[arg-type]
    )


def clear_caches() -> None:
    """Drop every module-level cache this module keeps.

    Atlas objects (:func:`get_atlas`), the one resident volume
    (:func:`canonical_annotation`), lookup tables (:func:`label_ids`),
    derived origins (:func:`derive_origin_from_ac`) and resolved region info
    (:func:`_region_info_from_id`). Tests use this in an autouse fixture so a
    fake atlas from one test never leaks into the next; production code has
    no reason to call it — the bounded caches above are what keep memory flat
    over a long-running server's lifetime.

    Tolerant of any of these names having been monkeypatched to something
    without ``cache_clear`` (tests do this, e.g. replacing ``get_atlas``
    outright) — skips it rather than raising, so fixture teardown ordering
    relative to ``monkeypatch``'s own undo can't turn this into a spurious
    failure.
    """
    for fn in (get_atlas, canonical_annotation, label_ids,
               derive_origin_from_ac, _region_info_from_id):
        cache_clear = getattr(fn, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()
    gc.collect()
