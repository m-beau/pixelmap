"""High-level region lookup: probe-local positions + pose → regions."""

from __future__ import annotations

import numpy as np

from pixelmap.anatomy.atlas import RegionInfo, canonical_annotation, lookup_regions
from pixelmap.anatomy.transform import AtlasWarp, ProbePlacement, probe_axis_up


def regions_for_positions(
    electrode_xy: np.ndarray,
    tip_atlas: np.ndarray | tuple[float, float, float],
    *,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    shank_orientation_deg: float = 0.0,
    atlas_name: str = "allen_mouse_25um",
    warp: AtlasWarp | None = None,
) -> list[RegionInfo | None]:
    """Resolve atlas regions for an array of probe-local electrode positions.

    Args:
        electrode_xy: ``(N, 2)`` probe-local ``(xp, yp)`` in µm.
        tip_atlas: ``(AP, ML, DV)`` of shank-0 lowest electrode in atlas µm.
        pitch_deg, yaw_deg, shank_orientation_deg: insertion pose — see
            :mod:`pixelmap.anatomy.transform`.
        atlas_name: brainglobe atlas identifier.
        warp: stereotaxic correction of the atlas (squish + tilt about
            bregma). When given, ``tip_atlas`` and the pose are in the
            *stereotaxic* frame, and the regions are those of the warped
            brain — the one the locator draws.

    Returns:
        Length-``N`` list; entries are ``None`` for positions outside the volume.
    """
    placement = ProbePlacement(
        tuple(float(c) for c in np.asarray(tip_atlas, dtype=float).reshape(3)),
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        shank_orientation_deg=shank_orientation_deg,
        warp=warp,
    )
    return lookup_regions(atlas_name, placement.to_atlas(electrode_xy))


def tip_depth_below_surface_um(
    tip_atlas: np.ndarray | tuple[float, float, float],
    *,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    atlas_name: str = "allen_mouse_25um",
    step_um: float | None = None,
    max_search_um: float = 30_000.0,
    warp: AtlasWarp | None = None,
) -> float | None:
    """Distance from the probe tip to the brain surface, *along the probe axis*.

    This is the quantity experimenters read off the micromanipulator: how far
    the tip has travelled since the shank broke the brain surface. It is a
    purely derived readout — the pose (tip + pitch + yaw) fixes it, so it
    cannot be used as an input (a surface-relative depth alone does not
    determine where the probe sits in the atlas).

    The shank axis is walked upward from the tip and the *outermost* crossing
    out of the annotated volume is taken as the surface, so an insertion that
    passes through a ventricle or between two structures still reports the
    depth below the true brain surface rather than below the first gap.

    Args:
        tip_atlas: ``(AP, ML, DV)`` of the tip in canonical atlas µm.
        pitch_deg, yaw_deg: insertion tilts — see
            :mod:`pixelmap.anatomy.transform`.
        atlas_name: brainglobe atlas identifier.
        step_um: sampling step along the axis. Defaults to the atlas's finest
            voxel size, which is the resolution the answer can have anyway.
        max_search_um: give up beyond this distance from the tip.
        warp: stereotaxic correction of the atlas (squish + tilt about
            bregma). When given, ``tip_atlas`` and the returned distance are
            in the *stereotaxic* frame — the brain as the locator draws it.

    Returns:
        Distance in µm, or ``None`` when it is undefined: the tip is outside
        the annotated volume (not in the brain), or the axis never leaves the
        brain within ``max_search_um``.
    """
    annotation, voxel_size = canonical_annotation(atlas_name)  # indexed (AP, DV, ML)
    tip = np.asarray(tip_atlas, dtype=float).reshape(3)
    direction = probe_axis_up(pitch_deg, yaw_deg)

    step = _sampling_step(step_um, voxel_size, warp)

    distances = np.arange(0.0, float(max_search_um) + step, step)
    coords = tip + distances[:, None] * direction  # (N, 3) in (AP, ML, DV)
    if warp is not None:
        coords = warp.to_atlas(coords)
    inside = _in_brain(annotation, voxel_size, coords)

    if not inside[0]:
        return None  # tip is not in the brain — "depth below surface" is meaningless
    last_inside = int(np.flatnonzero(inside)[-1])
    if last_inside == len(distances) - 1:
        return None  # never surfaced within the search range

    # The surface lies between the last in-brain sample and the first sample
    # past it; take the midpoint so the answer isn't biased by half a step.
    return float(distances[last_inside] + step / 2.0)


def surface_point_um(
    point_atlas: np.ndarray | tuple[float, float, float],
    *,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    atlas_name: str = "allen_mouse_25um",
    step_um: float | None = None,
    warp: AtlasWarp | None = None,
) -> tuple[float, float, float] | None:
    """Where a probe trajectory crosses the brain surface, as ``(AP, ML, DV)`` µm.

    The counterpart of :func:`tip_depth_below_surface_um` for when the point
    you have is *not* in the brain — typically an insertion point, which sits
    on or above the surface by construction. The trajectory is the line
    through ``point_atlas`` along the probe axis; the surface is its
    *outermost* (most dorsal along the shank) crossing out of the annotated
    volume, by the same rule and with the same half-step midpoint as the
    depth readout.

    The result depends only on the *line*, not on which of its points is
    passed in — the line is clipped to the volume's bounding box and sampled
    from where it enters the box. Casting from the tip or from the insertion
    point therefore gives the same answer.

    Args:
        point_atlas: any ``(AP, ML, DV)`` µm point on the trajectory.
        pitch_deg, yaw_deg: insertion tilts — see
            :mod:`pixelmap.anatomy.transform`.
        atlas_name: brainglobe atlas identifier.
        step_um: sampling step along the axis. Defaults to the atlas's finest
            voxel size, so a thin dorsal layer is never stepped over.
        warp: stereotaxic correction of the atlas (squish + tilt about
            bregma). When given, ``point_atlas`` and the result are in the
            *stereotaxic* frame, and the surface found is that of the warped
            brain — the one the locator draws.

    Returns:
        The surface crossing, or ``None`` if the trajectory never passes
        through a labelled voxel.
    """
    annotation, voxel_size = canonical_annotation(atlas_name)  # indexed (AP, DV, ML)
    p = np.asarray(point_atlas, dtype=float).reshape(3)
    direction = probe_axis_up(pitch_deg, yaw_deg)

    step = _sampling_step(step_um, voxel_size, warp)

    # The warp is affine, so the trajectory is still a straight line in the
    # atlas frame: ``p_a + t * d_a``, with ``t`` the distance travelled in the
    # frame the caller works in (``d_a`` is not unit length under a squish).
    if warp is None:
        p_a, d_a = p, direction
    else:
        p_a = warp.to_atlas(p)
        d_a = warp.to_atlas(p + direction) - p_a

    # Clip the infinite line to the slab of coordinates that round onto a
    # voxel, in coordinate order (AP, ML, DV). A fixed window around ``p``
    # would not do: an insertion point is routinely outside the volume, and
    # the answer would then depend on which point of the line was passed in.
    n_ap, n_dv, n_ml = annotation.shape
    res = np.asarray(voxel_size, dtype=float)[[0, 2, 1]]
    lo = -0.5 * res
    hi = (np.array([n_ap, n_ml, n_dv]) - 0.5) * res
    t_lo, t_hi = -np.inf, np.inf
    for i in range(3):
        if abs(d_a[i]) < 1e-12:  # line parallel to this pair of faces
            if not lo[i] <= p_a[i] < hi[i]:
                return None
            continue
        t0, t1 = (lo[i] - p_a[i]) / d_a[i], (hi[i] - p_a[i]) / d_a[i]
        t_lo, t_hi = max(t_lo, min(t0, t1)), min(t_hi, max(t0, t1))
    if t_hi < t_lo:
        return None  # the line misses the volume altogether

    # Sample upward from where the line enters the box, so the sample grid
    # (and hence the answer) is a property of the line alone.
    ts = np.arange(t_lo, t_hi + step, step)
    inside = _in_brain(annotation, voxel_size, p_a + ts[:, None] * d_a)
    hits = np.flatnonzero(inside)
    if hits.size == 0:
        return None
    t_surface = ts[hits[-1]] + step / 2.0
    return tuple(float(c) for c in p + t_surface * direction)


def _sampling_step(step_um, voxel_size, warp: AtlasWarp | None) -> float:
    """Step along the axis: the finest voxel, as seen from the caller's frame.

    A squish below 1 shrinks the brain in the stereotaxic frame, so a voxel
    spans less of the trajectory there; the step shrinks with it so a thin
    layer is still never stepped over.
    """
    if step_um:
        step = float(step_um)
    else:
        step = float(np.min(voxel_size))
        if warp is not None:
            step *= min(1.0, warp.ap_squish, warp.ml_squish, warp.dv_squish)
    if step <= 0:
        raise ValueError(f"step must be positive; got {step_um} (check the squish factors)")
    return step


def _in_brain(annotation, voxel_size, coords: np.ndarray) -> np.ndarray:
    """Whether each ``(AP, ML, DV)`` µm sample lands on a labelled voxel.

    Samples outside the volume count as outside the brain, matching
    :func:`~pixelmap.anatomy.atlas.lookup_regions`.
    """
    ap_idx = np.round(coords[:, 0] / voxel_size[0]).astype(int)
    dv_idx = np.round(coords[:, 2] / voxel_size[1]).astype(int)
    ml_idx = np.round(coords[:, 1] / voxel_size[2]).astype(int)

    n_ap, n_dv, n_ml = annotation.shape
    in_volume = (
        (ap_idx >= 0) & (ap_idx < n_ap)
        & (dv_idx >= 0) & (dv_idx < n_dv)
        & (ml_idx >= 0) & (ml_idx < n_ml)
    )
    inside = np.zeros(len(coords), dtype=bool)
    if in_volume.any():
        # One bounding-box read for the whole line: the samples are
        # collinear, so the box is thin even when the line is 30 mm long.
        inside[in_volume] = (
            annotation.gather(
                ap_idx[in_volume], dv_idx[in_volume], ml_idx[in_volume]
            )
            != 0
        )
    return inside
