"""High-level region lookup: probe-local positions + pose → regions."""

from __future__ import annotations

import numpy as np

from pixelmap.anatomy.atlas import RegionInfo, canonical_annotation, lookup_regions
from pixelmap.anatomy.transform import probe_axis_up, probe_to_atlas


def regions_for_positions(
    electrode_xy: np.ndarray,
    tip_atlas: np.ndarray | tuple[float, float, float],
    *,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    shank_orientation_deg: float = 0.0,
    atlas_name: str = "allen_mouse_25um",
) -> list[RegionInfo | None]:
    """Resolve atlas regions for an array of probe-local electrode positions.

    Args:
        electrode_xy: ``(N, 2)`` probe-local ``(xp, yp)`` in µm.
        tip_atlas: ``(AP, ML, DV)`` of shank-0 lowest electrode in atlas µm.
        pitch_deg, yaw_deg, shank_orientation_deg: insertion pose — see
            :mod:`pixelmap.anatomy.transform`.
        atlas_name: brainglobe atlas identifier.

    Returns:
        Length-``N`` list; entries are ``None`` for positions outside the volume.
    """
    atlas_coords = probe_to_atlas(
        electrode_xy, tip_atlas,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        shank_orientation_deg=shank_orientation_deg,
    )
    return lookup_regions(atlas_name, atlas_coords)


def tip_depth_below_surface_um(
    tip_atlas: np.ndarray | tuple[float, float, float],
    *,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    atlas_name: str = "allen_mouse_25um",
    step_um: float | None = None,
    max_search_um: float = 30_000.0,
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

    Returns:
        Distance in µm, or ``None`` when it is undefined: the tip is outside
        the annotated volume (not in the brain), or the axis never leaves the
        brain within ``max_search_um``.
    """
    annotation, voxel_size = canonical_annotation(atlas_name)  # indexed (AP, DV, ML)
    tip = np.asarray(tip_atlas, dtype=float).reshape(3)
    direction = probe_axis_up(pitch_deg, yaw_deg)

    step = float(step_um) if step_um else float(np.min(voxel_size))
    if step <= 0:
        raise ValueError(f"step_um must be positive; got {step_um}")

    distances = np.arange(0.0, float(max_search_um) + step, step)
    coords = tip + distances[:, None] * direction  # (N, 3) in (AP, ML, DV)

    ap_idx = np.round(coords[:, 0] / voxel_size[0]).astype(int)
    dv_idx = np.round(coords[:, 2] / voxel_size[1]).astype(int)
    ml_idx = np.round(coords[:, 1] / voxel_size[2]).astype(int)

    n_ap, n_dv, n_ml = annotation.shape
    in_volume = (
        (ap_idx >= 0) & (ap_idx < n_ap)
        & (dv_idx >= 0) & (dv_idx < n_dv)
        & (ml_idx >= 0) & (ml_idx < n_ml)
    )
    inside = np.zeros(distances.shape, dtype=bool)
    inside[in_volume] = (
        annotation[ap_idx[in_volume], dv_idx[in_volume], ml_idx[in_volume]] != 0
    )

    if not inside[0]:
        return None  # tip is not in the brain — "depth below surface" is meaningless
    last_inside = int(np.flatnonzero(inside)[-1])
    if last_inside == len(distances) - 1:
        return None  # never surfaced within the search range

    # The surface lies between the last in-brain sample and the first sample
    # past it; take the midpoint so the answer isn't biased by half a step.
    return float(distances[last_inside] + step / 2.0)
