"""High-level region lookup: probe-local positions + pose → regions."""

from __future__ import annotations

import numpy as np

from pixelmap.anatomy.atlas import RegionInfo, lookup_regions
from pixelmap.anatomy.transform import probe_to_atlas


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
