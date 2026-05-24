"""Probe-local → atlas coordinate transform.

Conventions used throughout this module:

* **Probe-local frame** (2D, µm):

  * ``xp`` runs along the shank-line direction. For multi-shank probes,
    shanks are placed at increasing ``xp`` (e.g. 0, 250, 500, 750 µm for a
    Neuropixels 2.0-4shank).
  * ``yp`` runs along the shank length, with ``yp = 0`` at the position of
    the *lowest electrode* (the user picks the tip coordinate to match
    that reference, so the math stays clean).

* **Atlas frame** (3D, µm): brainglobe's standard ``(AP, ML, DV)`` ordering
  for Allen CCF mouse. AP is anterior(+); ML is the lateral axis where +
  is to the right of the midline; DV is the depth axis, + meaning ventral
  (so going dorsal/up = decreasing DV).

* **Pose**:

  * ``tip_atlas`` — atlas position of the shank-0 lowest electrode.
  * ``yaw_deg`` — rotation about the DV (vertical) axis. ``0`` orients the
    shank-line along +ML, ``90`` along +AP, etc.
  * ``pitch_deg`` — tilt about the rotated probe-x axis. ``0`` is vertical
    (probe pointing straight down into the brain). Positive pitch tips the
    *top* of the probe toward +AP (anterior), so the probe leans forward.

Two angles (yaw + pitch) are sufficient for the vast majority of
Neuropixels insertions. Roll (rotation of the probe about its own long
axis) is intentionally omitted in v1 — it would shift electrode positions
by at most the probe thickness, which is well below the atlas resolution.
"""

from __future__ import annotations

import numpy as np


def probe_to_atlas(
    electrode_xy: np.ndarray,
    tip_atlas: np.ndarray | tuple[float, float, float],
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> np.ndarray:
    """Map probe-local electrode positions to atlas coordinates.

    Args:
        electrode_xy: shape ``(N, 2)`` array of probe-local ``(xp, yp)`` in µm.
        tip_atlas: length-3 ``(AP, ML, DV)`` position of the shank-0 lowest
            electrode in atlas µm.
        pitch_deg: pitch angle in degrees (see module docstring).
        yaw_deg: yaw angle in degrees (see module docstring).

    Returns:
        ``(N, 3)`` array of atlas ``(AP, ML, DV)`` positions in µm.
    """
    electrode_xy = np.asarray(electrode_xy, dtype=float)
    if electrode_xy.ndim != 2 or electrode_xy.shape[1] != 2:
        raise ValueError(f"electrode_xy must be (N, 2); got {electrode_xy.shape}")
    tip = np.asarray(tip_atlas, dtype=float).reshape(3)

    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    # Probe-x axis in atlas frame (yaw rotation, stays in the horizontal plane).
    #   yaw = 0 → +ML; yaw = 90° → +AP.
    probe_x = np.array([np.sin(yaw), np.cos(yaw), 0.0])

    # Probe-y axis: starts as -DV (vertical, going dorsal from the tip), then
    # pitch tilts it in the AP-DV plane around the (already-yawed) probe-x axis.
    # The "horizontal direction perpendicular to probe-x" is the axis we tilt
    # toward at +pitch — by convention pointing forward in atlas-AP at yaw=0.
    horizontal_forward = np.array([np.cos(yaw), -np.sin(yaw), 0.0])
    probe_y = np.array([
        np.sin(pitch) * horizontal_forward[0],
        np.sin(pitch) * horizontal_forward[1],
        -np.cos(pitch),
    ])

    displacements = (
        np.outer(electrode_xy[:, 0], probe_x)
        + np.outer(electrode_xy[:, 1], probe_y)
    )
    return tip + displacements
