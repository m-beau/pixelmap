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

* **Pose** — three independent angles around fixed atlas axes plus a spin
  around the probe's own long axis:

  * ``tip_atlas`` — atlas position of the shank-0 lowest electrode.
  * ``pitch_deg`` — *AP tilt*. Rotation around the world ML axis. ``0`` is
    vertical; positive tips the top of the probe forward (+AP).
  * ``yaw_deg`` — *ML tilt*. Rotation around the world AP axis. ``0`` is
    vertical; positive tips the top of the probe to the right (+ML).
  * ``shank_orientation_deg`` — *shank line direction*. Rotation of the
    shank line around the probe's own long axis. ``0`` = shanks along +ML
    (the "side-on" insertion), ``90`` = along +AP (anterior), ``180`` =
    along -ML, ``270`` = along -AP. Only matters for multi-shank probes.

Order of composition: spin (in the probe's local frame) → pitch → yaw, so
that pitch and yaw remain the literal AP/ML tilts of the stereotax arm
regardless of how the shank line is oriented around the probe shaft.
"""

from __future__ import annotations

import numpy as np


def bregma_to_atlas_um(
    coords_bregma: tuple[float, float, float],
    *,
    bregma_um: tuple[float, float, float],
    dv_squish: float = 1.0,
    tilt_deg: float = 0.0,
) -> tuple[float, float, float]:
    """Map a bregma-relative stereotaxic coordinate to canonical atlas µm.

    This is the optional bregma-coordinate mode: instead of absolute atlas
    µm, the user gives a position relative to bregma, and we fold in the
    atlas's bregma location, DV "squish" and nose-up tilt (see
    :func:`pixelmap.anatomy.atlas.reference_params`).

    Args:
        coords_bregma: ``(ap, ml, dv)`` µm from bregma, with ``ap`` positive
            anterior, ``ml`` positive left (matching the atlas +ML direction),
            ``dv`` positive ventral (deeper).
        bregma_um: bregma position in canonical atlas ``(AP, ML, DV)`` µm.
        dv_squish: DV scale — atlas DV displacement = real DV / ``dv_squish``
            (corrects the Allen CCF's dorsoventral compression). ``1.0`` = none.
        tilt_deg: nose-up tilt of the stereotaxic frame about the ML axis,
            applied before placing the point. ``0`` = none.

    Returns:
        ``(AP, ML, DV)`` in canonical atlas µm, ready for
        :func:`probe_to_atlas` / region lookup.
    """
    ap_b, ml_b, dv_b = (float(c) for c in coords_bregma)
    b_ap, b_ml, b_dv = (float(c) for c in bregma_um)
    dv_s = dv_b / dv_squish if dv_squish else dv_b
    theta = np.deg2rad(tilt_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # Rotate the (AP, DV) offset nose-up about ML, then place relative to bregma.
    ap_r = ap_b * cos_t + dv_s * sin_t
    dv_r = -ap_b * sin_t + dv_s * cos_t
    return (b_ap - ap_r, b_ml + ml_b, b_dv + dv_r)


def atlas_to_bregma_um(
    coords_atlas: tuple[float, float, float],
    *,
    bregma_um: tuple[float, float, float],
    dv_squish: float = 1.0,
    tilt_deg: float = 0.0,
) -> tuple[float, float, float]:
    """Exact inverse of :func:`bregma_to_atlas_um`.

    Needed whenever a coordinate computed in the atlas frame has to be shown
    back to the user in whichever frame they are typing in.

    Args:
        coords_atlas: ``(AP, ML, DV)`` in canonical atlas µm.
        bregma_um: bregma position in canonical atlas ``(AP, ML, DV)`` µm.
        dv_squish: as in :func:`bregma_to_atlas_um`.
        tilt_deg: as in :func:`bregma_to_atlas_um`.

    Returns:
        ``(ap, ml, dv)`` µm from bregma, using the same sign conventions as
        :func:`bregma_to_atlas_um` takes.
    """
    a_ap, a_ml, a_dv = (float(c) for c in coords_atlas)
    b_ap, b_ml, b_dv = (float(c) for c in bregma_um)
    # Undo the placement, then un-rotate the nose-up tilt.
    ap_r, dv_r = b_ap - a_ap, a_dv - b_dv
    theta = np.deg2rad(tilt_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    ap_b = ap_r * cos_t - dv_r * sin_t
    dv_s = ap_r * sin_t + dv_r * cos_t
    # Mirror the forward's truthiness guard: dv_squish == 0 passes through
    # unscaled there, so it must here too or the round trip breaks at 0.
    dv_b = dv_s * dv_squish if dv_squish else dv_s
    return (ap_b, a_ml - b_ml, dv_b)


def _R_pitch(pitch_rad: float) -> np.ndarray:
    """AP tilt — rotation around world ML axis. ``probe-y`` rotates -DV → +AP."""
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    return np.array([
        [cp, 0.0, -sp],
        [0.0, 1.0, 0.0],
        [sp, 0.0,  cp],
    ])


def _R_yaw(yaw_rad: float) -> np.ndarray:
    """ML tilt — rotation around world AP axis. ``probe-y`` rotates -DV → +ML."""
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, cy, -sy],
        [0.0, sy,  cy],
    ])


# Probe-local "up the shank" direction before any tilt: a vertical probe's
# shank runs from the tip toward -DV.
_PROBE_Y_LOCAL = np.array([0.0, 0.0, -1.0])


def _pose_rotation(pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """World-axis tilt composition: pitch around ML first, then yaw around AP.

    Pitch is applied first so that it stays the literal AP tilt of the
    stereotax arm regardless of yaw.
    """
    return _R_yaw(np.deg2rad(yaw_deg)) @ _R_pitch(np.deg2rad(pitch_deg))


def probe_axis(pitch_deg: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    """Unit vector pointing from the tip *up* the shank, in atlas (AP, ML, DV).

    Equals ``(sin p, sin y·cos p, -cos y·cos p)``. Independent of
    ``shank_orientation_deg`` — the spin rotates *around* this axis, so it
    can never change it.

    This is the direction the probe travels along when it is lowered, which
    makes it the basis for converting between an insertion point and a tip
    (:func:`insertion_to_tip`, :func:`tip_to_insertion`).
    """
    return _pose_rotation(pitch_deg, yaw_deg) @ _PROBE_Y_LOCAL


# Depth reference
# ---------------
# ``tip_atlas`` everywhere in this module is the *lowest electrode* (the
# ``yp = 0`` reference), because that is what the electrode geometry is
# measured from. Insertion depth, however, is naturally read off a
# manipulator as travel of the *physical shank tip*, which sits
# ``tip_length_um`` of inactive silicon below the lowest electrode
# (209 µm on NP 1.0, 206 µm on NP 2.0 — see ``tip_length_um`` in
# ``PROBE_FEATURES``). The three functions below take that offset so callers
# can work in manipulator depth; pass ``0.0`` to measure to the lowest
# electrode instead.


def tip_to_insertion(
    tip_atlas: np.ndarray | tuple[float, float, float],
    depth_um: float,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    tip_length_um: float = 0.0,
) -> np.ndarray:
    """Insertion point for a probe whose shank tip is ``depth_um`` deep.

    Args:
        tip_atlas: atlas ``(AP, ML, DV)`` µm of the *lowest electrode*.
        depth_um: travel along the shank from the brain surface to the
            *physical shank tip*.
        pitch_deg, yaw_deg: pose — see module docstring.
        tip_length_um: physical-tip-to-lowest-electrode offset. ``0.0``
            measures ``depth_um`` to the lowest electrode instead.

    Returns:
        Atlas ``(AP, ML, DV)`` µm where the shank crosses the brain surface.
    """
    tip = np.asarray(tip_atlas, dtype=float).reshape(3)
    reach = float(depth_um) - float(tip_length_um)
    return tip + reach * probe_axis(pitch_deg, yaw_deg)


def insertion_to_tip(
    insertion_atlas: np.ndarray | tuple[float, float, float],
    depth_um: float,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    tip_length_um: float = 0.0,
) -> np.ndarray:
    """Lowest-electrode position for a probe lowered ``depth_um`` into the brain.

    Inverse of :func:`tip_to_insertion`. ``depth_um`` is travel *along the
    shank* (what the manipulator advances), not a vertical DV drop, so this
    stays well defined at any pitch/yaw — including ±90°, where a
    vertical-depth parameterisation would divide by zero.

    A ``depth_um`` smaller than ``tip_length_um`` is meaningful, not an
    error: the shank tip has entered the brain but the lowest electrode has
    not yet, so the returned point sits above the surface.
    """
    entry = np.asarray(insertion_atlas, dtype=float).reshape(3)
    reach = float(depth_um) - float(tip_length_um)
    return entry - reach * probe_axis(pitch_deg, yaw_deg)


def insertion_depth_um(
    tip_atlas: np.ndarray | tuple[float, float, float],
    insertion_atlas: np.ndarray | tuple[float, float, float],
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    tip_length_um: float = 0.0,
) -> float:
    """Signed along-shank travel from the insertion point to the shank tip.

    Positive when the tip is below the insertion point (the normal case).
    **Negative** when the whole shank sits above it — i.e. the probe has not
    reached that point yet — which is why this projects onto the axis rather
    than taking a norm: a norm would report a healthy positive depth for a
    probe that never entered the tissue.
    """
    tip = np.asarray(tip_atlas, dtype=float).reshape(3)
    entry = np.asarray(insertion_atlas, dtype=float).reshape(3)
    reach = float(np.dot(entry - tip, probe_axis(pitch_deg, yaw_deg)))
    return reach + float(tip_length_um)


def probe_to_atlas(
    electrode_xy: np.ndarray,
    tip_atlas: np.ndarray | tuple[float, float, float],
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    shank_orientation_deg: float = 0.0,
) -> np.ndarray:
    """Map probe-local electrode positions to atlas coordinates.

    Args:
        electrode_xy: shape ``(N, 2)`` array of probe-local ``(xp, yp)`` in µm.
        tip_atlas: length-3 ``(AP, ML, DV)`` position of the shank-0 lowest
            electrode in atlas µm.
        pitch_deg: AP tilt in degrees (see module docstring).
        yaw_deg: ML tilt in degrees (see module docstring).
        shank_orientation_deg: shank line direction in degrees (see module
            docstring). At ``0``, +xp points along atlas +ML; at ``90``,
            +xp points along atlas +AP.

    Returns:
        ``(N, 3)`` array of atlas ``(AP, ML, DV)`` positions in µm.
    """
    electrode_xy = np.asarray(electrode_xy, dtype=float)
    if electrode_xy.ndim != 2 or electrode_xy.shape[1] != 2:
        raise ValueError(f"electrode_xy must be (N, 2); got {electrode_xy.shape}")
    tip = np.asarray(tip_atlas, dtype=float).reshape(3)

    spin = np.deg2rad(shank_orientation_deg)

    # Probe-local axes before any tilt:
    #   probe-y points from the tip up the shank → -DV (a vertical probe).
    #   probe-x lies along the shank line. Spin rotates it around probe-y, with
    #     0° giving +ML and +90° giving +AP (rotation is in the horizontal
    #     plane of the un-tilted probe).
    probe_x_local = np.array([np.sin(spin), np.cos(spin), 0.0])

    # World-axis tilts (pitch around world ML; yaw around world AP). Order:
    # pitch first, then yaw — this keeps pitch acting purely as AP tilt.
    R = _pose_rotation(pitch_deg, yaw_deg)
    probe_x = R @ probe_x_local
    probe_y = R @ _PROBE_Y_LOCAL  # == probe_axis(pitch_deg, yaw_deg)

    displacements = (
        np.outer(electrode_xy[:, 0], probe_x)
        + np.outer(electrode_xy[:, 1], probe_y)
    )
    return tip + displacements
