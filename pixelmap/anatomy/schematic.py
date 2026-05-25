"""Locator schematic: a small brain silhouette with the probe drawn on it.

Two orthogonal silhouette views — sagittal (x = AP, y = DV) and coronal
(x = ML, y = DV) — projected from the atlas annotation, with the probe shank
trajectories overlaid, so the user can see at a glance roughly where in the
brain the current pose lands.

Like the rest of :mod:`pixelmap.anatomy`, this assumes brainglobe's
Allen-style ``(AP, DV, ML)`` annotation indexing.
"""

from __future__ import annotations

import functools

import numpy as np
from matplotlib.figure import Figure

from pixelmap.anatomy.atlas import get_atlas
from pixelmap.anatomy.transform import probe_to_atlas

_FILL = "#d7d7e0"
_EDGE = "#9a9aa8"
_PROBE = "#d62728"


@functools.lru_cache(maxsize=4)
def _silhouettes(atlas_name: str):
    """Brain silhouettes + µm axes for the sagittal & coronal views.

    Cached per atlas — depends only on the annotation volume, not the probe.
    The returned masks are already shaped ``(y, x)`` for ``contourf`` (y is DV
    in both views).
    """
    atlas = get_atlas(atlas_name)
    ann = atlas.annotation                            # (AP, DV, ML)
    res = np.asarray(atlas.resolution, dtype=float)   # (AP, DV, ML) µm/voxel
    inside = ann > 0
    sag = inside.any(axis=2).T   # (DV, AP): x = AP, y = DV
    cor = inside.any(axis=0)     # (DV, ML): x = ML, y = DV
    ap_um = np.arange(ann.shape[0]) * res[0]
    dv_um = np.arange(ann.shape[1]) * res[1]
    ml_um = np.arange(ann.shape[2]) * res[2]
    return ap_um, dv_um, ml_um, sag, cor


def render_locator(
    atlas_name: str,
    *,
    tip_atlas: tuple[float, float, float],
    pitch_deg: float,
    yaw_deg: float,
    shank_orientation_deg: float,
    shank_positions: dict[int, float],
    y_range: tuple[float, float],
) -> Figure:
    """Render the locator figure for the current insertion pose.

    Pose args mirror
    :func:`pixelmap.anatomy.visualization.compute_region_bands`. Returns a
    bare matplotlib :class:`~matplotlib.figure.Figure` for a Panel
    ``Matplotlib`` pane (no pyplot global state).
    """
    ap_um, dv_um, ml_um, sag, cor = _silhouettes(atlas_name)

    # Each shank's trajectory from its lowest electrode (y_lo) to its top
    # (y_hi); columns of every entry are atlas (AP, ML, DV) in µm.
    y_lo, y_hi = y_range
    trajectories = []
    for shank_id in sorted(shank_positions):
        xp = shank_positions[shank_id]
        xy = np.array([[xp, y_lo], [xp, y_hi]], dtype=float)
        trajectories.append(
            probe_to_atlas(xy, tip_atlas, pitch_deg, yaw_deg, shank_orientation_deg)
        )

    fig = Figure(figsize=(4.2, 2.4), dpi=110)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.04, wspace=0.06)
    ax_sag = fig.add_subplot(1, 2, 1)
    ax_cor = fig.add_subplot(1, 2, 2)

    ax_sag.contourf(ap_um, dv_um, sag, levels=[0.5, 1.5], colors=[_FILL])
    ax_sag.contour(ap_um, dv_um, sag, levels=[0.5], colors=[_EDGE], linewidths=0.6)
    ax_cor.contourf(ml_um, dv_um, cor, levels=[0.5, 1.5], colors=[_FILL])
    ax_cor.contour(ml_um, dv_um, cor, levels=[0.5], colors=[_EDGE], linewidths=0.6)

    for pts in trajectories:
        ax_sag.plot(pts[:, 0], pts[:, 2], "-", color=_PROBE, lw=1.3)  # AP vs DV
        ax_cor.plot(pts[:, 1], pts[:, 2], "-", color=_PROBE, lw=1.3)  # ML vs DV
    if trajectories:  # tip = lowest electrode of shank 0
        tip = trajectories[0][0]
        ax_sag.plot(tip[0], tip[2], "o", color=_PROBE, ms=3)
        ax_cor.plot(tip[1], tip[2], "o", color=_PROBE, ms=3)

    for ax, title in ((ax_sag, "Sagittal"), (ax_cor, "Coronal")):
        ax.set_aspect("equal")
        ax.invert_yaxis()  # DV increases downward → dorsal on top, ventral below
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    return fig
