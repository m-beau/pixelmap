"""Locator schematic: brain slices through the probe tip, with the probe drawn.

Three orthogonal slices taken at the tip coordinate — sagittal (x = AP, y = DV,
at the tip ML), coronal (x = ML, y = DV, at the tip AP) and horizontal /
top-down (x = ML, y = AP, at the tip DV) — with their atlas regions filled in
the atlas's own colors and outlined, and the probe shank trajectories overlaid.
If the tip falls outside the volume the view falls back to a plain whole-brain
silhouette so it's never blank.

Coordinates are taken in pixelmap's canonical ``(AP, DV, ML)`` frame (see
:func:`pixelmap.anatomy.atlas.canonical_annotation`), so it works for any
atlas orientation.
"""

from __future__ import annotations

import functools

import numpy as np
from matplotlib.figure import Figure

from pixelmap.anatomy.atlas import canonical_annotation, get_atlas
from pixelmap.anatomy.transform import probe_to_atlas

_FILL = "#d7d7e0"          # whole-brain silhouette fill (fallback view)
_EDGE = "#9a9aa8"          # silhouette / region outline color
_PROBE = "#d62728"         # probe trajectory + tip
_REGION_ALPHA = 0.6        # region fill opacity
_OUTLINE_RGBA = (0.2, 0.2, 0.25, 0.8)


@functools.lru_cache(maxsize=4)
def _atlas_data(atlas_name: str):
    """Cached ``(atlas, annotation, resolution)`` in canonical ``(AP, DV, ML)``.

    Going through :func:`~pixelmap.anatomy.atlas.canonical_annotation` means the
    slicing/projection below works for any atlas orientation, not just Allen's.
    """
    atlas = get_atlas(atlas_name)
    annotation, resolution = canonical_annotation(atlas_name)
    return atlas, annotation, resolution


def _region_rgb(atlas, region_id: int) -> tuple[int, int, int]:
    """Atlas color for a region id, falling back to a neutral gray."""
    try:
        triplet = atlas.structures[region_id].get("rgb_triplet", (200, 200, 200))
        return int(triplet[0]), int(triplet[1]), int(triplet[2])
    except (KeyError, TypeError, AttributeError):
        return (200, 200, 200)


def _region_fill_rgba(label_img: np.ndarray, atlas) -> np.ndarray:
    """Map a 2D label slice to an RGBA image colored by atlas region."""
    rgba = np.zeros((*label_img.shape, 4), dtype=float)
    for region_id in np.unique(label_img):
        if region_id == 0:  # outside-brain / undefined
            continue
        r, g, b = _region_rgb(atlas, int(region_id))
        mask = label_img == region_id
        rgba[mask, :3] = (r / 255.0, g / 255.0, b / 255.0)
        rgba[mask, 3] = _REGION_ALPHA
    return rgba


def _outline_rgba(label_img: np.ndarray) -> np.ndarray:
    """RGBA layer marking boundaries between differing labels (region edges)."""
    border = np.zeros(label_img.shape, dtype=bool)
    border[:-1, :] |= label_img[:-1, :] != label_img[1:, :]
    border[:, :-1] |= label_img[:, :-1] != label_img[:, 1:]
    rgba = np.zeros((*label_img.shape, 4), dtype=float)
    rgba[border] = _OUTLINE_RGBA
    return rgba


def _draw_view(ax, label_img, x_um, y_um, atlas, fallback_mask, title):
    """Draw one slice (colored regions + outlines), or a silhouette fallback.

    ``label_img`` and ``fallback_mask`` are shaped ``(len(y_um), len(x_um))``;
    y is DV in both views. The y-axis is oriented dorsal-up.
    """
    # extent=(left, right, bottom, top); top = y_um[0] (DV 0) puts dorsal up.
    extent = (float(x_um[0]), float(x_um[-1]), float(y_um[-1]), float(y_um[0]))
    if label_img.any():
        ax.imshow(_region_fill_rgba(label_img, atlas), extent=extent,
                  origin="upper", aspect="equal", interpolation="nearest", zorder=1)
        ax.imshow(_outline_rgba(label_img), extent=extent,
                  origin="upper", aspect="equal", interpolation="nearest", zorder=2)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    else:
        # Tip is outside the volume here — show the whole-brain silhouette.
        ax.contourf(x_um, y_um, fallback_mask, levels=[0.5, 1.5], colors=[_FILL])
        ax.contour(x_um, y_um, fallback_mask, levels=[0.5], colors=[_EDGE], linewidths=0.6)
        ax.invert_yaxis()
    ax.set_title(title, fontsize=8)


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
    :func:`pixelmap.anatomy.visualization.compute_region_bands`. Slices are
    taken at the tip's AP (coronal), ML (sagittal) and DV (horizontal).
    Returns a bare matplotlib :class:`~matplotlib.figure.Figure` for a Panel
    ``Matplotlib`` pane (no pyplot global state).
    """
    atlas, ann, res = _atlas_data(atlas_name)
    n_ap, n_dv, n_ml = ann.shape
    ap_um = np.arange(n_ap) * res[0]
    dv_um = np.arange(n_dv) * res[1]
    ml_um = np.arange(n_ml) * res[2]

    # Each shank's trajectory from its lowest electrode (y_lo) to its top
    # (y_hi); columns of every entry are atlas (AP, ML, DV) in µm.
    y_lo, y_hi = y_range
    trajectories = []
    for shank_id in sorted(shank_positions):
        xy = np.array([[shank_positions[shank_id], y_lo],
                       [shank_positions[shank_id], y_hi]], dtype=float)
        trajectories.append(
            probe_to_atlas(xy, tip_atlas, pitch_deg, yaw_deg, shank_orientation_deg)
        )

    tip_ap, tip_ml, tip_dv = tip_atlas
    ap_idx = int(np.clip(round(tip_ap / res[0]), 0, n_ap - 1))
    dv_idx = int(np.clip(round(tip_dv / res[1]), 0, n_dv - 1))
    ml_idx = int(np.clip(round(tip_ml / res[2]), 0, n_ml - 1))
    inside = ann > 0

    # 2×2 grid (4th cell empty) so the figure stays narrow enough to fit the
    # side panel without overflowing the page; a 1×3 row would be too wide.
    fig = Figure(figsize=(3.7, 3.3), dpi=110)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02,
                        wspace=0.08, hspace=0.20)
    ax_sag = fig.add_subplot(2, 2, 1)
    ax_cor = fig.add_subplot(2, 2, 2)
    ax_hor = fig.add_subplot(2, 2, 3)

    # Labels report the coordinate along each panel's visible horizontal axis:
    # AP on the sagittal view, ML on the coronal view.
    _draw_view(ax_sag, ann[:, :, ml_idx].T, ap_um, dv_um, atlas,
               fallback_mask=inside.any(axis=2).T,
               title=f"Sagittal · AP {ap_um[ap_idx]:.0f} µm")
    _draw_view(ax_cor, ann[ap_idx, :, :], ml_um, dv_um, atlas,
               fallback_mask=inside.any(axis=0),
               title=f"Coronal · ML {ml_um[ml_idx]:.0f} µm")
    _draw_view(ax_hor, ann[:, dv_idx, :], ml_um, ap_um, atlas,
               fallback_mask=inside.any(axis=1),
               title=f"Horizontal · DV {dv_um[dv_idx]:.0f} µm")

    # Probe overlay per view: (axis, x-column, y-column) into (AP, ML, DV).
    for ax, x_col, y_col in ((ax_sag, 0, 2), (ax_cor, 1, 2), (ax_hor, 1, 0)):
        for pts in trajectories:
            ax.plot(pts[:, x_col], pts[:, y_col], "-", color=_PROBE, lw=1.3, zorder=5)
        if trajectories:  # tip = lowest electrode of shank 0
            tip = trajectories[0][0]
            ax.plot(tip[x_col], tip[y_col], "o", color=_PROBE, ms=3, zorder=6)

    for ax in (ax_sag, ax_cor, ax_hor):
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    return fig
