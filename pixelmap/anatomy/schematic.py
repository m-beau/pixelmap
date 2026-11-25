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

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from pixelmap.anatomy.atlas import canonical_annotation, get_atlas
from pixelmap.anatomy.transform import AtlasWarp, probe_to_atlas

_FILL = "#d7d7e0"          # whole-brain silhouette fill (fallback view)
_EDGE = "#9a9aa8"          # silhouette / region outline color
_PROBE = "#d62728"         # probe trajectory + tip
_ENTRY = "#1f77b4"         # insertion point + the shank path down from it
_REGION_ALPHA = 0.6        # region fill opacity
_OUTLINE_RGBA = (0.2, 0.2, 0.25, 0.8)
# Largest bounding box (voxels) read in one go when resampling an oblique
# plane; ~32 MB of uint32. See _gather_bounded.
_GATHER_BUDGET = 8_000_000


def _atlas_data(atlas_name: str):
    """``(atlas, annotation, resolution)`` in canonical ``(AP, DV, ML)``.

    Going through :func:`~pixelmap.anatomy.atlas.canonical_annotation` means the
    slicing/projection below works for any atlas orientation, not just Allen's.

    Deliberately **not** memoised.  It used to be an ``lru_cache(maxsize=4)``,
    which made it a third independent cache holding atlas objects and their
    annotation volumes — keyed and evicted independently of the two in
    :mod:`pixelmap.anatomy.atlas`, so between them they could pin a dozen
    volumes (~700 MB each) while each looked individually well-bounded.  That
    was the server's monotonic RSS climb.  Both calls below are already cached
    upstream, so there is nothing to memoise here anyway.
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
    """Map a 2D label slice to an RGBA image colored by atlas region.

    One color per *distinct* region, then a single gather through the inverse
    index.  Masking the slice once per region instead (the obvious loop) costs
    a full pass per region — 2.4 s of a 3.5 s render on a 10 µm atlas, whose
    slices hold hundreds of regions.
    """
    region_ids, inverse = np.unique(label_img, return_inverse=True)
    palette = np.zeros((len(region_ids), 4), dtype=float)
    for i, region_id in enumerate(region_ids):
        if region_id == 0:  # outside-brain / undefined: stays transparent
            continue
        r, g, b = _region_rgb(atlas, int(region_id))
        palette[i] = (r / 255.0, g / 255.0, b / 255.0, _REGION_ALPHA)
    return palette[inverse.reshape(label_img.shape)]


def _outline_rgba(label_img: np.ndarray) -> np.ndarray:
    """RGBA layer marking boundaries between differing labels (region edges)."""
    border = np.zeros(label_img.shape, dtype=bool)
    border[:-1, :] |= label_img[:-1, :] != label_img[1:, :]
    border[:, :-1] |= label_img[:, :-1] != label_img[:, 1:]
    rgba = np.zeros((*label_img.shape, 4), dtype=float)
    rgba[border] = _OUTLINE_RGBA
    return rgba


def _hollow(color: str) -> dict:
    """Marker style: a colored outline with no fill, so the anatomy under the
    tip, insertion point and bregma stays visible."""
    return {"markerfacecolor": "none", "markeredgecolor": color, "markeredgewidth": 1.0}


def _gather_bounded(ann, ap, dv, ml) -> np.ndarray:
    """``ann.gather`` for points on an oblique plane, without reading a thick slab.

    ``gather`` reads the bounding box of its points. For a tilted plane that
    box is most of the volume (hundreds of MB on a 10 µm atlas), so the points
    are split along AP until each piece's box is small: a plane cut into short
    AP runs is a staircase of thin boxes.
    """
    out = np.zeros(ap.size, dtype=ann.dtype)
    stack = [np.arange(ap.size)]
    while stack:
        sel = stack.pop()
        a, d, m = ap[sel], dv[sel], ml[sel]
        a_lo, a_hi = int(a.min()), int(a.max())
        box = (a_hi - a_lo + 1) * (int(d.max()) - int(d.min()) + 1) * (int(m.max()) - int(m.min()) + 1)
        if box <= _GATHER_BUDGET or a_lo == a_hi:
            out[sel] = ann.gather(a, d, m)
        else:
            mid = (a_lo + a_hi) // 2
            stack.append(sel[a <= mid])
            stack.append(sel[a > mid])
    return out


def _warped_slice(ann, res, warp: AtlasWarp, fixed_col: int, fixed_um: float,
                  x_col: int, y_col: int):
    """Cross-section of the *warped* brain on a plane of the stereotaxic frame.

    The plane is ``coordinate[fixed_col] == fixed_um`` in the frame the probe
    lives in (columns index ``(AP, ML, DV)``). Each pixel is mapped back into
    the atlas and looked up there, so the picture is the true section of the
    squished/tilted brain — including the out-of-plane part of the tilt, which
    a 2D transform of an axis-aligned atlas slice cannot show.

    Returns ``(label_img, x_um, y_um)`` with ``label_img`` shaped
    ``(len(y_um), len(x_um))``; pixels outside the volume are 0.
    """
    n_ap, n_dv, n_ml = ann.shape
    res_c = np.asarray(res, dtype=float)[[0, 2, 1]]          # (AP, ML, DV) voxel size
    n_c = np.array([n_ap, n_ml, n_dv])
    # The window is the warped volume's bounding box, at the atlas's own pitch.
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float)
    corners = warp.from_atlas(corners * (n_c - 1) * res_c)
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    x_um = np.arange(lo[x_col], hi[x_col] + res_c[x_col], res_c[x_col])
    y_um = np.arange(lo[y_col], hi[y_col] + res_c[y_col], res_c[y_col])

    # The warp is affine, so each atlas index is (constant + x term + y term):
    # build it from two 1D ramps per axis rather than a full (N, 3) float
    # point cloud, whose temporaries cost hundreds of MB on a 10 µm atlas.
    origin = np.zeros(3)
    origin[fixed_col] = fixed_um
    e_x, e_y = np.zeros(3), np.zeros(3)
    e_x[x_col], e_y[y_col] = 1.0, 1.0
    base = warp.to_atlas(origin)
    d_x, d_y = warp.to_atlas(origin + e_x) - base, warp.to_atlas(origin + e_y) - base

    def _index(col):
        ramp = ((base[col] + y_um * d_y[col])[:, None] + (x_um * d_x[col])[None, :]) / res_c[col]
        return np.rint(ramp, out=ramp).astype(np.int32).ravel()

    ap, ml, dv = _index(0), _index(1), _index(2)
    inside = ((ap >= 0) & (ap < n_ap) & (dv >= 0) & (dv < n_dv) & (ml >= 0) & (ml < n_ml))

    label = np.zeros(ap.size, dtype=ann.dtype)
    if inside.any():
        label[inside] = _gather_bounded(ann, ap[inside], dv[inside], ml[inside])
    return label.reshape(len(y_um), len(x_um)), x_um, y_um


def _draw_view(ax, label_img, x_um, y_um, atlas, silhouette, title):
    """Draw one slice (colored regions + outlines), or a silhouette fallback.

    ``label_img`` is shaped ``(len(y_um), len(x_um))``; y is DV in the sagittal
    and coronal views. The y-axis is oriented dorsal-up.

    ``silhouette`` is a *callable* returning ``(x_um, y_um, mask)``, not a
    mask: building it is the one operation here that has to read every voxel,
    and it is only needed when the slice misses the brain and there is nothing
    to draw. Passing the array instead meant paying for it on every render,
    including the overwhelmingly common case where it goes unused.
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
        # The slice misses the brain here — show the whole-brain silhouette.
        sx, sy, mask = silhouette()
        ax.contourf(sx, sy, mask, levels=[0.5, 1.5], colors=[_FILL])
        ax.contour(sx, sy, mask, levels=[0.5], colors=[_EDGE], linewidths=0.6)
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
    bregma_um: tuple[float, float, float] | None = None,
    entry_um: tuple[float, float, float] | None = None,
    ap_squish: float = 1.0,
    ml_squish: float = 1.0,
    dv_squish: float = 1.0,
    tilt_deg: float = 0.0,
    warp: AtlasWarp | None = None,
    fig: Figure | None = None,
) -> Figure:
    """Render the locator figure for the current insertion pose.

    Pose args mirror
    :func:`pixelmap.anatomy.visualization.compute_region_bands`. Slices are
    taken at the tip's AP (coronal), ML (sagittal) and DV (horizontal).

    ``bregma_um`` draws a black bregma dot. ``entry_um`` draws the insertion
    point (where the shank enters the brain) as a blue diamond, joined to the
    tip by a dashed line along shank 0 — the inactive shank above the
    electrodes and the tip below them are not drawn otherwise. ``ap_squish`` / ``ml_squish`` /
    ``dv_squish`` and ``tilt_deg`` warp the *brain* about bregma (per-axis
    squash + nose-up AP tilt, see :class:`~pixelmap.anatomy.transform.AtlasWarp`)
    so the atlas is shown in its stereotaxic-corrected shape with the probe
    straight. Each view is then the true cross-section of that warped brain
    through the tip, resampled from the atlas — not a 2D-transformed atlas
    slice, which would miss the part of the tilt that is out of the view's
    plane and draw the wrong section away from bregma. Returns a bare matplotlib
    :class:`~matplotlib.figure.Figure` for a Panel ``Matplotlib`` pane (no
    pyplot global state).

    Pass ``fig`` to draw into an existing figure instead of building one. The
    overlay re-renders on every pose edit, and constructing a figure per render
    — with its Agg canvas and renderer — was measured at 3.1 MB of unreturned
    RSS per call inside the container, more than everything the atlas itself
    costs. Reusing one figure per session removes that; the caller owns it and
    must not keep references to artists across renders, since this clears it.
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

    # 2×2 grid (4th cell empty) so the figure stays narrow enough to fit the
    # side panel without overflowing the page; a 1×3 row would be too wide.
    if fig is None:
        fig = Figure(figsize=(3.7, 3.3), dpi=110)
    else:
        fig.clear()  # reuse the canvas and renderer; drop the previous render
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02,
                        wspace=0.08, hspace=0.20)
    ax_sag = fig.add_subplot(2, 2, 1)
    ax_cor = fig.add_subplot(2, 2, 2)
    ax_hor = fig.add_subplot(2, 2, 3)

    silhouettes = (
        lambda: (ap_um, dv_um, ann.projections()[0].T),
        lambda: (ml_um, dv_um, ann.projections()[1]),
        lambda: (ml_um, ap_um, ann.projections()[2]),
    )
    # ``warp`` wins; the separate squish/tilt arguments describe one about
    # ``bregma_um`` for callers that have no AtlasWarp to hand.
    if warp is None and bregma_um is not None:
        warp = AtlasWarp(tuple(float(c) for c in bregma_um),
                         ap_squish, ml_squish, dv_squish, tilt_deg)
    if warp is not None and warp.is_identity:
        warp = None

    # Labels report the tip coordinate along each panel's visible horizontal
    # axis: AP on the sagittal view, ML on the coronal view. ``projections()``
    # returns the silhouette reduced over ML, AP and DV respectively, and
    # caches, so at most one whole-volume scan happens — and only if some view
    # actually has no slice to draw.
    if warp is None:
        # Slices of the atlas as published: plain axis-aligned reads.
        views = (
            (ann[:, :, ml_idx].T, ap_um, dv_um),
            (ann[ap_idx, :, :], ml_um, dv_um),
            (ann[:, dv_idx, :], ml_um, ap_um),
        )
        at = (ap_um[ap_idx], ml_um[ml_idx], dv_um[dv_idx])
    else:
        # Sections of the warped brain through the tip; (fixed, x, y) columns
        # index (AP, ML, DV). The fallback silhouette stays unwarped: it only
        # says "the probe is out here", and warping it would mean resampling
        # the whole volume.
        views = (
            _warped_slice(ann, res, warp, 1, tip_ml, 0, 2),
            _warped_slice(ann, res, warp, 0, tip_ap, 1, 2),
            _warped_slice(ann, res, warp, 2, tip_dv, 1, 0),
        )
        at = (tip_ap, tip_ml, tip_dv)
    titles = (f"Sagittal · AP {at[0]:.0f} µm", f"Coronal · ML {at[1]:.0f} µm",
              f"Horizontal · DV {at[2]:.0f} µm")
    for ax, (label_img, x_um, y_um), silhouette, title in zip(
            (ax_sag, ax_cor, ax_hor), views, silhouettes, titles):
        _draw_view(ax, label_img, x_um, y_um, atlas, silhouette=silhouette, title=title)

    # Probe (red) + bregma estimate (black) per view: (axis, x-col, y-col)
    # indexing into (AP, ML, DV).
    for ax, x_col, y_col in ((ax_sag, 0, 2), (ax_cor, 1, 2), (ax_hor, 1, 0)):
        for pts in trajectories:
            ax.plot(pts[:, x_col], pts[:, y_col], "-", color=_PROBE, lw=1.3, zorder=5)
        if trajectories:  # tip = lowest electrode of shank 0
            tip = trajectories[0][0]
            ax.plot(tip[x_col], tip[y_col], "o", ms=4.5, zorder=6, **_hollow(_PROBE))
        if entry_um is not None:
            if trajectories:
                tip = trajectories[0][0]
                ax.plot([tip[x_col], entry_um[x_col]], [tip[y_col], entry_um[y_col]],
                        "--", color=_ENTRY, lw=0.8, zorder=4)
            ax.plot(entry_um[x_col], entry_um[y_col], "D", ms=4.5, zorder=7, **_hollow(_ENTRY))
        if bregma_um is not None:
            ax.plot(bregma_um[x_col], bregma_um[y_col], "o", ms=4.5, zorder=8, **_hollow("black"))

    for ax in (ax_sag, ax_cor, ax_hor):
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Legend in the empty 4th quadrant.
    ax_leg = fig.add_subplot(2, 2, 4)
    ax_leg.axis("off")
    handles = [Line2D([0], [0], marker="o", color="none", markersize=6,
                      label="probe tip", **_hollow(_PROBE))]
    if entry_um is not None:
        handles.append(Line2D([0], [0], marker="D", color="none", markersize=5,
                              label="insertion point", **_hollow(_ENTRY)))
    if bregma_um is not None:
        handles.append(Line2D([0], [0], marker="o", color="none", markersize=6,
                              label="bregma (est.)", **_hollow("black")))
    ax_leg.legend(handles=handles, loc="center", frameon=False, fontsize=8)
    return fig
