"""Aggregate atlas lookups into per-shank colored depth bands.

The GUI renders these as faint rectangles behind each shank to give a
quick visual readout of which brain regions a probe is passing through.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pixelmap.anatomy.atlas import lookup_regions
from pixelmap.anatomy.transform import ProbePlacement


@dataclass(frozen=True)
class RegionBand:
    """A contiguous depth range on one shank that sits inside a single region."""

    shank_id: int
    probe_xp: float       # probe-local x of this shank (µm)
    y_min: float          # bottom of the band in probe-local y (µm)
    y_max: float          # top of the band in probe-local y (µm)
    atlas_id: int
    acronym: str
    name: str
    rgb: tuple[int, int, int]


def compute_region_bands(
    shank_positions: dict[int, float],
    y_range: tuple[float, float],
    *,
    placement: ProbePlacement,
    atlas_name: str,
    step_um: float = 25.0,
) -> list[RegionBand]:
    """Sample regions along each shank and collapse into contiguous bands.

    Args:
        shank_positions: ``{shank_id: probe_xp}`` map giving each shank's
            position along the probe-local x axis (µm).
        y_range: ``(y_min, y_max)`` to sample along each shank (probe-local µm).
        placement: where the probe sits — tip, angles and the atlas warp in
            force. Pass the very instance given to
            :func:`pixelmap.anatomy.schematic.render_locator`, so the bands
            are the regions under the shanks that figure draws.
        atlas_name: brainglobe atlas identifier.
        step_um: sampling stride along each shank. Match or undercut the
            atlas resolution to avoid skipping thin regions.

    Returns:
        Flat list of :class:`RegionBand`, sorted by ``(shank_id, y_min)``.
        Samples that fall outside the atlas volume are omitted (no band).
    """
    y_lo, y_hi = y_range
    if y_hi <= y_lo:
        return []
    y_samples = np.arange(y_lo, y_hi + step_um, step_um)

    bands: list[RegionBand] = []
    for shank_id in sorted(shank_positions):
        probe_xp = shank_positions[shank_id]
        electrode_xy = np.column_stack(
            [np.full_like(y_samples, probe_xp, dtype=float), y_samples]
        )
        regions = lookup_regions(atlas_name, placement.to_atlas(electrode_xy))
        bands.extend(_collapse_to_bands(shank_id, probe_xp, y_samples, regions, step_um))
    return bands


def _collapse_to_bands(
    shank_id: int,
    probe_xp: float,
    y_samples: np.ndarray,
    regions: list,
    step_um: float,
) -> list[RegionBand]:
    """Merge runs of identical region IDs into single bands.

    Each band spans ``[y_sample - step/2, y_last_sample + step/2]`` so the
    bands tile the shank without gaps at the boundary samples.
    """
    out: list[RegionBand] = []
    current_id: int | None = None
    band_start: float | None = None
    last_region = None

    for y, region in zip(y_samples, regions):
        region_id = region.atlas_id if region is not None else None
        if region_id != current_id:
            if current_id is not None and last_region is not None and band_start is not None:
                out.append(RegionBand(
                    shank_id=shank_id,
                    probe_xp=probe_xp,
                    y_min=band_start - step_um / 2,
                    y_max=y - step_um / 2,
                    atlas_id=last_region.atlas_id,
                    acronym=last_region.acronym,
                    name=last_region.name,
                    rgb=last_region.rgb,
                ))
            current_id = region_id
            band_start = float(y) if region_id is not None else None
        last_region = region

    # Close out the final run.
    if current_id is not None and last_region is not None and band_start is not None:
        out.append(RegionBand(
            shank_id=shank_id,
            probe_xp=probe_xp,
            y_min=band_start - step_um / 2,
            y_max=float(y_samples[-1]) + step_um / 2,
            atlas_id=last_region.atlas_id,
            acronym=last_region.acronym,
            name=last_region.name,
            rgb=last_region.rgb,
        ))
    return out


def band_acronyms_at(bands: list[RegionBand], shank_ids, y_um) -> list[str | None]:
    """The region each electrode sits in, read off the bands on its shank.

    Deliberately *not* a second atlas lookup: an electrode's region is the
    band drawn behind it, so what is reported per electrode (the hover
    tooltip, the exported table) can never disagree with the picture.

    Args:
        bands: the output of :func:`compute_region_bands`.
        shank_ids: shank of each electrode, length ``N``.
        y_um: probe-local y of each electrode (µm), length ``N``.

    Returns:
        Length-``N`` list of acronyms; ``None`` where no band covers the
        electrode (it is outside the brain, or its shank has no bands).
    """
    shank_ids = np.asarray(shank_ids)
    y_um = np.asarray(y_um, dtype=float)
    out: list[str | None] = [None] * len(y_um)

    by_shank: dict[int, list[RegionBand]] = {}
    for band in bands:
        by_shank.setdefault(band.shank_id, []).append(band)

    for shank_id, shank_bands in by_shank.items():
        shank_bands.sort(key=lambda b: b.y_min)
        lows = np.array([b.y_min for b in shank_bands])
        highs = np.array([b.y_max for b in shank_bands])
        rows = np.flatnonzero(shank_ids == shank_id)
        # The band starting at or below each electrode; bands are half-open
        # [y_min, y_max), so an electrode on a boundary belongs to the upper one.
        which = np.searchsorted(lows, y_um[rows], side="right") - 1
        covered = (which >= 0) & (y_um[rows] < highs[np.clip(which, 0, None)])
        for row, band_idx in zip(rows[covered].tolist(), which[covered].tolist()):
            out[row] = shank_bands[band_idx].acronym
    return out
