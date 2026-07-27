"""Anatomical-overlay support for PixelMap.

Map Neuropixels electrode positions into a brain atlas (e.g. Allen CCF) so
the GUI can color electrodes by the region they sit in.

Atlas data is downloaded on first use by ``brainglobe-atlasapi`` and cached
on disk in ``~/.brainglobe/``.  Subsequent calls load from the local cache.
"""

from pixelmap.anatomy.atlas import RegionInfo, list_atlases, surface_point_um
from pixelmap.anatomy.regions import regions_for_positions
from pixelmap.anatomy.transform import (
    insertion_depth_um,
    insertion_to_tip,
    probe_axis,
    probe_to_atlas,
    tip_to_insertion,
)

__all__ = [
    "RegionInfo",
    "insertion_depth_um",
    "insertion_to_tip",
    "list_atlases",
    "probe_axis",
    "probe_to_atlas",
    "regions_for_positions",
    "surface_point_um",
    "tip_to_insertion",
]
