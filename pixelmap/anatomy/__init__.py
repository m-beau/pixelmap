"""Anatomical-overlay support for PixelMap.

Map Neuropixels electrode positions into a brain atlas (e.g. Allen CCF) so
the GUI can color electrodes by the region they sit in.

Atlas data is downloaded on first use by ``brainglobe-atlasapi`` and cached
on disk in ``~/.brainglobe/``.  Subsequent calls load from the local cache.
"""

from pixelmap.anatomy.atlas import RegionInfo, list_atlases
from pixelmap.anatomy.regions import regions_for_positions
from pixelmap.anatomy.transform import probe_to_atlas

__all__ = [
    "RegionInfo",
    "list_atlases",
    "probe_to_atlas",
    "regions_for_positions",
]
