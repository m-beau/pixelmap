"""Anatomical-overlay support for PixelMap.

Map Neuropixels electrode positions into a brain atlas (e.g. Allen CCF) so
the GUI can color electrodes by the region they sit in.

This subpackage is optional. Install with ``pip install pixelmap[anatomy]``
to pull in ``brainglobe-atlasapi`` and the atlases it manages on demand.
"""

from pixelmap.anatomy.atlas import RegionInfo, is_available, list_atlases
from pixelmap.anatomy.regions import regions_for_positions
from pixelmap.anatomy.transform import probe_to_atlas

__all__ = [
    "RegionInfo",
    "is_available",
    "list_atlases",
    "probe_to_atlas",
    "regions_for_positions",
]
