"""Anatomical-overlay support for PixelMap.

Map Neuropixels electrode positions into a brain atlas (e.g. Allen CCF) so
the GUI can color electrodes by the region they sit in.

Atlas data is downloaded on first use by ``brainglobe-atlasapi`` and cached
on disk under ``~/.brainglobe/``.  Subsequent calls load from the local cache.
Both brainglobe v2 and v3 are supported; v3 stores its atlases in the
``brainglobe-atlasapi/`` subdirectory and fetches the bulky arrays lazily
(see :mod:`pixelmap.anatomy.atlas`).
"""

from pixelmap.anatomy.atlas import RegionInfo, list_atlases
from pixelmap.anatomy.regions import regions_for_positions, tip_depth_below_surface_um
from pixelmap.anatomy.transform import probe_axis_up, probe_to_atlas

__all__ = [
    "RegionInfo",
    "list_atlases",
    "probe_axis_up",
    "probe_to_atlas",
    "regions_for_positions",
    "tip_depth_below_surface_um",
]
