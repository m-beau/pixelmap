"""
Channelmap Generator for Neuropixels Probes

A Python package for generating channel maps for Neuropixels probes with hardware constraints.
Includes both a programmatic API and an interactive browser-based GUI.
"""

from .backend import (
    find_electrode_coordinates,
    find_forbidden_electrodes,
    find_selected_electrodes,
    format_imro_string,
    format_wiring_df,
    generate_kilosort_channelmap_dict,
    get_electrodes,
    get_preset_candidates,
    make_wiring_maps,
    plot_probe_layout,
)
from .constants import (
    IMRO_FORMAT_TO_PROBE_TYPE,
    LEGACY_INT_TO_IMRO_FORMAT,
    LEGACY_PROBE_TYPE_MAP,
    PROBE_FEATURES,
    PROBE_TYPE_MAP,
    REF_BANKS,
    REF_ELECTRODES,
    WIRING_FILE_MAP,
    SUPPORTED_1shank_PRESETS,
    SUPPORTED_4shanks_PRESETS,
    SUPPORTED_QuadBase_PRESETS,
)
from .types import Electrode
from .utils.imro import (
    generate_imro_channelmap,
    parse_imro_file,
    parse_imro_list,
    read_imro_file,
    save_to_imro_file,
)

__all__ = [
    "IMRO_FORMAT_TO_PROBE_TYPE",
    "LEGACY_INT_TO_IMRO_FORMAT",
    "LEGACY_PROBE_TYPE_MAP",
    "PROBE_FEATURES",
    "PROBE_TYPE_MAP",
    "REF_BANKS",
    "REF_ELECTRODES",
    "WIRING_FILE_MAP",
    "Electrode",
    "SUPPORTED_1shank_PRESETS",
    "SUPPORTED_4shanks_PRESETS",
    "SUPPORTED_QuadBase_PRESETS",
    "find_electrode_coordinates",
    "find_forbidden_electrodes",
    "find_selected_electrodes",
    "format_imro_string",
    "format_wiring_df",
    "generate_imro_channelmap",
    "generate_kilosort_channelmap_dict",
    "get_electrodes",
    "get_preset_candidates",
    "make_wiring_maps",
    "parse_imro_file",
    "parse_imro_list",
    "plot_probe_layout",
    "read_imro_file",
    "save_to_imro_file",
]

__version__ = "1.5.0"
__author__ = "Maxime Beau"
__email__ = "maxime@princeton.edu"
