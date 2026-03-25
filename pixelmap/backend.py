#############
## Imports ##
#############

from pathlib import Path
import pickle

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import (
    PROBE_N,
    PROBE_TYPE_MAP,
    WIRING_FILE_MAP,
    REF_BANKS,
    REF_ELECTRODES,
    SUPPORTED_1shank_PRESETS,
    SUPPORTED_4shanks_PRESETS,
)

from .types import Electrode

############################
## Channel map generation ##
############################


def get_electrodes(probe_type, wiring_df, preset=None, custom_electrodes=None):
    """
    Get electrode selection based on preset, avoiding channel conflicts.
    - probe_type: Type of probe (e.g., "1.0", "2.0-1shank", "2.0-4shanks", "NXT")
    - wiring_df: Wiring DataFrame containing wiring information
    - preset: Preset layout configuration
    - custom_electrodes: Optional list of custom (shank_id, electrode_id) pairs
    """

    if custom_electrodes is None:
        assert preset is not None, "You must provide a preset if you do not provide custom electrodes!"
        selected_electrodes = get_preset_candidates(preset, probe_type, wiring_df)
    else:
        if custom_electrodes.ndim == 1:  # assume single shank
            custom_electrodes = np.vstack([custom_electrodes * 0, custom_electrodes]).T
        selected_electrodes = np.array(custom_electrodes).astype(int)

    _verify_hardware_violations(probe_type, selected_electrodes, wiring_df)

    return selected_electrodes


def _verify_hardware_violations(probe_type, selected_electrodes, wiring_df):
    # Check for probe type illegal numbers
    max_n_per_shank = PROBE_N[probe_type]["n_per_shank"]
    for shank in np.unique(selected_electrodes[:, 0]):
        n_electrodes_shank = np.sum(shank == selected_electrodes[:, 0])
        assert n_electrodes_shank <= max_n_per_shank, (
            f"Violation - too many electrodes ({n_electrodes_shank}) on shank {shank}"
        )

    # Check for wiring conflicts
    forbidden_electrodes = find_forbidden_electrodes(selected_electrodes, wiring_df)
    selected_e_str = [f"{se[0]}_{se[1]}" for se in selected_electrodes]
    forbidden_e_str = [f"{se[0]}_{se[1]}" for se in forbidden_electrodes]
    conflicts = np.isin(selected_e_str, forbidden_e_str)
    assert not np.any(conflicts), (
        f"Selected electrodes conflict with wiring diagram ({' and '.join(list(np.array(selected_e_str)[conflicts]))}). Please choose a different preset or custom electrodes."
    )


def find_forbidden_electrodes(selected_electrodes, wiring_df):
    "Return list of forbidden electrodes [[shank_id, electrode_id], ...] given selected electrodes and wiring diagram."

    df = format_wiring_df(wiring_df)

    table_ids = df.iloc[:, 1:].values
    table_shape = table_ids.shape
    flat_ids = table_ids.ravel()
    flat_shank_ids = np.array([id[0] for id in flat_ids])
    flat_electrode_ids = np.array([id[1] for id in flat_ids])

    forbidden_electrodes = []
    for shank_id, electrode_id in selected_electrodes:
        table_bool = ((flat_shank_ids == shank_id) & (flat_electrode_ids == electrode_id)).reshape(table_shape)
        assert np.any(table_bool), f"electrode {electrode_id} on shank {shank_id} not found in wiring map!"
        table_loc = np.nonzero(table_bool)
        table_row = table_loc[0][0]
        # grab non-nan and non-self electrodes on row of wiring table
        forbidden_electrodes += [
            se
            for se in table_ids[table_row]
            if (~np.any(np.isnan(se))) & ~((shank_id == se[0]) & (electrode_id == se[1]))
        ]

    return np.array(forbidden_electrodes).astype(int)

def make_wiring_maps(wiring_maps_dir):
    "Precomputes electrode wiring conflict maps for each probe type."
    wiring_maps_dir = Path(wiring_maps_dir)
    assert wiring_maps_dir.exists(), f"{wiring_maps_dir} does not exist!"
    cache_file = wiring_maps_dir / "wiring_maps.pkl"

    # If pickled result found, preloads it
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
                wiring_maps = pickle.load(f)
                return wiring_maps
    
    # Generate wiring maps
    wiring_maps = {}
    for probe_type in WIRING_FILE_MAP:

        pos_file, wire_file = WIRING_FILE_MAP[probe_type]
        positions_file = wiring_maps_dir / pos_file
        wiring_file = wiring_maps_dir / wire_file
        positions_df = pd.read_csv(positions_file)
        wiring_df = pd.read_csv(wiring_file)

        wiring_maps[probe_type] = {}
        for (shank_id, electrode_id, _, _) in positions_df.values:
            electrode = Electrode(shank_id=shank_id, electrode_id=electrode_id)
            conflicting_electrodes = find_forbidden_electrodes([(shank_id, electrode_id)], wiring_df)
            conflicting_electrodes = {Electrode(e[0], e[1]) for e in conflicting_electrodes}
            wiring_maps[probe_type][electrode] = conflicting_electrodes

    # Pickle result
    with open(cache_file, 'wb') as f:
        pickle.dump(wiring_maps, f)
    
    return wiring_maps


def format_wiring_df(wiring_df):
    """
    Fetch shank id from column header of wiring_df
    and replace electrode_id with (shank_id, electrode_id) tuple in each cell
    """
    # add shank id to electrode id, in dataframe's cells
    df = wiring_df.copy()
    for column in df.columns[1:]:
        shank_id = int(column[5])
        electrode_ids = df.loc[:, column].values
        shank_electrode_ids = np.vstack([np.zeros(len(electrode_ids)) + shank_id, electrode_ids]).T
        shank_electrode_ids = [tuple(se) for se in shank_electrode_ids]
        df[column] = df[column].astype(object)
        df[column] = shank_electrode_ids

    return df


def format_imro_string(electrodes, wiring_df, probe_type, probe_subtype, reference_id, ap_gain, lf_gain, hp_filter):
    """
    Format IMRO string based on probe type.
    See see https://billkarsh.github.io/SpikeGLX/help/imroTables/ for details.
    """

    assert probe_subtype is not None

    df_coordinates = find_electrode_coordinates(electrodes, wiring_df)
    electrodes = [[int(se[0]), int(se[1])] for se in electrodes]

    # Generate entries based on probe type
    entries = []
    if probe_type == "1.0":
        # Format: (channel_id bank ref ap_gain lf_gain hp_filter)
        ref_value = REF_ELECTRODES[probe_subtype][reference_id]

        for (shank_id, electrode_id), (row, col) in zip(electrodes, df_coordinates):
            channel = int(wiring_df.loc[row, "channel"])
            bank = int(wiring_df.columns[col][-1])
            entry = (channel, bank, ref_value, ap_gain, lf_gain, hp_filter)
            entries.append(entry)

    elif probe_type == "2.0-1shank":
        # Format: (channel_id bank_mask ref electrode_id)
        ref_value = REF_ELECTRODES[probe_subtype][reference_id]

        for (shank_id, electrode_id), (row, col) in zip(electrodes, df_coordinates):
            channel = int(wiring_df.loc[row, "channel"])
            bank = int(wiring_df.columns[col][-1])
            bank_mask = REF_BANKS["2.0-1shank"][bank]  # {1=bnk-0, 2=bnk-1, 4=bnk-2, 8=bnk-3}
            entry = (channel, bank_mask, ref_value, electrode_id)
            entries.append(entry)

    elif probe_type in ["2.0-4shanks", "NXT"]:
        if reference_id == "Join Tips":
            ref_values = REF_ELECTRODES[probe_subtype][reference_id]
        else:  # apart from join tips, all electrodes share the same reference.
            ref_values = [REF_ELECTRODES[probe_subtype][reference_id]] * PROBE_N[probe_type]['n']
        assert len(ref_values) == len(electrodes),\
            "Major error - mismatch between number of electrodes and number of reference ids"

        for (shank_id, electrode_id), (row, col) in zip(electrodes, df_coordinates):
            channel = int(wiring_df.loc[row, "channel"])
            bank = int(wiring_df.columns[col][-1])
            entry = [channel, shank_id, bank, electrode_id] # reference added after sorting per channel
            entries.append(entry)

    # Sort by channel and format
    entries.sort(key=lambda x: x[0])
    
    # Only apply ref_values formatting for 4-shank probes
    if probe_type in ["2.0-4shanks", "NXT"]:
        entries = [(e[0], e[1], e[2], ref, e[3]) for e, ref in zip(entries, ref_values)]
    
    header = (probe_subtype, len(entries))
    imro_list = [header] + entries

    return imro_list


def find_electrode_coordinates(electrodes, wiring_df):
    df = format_wiring_df(wiring_df)

    table_ids = df.iloc[:, 1:].values
    table_shape = table_ids.shape
    flat_ids = table_ids.ravel()
    flat_shank_ids = np.array([id[0] for id in flat_ids])
    flat_electrode_ids = np.array([id[1] for id in flat_ids])

    coordinates = []
    for shank_id, electrode_id in electrodes:
        table_bool = ((flat_shank_ids == shank_id) & (flat_electrode_ids == electrode_id)).reshape(table_shape)
        row, col = np.nonzero(table_bool)
        coordinates.append([int(row[0]), int(col[0]) + 1])

    return coordinates


###########################
## preset configurations ##
###########################


def get_preset_candidates(preset, probe_type, wiring_df):
    """Get candidate (shank_id, electrode_id) pairs for a preset based on probe type and wiring."""

    if probe_type in ["1.0", "2.0-1shank"]:
        assert preset in SUPPORTED_1shank_PRESETS, (
            f"Preset {preset} is not supported for probe type {probe_type}. Supported presets: {SUPPORTED_1shank_PRESETS}"
        )
    elif probe_type in ["2.0-4shanks", "NXT"]:
        assert preset in SUPPORTED_4shanks_PRESETS, (
            f"Preset {preset} is not supported for probe type {probe_type}. Supported presets: {SUPPORTED_4shanks_PRESETS}"
        )

    preset_electrodes = []

    if probe_type == "1.0":
        # Single shank configurations for 1.0
        if preset == "Tip":
            # 0-383 of bank 0
            for row in range(384):
                electrode_id = wiring_df.loc[row, "shank0-bank0"]
                if not pd.isna(electrode_id):
                    preset_electrodes.append([0, int(electrode_id)])

        elif preset == "tip_b0_top_b1":
            # 0-191 of bank 0, 192-383 of bank 1
            for row in range(192):
                electrode_id = wiring_df.loc[row, "shank0-bank0"]
                if not pd.isna(electrode_id):
                    preset_electrodes.append([0, int(electrode_id)])
            for row in range(192, 384):
                electrode_id = wiring_df.loc[row, "shank0-bank1"]
                if not pd.isna(electrode_id):
                    preset_electrodes.append([0, int(electrode_id)])

        elif preset == "top_b0_tip_b1":
            # 192-383 of bank 0, 0-191 of bank 1
            for row in range(192, 384):
                electrode_id = wiring_df.loc[row, "shank0-bank0"]
                if not pd.isna(electrode_id):
                    preset_electrodes.append([0, int(electrode_id)])
            for row in range(192):
                electrode_id = wiring_df.loc[row, "shank0-bank1"]
                if not pd.isna(electrode_id):
                    preset_electrodes.append([0, int(electrode_id)])

        elif preset == "zigzag":
            # channels 0, 2, 4, 6, 8, 10... of bank 0 (even channels)
            for row in range(0, 384, 2):
                electrode_id = wiring_df.loc[row, "shank0-bank0"]
                if not pd.isna(electrode_id):
                    preset_electrodes.append([0, int(electrode_id)])
            # channels 1, 3, 5, 7... of bank 1 (odd channels)
            for row in range(1, 384, 2):
                electrode_id = wiring_df.loc[row, "shank0-bank1"]
                if not pd.isna(electrode_id):
                    preset_electrodes.append([0, int(electrode_id)])

    elif probe_type == "2.0-1shank":
        # Single shank configurations for 2.0-1shank
        if preset == "Tip":
            # 0-383 of bank 0
            for row in range(384):
                electrode_id = wiring_df.loc[row, "shank0-bank0"]
                if not pd.isna(electrode_id):
                    preset_electrodes.append([0, int(electrode_id)])

        elif preset == "tip_b0_top_b1":
            # 0-191 of bank 0, 192-383 of bank 1
            for row in range(192):
                electrode_id = wiring_df.loc[row, "shank0-bank0"]
                if not pd.isna(electrode_id):
                    preset_electrodes.append([0, int(electrode_id)])
            for row in range(192, 384):
                electrode_id = wiring_df.loc[row, "shank0-bank1"]
                if not pd.isna(electrode_id):
                    preset_electrodes.append([0, int(electrode_id)])

        elif preset == "top_b0_tip_b1":
            # 192-383 of bank 0, 0-191 of bank 1
            for row in range(192, 384):
                electrode_id = wiring_df.loc[row, "shank0-bank0"]
                if not pd.isna(electrode_id):
                    preset_electrodes.append([0, int(electrode_id)])
            for row in range(192):
                electrode_id = wiring_df.loc[row, "shank0-bank1"]
                if not pd.isna(electrode_id):
                    preset_electrodes.append([0, int(electrode_id)])

        elif preset == "zigzag":
            # For 2.0 bank 0: channels 0, 3, 4, 7, 8, 11, 12... on bank 0
            # Pattern: 0,3 then +4 repeatedly: 4,7,8,11,12,15,16,19...
            channels_bank0 = []
            i = 0
            while i < 384:  # Limit to 192 channels for bank 0
                channels_bank0.append(i)
                channels_bank0.append(i + 3)
                i += 4

            # For 2.0 bank 1: channels 1, 2, 5, 6, 8, 10, 13... on bank 0
            channels_bank1 = []
            i = 0
            while i < 384:  # Limit to 192 channels for bank 0
                channels_bank1.append(i + 1)
                channels_bank1.append(i + 2)
                i += 4

            electrodes_bank0 = wiring_df.loc[channels_bank0, "shank0-bank0"]
            electrodes_bank1 = wiring_df.loc[channels_bank1, "shank0-bank1"]
            electrodes_bank0 = np.vstack([electrodes_bank0 * 0, electrodes_bank0]).T
            electrodes_bank1 = np.vstack([electrodes_bank1 * 0, electrodes_bank1]).T
            preset_electrodes = np.vstack([electrodes_bank0, electrodes_bank1])

    elif probe_type in ["2.0-4shanks", "NXT"]:
        # Multi-shank configurations
        if preset == "tips_all":
            # 0-95 (384/4) of each shank's bank 0
            preset_electrodes = np.arange(96)
            preset_electrodes = np.vstack([preset_electrodes * 0, preset_electrodes]).T
            for shank in range(3):
                preset_electrodes_ = preset_electrodes.copy()
                preset_electrodes_[:, 0] = shank + 1
                preset_electrodes = np.vstack([preset_electrodes, preset_electrodes_])

        elif preset.startswith("tip_s") and len(preset) == 6 and preset[5].isdigit():
            # "tip_sX" with X in [0-3] - 0-383 on shank X
            shank = int(preset[5])
            for row in range(384):
                col = f"shank{shank}-bank0"
                electrode_id = wiring_df.loc[row, col]
                if not pd.isna(electrode_id):
                    preset_electrodes.append([shank, int(electrode_id)])

        elif preset == "tips_0_3":
            # 0-191 on shank 0 and 0-191 on shank 3
            for e in range(192):
                preset_electrodes.append([0, e])
            for e in range(192):
                preset_electrodes.append([3, e])

        elif preset == "tips_1_2":
            # 0-191 on shank 1 and 0-191 on shank 2
            for e in range(192):
                preset_electrodes.append([1, e])
            for e in range(192):
                preset_electrodes.append([2, e])

        elif preset.startswith("tip_b0_top_b1_s"):
            # "tip_sXb0_top_sXb1" with X in [0-3]
            shank = int(preset[-1])
            # 0-191 of bank 0, 192-383 of bank 1
            for e in range(192):
                preset_electrodes.append([shank, e])
            for e in range(192 + 384, 384 + 384):
                preset_electrodes.append([shank, e])

        elif preset.startswith("top_b0_tip_b1_s"):
            # "top_sXb0_tip_sXb1" with X in [0-3]
            shank = int(preset[-1])
            # 0-191 of bank 1, 192-383 of bank 0
            for e in range(192, 384):
                preset_electrodes.append([shank, e])
            for e in range(384, 192 + 384):
                preset_electrodes.append([shank, e])

        elif preset == "tip_s0b0_top_s2b0":
            # 0-191 of bank 0 shank 0, 192-383 of bank 0 shank 2
            for e in range(192):
                preset_electrodes.append([0, e])
            for e in range(192 + 384, 384 + 384):
                preset_electrodes.append([2, e])

        elif preset == "tip_s2b0_top_s0b0":
            # 0-191 of bank 0 shank 2, 192-383 of bank 0 shank 0
            for e in range(192):
                preset_electrodes.append([2, e])
            for e in range(192 + 384, 384 + 384):
                preset_electrodes.append([0, e])

        elif preset == "tip_s1b0_top_s3b0":
            # 0-191 of bank 0 shank 1, 192-383 of bank 0 shank 3
            for e in range(192):
                preset_electrodes.append([1, e])
            for e in range(192 + 384, 384 + 384):
                preset_electrodes.append([3, e])

        elif preset == "tip_s3b0_top_s1b0":
            # 0-191 of bank 0 shank 3, 192-383 of bank 0 shank 1
            for e in range(192):
                preset_electrodes.append([3, e])
            for e in range(192 + 384, 384 + 384):
                preset_electrodes.append([1, e])

        elif preset == "gliding_0-3":
            # 0-95 of shank 0, 96-191 of shank 1, 192-287 of shank 2, 288-383 of shank 3
            for e in range(96):
                preset_electrodes.append([0, e])
            for e in range(96, 192):
                preset_electrodes.append([1, e])
            for e in range(192, 288):
                preset_electrodes.append([2, e])
            for e in range(288, 384):
                preset_electrodes.append([3, e])

        elif preset == "gliding_3-0":
            # 0-95 of shank 3, 96-191 of shank 2, 192-287 of shank 1, 288-383 of shank 0
            for e in range(96):
                preset_electrodes.append([3, e])
            for e in range(96, 192):
                preset_electrodes.append([2, e])
            for e in range(192, 288):
                preset_electrodes.append([1, e])
            for e in range(288, 384):
                preset_electrodes.append([0, e])

        elif preset.startswith("zigzag_") and len(preset) == 8 and preset[7].isdigit():
            # For 2.0 bank 0: channels 0, 3, 4, 7, 8, 11, 12... on bank 0
            # Pattern: 0,3 then +4 repeatedly: 4,7,8,11,12,15,16,19...
            shank = int(preset[7])
            channels_bank0 = []
            i = 0
            while i < 384:  # Limit to 192 channels for bank 0
                channels_bank0.append(i)
                channels_bank0.append(i + 3)
                i += 4

            # For 2.0 bank 1: channels 1, 2, 5, 6, 8, 10, 13... on bank 0
            channels_bank1 = []
            i = 0
            while i < 384:  # Limit to 192 channels for bank 0
                channels_bank1.append(i + 1)
                channels_bank1.append(i + 2)
                i += 4

            electrodes_bank0 = wiring_df.loc[channels_bank0, "shank0-bank0"]
            electrodes_bank1 = wiring_df.loc[channels_bank1, "shank0-bank1"]
            electrodes_bank0 = np.vstack([electrodes_bank0 * 0 + shank, electrodes_bank0]).T
            electrodes_bank1 = np.vstack([electrodes_bank1 * 0 + shank, electrodes_bank1]).T
            preset_electrodes = np.vstack([electrodes_bank0, electrodes_bank1])

    return np.array(preset_electrodes, dtype=int)  # (n_electrodes, 2) array - [[shank_id, electrode_id], ...]


##########################
## Channel map plotting ##
##########################


def find_selected_electrodes(imro_list):
    "imro_list: list of tuples starting with (version, )"
    probe_type, n_channels = imro_list[0]
    if probe_type in PROBE_TYPE_MAP["1.0"]:
        probe_type = "1.0"
    elif probe_type in PROBE_TYPE_MAP["2.0-1shank"]:
        probe_type = "2.0-1shank"
    elif probe_type in PROBE_TYPE_MAP["2.0-4shanks"]:
        probe_type = "2.0-4shanks"
    elif probe_type in PROBE_TYPE_MAP["NXT"]:
        probe_type = "NXT"

    imro_table = np.array(imro_list[1:])
    if probe_type in ["1.0"]:
        selected_electrodes = imro_table[:, 0] + 384 * imro_table[:, 1]
        selected_shanks = selected_electrodes * 0
    elif probe_type in ["2.0-1shank"]:
        selected_electrodes = imro_table[:, -1]
        selected_shanks = selected_electrodes * 0
    elif probe_type in ["2.0-4shanks", "NXT"]:
        selected_electrodes = imro_table[:, -1]
        selected_shanks = imro_table[:, 1]
    selected_electrodes = np.vstack([selected_shanks, selected_electrodes]).T

    return selected_electrodes


def generate_kilosort_channelmap_dict(imro_list, positions_file):
    """
    Generate Kilosort-compatible channelmap dictionary from IMRO list.

    Args:
        imro_list: list of tuples starting with (version, n_channels) header
        positions_file: Path to CSV file containing electrode positions

    Returns:
        dict: Kilosort channelmap with keys: chanMap, xc, yc, kcoords, n_chan
    """
    # Get selected electrodes [[shank_id, electrode_id], ...]
    selected_electrodes = find_selected_electrodes(imro_list)

    # Load position data
    positions_df = pd.read_csv(positions_file)

    # Create mapping from (shank_id, electrode_id) to position
    n_selected = len(selected_electrodes)
    chanMap = np.arange(n_selected)  # Sequential channel indices
    xc = np.zeros(n_selected)
    yc = np.zeros(n_selected)
    kcoords = np.zeros(n_selected, dtype=int)  # shank IDs

    for i, (shank_id, electrode_id) in enumerate(selected_electrodes):
        # Find position for this electrode
        mask = (positions_df['shank'] == shank_id) & (positions_df['electrode'] == electrode_id)
        if not mask.any():
            raise ValueError(f"Electrode {electrode_id} on shank {shank_id} not found in positions file")

        row = positions_df[mask].iloc[0]
        xc[i] = row['x']
        yc[i] = row['y']
        kcoords[i] = shank_id

    kilosort_dict = {
        'chanMap': chanMap,
        'xc': xc,
        'yc': yc,
        'kcoords': kcoords,
        'n_chan': n_selected
    }

    return kilosort_dict


def plot_probe_layout(
    probe_type, imro_list, positions_file, wiring_file, title, figsize=(2, 30), save_plot=False, saveDir=None
):
    """
    Create visualization of probe layout with selected electrodes

    figsize: (width, height) per shank in inch

    """

    # Format parameters
    selected_color = "red"
    unselected_color = "lightgrey"
    forbidden_color = "black"

    if probe_type in ["1.0", "2.0-1shank"]:
        n_shanks = 1
    elif probe_type in ["2.0-4shanks", "NXT"]:
        n_shanks = 4
    electrode_vpitch = {"1.0": 20, "2.0-1shank": 15, "2.0-4shanks": 15, "NXT": 15}

    # Get physical electrode ids
    selected_electrodes = find_selected_electrodes(imro_list)

    # Define forbidden electrodes from wiring map
    wiring_df = pd.read_csv(wiring_file)
    forbidden_electrodes = find_forbidden_electrodes(selected_electrodes, wiring_df)

    # Load positions
    positions_df = pd.read_csv(positions_file)
    positions = positions_df.values

    # Single shank
    if n_shanks == 1:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Scale factor to make electrodes visible
        tick_width = 1
        shank_width = 20  # Shank width parameter
        electrode_width_ratio = 0.2  # Electrode width as fraction of shank width
        electrode_height = 10  # Reasonable height
        electrode_width = shank_width * electrode_width_ratio

        # Draw shank outline with pointy tip
        max_y = np.max(positions[:, -1])
        min_y = np.min(positions[:, -1])
        shank_height = max_y - min_y
        tip_height = shank_height * 0.05  # 5% of total shank height

        # Create polygon for shank shape (rectangle + triangle tip)
        shank_points = [
            (-shank_width // 2, max_y + 50),  # top left
            (shank_width // 2, max_y + 50),  # top right
            (shank_width // 2, min_y),  # bottom right at electrode level
            (0, min_y - tip_height),  # tip point
            (-shank_width // 2, min_y),  # bottom left at electrode level
        ]
        shank_poly_back = patches.Polygon(
            shank_points, linewidth=0, edgecolor="black", facecolor=(0.9, 0.9, 0.9, 0.2), zorder=-100
        )
        ax.add_patch(shank_poly_back)
        shank_poly_front = patches.Polygon(
            shank_points, linewidth=2, edgecolor="grey", facecolor=(0.9, 0.9, 0.9, 0), zorder=100
        )
        ax.add_patch(shank_poly_front)

        # Draw bank borders
        x_borders = [-shank_width // 2, shank_width // 2]
        for bank_i in np.arange(0, len(positions), 384):
            bank_y = positions[bank_i, -1] - electrode_vpitch[probe_type] // 2
            text_y = bank_y
            if bank_i == 0:
                text_y -= 50
            ax.plot(x_borders, [bank_y, bank_y], ls="-", lw=1.5, c="grey", zorder=-10)
            ax.text(
                x_borders[1] + tick_width * 2,
                text_y,
                f"Bank {bank_i // 384}\nonset",
                ha="left",
                va="center",
                fontsize=8,
                color="grey",
                zorder=-10,
            )

        # Draw electrodes
        for electrode, orig_x, y in positions[:, 1:]:
            if electrode in selected_electrodes[:, 1]:
                color = selected_color
                alpha = 1.0
            elif electrode in forbidden_electrodes[:, 1]:
                color = forbidden_color
                alpha = 1
            else:
                color = unselected_color
                alpha = 0.7

            # Map original x positions to shank width (normalized then scaled)
            if probe_type in ["1.0"]:
                # Original positions: 11,27,43,59 μm (range: 48 μm, center at 35)
                # Normalize to [-1, 1] range: (x - 35) / 24
                x_norm = (orig_x - 35) / 24
                x = x_norm * (shank_width * 0.8) / 2  # Use 80% of shank width
            else:  # linear
                # Original positions: 0,32 μm (range: 32 μm, center at 16)
                # Normalize to [-1, 1] range: (x - 16) / 16
                x_norm = (orig_x - 16) / 16
                x = x_norm * (shank_width * 0.6) / 2  # Use 60% of shank width

            rect = patches.Rectangle(
                (x - electrode_width // 2, y - electrode_height // 2),
                electrode_width,
                electrode_height,
                linewidth=0,
                edgecolor="black",
                facecolor=color,
                alpha=alpha,
            )
            ax.add_patch(rect)

        ax.set_xlim(-shank_width // 2 - 20, shank_width // 2 + 20)
        ax.set_ylim(min_y - tip_height - 50, max_y + 100)

        title = title.replace("_", " ")
        ax.set_title(title)

        # Style the plot - remove frame, grid, x-ticks
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_xticks([])

        # Remove all default ticks and labels
        ax.set_yticks([])

        # Add electrode index ticks on the left (every 50 electrodes)
        electrode_ticks = range(0, len(positions), 50)
        for electrode_idx in electrode_ticks:
            if electrode_idx < len(positions):
                y_pos = positions[electrode_idx, -1]
                # Tick mark on left edge of rightmost active shank
                ax.plot([-shank_width // 2, -shank_width // 2 - tick_width], [y_pos, y_pos], "k-", linewidth=1)
                # Label on left side
                ax.text(
                    -shank_width // 2 - tick_width * 2, y_pos, str(electrode_idx), ha="right", va="center", fontsize=8
                )

        # Add distance ticks on the right (every 500 μm)
        distance_ticks = range(0, int(max_y), 500)
        for distance in distance_ticks:
            # Tick mark on right edge of shank
            ax.plot([shank_width // 2, shank_width // 2 + tick_width], [distance, distance], "k-", linewidth=1)
            # Label on right side
            ax.text(shank_width // 2 + tick_width * 2, distance, f"{distance}", ha="left", va="center", fontsize=8)

    # multi-shank
    else:
        fig, ax = plt.subplots(1, 1, figsize=(figsize[0] * n_shanks * 0.8, figsize[1]))

        shank_width = 80
        shank_spacing = 150
        true_shank_spacing = 250
        electrode_width_ratio = 0.18
        electrode_height = 10
        electrode_width = shank_width * electrode_width_ratio
        tick_width = 5

        for shank_id in range(n_shanks):
            shank_m = positions[:, 0] == shank_id
            shank_m_selected = selected_electrodes[:, 0] == shank_id
            x_center = shank_id * shank_spacing

            # Draw shank outline with pointy tip
            max_y = np.max(positions[shank_m, -1])
            min_y = np.min(positions[shank_m, -1])
            shank_height = max_y - min_y
            tip_height = shank_height * 0.05  # 5% of total shank height

            # Create polygon for shank shape (rectangle + triangle tip)
            shank_points = [
                (x_center - shank_width // 2, max_y + 50),  # top left
                (x_center + shank_width // 2, max_y + 50),  # top right
                (x_center + shank_width // 2, min_y),  # bottom right at electrode level
                (x_center, min_y - tip_height),  # tip point
                (x_center - shank_width // 2, min_y),  # bottom left at electrode level
            ]
            shank_poly = patches.Polygon(shank_points, linewidth=2, edgecolor="black", facecolor="lightgray", alpha=0.2)
            ax.add_patch(shank_poly)
            shank_poly_front = patches.Polygon(
                shank_points, linewidth=2, edgecolor="grey", facecolor=(0.9, 0.9, 0.9, 0), zorder=100
            )
            ax.add_patch(shank_poly_front)

            # Draw bank borders
            x_borders = [x_center - shank_width // 2, x_center + shank_width // 2]
            for bank_i in np.arange(0, len(positions[shank_m]), 384):
                bank_y = positions[shank_m][bank_i, -1] - electrode_vpitch[probe_type] // 2
                text_y = bank_y
                if bank_i == 0:
                    text_y -= 50
                ax.plot(x_borders, [bank_y, bank_y], ls="-", lw=1.5, c="grey", zorder=-10)
                if shank_id == 3:
                    ax.text(
                        x_borders[1] + tick_width * 2,
                        text_y,
                        f"Bank {bank_i // 384}\nonset",
                        ha="left",
                        va="center",
                        fontsize=8,
                        color="grey",
                        zorder=-10,
                    )

            # Draw electrodes
            for shank_id_pos, electrode, orig_x, y in positions[shank_m]:
                if shank_id_pos != shank_id:
                    continue
                # Check if this electrode is selected
                if np.isin(shank_id_pos, selected_electrodes[shank_m_selected, 0]) & np.isin(
                    electrode, selected_electrodes[shank_m_selected, 1]
                ):
                    electrode_color = selected_color
                    alpha = 1.0
                elif np.isin(shank_id_pos, forbidden_electrodes[:, 0]) & np.isin(electrode, forbidden_electrodes[:, 1]):
                    electrode_color = forbidden_color
                    alpha = 1
                else:
                    electrode_color = unselected_color
                    alpha = 0.7

                # Map electrode positions to shank width (normalized then scaled)
                # Original positions: 0,32 μm (center at 16) - always 2.0
                x_norm = (orig_x - true_shank_spacing * shank_id - 16) / 16
                x = x_center + x_norm * (shank_width * 0.5) / 2

                rect = patches.Rectangle(
                    (x - electrode_width // 2, y - electrode_height // 2),
                    electrode_width,
                    electrode_height,
                    linewidth=0,
                    edgecolor="black",
                    facecolor=electrode_color,
                    alpha=alpha,
                )
                ax.add_patch(rect)

            # Add shank label
            ax.text(
                x_center,
                min_y - tip_height - tip_height * 0.2,
                f"Shank {shank_id}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )

        # Get overall bounds for multi-shank layout

        overall_max_y = np.max(positions[:, -1])
        overall_min_y = np.min(positions[:, -1])
        overall_tip_height = (overall_max_y - overall_min_y) * 0.05

        ax.set_xlim(-shank_width, n_shanks * shank_spacing)
        ax.set_ylim(overall_min_y - overall_tip_height - 50, overall_max_y + 100)

        title = title.replace("_", " ")
        ax.set_title(title)

        # Style the plot - remove frame, grid, x-ticks
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_xticks([])

        # Remove all default ticks and labels
        ax.set_yticks([])

        # For multi-shank, add ticks to the leftmost and rightmost shank
        leftmost_x = 0 * shank_spacing - shank_width // 2
        rightmost_x = 3 * shank_spacing + shank_width // 2
        leftmost_positions = positions_df.loc[positions_df.shank == 0, "x":"y"].values
        rightmost_positions = positions_df.loc[positions_df.shank == 3, "x":"y"].values

        # Add electrode index ticks on the left of leftmost shank (every 50 electrodes)
        electrode_ticks = range(0, len(leftmost_positions), 50)
        for electrode_idx in electrode_ticks:
            if electrode_idx < len(rightmost_positions):
                y_pos = rightmost_positions[electrode_idx][1]
                # Tick mark on left edge of rightmost active shank
                ax.plot([leftmost_x, leftmost_x - tick_width], [y_pos, y_pos], "k-", linewidth=1)
                # Label on left side
                ax.text(leftmost_x - 2 * tick_width, y_pos, str(electrode_idx), ha="right", va="center", fontsize=8)

        # Add distance ticks on the right of rightmost active shank (every 500 μm)
        distance_ticks = range(0, int(overall_max_y), 500)
        for distance in distance_ticks:
            # Tick mark on right edge of rightmost active shank
            ax.plot([rightmost_x, rightmost_x + tick_width], [distance, distance], "k-", linewidth=1)
            # Label on right side
            ax.text(rightmost_x + 2 * tick_width, distance, f"{distance}", ha="left", va="center", fontsize=8)

    ax.set_ylabel("Vertical position (channel/μm)")
    # Remove grid
    ax.grid(False)

    # Add legend positioned away from probe
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=selected_color, markersize=10, label="Selected"),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=unselected_color,
            markersize=10,
            alpha=0.7,
            label="Unselected",
        ),
        Line2D(
            [0], [0], marker="s", color="w", markerfacecolor=forbidden_color, markersize=10, alpha=1, label="Forbidden"
        ),
    ]
    ax.legend(title="Electrode state:", handles=legend_elements, bbox_to_anchor=(0.9, 0))

    plt.tight_layout()

    if save_plot:
        if saveDir is None:
            saveDir = Path.cwd()
        pdf_filename = "_".join(title.replace("\n", " ").split(" ")) + ".pdf"
        plt.savefig(Path(saveDir) / pdf_filename, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {pdf_filename}")
