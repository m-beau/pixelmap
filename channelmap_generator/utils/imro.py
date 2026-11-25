################################
## IMRO file I/O utilities ##
################################


import numpy as np
import pandas as pd

import panel as pn

from channelmap_generator.backend import format_imro_string, get_electrodes
from channelmap_generator.constants import PROBE_N, PROBE_TYPE_MAP, REF_ELECTRODES

def save_to_imro_file(imro_list, filename="channelmap.imro"):
    """
    Save IMRO list to a text file in SpikeGLX format.

    Args:
        imro_list: List of tuples from generate_imro_channelmap
        filename: Output filename (default: "channelmap.imro")
    """
    if ".imro" not in filename:
        filename = filename + ".imro"
    filename = "_".join(filename.replace("\n", " ").split(" "))

    with open(filename, "w") as f:
        # Write header
        header = imro_list[0]
        f.write(f"({header[0]},{header[1]})")

        # Write channel entries
        for entry in imro_list[1:]:
            # Format tuple as space-separated values in parentheses
            entry_str = " ".join(str(x) for x in entry)
            f.write(f"({entry_str})")

        # Add newline at end of file
        f.write("\n")

    print(f"IMRO file saved: {filename}")


def parse_imro_file(content):
    """
    Parse IMRO file content and return imro_list format.

    Args:
        content: str, contents of .imro file

    Returns:
        imro_list: List of tuples matching generate_imro_channelmap output
    """
    # Split by ')(' to get individual entries
    entries = content.split(")(")

    # Clean up parentheses from first and last entries
    entries[0] = entries[0].lstrip("(")
    entries[-1] = entries[-1].rstrip(")")

    # Parse header (first entry: "24,384")
    header_parts = entries[0].split(",")
    header = (int(header_parts[0]), int(header_parts[1]))

    # Parse channel entries (format: "0 1 0 2 288")
    channel_entries = []
    for entry_str in entries[1:]:
        values = [int(x) for x in entry_str.split()]
        channel_entries.append(tuple(values))

    return [header] + channel_entries


def read_imro_file(filepath):
    """
    Read IMRO file and return imro_list format.

    Args:
        filepath: Path to .imro file

    Returns:
        imro_list: List of tuples matching generate_imro_channelmap output
    """
    with open(filepath, "r") as f:
        content = f.read().strip()

    return parse_imro_file(content)


def parse_imro_list(imro_list):
    """
    Parse imro_list to extract electrode selection and parameters.

    Args:
        imro_list: List from read_imro_file or generate_imro_channelmap

    Returns:
        tuple: (selected_electrodes, probe_type, probe_subtype, reference_id, ap_gain, lf_gain, hp_filter)
               selected_electrodes: numpy array of (shank_id, electrode_id) pairs
               probe_type: "1.0", "2.0-1shank", "2.0-4shanks", or "NXT"
               Other parameters: as used in original generation
    """

    header = imro_list[0]
    probe_subtype = header[0]
    entries = imro_list[1:]

    # Determine probe type from subtype
    if probe_subtype in PROBE_TYPE_MAP["1.0"]:
        probe_type = "1.0"
    elif probe_subtype in PROBE_TYPE_MAP["2.0-1shank"]:
        probe_type = "2.0-1shank"
    elif probe_subtype in PROBE_TYPE_MAP["2.0-4shanks"]:
        probe_type = "2.0-4shanks"
    # elif probe_subtype in PROBE_TYPE_MAP["NXT"]:
    #     probe_type = "NXT"

    selected_electrodes = []

    if probe_type == "1.0":
        # Format: (channel, bank, ref, ap_gain, lf_gain, hp_filter)
        check_entry_elements(6,  entries[0], probe_type)
        
        reference_id = entries[0][2]  # Same for all entries
        reference_string = ref_id_to_string(reference_id, probe_subtype)

        ap_gain = entries[0][3]
        lf_gain = entries[0][4]
        hp_filter = entries[0][5]

        # Extract electrodes: channel + 384*bank gives electrode_id, shank is always 0
        for entry in entries:
            channel, bank = entry[0], entry[1]
            electrode_id = channel + 384 * bank
            selected_electrodes.append([0, electrode_id])

    elif probe_type == "2.0-1shank":
        # Format: (channel, bank_mask, ref, electrode_id)
        check_entry_elements(4,  entries[0], probe_type)

        reference_id = entries[0][2]
        reference_string = ref_id_to_string(reference_id, probe_subtype)

        ap_gain = lf_gain = hp_filter = None  # Not used in 2.0 IMRO tables

        # Extract electrodes: electrode_id is directly stored, shank is always 0
        for entry in entries:
            electrode_id = entry[3]
            selected_electrodes.append([0, electrode_id])

    else:  # 2.0-4shanks or NXT in the future
        # Format: (channel, shank_id, bank, ref, electrode_id)
        check_entry_elements(5,  entries[0], probe_type)

        # Need the 2 first reference IDs to handle join_tips referencing
        reference_ids = [entries[0][3], entries[1][3]]
        if reference_ids[0] != reference_ids[1]:
            reference_string = 'Join Tips'
        else:
            reference_id = reference_ids[0]
            reference_string = ref_id_to_string(reference_id, probe_subtype)

        ap_gain = lf_gain = hp_filter = None  # Not used in 2.0 IMRO tables

        # Extract electrodes: shank_id and electrode_id are directly stored
        for entry in entries:
            shank_id = entry[1]
            electrode_id = entry[4]
            selected_electrodes.append([shank_id, electrode_id])

    selected_electrodes = np.array(selected_electrodes, dtype=int)

    return (selected_electrodes,
            probe_type,
            probe_subtype,
            reference_string,
            ap_gain,
            lf_gain,
            hp_filter)


def ref_id_to_string(reference_id, probe_subtype):
    ref_map = {v: k for k, v in REF_ELECTRODES[probe_subtype].items() if k != "Join Tips"}
    if reference_id not in ref_map:
        error_message = f"Unexpected reference id {reference_id} for probe subtype {probe_subtype}!"
        pn.state.notifications.error(error_message,
                                    duration=10_000)
        raise AssertionError(error_message)
    return ref_map[reference_id]


def check_entry_elements(n_expected_elements, entry, probe_type):
    n_entry_elements = len(entry)
    if n_entry_elements != n_expected_elements:
        error_message = f"Corrupt .imro file - IMRO table entries for {probe_type} probes should have {n_expected_elements} elements, not {n_entry_elements}."
        pn.state.notifications.error(error_message,
                                duration=10_000)
        raise AssertionError(error_message)


def generate_imro_channelmap(
    probe_type,
    layout_preset=None,
    reference_id="External",
    probe_subtype=None,
    custom_electrodes=None,
    wiring_file=None,
    ap_gain=500,
    lf_gain=250,
    hp_filter=1,
):
    """
    Generate IMRO-formatted channelmap for Neuropixels probes.

    Args:
        probe_type: Type of probe ("1.0", "2.0-1shank", "2.0-4shanks", "NXT")
        layout_preset: Preset layout configuration
        reference_id: Reference electrode selection ('External', 'Tip', 'Ground')
        probe_subtype: Specific SpikeGLX type number (optional)
        custom_electrodes: list of custom (shank_id, electrode_id) pairs (overrides preset)
        positions_file: Path to positions CSV file
        wiring_file: Path to wiring CSV file
        ap_gain: AP band gain (for 1.0 probes)
        lf_gain: LF band gain (for 1.0 probes)
        hp_filter: High-pass filter setting (for 1.0 probes)

    Returns:
        IMRO-formatted string for channelmap
    """

    # 1) Process probe type and load CSVs
    if probe_subtype is None:
        probe_subtype = PROBE_TYPE_MAP[probe_type][0]

    wiring_df = pd.read_csv(wiring_file)

    # 2) Select electrodes from presets or custom
    selected_electrodes = get_electrodes(probe_type, wiring_df, layout_preset, custom_electrodes)

    # 3) Generate IMRO table with appropriate format
    imro_list = format_imro_string(
        selected_electrodes, wiring_df, probe_type, probe_subtype, reference_id, ap_gain, lf_gain, hp_filter
    )

    n_selected = len(imro_list) - 1
    n_possible = PROBE_N[probe_type]["n"]

    if n_selected != n_possible:
        print(
            f"\n!! WARNING !!\nYou selected {n_selected} electrodes, but {probe_type} probes must record from {n_possible} simultaneously!\n"
        )

    return imro_list
