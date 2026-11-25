"""Tests for the per-electrode table behind the *Table* (CSV) download."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pixelmap import backend
from pixelmap.constants import (
    PROBE_TYPE_MAP,
    WIRING_FILE_MAP,
    SUPPORTED_1shank_PRESETS,
    SUPPORTED_4shanks_PRESETS,
    SUPPORTED_QuadBase_PRESETS,
)
from pixelmap.utils import imro

WIRING_DIR = Path(backend.__file__).parent / "wiring_maps"
PROBE_TYPES = ["1.0", "2.0-1shank", "2.0-4shanks", "QuadBase"]
PRESETS = {
    "1.0": SUPPORTED_1shank_PRESETS,
    "2.0-1shank": SUPPORTED_1shank_PRESETS,
    "2.0-4shanks": SUPPORTED_4shanks_PRESETS,
    "QuadBase": SUPPORTED_QuadBase_PRESETS,
}
# Where the bank sits in an IMRO entry, for the formats that write it plainly
# (single-shank 2.0 writes a bank *mask*).
BANK_FIELD = {"1.0": 1, "2.0-4shanks": 2, "QuadBase": 2}


def _frames(probe_type):
    positions_file, wiring_file = WIRING_FILE_MAP[probe_type]
    return pd.read_csv(WIRING_DIR / positions_file), pd.read_csv(WIRING_DIR / wiring_file)


@pytest.mark.parametrize("probe_type", PROBE_TYPES)
def test_one_row_per_electrode_in_position_order(probe_type):
    positions, wiring = _frames(probe_type)
    table = backend.electrode_table(positions, wiring)

    assert list(table.columns) == ["shank", "electrode", "channel", "bank", "x_um", "z_um"]
    assert len(table) == len(positions)
    assert table[["shank", "electrode"]].values.tolist() == positions[["shank", "electrode"]].values.tolist()
    assert table["x_um"].tolist() == positions["x"].tolist()
    assert table["z_um"].tolist() == positions["y"].tolist()
    assert not table.isna().any().any()
    assert table["channel"].between(0, len(wiring) - 1).all()


@pytest.mark.parametrize("probe_type", PROBE_TYPES)
def test_channel_and_bank_are_those_the_imro_writes(probe_type):
    """The table's wiring must be the IMRO's, electrode for electrode."""
    positions, wiring = _frames(probe_type)
    table = backend.electrode_table(positions, wiring).set_index(["shank", "electrode"])
    _, wiring_file = WIRING_FILE_MAP[probe_type]
    preset = next(p for p in PRESETS[probe_type] if p)
    imro_list = imro.generate_imro_channelmap(
        probe_type=probe_type, layout_preset=preset, wiring_file=WIRING_DIR / wiring_file,
        probe_subtype=PROBE_TYPE_MAP[probe_type][0],
    )
    selected = backend.find_selected_electrodes(imro_list)
    assert len(selected) == len(imro_list) - 1

    for (shank_id, electrode_id), entry in zip(selected, imro_list[1:]):
        row = table.loc[(int(shank_id), int(electrode_id))]
        assert row["channel"] == entry[0]
        if probe_type in BANK_FIELD:
            assert row["bank"] == entry[BANK_FIELD[probe_type]]
