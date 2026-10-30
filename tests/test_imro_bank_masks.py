"""Regression coverage for NP2 single-shank bank masks (issue #77)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pixelmap.constants import WIRING_FILE_MAP
from pixelmap.utils.imro import generate_imro_channelmap, read_imro_file, save_to_imro_file


@pytest.mark.parametrize("subtype", ["NP2000", 21, "NP2003", 2003])
@pytest.mark.parametrize("selection", ["bank0", "mixed_banks"])
def test_np2_single_shank_bank_masks(subtype, selection, wiring_maps_dir, tmp_path):
    """Modern and legacy headers must export nonzero, correct bank masks."""
    wiring_file = wiring_maps_dir / WIRING_FILE_MAP["2.0-1shank"][1]
    fixture = Path(__file__).parent / "fixtures" / f"np2_{selection}.imro"
    expected_bytes = fixture.read_bytes().replace(b"(NP2003,", f"({subtype},".encode(), 1)
    expected = read_imro_file(fixture)
    expected[0] = (subtype, 384)

    params = {"layout_preset": "Tip"}
    if selection == "mixed_banks":
        # One raw CSV cell per channel avoids conflicts. Reverse the input
        # order to verify channel sorting as well as all four bank masks.
        wiring = pd.read_csv(wiring_file)
        electrodes = []
        for row_index, row in wiring.iterrows():
            bank = row_index % 4
            value = row[f"shank0-bank{bank}"]
            if pd.isna(value):
                value = row["shank0-bank0"]
            electrodes.append([0, int(value)])
        params = {"custom_electrodes": np.array(electrodes[::-1])}

    actual = generate_imro_channelmap(
        probe_type="2.0-1shank",
        probe_subtype=subtype,
        reference_id="External",
        wiring_file=str(wiring_file),
        **params,
    )
    assert actual == expected
    assert {entry[1] for entry in actual[1:]} == ({1} if selection == "bank0" else {1, 2, 4, 8})
    assert all(entry[1] == 1 << (entry[3] // 384) for entry in actual[1:])
    destination = tmp_path / "channelmap.imro"
    save_to_imro_file(actual, str(destination))
    assert destination.read_bytes() == expected_bytes
    assert read_imro_file(destination) == actual
