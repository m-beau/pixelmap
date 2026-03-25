"""
Pytest fixtures shared across all tests.

Provides common test data including wiring DataFrames for all probe types
and sample electrode selections for testing.
"""

from pathlib import Path

import pandas as pd
import pytest

from pixelmap.constants import WIRING_FILE_MAP


@pytest.fixture
def wiring_maps_dir():
    """Path to wiring maps directory."""
    return Path(__file__).parent.parent / "pixelmap" / "wiring_maps"


@pytest.fixture
def wiring_df_1_0(wiring_maps_dir):
    """Load wiring DataFrame for Neuropixels 1.0 probe."""
    wiring_file = wiring_maps_dir / WIRING_FILE_MAP["1.0"][1]
    return pd.read_csv(wiring_file)


@pytest.fixture
def wiring_df_2_0_1shank(wiring_maps_dir):
    """Load wiring DataFrame for Neuropixels 2.0 single-shank probe."""
    wiring_file = wiring_maps_dir / WIRING_FILE_MAP["2.0-1shank"][1]
    return pd.read_csv(wiring_file)


@pytest.fixture
def wiring_df_2_0_4shanks(wiring_maps_dir):
    """Load wiring DataFrame for Neuropixels 2.0 four-shank probe."""
    wiring_file = wiring_maps_dir / WIRING_FILE_MAP["2.0-4shanks"][1]
    return pd.read_csv(wiring_file)


@pytest.fixture
def sample_electrodes_1shank():
    """Sample valid electrode selection for single-shank probe."""
    # Simple selection: first 10 electrodes on shank 0
    return [[0, i] for i in range(0, 20, 2)]


@pytest.fixture
def sample_electrodes_4shanks():
    """Sample valid electrode selection for four-shank probe."""
    # 5 electrodes from each shank
    electrodes = []
    for shank in range(4):
        for electrode in range(0, 10, 2):
            electrodes.append([shank, electrode])
    return electrodes


@pytest.fixture
def tmp_imro_file(tmp_path):
    """Provide temporary file path for IMRO file testing."""
    return tmp_path / "test_channelmap.imro"
