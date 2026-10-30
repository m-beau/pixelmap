"""
Core functionality tests for JOSS review.

These tests demonstrate that the software's core functionality works correctly:
- Hardware constraint validation
- Electrode selection with presets
- IMRO file generation for all supported probe types
- File I/O round-trip consistency
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from pixelmap.backend import (
    _load_wiring_maps,
    _verify_hardware_violations,
    find_forbidden_electrodes,
    get_electrodes,
    make_wiring_maps,
)
from pixelmap.constants import (
    WIRING_FILE_MAP,
    SUPPORTED_1shank_PRESETS,
    SUPPORTED_4shanks_PRESETS,
)
from pixelmap.gui.gui import Electrodes
from pixelmap.utils.imro import (
    generate_imro_channelmap,
    read_imro_file,
    save_to_imro_file,
)


class TestHardwareConstraints:
    """Test that hardware wiring constraints are correctly enforced."""

    def test_forbidden_electrodes_are_detected(self, wiring_df_1_0):
        """Test that selecting an electrode identifies conflicting electrodes."""
        selected = np.array([[0, 0]])
        forbidden = find_forbidden_electrodes(selected, wiring_df_1_0)

        assert isinstance(forbidden, np.ndarray)
        assert len(forbidden) > 0
        # The selected electrode should not be in its own forbidden list
        assert not any(f[0] == 0 and f[1] == 0 for f in forbidden)

    def test_too_many_electrodes_rejected(self, wiring_df_1_0):
        """Test that exceeding electrode limit per shank is rejected."""
        max_n = 384  # NP 1.0 readout cap per shank
        too_many = np.array([[0, i] for i in range(max_n + 10)])

        with pytest.raises(AssertionError, match="too many electrodes"):
            _verify_hardware_violations("1.0", too_many, wiring_df_1_0)

    def test_valid_selection_accepted(self, wiring_df_1_0):
        """Test that valid electrode selections pass validation."""
        valid_selection = np.array([[0, i] for i in range(0, 100, 10)])

        # Should not raise
        _verify_hardware_violations("1.0", valid_selection, wiring_df_1_0)


class TestPresetConfigurations:
    """Test that all preset configurations work correctly."""

    @pytest.mark.parametrize("preset", SUPPORTED_1shank_PRESETS)
    def test_single_shank_presets(self, preset, wiring_df_1_0):
        """Test all single-shank presets generate valid electrode selections."""
        electrodes = get_electrodes("1.0", wiring_df_1_0, preset=preset)

        assert isinstance(electrodes, np.ndarray)
        assert len(electrodes) > 0
        assert len(electrodes) <= 384  # NP 1.0 readout cap per shank
        assert electrodes.shape[1] == 2  # [shank_id, electrode_id] pairs

    @pytest.mark.parametrize("preset", SUPPORTED_4shanks_PRESETS)
    def test_four_shank_presets(self, preset, wiring_df_2_0_4shanks):
        """Test all four-shank presets generate valid electrode selections."""
        electrodes = get_electrodes("2.0-4shanks", wiring_df_2_0_4shanks, preset=preset)

        assert isinstance(electrodes, np.ndarray)
        assert len(electrodes) > 0
        # Note: Some presets may select more electrodes than can be recorded simultaneously
        # The IMRO generation will warn about this, but the selection itself is valid
        assert electrodes.shape[1] == 2

    def test_custom_electrode_selection(self, wiring_df_1_0):
        """Test that custom electrode selection works."""
        custom = np.array([[0, 10], [0, 20], [0, 30]])
        electrodes = get_electrodes("1.0", wiring_df_1_0, custom_electrodes=custom)

        assert np.array_equal(electrodes, custom)


class TestIMROFileGeneration:
    """Test IMRO file generation for all probe types."""

    def test_generate_imro_1_0_probe(self, wiring_maps_dir):
        """Test IMRO generation for Neuropixels 1.0 probe."""
        wiring_file = wiring_maps_dir / WIRING_FILE_MAP["1.0"][1]

        imro_list = generate_imro_channelmap(
            probe_type="1.0",
            layout_preset="Tip",
            reference_id="External",
            wiring_file=str(wiring_file),
            ap_gain=500,
            lf_gain=250,
            hp_filter=1
        )

        assert isinstance(imro_list, list)
        assert len(imro_list) > 1  # Header + electrodes
        header = imro_list[0]
        assert isinstance(header[0], (int, str))  # Probe subtype (int legacy or str part number)
        assert isinstance(header[1], int)  # Number of channels

    def test_generate_imro_2_0_1shank(self, wiring_maps_dir):
        """Test IMRO generation for Neuropixels 2.0 single-shank probe."""
        wiring_file = wiring_maps_dir / WIRING_FILE_MAP["2.0-1shank"][1]

        imro_list = generate_imro_channelmap(
            probe_type="2.0-1shank",
            layout_preset="Tip",
            reference_id="External",
            wiring_file=str(wiring_file)
        )

        assert isinstance(imro_list, list)
        assert len(imro_list) > 1

    def test_generate_imro_2_0_4shanks(self, wiring_maps_dir):
        """Test IMRO generation for Neuropixels 2.0 four-shank probe."""
        wiring_file = wiring_maps_dir / WIRING_FILE_MAP["2.0-4shanks"][1]

        # Use tip_s0 preset which selects appropriate number of electrodes
        imro_list = generate_imro_channelmap(
            probe_type="2.0-4shanks",
            layout_preset="tip_s0",
            reference_id="External",
            wiring_file=str(wiring_file)
        )

        assert isinstance(imro_list, list)
        assert len(imro_list) > 1


class TestFileIO:
    """Test IMRO file reading and writing."""

    def test_save_and_load_imro_file(self, tmp_imro_file, wiring_maps_dir):
        """Test that IMRO files can be saved and loaded correctly."""
        wiring_file = wiring_maps_dir / WIRING_FILE_MAP["1.0"][1]

        # Generate IMRO list
        imro_list = generate_imro_channelmap(
            probe_type="1.0",
            layout_preset="Tip",
            reference_id="External",
            wiring_file=str(wiring_file),
            ap_gain=500,
            lf_gain=250,
            hp_filter=1
        )

        # Save to file
        save_to_imro_file(imro_list, str(tmp_imro_file))
        assert tmp_imro_file.exists()

        # Load back
        loaded = read_imro_file(str(tmp_imro_file))

        # Verify round-trip consistency
        assert loaded == imro_list

    def test_load_sample_files(self):
        """Test loading sample IMRO files."""
        fixtures_dir = Path(__file__).parent / "fixtures"

        # Test NP1.0 sample
        sample_1_0 = fixtures_dir / "sample_1.0.imro"
        if sample_1_0.exists():
            imro_list = read_imro_file(str(sample_1_0))
            assert len(imro_list) > 0
            assert imro_list[0][0] == 0  # Probe type 0

        # Test NP2.0-1shank sample
        sample_2_0_1s = fixtures_dir / "sample_2.0-1shank.imro"
        if sample_2_0_1s.exists():
            imro_list = read_imro_file(str(sample_2_0_1s))
            assert len(imro_list) > 0
            assert imro_list[0][0] == 21  # Probe type 21

        # Test NP2.0-4shanks sample
        sample_2_0_4s = fixtures_dir / "sample_2.0-4shanks.imro"
        if sample_2_0_4s.exists():
            imro_list = read_imro_file(str(sample_2_0_4s))
            assert len(imro_list) > 0
            assert imro_list[0][0] == 24  # Probe type 24


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    def test_complete_workflow_np1_0(self, wiring_maps_dir, tmp_path):
        """Test complete workflow for NP1.0: preset -> generate -> save -> load."""
        wiring_file = wiring_maps_dir / WIRING_FILE_MAP["1.0"][1]
        output_file = tmp_path / "test_np1.0.imro"

        # Generate with preset
        imro_list = generate_imro_channelmap(
            probe_type="1.0",
            layout_preset="Tip",
            reference_id="External",
            wiring_file=str(wiring_file),
            ap_gain=500,
            lf_gain=250,
            hp_filter=1
        )

        # Save
        save_to_imro_file(imro_list, str(output_file))

        # Load and verify
        loaded = read_imro_file(str(output_file))
        assert loaded == imro_list

    def test_complete_workflow_custom_electrodes(self, wiring_maps_dir, tmp_path):
        """Test workflow with custom electrode selection."""
        wiring_file = wiring_maps_dir / WIRING_FILE_MAP["1.0"][1]
        output_file = tmp_path / "test_custom.imro"

        # Custom selection - smaller number to avoid wiring conflicts
        custom = np.array([[0, 0], [0, 100], [0, 200]])

        # Generate
        imro_list = generate_imro_channelmap(
            probe_type="1.0",
            custom_electrodes=custom,
            reference_id="External",
            wiring_file=str(wiring_file),
            ap_gain=500,
            lf_gain=250,
            hp_filter=1
        )

        # Save
        save_to_imro_file(imro_list, str(output_file))

        # Verify
        assert output_file.exists()
        loaded = read_imro_file(str(output_file))
        assert len(loaded) == len(custom) + 1  # +1 for header

    def test_multiple_presets_same_probe(self, wiring_maps_dir, tmp_path):
        """Test generating multiple configurations for the same probe."""
        wiring_file = wiring_maps_dir / WIRING_FILE_MAP["1.0"][1]

        for preset in ["Tip", "zigzag"]:
            output_file = tmp_path / f"test_{preset}.imro"

            imro_list = generate_imro_channelmap(
                probe_type="1.0",
                layout_preset=preset,
                reference_id="External",
                wiring_file=str(wiring_file),
                ap_gain=500,
                lf_gain=250,
                hp_filter=1
            )

            save_to_imro_file(imro_list, str(output_file))
            assert output_file.exists()


class TestWiringMapsSharing:
    """make_wiring_maps must return one shared, process-wide object per
    directory instead of unpickling a fresh copy on every call."""

    @pytest.fixture(autouse=True)
    def _isolated_cache(self):
        """Start and end each test with an empty cache so no test sees another's map."""
        _load_wiring_maps.cache_clear()
        yield
        _load_wiring_maps.cache_clear()

    def test_make_wiring_maps_returns_same_object_for_path_and_str(self, wiring_maps_dir):
        by_path = make_wiring_maps(wiring_maps_dir)
        by_str = make_wiring_maps(str(wiring_maps_dir))

        assert by_path is by_str

        # Sanity: still contains every probe type in WIRING_FILE_MAP.
        assert set(by_path.keys()) == set(WIRING_FILE_MAP.keys())

    def test_make_wiring_maps_returns_same_object_across_calls(self, wiring_maps_dir):
        first = make_wiring_maps(wiring_maps_dir)
        second = make_wiring_maps(wiring_maps_dir)

        assert first is second

    def test_make_wiring_maps_cache_clear_forces_reload(self, wiring_maps_dir):
        """Tests (or callers) that mutate a wiring_maps_dir between calls
        must clear the cache to see a fresh reload."""
        first = make_wiring_maps(wiring_maps_dir)
        _load_wiring_maps.cache_clear()
        second = make_wiring_maps(wiring_maps_dir)

        assert first is not second
        assert set(first.keys()) == set(second.keys())

    def test_make_wiring_maps_rejects_missing_directory(self, tmp_path):
        with pytest.raises(AssertionError, match="does not exist"):
            make_wiring_maps(tmp_path / "missing")

    def test_make_wiring_maps_is_read_only(self, wiring_maps_dir):
        """The map is shared by every session, so mutating it must fail loudly."""
        wiring_maps = make_wiring_maps(wiring_maps_dir)
        conflicts = wiring_maps["1.0"]
        electrode, conflicting = next((e, c) for e, c in conflicts.items() if c)

        with pytest.raises(TypeError):
            wiring_maps["1.0"] = {}
        with pytest.raises(TypeError):
            conflicts[electrode] = set()
        with pytest.raises(TypeError):
            del conflicts[electrode]
        with pytest.raises(AttributeError):
            conflicting.add(electrode)
        with pytest.raises(AttributeError):
            conflicting.clear()

    def test_electrodes_selection_leaves_shared_map_untouched(self, wiring_maps_dir):
        wiring_map = make_wiring_maps(wiring_maps_dir)["1.0"]
        electrode, conflicting = next((e, c) for e, c in wiring_map.items() if c)
        snapshot = {e: set(c) for e, c in wiring_map.items()}

        first = Electrodes(wiring_map, n_maximum_electrodes=384)
        second = Electrodes(wiring_map, n_maximum_electrodes=384)
        first.push_undo_snapshot()
        first.select(electrode)

        assert electrode in first.selected
        assert conflicting <= first.unavailable
        assert isinstance(first.available, set) and isinstance(first.unavailable, set)
        assert second.selected == set() and second.unavailable == set()

        first.undo()
        assert first.selected == set() and first.unavailable == set()
        first.redo()
        first.deselect(electrode)
        assert first.selected == set() and first.unavailable == set()
        assert {e: set(c) for e, c in wiring_map.items()} == snapshot

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_rebuilt_wiring_maps_are_pickled_and_reloaded(self, wiring_maps_dir, tmp_path, monkeypatch):
        """A rebuild must still write a loadable pickle, and both paths return the read-only map."""
        # Rebuilding every probe type takes minutes; one probe exercises the same code.
        file_map = {"1.0": WIRING_FILE_MAP["1.0"]}
        monkeypatch.setattr("pixelmap.backend.WIRING_FILE_MAP", file_map)
        for filename in file_map["1.0"]:
            shutil.copy(wiring_maps_dir / filename, tmp_path / filename)

        built = make_wiring_maps(tmp_path)
        assert (tmp_path / "wiring_maps.pkl").exists()
        _load_wiring_maps.cache_clear()
        loaded = make_wiring_maps(tmp_path)

        assert built is not loaded
        for wiring_maps in (built, loaded):
            assert set(wiring_maps) == {"1.0"}
            with pytest.raises(TypeError):
                wiring_maps["1.0"] = {}
        assert dict(built["1.0"]) == dict(loaded["1.0"])
