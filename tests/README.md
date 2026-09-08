# Test Suite Documentation

This directory contains the automated test suite for PixelMap (Neuropixels Channelmap Generator).

## Overview

The test suite validates PixelMap end to end: reliable generation of IMRO files for Neuropixels probes, and both power features — the activity survey ⚡ and anatomical 🧠 overlays — at the computation layer *and* through the GUI handlers users actually trigger. Tests are designed to be comprehensive yet fast (the whole suite runs in a few seconds and needs no network access), providing confidence that the software works correctly.

## Test Structure

### `conftest.py`
Shared pytest fixtures providing:
- Wiring DataFrames for all probe types (1.0, 2.0-1shank, 2.0-4shanks)
- Sample electrode selections
- Temporary file paths for testing file I/O

### `test_core_functionality.py`
Main test file containing 41 tests organized into 5 test classes:

#### 1. **TestHardwareConstraints** (3 tests)
Validates that hardware wiring constraints are correctly enforced:
- Forbidden electrode detection when electrodes share ADC wiring
- Rejection of selections exceeding per-shank electrode limits
- Acceptance of valid electrode selections

#### 2. **TestPresetConfigurations** (30 tests)
Parametrized tests ensuring all preset configurations work:
- All 4 single-shank presets (Tip, tip_b0_top_b1, top_b0_tip_b1, zigzag)
- All 25 four-shank presets (tips_all, tip_s0-3, tips_0_3, gliding, zigzag_0-3, etc.)
- Custom electrode selection

#### 3. **TestIMROFileGeneration** (3 tests)
Tests IMRO file generation for all supported probe types:
- Neuropixels 1.0
- Neuropixels 2.0 single-shank
- Neuropixels 2.0 four-shank

#### 4. **TestFileIO** (2 tests)
Validates file reading/writing operations:
- Round-trip consistency (save then load)
- Loading sample IMRO files from fixtures

#### 5. **TestEndToEndWorkflows** (3 tests)
Complete workflow tests simulating real usage:
- Preset selection → IMRO generation → file saving → file loading
- Custom electrode selection workflow
- Multiple presets for the same probe

### Activity survey overlay ⚡

#### `test_survey.py` (17 tests)
The survey computation layer (`utils/survey.py`):
- Parsing SpikeGLX-exported survey `.txt` files (header validation, dtypes, malformed input)
- Matching survey rows to probe electrodes, including the shank-local vs global x-offset convention
- Rejecting surveys exported from a different probe type
- `default_survey_range()` — the percentile-based initial colormap bounds (heavy-tailed
  surveys, constant surveys, non-finite values, empty input)

### Anatomical overlay 🧠

Every anatomy test runs against a **small fake atlas** patched over
`BrainGlobeAtlas`, so the suite never downloads atlas data and stays offline-safe.

#### `test_anatomy_atlas.py` (10 tests)
The BrainGlobe wrapper (`anatomy/atlas.py`): voxel indexing, out-of-bounds and
unannotated lookups, region metadata resolution, and reorientation of atlases whose
native voxel orientation is not Allen's `asr`.

#### `test_anatomy_transform.py` (18 tests)
The probe-local → atlas transform (`anatomy/transform.py`): pitch, yaw, shank
orientation, composition order, and bregma-relative coordinate conversion.

#### `test_anatomy_visualization.py` (5 tests)
Sampling regions along a shank and collapsing them into contiguous depth bands
(`anatomy/visualization.py`).

#### `test_anatomy_schematic.py` (5 tests)
The three-slice locator figure (`anatomy/schematic.py`).

#### `test_anatomy_reference.py` (6 tests)
Per-atlas bregma / squish / tilt reference parameters and landmark policy.

#### `test_anatomy_surface_depth.py` (20 tests)
The tip-depth-below-brain-surface readout (`anatomy/regions.py`,
`anatomy/transform.probe_axis_up`): depth against a known surface, tilted
trajectories, the outermost-crossing rule (an internal gap such as a ventricle is
not mistaken for the surface), and the cases where depth is undefined.

### GUI overlays (both)

#### `test_gui_overlays.py` (33 tests)
Drives the `ChannelmapGUI` handlers themselves — the code path a user triggers by
clicking *Load survey overlay* or *Compute anatomical overlay* — and asserts on the
resulting Bokeh data sources and widget state:
- Survey loading: values mapped to the right electrodes, sidebar bars drawn and
  colored, electrode selection colors left untouched
- Survey colormap defaults: a heavy-tailed survey renders with visible contrast
  *before* any vmin/vmax edit (regression test), live rescaling, inverted ranges refused
- Survey rejection paths (wrong extension, unparseable, wrong probe, no file) and clearing,
  including the automatic clear on probe-type switch
- Anatomy overlay: bands / labels / boundaries / legend / locator populated for the
  traversed regions, per-shank coverage, live update when the pose changes, clearing
- Tip depth readout: value, tilt behaviour, out-of-brain and not-yet-downloaded states
  (asserting a readout refresh never triggers an atlas download)
- Both overlays active at once, with region labels shifting outward to clear the survey bars

### `fixtures/`
Sample IMRO files for testing file I/O:
- `sample_1.0.imro` - Neuropixels 1.0 example
- `sample_2.0-1shank.imro` - NP2.0 single-shank example
- `sample_2.0-4shanks.imro` - NP2.0 four-shank example

## Running the Tests

### Run all tests:
```bash
pytest tests/
```

### Run with verbose output:
```bash
pytest tests/ -v
```

### Run with coverage report:
```bash
pytest tests/ --cov=pixelmap --cov-report=term
```

### Run specific test class:
```bash
pytest tests/test_core_functionality.py::TestHardwareConstraints -v
```

### Run specific test:
```bash
pytest tests/test_core_functionality.py::TestHardwareConstraints::test_forbidden_electrodes_are_detected -v
```

## Test Coverage

The test suite covers:
- **Backend logic** (`backend.py`):
  - `find_forbidden_electrodes()` - Wiring conflict detection
  - `_verify_hardware_violations()` - Constraint validation
  - `get_electrodes()` - Electrode selection with presets
  - `get_preset_candidates()` - All 29 presets across probe types

- **IMRO utilities** (`utils/imro.py`):
  - `generate_imro_channelmap()` - IMRO list generation
  - `save_to_imro_file()` - File writing
  - `read_imro_file()` - File reading
  - `parse_imro_file()` - Content parsing

- **Activity survey overlay** (`utils/survey.py`, `gui/gui.py`):
  - `parse_survey_file()` - SpikeGLX `.txt` parsing
  - `match_survey_to_electrodes()` / `validate_probe_match()` - contact matching
  - `default_survey_range()` - initial colormap bounds
  - `ChannelmapGUI.load_survey_file()` / `clear_survey_overlay()` / `_on_survey_range_change()`

- **Anatomical overlay** (`anatomy/*`, `gui/gui.py`):
  - `lookup_regions()` / `canonical_annotation()` - atlas indexing and reorientation
  - `probe_to_atlas()` / `probe_axis_up()` / `bregma_to_atlas_um()` - pose geometry
  - `compute_region_bands()` - depth bands along each shank
  - `render_locator()` - three-slice locator figure
  - `tip_depth_below_surface_um()` - depth-below-surface readout
  - `ChannelmapGUI.compute_anatomy_overlay()` / `clear_anatomy_overlay()` /
    `_update_tip_depth_readout()`

- **All probe types**:
  - Neuropixels 1.0 (10 subtypes)
  - Neuropixels 2.0 single-shank (3 subtypes)
  - Neuropixels 2.0 four-shank (3 subtypes)

**No network access is required.** The anatomy tests patch `BrainGlobeAtlas` with a
small in-memory fake volume, so no atlas is ever downloaded during a test run.

## Continuous Integration

Tests run automatically via GitHub Actions on:
- Every push to `main` and `dev` branches
- Every pull request to `main`
- Manual workflow dispatch

The workflow tests across multiple Python versions (3.10, 3.11, 3.12) to ensure compatibility.

## Adding New Tests

When adding new functionality:

1. Add test fixtures to `conftest.py` if needed
2. Create tests in `test_core_functionality.py` following existing patterns
3. Use descriptive test names: `test_<what_is_being_tested>`
4. Include docstrings explaining what each test validates
5. Run tests locally before committing
6. Verify CI passes after pushing