# Test Suite Documentation

This directory contains the automated test suite for PixelMap (Neuropixels Channelmap Generator).

## Overview

The test suite validates the core functionality of PixelMap to ensure reliable generation of IMRO files for Neuropixels probes. Tests are designed to be comprehensive yet fast, providing confidence that the software works correctly.

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

- **All probe types**:
  - Neuropixels 1.0 (10 subtypes)
  - Neuropixels 2.0 single-shank (3 subtypes)
  - Neuropixels 2.0 four-shank (3 subtypes)

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