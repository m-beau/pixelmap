# Development Guide

This guide is for contributors and developers who want to work on PixelMap.

## Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
git clone https://github.com/m-beau/channelmap_generator.git
cd channelmap_generator

# Install with dev dependencies
uv sync --all-extras

# Or with pip
pip install -e ".[dev]"
```

### Running Locally

```bash
# Launch the GUI
uv run cmap_gui

# Or if installed with pip
cmap_gui
```

## Architecture Overview

```
channelmap_generator/
├── __init__.py          # Public API exports
├── backend.py           # Core electrode selection logic
├── constants.py         # Probe configurations and presets
├── types.py             # Data classes (Electrode, etc.)
├── gui/
│   ├── gui.py           # NiceGUI frontend
│   ├── app_util.py      # GUI helper functions
│   └── assets/          # Images and static files
├── utils/
│   └── imro.py          # IMRO file reading/writing
└── wiring_maps/         # CSV files defining electrode-ADC wiring
```

### Key Modules

**`backend.py`**
: Core logic for electrode selection and validation. Handles ADC wiring constraints via `find_forbidden_electrodes()` and generates valid channel configurations.

**`gui/gui.py`**
: Browser-based interface built with [NiceGUI](https://nicegui.io/). Handles user interactions, electrode visualization, and IMRO file downloads.

**`utils/imro.py`**
: Functions for reading, writing, and parsing IMRO files. The `generate_imro_channelmap()` function is the main entry point for programmatic use.

**`constants.py`**
: Probe type definitions, electrode counts, and preset configurations. Edit this when adding new probe types or presets.

**`types.py`**
: Data classes used throughout the codebase, including the `Electrode` class.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=channelmap_generator

# Run specific test file
pytest tests/test_backend.py
```

### Writing Tests

Tests are in the `tests/` directory. When adding new functionality:

1. Add tests for the new feature
2. Ensure existing tests still pass
3. Aim for coverage of edge cases

## Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

## Adding New Probe Types

To add support for a new probe (e.g., Quadbase, NXT):

1. **Add wiring map**: Create `wiring_maps/<probe_type>_wiring.csv` defining electrode-to-ADC mappings

2. **Update constants**: In `constants.py`, add entries to:
   - `PROBE_TYPE_MAP` - SpikeGLX subtype numbers
   - `PROBE_N` - Electrode and channel counts
   - `SUPPORTED_*_PRESETS` - Available presets (if applicable)

3. **Update GUI**: In `gui/gui.py`, add the probe to the dropdown options

4. **Add tests**: Create tests for the new probe type

5. **Update docs**: Add to the supported probes table in `index.md` and `README.md`

## Contributing

See [CONTRIBUTING.md](https://github.com/m-beau/channelmap_generator/blob/main/CONTRIBUTING.md) for guidelines on:

- Reporting bugs
- Requesting features
- Submitting pull requests
