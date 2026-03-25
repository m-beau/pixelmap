# Python API Reference

PixelMap can be used programmatically from Python scripts or Jupyter notebooks.

## Channel Map Generation

```{eval-rst}
.. module:: pixelmap.utils.imro

.. autofunction:: generate_imro_channelmap

.. autofunction:: save_to_imro_file

.. autofunction:: read_imro_file

.. autofunction:: parse_imro_file

.. autofunction:: parse_imro_list
```

## Electrode Selection

```{eval-rst}
.. module:: pixelmap.backend

.. autofunction:: get_electrodes

.. autofunction:: find_forbidden_electrodes

.. autofunction:: make_wiring_maps

.. autofunction:: format_imro_string
```

## Types

```{eval-rst}
.. module:: pixelmap.types

.. autoclass:: Electrode
   :members:
```

## Constants

The following constants are defined in `pixelmap.constants`:

`PROBE_TYPE_MAP`
: Mapping from probe type names (`"1.0"`, `"2.0-1shank"`, `"2.0-4shanks"`) to lists of SpikeGLX subtype numbers.

`PROBE_N`
: Number of physical electrodes (`N`), ADC channels (`n`), and channels per shank (`n_per_shank`) for each probe type.

`SUPPORTED_1shank_PRESETS`
: Available preset names for single-shank probes. See the [preset reference](presets.md).

`SUPPORTED_4shanks_PRESETS`
: Available preset names for four-shank probes. See the [preset reference](presets.md).

## Usage Examples

### Generate a channelmap from a preset

```python
import pixelmap as cmg

imro_list = cmg.generate_imro_channelmap(
    probe_type="2.0-4shanks",
    layout_preset="tips_all",
    wiring_file="wiring_maps/2.0-4shanks_wiring.csv"
)

cmg.save_to_imro_file(imro_list, "my_channelmap.imro")
```

### Generate a channelmap with custom electrodes

```python
import numpy as np
import pixelmap as cmg

# Custom electrode selection: array of (shank_id, electrode_id) pairs
custom = np.array([[0, i] for i in range(384)])

imro_list = cmg.generate_imro_channelmap(
    probe_type="1.0",
    custom_electrodes=custom,
    wiring_file="wiring_maps/1.0_wiring.csv"
)

cmg.save_to_imro_file(imro_list, "custom_channelmap.imro")
```

### Read and inspect an existing IMRO file

```python
import pixelmap as cmg

imro_list = cmg.read_imro_file("my_channelmap.imro")
(
    selected_electrodes,
    probe_type,
    probe_subtype,
    reference_string,
    ap_gain,
    lf_gain,
    hp_filter,
) = cmg.parse_imro_list(imro_list)

print(f"Probe type: {probe_type} (subtype {probe_subtype})")
print(f"Reference: {reference_string}")
print(f"Selected {len(selected_electrodes)} electrodes")
```
