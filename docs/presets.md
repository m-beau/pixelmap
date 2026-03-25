# Preset Reference

PixelMap includes pre-configured electrode layouts (presets) for common recording geometries. All presets respect the ADC wiring constraints of each probe type.

## Single-Shank Presets

These presets apply to **NP 1.0** and **NP 2.0 single-shank** probes:

| Preset | Description |
|--------|-------------|
| `Tip` | Electrodes starting from the tip of the probe |
| `tip_b0_top_b1` | Tip of bank 0 combined with top of bank 1 |
| `top_b0_tip_b1` | Top of bank 0 combined with tip of bank 1 |
| `zigzag` | Alternating (checkerboard) electrode pattern |

## Four-Shank Presets

These presets apply to **NP 2.0 four-shank** probes:

### Full-probe presets

| Preset | Description |
|--------|-------------|
| `tips_all` | Tips of all four shanks |
| `tips_0_3` | Tips of shanks 0 and 3 |
| `tips_1_2` | Tips of shanks 1 and 2 |

### Single-shank tip presets

| Preset | Description |
|--------|-------------|
| `tip_s0` | Tip of shank 0 only |
| `tip_s1` | Tip of shank 1 only |
| `tip_s2` | Tip of shank 2 only |
| `tip_s3` | Tip of shank 3 only |

### Bank-split presets (tip of bank 0, top of bank 1)

| Preset | Description |
|--------|-------------|
| `tip_b0_top_b1_s0` | Shank 0: tip of bank 0 + top of bank 1 |
| `tip_b0_top_b1_s1` | Shank 1: tip of bank 0 + top of bank 1 |
| `tip_b0_top_b1_s2` | Shank 2: tip of bank 0 + top of bank 1 |
| `tip_b0_top_b1_s3` | Shank 3: tip of bank 0 + top of bank 1 |

### Bank-split presets (top of bank 0, tip of bank 1)

| Preset | Description |
|--------|-------------|
| `top_b0_tip_b1_s0` | Shank 0: top of bank 0 + tip of bank 1 |
| `top_b0_tip_b1_s1` | Shank 1: top of bank 0 + tip of bank 1 |
| `top_b0_tip_b1_s2` | Shank 2: top of bank 0 + tip of bank 1 |
| `top_b0_tip_b1_s3` | Shank 3: top of bank 0 + tip of bank 1 |

### Cross-shank presets

| Preset | Description |
|--------|-------------|
| `tip_s0b0_top_s2b0` | Tip of shank 0 bank 0 + top of shank 2 bank 0 |
| `tip_s2b0_top_s0b0` | Tip of shank 2 bank 0 + top of shank 0 bank 0 |
| `tip_s1b0_top_s3b0` | Tip of shank 1 bank 0 + top of shank 3 bank 0 |
| `tip_s3b0_top_s1b0` | Tip of shank 3 bank 0 + top of shank 1 bank 0 |

### Gliding presets

| Preset | Description |
|--------|-------------|
| `gliding_0-3` | Gliding pattern from shank 0 to shank 3 |
| `gliding_3-0` | Gliding pattern from shank 3 to shank 0 |

### Zigzag presets

| Preset | Description |
|--------|-------------|
| `zigzag_0` | Alternating electrode pattern on shank 0 |
| `zigzag_1` | Alternating electrode pattern on shank 1 |
| `zigzag_2` | Alternating electrode pattern on shank 2 |
| `zigzag_3` | Alternating electrode pattern on shank 3 |

## Using Presets

### In the GUI

Select a preset from the dropdown menu. The probe visualization will update to show the selected electrodes. You can then modify the selection manually before downloading.

### In Python

```python
import pixelmap as cmg

imro_list = cmg.generate_imro_channelmap(
    probe_type="2.0-4shanks",
    layout_preset="tips_all",
    wiring_file="wiring_maps/2.0-4shanks_wiring.csv"
)
```

See the [API reference](api.md) for full details.
