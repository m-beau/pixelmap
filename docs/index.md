# PixelMap

**A browser-based GUI to generate Neuropixels channelmaps that respect electrode-ADC wiring constraints.**

```{image} _static/GUI_screenshot.png
:alt: PixelMap GUI screenshot
:width: 100%
```

PixelMap lets you design custom [IMRO tables](https://billkarsh.github.io/SpikeGLX/help/imroTables/) for SpikeGLX — the files that control which electrodes to record from on Neuropixels probes. It enforces the hardware wiring constraints that make manual IMRO editing error-prone.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Getting Started
:link: getting-started
:link-type: doc

Installation, first launch, and your first channelmap.
:::

:::{grid-item-card} GUI Guide
:link: gui-guide
:link-type: doc

Detailed walkthrough of the browser-based interface.
:::

:::{grid-item-card} Python API
:link: api
:link-type: doc

Use PixelMap programmatically from Python or Jupyter notebooks.
:::

:::{grid-item-card} Troubleshooting
:link: troubleshooting
:link-type: doc

Common issues and how to resolve them.
:::

:::{grid-item-card} Development
:link: development
:link-type: doc

Contributing, architecture, and adding new probes.
:::

::::

## Supported Probes

| Probe type     | Status |
|---------------|--------|
| 1.0           | Supported |
| 2.0, 1-shank  | Supported |
| 2.0, 4-shanks | Supported |
| Quadbase      | Planned |
| NXT           | Planned |

UHD and Opto are not currently on our roadmap — [open an issue](https://github.com/m-beau/pixelmap/issues) if you need support for these.

```{toctree}
:maxdepth: 2
:hidden:

getting-started
gui-guide
api
troubleshooting
presets
development
```
