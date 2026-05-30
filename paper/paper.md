---
title: 'PixelMap: An Application for Flexible Electrode Selection on Neuropixels Probes'
tags:
  - Python
  - neuroscience
  - electrophysiology
  - neuropixels
authors:
  - name: Maxime Beau
    orcid: 0000-0002-8907-6612
    corresponding: true
    affiliation: "1, 2"
  - name: Julie Fabre
    affiliation: 1
  - name: Christian Tabedzki
    orcid: 0000-0001-8409-6094
    affiliation: 1
  - name: Jorge Janar
    affiliation: 1
  - name: Carlos D. Brody
    orcid: 0000-0002-4201-561X
    affiliation: "1, 2"
affiliations:
  - name: Princeton Neuroscience Institute, Princeton University, USA
    index: 1
  - name: Howard Hughes Medical Institute, USA
    index: 2
date: 29 May 2026
bibliography: paper.bib
---

# Abstract

PixelMap is a browser-based application for creating custom channelmaps for Neuropixels probes that respects electrode wiring constraints. Neuropixels probes, widely used for high-density neural recordings, have more physical electrodes than can be used for simultaneous recording because they contain fewer readout channels and analogue-to-digital converters (ADCs) than electrodes. Each ADC and readout channel is hard-wired to several electrodes, creating complex interdependencies where selecting one electrode makes others unavailable. PixelMap provides an installation-free, browser-based interface for researchers to design arbitrary recording configurations that meet their experimental requirements while satisfying these hardware constraints. Beyond constraint-aware electrode selection, PixelMap supports two guided design workflows: overlaying a SpikeGLX activity survey heatmap to identify the most active channels after implantation, and overlaying anatomical region boundaries from any brain atlas available through BrainGlobe [@claudi2020] to plan recordings around specific brain regions. The tool generates IMRO (IMec Read Out) files compatible with SpikeGLX, the most common acquisition software for Neuropixels recordings.

# Statement of need

Neuropixels probes have revolutionised systems neuroscience by enabling simultaneous recordings from hundreds of neurons across multiple brain regions at any depth [@jun2017; @beau2021; @steinmetz2021; @bondy2024; @ye2025; @beau2025]. However, configuring these probes presents challenges. Limited by the number of readout channels and integrated analogue-to-digital converters (ADCs), Neuropixels probes contain 960–5120 electrodes but can only record from 384–1536 channels simultaneously (Table 1). Users must therefore select a subset of electrodes to activate for each recording, a "channelmap". Researchers often need to create custom channelmaps to target specific brain regions, and sometimes must adjust them rapidly based on feedback from ongoing recordings. Because the electrode-to-ADC wiring follows complex, probe version-dependent patterns, manual channelmap design is error-prone and time-consuming.

[SpikeGLX](https://billkarsh.github.io/SpikeGLX) is the reference acquisition software for Neuropixels recordings and uses the `.imro` file format to encode channelmaps. While SpikeGLX provides tools to edit channelmaps, it requires a desktop app, comes with limited preset channelmaps, and does not easily allow selection of fully arbitrary electrode geometries or anatomical-guided design. Other acquisition software like Open Ephys also support `.imro` files but also require desktop app and does not allow anatomical-guided design. Researchers often resort to hand-engineering `.imro` files or using custom scripts, which can lead to errors and inefficiencies.

PixelMap addresses these needs by:

1. **Being available on any machine installation-free**: The tool is available both as a browser-based web application at [https://pixelmap.pni.princeton.edu](https://pixelmap.pni.princeton.edu), as a Docker image, and a Python package.
2. **Visualising wiring constraints interactively**: When users select electrodes, the interface immediately shows which other electrodes become unavailable (marked in black) due to shared ADC lines, preventing invalid configurations.
3. **Supporting arbitrary electrode geometries**: Users can select electrodes by choosing from common preset geometries, entering electrode ranges as text for reproducibility, directly clicking or dragging on the probe visualization, or loading pre-existing `.imro` files. These four selection methods are fully intercompatible and can be combined.
4. **Guiding channelmap design with activity and anatomy information**: Users can overlay a SpikeGLX activity survey heatmap to identify the most spiking-active channels after probe implantation, and independently overlay anatomical region boundaries from any brain atlas available through BrainGlobe [@claudi2020] to plan recordings around specific brain regions of interest.

| Probe Version | Physical Channels | Simultaneously Recordable Channels |
|---------------|-------------------|-------------------------------------|
| Neuropixels 1.0 | 960 | 384 |
| Neuropixels 2.0 (single shank) | 1,280 | 384 |
| Neuropixels 2.0 (4-shank) | 5,120 (1,280 per shank) | 384 |
| Neuropixels 2.0 Quad Base | 5,120 (1,280 per shank) | 1,536 |

**Table 1**: Number of physical and simultaneously addressable electrodes across Neuropixels probe versions supported by PixelMap.

# Software Design

PixelMap is implemented in Python using [Holoviz Panel](https://panel.holoviz.org/) for the web interface, providing an interactive and responsive user experience. The software architecture consists of four main components.

First, the **probe type database** derives the mapping between probe part numbers, SpikeGLX probe type identifiers, and IMRO format numbers directly from the authoritative source maintained by SpikeGLX developer Bill Karsh. This replaces an earlier hand-engineered mapping and provides both accuracy and forward compatibility with new probe versions as they are added to SpikeGLX.

Second, the **wiring maps** at `./wiring_maps/*.csv` are CSV files describing the electrode-to-ADC mappings for each supported probe type. They were adapted from files provided by IMEC (Neuropixels manufacturer - downloadable [here](https://www.neuropixels.org/support)) and SpikeGLX (https://github.com/billkarsh/SpikeGLX/tree/a9024ba79f4481883766f5468563accb74fd58fd/Src-imro). PixelMap currently supports Neuropixels 1.0, 2.0 single-shank, 2.0 four-shank, and 2.0 Quad Base probes.

Third, the **core logic** at `./backend.py` implements the constraint-checking algorithms that validate electrode selections against probe-specific wiring maps. Hash tables (Python dictionaries) are used to query incompatible electrode pairs fast, with O(1) complexity.

Finally, the **graphical user interface** at `./gui/gui.py` was built with Holoviz Panel. The interface provides real-time visualisation of the probe layout with electrodes colour-coded by selection state (available in grey, selected in red, or unavailable in black). The interface supports the four selection modes described above, including interactive single-click and drag-box selection and deselection. Five distinct box tools cover the most common channelmap patterns: a standard selection box, a deselection box, a zigzag (checkerboard) selector for dense interleaved coverage, an interleaved-pairs selector that selects alternating row pairs, and a "deselect dependents" box that identifies and releases the selected electrodes that are blocking a target region of unavailable electrodes. User interactions trigger immediate recalculation of available electrodes, ensuring users receive instant feedback about constraint violations.

The GUI also supports two powerful overlay modes displayed alongside the probe visualization. The **activity survey overlay** reads a per-contact activity file exported from SpikeGLX and renders a coloured bar beside each electrode, allowing users to identify channels with high spiking activity (typically grey matter) and adjust their channelmap accordingly. The **anatomical overlay** computes the brain region traversed by each electrode along a user-specified insertion trajectory—accounting for pitch, yaw, and multi-shank orientation—using any atlas registered in the BrainGlobe atlas API [@claudi2020]. It renders colour-coded depth bands on the probe and displays three orthogonal atlas slices (sagittal, coronal, horizontal) through the probe tip. Both overlays are preserved when downloading the PDF rendering of the channelmap, providing a complete experimental record.

![PixelMap's browser-based graphical user interface.\
**Center:** Main panel featuring the probe's physical layout with one or four shanks that exhibit the 960 (Neuropixels 1.0) or 1,280 (Neuropixels 2.0) physical electrodes per shank to be selected from. Electrodes available for selection are light grey, selected electrodes turn red, and electrodes that become unavailable due to hardware wiring constraints turn black. In this example, 384 electrodes have been selected (matching the maximum simultaneous recording capacity), with a distributed pattern across multiple banks, illustrating that PixelMap allows selection of arbitrary channelmap geometries. An anatomical overlay (colored depth bands with region labels) and is shown as an optional layers that guides channelmap design.\
**Left:** panel to input probe metadata (also part of `.imro` files) as well as three methods of electrode selection: preset geometries, manual textual input of electrode ranges, and pre-loading an existing `.imro` file. These three methods of electrode selection can be mixed together with five interactive click-and-drag box tools.\
**Right:** electrode status indicator that turns green to confirm the selection is complete and is ready for IMRO file generation. Users can export their configuration via the "Download IMRO" button for direct use in SpikeGLX and optionally save a PDF visualisation to easily remember the geometry of the corresponding `.imro` file in the future. Below the status indicator are PixelMap's instructions.](Figure1.png)

# Installation and Usage

PixelMap can be used through:

1. **Web application**: Available at [https://pixelmap.pni.princeton.edu](https://pixelmap.pni.princeton.edu) for immediate use without installation.
2. **Local installation**: Via pip (`pip install .`) or uv (`uv run pixelmap`) from the cloned GitHub repository.
3. **Docker container**: Users can download the image used for the website and run the container locally.
4. **Programmatic API**: Python scripts can directly call `generate_imro_channelmap()` for batch processing or integration into analysis pipelines.

For more details, see the project repository at [https://github.com/m-beau/pixelmap](https://github.com/m-beau/pixelmap).

The software includes an automated test suite covering hardware constraint validation, all preset configurations, IMRO file generation for all supported probe types, and end-to-end workflows. Tests run automatically via GitHub Actions continuous integration on every code change, ensuring software reliability. See the repository's `tests/` directory for details.

# Research Impact Statement

PixelMap addresses a practical bottleneck in Neuropixels experimental workflows. Neuropixels have become the dominant technology for large-scale electrophysiology, with exponential growth in publications using the technology ([PubMed](https://esperr.github.io/pubmed-by-year/?q1=Neuropixels)). Yet no existing tool provided installation-free channelmap design with support for arbitrary electrode geometries and anatomy-guided planning (see **Statement of Need**).

PixelMap demonstrates community-readiness through comprehensive documentation—including a dedicated contributor guide (CONTRIBUTING.md) with contribution workflows and a permissive open-source license (GPL3). The tool is immediately accessible via web application, Python package, Docker container, or programmatic API. The tool builds on the authors' established track record using Neuropixels probes in their research [@steinmetz2021; @bondy2024; @beau2025] and developing Neuropixels software [@beau2021].

Evidence of adoption includes deployment at Princeton Neuroscience Institute's public server, community engagement on the project repository (37 GitHub stars), and measured web traffic: cookie analytics recorded approximately 400 unique visitors between March and May 2026, averaging roughly 45 unique visitors per week. The package has been under active development for over ten months, with external contributors implementing new features (anatomical overlay, activity survey overlay).

# AI Usage Disclosure

**AI-assisted technologies used:** Claude Sonnet 4.5, Sonnet 4.6, and Opus 4.6 (Anthropic).
AI assistance was used for (1) optimization suggestions and documentation improvements (docstrings, code comments) in `backend.py`, (2) initial scaffolding of the Holoviz Panel GUI architecture in `gui/gui.py`, (3) manuscript grammatical and syntactical review. AI was not used for project conceptualization, core backend design, electrode wiring map construction. App hosting infrastructure was designed independently of AI assistance.

# Author Contributions

|                                    | Maxime Beau | Julie Fabre | Christian Tabedzki | Jorge Janar | Carlos D. Brody |
|------------------------------------|:-----------:|:-----------:|:------------------:|:-----------:|:---------------:|
| Conceptualisation                  |      X      |             |                    |             |                 |
| Backend and GUI                    |      X      |             |                    |             |                 |
| Anatomical overlay                 |             |      X      |                    |             |                 |
| Activity survey overlay            |             |             |                    |      X      |                 |
| App hosting                        |             |             |         X          |             |                 |
| Supervision and funding            |             |             |                    |             |        X        |


# Acknowledgements

We thank Jesse C. Kaminsky and members of the Brody laboratory for testing and feedback during development, and PNI IT members Garrett McGrath and Gary Lyons for their advice concerning hosting. We also thank Bill Karsh for making SpikeGLX's probe type database publicly available. Finally, we thank the Princeton Neuroscience Institute for hosting the web application. This work was supported by Howard Hughes Medical Institute and the National Institutes of Health.

# References
