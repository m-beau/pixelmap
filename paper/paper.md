---
title: "PixelMap: A Browser-Based Tool for Wiring-Aware, Anatomy- and Activity-Guided Design of Neuropixels Channelmaps"
tags:
  - Python
  - Neuroscience
  - Electrophysiology
  - Neuropixels
  - Panel
authors:
  - given-names: Maxime
    surname: Beau
    orcid: 0000-0002-8907-6612
    corresponding: true
    affiliation: "1, 2"
  - given-names: Julie M. J.
    surname: Fabre
    orcid: 0000-0003-0550-0410
    affiliation: "1"
  - given-names: Christian
    surname: Tabedzki
    orcid: 0000-0001-8409-6094
    affiliation: "1"
  - given-names: Jorge
    surname: Yanar
    orcid: 0000-0003-1416-3567
    affiliation: "1"
  - given-names: Carlos D.
    surname: Brody
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

# Summary

PixelMap is a browser-based application for creating custom channelmaps for Neuropixels probes that respects electrode wiring constraints. Neuropixels probes, widely used for high-density neural recordings, have more physical electrodes than can be used for simultaneous recording because they contain fewer readout channels and analogue-to-digital converters (ADCs) than electrodes. Each ADC and readout channel is hard-wired to several electrodes, creating complex interdependencies where selecting one electrode makes others unavailable. PixelMap provides an installation-free, browser-based interface for researchers to design arbitrary recording configurations that meet their experimental requirements while satisfying these hardware constraints. Beyond constraint-aware electrode selection, PixelMap supports two guided workflows: overlaying anatomical region boundaries from any BrainGlobe atlas to plan recordings around specific regions, and overlaying a SpikeGLX activity survey heatmap to identify the most active channels after implantation. Both overlays can be used simultaneously during electrode selection. PixelMap generates IMRO (IMec Read Out) files compatible with SpikeGLX, the reference acquisition software for Neuropixels recordings.

# Statement of need

Neuropixels probes have revolutionised systems neuroscience by enabling simultaneous recordings from hundreds of neurons across multiple brain regions at any depth [@jun2017; @beau2021; @steinmetz2021; @bondy2024; @ye2025; @beau2025]. However, configuring these probes presents challenges. Limited by the number of readout channels and integrated analogue-to-digital converters (ADCs), Neuropixels probes contain 960–5120 physical electrodes per shank but can only record from 384–1536 channels simultaneously (Table 1). Users must therefore select a subset of electrodes to activate for each recording, forming an electrode selection set colloquially called "channelmap".

Because readout lines within each shank and ADCs in the probe head are each shared by multiple physical electrodes, activating one electrode makes others unavailable. For Neuropixels 1.0, these dependencies follow a regular, bank-aligned pattern that is relatively intuitive. With single-shank Neuropixels 2.0 probes [@steinmetz2021], whose wiring is intentionally scrambled to enable recording from several banks simultaneously, and four-shank Neuropixels 2.0 probes, whose wiring become untractable due to the high number of electrode banks, the inter-electrode incompatibilities become difficult to anticipate.

Beyond satisfying hardware constraints, channelmaps must be tailored to the experiment. Before implantation, researchers plan which brain regions each probe will traverse given its intended insertion trajectory; this requires mapping anatomical boundaries onto the physical electrode layout. After implantation (chronic or acute), researchers must adjust their channelmap based on observed neural activity. Regions with large waveforms typically correspond to grey matter (rich in axon initial segments), marking target region entry and exit points. The typical workflow is therefore: plan anatomically before implantation, survey activity afterwards, and refine the channelmap to maximise unit yield in the target regions. Using multiple probes simultaneously, where the difficulty compounds, is becoming commonplace [@bondy2024], making a fast wiring-aware design tool increasingly essential. Developing a tool to support this workflow within the Brody laboratory motivated the creation of PixelMap, which addresses these needs by:

1. **Being available on any machine installation-free** at [https://pixelmap.pni.princeton.edu](https://pixelmap.pni.princeton.edu).
2. **Visualising wiring constraints in real time**: When users select electrodes (red), the interface immediately shows which become unavailable (black) due to shared lines or ADCs, preventing invalid configurations.
3. **Supporting arbitrary electrode geometries** through 1) common preset geometries, 2) entering electrode ranges as text for reproducibility, 3) directly clicking or dragging on the probe visualization, or 4) loading pre-existing `.imro` files. These four selection methods are intercompatible and meant to be combined.
4. **Guiding channelmap design with anatomy and activity overlays**: Users can overlay anatomical region boundaries from any brain atlas available through BrainGlobe [@claudi2020] to plan recordings around specific brain regions of interest, and **simultaneously** a SpikeGLX activity survey heatmap to identify the most spiking-active channels and confirm anatomical predictions after probe implantation.

| Probe Version | Physical Channels | Simultaneously Recordable Channels |
|---------------|-------------------|-------------------------------------|
| Neuropixels 1.0 | 960 | 384 |
| Neuropixels 2.0 (single shank) | 1,280 | 384 |
| Neuropixels 2.0 (4-shank) | 5,120 (1,280 per shank) | 384 |
| Neuropixels 2.0 Quad Base | 5,120 (1,280 per shank) | 1,536 (384 max per shank)|

**Table 1**: Number of physical and simultaneously addressable electrodes across Neuropixels probe versions supported by PixelMap.

# State of the Field

SpikeGLX ([https://billkarsh.github.io/SpikeGLX](https://billkarsh.github.io/SpikeGLX)) and Open Ephys ([https://open-ephys.github.io/gui-docs/User-Manual/Plugins/Neuropixels-PXI.html](https://open-ephys.github.io/gui-docs/User-Manual/Plugins/Neuropixels-PXI.html)), the two most widely used systems for acquiring Neuropixels data, allow channelmap editing but their primary purpose is to enable reliable data acquisition. Both prioritize stable high-throughput data streaming and visualisation; channelmap editing is a configuration step performed at the rig on the machine connected to the probe, optimised for fast on-the-fly setup at recording time. SpikeGLX's editor is also the field's authoritative reference for wiring constraints, obtained directly from the manufacturer (IMEC); for this reason, PixelMap derives its information from SpikeGLX and writes channelmaps as `.imro` files compatible with SpikeGLX. NeuroCarto [@su2025], a recently published browser-based editor that runs locally, offers a GUI that is built around a novel automated channelmap-generation algorithm: the user specifies a target electrode density for each region along the shank, and a channelmap meeting those constraints is generated.

PixelMap, by contrast, is built as a no-install generalist design tool. It brings together in a single tool the features distributed across the solutions above (real-time visualisation of wiring constraints, .imro output, an interactive click-and-drag editor, anatomical reference) and adds several capabilities not available elsewhere:

- PixelMap is the only solution that requires no installation, hosted at [https://pixelmap.pni.princeton.edu](https://pixelmap.pni.princeton.edu), so a channelmap can be built on any machine without desktop acquisition software or a connected probe.
- Probe support follows the authoritative reference. Probe-group and subgroup definitions, which alter `.imro` file formats, are derived from SpikeGLX ([http://billkarsh.github.io/SpikeGLX/help/imroTables](http://billkarsh.github.io/SpikeGLX/help/imroTables)). This guarantees present accuracy and forward compatibility.
- The full probe-dependent `.imro` metadata is exposed, including the Neuropixels 1.0 hardware filter, gain settings, and electrode reference (external, tip, or join-tips).
- A wider set of arbitrary geometries can be selected. Beyond custom presets, text entry of electrode ranges, and an `.imro` loader, it provides single-click and five drag-box tools, including interleaved and checkerboard selectors for broader coverage, and a "deselect dependents" box, unavailable elsewhere, that releases selected electrodes blocking a target region.
- The anatomical overlay is built on the BrainGlobe Atlas API [@claudi2020], which makes any BrainGlobe atlas available immediately. PixelMap renders the region each electrode traverses beside the probe.
- An activity overlay based on a SpikeGLX survey can be displayed alongside the anatomy ovelay simultaneously while allowing electrode selection, which is really helpful for experiment planning and is not offered by any of the currently available software.

# Software Design

PixelMap is implemented in Python and consists of three main components:

- The **probe type database and wiring maps** (distributed in `./constants.py`. `./utils/probe_features.py`, `./wiring_maps/*`) derives the mapping between probe part numbers, SpikeGLX probe types, and IMRO format numbers directly from the authoritative source maintained by SpikeGLX [http://billkarsh.github.io/SpikeGLX/help/imroTables](http://billkarsh.github.io/SpikeGLX/help/imroTables). The **wiring maps** at `./wiring_maps/*.csv` are CSV files describing the electrode-to-ADC mappings for each supported probe type. They were adapted from files provided by IMEC (Neuropixels manufacturer - downloadable [here](https://www.neuropixels.org/support)) and SpikeGLX (https://github.com/billkarsh/SpikeGLX/tree/a9024ba79f4481883766f5468563accb74fd58fd/Src-imro).
- The **electrode selection logic** at `./backend.py` implements the constraint-checking algorithms that validate electrode selections against probe-specific wiring maps. Cached (memoized) hash tables (Python dictionaries) query incompatible electrode pairs in O(1) (so, fast).
- The **graphical user interface** at `./gui/gui.py` was built with Holoviz Panel. User interactions trigger immediate recalculation of available electrodes, ensuring users receive instant visual feedback about constraint violations. The **activity survey overlay** reads a tab-separated file exported from SpikeGLX (one row per electrode, with columns `Shank`, `Xum`, `Zum`, `Val`) and renders a coloured bar beside each electrode. The **anatomical overlay** computes the brain region traversed by each electrode along a user-specified insertion trajectory—accounting for pitch, yaw, and multi-shank orientation—using any atlas registered in the BrainGlobe atlas API [@claudi2020]. The two most widely used atlases (Allen Mouse 25 µm `allen_mouse_25um` and Waxholm rat 39 µm `whs_sd_rat_39um`) are pre-downloaded so they are available instantly. Additional atlases are downloaded on demand and stored in a shared Docker volume (`brainglobe_cache`) that persists across users.

## Forward Compatibility and Extensibility

PixelMap's architecture is designed to easily support future hardware and atlases, lowering the barrier for both maintainers and contributors.

**Probe support.** Wiring constraints are encoded as standalone CSV files (`wiring_maps/*.csv`), one per probe type, and probe metadata (part numbers, readout counts, electrode pitches) is derived directly from the authoritative JSON maintained by SpikeGLX developer Bill Karsh.

**Atlas support.** As the anatomical overlay is built on the BrainGlobe Atlas API [@claudi2020], any atlas registered with BrainGlobe in the future will be automatically available to PixelMap users without code changes.

**Contributor extensibility.** The codebase is written as comprehensively as possible with contributions in mind. New selection presets can be added to backend.py, new overlay types can follow the pattern established by the activity-survey and anatomy overlays, and the GUI layer can be extended or replaced without modifying the core constraint logic. This modular design, together with comprehensive tests and a contributor guide ([https://pixelmap-neuropixels.readthedocs.io/en/latest/development.html](https://pixelmap-neuropixels.readthedocs.io/en/latest/development.html)), is intended to make contribution straightforward. These design choices have already enabled two new contributors to contribute significant features in 2026, warranting authorship on this paper: the anatomical overlay (J.M.J.F) and the activity survey overlay (J.Y.).

![PixelMap's browser-based graphical user interface.\
**Center:** main panel featuring the probe's physical layout with one or four shanks that exhibit the 960 (Neuropixels 1.0) or 1,280 (Neuropixels 2.0, QuadBase) physical electrodes per shank to be selected from. Electrodes available for selection are light grey, selected electrodes turn red, and electrodes that become unavailable due to hardware wiring constraints turn black. In this example, 384 electrodes have been selected (matching the maximum simultaneous recording capacity), with a distributed pattern across multiple banks, illustrating that PixelMap allows selection of arbitrary channelmap geometries. An anatomical overlay (colored depth bands with region labels) and is shown as an optional layers that guides channelmap design.\
**Left:** panel to input probe metadata (also part of `.imro` files) as well as three methods of electrode selection: preset geometries, manual textual input of electrode ranges, and pre-loading an existing `.imro` file. These three methods of electrode selection can be mixed together with five interactive click-and-drag box tools.\
**Right:** electrode status indicator that turns green to confirm the selection is complete and is ready for IMRO file generation. Users can export their configuration via the "Download IMRO" button for direct use in SpikeGLX and optionally save a PDF visualisation to easily remember the geometry of the corresponding `.imro` file in the future. Below the status indicator are PixelMap's instructions.](Figure1.png)

# Installation and Usage

PixelMap can be used through 1) a **Web app** available at [https://pixelmap.pni.princeton.edu](https://pixelmap.pni.princeton.edu) , 2) **Local browser app** via pip/uv, or 3) **Programmatic API**: Python scripts can directly call `generate_imro_channelmap()` for batch processing or integration into analysis pipelines.

For more details, see the documentation ([https://pixelmap-neuropixels.readthedocs.io](https://pixelmap-neuropixels.readthedocs.io)).

# Research Impact Statement

PixelMap addresses a practical bottleneck in Neuropixels experimental workflows. Neuropixels have become the dominant technology for large-scale electrophysiology, with exponential growth in publications using the technology ([PubMed](https://esperr.github.io/pubmed-by-year/?q1=Neuropixels)). Yet no existing tool provided installation-free channelmap design with support for arbitrary electrode geometries and simultaneous anatomy- and activity-guided planning (see **State of the Field**).

PixelMap provides comprehensive documentation, including a contributor guide ([https://pixelmap-neuropixels.readthedocs.io/en/latest/development.html](https://pixelmap-neuropixels.readthedocs.io/en/latest/development.html)), under a permissive open-source license (GPL3), and is accessible via web application, Python package, Docker container, or programmatic API. It builds on the authors' track record using Neuropixels probes [@steinmetz2021; @bondy2024; @beau2025; @fabre2026basal] and developing Neuropixels software [@beau2021].

Adoption is evidenced by deployment at PNI's public server, 59 GitHub stars, and web traffic of roughly 400 unique visitors between March and May 2026 (~45/week; see https://github.com/m-beau/pixelmap/tree/main/analytics). The package has been under active development for over ten months, with external contributors implementing new features (anatomical and activity overlays).

# AI Usage Disclosure

**AI-assisted technologies used:** Claude Sonnet 4.5/4.6, and Opus 4.6/4.7 (Anthropic).
AI assistance was used for (1) optimization suggestions and documentation improvements (docstrings, code comments) in `backend.py`, (2) initial scaffolding of the Holoviz Panel GUI architecture in `gui/gui.py`, (3) manuscript grammatical and syntactical review. AI was not used for project conceptualization, core backend design, electrode wiring map construction. App hosting infrastructure was designed independently of AI assistance. All AI-generated code suggestions were reviewed and validated by human authors before integration; all AI-assisted manuscript edits were reviewed and approved by the corresponding author.

# Author Contributions

|                                    | Maxime Beau | Julie Fabre | Christian Tabedzki | Jorge Yanar | Carlos D. Brody |
|------------------------------------|:-----------:|:-----------:|:------------------:|:-----------:|:---------------:|
| Conceptualisation                  |      X      |             |                    |             |                 |
| Backend                            |      X      |             |                    |             |                 |
| GUI - general                      |      X      |      X      |                    |             |                 |
| GUI - Anatomical overlay           |             |      X      |                    |             |                 |
| GUI - Activity overlay      |             |             |                    |      X      |                 |
| App hosting                        |             |             |         X          |             |                 |
| Supervision and funding            |             |             |                    |             |        X        |

# Conflict of Interest Statement

The authors declare no competing interests.

# Acknowledgements

We thank Jesse C. Kaminsky and members of the Brody laboratory for testing and feedback during development, and PNI IT members Garrett McGrath and Gary Lyons for their advice concerning hosting. We also thank Bill Karsh for help with development and navigation of SpikeGLX resources. Finally, we thank the Princeton Neuroscience Institute for hosting the web application. J.M.J.F was supported by the Schmidt Science Fellows, in partnership with the Rhodes Trust. This work was supported by Howard Hughes Medical Institute and the National Institutes of Health.

# References
