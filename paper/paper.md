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

PixelMap is a browser-based application for creating custom channelmaps for Neuropixels probes that respects electrode wiring constraints. Neuropixels probes, widely used for high-density neural recordings, have more physical electrodes than can be used for simultaneous recording because they contain fewer readout channels and analogue-to-digital converters (ADCs) than electrodes. Each ADC and readout channel is hard-wired to several electrodes, creating complex interdependencies where selecting one electrode makes others unavailable. PixelMap provides an installation-free, browser-based interface for researchers to design arbitrary recording configurations that meet their experimental requirements while satisfying these hardware constraints. Beyond constraint-aware electrode selection, PixelMap supports two guided design workflows: overlaying a SpikeGLX activity survey heatmap to identify the most active channels after implantation, and overlaying anatomical region boundaries from any brain atlas available through BrainGlobe [@claudi2020] to plan recordings around specific brain regions. The tool generates IMRO (IMec Read Out) files compatible with SpikeGLX, the most common acquisition software for Neuropixels recordings.

# Statement of need

Neuropixels probes have revolutionised systems neuroscience by enabling simultaneous recordings from hundreds of neurons across multiple brain regions at any depth [@jun2017; @beau2021; @steinmetz2021; @bondy2024; @ye2025; @beau2025]. However, configuring these probes presents challenges. Limited by the number of readout channels and integrated analogue-to-digital converters (ADCs), Neuropixels probes contain 960–5120 physical electrodes per shank but can only record from 384–1536 channels simultaneously (Table 1). Users must therefore select a subset of electrodes to activate for each recording, forming a "channelmap".

Because readout lines within each shank and ADCs in the probe head are each shared by multiple physical electrodes, activating one electrode makes others unavailable. For Neuropixels 1.0, these dependencies follow a regular, bank-aligned pattern that is relatively intuitive. With single-shank Neuropixels 2.0 probes [@steinmetz2021], whose wiring is intentionally scrambled to enable recording from several banks simultaneously, and four-shank Neuropixels 2.0 probes, whose wiring become untractable due to the high number of electrode banks, the inter-electrode incompatibilities become difficult to anticipate manually. The difficulty compounds when multiple probes are used simultaneously: experiments can use up to eight simultaneous Neuropixels probes [@bondy2024], each requiring its own valid channelmap, making a reliable wiring-aware design tool increasingly essential as experimental scale grows.

Beyond satisfying hardware constraints, channelmaps must be tailored to the experiment. Before implantation, researchers plan which brain regions each probe will traverse given its intended insertion trajectory; this requires mapping anatomical boundaries onto the physical electrode layout. After implantation — whether in a chronic preparation where the probe remains in place for weeks, or an acute preparation on the same day as surgery — researchers must adjust their channelmap based on observed neural activity. Sites with high neural activity typically correspond to grey matter with neuron somata, which should be recording with the highest priority. The most effective workflow is therefore iterative: plan anatomically before implantation, survey activity afterwards, and refine the channelmap to maximise unit yield in the target regions. Developing a tool to support this workflow within the Brody laboratory — which reuns experiments featuring eight probes simultaneously [@bondy2024] — motivated the creation of PixelMap, which is now made available to the wider community. PixelMap addresses these needs by:

1. **Being available on any machine installation-free**: The tool is available both as a browser-based web application at [https://pixelmap.pni.princeton.edu](https://pixelmap.pni.princeton.edu), as a Docker image, and a Python package.
2. **Visualising in real time Neuropixels wiring constraints, interactively**: When users select electrodes, the interface immediately shows which other electrodes become unavailable (marked in black) due to shared ADC lines, preventing invalid configurations.
3. **Supporting arbitrary electrode geometries**: Users can select electrodes by 1) choosing from common preset geometries, 2) entering electrode ranges as text for reproducibility, 3) directly clicking or dragging on the probe visualization, or 4) loading pre-existing `.imro` files. These four selection methods are fully intercompatible and can be combined. See the [documentation](https://pixelmap-neuropixels.readthedocs.io/en/latest/) for details on supported preset geometries and selection methods, in particular the five different dragging boxes that enable great flexibility in electrode selection.
4. **Guiding channelmap design with activity and anatomy information**: Users can overlay a SpikeGLX activity survey heatmap to identify the most spiking-active channels after probe implantation, and independently overlay anatomical region boundaries from any brain atlas available through BrainGlobe [@claudi2020] to plan recordings around specific brain regions of interest.

| Probe Version | Physical Channels | Simultaneously Recordable Channels |
|---------------|-------------------|-------------------------------------|
| Neuropixels 1.0 | 960 | 384 |
| Neuropixels 2.0 (single shank) | 1,280 | 384 |
| Neuropixels 2.0 (4-shank) | 5,120 (1,280 per shank) | 384 |
| Neuropixels 2.0 Quad Base | 5,120 (1,280 per shank) | 1,536 (384 max per shank)|

**Table 1**: Number of physical and simultaneously addressable electrodes across Neuropixels probe versions supported by PixelMap.

# State of the Field

SpikeGLX [@karsh_spikeglx] and Open Ephys [@siegle2017] are the two most widely used systems for acquiring Neuropixels data. Both are built with the purpose to run and monitor recordings: their design priorities are stable, high-throughput streaming from high-channel-count probes, synchronisation with auxiliary data streams, and online visualisation. SpikeGLX's channelmap editor is the field's authoritative reference for wiring constraints, because it receives the electrode-to-readout correspondence directly from the probe manufacturer (IMEC); for this reason PixelMap derives its own version-specific constraints from the same source and, like SpikeGLX, encodes channelmaps as `.imro` files. Within these platforms, channelmap editing is designed as a configuration step at the rig: it requires the application to be installed on the acquisition machine and a probe to be physically connected. The editors are correspondingly optimised for quick, valid configuration at acquisition time rather than for deliberate planning away from the rig. SpikeGLX offers a small set of contiguous-group electrode selection presets; Open Ephys provides a more user-friendly editor with selection boxes; neither enable easy selection of fully arbitrary electrode geometries, i.e. interleaved selection across banks, or anatomy-guided design, as those needs lie outside the acquisition role these tools were built to fill. NeuroCarto [@su2025], a recently published, local-browser-based channelmap editor for Neuropixels, also allows users to design channelmaps encoded as `.imro` files. It is organised around a specific methodological contribution: a *blueprint* abstraction in which the user specifies a desired electrode density for each region of interest along the shank, together with an accompanying algorithm that automatically generates a channelmap around these constraints. NeuroCarto implements a GUI that lets users design channelmaps according to its algorithm; it also provides wiring-conflict visualisation and a mouse-brain atlas shown as a background image.

PixelMap, rather than being aimed at data acquisition or serving a specific channelmap-generation algorithm, is built to cover the breadth of features that routine channelmap design for multi-probe, multi-species Neuropixels experiments requires. Several design choices follow from this. First, it is served installation-free by the Princeton Neuroscience Institute as a hosted web application backed by a Docker image, so that any user can design a channelmap from any machine without setup. Second, PixelMap derives its probe-group and -subgroup definitions directly from the authoritative source maintained by the SpikeGLX developer (see Acknowledgements), which keeps probe support exhaustive and tracks new releases as they are added to SpikeGLX; PixelMap currently supports Neuropixels 1.0, 2.0 single- and four-shank, and 2.0 Quad Base. PixelMap also exposes the full, probe-type-dependent metadata encoded in `.imro` files, including the hardware high/low-pass filter toggle and gains specific to Neuropixels 1.0, and the electrode reference specification (external, tip, or join-tips referencing). Third, PixelMap implements an exhaustive collection of electrode selection methods: a large set of common probe-specific preset geometries; reproducible text-based entry of electrode ranges; five interactive drag-box tools, including interleaved and checkerboard selectors and a "deselect dependents" tool that frees the electrodes blocking an unavailable target region; and an `.imro` file loader to start from pre-existing channelmaps. Fourth, PixelMap's anatomical overlay is built on the BrainGlobe Atlas API [@claudi2020], making any registered atlas across species available, computing the region traversed by each electrode along a given insertion trajectory, and rendering named-region depth bands along the probe schematic as well as three orthogonal zoomable atlas slices through the probe. Finally, PixelMap overlays SpikeGLX activity surveys beside the probe and its predicted anatomy to guide post-implantation refinement. In summary, PixelMap contributes a feature-rich, general-purpose, installation-free Neuropixels channelmap-design toolkit spanning the full recording planning workflow, from anatomy-guided design with broad multi-species atlas integration to activity-guided refinement, with probe-type and metadata support tied to the field's authoritative reference.

# Software Design

PixelMap is implemented in Python using [Holoviz Panel](https://panel.holoviz.org/) for the web interface, providing an interactive and responsive user experience. The software architecture consists of four main components.

First, the **probe type database and wiring maps** (distributed in `./constants.py`. `./utils/probe_features.py`, `./wiring_maps/*`) derives the mapping between probe part numbers, SpikeGLX probe type identifiers, and IMRO format numbers directly from the authoritative source maintained by SpikeGLX developer Bill Karsh. This provides both accuracy and forward compatibility with new probe versions as they are added to SpikeGLX. The **wiring maps** at `./wiring_maps/*.csv` are CSV files describing the electrode-to-ADC mappings for each supported probe type. They were adapted from files provided by IMEC (Neuropixels manufacturer - downloadable [here](https://www.neuropixels.org/support)) and SpikeGLX (https://github.com/billkarsh/SpikeGLX/tree/a9024ba79f4481883766f5468563accb74fd58fd/Src-imro). PixelMap currently supports Neuropixels 1.0, 2.0 single-shank, 2.0 four-shank, and 2.0 Quad Base probes.

Second, the **core logic** at `./backend.py` implements the constraint-checking algorithms that validate electrode selections against probe-specific wiring maps. Hash tables (Python dictionaries) are used to query incompatible electrode pairs fast, with O(1) complexity.

Finally, the **graphical user interface** at `./gui/gui.py` was built with Holoviz Panel. The interface provides real-time visualisation of the probe layout with electrodes colour-coded by selection state (available in grey, selected in red, or unavailable in black). The interface supports the four selection modes described above, including interactive single-click and drag-box selection and deselection. Five distinct box tools cover the most common needs: a standard selection box, a deselection box, a zigzag (checkerboard) selector for broader interleaved coverage, an interleaved-pairs selector that selects alternating row pairs, and a "deselect dependents" box that identifies and releases the selected electrodes that are blocking a target region of unavailable electrodes. User interactions trigger immediate recalculation of available electrodes, ensuring users receive instant visual feedback about constraint violations.

Notably, the GUI supports two powerful overlay modes displayed alongside the probe visualization. The **activity survey overlay** reads a per-contact activity file exported from SpikeGLX and renders a coloured bar beside each electrode, allowing users to identify channels with high spiking activity (typically grey matter) and adjust their channelmap accordingly. The **anatomical overlay** computes the brain region traversed by each electrode along a user-specified insertion trajectory—accounting for pitch, yaw, and multi-shank orientation—using any atlas registered in the BrainGlobe atlas API [@claudi2020]. It renders colour-coded depth bands on the probe and displays three orthogonal atlas slices (sagittal, coronal, horizontal) through the probe tip. Both overlays are preserved when downloading the PDF rendering of the channelmap, providing a complete experimental record.

## Forward Compatibility and Extensibility

PixelMap's architecture is designed to accommodate new hardware and atlas resources with minimal effort, lowering the barrier for both maintainers and contributors.

**Probe support.** Wiring constraints are encoded as standalone CSV files (`wiring_maps/*.csv`), one per probe type, and probe metadata (part numbers, readout counts, electrode pitches) is derived directly from the authoritative JSON maintained by SpikeGLX developer Bill Karsh. Adding support for a new Neuropixels probe released by IMEC requires only a new wiring CSV and an updated entry in the Bill Karsh JSON—no changes to the constraint-checking backend are needed, since it operates generically over hash-map lookups.

**Atlas support.** The anatomical overlay is built on the BrainGlobe Atlas API [@claudi2020], which provides a growing catalogue of brain atlases across species. Any atlas registered with BrainGlobe is automatically available to PixelMap users without code changes. In the Docker deployment, the two most widely used atlases—Allen Mouse Brain Common Coordinate Framework at 25 µm (`allen_mouse_25um`) and Waxholm Space Atlas of the Sprague Dawley Rat at 39 µm (`whs_sd_rat_39um`)—are pre-downloaded into the container image so they are available instantly. Additional atlases are downloaded on demand: when a user selects an atlas that has not yet been cached, the button label changes to "Download & compute atlas" to indicate a one-time download is in progress. Downloaded atlases are stored in a shared Docker volume (`brainglobe_cache`) that persists across container restarts, so each atlas only needs to be fetched once and is then available to all subsequent users of the server.

**Contributor extensibility.** The codebase is written as comprehensively as possible with contributions in mind. New selection presets can be added to backend.py, new overlay types can follow the pattern established by the activity-survey and anatomy overlays, and the GUI layer can be extended or replaced without modifying the core constraint logic. This modular design, together with comprehensive tests and a contributor guide (`CONTRIBUTING.md`), is intended to make it straightforward for the community to contribute features that serve their specific experimental workflows. These design choices have already enabled two new contributors to contribute significant features in 2026, warranting authorship on this paper: the anatomical overlay (J.M.J.F) and the activity survey overlay (J.Y.).

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

Evidence of adoption includes deployment at Princeton Neuroscience Institute's public server, community engagement on the project repository (37 GitHub stars), and measured web traffic: cookie analytics recorded approximately 400 unique visitors between March and May 2026, averaging roughly 45 unique visitors per week (see `https://github.com/m-beau/pixelmap/tree/main/analytics`). The package has been under active development for over ten months, with external contributors implementing new features (anatomical overlay, activity survey overlay).

# AI Usage Disclosure

**AI-assisted technologies used:** Claude Sonnet 4.5, Sonnet 4.6, and Opus 4.6 (Anthropic).
AI assistance was used for (1) optimization suggestions and documentation improvements (docstrings, code comments) in `backend.py`, (2) initial scaffolding of the Holoviz Panel GUI architecture in `gui/gui.py`, (3) manuscript grammatical and syntactical review. AI was not used for project conceptualization, core backend design, electrode wiring map construction. App hosting infrastructure was designed independently of AI assistance. All AI-generated code suggestions were reviewed and validated by human authors before integration; all AI-assisted manuscript edits were reviewed and approved by the corresponding author.

# Author Contributions

|                                    | Maxime Beau | Julie Fabre | Christian Tabedzki | Jorge Yanar | Carlos D. Brody |
|------------------------------------|:-----------:|:-----------:|:------------------:|:-----------:|:---------------:|
| Conceptualisation                  |      X      |             |                    |             |                 |
| Backend                            |      X      |             |                    |             |                 |
| GUI - general                      |      X      |      X      |                    |             |                 |
| GUI - Anatomical overlay           |             |      X      |                    |             |                 |
| GUI - Activity survey overlay      |             |             |                    |      X      |                 |
| App hosting                        |             |             |         X          |             |                 |
| Supervision and funding            |             |             |                    |             |        X        |

# Conflict of Interest Statement

The authors declare no competing interests.


# Acknowledgements

We thank Jesse C. Kaminsky and members of the Brody laboratory for testing and feedback during development, and PNI IT members Garrett McGrath and Gary Lyons for their advice concerning hosting. We also thank Bill Karsh for help with development and navigation of SpikeGLX resources. Finally, we thank the Princeton Neuroscience Institute for hosting the web application. J.M.J.F was supported by the Schmidt Science Fellows, in partnership with the Rhodes Trust. This work was supported by Howard Hughes Medical Institute and the National Institutes of Health.

# References
