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

PixelMap is a browser-based application for creating custom channelmaps for Neuropixels probes that respects electrode wiring constraints. Neuropixels probes, widely used for high-density neural recordings, have more physical electrodes than can be used for simultaneous recording because they contain fewer readout channels and analogue-to-digital converters (ADCs) than electrodes. Each ADC and readout channel is hard-wired to several electrodes, creating complex interdependencies where selecting one electrode makes others unavailable. PixelMap provides an installation-free, browser-based interface for researchers to design arbitrary recording configurations that meet their experimental requirements while satisfying these hardware constraints. Beyond constraint-aware user-friendly electrode selection, PixelMap supports two guided design workflows: overlaying anatomical region boundaries from any brain atlas available through BrainGlobe to plan recordings around specific brain regions, and overlaying a SpikeGLX activity survey heatmap to identify the most active channels after implantation. Both overlays can be used simultaneously during electrode selection. PixelMap generates IMRO (IMec Read Out) files compatible with SpikeGLX, the reference acquisition software for Neuropixels recordings.

# Statement of need

Neuropixels probes have revolutionised systems neuroscience by enabling simultaneous recordings from hundreds of neurons across multiple brain regions at any depth [@jun2017; @beau2021; @steinmetz2021; @bondy2024; @ye2025; @beau2025]. However, configuring these probes presents challenges. Limited by the number of readout channels and integrated analogue-to-digital converters (ADCs), Neuropixels probes contain 960–5120 physical electrodes per shank but can only record from 384–1536 channels simultaneously (Table 1). Users must therefore select a subset of electrodes to activate for each recording, forming an electrode selection set colloquially called "channelmap".

Because readout lines within each shank and ADCs in the probe head are each shared by multiple physical electrodes, activating one electrode makes others unavailable. For Neuropixels 1.0, these dependencies follow a regular, bank-aligned pattern that is relatively intuitive. With single-shank Neuropixels 2.0 probes [@steinmetz2021], whose wiring is intentionally scrambled to enable recording from several banks simultaneously, and four-shank Neuropixels 2.0 probes, whose wiring become untractable due to the high number of electrode banks, the inter-electrode incompatibilities become difficult to anticipate.

Beyond satisfying hardware constraints, channelmaps must be tailored to the experiment. Before implantation, researchers plan which brain regions each probe will traverse given its intended insertion trajectory; this requires mapping anatomical boundaries onto the physical electrode layout. After implantation, either in a chronic preparation where the probe remains in place for weeks, or an acute preparation on the same day as surgery, researchers must adjust their channelmap based on observed neural activity. Sites with large extracellular waveforms typically correspond to areas with a high density of neuron somata and axon initial segments (grey matter), which can be used to identify target regions entry and exit points. The typical workflow is therefore the following: plan anatomically before implantation, survey activity afterwards, and refine the channelmap to maximise unit yield in the target regions. The difficulty compounds when multiple probes are used simultaneously: experiments can use up to eight simultaneous Neuropixels probes [@bondy2024], each requiring to iterate on its own valid channelmap, making a fast and reliable wiring-aware design tool increasingly essential. Developing a tool to support this workflow within the Brody laboratory motivated the creation of PixelMap, which addresses these needs by:

1. **Being available on any machine installation-free**: The tool is available both as a browser-based web application at [https://pixelmap.pni.princeton.edu](https://pixelmap.pni.princeton.edu), as a Docker image, and a Python package.
2. **Visualising in real time Neuropixels wiring constraints, interactively**: When users select electrodes, the interface immediately shows which other electrodes become unavailable (marked in black) due to shared ADC lines, preventing invalid configurations.
3. **Supporting arbitrary electrode geometries**: Users can select electrodes by 1) choosing from common preset geometries, 2) entering electrode ranges as text for reproducibility, 3) directly clicking or dragging on the probe visualization, or 4) loading pre-existing `.imro` files. These four selection methods are fully intercompatible and can be combined. See the documentation [https://pixelmap-neuropixels.readthedocs.io](https://pixelmap-neuropixels.readthedocs.io/en/latest/) for details on supported preset geometries and selection methods, in particular the five different dragging boxes that enable great flexibility in electrode selection.
4. **Guiding channelmap design with anatomy and activity overlays**: Users can overlay anatomical region boundaries from any brain atlas available through BrainGlobe [@claudi2020] to plan recordings around specific brain regions of interest, and **simultaneously** a SpikeGLX activity survey heatmap to identify the most spiking-active channels after probe implantation.

| Probe Version | Physical Channels | Simultaneously Recordable Channels |
|---------------|-------------------|-------------------------------------|
| Neuropixels 1.0 | 960 | 384 |
| Neuropixels 2.0 (single shank) | 1,280 | 384 |
| Neuropixels 2.0 (4-shank) | 5,120 (1,280 per shank) | 384 |
| Neuropixels 2.0 Quad Base | 5,120 (1,280 per shank) | 1,536 (384 max per shank)|

**Table 1**: Number of physical and simultaneously addressable electrodes across Neuropixels probe versions supported by PixelMap.

# State of the Field

SpikeGLX ([https://billkarsh.github.io/SpikeGLX](https://billkarsh.github.io/SpikeGLX)) and Open Ephys ([https://open-ephys.github.io/gui-docs/User-Manual/Plugins/Neuropixels-PXI.html](https://open-ephys.github.io/gui-docs/User-Manual/Plugins/Neuropixels-PXI.html)), the two most widely used systems for acquiring Neuropixels data, allow channelmap editing but their primary purpose is to enable reliable data acquisition. Both are designed to run and monitor recordings; their priorities are stable high-throughput streaming, synchronisation with auxiliary data streams, and online visualisation. In both, channelmap editing is a configuration step performed at the rig, on the machine connected to the probe, and is optimised for fast on-the-fly setup at recording time. SpikeGLX's editor is also the field's authoritative reference for wiring constraints, since it obtains the electrode-to-readout correspondence directly from the manufacturer (IMEC); for this reason, PixelMap takes its version-specific constraints from this same source and, like SpikeGLX, writes channelmaps as `.imro` files. NeuroCarto [@su2025], a recently published browser-based editor that runs locally, offers a GUI that is built around a novel automated channelmap-generation algorithm: the user specifies a target electrode density for each region along the shank, and a channelmap meeting those constraints is generated.

PixelMap, by contrast, is built with the primary purpose of being a no-install generalist Neuropixels channelmap design tool. It brings together in a single tool the features otherwise distributed across the solutions above, including real-time visualisation of wiring constraints, `.imro` output from the same IMEC source, an interactive click-and-drag electrode selection editor, and anatomical reference, and adds several capabilities not available elsewhere. First, PixelMap is the only solution that requires no installation: it is hosted at [https://pixelmap.pni.princeton.edu](https://pixelmap.pni.princeton.edu) as a web app backed by a Docker image, and is also distributed as a Python package and a programmatic API, so a channelmap can be built on any machine without desktop acquisition software or a connected probe. Second, probe support follows the authoritative reference. Probe-group and subgroup definitions, which impact electrode wiring constraints as well as `.imro` file formats, are read from the source maintained by the SpikeGLX developer (see Acknowledgements and [http://billkarsh.github.io/SpikeGLX/help/imroTables](http://billkarsh.github.io/SpikeGLX/help/imroTables)). New probe types can become supported easily as they are added to SpikeGLX; current coverage is Neuropixels 1.0, 2.0 single- and four-shank, and 2.0 Quad Base. PixelMap also exposes the full probe-dependent `.imro` metadata, including the Neuropixels 1.0 hardware filter and gain settings and the electrode reference (external, tip, or join-tips). Third, PixelMap allows arbitrary electrode geometries: in addition to a large set of custom preset geometries, text entry of electrode ranges, and an `.imro` loader, it provides five drag-box tools, among them interleaved and checkerboard selectors and a "deselect dependents" box, unique to PixelMap, that frees the electrodes blocking an unavailable target region. Fourth, PixelMap implements an anatomical overlay built on the BrainGlobe Atlas API [@claudi2020], which makes any BrainGlobe atlas available immediately. PixelMap computes the region each electrode traverses along a user-specified insertion trajectory, and renders named depth bands beside the probe with three orthogonal atlas slices through the tip. Fifth, PixelMap provides an activity overlay that places a SpikeGLX activity survey next to the probe for channelmap refinement. Critically, the anatomical and activity overlays can be displayed simultaneously during electrode selection, which is very helpful for experiment planning and is not offered by any of the currently available software.

# Software Design

PixelMap is implemented in Python using [Holoviz Panel](https://panel.holoviz.org/) for the web interface, providing an interactive and responsive user experience. The software architecture consists of three main components.

First, the **probe type database and wiring maps** (distributed in `./constants.py`. `./utils/probe_features.py`, `./wiring_maps/*`) derives the mapping between probe part numbers, SpikeGLX probe type identifiers, and IMRO format numbers directly from the authoritative source maintained by SpikeGLX [http://billkarsh.github.io/SpikeGLX/help/imroTables](http://billkarsh.github.io/SpikeGLX/help/imroTables). This provides both accuracy and forward compatibility with new probe versions as they are added to SpikeGLX. The **wiring maps** at `./wiring_maps/*.csv` are CSV files describing the electrode-to-ADC mappings for each supported probe type. They were adapted from files provided by IMEC (Neuropixels manufacturer - downloadable [here](https://www.neuropixels.org/support)) and SpikeGLX (https://github.com/billkarsh/SpikeGLX/tree/a9024ba79f4481883766f5468563accb74fd58fd/Src-imro). PixelMap currently supports Neuropixels 1.0, 2.0 single-shank, 2.0 four-shank, and 2.0 Quad Base probes.

Second, the **electrode selection logic** at `./backend.py` implements the constraint-checking algorithms that validate electrode selections against probe-specific wiring maps. Cached (memoized) hash tables (Python dictionaries) are used to query incompatible electrode pairs fast, with O(1) complexity (so, fast).

Finally, the **graphical user interface** at `./gui/gui.py` was built with Holoviz Panel. The interface provides real-time visualisation of the probe layout with electrodes colour-coded by selection state (available in grey, selected in red, or unavailable in black). The interface supports the four selection modes described above, including interactive single-click and drag-box selection and deselection. Five distinct box tools cover the most common needs: a standard selection box, a deselection box, a zigzag (checkerboard) selector for broader interleaved coverage, an interleaved-pairs selector that selects alternating row pairs, and a "deselect dependents" box that identifies and releases the selected electrodes that are blocking a target region of unavailable electrodes. User interactions trigger immediate recalculation of available electrodes, ensuring users receive instant visual feedback about constraint violations.

Notably, the GUI supports two powerful overlay modes displayed alongside the probe visualization. The **activity survey overlay** reads a per-contact activity file exported from SpikeGLX and renders a coloured bar beside each electrode, allowing users to identify channels with high spiking activity (typically grey matter) and adjust their channelmap accordingly. The **anatomical overlay** computes the brain region traversed by each electrode along a user-specified insertion trajectory—accounting for pitch, yaw, and multi-shank orientation—using any atlas registered in the BrainGlobe atlas API [@claudi2020]. It renders colour-coded depth bands on the probe and displays three orthogonal atlas slices (sagittal, coronal, horizontal) through the probe tip. Both overlays are preserved when downloading the PDF rendering of the channelmap, providing a complete experimental record.

## Forward Compatibility and Extensibility

PixelMap's architecture is designed to accommodate new hardware and atlas resources with minimal effort, lowering the barrier for both maintainers and contributors.

**Probe support.** Wiring constraints are encoded as standalone CSV files (`wiring_maps/*.csv`), one per probe type, and probe metadata (part numbers, readout counts, electrode pitches) is derived directly from the authoritative JSON maintained by SpikeGLX developer Bill Karsh. Adding support for a new Neuropixels probe released by IMEC requires only a new wiring CSV and an updated entry in the JSON - no changes to the constraint-checking backend are needed, since it operates generically over hash-map lookups.

**Atlas support.** The anatomical overlay is built on the BrainGlobe Atlas API [@claudi2020], which provides a growing catalogue of brain atlases across species. Any atlas registered with BrainGlobe is automatically available to PixelMap users without code changes. In the Docker deployment, the two most widely used atlases, Allen Mouse Brain Common Coordinate Framework at 25 µm (`allen_mouse_25um`) and Waxholm Space Atlas of the Sprague Dawley Rat at 39 µm (`whs_sd_rat_39um`), are pre-downloaded into the container image so they are available instantly. Additional atlases are downloaded on demand: when a user selects an atlas that has not yet been cached, the button label changes to "Download & compute atlas" to indicate a one-time download is in progress. Downloaded atlases are stored in a shared Docker volume (`brainglobe_cache`) that persists across container restarts, so each atlas only needs to be fetched once and is then available to all subsequent users of the server.

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

For more details, see the documentation at [https://pixelmap-neuropixels.readthedocs.io](https://pixelmap-neuropixels.readthedocs.io).

The software includes an automated test suite covering hardware constraint validation, all preset configurations, IMRO file generation for all supported probe types, and end-to-end workflows. Tests run automatically via GitHub Actions continuous integration on every code change, ensuring software reliability. See the repository's `tests/` and `https://github.com/m-beau/pixelmap/blob/main/.github/workflows/tests.yml` for details.

# Research Impact Statement

PixelMap addresses a practical bottleneck in Neuropixels experimental workflows. Neuropixels have become the dominant technology for large-scale electrophysiology, with exponential growth in publications using the technology ([PubMed](https://esperr.github.io/pubmed-by-year/?q1=Neuropixels)). Yet no existing tool provided installation-free channelmap design with support for arbitrary electrode geometries and simultaneous anatomy- and activity-guided planning (see **Statement of Need**).

PixelMap demonstrates community-readiness through comprehensive documentation—including a dedicated contributor guide (CONTRIBUTING.md) with contribution workflows and a permissive open-source license (GPL3). The tool is immediately accessible via web application, Python package, Docker container, or programmatic API. The tool builds on the authors' established track record using Neuropixels probes in their research [@steinmetz2021; @bondy2024; @beau2025; @fabre2026basal] and developing Neuropixels software [@beau2021].

Evidence of adoption includes deployment at Princeton Neuroscience Institute's public server, community engagement on the project repository (56 GitHub stars), and measured web traffic: cookie analytics recorded approximately 400 unique visitors between March and May 2026, averaging roughly 45 unique visitors per week (see `https://github.com/m-beau/pixelmap/tree/main/analytics`). The package has been under active development for over ten months, with external contributors implementing new features (anatomical overlay, activity survey overlay).

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
