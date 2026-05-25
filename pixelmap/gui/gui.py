#!/usr/bin/env python3
"""
Interactive GUI for Neuropixels Channelmap Generation
Using Bokeh for better interactivity with hover, click, and rectangular selection
"""

import gc
import json
import logging
import re
from dataclasses import dataclass, field
from io import BytesIO, StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn
import param
from bokeh import events
from bokeh.models import (
    BoxSelectTool,
    ColumnDataSource,
    CustomJS,
    HoverTool,
    PanTool,
    Range1d,
    ResetTool,
    TapTool,
    WheelZoomTool,
)
from bokeh.plotting import figure
from bokeh.util.logconfig import basicConfig

from pixelmap import __version__, backend
from pixelmap.analytics import AnalyticsSessionTracker
from pixelmap.constants import (
    PROBE_N,
    PROBE_TYPE_MAP,
    REF_ELECTRODES,
    WIRING_FILE_MAP,
    SUPPORTED_1shank_PRESETS,
    SUPPORTED_4shanks_PRESETS,
)
from pixelmap.types import Electrode
from pixelmap.utils import imro
from pixelmap.utils import url_share
from pixelmap.anatomy import atlas as anatomy_atlas
from pixelmap.anatomy.schematic import render_locator
from pixelmap.anatomy.visualization import compute_region_bands

## Configure logging
basicConfig(level=logging.ERROR)  # no warnings

# Paths to assets
WIRING_MAPS_DIR = Path(__file__).resolve().parent.parent / "wiring_maps"
GUI_ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# Default port
DEFAULT_PORT = 5007

# Enable Panel extensions
pn.extension("tabulator", notifications=True)


def _darken_hex(hex_color: str, factor: float = 0.6) -> str:
    """Return ``hex_color`` darkened by ``factor`` (0=black, 1=unchanged)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (max(0, min(255, int(c * factor))) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


###################################
#### Electrode selection logic ####
###################################


@dataclass
class Electrodes:
    """Manages electrode selection state."""

    wiring_map: dict[Electrode, set[Electrode]]
    n_maximum_electrodes: int # TODO: implement per-shank maximum for NXTs
    available: set[Electrode] = field(default_factory=set)
    selected: set[Electrode] = field(default_factory=set)
    unavailable: set[Electrode] = field(default_factory=set)
    # Undo/redo stacks store frozen snapshots of `selected`; available and
    # unavailable are re-derived on restore so we never store stale views.
    _undo_stack: list[frozenset[Electrode]] = field(default_factory=list)
    _redo_stack: list[frozenset[Electrode]] = field(default_factory=list)
    _max_history: int = 100

    def __post_init__(self):
        self.available = set(self.wiring_map.keys())

    def select(self, electrode: Electrode):
        if electrode in self.available \
            and len(self.selected) < self.n_maximum_electrodes:
            self.selected.add(electrode)
            self.available.discard(electrode)

            conflicting_electrodes = self.wiring_map[electrode]
            newly_unavailable = conflicting_electrodes & self.available
            self.available -= newly_unavailable
            self.unavailable |= newly_unavailable

    def deselect(self, electrode: Electrode):
        if electrode in self.selected:
            self.selected.discard(electrode)
            self.available.add(electrode)

            nomore_conflicting_electrodes = self.wiring_map[electrode]
            newly_available = nomore_conflicting_electrodes & self.unavailable
            self.available |= newly_available
            self.unavailable -= newly_available

    def clear_selection(self):
        self.available = set(self.wiring_map.keys())
        self.selected = set()
        self.unavailable = set()

    # --- undo / redo --------------------------------------------------

    def push_undo_snapshot(self):
        """Snapshot the current selection so the next mutation is undoable.

        Must be called *before* any mutation that should be one undo step.
        The redo stack is cleared because any new action invalidates the
        previously-undone branch of history.
        """
        self._undo_stack.append(frozenset(self.selected))
        if len(self._undo_stack) > self._max_history:
            del self._undo_stack[0]
        self._redo_stack.clear()

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(frozenset(self.selected))
        self._restore_selection(self._undo_stack.pop())
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(frozenset(self.selected))
        self._restore_selection(self._redo_stack.pop())
        return True

    def _restore_selection(self, target: frozenset[Electrode]):
        """Replace `selected` with `target` and recompute derived sets."""
        self.available = set(self.wiring_map.keys())
        self.selected = set()
        self.unavailable = set()
        for electrode in target:
            if electrode in self.available:
                self.selected.add(electrode)
                self.available.discard(electrode)
                conflicting = self.wiring_map[electrode]
                newly_unavailable = conflicting & self.available
                self.available -= newly_unavailable
                self.unavailable |= newly_unavailable


#################
#### GUI app ####
#################

class ChannelmapGUI(param.Parameterized):
    """Main GUI class for interactive channelmap generation using Bokeh"""

    # Parameters - will take their value as attributes after class initialization
    default_type = "2.0-4shanks"
    download_button_color = param.String(default="default")
    download_button_label = param.String(default="Select...")

    probe_type = param.Selector(default=default_type, objects=list(PROBE_TYPE_MAP.keys()), doc="Neuropixels probe type")

    probe_subtype = param.Selector(
        default=PROBE_TYPE_MAP[default_type][0],
        objects=PROBE_TYPE_MAP[default_type],
        doc="Specific probe subtype (does not affect probe geometry, but affects indexing of reference and bank. Plug your probe in SpikeGLX and save an imro file to find out its subtype.)",
    )

    reference_id = param.Selector(
        default="External",
        objects=list(REF_ELECTRODES[PROBE_TYPE_MAP[default_type][0]].keys()),
        doc=(
            "Reference to use for recording (probe tip, external pad, or circuit ground (possible for some versions)."
            " Specific channels not implemented."
        ),
    )

    hardware_hp_filter_on = param.Selector(
        default=1,
        objects=[0, 1, None],
        doc=(
            "Whether to turn on Neuropixels 1.0 on-board hardware high-pass filter"
            " (analog equivalent to 1st-order 300Hz hp butterworth)"
        ),
    )

    preset = param.Selector(
        default=SUPPORTED_4shanks_PRESETS[0],
        objects=SUPPORTED_4shanks_PRESETS,
        doc="Channel map common presets",
    )

    shank_selector = param.Selector(
        default=0, objects=[0, 1, 2, 3], doc="Shank of electrodes selected in textbox below"
    )


    def __init__(self, **params):
        super().__init__(**params)

        # Initialize data
        self.wiring_maps_dir = WIRING_MAPS_DIR
        self.wiring_maps = backend.make_wiring_maps(self.wiring_maps_dir)

        # Probe plot geometry
        self.probe_plot_height = 2500
        self.probe_plot_width = 800

        # Probe metadata
        self.ap_gain_default = 500
        self.lf_gain_default = 250

        # Load initial data
        self.load_probe_data()

        # Create Bokeh plot
        self.setup_bokeh_plot()

        # Create widgets
        self.create_widgets()

        # Register cleanup on session destroy
        if pn.state.curdoc:
            pn.state.curdoc.on_session_destroyed(self._cleanup_session)

    def _cleanup_session(self, session_context):
        """Clean up resources when session ends"""

        self.clear_bokeh_data()

        if hasattr(self, 'electrode_source'):
            del self.electrode_source
        if hasattr(self, 'tool_state_source'):
            del self.tool_state_source
        if hasattr(self, 'plot'):
            del self.plot
        if hasattr(self, 'plot_pane'):
            del self.plot_pane

        gc.collect()
        print(f"🧹🧹🧹 Session {session_context.id} cleaned up")


    #####################################
    ##### Probe geometry and wiring #####
    #####################################

    def load_probe_data(self):
        """Load wiring and position data for current probe type"""
        # File mapping
        pos_file, wire_file = WIRING_FILE_MAP[self.probe_type]
        self.positions_file = self.wiring_maps_dir / pos_file
        self.wiring_file = self.wiring_maps_dir / wire_file

        # Probe subtype update
        self.param.probe_subtype.objects = PROBE_TYPE_MAP[self.probe_type]
        self.probe_subtype = self.param.probe_subtype.objects[0]

        # Load data
        self.positions_df = pd.read_csv(self.positions_file)
        self.wiring_df = pd.read_csv(self.wiring_file)

        # Initialize electrodes
        self.electrodes = Electrodes(self.wiring_maps[self.probe_type], PROBE_N[self.probe_type]['n'])

        # Update preset options based on probe type
        if self.probe_type in ["1.0", "2.0-1shank"]:
            self.param.preset.objects = SUPPORTED_1shank_PRESETS
            # For single shank probes, only shank 0 is available
            self.param.shank_selector.objects = [0]
            self.shank_selector = 0
        else:
            self.param.preset.objects = SUPPORTED_4shanks_PRESETS
            # For multi-shank probes, all 4 shanks are available
            self.param.shank_selector.objects = [0, 1, 2, 3]
            if self.shank_selector not in [0, 1, 2, 3]:
                self.shank_selector = 0
        self.preset = self.param.preset.objects[0]


    def create_electrode_data(self):
        """Create the electrode data for Bokeh visualization"""
        positions = self.positions_df.values

        # Calculate electrode positions and colors
        electrode_data = {
            "x": [],
            "y": [],
            "width": [],
            "height": [],
            "shank_id": [],
            "electrode_id": [],
            "color": [],
            "alpha": [],
            "line_color": [],
            "line_width": [],
            "status": [],
        }

        # Parameters for visualization
        if self.probe_type in ["1.0", "2.0-1shank"]:
            # Single shank
            shank_width = 100
            electrode_width = 15
            electrode_height = 9 if self.probe_type == "2.0-1shank" else 14
        else:
            # Multi-shank
            shank_width = 100
            shank_spacing = 250
            electrode_width = 12
            electrode_height = 8

        for shank_id, electrode_id, orig_x, y in positions:

            # Map x position to shank width
            if self.probe_type == "1.0":
                x_norm = (orig_x - 35) / 24
                x = x_norm * (shank_width * 0.7) / 2
            elif self.probe_type == "2.0-1shank":
                x_norm = (orig_x - 16) / 16
                x = x_norm * (shank_width * 0.7) / 2
            else: # 4-shanks 2.0 or NXT
                # Calculate shank center
                x_center = shank_id * shank_spacing
                # Map electrode position within shank
                x_norm = (orig_x - x_center - 16) / 16
                x = x_center + x_norm * (shank_width * 0.7) / 2

            # Determine electrode status and color
            status, color, alpha, line_color, line_width = self.get_electrode_plotting_params(Electrode(shank_id, electrode_id))

            electrode_data["x"].append(x)
            electrode_data["y"].append(y)
            electrode_data["width"].append(electrode_width)
            electrode_data["height"].append(electrode_height)
            electrode_data["shank_id"].append(shank_id)
            electrode_data["electrode_id"].append(electrode_id)
            electrode_data["color"].append(color)
            electrode_data["alpha"].append(alpha)
            electrode_data["line_color"].append(line_color)
            electrode_data["line_width"].append(line_width)
            electrode_data["status"].append(status)

        # Create ColumnDataSource
        self.electrode_source = ColumnDataSource(data=electrode_data)


    def get_electrode_plotting_params(self, electrode: Electrode):
        """
        Get electrode appearance based on its status
        status, color, alpha, line_color, line_width
        """
        if electrode in self.electrodes.selected:
            return "Selected", "red", 1.0, "darkred", 0
        elif electrode in self.electrodes.unavailable:
            return "Unavailable", "black", 1.0, "darkgray", 0
        else:  # unselected electrodes
            return "Unselected", "lightgray", 0.8, "gray", 0


    def setup_electrode_visualization(self):
        """Setup the electrode rectangles in Bokeh"""
        # Draw electrodes as rectangles
        self.electrode_renderer = self.plot.rect(
            x="x",
            y="y",
            width="width",
            height="height",
            fill_color="color",
            fill_alpha="alpha",
            line_color="line_color",
            line_width="line_width",
            source=self.electrode_source,
            hover_fill_color="yellow",
            hover_line_color="orange",
            hover_line_width=3,
        )

        # Add shank outlines and labels
        self.add_shank_outlines()


    def add_shank_outlines(self):
        """Add shank outlines and bank labels"""
        positions = self.positions_df.values

        if self.probe_type in ["1.0", "2.0-1shank"]:
            # Single shank outline
            shank_width = 100
            xlim = [-shank_width / 2 - 150, shank_width / 2 + 150]
            max_y = np.max(positions[:, -1])
            min_y = np.min(positions[:, -1])
            tip_height = (max_y - min_y) * 0.1

            # Shank outline
            shank_x = [-shank_width / 2, shank_width / 2, shank_width / 2, 0, -shank_width / 2, -shank_width / 2]
            shank_y = [max_y + 100, max_y + 100, min_y, min_y - tip_height, min_y, max_y + 100]

            self.plot.line(shank_x, shank_y, line_width=3, color="black", alpha=1)
            self.plot.x_range = Range1d(xlim[0], xlim[1])

            # Bank labels
            for bank_i in np.arange(0, len(positions), 384):
                if bank_i < len(positions):
                    bank_y = positions[bank_i, -1]
                    self.plot.line(
                        [-shank_width / 2, shank_width / 2], [bank_y, bank_y], line_width=2, color="gray", alpha=0.7
                    )
                    self.plot.text(
                        [shank_width / 2 + 3],
                        [bank_y],
                        text=[f"Bank {bank_i // 384}"],
                        text_font_size="10pt",
                        text_color="gray",
                    )

        else:
            # Multi-shank outlines
            shank_width = 100
            shank_spacing = 250
            xlim = [-shank_width / 2 - 100, 3 * shank_spacing + shank_width / 2 + 100]

            for shank_id in range(4):
                shank_mask = positions[:, 0] == shank_id
                x_center = shank_id * shank_spacing

                max_y = np.max(positions[shank_mask, -1])
                min_y = np.min(positions[shank_mask, -1])
                tip_height = (max_y - min_y) * 0.08

                # Shank outline
                shank_x = [
                    x_center - shank_width / 2,
                    x_center + shank_width / 2,
                    x_center + shank_width / 2,
                    x_center,
                    x_center - shank_width / 2,
                    x_center - shank_width / 2,
                ]
                shank_y = [max_y + 100, max_y + 100, min_y, min_y - tip_height, min_y, max_y + 100]

                self.plot.line(shank_x, shank_y, line_width=3, color="black", alpha=1)
                self.plot.x_range = Range1d(xlim[0], xlim[1])

                # Shank label
                self.plot.text(
                    [x_center],
                    [min_y - tip_height - 250],
                    text=[f"Shank {shank_id}"],
                    text_font_size="12pt",
                    text_color="black",
                    text_align="center",
                )

                # Bank lines and labels
                for bank_i in np.arange(0, len(positions[shank_mask]), 384):
                    if bank_i < len(positions[shank_mask]):
                        bank_y = positions[shank_mask][bank_i, -1]
                        self.plot.line(
                            [x_center - shank_width / 2, x_center + shank_width / 2],
                            [bank_y, bank_y],
                            line_width=2,
                            color="gray",
                            alpha=0.7,
                        )
                        if shank_id == 3:  # Bank labels (only on rightmost shank)
                            self.plot.text(
                                [x_center + shank_width / 2 + 5],
                                [bank_y],
                                text=[f"Bank {bank_i // 384}"],
                                text_font_size="10pt",
                                text_color="gray",
                            )

        # Set appropriate axis limits
        self.plot.axis.visible = False
        self.plot.grid.visible = False


    def update_electrode_colors(self):
        """Update electrode colors in the Bokeh plot"""
        n_electrodes = len(self.electrode_source.data["shank_id"])

        # Pre-allocate numpy arrays for better memory efficiency
        colors = np.empty(n_electrodes, dtype=object)
        alphas = np.empty(n_electrodes, dtype=np.float64)
        line_colors = np.empty(n_electrodes, dtype=object)
        line_widths = np.empty(n_electrodes, dtype=np.int32)
        statuses = np.empty(n_electrodes, dtype=object)

        for i in range(n_electrodes):
            shank_id = self.electrode_source.data["shank_id"][i]
            electrode_id = self.electrode_source.data["electrode_id"][i]
            electrode = Electrode(shank_id, electrode_id)

            status, color, alpha, line_color, line_width = self.get_electrode_plotting_params(electrode)

            colors[i] = color
            alphas[i] = alpha
            line_colors[i] = line_color
            line_widths[i] = line_width
            statuses[i] = status

        # Update the data source with new data, replacing old arrays
        self.electrode_source.data.update(
            {"color": colors.tolist(), "alpha": alphas.tolist(),
             "line_color": line_colors.tolist(), "line_width": line_widths.tolist(),
             "status": statuses.tolist()}
        )


    #####################################
    ##### Electrode selection logic #####
    #####################################

    def on_electrode_selection(self, _, old, new):
        """Handle electrode selection changes (both tap and box select)"""
        if not new:  # No selection
            return

        print(f"Selection changed: {old} -> {new}")

        # One interactive gesture (tap, box drag, zigzag drag) = one undo step.
        self.electrodes.push_undo_snapshot()

        # Single electrode selection - toggle
        if len(new) == 1:
            shank_id = self.electrode_source.data["shank_id"][new[0]]
            electrode_id = self.electrode_source.data["electrode_id"][new[0]]
            electrode = Electrode(shank_id, electrode_id)

            if electrode in self.electrodes.selected:
                self.electrodes.deselect(electrode)
            else:
                self.electrodes.select(electrode)

        # Box selecttion (multiple electrodes)
        elif len(new) > 1:

            if self.select_mode == "deselect":
                print(f"Box deselect: {len(new)} electrodes")
                for idx in new:
                    shank_id = self.electrode_source.data["shank_id"][idx]
                    electrode_id = self.electrode_source.data["electrode_id"][idx]
                    self.electrodes.deselect(Electrode(shank_id, electrode_id))


            elif self.select_mode == "select":
                print(f"Box select: {len(new)} electrodes")
                for idx in new:
                    shank_id = self.electrode_source.data["shank_id"][idx]
                    electrode_id = self.electrode_source.data["electrode_id"][idx]
                    self.electrodes.select(Electrode(shank_id, electrode_id))

            elif self.select_mode == "zigzag_select":
                # zigzag logic - even electrodes in 1.0, 0, 3, 4, 7, 8... if 2.0
                print(f"Box zigzag select: {len(new)} electrodes")
                zigzag_subset = self.get_zigzag_subset()
                for idx in new:
                    shank_id = self.electrode_source.data["shank_id"][idx]
                    electrode_id = self.electrode_source.data["electrode_id"][idx]
                    if electrode_id in zigzag_subset:
                        self.electrodes.select(Electrode(shank_id, electrode_id))

        # Update electrode visualization
        self.update_electrode_colors()
        self.update_electrode_counter()
        self.refresh_undo_buttons()

        # Clear the selection to allow for new interactions
        self.electrode_source.selected.indices = []

        gc.collect()


    def get_zigzag_subset(self):
        N_per_shank = PROBE_N[self.probe_type]["N"]
        if self.probe_type == "1.0":
            zigzag_subset = np.arange(0, N_per_shank, 2)
        else:
            zigzag_subset = []
            i = 0
            while i < N_per_shank:  # Limit to 192 channels for bank 0
                zigzag_subset.append(i)
                zigzag_subset.append(i + 3)
                i += 4
            zigzag_subset = np.array(zigzag_subset)
        return zigzag_subset


    def apply_preset(self):
        """Apply selected preset configuration"""
        if self.preset:
            self.electrodes.push_undo_snapshot()
            self.electrodes.clear_selection()
            preset_electrodes = backend.get_preset_candidates(self.preset, self.probe_type, self.wiring_df)
            for shank_id, electrode_id in preset_electrodes:
                self.electrodes.select(Electrode(shank_id, electrode_id))

            # Update visualization
            self.update_electrode_colors()
            self.update_electrode_counter()
            self.refresh_undo_buttons()


    def parse_electrode_input(self, text):
        """Parse electrode input string like '1,2,3,5,7' or '1-5,7'"""
        if not text.strip():
            return []

        electrodes = []
        # Use the selected shank from the shank selector
        shank_id = self.shank_selector

        # Split by commas and process each part
        parts = text.split(",")
        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Check for range notation (e.g., '1-5' or '1..5')
            range_match = re.match(r"(\d+)[-.]\.?(\d+)", part)
            if range_match:
                start, end = map(int, range_match.groups())
                for e_id in range(start, end + 1):
                    electrodes.append((shank_id, e_id))
            else:
                # Single electrode
                try:
                    e_id = int(part)
                    electrodes.append((shank_id, e_id))
                except ValueError:
                    continue

        return electrodes


    def apply_electrode_input(self):
        """Add electrodes from text input to current selection (same logic as box selection)"""
        try:
            text = self.electrode_input.value
            electrodes = self.parse_electrode_input(text)
            if not electrodes:
                return

            self.electrodes.push_undo_snapshot()
            for (shank_id, electrode_id) in electrodes:
                self.electrodes.select(Electrode(shank_id, electrode_id))

            # Update visualization
            self.update_electrode_colors()
            self.update_electrode_counter()
            self.refresh_undo_buttons()

        except Exception as e:
            print(f"Error parsing electrode input: {e}")


    def clear_selection(self):
        """Clear all selected electrodes"""
        if self.electrodes.selected:
            self.electrodes.push_undo_snapshot()
        self.electrodes.clear_selection()

        # Update visualization
        self.update_electrode_colors()
        self.update_electrode_counter()
        self.refresh_undo_buttons()


    ###############################
    ##### IMRO and PDF output #####
    ###############################

    def ready_to_download(self):
        # Check number of selected electrodes
        if not any(self.electrodes.selected):
            print("No electrodes selected")
            return False
        n_selected = len(self.electrodes.selected)
        max_allowed = self.electrodes.n_maximum_electrodes
        if n_selected < max_allowed:
            print(f"IMRO file NOT generated - you still have {max_allowed - n_selected} electrodes to select!")
            return False
        return True

    def update_filename(self, reset=False):
        if reset:
            self.filename_input.value = f"manual_selection_{self.probe_type}"
        self.filename_input.value = self.filename_input.value.replace(".imro", "").replace(".pdf", "").replace(".json", "")
        self.download_imro_button.filename = f"{self.filename_input.value}.imro"
        self.download_pdf_button.filename = f"{self.filename_input.value}.pdf"
        self.download_kilosort_button.filename = f"{self.filename_input.value}.json"

    def generate_imro(self):
        """Generate IMRO file from current selection"""

        # Convert selection to array format
        selected_array = np.array([[e.shank_id, e.electrode_id] for e in  self.electrodes.selected])

        # Generate IMRO list
        self.imro_list = imro.generate_imro_channelmap(
            probe_type=self.probe_type,
            custom_electrodes=selected_array,
            wiring_file=self.wiring_file,
            layout_preset=None,
            reference_id=self.reference_id,
            probe_subtype=self.probe_subtype,
            ap_gain=self.ap_gain_input.value,
            lf_gain=self.lf_gain_input.value,
            hp_filter=self.hardware_hp_filter_on,
        )

    def generate_imro_content(self):
        if not self.ready_to_download():
            return

        # Update filename and generate IMRO list
        self.update_filename()
        self.generate_imro()

        # Convert list to string
        content_lines = []
        header = self.imro_list[0]
        content_lines.append(f"({header[0]},{header[1]})")

        for entry in self.imro_list[1:]:
            entry_str = " ".join(str(x) for x in entry)
            content_lines.append(f"({entry_str})")

        content = "".join(content_lines)
        return StringIO(content)

    def generate_pdf_content(self):
        if not self.ready_to_download():
            return

        # Update filename and generate IMRO list
        self.update_filename()
        self.generate_imro()

        # Create memory buffer
        buffer = BytesIO()

        try:
            # Make figure
            title = self.filename_input.value
            backend.plot_probe_layout(
                self.probe_type,
                self.imro_list,
                self.positions_file,
                self.wiring_file,
                title,
                figsize=(2, 30),
                save_plot=False,
            )

            # Save current figure to buffer
            plt.savefig(buffer, format="pdf", dpi=300, bbox_inches="tight")
            buffer.seek(0)

        finally:
            plt.close('all')
            gc.collect()

        return buffer

    def generate_kilosort_channelmap(self):
        if not self.ready_to_download():
            return

        # Update filename and generate IMRO list
        self.update_filename()
        self.generate_imro()

        # Generate Kilosort channelmap dictionary
        kilosort_dict = backend.generate_kilosort_channelmap_dict(
            self.imro_list,
            self.positions_file
        )

        # Convert numpy arrays to lists for JSON serialization
        kilosort_dict_serializable = {
            'chanMap': kilosort_dict['chanMap'].tolist(),
            'xc': kilosort_dict['xc'].tolist(),
            'yc': kilosort_dict['yc'].tolist(),
            'kcoords': kilosort_dict['kcoords'].tolist(),
            'n_chan': int(kilosort_dict['n_chan'])
        }

        # Convert to JSON string and return as StringIO
        json_str = json.dumps(kilosort_dict_serializable, indent=2)
        return StringIO(json_str)

    def apply_uploaded_imro(self):

        if self.imro_file_loader.value is None:
            pn.state.notifications.warning("No .imro file found - upload one before clicking this button.",
                                    duration=10_000)
            return

        file_extension = str(self.imro_file_loader.filename).split(".")[-1]
        if file_extension != "imro":
            pn.state.notifications.error(f"You must upload an .imro file, not .{file_extension}!",
                                    duration=10_000)
            return

        imro_file_content = self.imro_file_loader.value
        if isinstance(imro_file_content, bytes):
            imro_file_content = imro_file_content.decode("utf-8")
        imro_list = imro.parse_imro_file(imro_file_content.strip())

        try:
            (imro_electrodes,
            self.probe_type,  # probe_type value is a monitored param - simply setting its value will update the plot
            self.probe_subtype,
            self.reference_id,
            self.ap_gain_input.value,
            self.lf_gain_input.value,
            self.hardware_hp_filter_on,
            ) = imro.parse_imro_list(imro_list)
        except:
            pn.state.notifications.error("Failed to parse uploaded imro file.")
            return

        self.electrodes.push_undo_snapshot()
        self.electrodes.clear_selection()
        for shank_id, electrode_id in imro_electrodes:
            self.electrodes.select(Electrode(shank_id, electrode_id))

        self.update_electrode_colors()
        self.update_electrode_counter()
        self.refresh_undo_buttons()

    ###################################
    ##### Anatomical overlay logic ####
    ###################################

    def setup_region_bands(self):
        """Initialize the empty Bokeh sources + glyphs for the anatomy overlay.

        Three layers, drawn back-to-front so labels stay on top of bands:
          - colored band rectangles (fill behind the shank),
          - boundary tick marks (horizontal lines at region transitions),
          - region acronym text labels next to each band.
        """
        self.region_band_source = ColumnDataSource(
            data={"x": [], "y": [], "width": [], "height": [], "color": [], "acronym": []}
        )
        self.region_band_renderer = self.plot.rect(
            x="x", y="y", width="width", height="height",
            fill_color="color", fill_alpha=0.25, line_color=None,
            source=self.region_band_source,
        )

        self.region_boundary_source = ColumnDataSource(
            data={"x0": [], "x1": [], "y0": [], "y1": []}
        )
        self.region_boundary_renderer = self.plot.segment(
            x0="x0", y0="y0", x1="x1", y1="y1",
            line_color="#444444", line_width=1, line_dash="dashed",
            source=self.region_boundary_source,
        )

        self.region_label_source = ColumnDataSource(
            data={"x": [], "y": [], "text": [], "color": []}
        )
        self.region_label_renderer = self.plot.text(
            x="x", y="y", text="text", text_color="color",
            text_font_size="10pt", text_baseline="middle", text_align="left",
            source=self.region_label_source,
        )

    def _available_atlases(self) -> tuple[list[str], str]:
        """List atlases known to brainglobe; fall back to a single default."""
        default = "allen_mouse_25um"
        if not anatomy_atlas.is_available():
            return [default], default
        try:
            atlases = anatomy_atlas.list_atlases()
        except Exception:
            return [default], default
        if not atlases:
            return [default], default
        if default not in atlases:
            atlases = [default] + list(atlases)
        return list(atlases), default

    def _shank_probe_xs(self) -> dict[int, float]:
        """Probe-local x (µm) at the *median* of each shank's electrodes."""
        return {
            int(s): float(self.positions_df[self.positions_df["shank"] == s]["x"].median())
            for s in sorted(self.positions_df["shank"].unique())
        }

    def _shank_plot_centers(self) -> dict[int, float]:
        """Bokeh-plot x at the center of each shank's drawn outline."""
        if self.probe_type in ["1.0", "2.0-1shank"]:
            return {0: 0.0}
        return {sid: sid * 250.0 for sid in self._shank_probe_xs()}

    def _shank_plot_width(self) -> float:
        """Width (in plot units) to draw region bands at, narrower than the shank."""
        return 90.0  # shank outlines are 100 wide

    def compute_anatomy_overlay(self):
        """Compute region bands for the current pose and render them."""
        if not anatomy_atlas.is_available():
            if pn.state.notifications is not None:
                pn.state.notifications.error(
                    "Anatomy support not installed. Run: pip install 'pixelmap[anatomy]'",
                    duration=10_000,
                )
            return

        probe_xs = self._shank_probe_xs()
        plot_centers = self._shank_plot_centers()
        y_max = float(self.positions_df["y"].max())

        tip = (
            float(self.tip_ap_input.value),
            float(self.tip_ml_input.value),
            float(self.tip_dv_input.value),
        )

        try:
            bands = compute_region_bands(
                shank_positions=probe_xs,
                y_range=(0.0, y_max),
                tip_atlas=tip,
                pitch_deg=float(self.pitch_input.value),
                yaw_deg=float(self.yaw_input.value),
                shank_orientation_deg=float(self.shank_orientation_input.value),
                atlas_name=str(self.atlas_name_input.value).strip(),
                step_um=25.0,
            )
        except ImportError as exc:
            if pn.state.notifications is not None:
                pn.state.notifications.error(str(exc), duration=10_000)
            return
        except Exception as exc:
            print(f"Anatomy lookup failed: {exc}")
            if pn.state.notifications is not None:
                pn.state.notifications.error(f"Anatomy lookup failed: {exc}", duration=10_000)
            return

        width = self._shank_plot_width()
        half_width = width / 2
        # Pull label text slightly outside the shank outline so it doesn't
        # overlap the electrode rects.
        label_offset = half_width + 6

        band_data = {"x": [], "y": [], "width": [], "height": [], "color": [], "acronym": []}
        label_data = {"x": [], "y": [], "text": [], "color": []}
        boundary_data = {"x0": [], "x1": [], "y0": [], "y1": []}
        # Track which boundaries we've already drawn per shank so adjacent
        # bands sharing a transition don't double up on the line.
        seen_boundaries: set[tuple[int, float]] = set()

        for band in bands:
            plot_center = plot_centers.get(band.shank_id)
            if plot_center is None:
                continue
            y_mid = (band.y_min + band.y_max) / 2
            color_hex = "#{:02x}{:02x}{:02x}".format(*band.rgb)

            band_data["x"].append(plot_center)
            band_data["y"].append(y_mid)
            band_data["width"].append(width)
            band_data["height"].append(band.y_max - band.y_min)
            band_data["color"].append(color_hex)
            band_data["acronym"].append(band.acronym)

            label_data["x"].append(plot_center + label_offset)
            label_data["y"].append(y_mid)
            label_data["text"].append(band.acronym)
            # Slightly darker than the swatch so labels stay readable on white.
            label_data["color"].append(_darken_hex(color_hex, factor=0.6))

            for boundary_y in (band.y_min, band.y_max):
                key = (band.shank_id, round(boundary_y, 3))
                if key in seen_boundaries:
                    continue
                seen_boundaries.add(key)
                boundary_data["x0"].append(plot_center - half_width)
                boundary_data["x1"].append(plot_center + half_width)
                boundary_data["y0"].append(boundary_y)
                boundary_data["y1"].append(boundary_y)

        self.region_band_source.data = band_data
        self.region_label_source.data = label_data
        self.region_boundary_source.data = boundary_data

        self._update_anatomy_legend(bands)
        # The atlas is now downloaded, so the origin corner can be shown.
        self._update_anatomy_origin_note()

        try:
            self.anatomy_locator.object = render_locator(
                str(self.atlas_name_input.value).strip(),
                tip_atlas=tip,
                pitch_deg=float(self.pitch_input.value),
                yaw_deg=float(self.yaw_input.value),
                shank_orientation_deg=float(self.shank_orientation_input.value),
                shank_positions=probe_xs,
                y_range=(0.0, y_max),
            )
        except Exception as exc:  # a failed locator shouldn't sink the overlay
            print(f"Locator render failed: {exc}")

        if pn.state.notifications is not None and bands:
            n_regions = len({b.atlas_id for b in bands})
            pn.state.notifications.success(
                f"Found {n_regions} region(s) across {len(probe_xs)} shank(s).",
                duration=4_000,
            )

    def clear_anatomy_overlay(self):
        """Wipe the region bands, labels, boundaries and reset the legend."""
        self.region_band_source.data = {
            "x": [], "y": [], "width": [], "height": [], "color": [], "acronym": [],
        }
        self.region_label_source.data = {"x": [], "y": [], "text": [], "color": []}
        self.region_boundary_source.data = {"x0": [], "x1": [], "y0": [], "y1": []}
        self.anatomy_legend.object = self._empty_legend_html()
        self.anatomy_locator.object = None

    def _empty_legend_html(self) -> str:
        return (
            '<div style="font-size: 12px; color: #666; padding: 4px 12px;">'
            "No anatomical overlay computed yet."
            "</div>"
        )

    def _anatomy_coord_note_html(self) -> str:
        """Coordinate-space note, including the selected atlas's origin corner.

        The origin differs per atlas, so it's fetched from the atlas itself —
        but only when already downloaded, to keep dropdown changes cheap.
        """
        name = str(self.atlas_name_input.value).strip()
        origin = None
        if anatomy_atlas.is_available():
            try:
                if anatomy_atlas.is_downloaded(name):
                    origin = anatomy_atlas.origin_corner(name)
            except Exception:
                origin = None
        if origin:
            origin_line = (
                f"For <b>{name}</b>, (0,0,0) is the <b>{origin}</b> corner of the "
                "volume; coordinates increase away from it."
            )
        else:
            origin_line = (
                f"The (0,0,0) origin corner for <b>{name}</b> is shown once the "
                "atlas is downloaded (on first overlay compute)."
            )
        return (
            '<div style="font-size: 11px; color: #555; padding: 2px 0 6px 0;">'
            "Tip coordinates are in <b>CCF / atlas space</b> — µm from the atlas "
            "volume's origin corner, <i>not</i> bregma-relative stereotaxic "
            "coordinates. Brainglobe atlases don't define a bregma, so there's no "
            f"built-in conversion.<br>{origin_line}"
            "</div>"
        )

    def _update_anatomy_origin_note(self, *events):
        """Refresh the coordinate note when the atlas selection changes."""
        self.anatomy_coord_note.object = self._anatomy_coord_note_html()

    def _update_anatomy_legend(self, bands: list):
        """Render a deduplicated region legend on the right-side panel."""
        if not bands:
            self.anatomy_legend.object = (
                '<div style="font-size: 12px; color: #b00; padding: 4px 12px;">'
                "No regions found at the given pose (probe outside atlas volume?)."
                "</div>"
            )
            return

        # Deduplicate by atlas_id, preserve a stable display order.
        seen: dict[int, "RegionBand"] = {}
        for band in bands:
            seen.setdefault(band.atlas_id, band)

        rows = []
        for band in seen.values():
            swatch = "#{:02x}{:02x}{:02x}".format(*band.rgb)
            rows.append(
                f'<div style="display: flex; align-items: center; gap: 8px; padding: 2px 0; font-size: 12px;">'
                f'<span style="display: inline-block; width: 14px; height: 14px; '
                f'background: {swatch}; border: 1px solid #888;"></span>'
                f'<span style="font-weight: bold; min-width: 60px;">{band.acronym}</span>'
                f'<span style="color: #444;">{band.name}</span>'
                f"</div>"
            )
        self.anatomy_legend.object = (
            '<div style="padding: 4px 12px; max-height: 280px; overflow-y: auto;">'
            + "".join(rows)
            + "</div>"
        )

    def apply_url_state(self, encoded: str) -> bool:
        """Restore a selection previously serialized via build_share_link.

        Returns True if the state was applied, False if the payload was
        invalid or the embedded probe/subtype is unknown.
        """
        state = url_share.decode_state(encoded)
        if state is None:
            return False
        if state["probe_type"] not in PROBE_TYPE_MAP:
            return False

        # Setting probe_type triggers on_probe_type_change which rebuilds
        # widgets and clears the selection — must be set first.
        if state["probe_type"] != self.probe_type:
            self.probe_type = state["probe_type"]

        subtype = state["probe_subtype"]
        if subtype is not None and subtype in self.param.probe_subtype.objects:
            self.probe_subtype = subtype

        ref = state["reference_id"]
        if ref is not None and ref in self.param.reference_id.objects:
            self.reference_id = ref

        if state["ap_gain"] is not None:
            self.ap_gain_input.value = float(state["ap_gain"])
        if state["lf_gain"] is not None:
            self.lf_gain_input.value = float(state["lf_gain"])
        if state["hp_filter"] in self.param.hardware_hp_filter_on.objects:
            self.hardware_hp_filter_on = state["hp_filter"]

        if self.electrodes.selected:
            self.electrodes.push_undo_snapshot()
        self.electrodes.clear_selection()
        for shank_id, electrode_id in state["electrodes"]:
            self.electrodes.select(Electrode(shank_id, electrode_id))

        self.update_electrode_colors()
        self.update_electrode_counter()
        self.refresh_undo_buttons()
        return True

    def update_share_link(self):
        """Refresh the share-link text field and update the URL bar."""
        query = self.build_share_link()
        full_url = query
        try:
            location = pn.state.location
            if location is not None:
                origin = (getattr(location, "protocol", "") or "") + "//" + (getattr(location, "hostname", "") or "")
                port = getattr(location, "port", "") or ""
                if port and port not in ("80", "443"):
                    origin = f"{origin}:{port}"
                pathname = getattr(location, "pathname", "") or ""
                if origin.startswith("//"):  # no protocol info available
                    full_url = pathname + query
                else:
                    full_url = origin + pathname + query
                # Reflect the new state in the address bar so it's also copyable from there.
                location.search = query
        except Exception:
            pass
        self.share_link_output.value = full_url
        if pn.state.notifications is not None:
            pn.state.notifications.success("Share link ready — copy from the field below.", duration=5_000)

    def do_undo(self):
        """Roll back to the previous selection snapshot."""
        if self.electrodes.undo():
            self.update_electrode_colors()
            self.update_electrode_counter()
            self.refresh_undo_buttons()

    def do_redo(self):
        """Re-apply a previously-undone selection snapshot."""
        if self.electrodes.redo():
            self.update_electrode_colors()
            self.update_electrode_counter()
            self.refresh_undo_buttons()

    def refresh_undo_buttons(self):
        """Sync the disabled flag of the undo/redo buttons with stack state."""
        if hasattr(self, "undo_button"):
            self.undo_button.disabled = not self.electrodes.can_undo()
        if hasattr(self, "redo_button"):
            self.redo_button.disabled = not self.electrodes.can_redo()

    def build_share_link(self) -> str:
        """Encode the current selection as a query string fragment.

        Returns just the query string (e.g. ``?cfg=...``) — the GUI prepends
        the live origin + path on the client side so this method stays
        independent of how the app is being served.
        """
        encoded = url_share.encode_state(
            probe_type=self.probe_type,
            probe_subtype=self.probe_subtype,
            reference_id=self.reference_id,
            ap_gain=self.ap_gain_input.value,
            lf_gain=self.lf_gain_input.value,
            hp_filter=self.hardware_hp_filter_on,
            electrodes=[(e.shank_id, e.electrode_id) for e in self.electrodes.selected],
        )
        return f"?{url_share.QUERY_PARAM}={encoded}"

    ######################
    ##### GUI layout #####
    ######################

    def setup_bokeh_plot(self):
        """Setup the Bokeh plot for electrode visualization"""
        # Create custom box select tools with visual distinction
        self.select_mode = None

        select_box_string = "Select Electrodes"
        deselect_box_string = "Deselect Electrodes"
        zigzagselect_box_string = "Zigzag-select Electrodes"

        self.box_select_tool = BoxSelectTool(description=select_box_string, icon=GUI_ASSETS_DIR / "selector.png")

        self.box_deselect_tool = BoxSelectTool(
            description=deselect_box_string, icon=GUI_ASSETS_DIR / "deselector.png"
        )

        self.box_zigzagselect_tool = BoxSelectTool(
            description=zigzagselect_box_string, icon=GUI_ASSETS_DIR / "zigzag_selector.png"
        )

        # Create figure with proper tools.
        # We keep explicit references to PanTool and WheelZoomTool so we can
        # mark them as the default active drag and scroll tools. Without
        # this, Bokeh would require the user to click the toolbar icon first
        # before click-drag could pan or wheel could zoom.
        pan_tool = PanTool()
        wheel_zoom_tool = WheelZoomTool()
        tools = [
            pan_tool,
            wheel_zoom_tool,
            self.box_select_tool,
            self.box_deselect_tool,
            self.box_zigzagselect_tool,
            TapTool(),
            ResetTool(),
            HoverTool(
                tooltips=[
                    ("Electrode", "@electrode_id"),
                    ("Shank", "@shank_id"),
                    ("Z position", "@y μm"),
                    ("Status", "@status"),
                ]
            ),
        ]

        self.plot = figure(
            width=self.probe_plot_width,
            height=self.probe_plot_height,
            tools=tools,
            title=f"Neuropixels {self.probe_type} Electrode Layout",
            toolbar_location="right",
            active_drag=pan_tool,
            active_scroll=wheel_zoom_tool,
        )

        # Region bands must be added *before* the electrode rects so they
        # render behind everything else.
        self.setup_region_bands()

        # Create electrode data and visualization
        self.create_electrode_data()
        self.setup_electrode_visualization()
        self.setup_interactions()  # Only necessary for the tap tool

        # Hidden data source for tool state communication and CustomJS to monitor tool changes
        self.tool_state_source = ColumnDataSource(data={"active_tool": [""]})
        self.setup_tool_monitoring(select_box_string, deselect_box_string, zigzagselect_box_string)

    def setup_interactions(self):
        """Setup click and selection interactions"""
        # Set up the TapTool to enable single electrode selection
        tap_tool = self.plot.select_one(TapTool)
        if tap_tool:
            tap_tool.callback = CustomJS(
                args=dict(source=self.electrode_source),
                code="""
                console.log('Tap tool activated');
                const indices = source.selected.indices;
                console.log('Selected indices:', indices);
            """,
            )

        # Python callbacks for interactions - this is the key part
        self.electrode_source.selected.on_change("indices", self.on_electrode_selection)

    def setup_tool_monitoring(self, select_box_string, deselect_box_string, zigzagselect_box_string):
        """JS callback to expose active tools in toolbar"""
        # workaround https://stackoverflow.com/questions/58210752/how-to-get-currently-active-tool-in-bokeh-figure

        js_code = """
        let tools_info = 'Could not check tools';
        const tools = cb_obj.origin.toolbar.tools;

        if (tools && tools.length > 0) {
            // Find our specific tools
            let select_tool = null;
            let deselect_tool = null;
            let zigzagselect_tool = null;

            tools.forEach((tool, index) => {
                if (tool.description === 'select_box_string') {
                    select_tool = tool;
                }
                if (tool.description === 'deselect_box_string') {
                    deselect_tool = tool;
                }
                if (tool.description === 'zigzagselect_box_string') {
                    zigzagselect_tool = tool;
                }
            });

            if (select_tool && deselect_tool && zigzagselect_tool) {
                if (select_tool.active) {
                    tools_info = 'select';
                } else if (deselect_tool.active) {
                    tools_info = 'deselect';
                } else if (zigzagselect_tool.active) {
                    tools_info = 'zigzag_select';
                } else {
                    tools_info = 'neither_active';
                }
            } else {
                tools_info = 'tools_not_found';
            }
        }

        tool_state.data = {active_tool: [tools_info]};
        """
        js_code = js_code.replace("zigzagselect_box_string", zigzagselect_box_string)
        js_code = js_code.replace("deselect_box_string", deselect_box_string)
        js_code = js_code.replace("select_box_string", select_box_string)

        JS_selection_monitor = CustomJS(args=dict(tool_state=self.tool_state_source), code=js_code)

        # Hack - monitor random GUI events to trigger javascript fetch of selection box type
        self.plot.js_on_event("selectiongeometry", JS_selection_monitor)  # Selection events
        self.plot.js_on_event("tap", JS_selection_monitor)  # Plot clicks
        self.plot.js_on_event(events.MouseMove, JS_selection_monitor)  # Mouse move
        self.tool_state_source.on_change("data", self.on_tool_state_change)

    def on_tool_state_change(self, attr, old, new):
        """Handle tool detection during selection"""
        if new.get("active_tool"):
            active_tool = new["active_tool"][0]
            print(f"Selection made with tool: {active_tool}")

            if active_tool == "select":
                self.select_mode = "select"
                print("→ SELECT box activated")
            elif active_tool == "deselect":
                self.select_mode = "deselect"
                print("→ DESELECT box activated")
            elif active_tool == "zigzag_select":
                self.select_mode = "zigzag_select"
                print("→ ZIGZAG-SELECT box activated")
            else:
                self.select_mode = "select"
                print(f"→ Unexpected result: {active_tool}, defaulting to SELECT box")

    def create_widgets(self):
        """Create Panel widgets"""
        # Probe type selector
        self.probe_type_selector = pn.Param(
            self,
            parameters=["probe_type"],
            widgets={"probe_type": {"type": pn.widgets.Select, "width": 300, "margin": 0}},
            show_name=False,
        )
        self.probe_subtype_selector = pn.Param(
            self,
            parameters=["probe_subtype"],
            widgets={"probe_subtype": {"type": pn.widgets.Select, "width": 140, "margin": 0}},
            show_name=False,
        )

        # Preset selector
        self.preset_selector = pn.Param(
            self,
            parameters=["preset"],
            widgets={"preset": {"type": pn.widgets.Select, "margin": 0}},
            show_name=False,
        )

        # Apply preset button
        self.apply_button = pn.widgets.Button(name="Apply Preset", button_type="primary", width=250)
        self.apply_button.on_click(lambda event: self.apply_preset())

        # Clear selection button (moved to top, orange styling)
        self.clear_button = pn.widgets.Button(
            name="Clear Selection",
            button_type="danger",
            width=120,
            margin=(0, 10, 10, 10),
            align="center",
        )
        self.clear_button.on_click(lambda event: self.clear_selection())

        # Shank selector for text input
        self.shank_selector_widget = pn.Param(
            self,
            parameters=["shank_selector"],
            widgets={"shank_selector": {"type": pn.widgets.Select, "margin": 0}},
            show_name=False,
        )

        # Electrode input
        self.electrode_input = pn.widgets.TextInput(
            name="Electrode Selection", placeholder="e.g., 1,2,3,5,7 or 1-5,7", width=300
        )

        # Apply electrode input button
        self.apply_input_button = pn.widgets.Button(
            name="Add Electrodes to Selection", button_type="primary", width=250
        )
        self.apply_input_button.on_click(lambda event: self.apply_electrode_input())

        # IMRO directory and filename inputs (split into two)
        self.filename_input = pn.widgets.TextInput(
            name="IMRO or PDF filename (omit extension)",
            value=f"manual_selection_{self.probe_type}",  # default filename defined here
            width=300,
            margin=(0, 0, 0, 30)
        )

        # Download buttons creation
        # (in their own function to handle their reactive appearance)
        self.create_download_buttons()

        # Prominent electrode counter (moved to top)
        self.electrode_counter = pn.pane.HTML(
            """
            <div style="background: #f0f8ff; border: 2px solid #4a90e2; border-radius: 8px;
                        padding: 12px; text-align: center; font-size: 16px; font-weight: bold;
                        color: #2c3e50;">
                Selected Electrodes: 0/384
            </div>
            """,
            width=300,
            margin=(30, 10, 10, 10),
            align="center",
        )

        # Bokeh pane with proper scrolling - larger viewport for better visibility
        self.plot_pane = pn.pane.Bokeh(
            self.plot,
            width=self.probe_plot_width,
            height=self.probe_plot_height,  # Larger viewport height for better electrode visibility
        )

        # Reference and gain inputs
        self.reference_selector_widget = pn.Param(
            self,
            parameters=["reference_id"],
            widgets={"reference_id": {"type": pn.widgets.Select, "width": 140, "margin": 0}},
            show_name=False,
        )

        self.ap_gain_input = pn.widgets.FloatInput(
            name="High-pass 'ap' gain", value=self.ap_gain_default, step=1e-1, start=0, end=5000, width=140
        )

        self.lf_gain_input = pn.widgets.FloatInput(
            name="Low-pass 'lf' gain", value=self.lf_gain_default, step=1e-1, start=0, end=5000, width=140
        )

        self.hardware_hp_filter_on_selector_widget = pn.Param(
            self,
            parameters=["hardware_hp_filter_on"],
            widgets={"hardware_hp_filter_on": {"type": pn.widgets.Select, "width": 300, "margin": 0}},
            show_name=False,
        )

        # Undo / redo widgets
        self.undo_button = pn.widgets.Button(
            name="↶ Undo",
            button_type="default",
            disabled=True,
            width=120,
            margin=(0, 5, 5, 10),
            css_classes=["pixelmap-undo"],
        )
        self.redo_button = pn.widgets.Button(
            name="↷ Redo",
            button_type="default",
            disabled=True,
            width=120,
            margin=(0, 10, 5, 5),
            css_classes=["pixelmap-redo"],
        )
        self.undo_button.on_click(lambda event: self.do_undo())
        self.redo_button.on_click(lambda event: self.do_redo())

        # Keyboard shortcuts (Cmd/Ctrl+Z, Cmd/Ctrl+Shift+Z) trigger the buttons
        # via a one-time document-level keydown listener. Wrapped in a guard so
        # repeated session loads do not stack listeners.
        self.undo_keyboard_pane = pn.pane.HTML(
            """
            <script>
            (function() {
              if (window._pixelmapUndoBound) return;
              window._pixelmapUndoBound = true;
              const clickFirst = (sel) => {
                const el = document.querySelector(sel + " button");
                if (el && !el.disabled) el.click();
              };
              document.addEventListener("keydown", (e) => {
                const tag = (e.target && e.target.tagName) || "";
                if (tag === "INPUT" || tag === "TEXTAREA") return;
                if (!(e.metaKey || e.ctrlKey)) return;
                if (!e.key || e.key.toLowerCase() !== "z") return;
                e.preventDefault();
                clickFirst(e.shiftKey ? ".pixelmap-redo" : ".pixelmap-undo");
              });
            })();
            </script>
            """,
            width=0,
            height=0,
            margin=0,
        )

        # Share-link widgets
        self.share_link_button = pn.widgets.Button(
            name="Get shareable link 🔗", button_type="primary", width=250
        )
        self.share_link_output = pn.widgets.TextInput(
            name="Share link (copy to clipboard)",
            value="",
            disabled=True,
            width=300,
        )
        self.share_link_button.on_click(lambda event: self.update_share_link())

        # Anatomy widgets
        atlas_options, atlas_default = self._available_atlases()
        self.atlas_name_input = pn.widgets.Select(
            name="Atlas",
            value=atlas_default,
            options=atlas_options,
            width=300,
        )
        self.tip_ap_input = pn.widgets.FloatInput(
            name="Tip AP (µm)", value=5000.0, step=10.0, width=95
        )
        self.tip_ml_input = pn.widgets.FloatInput(
            name="Tip ML (µm)", value=2500.0, step=10.0, width=95
        )
        self.tip_dv_input = pn.widgets.FloatInput(
            name="Tip DV (µm)", value=3000.0, step=10.0, width=95
        )
        self.pitch_input = pn.widgets.FloatInput(
            name="Pitch (AP tilt, deg)", value=0.0, step=1.0,
            start=-90.0, end=90.0, width=140,
        )
        self.yaw_input = pn.widgets.FloatInput(
            name="Yaw (ML tilt, deg)", value=0.0, step=1.0,
            start=-90.0, end=90.0, width=140,
        )
        self.shank_orientation_input = pn.widgets.FloatInput(
            name="Shank orientation (deg)", value=0.0, step=1.0,
            start=-180.0, end=360.0, width=200,
        )
        self.anatomy_coord_note = pn.pane.HTML(
            self._anatomy_coord_note_html(), width=320
        )
        self.atlas_name_input.param.watch(self._update_anatomy_origin_note, "value")
        self.anatomy_help = pn.pane.HTML(
            (
                '<div style="font-size: 11px; color: #555; padding: 2px 0 6px 0;">'
                "<b>Pitch</b> = AP tilt (probe leaning forward / backward). "
                "<b>Yaw</b> = ML tilt (probe leaning left / right). "
                "<b>Shank orientation</b> = direction of shanks 0→3 in the "
                "horizontal plane; 0° = lateral (+ML), 90° = anterior (+AP)."
                "</div>"
            ),
            width=320,
        )
        self.compute_anatomy_button = pn.widgets.Button(
            name="Compute anatomical overlay 🧠", button_type="primary", width=250
        )
        self.clear_anatomy_button = pn.widgets.Button(
            name="Clear overlay", button_type="default", width=120
        )
        self.compute_anatomy_button.on_click(lambda event: self.compute_anatomy_overlay())
        self.clear_anatomy_button.on_click(lambda event: self.clear_anatomy_overlay())
        self.anatomy_legend = pn.pane.HTML(self._empty_legend_html(), width=320)
        self.anatomy_locator = pn.pane.Matplotlib(
            object=None, width=360, height=215, tight=True, format="png",
        )

        if not anatomy_atlas.is_available():
            install_hint = (
                "<i>Optional: install with </i><code>pip install 'pixelmap[anatomy]'</code>"
                "<i> to enable atlas lookups.</i>"
            )
            self.anatomy_install_hint = pn.pane.HTML(
                f'<div style="font-size: 12px; color: #b00; padding: 4px 0;">{install_hint}</div>',
                width=300,
            )
        else:
            self.anatomy_install_hint = pn.Spacer(height=0)

        # IMRO file dropper
        self.imro_file_loader = pn.widgets.FileInput(width=300)
        self.apply_uploaded_imro_button = pn.widgets.Button(
            name="Apply uploaded IMRO file to selection ⬆", button_type="primary", width=250
        )
        self.apply_uploaded_imro_button.on_click(lambda event: self.apply_uploaded_imro())


    def create_download_buttons(self):
        """Create download buttons as a reactive pane"""
        # This method recreates the buttons when color changes
        self.download_imro_button = pn.widgets.FileDownload(
            callback=self.generate_imro_content,
            filename=f"{self.filename_input.value}.imro",
            button_type=self.download_button_color,
            width=87,
            icon="file-text",
            label=self.download_button_label,
        )

        self.download_pdf_button = pn.widgets.FileDownload(
            callback=self.generate_pdf_content,
            filename=f"{self.filename_input.value}.pdf",
            button_type=self.download_button_color,
            width=87,
            icon="file-type-pdf",
            label=self.download_button_label.replace("IMRO", "PDF"),
        )

        self.download_kilosort_button = pn.widgets.FileDownload(
            callback=self.generate_kilosort_channelmap,
            filename=f"{self.filename_input.value}.json",
            button_type=self.download_button_color,
            width=87,
            icon="file-code",
            label=self.download_button_label.replace("IMRO", "Kilosort"),
        )

        return pn.Row(
            self.download_imro_button,
            self.download_pdf_button,
            self.download_kilosort_button,
            sizing_mode="stretch_width",
            margin=(0, 0, 0, 20),
        )

    @pn.depends('download_button_color', 'download_button_label')
    def get_download_buttons(self):
        """Reactive method that recreates buttons when color changes"""
        return self.create_download_buttons()

    def create_layout(self):
        """Create the main Panel layout"""

        # Counter and Downloader (fixed on the right)
        counter_downloader = pn.Column(
            self.undo_keyboard_pane,
            pn.Column(
                self.electrode_counter,
                pn.Row(self.undo_button, self.redo_button, margin=(0, 0, 0, 10)),
                self.clear_button,
                margin=(0, 0, -10, 20),
            ),
            pn.pane.Markdown("## Export IMRO table", margin=(10, 0, -5, 30)),
            self.filename_input,
            self.get_download_buttons,

            pn.pane.Markdown("## Share this layout", margin=(15, 0, -5, 30)),
            pn.Column(
                self.share_link_button,
                self.share_link_output,
                margin=(0, 0, 0, 20),
            ),

            pn.pane.Markdown("## Probe location", margin=(15, 0, -5, 30)),
            self.anatomy_locator,

            pn.pane.Markdown("## Region legend", margin=(15, 0, -5, 30)),
            self.anatomy_legend,

            pn.pane.Markdown("## PixelMap instructions", margin=(10, 0, -5, 10)),
            pn.pane.HTML("""
            <div style="font-size: 13px; line-height: 1.4; text-align: justify;">
            <b>Neuropixels hardware Constraints:</b><br>
            Neuropixels electrodes are <a href='https://www.neuropixels.org/support' target='_blank'>hardwired</a> to specific ADCs in the probe's head. When you select an electrode, others become unavailable because they share the same recording lines.
            This GUI allows you to build a channelmap around those constraints: when you select channels, they turn <font color="#c00000"><b>red</b></font>, and those that become unavailable because they share the same lines turn <b>black</b>.<br><br>

            <b>You can mix and match four selection methods:</b><br>
            • <b>Presets:</b> Pre-configured channelmaps that respect wiring constraints<br>
            • <b>Textual selection:</b> Type electrode ranges (e.g., "1-10,20-25") to add to the current selection<br>
            • <b>Interactive:</b> Click electrodes directly or drag boxes (selection, deselection, or "zigzag selection") to maually select multiple sites<br>
            • <b>Selection from pre-existing IMRO file</b>: you can pre-load an IMRO file as a starting point before doing any of the above.<br><br>

            Once you reach the <b>target number of electrodes</b> for the selected probe type (384 or 1536), you can <b>download your channelmap</b> as an IMRO file alongside a PDF rendering to easily remember what your channelmap looks like.
            </div>
            """),

            styles={
                "position": "sticky",
                "top": "0px",
                "height": "100vh",
                "overflow-y": "auto",
                "background": "white",
                "border-left": "1px solid #ddd",
            },
            width=350,
            scroll=False,
        )

        # Controls panel (fixed on left)
        controls = pn.Column(
            pn.pane.Markdown(
                (
                    f"<div style='text-align: center; padding: 12px;'><strong>See project (v{__version__}) at:"
                    "<br><a href='https://github.com/m-beau/pixelmap' "
                    "target='_blank'>github.com/m-beau/pixelmap</a></strong>"
                    "<br><span style='font-size: 1.0em;'>We're working towards publishing PixelMap in JOSS, and need to demonstrate impact. "
                    "If you find PixelMap helpful and want to see it improved in the future, <br><b style='color: red;'>you can help us by filling this 1-minute form: "
                    "<a href='https://docs.google.com/forms/d/e/1FAIpQLSeZyha1XEcKHRqY7oW5lNL5ER1YW2XIDagtCfRECCk5PKhBsw/viewform?usp=publish-editor'"
                    "target='_blank'>link</a></b><br> (and adding a star ⭐ to the <a href='https://github.com/m-beau/pixelmap' "
                    "target='_blank'>repo</a> 😊)</span></div>"
                ),
                margin=(0, 0, 0, 40),
            ),
            pn.Column(
                pn.pane.Markdown("## Probe and recording metadata", margin=(-5, 0, 0, 10)),
                pn.pane.Markdown(
                    "(see <a href='https://billkarsh.github.io/SpikeGLX/help/imroTables' target='_blank'>IMRO table anatomy</a>)",
                    margin=(-15, 0, -5, 10),
                ),
                self.probe_type_selector,
                pn.Row(
                    self.probe_subtype_selector,
                    self.reference_selector_widget,
                    sizing_mode="stretch_width",
                ),
                pn.pane.Markdown("<b>For 1.0 only (2.0 gains not in channelmap):</b>", margin=(0, 0, -15, 10)),
                self.hardware_hp_filter_on_selector_widget,
                pn.Row(
                    self.ap_gain_input,
                    self.lf_gain_input,
                    sizing_mode="stretch_width",
                ),
                styles={"background": "#e6e6e6", "padding": "10px", "border-radius": "5px"},
                margin=(0, 5, 0, 5),
            ),
            pn.pane.Markdown("## Preset Selection", margin=(10, 0, -5, 10)),
            self.preset_selector,
            self.apply_button,
            pn.pane.Markdown("## Textual Selection", margin=(10, 0, -5, 10)),
            self.shank_selector_widget,
            self.electrode_input,
            self.apply_input_button,
            pn.pane.Markdown("## Selection from IMRO file", margin=(10, 0, -5, 10)),
            self.imro_file_loader,
            # pn.Spacer(height=30),
            self.apply_uploaded_imro_button,
            pn.pane.Markdown("## Anatomical overlay", margin=(15, 0, -5, 10)),
            self.anatomy_install_hint,
            self.atlas_name_input,
            self.anatomy_coord_note,
            pn.Row(self.tip_ap_input, self.tip_ml_input, self.tip_dv_input, sizing_mode="stretch_width"),
            pn.Row(self.pitch_input, self.yaw_input, sizing_mode="stretch_width"),
            self.shank_orientation_input,
            self.anatomy_help,
            self.compute_anatomy_button,
            self.clear_anatomy_button,
            styles={
                "position": "fixed",
                "top": "0px",
                "left": "0px",
                "height": "100vh",
                "overflow-y": "auto",  # Allow scrolling within controls if needed
                "z-index": "1000",
                "background": "white",
                "border-right": "1px solid #ddd",
            },
            width=370,
            scroll=False,
        )

        # Main layout with properly scrollable plot container
        plot_container = pn.Column(
            self.plot_pane,
            width=self.probe_plot_width,
            height=self.probe_plot_height,  # Match the plot pane height
            scroll=True,  # Enable scrolling for the plot container
        )

        layout = pn.Row(
            controls,
            pn.Spacer(width=370),  # Spacer for fixed controls panel on left
            plot_container,
            counter_downloader,
            sizing_mode="fixed",
        )

        return layout

    ################################
    ##### Probe version update #####
    ################################

    def clear_bokeh_data(self):

        # Clean up old plot resources before creating new ones
        if hasattr(self, 'plot') and self.plot is not None:
            # Clear all renderers and tools to prevent memory accumulation
            self.plot.renderers.clear()
            self.plot.toolbar.tools.clear()

        if hasattr(self, 'electrode_source') and self.electrode_source is not None:
            # Clear data source
            self.electrode_source.data.clear()

        if hasattr(self, 'tool_state_source') and self.tool_state_source is not None:
            # Clear tool state source
            self.tool_state_source.data.clear()

    @param.depends("probe_type", watch=True)
    def on_probe_type_change(self):
        """
        Handle probe type changes, which require to reset the whole plot.
        Param module monitors value changes of probe_type.
        """
        self.clear_bokeh_data()
        self.load_probe_data()
        self.setup_bokeh_plot()
        self.update_electrode_counter()
        self.update_filename(reset=True)

        # Update the plot pane
        # ("object" attr of plot_pane is the bokeh plot self.plot)
        if hasattr(self, "plot_pane"):
            self.plot_pane.object = self.plot

    @param.depends("probe_subtype", watch=True)
    def on_probe_subtype_change(self):
        """
        Handle probe subtype changes - update reference_id objects based on REF_ELECTRODES mapping.
        Param module monitors value changes of probe_subtype.
        """
        # Update reference_id parameter objects based on the new probe_subtype
        self.param.reference_id.objects = list(REF_ELECTRODES[self.probe_subtype].keys())

        # Set reference_id to the first available option if current selection is not available
        if self.reference_id not in self.param.reference_id.objects:
            self.reference_id = self.param.reference_id.objects[0]

    def update_electrode_counter(self):
        """Update status information"""
        n_selected = len(self.electrodes.selected)
        # n_unavailable = len(self.electrodes.unavailable)
        max_allowed = self.electrodes.n_maximum_electrodes
        # n_remaining = max_allowed - n_selected

        # Update electrode counter
        if n_selected < max_allowed:
            counter_html = f"""
            <div style="background: #f0f8ff; border: 2px solid #4a90e2; border-radius: 8px;
                        padding: 12px; text-align: center; font-size: 16px; font-weight: bold;
                        color: #AA4A44;">
                Selected Electrodes: {n_selected}/{max_allowed}
            </div>
            """
            self.download_button_color = 'default'
            self.download_button_label = "Select..."
        else:
            counter_html = f"""
            <div style="background: #f0f8ff; border: 2px solid #4a90e2; border-radius: 8px;
                        padding: 12px; text-align: center; font-size: 16px; font-weight: bold;
                        color: #008000;">
                Selected Electrodes: {n_selected}/{max_allowed}<br>
                Ready for IMRO file generation!
            </div>
            """
            self.download_button_color = 'success'
            self.download_button_label = "IMRO ⬇"

        self.electrode_counter.object = counter_html


## App creation utilities
def _extract_cfg_from_session_args() -> str | None:
    """Pull the ``cfg`` query param out of pn.state.session_args, if any."""
    try:
        args = pn.state.session_args or {}
    except Exception:
        return None
    raw = args.get(url_share.QUERY_PARAM)
    if not raw:
        return None
    # session_args values come back as lists of bytes; take the first entry.
    value = raw[0] if isinstance(raw, list) else raw
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value or None


def create_app():
    """Create and configure the Panel app"""
    gui = ChannelmapGUI()

    cfg = _extract_cfg_from_session_args()
    if cfg:
        try:
            applied = gui.apply_url_state(cfg)
            if not applied:
                print("Ignored invalid or unsupported ?cfg= share link.")
        except Exception as e:
            print(f"Failed to apply shared layout from URL: {e}")

    analytics_tracker = AnalyticsSessionTracker()
    layout = gui.create_layout()

    # Update status initially
    gui.update_electrode_counter()

    if pn.state.curdoc:
        pn.state.curdoc.on_session_destroyed(analytics_tracker.handle_session_destroyed)

    app_layout = pn.Column(analytics_tracker.cookie_bridge, layout, margin=0)
    app_layout._analytics_tracker = analytics_tracker
    return app_layout
