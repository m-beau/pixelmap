"""The gain / hardware-filter inputs show only for probes whose IMRO table stores them.

That is decided by the part number's IMRO format, not by the probe group:
NHP probes run on the 1.0 ASIC and write 1.0-format tables, so they need the
inputs; 2.0 and later have fixed gains and no field for them.
"""

from __future__ import annotations

import pytest

from pixelmap.constants import (
    IMRO_FORMATS_WITH_GAINS,
    PROBE_FEATURES,
    PROBE_TYPE_MAP,
    WIRING_FILE_MAP,
)
from pixelmap.gui.gui import ChannelmapGUI
from pixelmap.utils.imro import generate_imro_channelmap

SELECTABLE = [(group, part) for group in WIRING_FILE_MAP for part in PROBE_TYPE_MAP[group]]


@pytest.fixture(scope="module")
def gui():
    gui = ChannelmapGUI()
    gui.create_layout()
    return gui


def _select(gui, group, part):
    gui.probe_type = group
    gui.probe_subtype = part


def test_hidden_on_the_default_2_0_probe():
    fresh = ChannelmapGUI()
    assert fresh.probe_type == "2.0-4shanks"
    assert fresh.gain_settings_section.visible is False


@pytest.mark.parametrize("group", ["2.0-1shank", "2.0-4shanks", "QuadBase"])
def test_hidden_for_every_2_0_part_number(gui, group):
    for part in PROBE_TYPE_MAP[group]:
        _select(gui, group, part)
        assert gui.gain_settings_section.visible is False, part


def test_shown_for_every_part_number_of_the_1_0_group(gui):
    for part in PROBE_TYPE_MAP["1.0"]:
        _select(gui, "1.0", part)
        assert gui.gain_settings_section.visible is True, part


@pytest.mark.parametrize("part", ["NP1010", "NP1011", "NP1012", "NP1013", "NP1014"])
def test_shown_for_primate_probes_that_write_1_0_format_tables(gui, part):
    assert "NHP" in PROBE_FEATURES[part]["probe_name"]
    _select(gui, "1.0", part)
    assert gui.gain_settings_section.visible is True


def test_follows_the_probe_back_and_forth(gui):
    _select(gui, "1.0", "NP1000")
    assert gui.gain_settings_section.visible is True
    gui.probe_type = "2.0-4shanks"
    assert gui.gain_settings_section.visible is False
    gui.probe_type = "1.0"
    assert gui.gain_settings_section.visible is True


def test_the_section_holds_the_three_inputs(gui):
    section = gui.gain_settings_section
    assert gui.ap_gain_input in section.select()
    assert gui.lf_gain_input in section.select()
    assert gui.hardware_hp_filter_on_selector_widget in section.select()


@pytest.mark.parametrize(("group", "part"), SELECTABLE)
def test_visible_exactly_when_the_generated_table_stores_the_gains(gui, group, part):
    """Entries carry (ap_gain, lf_gain, hp_filter) if and only if the inputs are shown."""
    _select(gui, group, part)
    imro_list = generate_imro_channelmap(
        probe_type=group, probe_subtype=part, wiring_file=gui.wiring_file,
        layout_preset=gui.param.preset.objects[0], reference_id=gui.reference_id,
        ap_gain=1234, lf_gain=567, hp_filter=0,
    )
    stores_gains = any(tuple(entry[-3:]) == (1234, 567, 0) for entry in imro_list[1:])

    assert stores_gains == (PROBE_FEATURES[part]["IMRO_format"] in IMRO_FORMATS_WITH_GAINS)
    assert gui.gain_settings_section.visible is stores_gains
