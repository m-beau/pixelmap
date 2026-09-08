"""Tests for pixelmap.utils.survey — survey .txt parsing and probe-match."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pixelmap.constants import WIRING_FILE_MAP
from pixelmap.types import Electrode
from pixelmap.utils import survey

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def positions_df_1_0(wiring_maps_dir):
    return pd.read_csv(wiring_maps_dir / WIRING_FILE_MAP["1.0"][0])


@pytest.fixture
def positions_df_2_0_1shank(wiring_maps_dir):
    return pd.read_csv(wiring_maps_dir / WIRING_FILE_MAP["2.0-1shank"][0])


@pytest.fixture
def positions_df_2_0_4shanks(wiring_maps_dir):
    return pd.read_csv(wiring_maps_dir / WIRING_FILE_MAP["2.0-4shanks"][0])


def _survey_content_from_positions(positions_df: pd.DataFrame, val_of) -> str:
    lines = ["Shank\tXum\tZum\tVal"]
    for shank, _elec, x, y in positions_df[["shank", "electrode", "x", "y"]].itertuples(index=False):
        lines.append(f"{int(shank)}\t{int(x)}\t{int(y)}\t{val_of(shank, x, y)}")
    return "\n".join(lines) + "\n"


def test_parse_survey_file_happy_path():
    content = "Shank\tXum\tZum\tVal\n0\t27\t0\t1.5\n0\t59\t0\t2.0\n"
    df = survey.parse_survey_file(content)
    assert list(df.columns) == ["shank", "xum", "zum", "val"]
    assert len(df) == 2
    assert df.iloc[1]["val"] == pytest.approx(2.0)


def test_parse_survey_file_4shank_export():
    df = survey.parse_survey_file((FIXTURES / "survey_2.0-4shanks.txt").read_text())
    assert set(df.columns) == {"shank", "xum", "zum", "val"}
    assert df["shank"].min() >= 0
    assert df["shank"].max() <= 3
    assert set(df["shank"].unique()) == {0, 1, 2, 3}


def test_parse_survey_file_1_0_export():
    df = survey.parse_survey_file((FIXTURES / "survey_1.0.txt").read_text())
    assert set(df.columns) == {"shank", "xum", "zum", "val"}
    assert set(df["shank"].unique()) == {0}


def test_parse_survey_file_2_0_1shank_export():
    df = survey.parse_survey_file((FIXTURES / "survey_2.0-1shank.txt").read_text())
    assert set(df.columns) == {"shank", "xum", "zum", "val"}
    assert set(df["shank"].unique()) == {0}


def test_validate_probe_match_1_0_export(positions_df_1_0):
    df = survey.parse_survey_file((FIXTURES / "survey_1.0.txt").read_text())
    values, n_unmatched = survey.validate_probe_match(df, positions_df_1_0)
    assert n_unmatched == 0
    assert len(values) == len(positions_df_1_0)


def test_validate_probe_match_2_0_1shank_export(positions_df_2_0_1shank):
    df = survey.parse_survey_file((FIXTURES / "survey_2.0-1shank.txt").read_text())
    values, n_unmatched = survey.validate_probe_match(df, positions_df_2_0_1shank)
    assert n_unmatched == 0
    assert len(values) == len(positions_df_2_0_1shank)


def test_parse_survey_file_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        survey.parse_survey_file("foo\tbar\n1\t2\n")


def test_validate_probe_match_success(positions_df_1_0):
    content = _survey_content_from_positions(positions_df_1_0, lambda s, x, y: float(y))
    df = survey.parse_survey_file(content)
    values, n_unmatched = survey.validate_probe_match(df, positions_df_1_0)
    assert n_unmatched == 0
    assert len(values) == len(positions_df_1_0)
    # Spot-check: Val encodes y, so Electrode(0, 0) (at y=0) should have val 0
    assert values[Electrode(0, 0)] == pytest.approx(0.0)


def test_validate_probe_match_rejects_wrong_probe(positions_df_2_0_4shanks, positions_df_1_0):
    # Build survey from 4-shank positions, but validate against 1.0 positions
    content = _survey_content_from_positions(positions_df_2_0_4shanks, lambda s, x, y: 1.0)
    df = survey.parse_survey_file(content)
    with pytest.raises(ValueError, match="different probe type|No survey rows matched"):
        survey.validate_probe_match(df, positions_df_1_0)


def test_match_handles_constant_offset(positions_df_2_0_4shanks):
    """SpikeGLX ships shank-local coords; positions CSV uses global x.
    Survey with an arbitrary per-shank offset should still match within tolerance."""
    rows = []
    for shank, _elec, x, y in positions_df_2_0_4shanks[["shank", "electrode", "x", "y"]].itertuples(index=False):
        # Simulate SpikeGLX: subtract shank's min-x (shank-local) and add 27µm offset
        shank_min_x = positions_df_2_0_4shanks[positions_df_2_0_4shanks["shank"] == shank]["x"].min()
        local_x = int(x) - int(shank_min_x) + 27
        rows.append(f"{int(shank)}\t{local_x}\t{int(y)}\t1.0")
    content = "Shank\tXum\tZum\tVal\n" + "\n".join(rows) + "\n"
    df = survey.parse_survey_file(content)
    values, n_unmatched = survey.validate_probe_match(df, positions_df_2_0_4shanks)
    assert n_unmatched == 0
    assert len(values) == len(positions_df_2_0_4shanks)


class TestDefaultSurveyRange:
    """Initial colormap bounds: robust enough that the overlay reads on sight."""

    def test_tails_are_clipped_off_a_heavy_tailed_survey(self):
        rng = np.random.default_rng(0)
        vals = list(rng.gamma(2.0, 1.0, size=500)) + [500.0, 900.0]
        vmin, vmax = survey.default_survey_range(vals)
        assert vmin > min(vals)
        assert vmax < max(vals) / 10  # the two outliers no longer set the scale

    def test_bulk_of_the_data_stays_inside_the_range(self):
        rng = np.random.default_rng(1)
        vals = rng.gamma(2.0, 1.0, size=1000)
        vmin, vmax = survey.default_survey_range(vals)
        inside = ((vals >= vmin) & (vals <= vmax)).mean()
        assert inside > 0.9

    def test_percentiles_are_configurable(self):
        vals = list(range(101))
        assert survey.default_survey_range(vals, percentiles=(0.0, 100.0)) == (0.0, 100.0)
        assert survey.default_survey_range(vals, percentiles=(10.0, 90.0)) == (10.0, 90.0)

    def test_constant_survey_gets_a_non_degenerate_range(self):
        vmin, vmax = survey.default_survey_range([4.2] * 50)
        assert vmax > vmin

    def test_nearly_constant_survey_falls_back_to_the_full_range(self):
        # Percentiles collapse (99% of values identical) but min/max still differ.
        vals = [1.0] * 500 + [9.0]
        vmin, vmax = survey.default_survey_range(vals)
        assert (vmin, vmax) == (1.0, 9.0)

    def test_non_finite_values_are_ignored(self):
        vmin, vmax = survey.default_survey_range(
            [float("nan"), float("inf"), -float("inf")] + list(range(101))
        )
        assert np.isfinite(vmin) and np.isfinite(vmax)
        assert 0.0 <= vmin < vmax <= 100.0

    def test_empty_survey_gets_a_safe_default(self):
        assert survey.default_survey_range([]) == (0.0, 1.0)
        assert survey.default_survey_range([float("nan")]) == (0.0, 1.0)
