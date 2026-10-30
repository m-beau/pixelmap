"""Equivalence tests for the wiring-table lookups behind IMRO generation.

``find_forbidden_electrodes`` and ``find_electrode_coordinates`` used to scan
the whole wiring table once per selected electrode -- O(selected x table),
~0.5 s per IMRO for a QuadBase probe, on the server's event loop.  They now
build one ``(shank, electrode) -> (row, col)`` index per call.  The reference
implementations below are verbatim copies of the originals and pin the exact
output, including order, dtype and shape.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pixelmap import backend
from pixelmap.constants import (
    WIRING_FILE_MAP,
    SUPPORTED_1shank_PRESETS,
    SUPPORTED_4shanks_PRESETS,
    SUPPORTED_QuadBase_PRESETS,
)

WIRING_DIR = Path(__file__).parent.parent / "pixelmap" / "wiring_maps"
PROBE_TYPES = ["1.0", "2.0-1shank", "2.0-4shanks", "QuadBase"]


# --------------------------------------------------------------------------
# Reference (pre-optimisation) implementations
# --------------------------------------------------------------------------

def _reference_forbidden(selected_electrodes, wiring_df):
    df = backend.format_wiring_df(wiring_df)
    table_ids = df.iloc[:, 1:].values
    table_shape = table_ids.shape
    flat_ids = table_ids.ravel()
    flat_shank_ids = np.array([id[0] for id in flat_ids])
    flat_electrode_ids = np.array([id[1] for id in flat_ids])

    forbidden_electrodes = []
    for shank_id, electrode_id in selected_electrodes:
        table_bool = ((flat_shank_ids == shank_id) & (flat_electrode_ids == electrode_id)).reshape(table_shape)
        assert np.any(table_bool), f"electrode {electrode_id} on shank {shank_id} not found in wiring map!"
        table_loc = np.nonzero(table_bool)
        table_row = table_loc[0][0]
        forbidden_electrodes += [
            se
            for se in table_ids[table_row]
            if (~np.any(np.isnan(se))) & ~((shank_id == se[0]) & (electrode_id == se[1]))
        ]
    return np.array(forbidden_electrodes).astype(int)


def _reference_coordinates(electrodes, wiring_df):
    df = backend.format_wiring_df(wiring_df)
    table_ids = df.iloc[:, 1:].values
    table_shape = table_ids.shape
    flat_ids = table_ids.ravel()
    flat_shank_ids = np.array([id[0] for id in flat_ids])
    flat_electrode_ids = np.array([id[1] for id in flat_ids])

    coordinates = []
    for shank_id, electrode_id in electrodes:
        table_bool = ((flat_shank_ids == shank_id) & (flat_electrode_ids == electrode_id)).reshape(table_shape)
        row, col = np.nonzero(table_bool)
        coordinates.append([int(row[0]), int(col[0]) + 1])
    return coordinates


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module", params=PROBE_TYPES)
def probe(request):
    probe_type = request.param
    wiring_df = pd.read_csv(WIRING_DIR / WIRING_FILE_MAP[probe_type][1])
    return probe_type, wiring_df


def _all_electrodes(wiring_df):
    """Every (shank, electrode) present in the table, in table order."""
    out = []
    for col in wiring_df.columns[1:]:
        shank = int(col[5])
        for e in wiring_df[col].dropna():
            out.append((shank, int(e)))
    return out


def _selections(probe_type, wiring_df):
    """Preset selections plus random and edge-case ones."""
    rng = np.random.default_rng(0)
    every = _all_electrodes(wiring_df)
    presets = {
        "1.0": SUPPORTED_1shank_PRESETS,
        "2.0-1shank": SUPPORTED_1shank_PRESETS,
        "2.0-4shanks": SUPPORTED_4shanks_PRESETS,
        "QuadBase": SUPPORTED_QuadBase_PRESETS,
    }[probe_type]
    sels = {}
    for preset in presets:
        sels[f"preset:{preset}"] = np.array(backend.get_preset_candidates(preset, probe_type, wiring_df))
    for i in range(3):
        idx = rng.choice(len(every), size=min(200, len(every)), replace=False)
        sels[f"random{i}"] = np.array([every[j] for j in idx])
    sels["first"] = np.array([every[0]])
    sels["last"] = np.array([every[-1]])
    sels["all"] = np.array(every)
    return sels


# --------------------------------------------------------------------------
# Equivalence
# --------------------------------------------------------------------------

def test_forbidden_matches_reference(probe):
    probe_type, wiring_df = probe
    for name, sel in _selections(probe_type, wiring_df).items():
        got = backend.find_forbidden_electrodes(sel, wiring_df)
        want = _reference_forbidden(sel, wiring_df)
        assert got.dtype == want.dtype, name
        assert got.shape == want.shape, name
        np.testing.assert_array_equal(got, want, err_msg=name)


def test_coordinates_match_reference(probe):
    probe_type, wiring_df = probe
    for name, sel in _selections(probe_type, wiring_df).items():
        got = backend.find_electrode_coordinates(sel, wiring_df)
        want = _reference_coordinates(sel, wiring_df)
        assert got == want, name
        assert all(type(v) is int for pair in got for v in pair), name


def test_forbidden_accepts_list_of_tuples(probe):
    """make_wiring_maps passes ``[(shank, electrode)]``, not an array."""
    _, wiring_df = probe
    shank, electrode = _all_electrodes(wiring_df)[5]
    got = backend.find_forbidden_electrodes([(shank, electrode)], wiring_df)
    np.testing.assert_array_equal(got, _reference_forbidden([(shank, electrode)], wiring_df))


def test_float_ids_are_accepted(probe):
    """IDs may arrive as floats (e.g. from a NaN-holding table)."""
    _, wiring_df = probe
    sel = np.array(_all_electrodes(wiring_df)[:10], dtype=float)
    np.testing.assert_array_equal(
        backend.find_forbidden_electrodes(sel, wiring_df), _reference_forbidden(sel, wiring_df)
    )
    assert backend.find_electrode_coordinates(sel, wiring_df) == _reference_coordinates(sel, wiring_df)


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------

def test_empty_selection(probe):
    _, wiring_df = probe
    empty = np.zeros((0, 2), dtype=int)
    got = backend.find_forbidden_electrodes(empty, wiring_df)
    want = _reference_forbidden(empty, wiring_df)
    # The original returns a 1-D empty array; callers depend on that shape.
    assert got.shape == want.shape == (0,)
    assert backend.find_electrode_coordinates(empty, wiring_df) == []


def test_unknown_electrode_raises(probe):
    _, wiring_df = probe
    with pytest.raises(AssertionError, match="electrode 99999 on shank 0 not found in wiring map"):
        backend.find_forbidden_electrodes([(0, 99999)], wiring_df)


def test_unknown_shank_raises(probe):
    _, wiring_df = probe
    with pytest.raises(AssertionError, match="on shank 7 not found in wiring map"):
        backend.find_forbidden_electrodes([(7, 0)], wiring_df)


def test_unknown_electrode_coordinates_raises(probe):
    """The original raised IndexError (``row[0]`` of an empty match)."""
    _, wiring_df = probe
    with pytest.raises((IndexError, KeyError, AssertionError)):
        backend.find_electrode_coordinates([(0, 99999)], wiring_df)


def test_duplicate_selection_repeats_output(probe):
    _, wiring_df = probe
    e = _all_electrodes(wiring_df)[3]
    sel = np.array([e, e])
    np.testing.assert_array_equal(
        backend.find_forbidden_electrodes(sel, wiring_df), _reference_forbidden(sel, wiring_df)
    )
    assert backend.find_electrode_coordinates(sel, wiring_df) == _reference_coordinates(sel, wiring_df)


def test_wiring_df_not_mutated(probe):
    _, wiring_df = probe
    before = wiring_df.copy()
    backend.find_forbidden_electrodes(np.array(_all_electrodes(wiring_df)[:20]), wiring_df)
    backend.find_electrode_coordinates(np.array(_all_electrodes(wiring_df)[:20]), wiring_df)
    pd.testing.assert_frame_equal(wiring_df, before)


# --------------------------------------------------------------------------
# Speed (relative, so it holds on any machine)
# --------------------------------------------------------------------------

def _best_of(fn, n=3):
    import time

    best = float("inf")
    for _ in range(n):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def test_lookups_much_faster_than_per_electrode_scan():
    """QuadBase, 1536 selected: the per-electrode scan took ~0.45 s here."""
    wiring_df = pd.read_csv(WIRING_DIR / WIRING_FILE_MAP["QuadBase"][1])
    sel = np.array(_all_electrodes(wiring_df)[:1536])

    old = _best_of(lambda: (_reference_forbidden(sel, wiring_df), _reference_coordinates(sel, wiring_df)))
    new = _best_of(lambda: (
        backend.find_forbidden_electrodes(sel, wiring_df),
        backend.find_electrode_coordinates(sel, wiring_df),
    ))
    assert new * 5 < old, f"new {new * 1000:.0f} ms vs reference {old * 1000:.0f} ms"
