################################
## SpikeGLX survey file I/O   ##
################################

from io import StringIO

import numpy as np
import pandas as pd

from pixelmap.types import Electrode


SURVEY_COLUMNS = ("shank", "xum", "zum", "val")

# Tolerance for matching a survey (Xum, Zum) to a probe electrode.
# SpikeGLX uses shank-local coordinates that can differ from our
# positions CSVs by a constant per-probe offset; the tolerance absorbs
# that while still rejecting surveys from a genuinely different probe.
DEFAULT_MATCH_TOLERANCE_UM = 35.0


def parse_survey_file(content: str) -> pd.DataFrame:
    """
    Parse a SpikeGLX-exported survey .txt file.

    The file is tab-separated with a header line: Shank, Xum, Zum, Val.
    Each subsequent row corresponds to one probe contact and its measured
    value (spike rate, amplitude, LFP power, etc).

    Returns a DataFrame with lowercase columns (shank, xum, zum, val).
    Raises ValueError if the header or dtypes are unexpected.
    """
    try:
        df = pd.read_csv(StringIO(content), sep="\t")
    except Exception as e:
        raise ValueError(f"Could not parse survey file as TSV: {e}") from e

    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in SURVEY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Survey file is missing required columns: {missing}. "
            f"Expected header: Shank\\tXum\\tZum\\tVal."
        )

    df = df[list(SURVEY_COLUMNS)]
    df = df.assign(
        shank=pd.to_numeric(df["shank"], errors="raise"),
        xum=pd.to_numeric(df["xum"], errors="raise"),
        zum=pd.to_numeric(df["zum"], errors="raise"),
        val=pd.to_numeric(df["val"], errors="raise"),
    ).astype({"shank": int, "xum": int, "zum": int, "val": float})
    return df


def _shank_local_positions(positions_df: pd.DataFrame) -> dict[int, np.ndarray]:
    """
    Return {shank_id: array of (electrode_id, local_x, y)} per shank.

    Positions CSVs for multi-shank probes store x in *global* coordinates
    (offset by shank index × shank spacing). SpikeGLX Xum is exported in
    *shank-local* coordinates. We normalise by subtracting the per-shank
    minimum x so lookups work regardless of convention.
    """
    out: dict[int, np.ndarray] = {}
    for shank_id, group in positions_df.groupby("shank"):
        x = group["x"].to_numpy(dtype=float)
        y = group["y"].to_numpy(dtype=float)
        local_x = x - x.min()
        elec = group["electrode"].to_numpy(dtype=int)
        out[int(shank_id)] = np.column_stack([elec, local_x, y])
    return out


def match_survey_to_electrodes(
    survey_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    tolerance_um: float = DEFAULT_MATCH_TOLERANCE_UM,
) -> tuple[dict[Electrode, float], int]:
    """
    Map each survey row to its nearest electrode on the same shank.

    A row is accepted if the nearest electrode is within ``tolerance_um``
    microns (Euclidean, after shank-local normalisation). Any row whose
    shank is absent from ``positions_df`` or whose nearest electrode is
    beyond the tolerance is counted as unmatched.

    Returns (values_by_electrode, n_unmatched).
    """
    shank_tables = _shank_local_positions(positions_df)
    survey_shanks = survey_df["shank"].to_numpy(dtype=int)
    survey_x = survey_df["xum"].to_numpy(dtype=float)
    survey_z = survey_df["zum"].to_numpy(dtype=float)
    survey_v = survey_df["val"].to_numpy(dtype=float)

    # Normalise survey x per-shank the same way the probe positions are
    # normalised, so the two coordinate systems line up regardless of
    # absolute offset.
    local_survey_x = survey_x.copy()
    for shank_id in np.unique(survey_shanks):
        mask = survey_shanks == shank_id
        local_survey_x[mask] -= survey_x[mask].min()

    values: dict[Electrode, float] = {}
    n_unmatched = 0
    tol2 = tolerance_um ** 2
    for i in range(len(survey_df)):
        shank_id = int(survey_shanks[i])
        table = shank_tables.get(shank_id)
        if table is None:
            n_unmatched += 1
            continue
        dx = table[:, 1] - local_survey_x[i]
        dy = table[:, 2] - survey_z[i]
        d2 = dx * dx + dy * dy
        j = int(np.argmin(d2))
        if d2[j] > tol2:
            n_unmatched += 1
            continue
        electrode_id = int(table[j, 0])
        values[Electrode(shank_id, electrode_id)] = float(survey_v[i])
    return values, n_unmatched


def validate_probe_match(
    survey_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    tolerance_um: float = DEFAULT_MATCH_TOLERANCE_UM,
) -> tuple[dict[Electrode, float], int]:
    """
    Confirm the survey file matches the currently loaded probe type.

    Returns (values_by_electrode, n_unmatched) on success.
    Raises ValueError if no survey rows map to the probe (strong signal
    of a probe-type mismatch) or if the match rate is implausibly low.
    """
    values, n_unmatched = match_survey_to_electrodes(survey_df, positions_df, tolerance_um)
    total = len(survey_df)
    if not values:
        raise ValueError(
            "No survey rows matched the currently selected probe. "
            "Check that the survey was exported from the same probe type "
            "that is selected in the GUI."
        )
    # If fewer than half the rows mapped, the shapes don't align — almost
    # certainly a probe-type mismatch even if a few rows happen to fit.
    if total and len(values) / total < 0.5:
        raise ValueError(
            f"Only {len(values)}/{total} survey rows matched the selected probe — "
            "this survey was likely exported from a different probe type."
        )
    return values, n_unmatched
