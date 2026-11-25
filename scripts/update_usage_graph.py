#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.7"]
# ///
"""Redraw the graph of unique visitors per week of the hosted GUI.

Fetches the weekly counts the server publishes at ``/usage-stats`` (see
``pixelmap/usage_stats.py``) and writes two files into ``--out-dir``:

- ``weekly_unique_visitors.csv``: one row per complete week;
- ``weekly_unique_visitors.png``: the graph of that table.

The CSV is also the memory of this script. Weeks already in it are kept when
the server no longer reports them, so the graph survives a reset of the
server's database; where both have a week, the server's count wins. The week
in progress is left out until it is over, so every bar is a full week.

``.github/workflows/usage-stats.yml`` runs this every Monday and commits the
result to the ``usage-stats`` branch. To run it by hand::

    uv run scripts/update_usage_graph.py --out-dir /tmp/usage
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.request
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

DEFAULT_URL = "https://pixelmap.pni.princeton.edu/usage-stats"
CSV_NAME = "weekly_unique_visitors.csv"
PNG_NAME = "weekly_unique_visitors.png"
CSV_FIELDS = ("week_start", "unique_visitors")

# One series, so one hue; text and grid stay in neutral ink.
SURFACE = "#fcfcfb"
SERIES = "#2a78d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e0"


def parse_weeks(rows: list[dict]) -> dict[date, int]:
    """Validate ``[{"week_start": "YYYY-MM-DD", "unique_visitors": n}, ...]``."""
    weeks: dict[date, int] = {}
    for row in rows:
        week_start = date.fromisoformat(str(row["week_start"]))
        if week_start.weekday() != 0:
            raise ValueError(f"week_start {week_start} is not a Monday")
        count = int(row["unique_visitors"])
        if count < 0:
            raise ValueError(f"negative visitor count for the week of {week_start}")
        weeks[week_start] = count
    return weeks


def fetch_weeks(url: str) -> dict[date, int]:
    request = urllib.request.Request(url, headers={"User-Agent": "usage-graph-updater"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    return parse_weeks(payload["weeks"])


def read_history(csv_path: Path) -> dict[date, int]:
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="") as handle:
        return parse_weeks(list(csv.DictReader(handle)))


def write_history(csv_path: Path, weeks: dict[date, int]) -> None:
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CSV_FIELDS)
        for week_start, count in sorted(weeks.items()):
            writer.writerow([week_start.isoformat(), count])


def merge_weeks(history: dict[date, int], fetched: dict[date, int], today: date) -> dict[date, int]:
    """Server counts over remembered ones, complete weeks only."""
    merged = {**history, **fetched}
    return {
        week_start: count
        for week_start, count in sorted(merged.items())
        if week_start + timedelta(days=7) <= today
    }


def render(weeks: dict[date, int], png_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    # Each bar sits mid-week, on the Thursday of its Monday-to-Sunday week.
    mids = [
        datetime.combine(week_start, datetime.min.time()) + timedelta(days=3.5)
        for week_start in weeks
    ]
    counts = list(weeks.values())

    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.80, bottom=0.10)
    ax.set_facecolor(SURFACE)

    # Bars fill 80% of their week, but stay thin while there are few weeks.
    x_min, x_max = mids[0] - timedelta(days=7), mids[-1] + timedelta(days=7)
    days_per_inch = (x_max - x_min).days / (ax.get_position().width * fig.get_figwidth())
    bar_days = min(5.6, 0.25 * days_per_inch)
    ax.bar(mids, counts, width=bar_days, color=SERIES, linewidth=0, zorder=2)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, max(max(counts), 1) * 1.15)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, show_offset=False))

    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", colors=TEXT_SECONDARY, labelsize=9, length=0)

    # Label two bars, not all of them: the latest week and the busiest one.
    latest, peak = len(counts) - 1, counts.index(max(counts))
    for index in {latest, peak}:
        ax.annotate(
            f"{counts[index]:,}",
            xy=(mids[index], counts[index]),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=TEXT_PRIMARY,
        )

    fig.text(
        0.012, 0.955, "Unique visitors per week",
        fontsize=13, fontweight="bold", color=TEXT_PRIMARY, va="top",
    )
    last_day = max(weeks) + timedelta(days=6)
    fig.text(
        0.012, 0.885,
        "PixelMap online GUI · weeks run Monday to Sunday (UTC)"
        f" · through {last_day:%b} {last_day.day}, {last_day.year}",
        fontsize=9, color=TEXT_SECONDARY, va="top",
    )
    fig.savefig(png_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--url",
        default=os.getenv("USAGE_STATS_URL") or DEFAULT_URL,
        help="where to fetch the weekly counts (default: $USAGE_STATS_URL, else %(default)s)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help=f"directory holding {CSV_NAME} and {PNG_NAME}",
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / CSV_NAME
    history = read_history(csv_path)
    fetched = fetch_weeks(args.url)
    weeks = merge_weeks(history, fetched, today=datetime.now(UTC).date())
    if not weeks:
        print(f"No complete week of data at {args.url} or in {csv_path}.", file=sys.stderr)
        return 1

    write_history(csv_path, weeks)
    render(weeks, args.out_dir / PNG_NAME)
    latest = max(weeks)
    print(
        f"{len(weeks)} weeks ({len(fetched)} from the server, {len(history)} remembered); "
        f"week of {latest}: {weeks[latest]} unique visitors"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
