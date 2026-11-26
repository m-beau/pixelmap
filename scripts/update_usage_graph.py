#!/usr/bin/env python3
"""Redraw the README's usage graph and badge of the hosted GUI.

Fetches the visitor counts the server publishes at ``/usage-stats`` (see
``pixelmap/usage_stats.py``) and writes these files into ``--out-dir``:

- ``weekly_unique_visitors.csv``: unique visitors of each complete week;
- ``weekly_unique_visitors_light.svg`` and ``_dark.svg``: the graph of that
  table, one per GitHub colour scheme;
- ``unique_visitors_badge.json``: the all-time number of unique visitors, in
  the format of a shields.io endpoint badge.

The CSV is also the memory of this script. Weeks already in it are kept when
the server no longer reports them, so the graph survives a reset of the
server's database; where both have a week, the server's count wins. The week
in progress is left out until it is over, so every bar is a full week.

``.github/workflows/usage-stats.yml`` runs this every Monday and commits the
result to the ``usage-stats`` branch. It needs nothing but Python::

    python3 scripts/update_usage_graph.py --out-dir /tmp/usage
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import urllib.request
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from xml.sax.saxutils import escape

DEFAULT_URL = "https://pixelmap.pni.princeton.edu/usage-stats"
CSV_NAME = "weekly_unique_visitors.csv"
SVG_NAME = "weekly_unique_visitors_{theme}.svg"
BADGE_NAME = "unique_visitors_badge.json"
# Earlier versions drew one PNG; remove it so the branch holds no stale graph.
LEGACY_PNG_NAME = "weekly_unique_visitors.png"
CSV_FIELDS = ("week_start", "unique_visitors")

# The graphs have no background: they sit on GitHub's own page colour, so
# each scheme gets GitHub's text colours for it. One series, so one hue.
THEMES = {
    "light": {"series": "#2a78d6", "text": "#1f2328", "muted": "#59636e", "grid": "#d8dee4"},
    "dark": {"series": "#3987e5", "text": "#f0f6fc", "muted": "#9198a1", "grid": "#30363d"},
}
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans',Helvetica,Arial,sans-serif"

# Drawn at the size the README shows it, so the text is not scaled.
WIDTH, HEIGHT = 480, 150
PLOT_LEFT, PLOT_RIGHT = 26, WIDTH - 2
PLOT_TOP, PLOT_BOTTOM = 42, HEIGHT - 20  # PLOT_TOP is where the tallest bar ends
MAX_BAR_WIDTH = 14
MAX_MONTH_LABELS = 9


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


def fetch_usage(url: str) -> tuple[dict[date, int], int | None]:
    """Return the weekly counts and the all-time count (``None`` if not served)."""
    request = urllib.request.Request(url, headers={"User-Agent": "usage-graph-updater"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    total = payload.get("total_unique_visitors")
    return parse_weeks(payload["weeks"]), None if total is None else int(total)


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


def write_badge(badge_path: Path, total: int) -> None:
    badge = {"schemaVersion": 1, "label": "unique users", "message": f"{total:,}", "color": "blue"}
    badge_path.write_text(json.dumps(badge, indent=2) + "\n")


def _y_ticks(highest: int) -> list[int]:
    """Round gridline values, at most five above zero."""
    magnitude = 10 ** max(math.floor(math.log10(max(highest, 1))), 0)
    step = next(
        candidate
        for scale in (magnitude // 10 or 1, magnitude, magnitude * 10)
        for candidate in (scale, 2 * scale, 5 * scale)
        if highest / candidate <= 5
    )
    return list(range(0, highest + 1, step))


def _month_labels(slots: list[date]) -> dict[int, str]:
    """Label the first week of each month, thinned to what fits; years on January."""
    firsts = [i for i in range(1, len(slots)) if slots[i].month != slots[i - 1].month]
    every = next(
        (n for n in (1, 2, 3, 6, 12) if len(firsts) / n <= MAX_MONTH_LABELS),
        12,
    )
    return {
        i: str(slots[i].year) if slots[i].month == 1 else f"{slots[i]:%b}"
        for i in firsts
        if (slots[i].month - 1) % every == 0
    }


def render_svg(weeks: dict[date, int], theme: str) -> str:
    colors = THEMES[theme]
    first, last = min(weeks), max(weeks)
    # One slot per week of the span, so a week without a visit is a gap.
    slots = [first + timedelta(weeks=i) for i in range((last - first).days // 7 + 1)]
    counts = [weeks.get(week_start, 0) for week_start in slots]
    highest = max(max(counts), 1)

    slot_width = (PLOT_RIGHT - PLOT_LEFT) / len(slots)
    bar_width = min(slot_width * 0.72, MAX_BAR_WIDTH)
    radius = min(2.0, bar_width / 2)

    def y_of(value: float) -> float:
        return PLOT_BOTTOM - (PLOT_BOTTOM - PLOT_TOP) * value / highest

    def center_of(index: int) -> float:
        return PLOT_LEFT + (index + 0.5) * slot_width

    parts = []
    for tick in _y_ticks(highest):
        y = y_of(tick)
        parts.append(f'<line x1="{PLOT_LEFT}" x2="{PLOT_RIGHT}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{PLOT_LEFT - 6}" y="{y + 3.5:.1f}">{tick:,}</text>')
    grid = "".join(parts)

    # Bars: square on the baseline, rounded at the data end.
    parts = []
    for index, count in enumerate(counts):
        if count == 0:
            continue
        left, top = center_of(index) - bar_width / 2, y_of(count)
        right, r = left + bar_width, min(radius, PLOT_BOTTOM - top)
        parts.append(
            f"M{left:.1f},{PLOT_BOTTOM}V{top + r:.1f}Q{left:.1f},{top:.1f} {left + r:.1f},{top:.1f}"
            f"H{right - r:.1f}Q{right:.1f},{top:.1f} {right:.1f},{top + r:.1f}V{PLOT_BOTTOM}Z"
        )
    bars = "".join(parts)

    months = "".join(
        f'<text class="month" x="{center_of(index):.1f}" y="{PLOT_BOTTOM + 14}">{label}</text>'
        for index, label in _month_labels(slots).items()
    )

    # Label two bars, not all of them: the latest week, and the busiest one
    # unless its label would run into the latest's.
    latest, peak = len(counts) - 1, counts.index(max(counts))
    labelled = {latest} if (latest - peak) * slot_width < 24 else {latest, peak}
    values = "".join(
        f'<text class="value" x="{min(center_of(index), WIDTH - 8):.1f}" '
        f'y="{y_of(counts[index]) - 4:.1f}">{counts[index]:,}</text>'
        for index in sorted(labelled)
    )

    last_day = last + timedelta(days=6)
    through = f"through {last_day:%b} {last_day.day}, {last_day.year}"
    title = "Unique visitors per week"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{escape(title)}, {through}">\n'
        f"<title>{escape(title)}, {through}</title>\n"
        f"<style>text{{font-family:{FONT};font-size:10px;fill:{colors['muted']}}}"
        f".title{{font-size:12px;font-weight:600}}.tick{{text-anchor:end}}"
        f".month,.value{{text-anchor:middle}}.through{{text-anchor:end}}"
        f".value{{fill:{colors['text']}}}line{{stroke:{colors['grid']};stroke-width:1}}</style>\n"
        f'<text class="title" x="0" y="12">{escape(title)}</text>\n'
        f'<text class="through" x="{WIDTH - 2}" y="12">{through}</text>\n'
        f"<g>{grid}</g>\n"
        f'<path fill="{colors["series"]}" d="{bars}"/>\n'
        f"<g>{months}</g>\n"
        f"<g>{values}</g>\n"
        "</svg>\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--url",
        default=os.getenv("USAGE_STATS_URL") or DEFAULT_URL,
        help="where to fetch the counts (default: $USAGE_STATS_URL, else %(default)s)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help=f"directory holding {CSV_NAME}, the graphs and the badge",
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / CSV_NAME
    history = read_history(csv_path)
    fetched, total = fetch_usage(args.url)
    weeks = merge_weeks(history, fetched, today=datetime.now(UTC).date())
    if not weeks:
        print(f"No complete week of data at {args.url} or in {csv_path}.", file=sys.stderr)
        return 1

    write_history(csv_path, weeks)
    for theme in THEMES:
        (args.out_dir / SVG_NAME.format(theme=theme)).write_text(render_svg(weeks, theme))
    (args.out_dir / LEGACY_PNG_NAME).unlink(missing_ok=True)
    if total is None:
        # A server older than the badge: keep the graph going, leave the badge alone.
        print(f"{args.url} serves no total_unique_visitors: badge not updated.", file=sys.stderr)
    else:
        write_badge(args.out_dir / BADGE_NAME, total)

    latest = max(weeks)
    print(
        f"{len(weeks)} weeks ({len(fetched)} from the server, {len(history)} remembered); "
        f"week of {latest}: {weeks[latest]} unique visitors; all time: {total}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
