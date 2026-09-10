#!/usr/bin/env python3
"""Regenerate ``pixelmap/anatomy/_registry_snapshot.py`` from the live registry.

The snapshot is what fills the GUI's atlas dropdown when the on-disk registry
cache is missing *and* the atlas host is unreachable, so it only needs to be
refreshed when BrainGlobe adds atlases — a few times a year. Being stale costs
a missing entry in a dropdown; being absent costs a blocking network call on
the server's event loop (see ``pixelmap.anatomy.atlas.list_atlases``).

Usage::

    python scripts/refresh_registry_snapshot.py
"""

import datetime
import sys
import textwrap
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "pixelmap" / "anatomy" / "_registry_snapshot.py"


def main() -> int:
    import brainglobe_atlasapi as bg
    from brainglobe_atlasapi.list_atlases import get_all_atlases_lastversions

    names = sorted(get_all_atlases_lastversions().keys())
    if not names:
        print("Registry came back empty — refusing to overwrite the snapshot.")
        return 1

    body = [
        '"""Offline snapshot of the BrainGlobe atlas registry.',
        "",
        textwrap.fill(
            "Populates the GUI's atlas dropdown when the on-disk registry cache is "
            "missing and the network is unreachable, so a cold cache degrades to a "
            "slightly stale list instead of a blocking HTTP call on the server's "
            "event loop. See :func:`pixelmap.anatomy.atlas.list_atlases`.",
            79,
        ),
        "",
        textwrap.fill(
            "Generated, do not hand-edit. Refresh with "
            "``python scripts/refresh_registry_snapshot.py``.",
            79,
        ),
        "",
        f"Snapshot taken {datetime.date.today().isoformat()} "
        f"from brainglobe-atlasapi {bg.__version__}.",
        '"""',
        "",
        "REGISTRY_SNAPSHOT: tuple[str, ...] = (",
        *[f'    "{n}",' for n in names],
        ")",
        "",
    ]
    TARGET.write_text("\n".join(body))
    print(f"Wrote {len(names)} atlas names to {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
