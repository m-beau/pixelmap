"""Ensure the atlases PixelMap needs are present in the local brainglobe cache.

Used in two places:

* At Docker build time (under `timeout`), to bake the atlases into the image
  layer.  The build *requires* this to succeed: a non-zero exit fails the build
  so we never ship an image without atlases, and `timeout` turns a stalled GIN
  server into a bounded failure instead of an infinite hang.
* At container startup (via entrypoint.sh), as a cheap safety net: it re-verifies
  the atlases are present (e.g. in case a runtime volume mount shadowed
  ~/.brainglobe/) and re-fetches anything missing before the app boots.

Because both callers run the same code, the startup check is a no-op in the
normal case — brainglobe stores atlases on disk and we skip any that are already
downloaded.

Exit codes:
    0  all required atlases are present (downloaded now or already cached)
    1  at least one required atlas is still missing after attempting a download
"""

from __future__ import annotations

import sys

from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_atlasapi.list_atlases import get_downloaded_atlases

# Atlases baked into the image / required for the app to be useful offline.
REQUIRED_ATLASES = (
    "allen_mouse_25um",
    "whs_sd_rat_39um",
)


def ensure_atlas(name: str) -> bool:
    """Make sure `name` is downloaded. Return True if present afterwards."""
    if name in get_downloaded_atlases():
        print(f"atlas already cached: {name}")
        return True
    try:
        # Constructing the atlas triggers the download into ~/.brainglobe/.
        # check_latest=False avoids the remote version check hanging the boot.
        print(f"downloading atlas: {name} ...")
        BrainGlobeAtlas(name, check_latest=False)
    except Exception as exc:  # network/GIN failure — report, don't crash caller
        print(f"failed to download {name}: {exc}", file=sys.stderr)
    return name in get_downloaded_atlases()


def main() -> int:
    missing = [name for name in REQUIRED_ATLASES if not ensure_atlas(name)]
    if missing:
        print(f"missing atlases after fetch attempt: {', '.join(missing)}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
