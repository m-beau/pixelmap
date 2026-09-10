#!/usr/bin/env python3
"""Warm the BrainGlobe atlas cache at image-build time.

Pre-downloading the common atlases means users don't sit through a download
the first time they open the anatomy overlay, and — far more importantly —
means the deployed container never has to reach the atlas host at all.

**This fails the build when it cannot warm the cache, and that is the point.**

An earlier version exited 0 regardless, on the reasoning that a warm cache is
"an optimisation, not a requirement: the app downloads whatever is missing on
demand".  That reasoning was backwards, and it shipped three broken releases
(v1.2.0, v1.2.1, v1.2.3) whose images contained no atlas data at all, because
GIN answered the CI runner with 403 while answering everyone else normally.
"Downloads on demand at runtime" meant, in production: network I/O on the
single-threaded Bokeh event loop, hundreds of MB decoded inside the serving
process, and a container that fell over under its own memory once someone
opened the anatomy panel.  A failed build is cheap; a silently cold image is
not.  See the module docstring of ``pixelmap/anatomy/atlas.py``.

Set ``PREFETCH_ALLOW_PARTIAL=1`` to downgrade the failure to a warning.  That
is for local experimentation only — never for a release build.

Why reading ``.annotation`` matters
-----------------------------------
``BrainGlobeAtlas(...)`` only fetches the manifest and metadata — a few
hundred KB — and pulls the annotation array from S3 lazily, the first time
something reads ``atlas.annotation``.  Constructing the atlas would therefore
"succeed" while leaving the expensive part for the first user to hit at
runtime, which is exactly what this script exists to prevent.
"""

import os
import sys
import time

DEFAULT_ATLASES = ("allen_mouse_25um", "whs_sd_rat_39um")

# Transient S3/CDN hiccups clear quickly; a real outage won't, and we want to
# hear about that rather than stall the build for minutes.
RETRY_DELAYS = (10, 30)


def warm_registry() -> bool:
    """Cache the atlas registry index (``last_versions.conf``) on disk.

    The GUI reads this to populate its atlas dropdown.  Baking it into the
    image keeps that read local, so building a session never waits on the
    network — see ``pixelmap.anatomy.atlas.list_atlases``.
    """
    from brainglobe_atlasapi.list_atlases import get_all_atlases_lastversions

    try:
        n = len(get_all_atlases_lastversions())
        print(f"[prefetch] registry: ok ({n} atlases)", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        print(f"[prefetch] registry: failed: {type(exc).__name__}: {exc}", flush=True)
        return False


def fetch(atlas_name: str) -> bool:
    """Download one atlas, retrying transient failures. True if it landed."""
    from brainglobe_atlasapi import BrainGlobeAtlas

    for attempt, delay in enumerate((*RETRY_DELAYS, None), start=1):
        try:
            atlas = BrainGlobeAtlas(atlas_name, check_latest=False)
            # Force the annotation onto disk (see the module docstring).
            _ = atlas.annotation
            print(f"[prefetch] {atlas_name}: ok", flush=True)
            return True
        except Exception as exc:  # noqa: BLE001 - retried, then reported below
            print(
                f"[prefetch] {atlas_name}: attempt {attempt} failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if delay is None:
                return False
            print(f"[prefetch] retrying in {delay}s...", flush=True)
            time.sleep(delay)

    return False


def main(argv: list[str]) -> int:
    atlases = tuple(argv[1:]) or DEFAULT_ATLASES
    allow_partial = os.environ.get("PREFETCH_ALLOW_PARTIAL") == "1"

    registry_ok = warm_registry()
    fetched = [name for name in atlases if fetch(name)]
    missing = [name for name in atlases if name not in fetched]

    print(f"[prefetch] cached {len(fetched)}/{len(atlases)} atlases", flush=True)
    if not missing and registry_ok:
        return 0

    problems = []
    if missing:
        problems.append(f"atlases not cached: {', '.join(missing)}")
    if not registry_ok:
        problems.append("registry index not cached")
    summary = "; ".join(problems)

    if allow_partial:
        print(f"[prefetch] WARNING: {summary} (PREFETCH_ALLOW_PARTIAL=1)", flush=True)
        return 0

    print(
        f"[prefetch] ERROR: {summary}.\n"
        "[prefetch] Refusing to build a cold image — it would download inside "
        "the serving process at runtime. Re-run the build once the atlas host "
        "is reachable.",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
