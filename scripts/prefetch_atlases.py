#!/usr/bin/env python3
"""Warm the BrainGlobe atlas cache at image-build time.

Pre-downloading the common atlases means users don't sit through a
multi-hundred-MB download the first time they open the anatomy overlay. It is
purely an optimisation, so a failure here must never fail the build: the app
downloads any missing atlas on demand at runtime, and production mounts a
volume at ~/.brainglobe so those downloads persist.

This exits 0 even when nothing could be fetched. It reports what it managed to
get so a half-warmed image is visible in the build log rather than silent.

Why this needs to be defensive
------------------------------
``BrainGlobeAtlas(..., check_latest=False)`` is *not* enough on its own.
``__init__`` reads ``self.remote_version`` before it ever consults
``check_latest``, and ``remote_version`` only swallows ``ConnectionError``.
When GIN answers but refuses (it intermittently 403s CI runners),
``conf_from_url`` falls back to a cached ``last_versions.conf`` that a fresh
image does not have yet, and the resulting ``FileNotFoundError`` escapes.
"""

import sys
import time

DEFAULT_ATLASES = ("allen_mouse_25um", "whs_sd_rat_39um")

# GIN rate-limits in bursts, so back off far enough to outlast a short block
# without stalling the build for minutes when it is genuinely down.
RETRY_DELAYS = (10, 30)


def fetch(atlas_name: str) -> bool:
    """Download one atlas, retrying transient failures. True if it landed."""
    from brainglobe_atlasapi import BrainGlobeAtlas

    for attempt, delay in enumerate((*RETRY_DELAYS, None), start=1):
        try:
            BrainGlobeAtlas(atlas_name, check_latest=False)
            print(f"[prefetch] {atlas_name}: ok", flush=True)
            return True
        except Exception as exc:  # noqa: BLE001 - any failure is non-fatal here
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

    fetched = [name for name in atlases if fetch(name)]
    missing = [name for name in atlases if name not in fetched]

    print(f"[prefetch] cached {len(fetched)}/{len(atlases)} atlases", flush=True)
    if missing:
        print(
            "[prefetch] WARNING: not pre-cached, will download on first use at "
            f"runtime: {', '.join(missing)}",
            flush=True,
        )

    # Always succeed - see the module docstring.
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
