"""Early warnings for the deployed server: event-loop stalls and high RSS.

The production container used to die with no output at all -- a SIGKILL from
the OOM killer or an orchestrator acting on a failed healthcheck leaves no
Python traceback.  The only clue was that the periodic ``Memory usage`` log
sometimes fired 20-90 s late, i.e. the single Bokeh event loop had been
blocked that long.

This module turns those conditions into loud, greppable ``[watchdog]`` WARNING
lines while the app is still alive, so the Slack log forwarder flags them:

* :class:`StallWatchdog` -- the event loop beats a heartbeat every second; a
  separate thread notices when it stops and logs the *blocked thread's stack*,
  which names the handler responsible.  Logs again with the duration once the
  loop recovers.
* :class:`MemoryWatch` -- warns once when RSS crosses a soft limit, re-arming
  only after it drops back below, so RSS hovering at the limit doesn't spam.

Neither ever raises into the app or kills anything: they only log.

Configuration (environment, read by :func:`install`):

``PIXELMAP_STALL_WARN_S``  seconds of blocking before warning (default 5;
                           ``0`` disables the watchdog entirely)
``PIXELMAP_RSS_WARN_MB``   RSS soft limit in MB (default unset = no memory
                           warnings)
"""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable

log = logging.getLogger("pixelmap.watchdog")

#: The process-wide watchdog, so :func:`install` is idempotent -- ``app.py`` is
#: re-executed for every Bokeh session, but this module is imported once.
_installed: _Watchdog | None = None
_install_lock = threading.Lock()


def _env_float(name: str, default: float) -> float:
    """Read a threshold from the environment; blank or unset means ``default``.

    Finite zero and negatives are returned as-is (callers treat them as "off").
    ``nan`` and ``±inf`` parse as floats but aren't usable thresholds -- NaN
    breaks every comparison, and infinity silently disables the check -- so they
    are rejected like any other non-number.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value):
        log.warning("[watchdog] ignoring %s=%r: not a finite number, using %s", name, raw, default)
        return default
    return value


def _rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 1024 / 1024


class StallWatchdog:
    """Detect a blocked event loop and report what it is stuck on.

    :meth:`beat` runs *on* the loop; :meth:`check` runs from any other thread.
    """

    def __init__(self, threshold_s: float = 5.0, clock: Callable[[], float] = time.monotonic):
        if threshold_s <= 0:
            raise ValueError(f"threshold_s must be positive, got {threshold_s}")
        self.threshold_s = threshold_s
        self._clock = clock
        self._last_beat = clock()
        self._loop_thread_id: int | None = None
        self._reported = False

    def beat(self) -> None:
        now = self._clock()
        if self._reported:
            log.warning(
                "[watchdog] event loop recovered after being blocked %.1f s",
                now - self._last_beat,
            )
            self._reported = False
        self._last_beat = now
        self._loop_thread_id = threading.get_ident()

    def check(self) -> bool:
        """Log a warning if the loop is newly stalled.  Returns whether it did."""
        lag = self._clock() - self._last_beat
        if lag < self.threshold_s or self._reported:
            return False
        self._reported = True
        log.warning(
            "[watchdog] event loop blocked for %.1f s (threshold %.1f s); "
            "all sessions and the healthcheck are frozen. Blocked thread:\n%s",
            lag,
            self.threshold_s,
            self._loop_stack(),
        )
        return True

    def _loop_stack(self) -> str:
        frame = sys._current_frames().get(self._loop_thread_id) if self._loop_thread_id else None
        if frame is None:
            return "  (stack unavailable: loop thread unknown)"
        return "".join(traceback.format_stack(frame))


class MemoryWatch:
    """Warn once when RSS reaches ``limit_mb``; re-arm below ``rearm_fraction``."""

    def __init__(self, limit_mb: float | None, rearm_fraction: float = 0.9):
        if not 0 < rearm_fraction <= 1:
            raise ValueError(f"rearm_fraction must be in (0, 1], got {rearm_fraction}")
        self.limit_mb = limit_mb if limit_mb and limit_mb > 0 else None
        self.rearm_fraction = rearm_fraction
        self._armed = True

    def check(self, rss_mb: float) -> bool:
        """Log a warning if RSS newly crossed the limit.  Returns whether it did."""
        if self.limit_mb is None:
            return False
        if rss_mb < self.limit_mb * self.rearm_fraction:
            self._armed = True
            return False
        if rss_mb < self.limit_mb or not self._armed:
            return False
        self._armed = False
        log.warning(
            "[watchdog] RSS %.0f MB reached the %.0f MB warning threshold; "
            "the container risks an OOM kill",
            rss_mb,
            self.limit_mb,
        )
        return True


class _Watchdog:
    """Wires both checks to the running loop and a daemon polling thread."""

    def __init__(self, stall: StallWatchdog, memory: MemoryWatch, poll_s: float = 1.0):
        from tornado.ioloop import PeriodicCallback

        self.stall = stall
        self.memory = memory
        self._stop = threading.Event()
        self._poll_s = poll_s
        # Created on the loop thread, so it schedules onto the server's loop.
        self._beat = PeriodicCallback(stall.beat, poll_s * 1000)
        self._beat.start()
        stall.beat()
        self._thread = threading.Thread(target=self._run, name="pixelmap-watchdog", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._poll_s):
            try:
                self.stall.check()
                self.memory.check(_rss_mb())
            except Exception:  # a broken watchdog must never take the app down
                log.exception("[watchdog] check failed")

    def stop(self) -> None:
        self._stop.set()
        self._beat.stop()
        self._thread.join(timeout=5)


def install(
    stall_threshold_s: float | None = None,
    rss_warn_mb: float | None = None,
) -> _Watchdog | None:
    """Start the watchdogs once per process.  Call from the event-loop thread.

    Arguments default to the environment (see module docstring).  Returns the
    running watchdog, or ``None`` when disabled by a non-positive threshold.
    """
    global _installed

    if stall_threshold_s is None:
        stall_threshold_s = _env_float("PIXELMAP_STALL_WARN_S", 5.0)
    if rss_warn_mb is None:
        rss_warn_mb = _env_float("PIXELMAP_RSS_WARN_MB", 0.0)

    with _install_lock:
        if _installed is not None:
            return _installed
        if stall_threshold_s <= 0:
            return None
        _installed = _Watchdog(StallWatchdog(stall_threshold_s), MemoryWatch(rss_warn_mb))
        log.info(
            "[watchdog] started: stall warning at %.1f s, RSS warning at %s",
            stall_threshold_s,
            f"{rss_warn_mb:.0f} MB" if rss_warn_mb and rss_warn_mb > 0 else "off",
        )
        return _installed
