"""Tests for the event-loop stall and memory watchdogs.

They exist so production problems show up as a loud log line with a stack
trace (which the Slack log forwarder picks up) long before the container gets
OOM-killed or restarted for failing its healthcheck.
"""

from __future__ import annotations

import asyncio
import logging
import threading

import pytest

from pixelmap.gui import watchdog
from pixelmap.gui.watchdog import MemoryWatch, StallWatchdog


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def caplog_warn(caplog):
    caplog.set_level(logging.INFO, logger="pixelmap.watchdog")
    return caplog


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING]


# --------------------------------------------------------------------------
# StallWatchdog
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, -1, -0.5])
def test_stall_threshold_must_be_positive(bad):
    with pytest.raises(ValueError):
        StallWatchdog(threshold_s=bad)


def test_no_report_below_threshold(clock, caplog_warn):
    wd = StallWatchdog(threshold_s=5, clock=clock)
    wd.beat()
    clock.t += 4.99
    assert wd.check() is False
    assert _warnings(caplog_warn) == []


def test_report_at_exact_threshold(clock, caplog_warn):
    wd = StallWatchdog(threshold_s=5, clock=clock)
    wd.beat()
    clock.t += 5
    assert wd.check() is True


def test_stall_reported_once_per_stall(clock, caplog_warn):
    wd = StallWatchdog(threshold_s=5, clock=clock)
    wd.beat()
    clock.t += 6
    assert wd.check() is True
    clock.t += 30
    assert wd.check() is False
    assert len(_warnings(caplog_warn)) == 1
    assert "[watchdog]" in _warnings(caplog_warn)[0].getMessage()


def test_recovery_logs_duration_and_rearms(clock, caplog_warn):
    wd = StallWatchdog(threshold_s=5, clock=clock)
    wd.beat()
    clock.t += 42
    wd.check()
    wd.beat()  # loop came back
    recovered = [r for r in caplog_warn.records if "recovered" in r.getMessage()]
    assert len(recovered) == 1
    assert "42.0" in recovered[0].getMessage()

    # A second, separate stall is reported again.
    clock.t += 10
    assert wd.check() is True
    assert len([r for r in _warnings(caplog_warn) if "blocked for" in r.getMessage()]) == 2


def test_beat_without_stall_logs_nothing(clock, caplog_warn):
    wd = StallWatchdog(threshold_s=5, clock=clock)
    for _ in range(3):
        wd.beat()
        clock.t += 1
    assert caplog_warn.records == []


def test_check_before_first_beat_does_not_raise(clock, caplog_warn):
    """Before the loop ever beats there is no thread to dump."""
    wd = StallWatchdog(threshold_s=5, clock=clock)
    clock.t += 10
    assert wd.check() is True
    assert "unavailable" in _warnings(caplog_warn)[0].getMessage()


def test_report_includes_blocked_threads_stack(clock, caplog_warn):
    """The whole point: the warning says *what* is blocking the loop."""
    wd = StallWatchdog(threshold_s=5, clock=clock)
    beaten = threading.Event()
    release = threading.Event()

    def _blocking_handler():
        wd.beat()
        beaten.set()
        release.wait(timeout=10)

    t = threading.Thread(target=_blocking_handler, daemon=True)
    t.start()
    try:
        assert beaten.wait(timeout=5)
        clock.t += 10
        assert wd.check() is True
    finally:
        release.set()
        t.join(timeout=5)
    assert "_blocking_handler" in _warnings(caplog_warn)[0].getMessage()


# --------------------------------------------------------------------------
# MemoryWatch
# --------------------------------------------------------------------------

@pytest.mark.parametrize("limit", [None, 0, -100])
def test_memory_watch_disabled(limit, caplog_warn):
    mw = MemoryWatch(limit_mb=limit)
    assert mw.check(10_000_000) is False
    assert caplog_warn.records == []


def test_memory_below_limit_is_quiet(caplog_warn):
    mw = MemoryWatch(limit_mb=1000)
    assert mw.check(999.9) is False
    assert caplog_warn.records == []


def test_memory_at_limit_warns(caplog_warn):
    mw = MemoryWatch(limit_mb=1000)
    assert mw.check(1000) is True
    assert "[watchdog]" in _warnings(caplog_warn)[0].getMessage()


def test_memory_warns_once_until_rearmed(caplog_warn):
    mw = MemoryWatch(limit_mb=1000, rearm_fraction=0.9)
    assert mw.check(1500) is True
    assert mw.check(1600) is False
    # Dipping below the limit but above the re-arm point doesn't re-arm --
    # otherwise RSS hovering around the limit would spam the channel.
    assert mw.check(950) is False
    assert mw.check(1100) is False
    # Dropping below 90% re-arms.
    assert mw.check(899) is False
    assert mw.check(1200) is True
    assert len(_warnings(caplog_warn)) == 2


@pytest.mark.parametrize("bad", [0, -0.1, 1.5])
def test_memory_rearm_fraction_validated(bad):
    with pytest.raises(ValueError):
        MemoryWatch(limit_mb=1000, rearm_fraction=bad)


# --------------------------------------------------------------------------
# Environment config
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [(None, 7.0), ("", 7.0), ("12", 12.0), ("2.5", 2.5), ("0", 0.0), ("nope", 7.0)],
)
def test_env_float(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("PIXELMAP_TEST_KNOB", raising=False)
    else:
        monkeypatch.setenv("PIXELMAP_TEST_KNOB", raw)
    assert watchdog._env_float("PIXELMAP_TEST_KNOB", 7.0) == expected


@pytest.mark.parametrize(
    "raw, expected",
    # Finite zero and negatives stay accepted: they deliberately switch a check off.
    [(" 12 ", 12.0), ("-1", -1.0), ("-0", 0.0), ("1e3", 1000.0)],
)
def test_env_float_accepts_finite_values(monkeypatch, raw, expected):
    monkeypatch.setenv("PIXELMAP_TEST_KNOB", raw)
    assert watchdog._env_float("PIXELMAP_TEST_KNOB", 7.0) == expected


@pytest.mark.parametrize("raw", ["nan", "NaN", "inf", "-inf", "Infinity", " inf "])
def test_env_float_rejects_non_finite_values(monkeypatch, caplog, raw):
    """``float()`` parses these, but none is a usable threshold: NaN makes every
    stall comparison misbehave, and ±inf silently disables a check."""
    monkeypatch.setenv("PIXELMAP_TEST_KNOB", raw)
    with caplog.at_level(logging.WARNING, logger=watchdog.log.name):
        assert watchdog._env_float("PIXELMAP_TEST_KNOB", 7.0) == 7.0
    assert "PIXELMAP_TEST_KNOB" in caplog.text


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_env_float_unset_or_blank_is_silent(monkeypatch, caplog, raw):
    if raw is None:
        monkeypatch.delenv("PIXELMAP_TEST_KNOB", raising=False)
    else:
        monkeypatch.setenv("PIXELMAP_TEST_KNOB", raw)
    with caplog.at_level(logging.WARNING, logger=watchdog.log.name):
        assert watchdog._env_float("PIXELMAP_TEST_KNOB", 7.0) == 7.0
    assert caplog.text == ""


# --------------------------------------------------------------------------
# install()
# --------------------------------------------------------------------------

@pytest.fixture
def fresh_install(monkeypatch):
    monkeypatch.setattr(watchdog, "_installed", None)
    yield
    if watchdog._installed is not None:
        watchdog._installed.stop()


def test_install_is_idempotent(fresh_install):
    """app.py is re-executed per session; the watchdog must start once per process."""

    async def run():
        first = watchdog.install(stall_threshold_s=5, rss_warn_mb=None)
        second = watchdog.install(stall_threshold_s=5, rss_warn_mb=None)
        return first, second

    first, second = asyncio.run(run())
    assert first is not None
    assert first is second
    assert sum(t.name == "pixelmap-watchdog" for t in threading.enumerate()) == 1


@pytest.mark.parametrize("threshold", [0, -1])
def test_install_disabled_by_nonpositive_threshold(fresh_install, threshold):
    async def run():
        return watchdog.install(stall_threshold_s=threshold, rss_warn_mb=None)

    assert asyncio.run(run()) is None
