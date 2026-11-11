"""Tests for scripts/warm_first_session.py, which explains why the warmup exists."""

from __future__ import annotations

import logging
import re
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock

import panel as pn

from pixelmap.anatomy import atlas as atlas_module
from pixelmap.gui import gui as gui_module

ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT = ROOT / "entrypoint.sh"
SCRIPT = ROOT / "scripts" / "warm_first_session.py"
DEFAULT_ATLAS = "allen_mouse_25um"


def _run_warmup(caplog):
    """Run the script as Panel would; return its namespace and logged failures."""
    with caplog.at_level(logging.WARNING, logger="pixelmap.startup"):
        namespace = runpy.run_path(str(SCRIPT))  # Panel aborts startup if this raises
    return namespace, [r for r in caplog.records if r.name == "pixelmap.startup"]


def test_entrypoint_runs_the_warmup_before_serving():
    entrypoint = ENTRYPOINT.read_text()
    match = re.search(r"--setup\s+(\./\S+)", entrypoint)
    assert match, "entrypoint.sh must pass --setup to panel serve"
    assert (ROOT / match.group(1)).resolve() == SCRIPT


def test_warmup_fills_the_wiring_cache_in_a_fresh_process():
    probe = (
        "import runpy\n"
        f"runpy.run_path({str(SCRIPT)!r})\n"
        "from pixelmap import backend\n"
        "assert backend._load_wiring_maps.cache_info().currsize == 1\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=ROOT, timeout=300, check=False
    )
    assert done.returncode == 0, done.stderr


def test_warmup_builds_one_gui_and_keeps_no_session_state(monkeypatch, caplog):
    real = gui_module.ChannelmapGUI
    spy = Mock(wraps=real)
    monkeypatch.setattr(gui_module, "ChannelmapGUI", spy)
    views_before = set(pn.state._views)

    namespace, failures = _run_warmup(caplog)

    spy.assert_called_once()
    assert failures == []  # the script swallows errors, so the spy alone proves nothing
    assert pn.state.curdoc is None
    assert set(pn.state._views) == views_before
    assert not any(isinstance(value, real) for value in namespace.values())


def test_a_failing_warmup_does_not_stop_the_server(monkeypatch, caplog):
    monkeypatch.setattr(gui_module, "ChannelmapGUI", Mock(side_effect=RuntimeError("no wiring")))
    _, failures = _run_warmup(caplog)
    assert [r for r in failures if r.exc_info]


def test_a_failing_import_does_not_stop_the_server(monkeypatch, caplog):
    # None in sys.modules makes `import pixelmap.gui.gui` raise ImportError.
    monkeypatch.setitem(sys.modules, "pixelmap.gui.gui", None)
    _, failures = _run_warmup(caplog)
    assert [r for r in failures if r.exc_info]


def test_warmup_opens_the_default_atlas_when_it_is_on_disk(monkeypatch, caplog):
    fake_atlas = MagicMock(return_value=MagicMock(orientation="asr", resolution=(25.0, 25.0, 25.0)))
    monkeypatch.setattr(atlas_module, "get_downloaded_atlases", lambda: [DEFAULT_ATLAS])
    monkeypatch.setattr(atlas_module, "_annotation_is_cached", lambda atlas: True)
    monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", fake_atlas)
    atlas_module.get_atlas.cache_clear()
    try:
        _, failures = _run_warmup(caplog)
        assert failures == []
        fake_atlas.assert_any_call(DEFAULT_ATLAS, check_latest=False)
        assert atlas_module.get_atlas.cache_info().currsize == 1
    finally:
        atlas_module.get_atlas.cache_clear()


def test_warmup_with_no_atlas_on_disk_never_opens_or_downloads_one(monkeypatch, caplog):
    fake_atlas = Mock()
    monkeypatch.setattr(atlas_module, "get_downloaded_atlases", list)  # always empty
    monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", fake_atlas)
    atlas_module.get_atlas.cache_clear()
    try:
        _, failures = _run_warmup(caplog)
    finally:
        atlas_module.get_atlas.cache_clear()
    assert failures == []
    fake_atlas.assert_not_called()
