"""The first session must not pay one-time startup costs on the event loop.

``panel serve`` imports ``app.py`` per session, on the Bokeh event loop that
serves every session and ``/liveness``.  Whatever is cold (the imports, the
wiring pickle, the default atlas's metadata) therefore froze the server during
the first visit after every start.  ``entrypoint.sh`` warms all of it through
Panel's ``--setup`` hook, before the server accepts connections.
"""

from __future__ import annotations

import logging
import re
import runpy
import subprocess
import sys
from pathlib import Path

import panel as pn
import pytest

from pixelmap.anatomy import atlas as atlas_module
from pixelmap.gui import gui as gui_module

ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT = ROOT / "entrypoint.sh"
SCRIPT = ROOT / "scripts" / "warm_first_session.py"


def test_entrypoint_runs_the_warmup_before_serving():
    entrypoint = ENTRYPOINT.read_text()
    match = re.search(r"--setup\s+(\./\S+)", entrypoint)
    assert match, "entrypoint.sh must pass --setup to panel serve"
    assert (ROOT / match.group(1)).resolve() == SCRIPT


def test_warmup_fills_the_one_time_caches_in_a_fresh_process():
    probe = (
        "import runpy, sys\n"
        f"runpy.run_path({str(SCRIPT)!r})\n"
        "assert 'pixelmap.gui.gui' in sys.modules\n"
        "from pixelmap import backend\n"
        "assert backend._load_wiring_maps.cache_info().currsize == 1\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=ROOT, timeout=300, check=False
    )
    assert done.returncode == 0, done.stderr


def test_warmup_builds_one_gui_and_keeps_no_session_state(monkeypatch):
    built = []
    real = gui_module.ChannelmapGUI

    def spy(*args, **kwargs):
        built.append(real(*args, **kwargs))
        return built[-1]

    monkeypatch.setattr(gui_module, "ChannelmapGUI", spy)
    views_before = set(pn.state._views)

    namespace = runpy.run_path(str(SCRIPT))

    assert len(built) == 1
    assert pn.state.curdoc is None
    assert set(pn.state._views) == views_before
    assert not any(isinstance(v, real) for v in namespace.values())


@pytest.mark.parametrize("error", [RuntimeError("no wiring"), OSError("disk"), ValueError("bad")])
def test_a_failing_warmup_does_not_stop_the_server(monkeypatch, caplog, error):
    def boom():
        raise error

    monkeypatch.setattr(gui_module, "ChannelmapGUI", boom)
    with caplog.at_level(logging.WARNING, logger="pixelmap.startup"):
        runpy.run_path(str(SCRIPT))  # Panel aborts startup if this raises
    assert any(str(error) in r.getMessage() or r.exc_info for r in caplog.records)


def test_a_failing_import_does_not_stop_the_server(monkeypatch, caplog):
    # None in sys.modules makes `import pixelmap.gui.gui` raise ImportError.
    monkeypatch.setitem(sys.modules, "pixelmap.gui.gui", None)
    with caplog.at_level(logging.WARNING, logger="pixelmap.startup"):
        runpy.run_path(str(SCRIPT))
    assert any(r.exc_info for r in caplog.records)


def test_warmup_with_no_atlas_on_disk_never_opens_or_downloads_one(monkeypatch):
    opened = []
    monkeypatch.setattr(atlas_module, "get_downloaded_atlases", list)
    monkeypatch.setattr(atlas_module, "BrainGlobeAtlas", lambda *a, **k: opened.append(a))
    atlas_module.get_atlas.cache_clear()
    try:
        runpy.run_path(str(SCRIPT))
    finally:
        atlas_module.get_atlas.cache_clear()
    assert opened == []
