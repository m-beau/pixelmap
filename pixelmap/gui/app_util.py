"""For general panel serving, both in Docker and in Ploomber"""

import socket
import threading
import time
from pathlib import Path

import panel as pn
import psutil

from pixelmap.error_reporting import install_global_handlers
from pixelmap.gui import watchdog
from pixelmap.gui.gui import create_app

pn.extension(notifications=True)
# Background-thread and top-level errors go to the error webhook too.
install_global_handlers()

# Browser tab icon, served by Bokeh at /favicon.ico -- assets/npix_map_logo.png
# centred on a transparent square canvas, at 16-256px. Passed to pn.serve below;
# the `panel serve` deployment gets it through --ico-path in entrypoint.sh
# instead, because that server is built before this module is ever imported.
FAVICON_PATH = Path(__file__).resolve().parent / "assets" / "favicon.ico"


def find_free_port(start_port=5007):
    """Find next available port starting from start_port"""
    for port in range(start_port, start_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free ports found")


def memory_monitorer():
    process = psutil.Process()
    while True:
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"Memory usage: {memory_mb:.1f} MB")
        time.sleep(10)


def monitor_memory():
    threading.Thread(target=memory_monitorer, daemon=True).start()


def _log_session_start():
    """Log one informative line per Bokeh session build.

    Each call to this (via `main(local=False)`) re-executes app.py inside a
    fresh Bokeh session, so this line is the per-session equivalent of the
    periodic `--mem-log-frequency` RSS log in entrypoint.sh -- it is what
    shows up in the logs every time a session is built (e.g. from a
    healthcheck hitting the wrong endpoint, see entrypoint.sh).
    """
    try:
        session_context = pn.state.curdoc.session_context
        session_id = session_context.id
        num_sessions = len(session_context.server_context.sessions)
        rss_mb = psutil.Process().memory_info().rss / 1024 / 1024
        print(
            f"Session {session_id} starting "
            f"({num_sessions} live sessions, RSS {rss_mb:.1f} MB)"
        )
    except Exception:  # noqa: BLE001 - falls back to the old message, never raises
        # session_context (and its server_context) is only populated when
        # running under an actual Bokeh server -- fall back rather than
        # error out of the app.
        print("Starting app...")


def main(show=True, local=True):
    # Monitor potential memory leak
    # monitor_memory()

    # Serve the app
    if local:
        print("Starting app...")
        port = find_free_port(5003)
        pn.serve(
            create_app,
            port=port,
            show=show,
            title="Neuropixels Channelmap Generator",
            ico_path=str(FAVICON_PATH),
            verbose=True,
        )
    else:
        # Once per process (app.py re-runs per session); we're on the server's
        # event loop here, which is the loop the stall watchdog must watch.
        watchdog.install()
        _log_session_start()
        create_app().servable(title="Neuropixels Channelmap Generator")
