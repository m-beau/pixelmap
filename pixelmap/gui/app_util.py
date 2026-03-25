"""For general panel serving, both in Docker and in Ploomber"""

import socket
import threading
import time

import panel as pn
import psutil

from pixelmap.gui.gui import create_app

pn.extension(notifications=True)


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


def main(show=True, local=True):
    # Monitor potential memory leak
    # monitor_memory()

    # Serve the app
    print("Starting app...")
    if local:
        port = find_free_port(5003)
        pn.serve(create_app, port=port, show=show, title="Neuropixels Channelmap Generator", verbose=True)
    else:
        create_app().servable(title="Neuropixels Channelmap Generator")
