"""Pay the first session's one-time costs before the server accepts connections.

``panel serve`` runs ``app.py`` per session on the Bokeh event loop, which also
serves every other session and ``/liveness``.  Whatever is still cold when the
first visitor arrives (the imports, the wiring pickle, the default atlas's
metadata) would freeze that loop.

``entrypoint.sh`` runs this through ``panel serve --setup``.  Building one
throwaway GUI warms the one-time costs of a session's build, without a list here
to keep in step with the constructor.  It has no document, so it creates no
session or analytics event.

This runs before the port is bound only with ``NUM_PROCS=1``, the image's
default.  Above that, Panel runs it on each process's event loop after the
server starts, which costs about what the first session would pay anyway.

A failure only costs the speed-up, and the first session reports it properly,
so it must not stop the server: Panel aborts startup if this script raises.
"""

import logging
import time

_started = time.perf_counter()
try:
    from pixelmap.gui.gui import ChannelmapGUI

    ChannelmapGUI()
except Exception:  # see the module docstring
    logging.getLogger("pixelmap.startup").warning(
        "First-session warmup failed; the first visitor will pay for it",
        exc_info=True,
    )
else:
    print(f"First-session warmup finished in {time.perf_counter() - _started:.1f} s")
