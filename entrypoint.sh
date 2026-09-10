#!/bin/bash
set -euo pipefail

# Set default values if environment variables are not set
INTERNAL_PORT=${INTERNAL_PORT:-5006}
NUM_PROCS=${NUM_PROCS:-1}
ADDRESS=${ADDRESS:-localhost}
ALLOW_WEBSOCKET_ORIGIN=${ALLOW_WEBSOCKET_ORIGIN:-localhost:$INTERNAL_PORT}
# Browser tab icon (the PixelMap logo). `pn.serve` gets this from
# pixelmap.gui.app_util.FAVICON_PATH; `panel serve` builds its server before
# that module is imported, so it has to be passed on the command line.
ICO_PATH=${ICO_PATH:-./pixelmap/gui/assets/favicon.ico}
BRAINGLOBE_SEED=${BRAINGLOBE_SEED:-/opt/brainglobe-seed}
BRAINGLOBE_DIR=${BRAINGLOBE_DIR:-/root/.brainglobe}

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*"; }

# Restore anything the image baked in that the mounted volume doesn't have.
#
# Production mounts a named volume over ~/.brainglobe. Docker seeds such a
# volume from the image only while it is empty, so an existing volume masks
# the atlases baked into every later image. Copying no-clobber means the
# volume keeps whatever users downloaded at runtime while still gaining the
# prefetched atlases and the registry index, on every start.
seed_brainglobe_cache() {
    if [ ! -d "$BRAINGLOBE_SEED" ]; then
        log "brainglobe seed $BRAINGLOBE_SEED missing - skipping"
        return
    fi
    mkdir -p "$BRAINGLOBE_DIR"
    cp -a --no-clobber "$BRAINGLOBE_SEED/." "$BRAINGLOBE_DIR/" 2>/dev/null || true
    local atlases_dir="$BRAINGLOBE_DIR/brainglobe-atlasapi/atlases"
    local n=0
    if [ -d "$atlases_dir" ]; then
        n=$(find "$atlases_dir" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
    fi
    log "brainglobe cache at $BRAINGLOBE_DIR: $n atlas(es) available"
    if [ "$n" -eq 0 ]; then
        log "WARNING: no atlases cached - the anatomy overlay will download on first use"
    fi
}

seed_brainglobe_cache

log "INTERNAL_PORT $INTERNAL_PORT"
log "ADDRESS $ADDRESS"
log "ALLOW_WEBSOCKET_ORIGIN $ALLOW_WEBSOCKET_ORIGIN"
log "NUM_PROCS $NUM_PROCS"

# Start the Panel application
exec uv run panel serve ./app.py \
    --address "$ADDRESS" \
    --port "$INTERNAL_PORT" \
    --allow-websocket-origin "$ALLOW_WEBSOCKET_ORIGIN" \
    --ico-path "$ICO_PATH" \
    --num-procs "$NUM_PROCS" \
    --session-token-expiration 3600000 \
    --check-unused-sessions 10000 \
    --unused-session-lifetime 60000 \
    --show
