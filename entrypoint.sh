#!/bin/bash

# Set default values if environment variables are not set
INTERNAL_PORT=${INTERNAL_PORT:-5006}
NUM_PROCS=${NUM_PROCS:-1}
ADDRESS=${ADDRESS:-localhost}
ALLOW_WEBSOCKET_ORIGIN=${ALLOW_WEBSOCKET_ORIGIN:-localhost:$INTERNAL_PORT}

echo "$(date '+%Y-%m-%d %H:%M:%S') INTERNAL_PORT $INTERNAL_PORT"
echo "$(date '+%Y-%m-%d %H:%M:%S') ADDRESS $ADDRESS"
echo "$(date '+%Y-%m-%d %H:%M:%S') ALLOW_WEBSOCKET_ORIGIN $ALLOW_WEBSOCKET_ORIGIN"
echo "$(date '+%Y-%m-%d %H:%M:%S') NUM_PROCS $NUM_PROCS"

# Verify the required atlases are present before serving.  They are normally
# baked into the image at build time, so this is a fast no-op; it only does work
# if a runtime volume mount shadowed ~/.brainglobe/.  REQUIRE_ATLASES=1
# (default) makes a still-missing atlas a hard startup failure; set
# REQUIRE_ATLASES=0 to warn and start anyway (atlases download lazily on use).
REQUIRE_ATLASES=${REQUIRE_ATLASES:-1}
echo "$(date '+%Y-%m-%d %H:%M:%S') verifying atlases ..."
if ! uv run python /fetch_atlases.py; then
    if [ "$REQUIRE_ATLASES" = "1" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') ERROR: required atlases unavailable; aborting startup" >&2
        exit 1
    fi
    echo "$(date '+%Y-%m-%d %H:%M:%S') WARNING: atlases unavailable; continuing (will download lazily)" >&2
fi

# Start the Panel application
exec uv run panel serve ./app.py \
    --address "$ADDRESS" \
    --port "$INTERNAL_PORT" \
    --allow-websocket-origin "$ALLOW_WEBSOCKET_ORIGIN" \
    --num-procs "$NUM_PROCS" \
    --session-token-expiration 3600000 \
    --check-unused-sessions 10000 \
    --unused-session-lifetime 60000 \
    --show
