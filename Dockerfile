FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
   git \
   build-essential \
   curl \
   && rm -rf /var/lib/apt/lists/*
# Set the working directory
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/root/.cache/uv

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the project files to the working directory

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-dev --frozen --no-install-workspace


COPY . /app

# Install the Python packages using uv
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-dev --frozen

# Pre-download the common mouse and rat atlases, plus the registry index, so
# the running container never has to reach the atlas host.  Under
# brainglobe-atlasapi v3 both atlases together are only ~9 MB of compressed
# OME-Zarr, so this is cheap.
#
# NOT best-effort: prefetch_atlases.py exits non-zero if it cannot warm the
# cache, failing the build.  A cold image is what took the server down in
# v1.2.x -- it pushes the download into the single-threaded Bokeh process at
# runtime.  See the script's docstring for the full story.
ENV PATH="/app/.venv/bin:$PATH"
RUN python /app/scripts/prefetch_atlases.py

# Keep a pristine copy of the warmed cache outside ~/.brainglobe.
#
# Production mounts a Docker volume at /root/.brainglobe to persist atlases
# users download at runtime.  A named volume is seeded from the image *only
# when the volume is empty*, so once it exists it permanently masks whatever
# the image baked in -- and a single deploy of a cold image leaves a cold
# volume that shadows every later image, warm or not.  entrypoint.sh copies
# anything missing out of this seed at startup, which makes that trap
# impossible regardless of the volume's history.
ENV BRAINGLOBE_SEED=/opt/brainglobe-seed
RUN cp -a /root/.brainglobe "$BRAINGLOBE_SEED"

# Expose the port
ENV INTERNAL_PORT=5008
ENV EXTERNAL_PORT=5008
ENV NUM_PROCS=1
ENV ADDRESS=0.0.0.0
ENV ALLOW_WEBSOCKET_ORIGIN=*
EXPOSE ${INTERNAL_PORT}

# Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Copy the Dockerfile to /dockerfile within the container
COPY Dockerfile /Dockerfile

# Health check
HEALTHCHECK CMD curl --fail http://localhost:${INTERNAL_PORT}/

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

