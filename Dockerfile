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

# Pre-download the most common mouse and rat atlases so users don't wait for
# multi-hundred-MB downloads on first use.  Stored in ~/.brainglobe/ inside
# the image layer; mount a Docker volume there in production to persist any
# additional atlases users request across container restarts.
# check_latest=False avoids hanging the build if the GIN server is down.
ENV PATH="/app/.venv/bin:$PATH"
RUN python -c "\
from brainglobe_atlasapi import BrainGlobeAtlas; \
BrainGlobeAtlas('allen_mouse_25um', check_latest=False); \
BrainGlobeAtlas('whs_sd_rat_39um', check_latest=False); \
"

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

