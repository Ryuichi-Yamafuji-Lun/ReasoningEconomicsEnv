# Multi-stage build for ReasoningEconomicsEnv OpenEnv environment.
# Build context: project root (ReasoningEconomicsEnv/). Copy into /app/env.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder
ARG BASE_IMAGE

WORKDIR /app

# Copy entire project (pyproject.toml, server/, env/, etc.)
COPY . /app/env

WORKDIR /app/env

# Ensure uv is available
RUN if ! command -v uv >/dev/null 2>&1; then \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install git for building from git repos (e.g. openenv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies; use --frozen only if uv.lock exists
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
    uv sync --frozen --no-install-project --no-editable; \
    else \
    uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
    uv sync --frozen --no-editable; \
    else \
    uv sync --no-editable; \
    fi

RUN echo "=== ReasoningEconomicsEnv Docker build (builder) ===" && \
    echo "BASE_IMAGE=${BASE_IMAGE}" && \
    cd /app/env && uv run python -c "\
from dataclasses import asdict; \
import json; \
from env.config import EnvConfig; \
print('EnvConfig defaults from repo (baked into image):'); \
print(json.dumps(asdict(EnvConfig()), indent=2))"

# Final runtime stage
FROM ${BASE_IMAGE}
ARG BASE_IMAGE

WORKDIR /app

COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env
COPY server/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

RUN echo "=== ReasoningEconomicsEnv Docker build (final) ===" && \
    echo "BASE_IMAGE=${BASE_IMAGE}" && \
    cd /app/env && python -c "\
from dataclasses import asdict; \
import json; \
from env.config import EnvConfig; \
print('EnvConfig defaults (final image):'); \
print(json.dumps(asdict(EnvConfig()), indent=2))"

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
