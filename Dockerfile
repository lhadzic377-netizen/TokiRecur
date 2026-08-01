# Toki Recur Dockerfile
#
# Multi-stage build for production deployment.
# Stage 1: install deps + build the wheel.
# Stage 2: lean runtime image with just Python + the installed package.

# ---- Stage 1: builder ----
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for building wheels (sentence-transformers, hnswlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

# Build wheel and install to a known prefix.
RUN pip install --upgrade pip wheel && \
    pip wheel --no-deps --wheel-dir=/wheels .

# ---- Stage 2: runtime ----
FROM python:3.11-slim AS runtime

# Minimal runtime labels.
LABEL org.opencontainers.image.title="Toki Recur" \
      org.opencontainers.image.description="KV cache reuse for LLM inference" \
      org.opencontainers.image.source="https://github.com/anomalyco/tokirecur" \
      org.opencontainers.image.licenses="MIT"

# Non-root user for runtime.
RUN useradd --create-home --uid 1000 toki
WORKDIR /home/toki

# Copy wheels from builder and install.
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && \
    rm -rf /wheels

USER toki

# Default ports: 8787 (daemon), 7777 (web UI).
EXPOSE 8787 7777

# Default: launch the daemon in foreground.
# Use `toki web` for the chat UI, or `toki up --detach` for both
# llama-server and the daemon.
ENTRYPOINT ["toki"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8787"]
