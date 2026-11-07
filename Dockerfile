# ============================================================
# Multi-stage Dockerfile for InfraLens
# ============================================================

# ---- Stage 1: Builder ----
FROM ultralytics/ultralytics:latest AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install from the uv lock so the image matches CI and local exactly.
# --extra-index-url is required: [tool.uv.sources] pins torch to the CPU
# wheel index, so uv.lock records torch==X+cpu, and plain pip cannot
# resolve a +cpu local version from PyPI alone.
# --no-install-project skips building this package itself; the source is copied
# in the runtime stage, so only the dependencies are needed here.
# pip's --prefix layout depends on the base image: Debian's sysconfig 'local'
# scheme nests everything under /install/local, while other images write
# straight into /install. The RUN below flattens it so the runtime stage has
# one path to copy regardless of base image.
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
    && uv export --frozen --no-emit-project --no-hashes --format requirements-txt -o /tmp/req.txt \
    && pip install --no-cache-dir --prefix=/install \
         --extra-index-url https://download.pytorch.org/whl/cpu \
         -r /tmp/req.txt \
    && if [ -d /install/local ]; then \
         cp -a /install/local/. /install/ && rm -rf /install/local; \
       fi

# ---- Stage 2: Runtime ----
FROM ultralytics/ultralytics:latest AS runtime

WORKDIR /app

# Install only runtime system dependencies
# libgl1 and libxcb1 are required by opencv-python-headless at import time:
# without them the container dies with
# "ImportError: libxcb.so.1: cannot open shared object file".
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
# The builder flattens pip's --prefix layout, so /install is always the
# sysconfig root. Copying /install/local instead would fail on base images
# that do not nest under 'local'.
COPY --from=builder /install /usr/local
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy application code
COPY . .

# Set environment variables
ENV PATH=/usr/local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Create directories and set ownership
RUN mkdir -p /app/models /app/logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

EXPOSE 8000

# Default command: run FastAPI backend.
# The app lives at src/app/main.py, so the module path is src.app.main.
# "src.main" does not exist and the container exited immediately with
# ModuleNotFoundError.
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]