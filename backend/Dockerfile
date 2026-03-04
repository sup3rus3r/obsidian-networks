# ── Stage 1: install dependencies ────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System libs required by python-magic and TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libmagic1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install all Python deps (pinned via pip, independent of uv.lock)
RUN pip install --no-cache-dir --prefix=/install \
    "fastapi[standard]>=0.128.0" \
    "sqlalchemy>=2.0.0" \
    "bcrypt>=4.0.0" \
    "pydantic>=2.0.0" \
    "motor>=3.7.1" \
    "pycryptodome>=3.20.0" \
    "python-dotenv>=1.0.0" \
    "python-jose[cryptography]>=3.3.0" \
    "slowapi>=0.1.9" \
    "celery[redis]>=5.4.0" \
    "aioredis>=2.0.1" \
    "python-magic>=0.4.27" \
    "pandas>=2.0.0" \
    "numpy>=1.26.0" \
    "scikit-learn>=1.4.0" \
    "tensorflow-cpu>=2.16.1" \
    "httpx>=0.27.0" \
    "beautifulsoup4>=4.12.0" \
    "feedparser>=6.0.0" \
    "nbformat>=5.10.0" \
    "matplotlib>=3.8.0" \
    "seaborn>=0.13.0" \
    "statsmodels>=0.14.0"

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create a restricted user for executing user-generated model code
RUN useradd --system --no-create-home --shell /sbin/nologin mlworker

# Create the sessions directory (overridden by Docker volume mount)
RUN mkdir -p /sessions && chmod 777 /sessions

WORKDIR /app

# Copy application code
COPY . .

EXPOSE 8000

# Default: run the API server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
