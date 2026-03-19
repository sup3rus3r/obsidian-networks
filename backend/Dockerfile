# ── Stage 1: install dependencies ────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System libs required by python-magic and TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libmagic1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Layer 1: heavy ML wheels (TF, torch-based) — cached separately so a timeout only redoes this layer
RUN pip install --no-cache-dir --timeout 300 --prefix=/install \
    "numpy>=1.26.0,<2.0.0" \
    "tensorflow-cpu>=2.16.1,<2.17.0" \
    "scikit-learn>=1.4.0" \
    "scipy>=1.13.0" \
    "pandas>=2.0.0" \
    "xgboost>=2.0.0" \
    "lightgbm>=4.3.0" \
    "catboost>=1.2.0" \
    "sentence-transformers>=3.0.0" \
    "faiss-cpu>=1.8.0"

# Layer 2: everything else
RUN pip install --no-cache-dir --timeout 300 --prefix=/install \
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
    "statsmodels>=0.14.0" \
    "httpx>=0.27.0" \
    "beautifulsoup4>=4.12.0" \
    "feedparser>=6.0.0" \
    "nbformat>=5.10.0" \
    "matplotlib>=3.8.0" \
    "seaborn>=0.13.0" \
    "shap>=0.45.0" \
    "optuna>=3.6.0" \
    "h5py>=3.11.0" \
    "pyarrow>=16.0.0" \
    "openpyxl>=3.1.0" \
    "tqdm>=4.66.0" \
    "joblib>=1.4.0" \
    "imageio>=2.34.0" \
    "Pillow>=10.0.0" \
    "more-itertools>=10.0.0" \
    "gymnasium>=0.29.0" \
    "pypdf>=4.0.0" \
    "arxiv>=2.0.0" \
    "sympy>=1.12" \
    "networkx>=3.0"

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
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--http", "httptools"]
