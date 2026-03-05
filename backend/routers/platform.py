"""
Platform router — anonymous session, dataset upload/preview,
SSE job progress, and file downloads.
"""
import asyncio
import io
import json
import os
import re
import time
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Cookie, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from sessions import (
    SessionData,
    create_session,
    get_session,
    session_expires_at,
)
from tasks import (
    celery_app, run_compilation_task, validate_code,
    patch_keras_mistakes, patch_categorical_encoding,
    patch_safe_concatenate, patch_normalizer_name,
)

try:
    import magic as _magic
    _HAS_MAGIC = True
except ImportError:
    _HAS_MAGIC = False

SESSIONS_DIR         = Path(os.environ.get("SESSIONS_DIR", "/sessions"))
MAX_FILE_BYTES       = int(os.environ.get("MAX_FILE_SIZE_MB", "500")) * 1024 * 1024
SESSION_TTL          = int(os.environ.get("SESSION_TTL_HOURS", "4")) * 3600
MAX_TRAINING_MINUTES = int(os.environ.get("MAX_TRAINING_MINUTES", "10"))
MAX_MEMORY_GB        = int(os.environ.get("MAX_MEMORY_GB", "12"))
MAX_OUTPUT_GB        = int(os.environ.get("MAX_OUTPUT_GB", "10"))
MAX_EPOCHS           = int(os.environ.get("MAX_EPOCHS", "200"))

ALLOWED_EXTENSIONS = {".csv", ".json"}
ALLOWED_MIME_TYPES = {"text/csv", "application/json", "text/plain", "application/octet-stream"}

router = APIRouter(prefix="/platform")


@router.get("/health")
async def platform_health():
    return {"status": "ok"}


@router.get("/limits")
async def platform_limits():
    """Return the current platform resource limits (driven by env vars)."""
    return {
        "max_training_minutes": MAX_TRAINING_MINUTES,
        "max_dataset_mb"      : MAX_FILE_BYTES // (1024 * 1024),
        "max_memory_gb"       : MAX_MEMORY_GB,
        "max_output_gb"       : MAX_OUTPUT_GB,
        "session_ttl_hours"   : SESSION_TTL // 3600,
        "max_epochs"          : MAX_EPOCHS,
    }


@router.post("/session")
async def new_session(response: JSONResponse.__class__ = None):
    """Create a new anonymous session. Returns session info and sets a cookie."""
    from fastapi.responses import JSONResponse as _JSONResponse
    sid     = create_session()
    session = get_session(sid)
    payload = {
        "session_id": sid,
        "created_at": int(session.created_at),
        "expires_at": session_expires_at(session),
    }
    resp = _JSONResponse(content=payload)
    resp.set_cookie(
        key      ="session_id",
        value    =sid,
        httponly =True,
        samesite ="lax",
        secure   =False,       # set to True in production behind HTTPS
        max_age  =SESSION_TTL,
    )
    return resp


@router.get("/session")
async def check_session(session_id: str = Cookie(default=None)):
    """Return current session info or 401 if no valid session."""
    if not session_id:
        raise HTTPException(status_code=401, detail="No session cookie")
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Session expired or not found")
    return {
        "session_id": session_id,
        "created_at": int(session.created_at),
        "expires_at": session_expires_at(session),
    }


def analyse_dataset(df: pd.DataFrame) -> dict:
    """Extract meta-features from a dataset DataFrame."""
    n_rows, n_features = df.shape

    # Detect datetime columns (native dtype + parseable string columns)
    datetime_cols: list[str] = [
        c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    for c in df.select_dtypes(include=["object", "str"]).columns:
        try:
            pd.to_datetime(df[c].dropna().head(100))
            if c not in datetime_cols:
                datetime_cols.append(c)
        except Exception:
            pass
    datetime_detected = len(datetime_cols) > 0

    # For time series: infer the datetime column name and sampling frequency
    datetime_col: str | None = datetime_cols[0] if datetime_cols else None
    ts_frequency: str | None = None
    if datetime_col:
        try:
            parsed = pd.to_datetime(df[datetime_col].dropna())
            inferred = pd.infer_freq(parsed.sort_values().head(50))
            ts_frequency = inferred
        except Exception:
            pass

    # Categorical vs numeric split
    cat_cols = list(df.select_dtypes(include=["object", "str", "category"]).columns)
    fraction_categorical = len(cat_cols) / n_features if n_features > 0 else 0.0

    # Missing value fraction (mean over all columns)
    fraction_missing = float(df.isnull().mean().mean())

    # Cardinality
    max_cardinality = int(max((df[c].nunique() for c in cat_cols), default=0))

    # Average string length — used to detect NLP-style text columns
    avg_string_length = 0.0
    if cat_cols:
        lengths = [float(df[c].dropna().astype(str).str.len().mean()) for c in cat_cols]
        avg_string_length = sum(lengths) / len(lengths)

    # Target column: heuristic — last column is most common convention
    target_col = str(df.columns[-1])
    n_classes   = int(df[target_col].nunique())

    # Class imbalance ratio (majority / minority count)
    class_imbalance_ratio: float | None = None
    if 2 <= n_classes <= 50:
        counts = df[target_col].value_counts()
        if counts.min() > 0:
            class_imbalance_ratio = round(float(counts.max() / counts.min()), 2)

    # Dataset type
    col_names_lower = " ".join(df.columns.str.lower())
    if avg_string_length > 50:
        dataset_type = "nlp"
    elif datetime_detected:
        dataset_type = "time_series"
    elif any(kw in col_names_lower for kw in ("image", " img ", "filepath", "filename", "path")):
        dataset_type = "image"
    else:
        dataset_type = "tabular"

    # Task type — OHLCV fingerprint takes priority (RL trading data)
    ohlcv_keywords = {"open", "high", "low", "close", "volume", "price", "bid", "ask"}
    col_set        = {c.lower() for c in df.columns}
    is_ohlcv       = len(ohlcv_keywords & col_set) >= 3

    if is_ohlcv and datetime_detected:
        task_type = "rl_trading"
    elif n_classes == 2:
        task_type = "binary_classification"
    elif n_classes > 2 and (n_classes <= 20 or not pd.api.types.is_numeric_dtype(df[target_col])):
        task_type = "multiclass_classification"
    else:
        task_type = "regression"

    return {
        "dataset_type"          : dataset_type,
        "task_type"             : task_type,
        "n_rows"                : n_rows,
        "n_features"            : n_features,
        "target_col"            : target_col,
        "n_classes"             : n_classes,
        "class_imbalance_ratio" : class_imbalance_ratio,
        "fraction_categorical"  : round(fraction_categorical, 3),
        "fraction_missing"      : round(fraction_missing, 3),
        "max_cardinality"       : max_cardinality,
        "datetime_detected"     : datetime_detected,
        "datetime_col"          : datetime_col,
        "ts_frequency"          : ts_frequency,
        "avg_string_length"     : round(avg_string_length, 1),
        "columns"               : list(df.columns),
        "dtypes"                : {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


@router.post("/upload/{session_id}")
async def upload_dataset(session_id: str, file: UploadFile = File(...)):
    """Upload a CSV or JSON dataset. Validates type and size."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    # Extension check
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Only .csv and .json files are accepted (got '{ext}')")

    # Stream to disk with size enforcement
    dest_path = session.session_dir / f"dataset{ext}"
    total_size = 0
    chunks: list[bytes] = []

    while True:
        chunk = await file.read(1024 * 1024)  # 1 MB chunks
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail=f"File too large (max {MAX_FILE_BYTES // (1024*1024)} MB)")
        chunks.append(chunk)

    raw_bytes = b"".join(chunks)

    # MIME validation via python-magic (if available)
    if _HAS_MAGIC:
        mime = _magic.from_buffer(raw_bytes[:2048], mime=True)
        if mime not in ALLOWED_MIME_TYPES:
            raise HTTPException(status_code=400, detail=f"Invalid file content type: {mime}")

    # Content validation — try to parse
    try:
        if ext == ".csv":
            df = pd.read_csv(io.BytesIO(raw_bytes))
        else:
            df = pd.read_json(io.BytesIO(raw_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty")
    if len(df.columns) < 2:
        raise HTTPException(status_code=400, detail="Dataset must have at least 2 columns")

    # Save to disk
    dest_path.write_bytes(raw_bytes)
    session.dataset_path = str(dest_path)

    # Run meta-feature extraction and cache the result
    analysis = analyse_dataset(df)
    session.analysis = analysis
    (session.session_dir / "analysis.json").write_text(json.dumps(analysis))

    return {
        "session_id" : session_id,
        "filename"   : file.filename,
        "size_bytes" : total_size,
        "rows"       : len(df),
        "columns"    : len(df.columns),
        "analysis"   : analysis,
    }


@router.get("/preview/{session_id}")
async def preview_dataset(session_id: str):
    """Return schema + first 5 rows of the uploaded dataset."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    if not session.dataset_path:
        raise HTTPException(status_code=404, detail="No dataset uploaded for this session")

    path = Path(session.dataset_path)
    try:
        df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_json(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read dataset: {e}")

    return {
        "columns"    : list(df.columns),
        "dtypes"     : {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample_rows": df.head(5).where(pd.notna(df.head(5)), other=None).to_dict(orient="records"),
        "row_count"  : len(df),
    }


@router.get("/analysis/{session_id}")
async def get_analysis(session_id: str):
    """Return cached meta-feature analysis for the uploaded dataset."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    if not session.analysis:
        raise HTTPException(status_code=404, detail="No analysis available — upload a dataset first")
    return session.analysis


_SETUP_NOTES: dict[str, str] = {
    "cpu": """\
## Environment Setup

### Requirements
- **Linux or WSL2** (Windows Subsystem for Linux)
- **Python 3.10 – 3.12** (TensorFlow does not support Python 3.13+ yet)

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies
```bash
pip install "tensorflow-cpu>=2.16" pandas scikit-learn matplotlib seaborn statsmodels
```

### Place your dataset
Copy `dataset.csv` into the **same directory as this notebook** before running.
""",

    "nvidia_gpu": """\
## Environment Setup

### Requirements
- **Linux or WSL2** with NVIDIA GPU
- NVIDIA drivers + CUDA 12.x installed
- **Python 3.10 – 3.12**

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies (GPU)
```bash
pip install "tensorflow[and-cuda]>=2.16" pandas scikit-learn matplotlib seaborn statsmodels
```

### Verify GPU is detected
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))  # should list your GPU
```

### Place your dataset
Copy `dataset.csv` into the **same directory as this notebook**.
""",

    "google_colab": """\
## Environment Setup

Google Colab already includes TensorFlow. Just install the extra dependencies:

```python
!pip install -q pandas scikit-learn matplotlib seaborn statsmodels
```

### Upload your dataset
Use the Colab file browser (left panel → Files icon) to upload `dataset.csv`,
or mount Google Drive and point `DATA_PATH` to the correct path.
""",
}

_PIP_INSTALL: dict[str, str] = {
    "cpu"         : 'pip install "tensorflow-cpu>=2.16" pandas scikit-learn matplotlib seaborn statsmodels',
    "nvidia_gpu"  : 'pip install "tensorflow[and-cuda]>=2.16" pandas scikit-learn matplotlib seaborn statsmodels',
    "google_colab": "# TensorFlow is pre-installed on Colab\n!pip install -q pandas scikit-learn matplotlib seaborn statsmodels",
}

_SECTION_RE = re.compile(r"^# ── .+ ──+\s*$", re.MULTILINE)


def _code_cell(source: str) -> dict:
    lines = [l + "\n" for l in source.rstrip().split("\n")]
    return {
        "cell_type"     : "code",
        "id"            : uuid.uuid4().hex[:8],
        "metadata"      : {},
        "outputs"       : [],
        "execution_count": None,
        "source"        : lines,
    }


def _md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id"       : uuid.uuid4().hex[:8],
        "metadata" : {},
        "source"   : [source],
    }


def build_notebook(script: str, description: str, hardware: str = "cpu") -> dict:
    """Wrap a Python training script in a .ipynb structure."""
    pip_line   = _PIP_INSTALL.get(hardware, _PIP_INSTALL["cpu"])
    setup_note = _SETUP_NOTES.get(hardware, _SETUP_NOTES["cpu"])

    cells: list[dict] = [
        _md_cell(f"# {description}\n\n*Generated by [Obsidian Networks](https://github.com)*"),
        _md_cell(setup_note),
        _code_cell(pip_line),
    ]

    # Split on our section-header style comments (# ── Name ──)
    headers   = _SECTION_RE.findall(script)
    parts     = _SECTION_RE.split(script)

    if headers:
        if parts[0].strip():
            cells.append(_code_cell(parts[0]))
        for header, content in zip(headers, parts[1:]):
            if content.strip():
                label = header.strip().lstrip("# ─").rstrip(" ─").strip()
                cells.append(_md_cell(f"## {label}"))
                cells.append(_code_cell(content))
    else:
        cells.append(_code_cell(script))

    return {
        "nbformat"      : 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec"   : {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }


def _validate_script(script: str) -> list[str]:
    """Return a list of validation error strings, empty if script looks correct.

    Checks the most critical requirements so the model can fix them before
    the script is saved and the user attempts to compile.
    """
    errors: list[str] = []

    # ── Syntax check ──────────────────────────────────────────────────────────
    import ast as _ast
    try:
        _ast.parse(script)
    except SyntaxError as e:
        errors.append(f"SyntaxError at line {e.lineno}: {e.msg}")
        return errors  # no point checking further if it won't parse

    # ── Security check via existing validator ─────────────────────────────────
    try:
        validate_code(script)
    except ValueError as e:
        errors.append(f"Security violation: {e}")
        return errors

    # ── Required structural elements ──────────────────────────────────────────
    if "model.fit(" not in script and "model.train" not in script and "env.step(" not in script:
        errors.append("Script does not call model.fit() — training loop is missing.")

    if not re.search(r'\.save\s*\(', script):
        errors.append("Script does not call model.save() — no model will be produced.")

    if "dataset.csv" not in script and "dataset.json" not in script:
        errors.append(
            "Script does not reference 'dataset.csv' or 'dataset.json'. "
            "Use DATA_PATH = 'dataset.csv' — do not use the original uploaded filename."
        )

    if re.search(r'\.save\s*\(["\'][^"\']*\.(?:keras|h5)["\']', script):
        # Check if any save path is bare (no directory separator)
        for m in re.finditer(r'\.save\s*\(["\']([^"\']+\.(?:keras|h5))["\']', script):
            path = m.group(1)
            if "/" not in path and "\\" not in path:
                errors.append(
                    f"model.save('{path}') is missing the output/ directory prefix. "
                    f"Use model.save('output/{path}') instead."
                )

    if "normalizer.adapt(" not in script and "norm.adapt(" not in script and \
       re.search(r'Normalization\s*\(', script):
        errors.append(
            "A Normalization layer is defined but .adapt() is never called. "
            "Call normalizer.adapt(X_train) before building the model."
        )

    return errors


@router.post("/notebook/{session_id}")
async def create_notebook(session_id: str, payload: dict):
    """Convert a generated Python script to a .ipynb and save it."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    script      = payload.get("script", "")
    description = payload.get("description", "Training Notebook")
    hardware    = (session.environment or {}).get("hardware", "cpu") if session.environment else "cpu"

    if not script.strip():
        raise HTTPException(status_code=400, detail="Script cannot be empty")

    # Apply all worker patches so the notebook matches what actually ran
    script = patch_keras_mistakes(script)
    script = patch_categorical_encoding(script)
    script = patch_safe_concatenate(script)
    script = patch_normalizer_name(script)

    # Validate — return errors as 422 so the AI tool caller sees them and retries
    validation_errors = _validate_script(script)
    if validation_errors:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Script validation failed. Fix the errors and call create_notebook again.",
                "errors" : validation_errors,
            },
        )

    nb   = build_notebook(script, description, hardware)
    path = session.session_dir / "output" / "training_notebook.ipynb"
    path.write_text(json.dumps(nb, indent=2))

    (session.session_dir / "generated_script.py").write_text(script)

    output_dir = session.session_dir / "output"
    for stale in output_dir.glob("*.keras"):
        stale.unlink()
    for stale in output_dir.iterdir():
        if stale.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".csv"}:
            stale.unlink()

    return {"ok": True, "path": str(path)}


@router.get("/status/{session_id}")
async def artifact_status(session_id: str):
    """Return which downloadable artifacts are ready for this session."""
    from celery.result import AsyncResult
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    output_dir = session.session_dir / "output"
    images = sorted(
        f.name for f in output_dir.iterdir()
        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}
    ) if output_dir.exists() else []

    # Pull epochs_run / epochs_max from the completed Celery task result
    epochs_run: int | None = None
    epochs_max: int | None = None
    if session.task_id:
        try:
            result = AsyncResult(session.task_id, app=celery_app)
            if result.state == "SUCCESS" and isinstance(result.result, dict):
                epochs_run = result.result.get("epochs_run")
                epochs_max = result.result.get("epochs_max")
        except Exception:
            pass

    nb_path = output_dir / "training_notebook.ipynb"
    return {
        "notebook"      : nb_path.exists(),
        "notebook_mtime": nb_path.stat().st_mtime if nb_path.exists() else None,
        "models"        : sorted(f.name for f in output_dir.glob("*.keras")),
        "images"        : images,
        "epochs_run"    : epochs_run,
        "epochs_max"    : epochs_max,
    }


@router.get("/download/{session_id}/image/{filename}")
async def download_image(session_id: str, filename: str):
    """Serve a plot image from the session output directory."""
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in {"png", "jpg", "jpeg", "svg"}:
        raise HTTPException(status_code=400, detail="Not an image file")

    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    image_path = session.session_dir / "output" / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not found")

    media_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "svg": "image/svg+xml"}
    return FileResponse(path=str(image_path), media_type=media_map[ext])


@router.get("/progress-once/{task_id}")
async def progress_once(task_id: str):
    """Single JSON snapshot of a Celery task's current state. Used for polling."""
    from celery.result import AsyncResult
    result = AsyncResult(task_id, app=celery_app)
    info   = result.info if isinstance(result.info, dict) else {}
    error  = info.get("error")
    if result.state == "FAILURE" and not error:
        exc   = result.info
        error = str(exc) if exc is not None else "Compilation failed"
    return {
        "state"   : result.state,
        "progress": info.get("progress", 0),
        "step"    : info.get("step", ""),
        "error"   : error,
        "metrics" : info.get("metrics"),
    }


@router.get("/progress/{task_id}")
async def stream_progress(task_id: str):
    """Server-Sent Events stream for Celery task progress."""
    from celery.result import AsyncResult

    async def event_generator():
        while True:
            result = AsyncResult(task_id, app=celery_app)
            info   = result.info if isinstance(result.info, dict) else {}

            # On FAILURE, result.info is the exception instance — extract its message
            error = info.get("error")
            if result.state == "FAILURE" and not error:
                exc = result.info
                error = str(exc) if exc is not None else "Compilation failed"

            data   = {
                "state"   : result.state,
                "progress": info.get("progress", 0),
                "step"    : info.get("step", ""),
                "error"   : error,
                "metrics" : info.get("metrics"),
            }
            yield f"data: {json.dumps(data)}\n\n"
            if result.state in ("SUCCESS", "FAILURE"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control"   : "no-cache",
            "X-Accel-Buffering": "no",
            "Connection"      : "keep-alive",
        },
    )


@router.get("/download/{session_id}/model/{filename}")
async def download_model(session_id: str, filename: str):
    """Download a compiled .keras model file by name (e.g. actor.keras, critic.keras)."""
    if not filename.endswith(".keras") or "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename — must be a .keras file")

    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    model_path = session.session_dir / "output" / filename
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not yet compiled for this session")

    return FileResponse(
        path      =str(model_path),
        filename  =filename,
        media_type="application/octet-stream",
    )


@router.get("/download/{session_id}/notebook")
async def download_notebook(session_id: str):
    """Download the generated training_notebook.ipynb file."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    notebook_path = session.session_dir / "output" / "training_notebook.ipynb"
    if not notebook_path.exists():
        raise HTTPException(status_code=404, detail="Notebook not yet generated for this session")

    return FileResponse(
        path    =str(notebook_path),
        filename="training_notebook.ipynb",
        media_type="application/octet-stream",
    )


@router.post("/compile/{session_id}")
async def trigger_compilation(session_id: str):
    """Enqueue a model compilation job for this session."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    script_path = session.session_dir / "generated_script.py"
    if not script_path.exists():
        raise HTTPException(status_code=400, detail="No generated script found — run the AI pipeline first")

    task = run_compilation_task.delay(session_id)
    session.task_id = task.id

    return {"task_id": task.id, "status": "queued"}
