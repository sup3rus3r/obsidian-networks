import ast
import os
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from celery import Celery

REDIS_URL           = os.environ.get("REDIS_URL", "redis://redis:6379/0")
SESSIONS_DIR        = Path(os.environ.get("SESSIONS_DIR", "/sessions"))
MAX_TRAINING_MINUTES = int(os.environ.get("MAX_TRAINING_MINUTES", "10"))
MAX_OUTPUT_GB        = int(os.environ.get("MAX_OUTPUT_GB", "10"))
MAX_EPOCHS           = int(os.environ.get("MAX_EPOCHS", "200"))

ALLOWED_IMPORTS = {
    "tensorflow", "keras", "numpy", "pandas",
    "sklearn", "scipy", "gymnasium", "gym",
    "matplotlib", "seaborn", "plotly", "statsmodels",
    "os", "sys", "pathlib", "json", "math", "collections", "typing", "functools", "itertools", "datetime",
    "warnings", "random", "time", "re", "csv", "io", "copy", "abc", "enum",
}
# Blocked as standalone function calls only (e.g. eval("..."), exec("..."))
# NOT as method calls — model.compile() is legitimate Keras API
BLOCKED_BUILTINS = {"exec", "eval"}
# Blocked as both standalone and attribute calls
BLOCKED_ATTRS    = {"system", "popen", "execve"}

celery_app = Celery("obsidian_worker", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_serializer  ="json",
    result_serializer="json",
    accept_content   =["json"],
    result_expires   =SESSION_TTL if (SESSION_TTL := int(os.environ.get("SESSION_TTL_HOURS", "4")) * 3600) else 14400,
)


# ── Universal categorical encoding patch ──────────────────────────────────────
#
# This snippet is injected once, right after the first `df = pd.read_*()` call.
# It runs at script execution time and encodes EVERY string/object/category column
# to float32 integer codes — regardless of variable names, pandas version, or
# column types. The target column is excluded only if TARGET_COL is defined.
#
_CAT_ENCODE_SNIPPET = '''\

# ── Universal categorical encoding (injected by Obsidian worker) ─────────────
def _obsidian_encode_cats(df):
    import pandas as _pd
    _target = globals().get("TARGET_COL", None)
    for _c in list(df.columns):
        if _c == _target:
            continue
        if (
            df[_c].dtype == object
            or str(df[_c].dtype) in ("string", "str", "category")
            or hasattr(df[_c], "cat")
            or _pd.api.types.is_string_dtype(df[_c])
            or _pd.api.types.is_object_dtype(df[_c])
        ):
            df[_c] = _pd.Categorical(df[_c]).codes.astype("float32")
    return df
df = _obsidian_encode_cats(df)
# ─────────────────────────────────────────────────────────────────────────────
'''

_ALREADY_PATCHED = re.compile(r"_obsidian_encode_cats|injected by Obsidian worker")


def patch_categorical_encoding(code: str) -> str:
    """Inject universal categorical encoding after the first df = pd.read_*() call.

    Uses the AST to find the exact line number of the read call, then inserts
    the snippet after the complete statement ends. Falls back to a simple regex
    search if AST traversal finds nothing.
    """
    if _ALREADY_PATCHED.search(code):
        return code

    lines = code.splitlines(keepends=True)

    # ── AST-based: find the line of the first df = pd.read_*(...) assignment ──
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None

    insert_after_line: int | None = None  # 1-based line number

    if tree:
        for node in ast.walk(tree):
            # Look for: df = pd.read_csv(...)  →  Assign where value is a Call
            if not isinstance(node, ast.Assign):
                continue
            # Target must be a simple name 'df'
            if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
                continue
            if node.targets[0].id != "df":
                continue
            val = node.value
            # Value must be a Call like pd.read_csv / pd.read_json etc.
            if not isinstance(val, ast.Call):
                continue
            func = val.func
            is_read_call = (
                isinstance(func, ast.Attribute)
                and func.attr.startswith("read_")
                and isinstance(func.value, ast.Name)
                and func.value.id == "pd"
            )
            if not is_read_call:
                continue
            # end_lineno is the last line of the statement (Python 3.8+)
            end_line = getattr(node, "end_lineno", None) or node.lineno
            insert_after_line = end_line
            break  # only patch after the FIRST load

    if insert_after_line is not None:
        idx = insert_after_line  # insert after this 1-based line → index insert_after_line
        return "".join(lines[:idx]) + _CAT_ENCODE_SNIPPET + "".join(lines[idx:])

    # ── Fallback: regex scan for df = pd.read_* ──────────────────────────────
    pattern = re.compile(r"^(\s*df\s*=\s*pd\.read_\w+\s*\()", re.MULTILINE)
    m = pattern.search(code)
    if not m:
        return code  # can't locate load point — leave untouched

    # Walk forward to find the end of the (possibly multi-line) call
    pos = m.start()
    depth = 0
    i = pos
    while i < len(code):
        ch = code[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                # Find end of line
                nl = code.find('\n', i)
                insert_pos = nl + 1 if nl != -1 else len(code)
                return code[:insert_pos] + _CAT_ENCODE_SNIPPET + code[insert_pos:]
        i += 1

    return code


# ── Safe Concatenate guard ─────────────────────────────────────────────────────
#
# Replaces any `layers.Concatenate(...)(some_list)` call where the list variable
# might be empty at runtime with a safe wrapper that skips concatenation when the
# list is empty. Prevents: TypeError: object of type 'NoneType' has no len()
#
_SAFE_CONCAT_SNIPPET = '''
# ── Safe Concatenate helper (injected by Obsidian worker) ────────────────────
def _obsidian_safe_concat(tensors, axis=-1):
    if not tensors:
        return None
    if len(tensors) == 1:
        return tensors[0]
    import keras
    return keras.layers.Concatenate(axis=axis)(tensors)
# ─────────────────────────────────────────────────────────────────────────────
'''

_CONCAT_CALL_RE = re.compile(
    r'layers\.Concatenate\s*\([^)]*\)\s*\((\w+)\)',
)
_ALREADY_SAFE_CONCAT = re.compile(r'_obsidian_safe_concat')


def patch_safe_concatenate(code: str) -> str:
    """Replace layers.Concatenate(...)(list_var) with _obsidian_safe_concat(list_var).

    This prevents crashes when the list is empty (e.g. no categorical columns
    after the worker pre-encodes everything to float32).
    """
    if _ALREADY_SAFE_CONCAT.search(code):
        return code
    if not _CONCAT_CALL_RE.search(code):
        return code

    # Inject helper at top (after first import block)
    # Find first blank line after imports
    lines = code.splitlines(keepends=True)
    insert_idx = 0
    in_imports = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            in_imports = True
            insert_idx = i + 1
        elif in_imports and stripped == '':
            insert_idx = i + 1
            break

    code = ''.join(lines[:insert_idx]) + _SAFE_CONCAT_SNIPPET + ''.join(lines[insert_idx:])

    # Replace all layers.Concatenate(...)(var) → _obsidian_safe_concat(var)
    code = _CONCAT_CALL_RE.sub(
        lambda m: f'_obsidian_safe_concat({m.group(1)})',
        code,
    )
    return code


def validate_code(code: str) -> None:
    """AST-level validation: allowlist imports, block dangerous builtins."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            # e.g. import tensorflow, import numpy as np
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in ALLOWED_IMPORTS:
                    raise ValueError(f"Import '{alias.name}' is not allowed in generated code")
        elif isinstance(node, ast.ImportFrom):
            # e.g. from keras import layers  →  module='keras'
            # e.g. from keras.layers import Dense  →  module='keras.layers'
            module_top = (node.module or "").split(".")[0]
            if module_top and module_top not in ALLOWED_IMPORTS:
                raise ValueError(f"Import from '{node.module}' is not allowed in generated code")
        elif isinstance(node, ast.Call):
            # Block dangerous standalone function calls: eval("..."), exec("...")
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_BUILTINS:
                raise ValueError(f"Call to built-in '{node.func.id}' is not permitted")
            # Block dangerous OS-level method calls: os.system(), os.popen(), os.execve()
            if isinstance(node.func, ast.Attribute) and node.func.attr in BLOCKED_ATTRS:
                raise ValueError(f"Call to '{node.func.attr}' is not permitted")


# Matches Keras verbose=1 epoch lines like:
#   Epoch 3/50
#   45/45 ━━━━━━━━━━ 1s 23ms/step - loss: 0.3421 - accuracy: 0.8712 - val_loss: 0.4102 - val_accuracy: 0.8401
_EPOCH_NUM_RE = re.compile(r"^Epoch\s+(\d+)/(\d+)", re.MULTILINE)

# Individual metric patterns — each searched independently to avoid optional-group matching bugs
_METRIC_LOSS     = re.compile(r"(?<![a-z_])loss:\s*([\d.eE+\-]+)")
_METRIC_ACC      = re.compile(r"(?<![a-z_])accuracy:\s*([\d.eE+\-]+)")
_METRIC_MAE      = re.compile(r"(?<![a-z_])mae:\s*([\d.eE+\-]+)")
_METRIC_VAL_LOSS = re.compile(r"val_loss:\s*([\d.eE+\-]+)")
_METRIC_VAL_ACC  = re.compile(r"val_accuracy:\s*([\d.eE+\-]+)")
_METRIC_VAL_MAE  = re.compile(r"val_mae:\s*([\d.eE+\-]+)")


def _parse_epoch_metrics(line: str, current_epoch: int, total_epochs: int) -> dict | None:
    """Return a metrics dict if `line` contains Keras per-epoch stats, else None."""
    # Step line ends with metric values (contains "loss:" but not "Epoch N/N")
    if "loss:" not in line or _EPOCH_NUM_RE.match(line):
        return None

    def _f(pattern: re.Pattern, s: str) -> float | None:
        m = pattern.search(s)
        try:
            return round(float(m.group(1)), 4) if m else None
        except ValueError:
            return None

    loss     = _f(_METRIC_LOSS,     line)
    accuracy = _f(_METRIC_ACC,      line) or _f(_METRIC_MAE,     line)
    val_loss = _f(_METRIC_VAL_LOSS, line)
    val_acc  = _f(_METRIC_VAL_ACC,  line) or _f(_METRIC_VAL_MAE, line)

    if loss is None and val_loss is None:
        return None

    return {
        "epoch"       : current_epoch,
        "total_epochs": total_epochs,
        "loss"        : loss,
        "accuracy"    : accuracy,
        "val_loss"    : val_loss,
        "val_accuracy": val_acc,
    }


@celery_app.task(bind=True, name="tasks.run_compilation_task")
def run_compilation_task(self, session_id: str) -> dict:
    """Execute the AI-generated model script. Expects one or more .keras files in the output dir."""
    session_dir = SESSIONS_DIR / session_id
    script_path = session_dir / "generated_script.py"
    output_dir  = session_dir / "output"

    if not script_path.exists():
        raise FileNotFoundError("No generated script found for this session")

    code = script_path.read_text()
    validate_code(code)
    code = patch_categorical_encoding(code)
    code = patch_safe_concatenate(code)
    script_path.write_text(code)  # overwrite so subprocess runs the patched version

    self.update_state(state="PROGRESS", meta={"step": "Compiling…", "progress": 5})

    safe_env = {
        "PATH"                   : "/usr/local/bin:/usr/bin:/bin",
        "PYTHONUNBUFFERED"       : "1",
        "HOME"                   : str(session_dir),
        "PYTHONPATH"             : "",
        # Limit TF thread pools to prevent pthread_create() exhaustion
        # TF spawns interop + intraop + data pipeline threads per epoch;
        # without limits these accumulate and exhaust the container's nproc.
        "TF_NUM_INTEROP_THREADS" : "2",
        "TF_NUM_INTRAOP_THREADS" : "2",
        "OMP_NUM_THREADS"        : "4",
    }

    def _apply_limits():
        """Set per-process resource limits (Linux only).
        Note: RLIMIT_AS is intentionally not set — TF spawns many threads each
        needing virtual address space, so AS limits cause pthread_create failures.
        Memory is capped at the Docker container level (mem_limit in compose).
        """
        if sys.platform == "linux":
            import resource
            cpu_soft = MAX_TRAINING_MINUTES * 60
            cpu_hard = cpu_soft + 60
            resource.setrlimit(resource.RLIMIT_CPU,   (cpu_soft, cpu_hard))
            resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_GB * 1024 ** 3, MAX_OUTPUT_GB * 1024 ** 3))

    try:
        proc = subprocess.Popen(
            ["python3", "-u", str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(session_dir),
            env=safe_env,
            preexec_fn=_apply_limits if sys.platform == "linux" else None,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to start subprocess: {e}") from e

    current_epoch = 0
    total_epochs  = 0
    stdout_lines: list[str] = []
    metrics_log:  list[dict] = []
    tf_loaded = False

    try:
        for raw_line in proc.stdout:  # type: ignore[union-attr]
            line = raw_line.rstrip()
            stdout_lines.append(line)

            # TensorFlow startup messages — show loading state until first epoch
            if not tf_loaded and not current_epoch:
                low = line.lower()
                if any(tok in low for tok in ("tensorflow", "keras", "cuda", "gpu", "cpu", "using")):
                    tf_loaded = True
                    self.update_state(state="PROGRESS", meta={
                        "step"    : "Loading TensorFlow…",
                        "progress": 8,
                    })
                    continue

            # Detect "Epoch N/T" header
            em = _EPOCH_NUM_RE.match(line)
            if em:
                current_epoch = int(em.group(1))
                total_epochs  = int(em.group(2))
                if current_epoch == 1:
                    self.update_state(state="PROGRESS", meta={
                        "step"    : "Building model…",
                        "progress": 15,
                    })
                progress = max(18, min(90, int(18 + 72 * (current_epoch - 1) / max(total_epochs, 1))))
                self.update_state(state="PROGRESS", meta={
                    "step"    : f"Epoch {current_epoch}/{total_epochs}",
                    "progress": progress,
                })
                continue

            # Detect metric line
            metrics = _parse_epoch_metrics(line, current_epoch, total_epochs)
            if metrics:
                metrics_log.append(metrics)
                progress = max(18, min(90, int(18 + 72 * current_epoch / max(total_epochs, 1))))
                self.update_state(state="PROGRESS", meta={
                    "step"    : f"Epoch {current_epoch}/{total_epochs}",
                    "progress": progress,
                    "metrics" : metrics,
                })
                continue

            # Post-training output (evaluation, plot generation, etc.)
            # Any non-empty line after we've seen at least one epoch = still working
            if current_epoch and line.strip():
                self.update_state(state="PROGRESS", meta={
                    "step"    : "Evaluating & saving outputs…",
                    "progress": 91,
                })
    except Exception:
        pass

    self.update_state(state="PROGRESS", meta={
        "step"    : "Saving outputs…",
        "progress": 92,
    })

    try:
        proc.wait(timeout=MAX_TRAINING_MINUTES * 60)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError(f"Compilation timed out ({MAX_TRAINING_MINUTES} minute limit)")

    keras_files = list(output_dir.glob("*.keras"))

    if proc.returncode != 0:
        # If the model was saved before the crash (e.g. post-training plot crash),
        # treat as partial success so the user can still download the model + any saved plots.
        if keras_files:
            return {
                "status"      : "partial",
                "models"      : [f.name for f in keras_files],
                "sizes"       : {f.name: f.stat().st_size for f in keras_files},
                "epochs_run"  : current_epoch,
                "epochs_max"  : total_epochs,
                "warning"     : ("\n".join(stdout_lines))[-1000:],
            }
        error_msg = "\n".join(stdout_lines[-50:]) or "Unknown error"
        raise RuntimeError(f"Script failed:\n{error_msg}")

    if not keras_files:
        raise FileNotFoundError("No .keras files produced — ensure the script calls model.save('<name>.keras')")

    return {
        "status"      : "success",
        "models"      : [f.name for f in keras_files],
        "sizes"       : {f.name: f.stat().st_size for f in keras_files},
        "epochs_run"  : current_epoch,
        "epochs_max"  : total_epochs,
    }
