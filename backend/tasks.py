import ast
import os
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from celery import Celery

REDIS_URL    = os.environ.get("REDIS_URL", "redis://redis:6379/0")
SESSIONS_DIR = Path(os.environ.get("SESSIONS_DIR", "/sessions"))

ALLOWED_IMPORTS = {
    "tensorflow", "keras", "numpy", "pandas",
    "sklearn", "scipy", "gymnasium", "gym",
    "matplotlib", "seaborn", "plotly", "statsmodels",
    "os", "pathlib", "json", "math", "collections", "typing", "functools", "itertools", "datetime",
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
_EPOCH_NUM_RE  = re.compile(r"^Epoch\s+(\d+)/(\d+)", re.MULTILINE)
_EPOCH_STAT_RE = re.compile(
    r"(?:loss:\s*([\d.]+))?"
    r"(?:.*?accuracy:\s*([\d.]+))?"
    r"(?:.*?val_loss:\s*([\d.]+))?"
    r"(?:.*?val_accuracy:\s*([\d.]+))?",
)


def _parse_epoch_metrics(line: str, current_epoch: int, total_epochs: int) -> dict | None:
    """Return a metrics dict if `line` contains Keras per-epoch stats, else None."""
    # Step line ends with metric values (contains "loss:" but not "Epoch N/N")
    if "loss:" not in line or _EPOCH_NUM_RE.match(line):
        return None
    m = _EPOCH_STAT_RE.search(line)
    if not m or not any(m.groups()):
        return None

    def _f(v: str | None) -> float | None:
        try:
            return round(float(v), 4) if v else None
        except ValueError:
            return None

    return {
        "epoch"       : current_epoch,
        "total_epochs": total_epochs,
        "loss"        : _f(m.group(1)),
        "accuracy"    : _f(m.group(2)),
        "val_loss"    : _f(m.group(3)),
        "val_accuracy": _f(m.group(4)),
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

    self.update_state(state="PROGRESS", meta={"step": "Compiling…", "progress": 5})

    safe_env = {
        "PATH"            : "/usr/local/bin:/usr/bin:/bin",
        "PYTHONUNBUFFERED": "1",
        "HOME"            : str(session_dir),
        "PYTHONPATH"      : "",
    }

    def _apply_limits():
        """Set per-process resource limits (Linux only)."""
        if sys.platform == "linux":
            import resource
            # Max CPU time: 360s (hard kill after 6 min — soft timeout is 5 min)
            resource.setrlimit(resource.RLIMIT_CPU, (360, 400))
            # Max output file size: 2 GB (prevents disk-fill attacks)
            resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024 ** 3, 2 * 1024 ** 3))
            # Max address space: 6 GB (TF needs headroom but cap runaway allocations)
            resource.setrlimit(resource.RLIMIT_AS, (6 * 1024 ** 3, 6 * 1024 ** 3))

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
    tf_loaded     = False

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
                    "step"   : f"Epoch {current_epoch}/{total_epochs}",
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
    except Exception:
        pass

    try:
        proc.wait(timeout=300)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError("Compilation timed out (5 minute limit)")

    if proc.returncode != 0:
        error_msg = "\n".join(stdout_lines)[-2000:] or "Unknown error"
        raise RuntimeError(f"Script failed:\n{error_msg}")

    keras_files = list(output_dir.glob("*.keras"))
    if not keras_files:
        raise FileNotFoundError("No .keras files produced — ensure the script calls model.save('<name>.keras')")

    return {
        "status"      : "success",
        "models"      : [f.name for f in keras_files],
        "sizes"       : {f.name: f.stat().st_size for f in keras_files},
        "epochs_run"  : current_epoch,
        "epochs_max"  : total_epochs,
    }
