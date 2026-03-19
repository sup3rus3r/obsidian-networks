"""
SafetyChecker — multi-layer validation of LLM-generated training code.

Checks:
1. Python syntax (ast.parse)
2. Blocked shell/eval patterns
3. Import whitelist
4. Code size limit
5. Suspicious forward-method patterns
"""
from __future__ import annotations

import ast
import re

# ── Allowlist ─────────────────────────────────────────────────────────────────

ALLOWED_IMPORTS = {
    # Deep learning
    "tensorflow", "keras", "torch", "torchvision", "torchaudio",
    # Data / numerics
    "numpy", "pandas", "scipy", "statsmodels", "sympy", "xarray",
    # ML / sklearn
    "sklearn", "xgboost", "lightgbm", "catboost", "shap", "optuna",
    # Vision
    "cv2", "PIL", "skimage", "imageio", "albumentations", "timm",
    # Audio
    "librosa", "soundfile",
    # Graph
    "networkx", "torch_geometric",
    # NLP / text
    "transformers", "tokenizers", "datasets", "nltk", "spacy", "gensim",
    # Visualization (no plt.show / plt.savefig in generated code)
    "matplotlib", "seaborn", "plotly",
    # Math / symbolic
    "math", "cmath", "statistics", "decimal", "fractions", "numbers",
    # Standard library safe subset
    "os", "sys", "pathlib", "json", "collections", "typing", "functools",
    "itertools", "datetime", "warnings", "random", "time", "re", "csv",
    "io", "copy", "abc", "enum", "dataclasses", "hashlib", "base64",
    "zipfile", "gzip", "pickle", "pprint", "textwrap", "string",
    "operator", "heapq", "bisect", "array", "queue", "contextlib",
    "gc", "traceback", "inspect", "types", "weakref", "logging",
    # Data engineering
    "pyarrow", "h5py", "joblib", "tqdm", "more_itertools",
    # Tabular
    "faker",
}

# Blocked as standalone calls (not method calls like model.compile())
_BLOCKED_BUILTINS = re.compile(
    r'(?<!\.)(?<!\w)(eval|exec)\s*\(',
    re.MULTILINE,
)

# Blocked anywhere including as attributes
_BLOCKED_ATTRS = re.compile(
    r'\b(os\.system|os\.popen|os\.execve|subprocess\.|__import__)\s*\(',
    re.MULTILINE,
)

_IMPORT_RE = re.compile(
    r'^(?:import\s+([\w]+)|from\s+([\w]+)\s+import)',
    re.MULTILINE,
)

MAX_CODE_CHARS = 150_000


# ── Public API ────────────────────────────────────────────────────────────────

def validate_code(code: str) -> tuple[bool, list[str]]:
    """
    Validate LLM-generated code before execution.

    Returns:
        (is_safe: bool, violations: list[str])
    """
    violations: list[str] = []

    # 1. Size limit
    if len(code) > MAX_CODE_CHARS:
        violations.append(f"Code too large ({len(code):,} chars > {MAX_CODE_CHARS:,} limit)")
        return False, violations

    # 2. Syntax check
    try:
        ast.parse(code)
    except SyntaxError as e:
        violations.append(f"SyntaxError at line {e.lineno}: {e.msg}")
        return False, violations

    # 3. Blocked builtins (eval / exec as standalone calls)
    for m in _BLOCKED_BUILTINS.finditer(code):
        violations.append(f"Blocked call: {m.group().strip()} at char {m.start()}")

    # 4. Blocked system-level patterns
    for m in _BLOCKED_ATTRS.finditer(code):
        violations.append(f"Blocked pattern: {m.group().strip()} at char {m.start()}")

    # 5. Import whitelist
    for m in _IMPORT_RE.finditer(code):
        pkg = m.group(1) or m.group(2)
        if pkg and pkg not in ALLOWED_IMPORTS:
            violations.append(f"Forbidden import: {pkg}")

    # 6. Suspicious forward-method exec/eval
    if re.search(r'def\s+\w+\s*\(.*\):\s*(?:exec|eval)\s*\(', code, re.DOTALL):
        violations.append("Suspicious exec/eval inside function definition")

    is_safe = len(violations) == 0
    return is_safe, violations


def assert_safe(code: str) -> None:
    """Raise ValueError with all violations if code is not safe."""
    is_safe, violations = validate_code(code)
    if not is_safe:
        raise ValueError(f"Generated code failed safety check:\n" + "\n".join(f"  - {v}" for v in violations))
