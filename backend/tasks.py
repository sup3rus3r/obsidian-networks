import ast
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

from celery import Celery

REDIS_URL           = os.environ.get("REDIS_URL", "redis://redis:6379/0")
SESSIONS_DIR        = Path(os.environ.get("SESSIONS_DIR", "/sessions"))
MAX_TRAINING_MINUTES = int(os.environ.get("MAX_TRAINING_MINUTES", "10"))
MAX_OUTPUT_GB        = int(os.environ.get("MAX_OUTPUT_GB", "10"))
MAX_EPOCHS           = int(os.environ.get("MAX_EPOCHS", "200"))

ALLOWED_IMPORTS = {
    # Deep learning
    "tensorflow", "keras", "torch", "torchvision", "torchaudio",
    "tensorflow_datasets",
    # Data / numerics
    "numpy", "pandas", "scipy", "statsmodels", "sympy", "xarray",
    # ML / sklearn ecosystem
    "sklearn", "xgboost", "lightgbm", "catboost", "shap", "optuna",
    # RL
    "gymnasium", "gym", "stable_baselines3",
    # Visualization
    "matplotlib", "seaborn", "plotly", "bokeh", "altair", "PIL", "cv2",
    # NLP / text
    "transformers", "tokenizers", "datasets", "nltk", "spacy", "gensim",
    # Image / audio
    "skimage", "imageio", "librosa", "soundfile",
    # Utilities
    "os", "sys", "pathlib", "json", "math", "collections", "typing",
    "functools", "itertools", "datetime", "warnings", "random", "time",
    "re", "csv", "io", "copy", "abc", "enum", "dataclasses", "struct",
    "hashlib", "base64", "urllib", "http", "zipfile", "tarfile", "gzip",
    "pickle", "shelve", "sqlite3", "pprint", "textwrap", "string",
    "operator", "heapq", "bisect", "array", "queue", "threading",
    "multiprocessing", "concurrent", "contextlib", "gc", "traceback",
    "inspect", "types", "weakref", "numbers", "decimal", "fractions",
    "statistics", "cmath", "logging",
    # Tabular / data engineering
    "pyarrow", "fastparquet", "openpyxl", "xlrd", "h5py", "tables",
    "joblib", "tqdm", "more_itertools",
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
        _dtype = df[_c].dtype
        # Detect non-numeric: use pandas helpers only — numpy issubdtype cannot
        # interpret pandas extension types (StringDtype, ArrowDtype, etc.) and raises.
        try:
            _is_numeric = _pd.api.types.is_numeric_dtype(df[_c])
        except Exception:
            _is_numeric = False
        if not _is_numeric:
            try:
                df[_c] = _pd.Categorical(df[_c].astype(str)).codes.astype("float32")
            except Exception:
                pass  # leave column as-is; to_numpy will raise a clear error
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

    Skipped for RL scripts — they manage their own state and the injection would
    land inside a class/method body, corrupting indentation.
    """
    if _ALREADY_PATCHED.search(code):
        return code

    # Skip RL scripts entirely
    _rl_re = re.compile(
        r'\benv\.step\s*\(|\benv\.reset\s*\(|\bgymnasium\b|\bgym\.Env\b'
        r'|\bGradientTape\b|\bPPO\b|\bDQN\b|\bSAC\b'
        r'|class\s+\w+\s*\(\s*(?:gymnasium\.Env|gym\.Env)',
    )
    if _rl_re.search(code):
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


_DF_NONE_GUARD = '''\
if df is None:
    raise RuntimeError(
        "Dataset failed to load — df is None. "
        "Check that dataset.csv exists in the working directory and that the "
        "load function returns a DataFrame."
    )
'''


def patch_synthetic_data_fallback(code: str) -> str:
    """Remove synthetic-data fallback blocks that hide real load failures.

    LLMs sometimes generate:
        if df is None:
            df = pd.DataFrame(np.random....)   # synthetic fallback
    or similar try/except blocks that swallow load errors and substitute fake
    data.  These must be removed so the real error surfaces.

    Strategy: use AST to find If nodes whose test is `df is None` (or
    `df is None or len(df) == 0` etc.) and whose body contains a DataFrame
    construction (pd.DataFrame, np.random, range).  Replace those blocks with
    a hard raise.
    """
    # Fast path — most scripts won't have this
    if 'synthetic' not in code and ('df is None' not in code and 'df == None' not in code):
        return code

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    lines = code.splitlines(keepends=True)
    # Collect line ranges of synthetic fallback if-blocks to replace
    replacements: list[tuple[int, int]] = []  # (start_lineno, end_lineno) 1-based inclusive

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        # Check test involves `df is None` or similar
        test_src = ast.unparse(node.test)
        if 'df' not in test_src or ('None' not in test_src and 'empty' not in test_src and 'len' not in test_src):
            continue
        # Check body contains synthetic data construction
        body_src = '\n'.join(ast.unparse(n) for n in node.body)
        if not any(kw in body_src for kw in ('DataFrame', 'random', 'synthetic', 'np.zeros', 'np.ones', 'range(')):
            continue
        start = node.lineno
        end   = getattr(node, 'end_lineno', node.lineno)
        replacements.append((start, end))

    if not replacements:
        return code

    # Replace each block (process in reverse so line numbers stay valid)
    for start, end in sorted(replacements, reverse=True):
        replacement = (
            'if df is None:\n'
            '    raise RuntimeError(\n'
            '        "Dataset failed to load — refusing to substitute synthetic data. "\n'
            '        "Ensure dataset.csv is present and the load function returns a DataFrame."\n'
            '    )\n'
        )
        lines[start - 1 : end] = [replacement]

    return ''.join(lines)


def patch_load_data_missing_return(code: str) -> str:
    """Fix load_data() / load_dataset() functions that read a file but forget to return df.

    Skipped for RL scripts — they use load functions intentionally without top-level returns.

    A common weak-model mistake:
        def load_data(path):
            df = pd.read_csv(path)
            df = df.dropna()
            # no return!

    Uses AST: finds FunctionDef nodes whose name contains 'load' or 'data',
    whose body contains a pd.read_* call assigned to a local name, and whose
    last statement is NOT a Return.  Appends `return <varname>`.
    """
    _rl_re = re.compile(
        r'\benv\.step\s*\(|\benv\.reset\s*\(|\bgymnasium\b|\bgym\.Env\b'
        r'|\bGradientTape\b|\bPPO\b|\bDQN\b|\bSAC\b'
        r'|class\s+\w+\s*\(\s*(?:gymnasium\.Env|gym\.Env)',
    )
    if _rl_re.search(code):
        return code

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    lines = code.splitlines(keepends=True)
    inserts: list[tuple[int, str]] = []  # (after_lineno, text)

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        name = node.name.lower()
        if not any(k in name for k in ('load', 'data', 'read', 'fetch')):
            continue

        # Find a local var assigned from pd.read_*
        read_var: str | None = None
        for stmt in ast.walk(node):
            if not isinstance(stmt, ast.Assign):
                continue
            if not (len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
                continue
            val = stmt.value
            if (isinstance(val, ast.Call)
                    and isinstance(val.func, ast.Attribute)
                    and val.func.attr.startswith('read_')
                    and isinstance(val.func.value, ast.Name)
                    and val.func.value.id == 'pd'):
                read_var = stmt.targets[0].id

        if read_var is None:
            continue

        # Check last statement of function body is not a Return
        last = node.body[-1]
        if isinstance(last, ast.Return):
            continue

        indent = '    '  # function body indentation
        after_line = getattr(last, 'end_lineno', last.lineno)
        inserts.append((after_line, f'{indent}return {read_var}\n'))

    # Apply in reverse order
    for after_line, text in sorted(inserts, key=lambda x: x[0], reverse=True):
        lines.insert(after_line, text)

    return ''.join(lines)


def patch_df_none_guard(code: str) -> str:
    """Inject a None-check after the first `df = ...` assignment.

    Catches cases where the LLM wraps pd.read_csv in a helper that forgets to
    return, or uses a file path that silently returns None.

    Skipped for RL scripts — df is loaded inside class/method scope.
    """
    if 'df is None' in code or 'df == None' in code:
        return code  # already guarded (possibly by patch_synthetic_data_fallback)

    _rl_re = re.compile(
        r'\benv\.step\s*\(|\benv\.reset\s*\(|\bgymnasium\b|\bgym\.Env\b'
        r'|\bGradientTape\b|\bPPO\b|\bDQN\b|\bSAC\b'
        r'|class\s+\w+\s*\(\s*(?:gymnasium\.Env|gym\.Env)',
    )
    if _rl_re.search(code):
        return code

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    insert_after_line: int | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        if node.targets[0].id != 'df':
            continue
        insert_after_line = getattr(node, 'end_lineno', None) or node.lineno
        break  # first assignment only

    if insert_after_line is None:
        return code

    lines = code.splitlines(keepends=True)
    idx = insert_after_line  # 1-based → slice index after that line
    return ''.join(lines[:idx]) + _DF_NONE_GUARD + ''.join(lines[idx:])


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


def patch_normalizer_name(code: str) -> str:
    """Fix inconsistent Normalization layer variable names using AST.

    Weak models sometimes assign the layer to e.g. 'ormalizer' then call
    'normalizer.adapt(...)' — causing a NameError at runtime. This patch:
    1. Finds the variable name actually assigned keras.layers.Normalization()
    2. Finds the variable name used in .adapt() calls
    3. If they differ, renames all occurrences of the adapt-caller to the
       assigned name (or vice-versa, picking the canonical 'normalizer').
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    # Find all assignments: <name> = keras.layers.Normalization(...)
    assigned_names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        val = node.value
        if not isinstance(val, ast.Call):
            continue
        func = val.func
        is_norm = (
            (isinstance(func, ast.Attribute) and func.attr == 'Normalization')
        )
        if is_norm:
            assigned_names.append(node.targets[0].id)

    if not assigned_names:
        return code

    assigned = assigned_names[0]

    # Find all <name>.adapt(...) call targets
    adapt_targets: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == 'adapt':
            if isinstance(func.value, ast.Name):
                adapt_targets.append(func.value.id)

    if not adapt_targets:
        return code

    adapt_name = adapt_targets[0]

    # If they match, nothing to do
    if assigned == adapt_name:
        return code

    # assigned name is ground truth (it's what was actually created).
    # Rename all uses of the wrong adapt-caller to match.
    code = re.sub(rf'\b{re.escape(adapt_name)}\b', assigned, code)
    return code


_LIVE_DATA_IMPORTS = re.compile(
    r'^\s*(?:import\s+(?:MetaTrader5|mt5|yfinance|ccxt|alpaca|binance|ib_insync|tvdatafeed|twelvedata)|'
    r'from\s+(?:MetaTrader5|mt5|yfinance|ccxt|alpaca|binance|ib_insync|tvdatafeed|twelvedata)\b)',
    re.MULTILINE,
)

_LIVE_DATA_REPLACEMENT = '''\
# [obsidian patch] live-data import removed — reading uploaded dataset instead
df = pd.read_csv("dataset.csv")
'''


_COL_PRINT_MARKER = "_obsidian_col_print"

def patch_column_names_print(code: str) -> str:
    """Inject a print(<var>.columns.tolist()) after the first pd.read_*() call.

    This ensures every run outputs the actual column names at the top of stdout,
    so if the script crashes on a KeyError the error message already contains
    the real column list — making it trivial for the model to fix.

    Skipped for RL scripts — they don't have a top-level DataFrame variable.
    """
    if _COL_PRINT_MARKER in code:
        return code

    # Skip RL scripts — injecting df.columns would NameError if the var isn't 'df'
    _rl_re = re.compile(
        r'\benv\.step\s*\(|\benv\.reset\s*\(|\bgymnasium\b|\bgym\.Env\b'
        r'|\bGradientTape\b|\bPPO\b|\bDQN\b|\bSAC\b'
        r'|class\s+\w+\s*\(\s*(?:gymnasium\.Env|gym\.Env)',
    )
    if _rl_re.search(code):
        return code

    # Find insert point: after the first <varname> = pd.read_*() call
    # Capture the assigned variable name so we use the right name in the print.
    pattern = re.compile(r'^(\s*)(\w+)\s*=\s*pd\.read_\w+\s*\(', re.MULTILINE)
    m = pattern.search(code)
    if not m:
        return code

    var_name = m.group(2)  # actual variable name (e.g. 'df', 'df_raw', 'data')

    # Walk forward to end of the statement (matching parens)
    pos = m.start()
    depth = 0
    i = pos
    while i < len(code):
        if code[i] == '(':
            depth += 1
        elif code[i] == ')':
            depth -= 1
            if depth == 0:
                nl = code.find('\n', i)
                insert_pos = nl + 1 if nl != -1 else len(code)
                indent = m.group(1)
                snippet = (
                    f"{indent}print(f'[obsidian] columns ({_COL_PRINT_MARKER}):', "
                    f"list({var_name}.columns))\n"
                    f"{indent}print(f'[obsidian] shape:', {var_name}.shape)\n"
                )
                return code[:insert_pos] + snippet + code[insert_pos:]
        i += 1
    return code


def patch_live_data_sources(code: str) -> str:
    """Replace live market-data API calls with pd.read_csv("dataset.csv").

    Models sometimes generate scripts that fetch live data from MT5, yfinance,
    ccxt, etc. These APIs are not installed and have no network access in the
    sandbox. Replace the entire loading block with a read of the uploaded file.
    """
    if not _LIVE_DATA_IMPORTS.search(code):
        return code

    lines = code.splitlines(keepends=True)
    out: list[str] = []
    injected = False

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Remove live-data import lines
        if _LIVE_DATA_IMPORTS.match(line):
            i += 1
            continue

        # Remove mt5.initialize() / mt5.shutdown() / yf.download() / ccxt calls
        # and any df = mt5.* / df = yf.* assignments
        if re.match(
            r'\s*(?:mt5|yf|yfinance|ccxt|MetaTrader5)\s*[.=(]',
            line,
        ) or re.match(
            r'\s*df\s*=\s*(?:mt5|yf|yfinance|ccxt|MetaTrader5)\b',
            line,
        ):
            if not injected:
                out.append(_LIVE_DATA_REPLACEMENT)
                injected = True
            i += 1
            continue

        out.append(line)
        i += 1

    return ''.join(out)


def patch_dataset_filename(code: str) -> str:
    """Rewrite any pd.read_csv / pd.read_json call whose first string argument
    is not 'dataset.csv' / 'dataset.json' to use the canonical filename.

    The upload handler saves every file as dataset.{ext} regardless of the
    original name, but weak models copy the original filename from the chat
    context (e.g. 'ohlcv_data.csv', 'heart_failure_dataset.csv').
    """
    # Rewrite pd.read_csv("anything.csv") → pd.read_csv("dataset.csv")
    # Skip paths that already have a directory component UNLESS it's output/dataset.csv
    # (output/ is never a valid location for the input dataset — the script created it wrong)
    code = re.sub(
        r'''(pd\.read_csv\s*\(\s*)(['"])(?!dataset\.csv\b)(?![^'"]*[/\\])[^'"]+\.csv\2''',
        r'''\g<1>\g<2>dataset.csv\g<2>''',
        code,
    )
    # Also rewrite output/dataset.csv → dataset.csv (common model mistake)
    code = re.sub(
        r'''(pd\.read_csv\s*\(\s*)(['"])output/dataset\.csv\2''',
        r'''\g<1>\g<2>dataset.csv\g<2>''',
        code,
    )
    # Rewrite pd.read_json("anything.json") → pd.read_json("dataset.json")
    code = re.sub(
        r'''(pd\.read_json\s*\(\s*)(['"])(?!dataset\.json\b)(?![^'"]*[/\\])[^'"]+\.json\2''',
        r'''\g<1>\g<2>dataset.json\g<2>''',
        code,
    )
    # Also fix bare DATA_PATH / FILE_PATH / CSV_PATH string assignments
    # Skip if the value already contains a path separator (derived file reference),
    # but always fix output/dataset.csv → dataset.csv
    code = re.sub(
        r'''((?:DATA_PATH|FILE_PATH|CSV_PATH|data_path|file_path|csv_path)\s*=\s*)(['"])output/dataset\.(csv|json)\2''',
        lambda m: f'{m.group(1)}{m.group(2)}dataset.{m.group(3)}{m.group(2)}',
        code,
    )
    code = re.sub(
        r'''((?:DATA_PATH|FILE_PATH|CSV_PATH|data_path|file_path|csv_path)\s*=\s*)(['"])(?!dataset\.)(?![^'"]*[/\\])[^'"]+\.(csv|json)\2''',
        lambda m: f'{m.group(1)}{m.group(2)}dataset.{m.group(3)}{m.group(2)}',
        code,
    )
    return code


def patch_keras_mistakes(code: str) -> str:
    """Fix common Keras/TF errors produced by weak models. Regex-based, order matters.

    Handles:
    1. matplotlib.use("Agg") before import matplotlib — move import before the call
    2. bare `import keras` → `from tensorflow import keras`
    3. `from keras import X` → `from tensorflow.keras import X`
    4. `keras.X` references → `tensorflow.keras.X` (only if not already prefixed)
    5. plt.show() → plt.close()  (headless environment — show() hangs)
    6. model.save("name.keras") missing output/ prefix → model.save("output/name.keras")
       (only for bare filenames, not already-prefixed paths)
    7. model.save("name.h5") → same output/ prefix fix
    """
    # ── 1. matplotlib.use("Agg") before import matplotlib ────────────────────
    # Pattern: matplotlib.use(...) appears before any `import matplotlib` line.
    # Fix: ensure `import matplotlib` appears immediately before the .use() call.
    use_re = re.compile(r'^([ \t]*)matplotlib\.use\s*\(["\']Agg["\']\)', re.MULTILINE)
    imp_re = re.compile(r'^[ \t]*import matplotlib\b', re.MULTILINE)
    if use_re.search(code) and not imp_re.search(code):
        # No import matplotlib at all — insert one before the .use() call
        code = use_re.sub(r'import matplotlib\n\1matplotlib.use("Agg")', code)
    else:
        # import exists but may be after the .use() call — reorder
        lines = code.splitlines(keepends=True)
        use_lineno   = next((i for i, l in enumerate(lines) if use_re.search(l.rstrip())), None)
        imp_lineno   = next((i for i, l in enumerate(lines) if imp_re.search(l.rstrip())), None)
        if use_lineno is not None and imp_lineno is not None and imp_lineno > use_lineno:
            # Move the import line to just before the .use() line
            imp_line = lines.pop(imp_lineno)
            lines.insert(use_lineno, imp_line)
            code = ''.join(lines)

    # ── 2. `import keras` → `from tensorflow import keras` ───────────────────
    # Only bare `import keras` — leave `import keras.layers` etc. for step 3.
    code = re.sub(
        r'^([ \t]*)import\s+keras\s*$',
        r'\1from tensorflow import keras',
        code, flags=re.MULTILINE,
    )

    # ── 3. `from keras import X` → `from tensorflow.keras import X` ──────────
    code = re.sub(
        r'\bfrom\s+keras\b',
        'from tensorflow.keras',
        code,
    )

    # ── 4. `import keras.X` → `import tensorflow.keras.X` ────────────────────
    code = re.sub(
        r'^([ \t]*)import\s+keras\.',
        r'\1import tensorflow.keras.',
        code, flags=re.MULTILINE,
    )

    # ── 5. bare `keras.` attribute access → `tensorflow.keras.` ──────────────
    # Only rewrite `keras.` not already preceded by `tensorflow.` or `.` (alias like tf.keras.)
    code = re.sub(r'(?<!tensorflow\.)(?<!\w)(?<!\.)keras\.', 'tensorflow.keras.', code)

    # ── 5a. fix double-namespace from alias scripts: `tf.tensorflow.keras.` → `tf.keras.`
    code = re.sub(r'\btf\.tensorflow\.keras\.', 'tf.keras.', code)

    # ── 5b. ensure `import tensorflow` is present when tensorflow.keras.* is used
    # After the rewrites above, the script may reference `tensorflow.keras.*` but
    # only have `from tensorflow import keras` — `tensorflow` as a bare name won't
    # be in scope.  Prepend `import tensorflow` if missing.
    if 'tensorflow.keras' in code and not re.search(r'^import\s+tensorflow\s*$', code, re.MULTILINE):
        code = 'import tensorflow\n' + code

    # ── 6. plt.show() → plt.close() ──────────────────────────────────────────
    code = re.sub(r'\bplt\.show\s*\(\s*\)', 'plt.close()', code)

    # ── 7. model.save("bare_name.keras") → model.save("output/bare_name.keras")
    #       Same for .h5. Skip paths that already start with output/ or / or .
    def _fix_save_path(m: re.Match) -> str:
        quote = m.group(1)   # ' or "
        path  = m.group(2)
        # Already has a directory component or is absolute → leave alone
        if '/' in path or '\\' in path or path.startswith('.'):
            return m.group(0)
        return f'.save({quote}output/{path}{quote})'

    code = re.sub(
        r'\.save\((["\'])([^"\']+\.(?:keras|h5))\1\)',
        _fix_save_path,
        code,
    )

    return code


_CLASSIFICATION_LOSSES = {
    "categorical_crossentropy", "sparse_categorical_crossentropy",
    "binary_crossentropy", "CategoricalCrossentropy",
    "SparseCategoricalCrossentropy", "BinaryCrossentropy",
}

_CANONICAL_PLOTS_BLOCK = '''
# ── Obsidian canonical plots (always generated) ───────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os as _os

def _obsidian_plots(history, model, X_test, y_test, task_type):
    """Generate a consistent set of diagnostic plots regardless of model type."""
    _os.makedirs("output", exist_ok=True)
    hist = history.history

    # ── 1. Loss curve ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist["loss"],     label="Train loss",      linewidth=2)
    if "val_loss" in hist:
        ax.plot(hist["val_loss"], label="Val loss", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 2. Accuracy / metric curve ─────────────────────────────────────────────
    _metric_key = next(
        (k for k in hist if k != "loss" and not k.startswith("val_") and k != "lr"),
        None,
    )
    if _metric_key:
        _val_key = f"val_{_metric_key}"
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(hist[_metric_key], label=f"Train {_metric_key}", linewidth=2)
        if _val_key in hist:
            ax.plot(hist[_val_key], label=f"Val {_metric_key}", linewidth=2, linestyle="--")
        ax.set_xlabel("Epoch"); ax.set_ylabel(_metric_key.capitalize())
        ax.set_title(f"Training & Validation {_metric_key.capitalize()}")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"output/{_metric_key}_curve.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ── 3a. Confusion matrix (classification) ─────────────────────────────────
    if task_type == "classification":
        try:
            from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
            _preds = model.predict(X_test, verbose=0)
            if _preds.shape[-1] > 1:
                _y_pred = np.argmax(_preds, axis=1)
                _y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 and y_test.shape[-1] > 1 else np.array(y_test).ravel()
            else:
                _y_pred = (_preds.ravel() >= 0.5).astype(int)
                _y_true = np.array(y_test).ravel()
            _cm = confusion_matrix(_y_true, _y_pred)
            _disp = ConfusionMatrixDisplay(_cm)
            fig, ax = plt.subplots(figsize=(max(5, _cm.shape[0]), max(5, _cm.shape[0])))
            _disp.plot(ax=ax, colorbar=False)
            ax.set_title("Confusion Matrix")
            plt.tight_layout()
            plt.savefig("output/confusion_matrix.png", dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as _e:
            print(f"[obsidian] confusion matrix skipped: {_e}")

    # ── 3b. Actual vs Predicted scatter (regression) ───────────────────────────
    if task_type == "regression":
        try:
            _preds = model.predict(X_test, verbose=0).ravel()
            _true  = np.array(y_test).ravel()
            _mask  = np.isfinite(_preds) & np.isfinite(_true)
            _preds, _true = _preds[_mask], _true[_mask]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            # Scatter
            axes[0].scatter(_true, _preds, alpha=0.4, s=12, color="#39FF14")
            _lo, _hi = min(_true.min(), _preds.min()), max(_true.max(), _preds.max())
            axes[0].plot([_lo, _hi], [_lo, _hi], "r--", linewidth=1.5, label="Perfect fit")
            axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
            axes[0].set_title("Actual vs Predicted")
            axes[0].legend(); axes[0].grid(True, alpha=0.3)
            # Residuals
            _resid = _preds - _true
            axes[1].hist(_resid, bins=40, color="#39FF14", edgecolor="black", alpha=0.7)
            axes[1].axvline(0, color="red", linewidth=1.5, linestyle="--")
            axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Count")
            axes[1].set_title("Residual Distribution")
            axes[1].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("output/predictions.png", dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as _e:
            print(f"[obsidian] predictions plot skipped: {_e}")

try:
    _obsidian_plots(history, model, X_test, y_test, "{task_type}")
except Exception as _plot_err:
    print(f"[obsidian] canonical plots failed: {_plot_err}")
# ─────────────────────────────────────────────────────────────────────────────
'''


def patch_tf_float_cast(code: str) -> str:
    """Fix float64/float32 dtype mismatches in TF operations.

    Weak models compute RL advantages/returns/rewards in numpy (float64) then
    multiply or add them to TF tensors (float32), causing InvalidArgumentError.

    Strategy: inject a cast helper `_to_f32` and wrap common RL variable names
    (advantages, returns, rewards, targets) at the point they are passed to TF
    operations by patching `tf.cast` wrap around numpy-to-tensor conversions and
    `tf.constant` / `tf.convert_to_tensor` calls for those variables.

    A simpler and more reliable approach: after every numpy array is used in a
    `tf.GradientTape` block, the model ops expect float32.  We insert a small
    helper and patch the most common patterns:
      - `tf.constant(var)` / `tf.convert_to_tensor(var)` for known RL names
      - bare variable references multiplied with TF tensors (ratio * adv_batch etc.)
    """
    # Only act on scripts that import tensorflow
    if 'import tensorflow' not in code and 'import tf' not in code:
        return code

    _HELPER = (
        '\ndef _obsidian_f32(x):\n'
        '    import numpy as _np\n'
        '    import tensorflow as _tf\n'
        '    if isinstance(x, _np.ndarray): return x.astype(_np.float32)\n'
        '    try: return _tf.cast(x, _tf.float32)\n'
        '    except Exception: return x\n\n'
    )

    if '_obsidian_f32' in code:
        return code  # already patched

    # Common RL variable name patterns that are typically float64 numpy arrays
    _RL_VARS = r'(?:advantages?|returns?|rewards?|adv_batch|ret_batch|reward_batch|targets?|discounted_rewards?|gae|td_targets?)'

    # Skip patching RL vars inside Obsidian-injected helpers (_TFNormalDist etc.)
    # by only applying regex to lines outside those blocks.
    def _patch_outside_helpers(src: str) -> str:
        out_lines = []
        in_helper = False
        for ln in src.splitlines(keepends=True):
            if ln.strip().startswith('class _TFNormalDist') or ln.strip().startswith('def _obsidian_'):
                in_helper = True
            elif in_helper and ln and not ln[0].isspace():
                in_helper = False
            if in_helper:
                out_lines.append(ln)
            else:
                ln = re.sub(
                    r'\btf\.(convert_to_tensor|constant)\s*\(\s*(' + _RL_VARS + r')\s*\)',
                    r'tf.\1(_obsidian_f32(\2))',
                    ln,
                )
                ln = re.sub(
                    r'(?<!\w)(' + _RL_VARS + r')(?=\s*[\*\+\-\/](?!=))',
                    r'_obsidian_f32(\1)',
                    ln,
                )
                ln = re.sub(
                    r'(?<=[\*\+\-\/]\s)(' + _RL_VARS + r')(?!\w)',
                    r'_obsidian_f32(\1)',
                    ln,
                )
                out_lines.append(ln)
        return ''.join(out_lines)

    code = _patch_outside_helpers(code)

    # Inject helper after the last TOP-LEVEL import only.
    # Indented imports inside Obsidian-injected helpers must be ignored —
    # only match lines with no leading whitespace, and stop at the first
    # injected block to avoid inserting inside it.
    lines = code.splitlines(keepends=True)
    insert_idx = 0
    for i, line in enumerate(lines):
        if 'injected by Obsidian' in line or '_obsidian_encode_cats' in line:
            break
        if (line.startswith('import ') or line.startswith('from ')):
            insert_idx = i + 1
    code = ''.join(lines[:insert_idx]) + _HELPER + ''.join(lines[insert_idx:])

    return code


def patch_tf_distributions(code: str) -> str:
    """Replace tf.distributions.Normal / tensorflow.distributions.Normal (removed in TF2)
    with a minimal inline implementation using TF math.

    tensorflow_probability may not be installed, so we provide a drop-in class
    _TFNormalDist that supports .log_prob() and .sample() via pure TF ops.
    """
    # Patterns to match any variant the model might write
    dist_pattern = re.compile(
        r'\b(?:tensorflow|tf)(?:\.keras)?\.distributions\.Normal\b'
    )
    if not dist_pattern.search(code):
        return code

    helper = '''
class _TFNormalDist:
    """Minimal Normal distribution using pure TensorFlow ops (no tensorflow_probability needed)."""
    import math as _math
    _LOG2PI = _math.log(2 * _math.pi)

    def __init__(self, loc, scale):
        import tensorflow as _tf
        self.loc   = _tf.cast(loc,   _tf.float32)
        self.scale = _tf.cast(scale, _tf.float32)

    def log_prob(self, x):
        import tensorflow as _tf
        x = _tf.cast(x, _tf.float32)
        var = self.scale ** 2
        log_scale = _tf.math.log(self.scale + 1e-8)
        return -0.5 * ((x - self.loc) ** 2 / (var + 1e-8) + self._LOG2PI) - log_scale

    def sample(self):
        import tensorflow as _tf
        return self.loc + self.scale * _tf.random.normal(shape=_tf.shape(self.loc))

    def entropy(self):
        import tensorflow as _tf
        import math as _math
        return 0.5 + 0.5 * _math.log(2 * _math.pi) + _tf.math.log(self.scale + 1e-8)

'''
    # Inject helper after the last import line in the header block
    lines = code.split('\n')
    last_import_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            last_import_idx = i
    lines.insert(last_import_idx + 1, helper)
    code = '\n'.join(lines)

    # Replace all usages
    code = dist_pattern.sub('_TFNormalDist', code)
    return code


def patch_gymnasium_int_cast(code: str) -> str:
    """Fix numpy int64 values from gymnasium being passed to Keras Dense(units=...).

    gymnasium's action_space.n returns a numpy.int64, not a Python int.
    Keras 3 rejects non-Python-int values for the `units` argument.

    Patches any assignment of the form:
        self.<attr> = <expr>.action_space.n
        self.<attr> = env.action_space.n
        n_actions = env.action_space.n
    to wrap the RHS with int(...).
    """
    if 'action_space' not in code:
        return code
    code = re.sub(
        r'(=\s*)(\w[\w.]*\.action_space\.n)\b',
        r'\1int(\2)',
        code,
    )
    return code


def patch_canonical_plots(code: str) -> str:
    """Remove all plt.savefig calls from the generated script and append
    a canonical, deterministic plot block at the end.

    This guarantees every run produces the same set of plots regardless of
    what the model wrote.  The canonical set is:
      - loss_curve.png            (always)
      - <metric>_curve.png        (always, first non-loss metric)
      - confusion_matrix.png      (classification only)
      - predictions.png           (regression only — scatter + residual histogram)

    RL scripts (gymnasium envs, custom training loops) are excluded — they manage
    their own plotting and don't have a Keras History object or model.predict().
    """
    # Skip RL scripts — they have custom training loops without a Keras History object.
    is_rl = bool(re.search(
        r'\benv\.step\s*\(|\benv\.reset\s*\(|\bgymnasium\b|\bgym\.Env\b'
        r'|\bGradientTape\b|\bPPO\b|\bDQN\b|\bSAC\b'
        r'|class\s+\w+\s*\(\s*(?:gymnasium\.Env|gym\.Env)',
        code,
    ))
    if is_rl:
        return code

    # Detect task type from loss function name
    task_type = "regression"
    for loss_name in _CLASSIFICATION_LOSSES:
        if loss_name in code:
            task_type = "classification"
            break

    # Strip existing savefig calls so we don't get duplicates / extra plots.
    # We remove whole lines containing plt.savefig(...) — multi-line savefig
    # calls are rare and not worth the complexity of AST removal.
    code = re.sub(r'[ \t]*plt\.savefig\s*\([^)]*\)\s*\n', '\n', code)

    # Remove seaborn-based plot saves (sns.heatmap(...).get_figure().savefig(...))
    code = re.sub(r'[ \t]*[^#\n]*\.savefig\s*\([^)]*\)\s*\n', '\n', code)

    # Append canonical block with task_type substituted
    code = code.rstrip() + "\n" + _CANONICAL_PLOTS_BLOCK.replace("{task_type}", task_type)
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
    logger.info("[patch] original load block (lines 1-80):\n%s", '\n'.join(code.splitlines()[:80]))
    validate_code(code)
    code = patch_live_data_sources(code)
    code = patch_column_names_print(code)
    code = patch_keras_mistakes(code)
    code = patch_load_data_missing_return(code)
    code = patch_synthetic_data_fallback(code)
    code = patch_categorical_encoding(code)
    code = patch_df_none_guard(code)
    code = patch_safe_concatenate(code)
    code = patch_normalizer_name(code)
    code = patch_tf_distributions(code)
    code = patch_tf_float_cast(code)
    code = patch_gymnasium_int_cast(code)
    code = patch_canonical_plots(code)
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

    def _preexec():
        os.setsid()  # put process in its own process group so killpg kills all children
        _apply_limits()

    try:
        proc = subprocess.Popen(
            ["python3", "-u", str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(session_dir),
            env=safe_env,
            preexec_fn=_preexec if sys.platform == "linux" else None,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to start subprocess: {e}") from e

    current_epoch = 0
    total_epochs  = 0
    stdout_lines: list[str] = []
    metrics_log:  list[dict] = []
    tf_loaded      = False
    preprocessing  = False
    last_detail    = ""   # last meaningful stdout line shown as subtitle in UI

    # Lines that are too noisy/generic to show as detail
    _DETAIL_SKIP = re.compile(
        r'^(=+|-+|─+|╔|╗|║|╚|╝|\s*$|Traceback|File "|  File )'
    )

    def _update(step: str, progress: int, **extra) -> None:
        meta: dict = {"step": step, "progress": progress, **extra}
        if last_detail:
            meta["detail"] = last_detail
        self.update_state(state="PROGRESS", meta=meta)

    # Connect to Redis for stop-flag polling
    try:
        import redis as _redis
        from celery_app import celery_app as _capp
        _stop_redis = _redis.from_url(_capp.conf.broker_url)
        _stop_redis.delete("worker:stop")  # clear any stale flag from previous run
    except Exception:
        _stop_redis = None

    # Background thread: poll Redis stop flag every 0.5s and kill proc if set.
    # This ensures Stop works even when the script is not printing (e.g. saving files).
    import threading
    _done = threading.Event()    # set when stdout loop finishes normally
    _killed = threading.Event()  # set when watcher kills the process

    def _stop_watcher():
        while not _done.is_set():
            if _stop_redis:
                try:
                    if _stop_redis.get("worker:stop"):
                        # Kill entire process group (Python + TF threads + children)
                        try:
                            import signal
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except Exception:
                            proc.kill()  # fallback
                        _stop_redis.delete("worker:stop")
                        _killed.set()
                        return
                except Exception:
                    pass
            _done.wait(0.5)

    _watcher = threading.Thread(target=_stop_watcher, daemon=True)
    _watcher.start()

    try:
        for raw_line in proc.stdout:  # type: ignore[union-attr]
            # Check if stop watcher already killed the process
            if _killed.is_set():
                break

            line = raw_line.rstrip()
            stdout_lines.append(line)

            # Track last meaningful line for UI subtitle
            stripped = line.strip()
            if stripped and not _DETAIL_SKIP.match(line):
                last_detail = stripped[:120]  # cap length

            # TensorFlow startup messages — show loading state until first epoch
            if not tf_loaded and not current_epoch:
                low = line.lower()
                if any(tok in low for tok in ("tensorflow", "keras", "cuda", "gpu", "cpu", "using")):
                    tf_loaded = True
                    _update("Loading TensorFlow…", 8)
                    continue

            # Preprocessing / data loading — detect common script output patterns
            if tf_loaded and not preprocessing and not current_epoch:
                low = line.lower()
                if any(tok in low for tok in (
                    "loading", "loaded", "reading", "dataset", "features",
                    "samples", "rows", "bars", "indicators", "preprocessing",
                    "extracting", "preparing", "train:", "val:", "test:",
                )):
                    preprocessing = True
                    _update("Preprocessing data…", 13)

            # Detect "[STAGE N]" or "[N]" or "[N/N]" style section headers from custom scripts
            stage_m = re.match(r'\[(?:STAGE\s+)?(\d+)(?:/\d+)?\]', line)
            if stage_m:
                stage_num = int(stage_m.group(1))
                if stage_num == 1:
                    preprocessing = True
                    _update("Preprocessing data…", 13)
                elif stage_num == 2:
                    _update("Building model…", 22)
                elif stage_num >= 3:
                    _update("Building model…", 24)
                continue

            # Detect any N/M counter that looks like training progress — script-agnostic
            # Matches: "Epoch 1/50", "Update 1/50", "[EP 0001/500]", "Episode 1/200", "Step 100/1000", etc.
            box_epoch_m = re.search(r'(?:epoch|update|episode|step|ep|iter)[^\d]*(\d+)\s*/\s*(\d+)', line, re.IGNORECASE)
            if box_epoch_m:
                current_epoch = int(box_epoch_m.group(1))
                total_epochs  = int(box_epoch_m.group(2))
                if total_epochs > 1:
                    progress = max(26, min(90, int(26 + 64 * (current_epoch - 1) / max(total_epochs, 1))))
                    _update(f"Training… {current_epoch}/{total_epochs}", progress)
                    continue

            # Generic training keyword detection — script-agnostic fallback
            # If we see training-related keywords after preprocessing, move to Training stage
            if preprocessing and not current_epoch:
                low = line.lower()
                if any(tok in low for tok in (
                    "training", "train loop", "starting ppo", "starting training",
                    "actor loss", "critic loss", "reward", "policy", "rollout",
                    "loss:", "acc:", "accuracy", "val_loss", "val_acc",
                )):
                    current_epoch = 1
                    total_epochs  = 1
                    _update("Training…", 30)

            # Detect "Epoch N/T" header
            em = _EPOCH_NUM_RE.match(line)
            if em:
                current_epoch = int(em.group(1))
                total_epochs  = int(em.group(2))
                if current_epoch == 1:
                    _update("Building model…", 22)
                progress = max(18, min(90, int(18 + 72 * (current_epoch - 1) / max(total_epochs, 1))))
                _update(f"Epoch {current_epoch}/{total_epochs}", progress)
                continue

            # Detect metric line
            metrics = _parse_epoch_metrics(line, current_epoch, total_epochs)
            if metrics:
                metrics_log.append(metrics)
                progress = max(18, min(90, int(18 + 72 * current_epoch / max(total_epochs, 1))))
                _update(f"Epoch {current_epoch}/{total_epochs}", progress, metrics=metrics)
                continue

            # Post-training output (evaluation, plot generation, etc.)
            # Any non-empty line after we've seen at least one epoch = still working
            if current_epoch and line.strip():
                _update("Evaluating & saving outputs…", 91)
    except Exception:
        pass
    finally:
        _done.set()   # signal the background watcher thread to exit

    if _killed.is_set():
        # Watcher killed the process — drain and return
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        return {"stopped": True}

    self.update_state(state="PROGRESS", meta={
        "step"    : "Saving outputs…",
        "progress": 92,
    })

    try:
        proc.wait(timeout=MAX_TRAINING_MINUTES * 60)
    except subprocess.TimeoutExpired:
        try:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()
        err = f"Compilation timed out ({MAX_TRAINING_MINUTES} minute limit)"
        self.update_state(state="FAILURE", meta={"error": err, "exc_type": "RuntimeError", "exc_message": err})
        raise RuntimeError(err)

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
        # Always include the column-names line (injected near the top) so the model
        # knows the actual column names when it sees a KeyError.
        col_lines = [l for l in stdout_lines if '_obsidian_col_print' in l or '[obsidian]' in l]
        tail_lines = stdout_lines[-50:]
        combined = col_lines + ([] if col_lines and col_lines[-1] in tail_lines else tail_lines)
        error_msg = "\n".join(dict.fromkeys(combined)) or "Unknown error"
        full_error = f"Script failed:\n{error_msg}"
        self.update_state(state="FAILURE", meta={"error": full_error, "exc_type": "RuntimeError", "exc_message": full_error})
        raise RuntimeError(full_error)

    if not keras_files:
        raise FileNotFoundError("No .keras files produced — ensure the script calls model.save('<name>.keras')")

    return {
        "status"      : "success",
        "models"      : [f.name for f in keras_files],
        "sizes"       : {f.name: f.stat().st_size for f in keras_files},
        "epochs_run"  : current_epoch,
        "epochs_max"  : total_epochs,
    }
