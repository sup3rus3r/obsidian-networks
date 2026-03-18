"""
Per-session FAISS vector store for the Obsidian Networks research pipeline.

- Embedding model : all-MiniLM-L6-v2  (384-dim, ~90 MB, loaded once at startup)
- Index type      : IndexFlatL2  (exact search, no training needed, <1 k chunks/session)
- Persistence     : /sessions/<sid>/vectorstore/index.faiss  +  chunks.json
- Thread safety   : per-session asyncio.Lock (single uvicorn process)
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── Chunking parameters ───────────────────────────────────────────────────────
CHUNK_SIZE    = 400   # words per chunk
CHUNK_OVERLAP = 80    # word overlap between consecutive chunks

# ── Per-session asyncio locks (prevents concurrent index corruption) ──────────
_index_locks: dict[str, asyncio.Lock] = {}


def get_lock(session_id: str) -> asyncio.Lock:
    if session_id not in _index_locks:
        _index_locks[session_id] = asyncio.Lock()
    return _index_locks[session_id]


# ── Lazy embedding model singleton ───────────────────────────────────────────
_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("SentenceTransformer loaded: all-MiniLM-L6-v2 (384-dim)")
    return _model


def warmup() -> None:
    """Pre-load the embedding model at startup to avoid first-request latency."""
    get_model()
    logger.info("Embedding model warm-up complete")


# ── Text chunking ─────────────────────────────────────────────────────────────
def _chunk_text(text: str, source_url: str, source_title: str = "") -> list[dict]:
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end])
        chunks.append({
            "text"   : chunk,
            "source" : source_url,
            "title"  : source_title,
            "chunk_i": len(chunks),
        })
        if end == len(words):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── Path helpers ──────────────────────────────────────────────────────────────
def _vs_dir(session_dir: Path) -> Path:
    d = session_dir / "vectorstore"
    d.mkdir(exist_ok=True)
    return d


def _index_path(session_dir: Path) -> Path:
    return _vs_dir(session_dir) / "index.faiss"


def _chunks_path(session_dir: Path) -> Path:
    return _vs_dir(session_dir) / "chunks.json"


# ── FAISS helpers ─────────────────────────────────────────────────────────────
def _load_or_create_index(session_dir: Path):
    import faiss
    p = _index_path(session_dir)
    if p.exists():
        return faiss.read_index(str(p))
    return faiss.IndexFlatL2(384)


def _save_index(index, session_dir: Path) -> None:
    import faiss
    faiss.write_index(index, str(_index_path(session_dir)))


def _load_chunks(session_dir: Path) -> list[dict]:
    p = _chunks_path(session_dir)
    return json.loads(p.read_text()) if p.exists() else []


def _save_chunks(chunks: list[dict], session_dir: Path) -> None:
    _chunks_path(session_dir).write_text(json.dumps(chunks))


# ── Public API ────────────────────────────────────────────────────────────────
def ingest_text(
    session_dir: Path,
    text: str,
    source_url: str,
    source_title: str = "",
) -> int:
    """Chunk, embed, and add text to the session FAISS index.

    Returns the total chunk count after ingestion.
    Caller must hold the session lock before calling this.
    """
    new_chunks = _chunk_text(text, source_url, source_title)
    if not new_chunks:
        return chunk_count(session_dir)

    model      = get_model()
    embeddings = model.encode([c["text"] for c in new_chunks], normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    index = _load_or_create_index(session_dir)
    index.add(embeddings)
    _save_index(index, session_dir)

    existing = _load_chunks(session_dir)
    existing.extend(new_chunks)
    _save_chunks(existing, session_dir)

    return len(existing)


def query(
    session_dir: Path,
    query_text: str,
    k: int = 6,
) -> list[dict]:
    """Return top-k chunks most similar to query_text, with a relevance score."""
    if not _index_path(session_dir).exists():
        return []

    import faiss
    model = get_model()
    vec   = model.encode([query_text], normalize_embeddings=True)
    vec   = np.array(vec, dtype="float32")

    index    = faiss.read_index(str(_index_path(session_dir)))
    actual_k = min(k, index.ntotal)
    if actual_k == 0:
        return []

    distances, indices = index.search(vec, actual_k)
    chunks  = _load_chunks(session_dir)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        c         = chunks[idx].copy()
        c["score"] = float(round(1.0 / (1.0 + dist), 4))
        results.append(c)
    return results


def chunk_count(session_dir: Path) -> int:
    return len(_load_chunks(session_dir))
