"""
Initialize MongoDB collections and FAISS novelty index for Research Labs.

Run once:
  python init_research_db.py

Creates:
  - research_sessions    collection with indexes
  - research_candidates  collection with indexes
  - FAISS novelty index file at RESEARCH_ARTIFACTS_DIR/novelty_index.faiss
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(os.environ.get("RESEARCH_ARTIFACTS_DIR", "/research_artifacts"))
MONGO_URI     = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "obsidian")


async def init_mongo():
    """Create MongoDB indexes for research collections."""
    try:
        import motor.motor_asyncio as motor
        client = motor.AsyncIOMotorClient(MONGO_URI)
        db     = client[MONGO_DB_NAME]

        # research_sessions
        await db["research_sessions"].create_index("session_id")
        await db["research_sessions"].create_index("status")
        await db["research_sessions"].create_index("created_at")
        logger.info("research_sessions indexes created")

        # research_candidates
        await db["research_candidates"].create_index("research_session_id")
        await db["research_candidates"].create_index(
            [("research_session_id", 1), ("architecture_name", 1)],
            unique=True,
        )
        await db["research_candidates"].create_index("composite_score")
        await db["research_candidates"].create_index("next_action")
        logger.info("research_candidates indexes created")

        client.close()
    except Exception as e:
        logger.error("MongoDB init failed: %s", e)
        raise


def init_faiss_index():
    """Create an empty FAISS index for novelty scoring (384-dim all-MiniLM-L6-v2)."""
    try:
        import faiss
        import numpy as np

        index_path = ARTIFACTS_DIR / "novelty_index.faiss"
        if index_path.exists():
            logger.info("FAISS novelty index already exists at %s", index_path)
            return

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        dim   = 384   # all-MiniLM-L6-v2 embedding size
        index = faiss.IndexFlatL2(dim)
        faiss.write_index(index, str(index_path))
        logger.info("FAISS novelty index created at %s (dim=%d)", index_path, dim)

    except ImportError:
        logger.warning("faiss not installed — novelty index not created. Install faiss-cpu.")
    except Exception as e:
        logger.error("FAISS init failed: %s", e)
        raise


def init_artifacts_dir():
    """Ensure the research artifacts directory exists."""
    dirs = [
        ARTIFACTS_DIR,
        ARTIFACTS_DIR / "checkpoints",
        ARTIFACTS_DIR / "papers",
        ARTIFACTS_DIR / "docker_jobs",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.info("Directory ready: %s", d)


if __name__ == "__main__":
    logger.info("Initializing Research Labs infrastructure...")
    init_artifacts_dir()
    asyncio.run(init_mongo())
    init_faiss_index()
    logger.info("Initialization complete.")
