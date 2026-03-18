import asyncio
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

SESSIONS_DIR = Path(os.environ.get("SESSIONS_DIR", "/sessions"))
SESSION_TTL  = int(os.environ.get("SESSION_TTL_HOURS", "4")) * 3600


@dataclass
class SessionData:
    created_at  : float
    session_dir : Path
    dataset_path: str | None = None
    analysis    : dict | None = None
    environment : dict | None = None
    task_id     : str | None = None
    # Research → Plan → Build state machine (v0.7.0)
    # Phases: idle | researching | planning | approved | building
    phase       : str        = "idle"
    plan_doc    : str | None = None


_sessions: dict[str, SessionData] = {}


def _persist_phase(session: "SessionData") -> None:
    """Write phase and plan_doc to disk so they survive backend restarts."""
    (session.session_dir / "phase.txt").write_text(session.phase)
    plan_path = session.session_dir / "plan.md"
    if session.plan_doc is not None:
        plan_path.write_text(session.plan_doc)
    elif plan_path.exists():
        plan_path.unlink()


def _restore_phase(session: "SessionData") -> None:
    """Re-hydrate phase and plan_doc from disk after a backend restart."""
    phase_path = session.session_dir / "phase.txt"
    if phase_path.exists():
        session.phase = phase_path.read_text().strip()
    plan_path = session.session_dir / "plan.md"
    if plan_path.exists():
        session.plan_doc = plan_path.read_text()


def create_session() -> str:
    sid         = str(uuid.uuid4())
    session_dir = SESSIONS_DIR / sid
    output_dir  = session_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    created_at  = time.time()
    (session_dir / "session_meta.json").write_text(
        json.dumps({"created_at": created_at})
    )
    _sessions[sid] = SessionData(created_at=created_at, session_dir=session_dir)
    return sid


def get_session(sid: str) -> SessionData | None:
    session = _sessions.get(sid)
    if not session:
        # Try to reconstruct from disk (survives backend restarts within TTL)
        session_dir = SESSIONS_DIR / sid
        meta_path   = session_dir / "session_meta.json"
        if meta_path.exists():
            try:
                meta    = json.loads(meta_path.read_text())
                session = SessionData(
                    created_at  = meta["created_at"],
                    session_dir = session_dir,
                    dataset_path= meta.get("dataset_path"),
                )
                _restore_phase(session)
                _sessions[sid] = session
            except Exception:
                return None
        else:
            return None
    if time.time() - session.created_at > SESSION_TTL:
        _delete_session(sid)
        return None
    return session


def session_expires_at(session: SessionData) -> int:
    return int(session.created_at + SESSION_TTL)


def _delete_session(sid: str) -> None:
    _sessions.pop(sid, None)
    shutil.rmtree(SESSIONS_DIR / sid, ignore_errors=True)


async def cleanup_expired_sessions() -> None:
    """Background asyncio task: runs every 5 min, purges expired sessions."""
    while True:
        await asyncio.sleep(300)
        now     = time.time()
        expired = [sid for sid, s in list(_sessions.items()) if now - s.created_at > SESSION_TTL]
        for sid in expired:
            _delete_session(sid)
