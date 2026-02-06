"""SQLite persistence pour JobStore — aiosqlite."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

from api.models import JobStatus

DB_PATH = Path(__file__).resolve().parent.parent / ".audioreader_data" / "jobs.db"

_db: Optional[aiosqlite.Connection] = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    type TEXT NOT NULL DEFAULT 'generate',
    status TEXT NOT NULL DEFAULT 'pending',
    progress REAL NOT NULL DEFAULT 0.0,
    phase TEXT,
    result TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    chapter_index INTEGER,
    chapter_title TEXT,
    segments_done INTEGER NOT NULL DEFAULT 0,
    segments_total INTEGER NOT NULL DEFAULT 0,
    total_chapters INTEGER NOT NULL DEFAULT 0,
    cancelled INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
"""


async def init_db() -> aiosqlite.Connection:
    """Initialise la base de données SQLite."""
    global _db
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _db = await aiosqlite.connect(str(DB_PATH))
    _db.row_factory = aiosqlite.Row
    await _db.executescript(SCHEMA)
    await _db.commit()
    return _db


async def get_db() -> aiosqlite.Connection:
    """Retourne la connexion DB, l'initialise si nécessaire."""
    global _db
    if _db is None:
        await init_db()
    return _db  # type: ignore


def _row_to_dict(row: aiosqlite.Row) -> Dict[str, Any]:
    """Convertit une Row SQLite en dict compatible avec l'ancien format."""
    d = dict(row)
    # Convertir status string en enum
    d["status"] = JobStatus(d["status"])
    # Désérialiser result JSON
    if d.get("result"):
        d["result"] = json.loads(d["result"])
    else:
        d["result"] = None
    d["cancelled"] = bool(d["cancelled"])
    return d


async def db_create_job(job_type: str = "generate") -> str:
    """Crée un nouveau job dans la DB."""
    db = await get_db()
    job_id = str(uuid.uuid4())[:8]
    now = datetime.now().isoformat()
    await db.execute(
        """INSERT INTO jobs (job_id, type, status, progress, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (job_id, job_type, JobStatus.pending.value, 0.0, now, now),
    )
    await db.commit()
    return job_id


async def db_get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Récupère un job par son ID."""
    db = await get_db()
    async with db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)) as cursor:
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None


async def db_list_jobs(
    status: Optional[str] = None,
    offset: int = 0,
    limit: int = 50,
) -> tuple[List[Dict[str, Any]], int]:
    """Liste les jobs avec pagination. Retourne (jobs, total)."""
    db = await get_db()

    where = ""
    params: list = []
    if status:
        where = "WHERE status = ?"
        params.append(status)

    # Count total
    async with db.execute(f"SELECT COUNT(*) FROM jobs {where}", params) as cursor:
        total = (await cursor.fetchone())[0]

    # Fetch page
    async with db.execute(
        f"SELECT * FROM jobs {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
        params + [limit, offset],
    ) as cursor:
        rows = await cursor.fetchall()
        jobs = [_row_to_dict(r) for r in rows]

    return jobs, total


async def db_update_job(job_id: str, **kwargs) -> None:
    """Met à jour un job dans la DB."""
    db = await get_db()
    kwargs["updated_at"] = datetime.now().isoformat()

    # Sérialiser result en JSON si présent
    if "result" in kwargs and kwargs["result"] is not None:
        kwargs["result"] = json.dumps(kwargs["result"], ensure_ascii=False)

    # Convertir status enum en string
    if "status" in kwargs:
        status = kwargs["status"]
        kwargs["status"] = status.value if hasattr(status, "value") else str(status)

    # Convertir cancelled bool en int
    if "cancelled" in kwargs:
        kwargs["cancelled"] = int(kwargs["cancelled"])

    sets = ", ".join(f"{k} = ?" for k in kwargs)
    vals = list(kwargs.values()) + [job_id]
    await db.execute(f"UPDATE jobs SET {sets} WHERE job_id = ?", vals)
    await db.commit()
