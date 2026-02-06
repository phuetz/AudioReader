"""Dépendances partagées — singletons TTS, job store, fichiers."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from api.models import JobStatus

# ── Chemins ──────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_DIR = BASE_DIR / ".audioreader_data"
DATA_DIR.mkdir(exist_ok=True)

UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

PROJECTS_DIR = DATA_DIR / "projects"
PROJECTS_DIR.mkdir(exist_ok=True)

CLONED_VOICES_DIR = DATA_DIR / "cloned_voices"
CLONED_VOICES_DIR.mkdir(exist_ok=True)

# ── Heure de démarrage ──────────────────────────────────────────────────────

START_TIME = time.time()

# ── TTS Engine (lazy singleton) ──────────────────────────────────────────────

_tts_engine = None


def get_tts_engine():
    """Retourne le moteur TTS unifié (lazy init)."""
    global _tts_engine
    if _tts_engine is None:
        try:
            from src.tts_unified import UnifiedTTS
            _tts_engine = UnifiedTTS()
        except ImportError:
            from tts_unified import UnifiedTTS
            _tts_engine = UnifiedTTS()
    return _tts_engine


# ── Job Store ────────────────────────────────────────────────────────────────

class JobStore:
    """Stockage des jobs avec SQLite + SSE pub-sub en mémoire."""

    def __init__(self):
        self._listeners: Dict[str, list] = {}  # job_id -> [asyncio.Queue]
        self._global_listeners: list = []  # SSE global
        self._cache: Dict[str, Dict[str, Any]] = {}  # cache mémoire pour accès rapide

    async def create(self, job_type: str = "generate") -> str:
        from api.database import db_create_job, db_get_job
        job_id = await db_create_job(job_type)
        job = await db_get_job(job_id)
        if job:
            self._cache[job_id] = job
        return job_id

    async def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        if job_id in self._cache:
            return self._cache[job_id]
        from api.database import db_get_job
        job = await db_get_job(job_id)
        if job:
            self._cache[job_id] = job
        return job

    async def list_all(
        self,
        status: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list, int]:
        """Liste les jobs avec pagination. Retourne (jobs, total)."""
        from api.database import db_list_jobs
        return await db_list_jobs(status=status, offset=offset, limit=limit)

    async def update(self, job_id: str, **kwargs):
        """Met à jour un job et notifie les listeners SSE."""
        from api.database import db_update_job
        await db_update_job(job_id, **kwargs)
        # Mettre à jour le cache
        if job_id in self._cache:
            self._cache[job_id].update(kwargs)
            self._cache[job_id]["updated_at"] = datetime.now().isoformat()
        else:
            from api.database import db_get_job
            job = await db_get_job(job_id)
            if job:
                self._cache[job_id] = job
        # Notifier les listeners SSE (per-job + global)
        self._notify(job_id)
        self._notify_global(job_id)

    async def cancel(self, job_id: str) -> bool:
        job = await self.get(job_id)
        if not job:
            return False
        if job["status"] in (JobStatus.completed, JobStatus.failed, JobStatus.cancelled):
            return False
        await self.update(job_id, status=JobStatus.cancelled, cancelled=True)
        return True

    async def is_cancelled(self, job_id: str) -> bool:
        job = await self.get(job_id)
        return job.get("cancelled", False) if job else False

    # ── SSE per-job ──

    def subscribe(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._listeners.setdefault(job_id, []).append(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue):
        listeners = self._listeners.get(job_id, [])
        if q in listeners:
            listeners.remove(q)

    def _notify(self, job_id: str):
        job = self._cache.get(job_id)
        if not job:
            return
        data = dict(job)
        data["job_id"] = job_id
        for q in self._listeners.get(job_id, []):
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                pass

    # ── SSE global ──

    def subscribe_global(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._global_listeners.append(q)
        return q

    def unsubscribe_global(self, q: asyncio.Queue):
        if q in self._global_listeners:
            self._global_listeners.remove(q)

    def _notify_global(self, job_id: str):
        job = self._cache.get(job_id)
        if not job:
            return
        data = dict(job)
        data["job_id"] = job_id
        for q in self._global_listeners:
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                pass

    def make_progress_callback(self, job_id: str) -> Callable:
        """Crée un callback compatible avec ExtendedHQPipeline.process_chapter()."""
        import asyncio as _asyncio

        def callback(progress: float, phase: str = "", **extra):
            try:
                loop = _asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.update(job_id, progress=progress, phase=phase, **extra))
                else:
                    loop.run_until_complete(self.update(job_id, progress=progress, phase=phase, **extra))
            except RuntimeError:
                # Cache-only update if no event loop
                if job_id in self._cache:
                    self._cache[job_id].update({"progress": progress, "phase": phase, **extra})
                    self._notify(job_id)
                    self._notify_global(job_id)
        return callback


# Singleton global
job_store = JobStore()


# ── File Store ───────────────────────────────────────────────────────────────

class FileStore:
    """Gestion des fichiers uploadés."""

    def __init__(self, upload_dir: Path = UPLOAD_DIR):
        self.upload_dir = upload_dir

    def save(self, filename: str, content: bytes) -> tuple[str, Path]:
        """Sauvegarde un fichier uploadé, retourne (file_id, path)."""
        file_id = str(uuid.uuid4())[:8]
        suffix = Path(filename).suffix
        dest = self.upload_dir / f"{file_id}{suffix}"
        dest.write_bytes(content)
        # Sauvegarder les métadonnées
        meta = {
            "id": file_id,
            "original_name": filename,
            "path": str(dest),
            "mime_type": self._guess_mime(suffix),
            "size": len(content),
            "created_at": datetime.now().isoformat(),
        }
        meta_path = self.upload_dir / f"{file_id}.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        return file_id, dest

    def get_path(self, file_id: str) -> Optional[Path]:
        meta_path = self.upload_dir / f"{file_id}.json"
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        p = Path(meta["path"])
        return p if p.exists() else None

    def get_meta(self, file_id: str) -> Optional[Dict[str, Any]]:
        meta_path = self.upload_dir / f"{file_id}.json"
        if not meta_path.exists():
            return None
        return json.loads(meta_path.read_text())

    @staticmethod
    def _guess_mime(suffix: str) -> str:
        mapping = {
            ".md": "text/markdown",
            ".txt": "text/plain",
            ".pdf": "application/pdf",
            ".epub": "application/epub+zip",
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".mp4": "video/mp4",
            ".mkv": "video/x-matroska",
            ".avi": "video/x-msvideo",
            ".mov": "video/quicktime",
            ".webm": "video/webm",
        }
        return mapping.get(suffix.lower(), "application/octet-stream")


file_store = FileStore()
