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
    """Stockage des jobs en mémoire avec support SSE."""

    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._listeners: Dict[str, list] = {}  # job_id -> [asyncio.Queue]

    def create(self, job_type: str = "generate") -> str:
        job_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()
        self._jobs[job_id] = {
            "status": JobStatus.pending,
            "progress": 0.0,
            "phase": None,
            "result": None,
            "error": None,
            "type": job_type,
            "created_at": now,
            "updated_at": now,
            "chapter_index": None,
            "chapter_title": None,
            "segments_done": 0,
            "segments_total": 0,
            "total_chapters": 0,
            "cancelled": False,
        }
        return job_id

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._jobs.get(job_id)

    def list_all(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._jobs)

    def update(self, job_id: str, **kwargs):
        """Met à jour un job et notifie les listeners SSE."""
        if job_id not in self._jobs:
            return
        self._jobs[job_id].update(kwargs)
        self._jobs[job_id]["updated_at"] = datetime.now().isoformat()
        # Notifier les listeners SSE
        self._notify(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job["status"] in (JobStatus.completed, JobStatus.failed, JobStatus.cancelled):
            return False
        job["status"] = JobStatus.cancelled
        job["cancelled"] = True
        self._notify(job_id)
        return True

    def is_cancelled(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        return job.get("cancelled", False) if job else False

    # ── SSE ──

    def subscribe(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._listeners.setdefault(job_id, []).append(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue):
        listeners = self._listeners.get(job_id, [])
        if q in listeners:
            listeners.remove(q)

    def _notify(self, job_id: str):
        job = self._jobs.get(job_id)
        if not job:
            return
        for q in self._listeners.get(job_id, []):
            try:
                q.put_nowait(dict(job))
            except asyncio.QueueFull:
                pass

    def make_progress_callback(self, job_id: str) -> Callable:
        """Crée un callback compatible avec ExtendedHQPipeline.process_chapter()."""
        def callback(progress: float, phase: str = "", **extra):
            self.update(job_id, progress=progress, phase=phase, **extra)
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
