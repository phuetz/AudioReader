"""File d'attente de projets — batch processing avec workers parallèles."""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.dependencies import job_store
from api.errors import APIError, ErrorCode
from api.models import JobStatus

router = APIRouter(prefix="/api/v2", tags=["Queue"])

# ── Nombre max de workers parallèles ──
MAX_WORKERS = 2
_semaphore = asyncio.Semaphore(MAX_WORKERS)


class QueuedJob(BaseModel):
    id: str
    file_id: str
    title: str = ""
    config: dict = {}
    priority: int = 0
    status: str = "waiting"  # waiting, processing, completed, failed
    job_id: Optional[str] = None  # linked generation job
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class QueueAddRequest(BaseModel):
    file_id: str = Field(..., min_length=1)
    title: str = ""
    config: dict = {}
    priority: int = Field(default=0, ge=0, le=10)


class QueueReorderRequest(BaseModel):
    order: List[str]  # list of queue item IDs in desired order


class QueueStatus(BaseModel):
    waiting: List[QueuedJob]
    processing: Optional[QueuedJob] = None
    completed: List[QueuedJob]
    failed: List[QueuedJob]
    paused: bool
    total: int
    max_workers: int = MAX_WORKERS


# In-memory queue state
_queue: list[dict] = []
_paused: bool = False
_worker_tasks: list[asyncio.Task] = []


def _get_by_status(status: str) -> list[dict]:
    return sorted(
        [q for q in _queue if q["status"] == status],
        key=lambda q: (-q["priority"], q["created_at"]),
    )


@router.get("/queue")
async def get_queue() -> QueueStatus:
    """État complet de la queue."""
    waiting = _get_by_status("waiting")
    processing_list = _get_by_status("processing")
    processing = QueuedJob(**processing_list[0]) if processing_list else None
    completed = _get_by_status("completed")
    failed = _get_by_status("failed")

    return QueueStatus(
        waiting=[QueuedJob(**q) for q in waiting],
        processing=processing,
        completed=[QueuedJob(**q) for q in completed],
        failed=[QueuedJob(**q) for q in failed],
        paused=_paused,
        total=len(_queue),
        max_workers=MAX_WORKERS,
    )


@router.post("/queue/add")
async def add_to_queue(request: QueueAddRequest) -> QueuedJob:
    """Ajoute un job à la queue."""
    item = {
        "id": str(uuid.uuid4())[:8],
        "file_id": request.file_id,
        "title": request.title or f"Job {len(_queue) + 1}",
        "config": request.config,
        "priority": request.priority,
        "status": "waiting",
        "job_id": None,
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "error": None,
    }
    _queue.append(item)

    # Start worker if not running
    _ensure_worker()

    return QueuedJob(**item)


@router.delete("/queue/{queue_id}")
async def remove_from_queue(queue_id: str) -> dict:
    """Retire un job de la queue (si pas en cours)."""
    global _queue
    item = next((q for q in _queue if q["id"] == queue_id), None)
    if not item:
        raise APIError(ErrorCode.NOT_FOUND, f"Queue item {queue_id} non trouvé", status_code=404)
    if item["status"] == "processing":
        raise APIError(ErrorCode.VALIDATION_ERROR, "Impossible de retirer un job en cours de traitement")
    _queue = [q for q in _queue if q["id"] != queue_id]
    return {"success": True, "message": f"Job {queue_id} retiré de la queue"}


@router.post("/queue/reorder")
async def reorder_queue(request: QueueReorderRequest) -> dict:
    """Réordonne les jobs en attente."""
    waiting = {q["id"]: q for q in _queue if q["status"] == "waiting"}
    for i, qid in enumerate(request.order):
        if qid in waiting:
            waiting[qid]["priority"] = len(request.order) - i  # Higher = first
    return {"success": True, "message": "Queue réordonnée"}


@router.post("/queue/pause")
async def pause_queue() -> dict:
    """Met la queue en pause."""
    global _paused
    _paused = True
    return {"success": True, "paused": True, "message": "Queue en pause"}


@router.post("/queue/resume")
async def resume_queue() -> dict:
    """Reprend le traitement."""
    global _paused
    _paused = False
    _ensure_worker()
    return {"success": True, "paused": False, "message": "Queue reprise"}


@router.post("/queue/clear")
async def clear_completed() -> dict:
    """Supprime les jobs terminés/échoués de la queue."""
    global _queue
    before = len(_queue)
    _queue = [q for q in _queue if q["status"] in ("waiting", "processing")]
    removed = before - len(_queue)
    return {"success": True, "removed": removed}


def _ensure_worker():
    """Démarre le worker si nécessaire."""
    global _worker_tasks
    # Clean done tasks
    _worker_tasks = [t for t in _worker_tasks if not t.done()]
    if not _worker_tasks:
        try:
            loop = asyncio.get_event_loop()
            task = loop.create_task(_worker_loop())
            _worker_tasks.append(task)
        except RuntimeError:
            pass


async def _process_item(item: dict):
    """Traite un item de la queue avec le semaphore."""
    async with _semaphore:
        item["status"] = "processing"
        item["started_at"] = datetime.now().isoformat()

        try:
            # Create a generation job
            gen_job_id = await job_store.create("audiobook")
            item["job_id"] = gen_job_id

            # Read file and generate
            from api.dependencies import file_store
            path = file_store.get_path(item["file_id"])
            if not path:
                raise FileNotFoundError(f"Fichier {item['file_id']} non trouvé")

            text = path.read_text(encoding="utf-8")
            await job_store.update(gen_job_id, status=JobStatus.processing, progress=5, phase="queue_processing")

            tts = None
            try:
                from api.dependencies import get_tts_engine
                tts = get_tts_engine()
            except Exception:
                pass

            if tts:
                config = item.get("config", {})
                voice = config.get("narrator_voice", "ff_siwis")
                speed = config.get("speed", 1.0)
                lang = config.get("language", "fr")

                audio, sr = tts.synthesize(text=text[:2000], voice=voice, speed=speed, lang=lang)

                from api.dependencies import OUTPUT_DIR
                output_name = f"queue_{item['id']}.wav"
                output_path = OUTPUT_DIR / output_name

                import soundfile as sf
                sf.write(str(output_path), audio, sr)

                duration = len(audio) / sr
                await job_store.update(
                    gen_job_id,
                    status=JobStatus.completed,
                    progress=100,
                    phase="done",
                    result={
                        "output_file": output_name,
                        "duration_seconds": round(duration, 2),
                        "download_url": f"/output/{output_name}",
                    },
                )

            item["status"] = "completed"
            item["completed_at"] = datetime.now().isoformat()

        except Exception as e:
            item["status"] = "failed"
            item["error"] = str(e)
            item["completed_at"] = datetime.now().isoformat()
            if item.get("job_id"):
                await job_store.update(item["job_id"], status=JobStatus.failed, error=str(e))


async def _worker_loop():
    """Boucle du worker qui traite la queue avec des workers parallèles."""
    while True:
        if _paused:
            await asyncio.sleep(2)
            continue

        waiting = _get_by_status("waiting")
        if not waiting:
            await asyncio.sleep(2)
            continue

        # Lancer jusqu'à MAX_WORKERS items en parallèle
        tasks = []
        for item in waiting[:MAX_WORKERS]:
            task = asyncio.create_task(_process_item(item))
            tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        await asyncio.sleep(1)
