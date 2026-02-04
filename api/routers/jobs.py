"""Endpoints jobs : liste, status, SSE stream, annulation."""
from __future__ import annotations

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from api.dependencies import job_store
from api.errors import APIError, ErrorCode
from api.models import JobResponse, JobStatus

router = APIRouter(prefix="/api/v2", tags=["Jobs"])


@router.get("/jobs")
async def list_jobs(status: Optional[str] = None, limit: int = 50):
    """Liste tous les jobs, optionnellement filtrés par status."""
    jobs = job_store.list_all()
    result = []
    for jid, j in jobs.items():
        if status and j["status"] != status:
            continue
        result.append(
            JobResponse(
                job_id=jid,
                status=j["status"],
                progress=j["progress"],
                phase=j.get("phase"),
                result=j.get("result"),
                error=j.get("error"),
                created_at=j.get("created_at", ""),
                updated_at=j.get("updated_at", ""),
            )
        )
    # Plus récents en premier
    result.sort(key=lambda j: j.created_at, reverse=True)
    return result[:limit]


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Status d'un job spécifique."""
    j = job_store.get(job_id)
    if not j:
        raise APIError(ErrorCode.JOB_NOT_FOUND, f"Job {job_id} non trouvé", status_code=404)
    return JobResponse(
        job_id=job_id,
        status=j["status"],
        progress=j["progress"],
        phase=j.get("phase"),
        result=j.get("result"),
        error=j.get("error"),
        created_at=j.get("created_at", ""),
        updated_at=j.get("updated_at", ""),
    )


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request):
    """SSE stream temps réel pour un job."""
    j = job_store.get(job_id)
    if not j:
        raise APIError(ErrorCode.JOB_NOT_FOUND, f"Job {job_id} non trouvé", status_code=404)

    async def event_stream():
        queue = job_store.subscribe(job_id)
        try:
            # Envoyer l'état initial
            initial = job_store.get(job_id)
            if initial:
                yield _format_sse(_make_event(job_id, initial))

            heartbeat_counter = 0
            while True:
                if await request.is_disconnected():
                    break

                try:
                    update = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield _format_sse(_make_event(job_id, update))

                    # Terminer le stream si le job est fini
                    if update.get("status") in (
                        JobStatus.completed,
                        JobStatus.failed,
                        JobStatus.cancelled,
                    ):
                        break
                except asyncio.TimeoutError:
                    # Heartbeat keepalive
                    heartbeat_counter += 1
                    yield f"event: heartbeat\ndata: {json.dumps({'tick': heartbeat_counter})}\n\n"
        finally:
            job_store.unsubscribe(job_id, queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Annule un job en cours."""
    if not job_store.get(job_id):
        raise APIError(ErrorCode.JOB_NOT_FOUND, f"Job {job_id} non trouvé", status_code=404)

    ok = job_store.cancel(job_id)
    if not ok:
        raise APIError(ErrorCode.JOB_ALREADY_CANCELLED, "Job déjà terminé ou annulé")

    return {"success": True, "message": f"Job {job_id} annulé"}


def _make_event(job_id: str, job: dict) -> dict:
    status = job.get("status", "pending")
    if status == JobStatus.pending:
        event_type = "job:pending"
    elif status == JobStatus.processing:
        event_type = "job:progress"
    elif status == JobStatus.completed:
        event_type = "job:completed"
    elif status == JobStatus.failed:
        event_type = "job:failed"
    elif status == JobStatus.cancelled:
        event_type = "job:cancelled"
    else:
        event_type = "job:progress"

    return {
        "event": event_type,
        "data": {
            "job_id": job_id,
            "status": status,
            "progress": job.get("progress", 0),
            "phase": job.get("phase"),
            "chapter_index": job.get("chapter_index"),
            "chapter_title": job.get("chapter_title"),
            "segments_done": job.get("segments_done", 0),
            "segments_total": job.get("segments_total", 0),
            "result": job.get("result"),
            "error": job.get("error"),
        },
    }


def _format_sse(event_data: dict) -> str:
    return f"event: {event_data['event']}\ndata: {json.dumps(event_data['data'], ensure_ascii=False)}\n\n"
