"""Mode Révision — édition segment-par-segment après génération."""
from __future__ import annotations

import uuid
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

from api.dependencies import OUTPUT_DIR, get_tts_engine, job_store
from api.errors import APIError, ErrorCode
from api.models import JobStatus

router = APIRouter(prefix="/api/v2", tags=["Review"])


class ReviewSegment(BaseModel):
    id: str
    index: int
    text: str
    audio_url: str
    duration: float
    speaker: Optional[str] = None
    emotion: Optional[str] = None
    voice_id: str = "ff_siwis"


class RegenerateOptions(BaseModel):
    voice_id: Optional[str] = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    emotion: Optional[str] = None
    language: str = "fr"


class UpdateSegmentText(BaseModel):
    text: str


# In-memory segment store (keyed by job_id)
_segments_store: dict[str, list[dict]] = {}


@router.get("/jobs/{job_id}/segments")
async def get_job_segments(job_id: str) -> dict:
    """Liste tous les segments audio d'un job."""
    job = await job_store.get(job_id)
    if not job:
        raise APIError(ErrorCode.NOT_FOUND, f"Job {job_id} non trouvé", status_code=404)

    segments = _segments_store.get(job_id, [])
    return {
        "job_id": job_id,
        "segments": [ReviewSegment(**s) for s in segments],
        "total": len(segments),
    }


@router.post("/jobs/{job_id}/segments/{segment_id}/regenerate")
async def regenerate_segment(
    job_id: str, segment_id: str, options: RegenerateOptions, background_tasks: BackgroundTasks
) -> dict:
    """Régénère un segment avec de nouvelles options."""
    segments = _segments_store.get(job_id)
    if not segments:
        raise APIError(ErrorCode.NOT_FOUND, f"Segments pour job {job_id} non trouvés", status_code=404)

    seg = next((s for s in segments if s["id"] == segment_id), None)
    if not seg:
        raise APIError(ErrorCode.NOT_FOUND, f"Segment {segment_id} non trouvé", status_code=404)

    regen_job_id = await job_store.create("regenerate")

    async def process():
        try:
            await job_store.update(regen_job_id, status=JobStatus.processing, progress=10, phase="regenerating")
            tts = get_tts_engine()

            voice = options.voice_id or seg["voice_id"]
            audio, sr = tts.synthesize(
                text=seg["text"],
                voice=voice,
                speed=options.speed,
                lang=options.language,
            )

            output_name = f"seg_{job_id}_{segment_id}.wav"
            output_path = OUTPUT_DIR / output_name

            import soundfile as sf
            sf.write(str(output_path), audio, sr)

            duration = len(audio) / sr
            seg["audio_url"] = f"/output/{output_name}"
            seg["duration"] = round(duration, 2)
            seg["voice_id"] = voice

            await job_store.update(
                regen_job_id,
                status=JobStatus.completed,
                progress=100,
                phase="done",
                result={"segment_id": segment_id, "audio_url": seg["audio_url"], "duration": seg["duration"]},
            )
        except Exception as e:
            await job_store.update(regen_job_id, status=JobStatus.failed, error=str(e))

    background_tasks.add_task(process)
    return {"job_id": regen_job_id, "message": "Régénération démarrée"}


@router.put("/jobs/{job_id}/segments/{segment_id}")
async def update_segment_text(job_id: str, segment_id: str, body: UpdateSegmentText) -> dict:
    """Modifie le texte d'un segment."""
    segments = _segments_store.get(job_id)
    if not segments:
        raise APIError(ErrorCode.NOT_FOUND, f"Segments pour job {job_id} non trouvés", status_code=404)

    seg = next((s for s in segments if s["id"] == segment_id), None)
    if not seg:
        raise APIError(ErrorCode.NOT_FOUND, f"Segment {segment_id} non trouvé", status_code=404)

    seg["text"] = body.text
    return {"success": True, "segment_id": segment_id, "text": body.text}


@router.post("/jobs/{job_id}/rebuild")
async def rebuild_from_segments(job_id: str, background_tasks: BackgroundTasks) -> dict:
    """Reconstruit l'audio final à partir des segments modifiés."""
    segments = _segments_store.get(job_id)
    if not segments:
        raise APIError(ErrorCode.NOT_FOUND, f"Segments pour job {job_id} non trouvés", status_code=404)

    rebuild_job_id = await job_store.create("rebuild")

    async def process():
        try:
            import numpy as np
            import soundfile as sf

            await job_store.update(rebuild_job_id, status=JobStatus.processing, progress=10, phase="loading_segments")

            parts = []
            for i, seg in enumerate(segments):
                audio_path = OUTPUT_DIR / seg["audio_url"].lstrip("/output/")
                if audio_path.exists():
                    data, sr = sf.read(str(audio_path))
                    parts.append(data)
                    # Add small silence between segments
                    parts.append(np.zeros(int(sr * 0.3)))

                await job_store.update(
                    rebuild_job_id,
                    progress=10 + int(70 * (i + 1) / len(segments)),
                    phase="assembling",
                )

            if not parts:
                await job_store.update(rebuild_job_id, status=JobStatus.failed, error="Aucun segment audio trouvé")
                return

            full_audio = np.concatenate(parts)
            output_name = f"rebuilt_{job_id}.wav"
            output_path = OUTPUT_DIR / output_name
            sf.write(str(output_path), full_audio, 24000)

            duration = len(full_audio) / 24000
            await job_store.update(
                rebuild_job_id,
                status=JobStatus.completed,
                progress=100,
                phase="done",
                result={
                    "output_file": output_name,
                    "duration_seconds": round(duration, 2),
                    "download_url": f"/output/{output_name}",
                },
            )
        except Exception as e:
            await job_store.update(rebuild_job_id, status=JobStatus.failed, error=str(e))

    background_tasks.add_task(process)
    return {"job_id": rebuild_job_id, "message": "Reconstruction démarrée"}


def save_segments(job_id: str, segments: list[dict]):
    """Sauvegarde les segments d'un job (appelé par le pipeline de génération)."""
    _segments_store[job_id] = segments


def get_segments_store() -> dict:
    """Retourne le store de segments (pour accès externe)."""
    return _segments_store
