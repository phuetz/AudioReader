"""Sous-titres synchronisés — SRT, VTT, JSON."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.dependencies import OUTPUT_DIR, job_store
from api.errors import APIError, ErrorCode
from api.models import JobStatus

router = APIRouter(prefix="/api/v2", tags=["Subtitles"])


class SubtitleOptions(BaseModel):
    format: str = "srt"  # srt, vtt, json
    mode: str = "estimated"  # estimated, whisper
    max_chars_per_line: int = Field(default=42, ge=20, le=80)
    highlight_speaker: bool = False


class SubtitleResult(BaseModel):
    success: bool
    format: str
    download_url: str = ""
    entries_count: int = 0
    message: str = ""


@router.post("/jobs/{job_id}/subtitles")
async def generate_subtitles(
    job_id: str, options: SubtitleOptions, background_tasks: BackgroundTasks
) -> dict:
    """Génère des sous-titres pour un job terminé."""
    job = job_store.get(job_id)
    if not job:
        raise APIError(ErrorCode.NOT_FOUND, f"Job {job_id} non trouvé", status_code=404)
    if job["status"] != JobStatus.completed:
        raise APIError(ErrorCode.VALIDATION_ERROR, "Le job doit être terminé pour générer des sous-titres")

    result = job.get("result", {})
    output_file = result.get("output_file")
    if not output_file:
        raise APIError(ErrorCode.NOT_FOUND, "Fichier audio du job non trouvé")

    audio_path = OUTPUT_DIR / output_file
    if not audio_path.exists():
        raise APIError(ErrorCode.NOT_FOUND, "Fichier audio du job non trouvé")

    sub_job_id = job_store.create("subtitles")

    async def process():
        try:
            job_store.update(sub_job_id, status=JobStatus.processing, progress=10, phase="generating_subtitles")

            from src.subtitle_generator import SubtitleGenerator

            generator = SubtitleGenerator()

            # Get text from original job context or use dummy text
            text = result.get("original_text", "")
            duration = result.get("duration_seconds", 60.0)

            ext = {"srt": ".srt", "vtt": ".vtt", "json": ".json"}.get(options.format, ".srt")
            sub_name = f"{audio_path.stem}{ext}"
            sub_path = OUTPUT_DIR / sub_name

            if options.format == "srt":
                success = generator.generate_srt(
                    str(audio_path), text, str(sub_path), audio_duration_s=duration
                )
            elif options.format == "vtt":
                success = generator.generate_vtt(
                    str(audio_path), text, str(sub_path), audio_duration_s=duration
                )
            elif options.format == "json":
                success = generator.generate_word_level_json(
                    str(audio_path), text, str(sub_path), audio_duration_s=duration
                )
            else:
                success = False

            if success:
                job_store.update(
                    sub_job_id,
                    status=JobStatus.completed,
                    progress=100,
                    phase="done",
                    result={
                        "download_url": f"/output/{sub_name}",
                        "format": options.format,
                        "subtitle_file": sub_name,
                    },
                )
            else:
                job_store.update(sub_job_id, status=JobStatus.failed, error="Échec de génération des sous-titres")
        except Exception as e:
            job_store.update(sub_job_id, status=JobStatus.failed, error=str(e))

    background_tasks.add_task(process)
    return {"job_id": sub_job_id, "message": "Génération des sous-titres démarrée"}


@router.get("/jobs/{job_id}/subtitles")
async def get_subtitles(job_id: str, format: str = "srt") -> FileResponse:
    """Télécharge les sous-titres d'un job."""
    job = job_store.get(job_id)
    if not job:
        raise APIError(ErrorCode.NOT_FOUND, f"Job {job_id} non trouvé", status_code=404)

    result = job.get("result", {})
    output_file = result.get("output_file", "")
    stem = Path(output_file).stem if output_file else job_id

    ext = {"srt": ".srt", "vtt": ".vtt", "json": ".json"}.get(format, ".srt")
    sub_path = OUTPUT_DIR / f"{stem}{ext}"

    if not sub_path.exists():
        raise APIError(ErrorCode.NOT_FOUND, "Sous-titres non trouvés. Générez-les d'abord.", status_code=404)

    media_types = {"srt": "text/plain", "vtt": "text/vtt", "json": "application/json"}
    return FileResponse(
        path=str(sub_path),
        media_type=media_types.get(format, "text/plain"),
        filename=sub_path.name,
    )
