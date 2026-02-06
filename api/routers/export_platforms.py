"""Export multi-plateformes — Spotify, YouTube, Podcast, Audible."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

from api.dependencies import OUTPUT_DIR, job_store
from api.errors import APIError, ErrorCode
from api.models import JobStatus

router = APIRouter(prefix="/api/v2", tags=["Export"])


class ExportRequest(BaseModel):
    job_id: str
    title: str = "audiobook"
    author: str = "AudioReader"
    description: str = ""
    language: str = "fr"


class YouTubeExportRequest(ExportRequest):
    background_color: str = "#1a1a2e"
    show_waveform: bool = True


class ExportResultResponse(BaseModel):
    success: bool
    download_url: str = ""
    format: str = ""
    file_size_mb: float = 0
    duration_seconds: float = 0
    message: str = ""


async def _find_job_audio(job_id: str) -> Optional[Path]:
    """Trouve le fichier audio d'un job terminé."""
    job = await job_store.get(job_id)
    if not job or job["status"] != JobStatus.completed:
        return None
    result = job.get("result", {})
    output_file = result.get("output_file")
    if not output_file:
        return None
    path = OUTPUT_DIR / output_file
    return path if path.exists() else None


@router.post("/export/spotify")
async def export_for_spotify(request: ExportRequest, background_tasks: BackgroundTasks) -> dict:
    """Export MP3 320kbps optimisé Spotify (-14 LUFS)."""
    audio_path = await _find_job_audio(request.job_id)
    if not audio_path:
        raise APIError(ErrorCode.NOT_FOUND, "Audio du job non trouvé", status_code=404)

    export_job_id = await job_store.create("export_spotify")

    async def process():
        try:
            await job_store.update(export_job_id, status=JobStatus.processing, progress=10, phase="exporting")

            from src.platform_exporter import PlatformExporter, ExportMetadata

            exporter = PlatformExporter()
            metadata = ExportMetadata(
                title=request.title,
                author=request.author,
                description=request.description,
                language=request.language,
            )
            output = OUTPUT_DIR / f"{audio_path.stem}_spotify.mp3"
            result = exporter.export_for_spotify(audio_path, output, metadata)

            job_store.update(
                export_job_id,
                status=JobStatus.completed if result.success else JobStatus.failed,
                progress=100,
                phase="done",
                result={
                    "download_url": f"/output/{output.name}",
                    "format": result.format,
                    "file_size_mb": round(result.file_size_mb, 2),
                    "duration_seconds": round(result.duration_seconds, 2),
                    "platform": "spotify",
                },
                error=None if result.success else result.message,
            )
        except Exception as e:
            await job_store.update(export_job_id, status=JobStatus.failed, error=str(e))

    background_tasks.add_task(process)
    return {"job_id": export_job_id, "message": "Export Spotify démarré"}


@router.post("/export/youtube")
async def export_for_youtube(request: YouTubeExportRequest, background_tasks: BackgroundTasks) -> dict:
    """Export MP4 avec waveform pour YouTube."""
    audio_path = await _find_job_audio(request.job_id)
    if not audio_path:
        raise APIError(ErrorCode.NOT_FOUND, "Audio du job non trouvé", status_code=404)

    export_job_id = await job_store.create("export_youtube")

    async def process():
        try:
            await job_store.update(export_job_id, status=JobStatus.processing, progress=10, phase="exporting")

            from src.platform_exporter import PlatformExporter, ExportMetadata

            exporter = PlatformExporter()
            metadata = ExportMetadata(
                title=request.title,
                author=request.author,
                description=request.description,
                language=request.language,
            )
            output = OUTPUT_DIR / f"{audio_path.stem}_youtube.mp4"
            result = exporter.export_for_youtube(
                audio_path, output, metadata,
                show_waveform=request.show_waveform,
            )

            job_store.update(
                export_job_id,
                status=JobStatus.completed if result.success else JobStatus.failed,
                progress=100,
                phase="done",
                result={
                    "download_url": f"/output/{output.name}",
                    "format": result.format,
                    "file_size_mb": round(result.file_size_mb, 2),
                    "duration_seconds": round(result.duration_seconds, 2),
                    "platform": "youtube",
                },
                error=None if result.success else result.message,
            )
        except Exception as e:
            await job_store.update(export_job_id, status=JobStatus.failed, error=str(e))

    background_tasks.add_task(process)
    return {"job_id": export_job_id, "message": "Export YouTube démarré"}


@router.post("/export/podcast")
async def export_for_podcast(request: ExportRequest, background_tasks: BackgroundTasks) -> dict:
    """Export MP3 optimisé podcast (-16 LUFS, mono)."""
    audio_path = await _find_job_audio(request.job_id)
    if not audio_path:
        raise APIError(ErrorCode.NOT_FOUND, "Audio du job non trouvé", status_code=404)

    export_job_id = await job_store.create("export_podcast")

    async def process():
        try:
            await job_store.update(export_job_id, status=JobStatus.processing, progress=10, phase="exporting")

            from src.platform_exporter import PlatformExporter, ExportMetadata

            exporter = PlatformExporter()
            metadata = ExportMetadata(
                title=request.title,
                author=request.author,
                description=request.description,
                language=request.language,
            )
            output = OUTPUT_DIR / f"{audio_path.stem}_podcast.mp3"
            result = exporter.export_for_podcast(audio_path, output, metadata)

            job_store.update(
                export_job_id,
                status=JobStatus.completed if result.success else JobStatus.failed,
                progress=100,
                phase="done",
                result={
                    "download_url": f"/output/{output.name}",
                    "format": result.format,
                    "file_size_mb": round(result.file_size_mb, 2),
                    "duration_seconds": round(result.duration_seconds, 2),
                    "platform": "podcast",
                },
                error=None if result.success else result.message,
            )
        except Exception as e:
            await job_store.update(export_job_id, status=JobStatus.failed, error=str(e))

    background_tasks.add_task(process)
    return {"job_id": export_job_id, "message": "Export Podcast démarré"}


@router.post("/export/audible")
async def export_for_audible(request: ExportRequest, background_tasks: BackgroundTasks) -> dict:
    """Export ACX-compliant pour Audible."""
    audio_path = await _find_job_audio(request.job_id)
    if not audio_path:
        raise APIError(ErrorCode.NOT_FOUND, "Audio du job non trouvé", status_code=404)

    export_job_id = await job_store.create("export_audible")

    async def process():
        try:
            await job_store.update(export_job_id, status=JobStatus.processing, progress=10, phase="exporting")

            from src.platform_exporter import PlatformExporter, ExportMetadata

            exporter = PlatformExporter()
            metadata = ExportMetadata(
                title=request.title,
                author=request.author,
                description=request.description,
                language=request.language,
            )
            output = OUTPUT_DIR / f"{audio_path.stem}_audible.mp3"
            result = exporter.export_for_acx(audio_path, output, metadata)

            job_store.update(
                export_job_id,
                status=JobStatus.completed if result.success else JobStatus.failed,
                progress=100,
                phase="done",
                result={
                    "download_url": f"/output/{output.name}",
                    "format": result.format,
                    "file_size_mb": round(result.file_size_mb, 2),
                    "duration_seconds": round(result.duration_seconds, 2),
                    "platform": "audible",
                },
                error=None if result.success else result.message,
            )
        except Exception as e:
            await job_store.update(export_job_id, status=JobStatus.failed, error=str(e))

    background_tasks.add_task(process)
    return {"job_id": export_job_id, "message": "Export Audible démarré"}
