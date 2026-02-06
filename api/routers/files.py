"""Endpoints fichiers : upload avec conversion auto, listing paginé, téléchargement, batch export."""
from __future__ import annotations

import io
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from api.dependencies import OUTPUT_DIR, file_store
from api.errors import APIError, ErrorCode
from api.models import FileInfo, PaginatedFiles, UploadResponse

router = APIRouter(prefix="/api/v2", tags=["Files"])

ALLOWED_EXTENSIONS = {".md", ".txt", ".pdf", ".epub", ".wav", ".mp3", ".mp4", ".mkv", ".avi", ".mov", ".webm"}
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB


@router.post("/files/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload un fichier (MD/PDF/EPUB/audio/vidéo) avec conversion auto en Markdown."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise APIError(
            ErrorCode.UNSUPPORTED_FORMAT,
            f"Format {suffix} non supporté. Formats acceptés : {', '.join(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise APIError(ErrorCode.FILE_TOO_LARGE, "Fichier trop volumineux (max 500 MB)")

    file_id, saved_path = file_store.save(file.filename, content)
    meta = file_store.get_meta(file_id)

    response = UploadResponse(
        file_id=file_id,
        original_name=file.filename,
        mime_type=meta["mime_type"],
    )

    # Conversion auto pour PDF/EPUB
    if suffix in (".pdf", ".epub"):
        try:
            from src.input_converter import InputConverter
            converter = InputConverter()
            md_path = converter.convert_to_markdown(saved_path)
            response.converted_path = str(md_path)
            text = md_path.read_text(encoding="utf-8")
            response.text_preview = text[:500]
            # Extraire les titres de chapitres
            chapters = [
                line.strip("# ").strip()
                for line in text.splitlines()
                if line.startswith("# ") or line.startswith("## ")
            ]
            response.chapters = chapters[:50]
        except Exception:
            pass

    # Preview pour Markdown/texte
    if suffix in (".md", ".txt"):
        text = saved_path.read_text(encoding="utf-8", errors="replace")
        response.text_preview = text[:500]
        chapters = [
            line.strip("# ").strip()
            for line in text.splitlines()
            if line.startswith("# ") or line.startswith("## ")
        ]
        response.chapters = chapters[:50]

    return response


@router.get("/files", response_model=PaginatedFiles)
async def list_files(extension: str = None, offset: int = 0, limit: int = 50):
    """Liste les fichiers audio générés dans output/ avec pagination."""
    all_files: list[FileInfo] = []
    patterns = [f"*.{extension}"] if extension else ["*.wav", "*.mp3", "*.m4b"]

    for pattern in patterns:
        for f in OUTPUT_DIR.glob(pattern):
            stat = f.stat()
            all_files.append(FileInfo(
                id=f.stem,
                name=f.name,
                path=str(f),
                size_mb=round(stat.st_size / 1024 / 1024, 2),
                mime_type=f"audio/{f.suffix[1:]}",
                created_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                download_url=f"/output/{f.name}",
            ))

    all_files.sort(key=lambda x: x.created_at, reverse=True)
    total = len(all_files)
    page = all_files[offset:offset + limit]
    return PaginatedFiles(files=page, total=total, offset=offset, limit=limit)


class BatchExportRequest(BaseModel):
    filenames: List[str]


@router.post("/files/batch-export")
async def batch_export(request: BatchExportRequest):
    """Exporte plusieurs fichiers dans un ZIP."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in request.filenames:
            path = OUTPUT_DIR / name
            if path.exists() and path.is_file():
                zf.write(path, name)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=audioreader_export.zip"},
    )


@router.get("/files/{filename}")
async def download_file(filename: str):
    """Télécharge un fichier audio généré."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise APIError(ErrorCode.NOT_FOUND, f"Fichier {filename} non trouvé", status_code=404)

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type=f"audio/{file_path.suffix[1:]}",
    )
