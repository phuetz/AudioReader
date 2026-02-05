"""Endpoints de streaming audio en temps réel."""
from __future__ import annotations

import asyncio
import base64
import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v2", tags=["Streaming"])


class StreamRequest(BaseModel):
    """Requête de synthèse en streaming."""
    text: str = Field(..., min_length=1, max_length=10000, description="Texte à synthétiser")
    voice: str = Field(default="ff_siwis", description="Identifiant de la voix")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Vitesse de parole")
    chunk_size_ms: int = Field(default=300, ge=100, le=1000, description="Taille des chunks en ms")


class StreamInfo(BaseModel):
    """Informations sur les capacités de streaming."""
    streaming_available: bool
    supported_formats: list[str]
    max_text_length: int
    default_chunk_size_ms: int


@router.get("/streaming/info", response_model=StreamInfo)
async def get_streaming_info():
    """Retourne les informations sur les capacités de streaming."""
    return StreamInfo(
        streaming_available=True,
        supported_formats=["raw_pcm_float32", "base64_pcm"],
        max_text_length=10000,
        default_chunk_size_ms=300,
    )


@router.post("/synthesize-stream")
async def synthesize_stream(request: StreamRequest):
    """
    Stream audio pendant la génération TTS.

    Retourne un flux SSE (Server-Sent Events) avec les chunks audio encodés en base64.

    Événements:
    - data: {"type": "chunk", "audio": "<base64>", "sample_rate": 24000, "timestamp_ms": 0}
    - data: {"type": "progress", "text_segment": "...", "progress_percent": 50}
    - data: {"type": "done", "total_duration_ms": 5000}
    - data: {"type": "error", "message": "..."}
    """
    async def event_generator():
        try:
            # Import lazy pour éviter les dépendances circulaires
            from src.tts_async import AsyncKokoroWrapper, StreamingConfig

            config = StreamingConfig(
                chunk_size_ms=request.chunk_size_ms,
                sample_rate=24000,
            )
            engine = AsyncKokoroWrapper(config=config)

            total_duration_ms = 0
            text_length = len(request.text)
            processed_chars = 0

            async for chunk in engine.synthesize_stream(
                text=request.text,
                voice_id=request.voice,
                speed=request.speed,
            ):
                # Encoder l'audio en base64
                audio_bytes = chunk.audio.astype('float32').tobytes()
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

                # Envoyer le chunk audio
                chunk_data = {
                    "type": "chunk",
                    "audio": audio_b64,
                    "sample_rate": chunk.sample_rate,
                    "timestamp_ms": chunk.timestamp_ms,
                    "is_final": chunk.is_final,
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"

                # Calculer la progression
                processed_chars += len(chunk.text_segment)
                progress = int((processed_chars / text_length) * 100) if text_length > 0 else 100
                total_duration_ms = chunk.timestamp_ms + int(len(chunk.audio) / chunk.sample_rate * 1000)

                # Envoyer un événement de progression
                progress_data = {
                    "type": "progress",
                    "text_segment": chunk.text_segment[:100],  # Limiter la longueur
                    "progress_percent": min(progress, 100),
                }
                yield f"data: {json.dumps(progress_data)}\n\n"

                # Petit délai pour permettre au client de traiter
                await asyncio.sleep(0.01)

            # Événement de fin
            done_data = {
                "type": "done",
                "total_duration_ms": total_duration_ms,
            }
            yield f"data: {json.dumps(done_data)}\n\n"

        except ImportError as e:
            error_data = {
                "type": "error",
                "message": f"Module non disponible: {str(e)}",
            }
            yield f"data: {json.dumps(error_data)}\n\n"
        except Exception as e:
            error_data = {
                "type": "error",
                "message": str(e),
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Désactiver le buffering nginx
        },
    )


@router.post("/synthesize-stream/start")
async def start_streaming_session(request: StreamRequest):
    """
    Démarre une session de streaming et retourne un session_id.

    Alternative à l'endpoint SSE pour les clients qui préfèrent
    le polling ou les WebSockets.
    """
    import uuid

    session_id = str(uuid.uuid4())

    # Stocker la session (implémentation simplifiée)
    # En production, utiliser Redis ou un store persistant
    _streaming_sessions[session_id] = {
        "request": request.model_dump(),
        "status": "pending",
        "chunks": [],
    }

    return {
        "session_id": session_id,
        "status": "created",
        "poll_url": f"/api/v2/synthesize-stream/{session_id}/poll",
    }


@router.get("/synthesize-stream/{session_id}/poll")
async def poll_streaming_session(session_id: str, offset: int = 0):
    """
    Récupère les chunks audio d'une session de streaming.

    Args:
        session_id: ID de la session
        offset: Index du premier chunk à récupérer

    Returns:
        Chunks audio depuis l'offset
    """
    if session_id not in _streaming_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _streaming_sessions[session_id]
    chunks = session.get("chunks", [])[offset:]

    return {
        "session_id": session_id,
        "status": session.get("status", "unknown"),
        "chunks": chunks,
        "next_offset": offset + len(chunks),
    }


# Store temporaire pour les sessions (en mémoire)
_streaming_sessions: dict = {}
