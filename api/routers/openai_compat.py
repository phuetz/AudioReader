"""
Endpoint compatible OpenAI — /v1/audio/speech.

Permet d'utiliser AudioReader comme drop-in replacement pour l'API TTS d'OpenAI.
Compatible avec le SDK OpenAI Python et les outils tiers.
"""
from __future__ import annotations

import io
import time
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from api.dependencies import get_tts_engine
from api.errors import APIError, ErrorCode

router = APIRouter(tags=["OpenAI Compatible"])

# Mapping des noms de modeles OpenAI vers les moteurs AudioReader
MODEL_MAP = {
    "tts-1": "kokoro",
    "tts-1-hd": "kokoro",
    "kokoro": "kokoro",
    "chatterbox": "chatterbox",
    "dia": "dia",
    "f5": "f5",
    "xtts": "xtts",
    "edge": "edge",
}

# Mapping des voix OpenAI vers les voix AudioReader
VOICE_MAP = {
    "alloy": "af_bella",
    "echo": "am_adam",
    "fable": "bf_emma",
    "onyx": "am_michael",
    "nova": "af_nicole",
    "shimmer": "af_sky",
}

# Formats de sortie supportes
SUPPORTED_FORMATS = {"wav", "mp3", "opus", "aac", "flac", "pcm"}


class SpeechRequest(BaseModel):
    """Requete compatible OpenAI /v1/audio/speech."""
    model: str = Field(default="tts-1", description="Modele TTS")
    input: str = Field(..., min_length=1, max_length=4096, description="Texte a synthetiser")
    voice: str = Field(default="alloy", description="Voix a utiliser")
    response_format: str = Field(default="wav", description="Format audio de sortie")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Vitesse de lecture")


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "audioreader"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class VoiceInfo(BaseModel):
    voice_id: str
    name: str


@router.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    """
    Genere de l'audio a partir de texte (compatible OpenAI).

    Drop-in replacement pour https://api.openai.com/v1/audio/speech
    """
    # Valider le format
    fmt = request.response_format.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise APIError(
            ErrorCode.VALIDATION_ERROR,
            f"Format non supporte: {fmt}. Supportes: {', '.join(SUPPORTED_FORMATS)}",
            status_code=400,
        )

    # Mapper la voix
    voice_id = VOICE_MAP.get(request.voice, request.voice)

    try:
        tts = get_tts_engine()
        audio, sample_rate = tts.synthesize(
            text=request.input,
            voice=voice_id,
            speed=request.speed,
            lang="fr",
        )

        import soundfile as sf
        import numpy as np

        buf = io.BytesIO()

        if fmt in ("wav", "pcm"):
            sf.write(buf, audio, sample_rate, format="WAV")
            media_type = "audio/wav"
        elif fmt == "mp3":
            # Fallback to WAV if MP3 encoding not available
            try:
                from pydub import AudioSegment
                wav_buf = io.BytesIO()
                sf.write(wav_buf, audio, sample_rate, format="WAV")
                wav_buf.seek(0)
                seg = AudioSegment.from_wav(wav_buf)
                seg.export(buf, format="mp3")
                media_type = "audio/mpeg"
            except ImportError:
                sf.write(buf, audio, sample_rate, format="WAV")
                media_type = "audio/wav"
        elif fmt == "flac":
            sf.write(buf, audio, sample_rate, format="FLAC")
            media_type = "audio/flac"
        elif fmt in ("opus", "aac"):
            # Fallback to WAV for opus/aac
            sf.write(buf, audio, sample_rate, format="WAV")
            media_type = "audio/wav"
        else:
            sf.write(buf, audio, sample_rate, format="WAV")
            media_type = "audio/wav"

        buf.seek(0)
        return Response(
            content=buf.read(),
            media_type=media_type,
            headers={"Content-Disposition": f'inline; filename="speech.{fmt}"'},
        )

    except Exception as e:
        raise APIError(ErrorCode.TTS_ENGINE_UNAVAILABLE, str(e), status_code=503)


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """Liste les modeles TTS disponibles."""
    models = []
    engines_to_check = {
        "kokoro": ("tts-1", "Kokoro TTS (rapide)"),
        "chatterbox": ("chatterbox", "Chatterbox (clonage voix)"),
        "dia": ("dia", "Dia 1.6B (multi-speakers)"),
        "f5": ("f5", "F5-TTS (flow matching)"),
        "xtts": ("xtts", "XTTS-v2 (clonage)"),
        "edge": ("edge", "Edge-TTS (Microsoft)"),
    }

    for engine_id, (model_id, _) in engines_to_check.items():
        models.append(ModelInfo(
            id=model_id,
            created=int(time.time()),
            owned_by="audioreader",
        ))

    return ModelsResponse(data=models)


@router.get("/v1/audio/voices")
async def list_voices():
    """Liste les voix disponibles."""
    voices = []

    # Voix OpenAI mappees
    for oai_name, ar_id in VOICE_MAP.items():
        voices.append(VoiceInfo(voice_id=oai_name, name=f"{oai_name} -> {ar_id}"))

    # Voix Kokoro directes
    kokoro_voices = [
        ("ff_siwis", "Siwis (FR femme)"),
        ("af_bella", "Bella (EN femme)"),
        ("af_heart", "Heart (EN femme)"),
        ("am_adam", "Adam (EN homme)"),
        ("am_michael", "Michael (EN homme)"),
        ("bf_emma", "Emma (EN-GB femme)"),
    ]
    for vid, name in kokoro_voices:
        voices.append(VoiceInfo(voice_id=vid, name=name))

    return {"voices": voices}
