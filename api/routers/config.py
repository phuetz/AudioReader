"""Endpoints configuration et health check."""
from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter

from api.dependencies import OUTPUT_DIR, START_TIME
from api.models import ConfigResponse, HealthResponse

router = APIRouter(prefix="/api/v2", tags=["Config"])


@router.get("/health", response_model=HealthResponse)
async def health():
    """Status système : engines disponibles, version, uptime."""
    engines: dict[str, bool] = {}
    try:
        from src.tts_kokoro_engine import KokoroTTSEngine
        engines["kokoro"] = True
    except Exception:
        engines["kokoro"] = False

    try:
        import edge_tts
        engines["edge_tts"] = True
    except Exception:
        engines["edge_tts"] = False

    try:
        from src.tts_xtts_engine import XTTSEngine
        engines["xtts"] = True
    except Exception:
        engines["xtts"] = False

    # v4.0 engines
    try:
        from src.tts_chatterbox_engine import ChatterboxEngine
        engines["chatterbox"] = ChatterboxEngine.is_available()
    except Exception:
        engines["chatterbox"] = False

    try:
        from src.tts_dia_engine import DiaEngine
        engines["dia"] = DiaEngine.is_available()
    except Exception:
        engines["dia"] = False

    try:
        from src.tts_f5_engine import F5Engine
        engines["f5"] = F5Engine.is_available()
    except Exception:
        engines["f5"] = False

    return HealthResponse(
        status="ok",
        version="4.0.0",
        engines=engines,
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


@router.get("/config", response_model=ConfigResponse)
async def get_config():
    """Configuration actuelle d'AudioReader."""
    return ConfigResponse(
        output_dir=str(OUTPUT_DIR),
        default_voice="ff_siwis",
        default_language="fr",
        features={
            "intonation_contours": True,
            "timing_humanization": True,
            "advanced_breaths": True,
            "emotion_analysis": True,
            "multi_voice": True,
            "acx_compliance": True,
        },
        styles_available=[
            "formal", "conversational", "dramatic",
            "storytelling", "documentary", "intimate", "energetic",
        ],
        version="4.0.0",
    )
