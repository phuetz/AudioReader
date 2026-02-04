"""Endpoints podcast : démarrage, arrêt, status du serveur RSS."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter

from api.dependencies import OUTPUT_DIR
from api.errors import APIError, ErrorCode
from api.models import PodcastStartRequest, PodcastStatusResponse

router = APIRouter(prefix="/api/v2", tags=["Podcast"])

# État global du serveur podcast
_podcast_server = None
_podcast_thread = None


@router.post("/podcast/start")
async def start_podcast(request: PodcastStartRequest):
    """Démarre le serveur podcast RSS local."""
    global _podcast_server, _podcast_thread

    if _podcast_server is not None:
        raise APIError(ErrorCode.PODCAST_ALREADY_RUNNING, "Le serveur podcast est déjà en cours")

    try:
        from src.podcast_server import PodcastServer
        import threading

        audio_dir = Path(request.audio_dir)
        if not audio_dir.is_absolute():
            audio_dir = OUTPUT_DIR.parent / request.audio_dir

        _podcast_server = PodcastServer(
            audio_dir=str(audio_dir),
            port=request.port,
            title=request.title,
        )

        _podcast_thread = threading.Thread(target=_podcast_server.start, daemon=True)
        _podcast_thread.start()

        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        url = f"http://{local_ip}:{request.port}"

        return {
            "success": True,
            "url": url,
            "port": request.port,
            "message": f"Serveur podcast démarré sur {url}",
        }
    except Exception as e:
        _podcast_server = None
        _podcast_thread = None
        raise APIError(ErrorCode.INTERNAL_ERROR, f"Erreur démarrage podcast : {e}", status_code=500)


@router.post("/podcast/stop")
async def stop_podcast():
    """Arrête le serveur podcast."""
    global _podcast_server, _podcast_thread

    if _podcast_server is None:
        raise APIError(ErrorCode.PODCAST_NOT_RUNNING, "Aucun serveur podcast en cours")

    try:
        if hasattr(_podcast_server, 'stop'):
            _podcast_server.stop()
    except Exception:
        pass

    _podcast_server = None
    _podcast_thread = None
    return {"success": True, "message": "Serveur podcast arrêté"}


@router.get("/podcast/status", response_model=PodcastStatusResponse)
async def podcast_status():
    """Status du serveur podcast."""
    if _podcast_server is None:
        return PodcastStatusResponse(running=False)

    episode_count = 0
    try:
        if hasattr(_podcast_server, 'audio_dir'):
            audio_dir = Path(_podcast_server.audio_dir)
            episode_count = len(list(audio_dir.glob("*.wav"))) + len(list(audio_dir.glob("*.mp3")))
    except Exception:
        pass

    import socket
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        url = f"http://{local_ip}:{_podcast_server.port}"
    except Exception:
        url = f"http://localhost:{_podcast_server.port}"

    return PodcastStatusResponse(
        running=True,
        url=url,
        port=_podcast_server.port,
        episode_count=episode_count,
    )
