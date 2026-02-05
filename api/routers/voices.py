"""Endpoints voix : liste, preview, clonage."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import Response

from api.dependencies import CLONED_VOICES_DIR, OUTPUT_DIR, get_tts_engine
from api.errors import APIError, ErrorCode
from api.models import VoiceCloneRequest, VoiceInfo, VoicePreviewRequest

router = APIRouter(prefix="/api/v2", tags=["Voices"])

# Voix Kokoro intégrées
KOKORO_VOICES = {
    "fr": [
        VoiceInfo(id="ff_siwis", name="Siwis", gender="F", language="fr", engine="kokoro", style="neutral"),
    ],
    "en": [
        VoiceInfo(id="af_heart", name="Heart", gender="F", language="en", engine="kokoro", style="warm"),
        VoiceInfo(id="af_bella", name="Bella", gender="F", language="en", engine="kokoro", style="gentle"),
        VoiceInfo(id="af_sarah", name="Sarah", gender="F", language="en", engine="kokoro", style="professional"),
        VoiceInfo(id="af_nicole", name="Nicole", gender="F", language="en", engine="kokoro", style="bright"),
        VoiceInfo(id="af_sky", name="Sky", gender="F", language="en", engine="kokoro", style="light"),
        VoiceInfo(id="am_adam", name="Adam", gender="M", language="en", engine="kokoro", style="neutral"),
        VoiceInfo(id="am_michael", name="Michael", gender="M", language="en", engine="kokoro", style="deep"),
        VoiceInfo(id="bf_emma", name="Emma", gender="F", language="en", engine="kokoro", style="british"),
        VoiceInfo(id="bm_george", name="George", gender="M", language="en", engine="kokoro", style="british"),
    ],
}

EDGE_VOICES = {
    "fr": [
        VoiceInfo(id="fr-FR-DeniseNeural", name="Denise", gender="F", language="fr", engine="edge", style="neural"),
        VoiceInfo(id="fr-FR-HenriNeural", name="Henri", gender="M", language="fr", engine="edge", style="neural"),
        VoiceInfo(id="fr-FR-AlainNeural", name="Alain", gender="M", language="fr", engine="edge", style="neural"),
        VoiceInfo(id="fr-FR-BrigitteNeural", name="Brigitte", gender="F", language="fr", engine="edge", style="neural"),
        VoiceInfo(id="fr-FR-CelesteNeural", name="Céleste", gender="F", language="fr", engine="edge", style="neural"),
        VoiceInfo(id="fr-FR-ClaudeNeural", name="Claude", gender="M", language="fr", engine="edge", style="neural"),
        VoiceInfo(id="fr-FR-CoralieNeural", name="Coralie", gender="F", language="fr", engine="edge", style="neural"),
        VoiceInfo(id="fr-FR-EloiseNeural", name="Éloise", gender="F", language="fr", engine="edge", style="children"),
        VoiceInfo(id="fr-FR-JacquelineNeural", name="Jacqueline", gender="F", language="fr", engine="edge", style="neural"),
        VoiceInfo(id="fr-FR-JeromeNeural", name="Jérôme", gender="M", language="fr", engine="edge", style="neural"),
        VoiceInfo(id="fr-FR-JosephineNeural", name="Joséphine", gender="F", language="fr", engine="edge", style="neural"),
        VoiceInfo(id="fr-FR-MauriceNeural", name="Maurice", gender="M", language="fr", engine="edge", style="neural"),
        VoiceInfo(id="fr-FR-YvesNeural", name="Yves", gender="M", language="fr", engine="edge", style="neural"),
        VoiceInfo(id="fr-FR-YvetteNeural", name="Yvette", gender="F", language="fr", engine="edge", style="neural"),
        VoiceInfo(id="fr-CA-SylvieNeural", name="Sylvie (CA)", gender="F", language="fr", engine="edge", style="canadian"),
        VoiceInfo(id="fr-CA-JeanNeural", name="Jean (CA)", gender="M", language="fr", engine="edge", style="canadian"),
        VoiceInfo(id="fr-CA-AntoineNeural", name="Antoine (CA)", gender="M", language="fr", engine="edge", style="canadian"),
        VoiceInfo(id="fr-BE-CharlineNeural", name="Charline (BE)", gender="F", language="fr", engine="edge", style="belgian"),
        VoiceInfo(id="fr-BE-GerardNeural", name="Gérard (BE)", gender="M", language="fr", engine="edge", style="belgian"),
        VoiceInfo(id="fr-CH-ArianeNeural", name="Ariane (CH)", gender="F", language="fr", engine="edge", style="swiss"),
        VoiceInfo(id="fr-CH-FabriceNeural", name="Fabrice (CH)", gender="M", language="fr", engine="edge", style="swiss"),
    ],
    "en": [
        VoiceInfo(id="en-US-JennyNeural", name="Jenny", gender="F", language="en", engine="edge", style="neural"),
        VoiceInfo(id="en-US-GuyNeural", name="Guy", gender="M", language="en", engine="edge", style="neural"),
        VoiceInfo(id="en-GB-SoniaNeural", name="Sonia", gender="F", language="en", engine="edge", style="british"),
    ],
}

# MMS-TTS voices (Meta Multilingual Speech)
MMS_VOICES = {
    "fr": [
        VoiceInfo(id="mms_fr", name="MMS Français", gender="N", language="fr", engine="mms", style="neural"),
    ],
    "en": [
        VoiceInfo(id="mms_en", name="MMS English", gender="N", language="en", engine="mms", style="neural"),
    ],
}


@router.get("/voices")
async def list_voices(language: Optional[str] = None):
    """Liste toutes les voix disponibles (kokoro + edge + clonées)."""
    voices: list[VoiceInfo] = []

    langs = [language] if language else ["fr", "en"]
    for lang in langs:
        voices.extend(KOKORO_VOICES.get(lang, []))
        voices.extend(EDGE_VOICES.get(lang, []))
        voices.extend(MMS_VOICES.get(lang, []))

    # Voix clonées
    for meta_file in CLONED_VOICES_DIR.glob("*.json"):
        import json
        meta = json.loads(meta_file.read_text())
        if language and meta.get("language") != language:
            continue
        voices.append(VoiceInfo(
            id=meta["id"],
            name=meta["name"],
            gender=meta.get("gender", "?"),
            language=meta.get("language", "fr"),
            engine="cloned",
            style="cloned",
        ))

    return {"voices": voices, "total": len(voices)}


@router.post("/voices/preview")
async def preview_voice(request: VoicePreviewRequest):
    """Génère un court extrait audio d'une voix (retourne WAV synchrone)."""
    try:
        tts = get_tts_engine()
        audio, sample_rate = tts.synthesize(
            text=request.text,
            voice=request.voice_id,
            speed=request.speed,
            lang=request.language,
        )

        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV")
        buf.seek(0)

        return Response(
            content=buf.read(),
            media_type="audio/wav",
            headers={"Content-Disposition": f'inline; filename="preview_{request.voice_id}.wav"'},
        )
    except Exception as e:
        raise APIError(ErrorCode.TTS_ENGINE_UNAVAILABLE, str(e), status_code=503)


@router.post("/voices/clone")
async def clone_voice(
    name: str = Form(...),
    language: str = Form("fr"),
    audio: UploadFile = File(...),
):
    """Clone une voix à partir d'un fichier audio (multipart)."""
    import json
    import uuid

    content = await audio.read()
    if len(content) < 1000:
        raise APIError(ErrorCode.VALIDATION_ERROR, "Fichier audio trop court (min 6 secondes)")

    voice_id = f"cloned_{uuid.uuid4().hex[:6]}"
    audio_path = CLONED_VOICES_DIR / f"{voice_id}.wav"
    audio_path.write_bytes(content)

    meta = {
        "id": voice_id,
        "name": name,
        "language": language,
        "gender": "?",
        "audio_file": str(audio_path),
    }
    meta_path = CLONED_VOICES_DIR / f"{voice_id}.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    return {"voice_id": voice_id, "name": name, "message": "Voix clonée avec succès"}


@router.get("/voices/cloned")
async def list_cloned_voices():
    """Liste les voix clonées."""
    import json
    voices = []
    for meta_file in CLONED_VOICES_DIR.glob("*.json"):
        meta = json.loads(meta_file.read_text())
        voices.append(meta)
    return {"voices": voices}


@router.delete("/voices/cloned/{voice_id}")
async def delete_cloned_voice(voice_id: str):
    """Supprime une voix clonée."""
    meta_path = CLONED_VOICES_DIR / f"{voice_id}.json"
    audio_path = CLONED_VOICES_DIR / f"{voice_id}.wav"
    if not meta_path.exists():
        raise APIError(ErrorCode.NOT_FOUND, f"Voix {voice_id} non trouvée", status_code=404)
    meta_path.unlink(missing_ok=True)
    audio_path.unlink(missing_ok=True)
    return {"success": True, "message": f"Voix {voice_id} supprimée"}
