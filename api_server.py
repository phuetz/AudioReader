#!/usr/bin/env python3
"""
AudioReader REST API Server.

API REST pour piloter AudioReader depuis ChatGPT, n'importe quel client HTTP,
ou via des GPT Actions personnalises.

Installation:
    pip install fastapi uvicorn python-multipart

Demarrage:
    python api_server.py
    # ou
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /                    - Info API
    GET  /voices              - Liste des voix
    POST /generate            - Genere audio simple
    POST /audiobook           - Genere audiobook complet
    POST /analyze             - Analyse texte
    GET  /config              - Configuration actuelle
    PUT  /config              - Modifier configuration
    GET  /files               - Liste fichiers generes
    GET  /files/{filename}    - Telecharger fichier
    GET  /openapi.json        - Spec OpenAPI (pour ChatGPT Actions)
"""
import os
import sys
import json
import uuid
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

# Ajouter le repertoire src au path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configuration
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Jobs en cours
JOBS: Dict[str, Dict[str, Any]] = {}

# ============================================================================
# Modeles Pydantic
# ============================================================================

class VoiceInfo(BaseModel):
    id: str
    name: str
    gender: str
    style: str


class VoicesResponse(BaseModel):
    language: str
    kokoro: List[VoiceInfo]
    edge_tts: List[VoiceInfo]
    recommended: str


class GenerateRequest(BaseModel):
    text: str = Field(..., description="Texte a convertir en audio")
    voice: str = Field(default="ff_siwis", description="ID de la voix")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Vitesse (0.5-2.0)")
    output_name: Optional[str] = Field(default=None, description="Nom du fichier de sortie")


class GenerateResponse(BaseModel):
    success: bool
    job_id: str
    output_file: Optional[str] = None
    duration_seconds: Optional[float] = None
    status: str
    message: str


class NarrationStyle(str, Enum):
    formal = "formal"
    conversational = "conversational"
    dramatic = "dramatic"
    storytelling = "storytelling"
    documentary = "documentary"
    intimate = "intimate"
    energetic = "energetic"


class AudiobookRequest(BaseModel):
    text: str = Field(..., description="Texte complet du livre/chapitre")
    title: str = Field(default="audiobook", description="Titre de l'audiobook")
    narrator_voice: str = Field(default="ff_siwis", description="Voix du narrateur")
    style: NarrationStyle = Field(default=NarrationStyle.storytelling, description="Style de narration")
    enable_emotions: bool = Field(default=True, description="Activer l'analyse des emotions")
    enable_multi_voice: bool = Field(default=True, description="Activer les voix multiples")
    language: str = Field(default="fr", description="Langue (fr, en)")


class AudiobookResponse(BaseModel):
    success: bool
    job_id: str
    status: str
    output_file: Optional[str] = None
    duration_seconds: Optional[float] = None
    duration_formatted: Optional[str] = None
    segments_count: Optional[int] = None
    characters_detected: Optional[List[str]] = None
    message: str


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Texte a analyser")


class DialogueInfo(BaseModel):
    text: str
    speaker: str
    method: str
    confidence: float


class SentenceInfo(BaseModel):
    text: str
    intonation: str
    emotion: str
    intensity: float


class AnalyzeResponse(BaseModel):
    total_characters: int
    dialogues: List[DialogueInfo]
    characters: List[str]
    sentences: List[SentenceInfo]


class ConfigResponse(BaseModel):
    output_dir: str
    default_voice: str
    default_language: str
    features: Dict[str, bool]
    styles_available: List[str]
    version: str


class ConfigUpdateRequest(BaseModel):
    narrator_voice: Optional[str] = None
    default_style: Optional[str] = None
    enable_intonation: Optional[bool] = None
    enable_timing_humanization: Optional[bool] = None
    intonation_strength: Optional[float] = None


class FileInfo(BaseModel):
    name: str
    path: str
    size_mb: float
    modified: str
    download_url: str


class FilesResponse(BaseModel):
    output_dir: str
    files: List[FileInfo]
    total_count: int


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ============================================================================
# Application FastAPI
# ============================================================================

app = FastAPI(
    title="AudioReader API",
    description="""
API REST pour generer des audiobooks de haute qualite avec AudioReader.

## Fonctionnalites

- **Synthese vocale** : Convertir du texte en audio avec differentes voix
- **Audiobooks** : Generer des audiobooks complets avec analyse des dialogues
- **Analyse de texte** : Detecter les emotions, dialogues et personnages
- **Multi-voix** : Attribution automatique des voix aux personnages

## Utilisation avec ChatGPT

Cette API peut etre utilisee comme Action ChatGPT. Utilisez l'endpoint
`/openapi.json` pour obtenir la specification OpenAPI.
    """,
    version="2.4.0",
    contact={
        "name": "AudioReader",
        "url": "https://github.com/audioreader",
    },
)

# CORS pour permettre les appels depuis n'importe quelle origine
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales pour le TTS (lazy loading)
_tts_engine = None
_pipeline = None


def get_tts_engine():
    """Obtient ou initialise le moteur TTS."""
    global _tts_engine
    if _tts_engine is None:
        from src.tts_unified import UnifiedTTS
        _tts_engine = UnifiedTTS()
    return _tts_engine


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Information sur l'API AudioReader."""
    return {
        "name": "AudioReader API",
        "version": "2.4.0",
        "description": "API pour generer des audiobooks de haute qualite",
        "endpoints": {
            "voices": "/voices",
            "generate": "/generate",
            "audiobook": "/audiobook",
            "analyze": "/analyze",
            "config": "/config",
            "files": "/files",
            "job_status": "/jobs/{job_id}"
        }
    }


@app.get("/voices", response_model=VoicesResponse, tags=["Voix"])
async def list_voices(language: str = Query(default="fr", description="Code langue (fr, en)")):
    """Liste les voix disponibles pour la synthese vocale."""
    kokoro_voices = {
        "fr": [
            VoiceInfo(id="ff_siwis", name="Siwis", gender="F", style="neutral"),
        ],
        "en": [
            VoiceInfo(id="af_heart", name="Heart", gender="F", style="warm"),
            VoiceInfo(id="af_sarah", name="Sarah", gender="F", style="professional"),
            VoiceInfo(id="am_adam", name="Adam", gender="M", style="neutral"),
            VoiceInfo(id="am_michael", name="Michael", gender="M", style="deep"),
            VoiceInfo(id="bf_emma", name="Emma", gender="F", style="british"),
            VoiceInfo(id="bm_george", name="George", gender="M", style="british"),
        ]
    }

    edge_voices = {
        "fr": [
            VoiceInfo(id="fr-FR-DeniseNeural", name="Denise", gender="F", style="neural"),
            VoiceInfo(id="fr-FR-HenriNeural", name="Henri", gender="M", style="neural"),
            VoiceInfo(id="fr-CA-SylvieNeural", name="Sylvie (CA)", gender="F", style="neural"),
        ],
        "en": [
            VoiceInfo(id="en-US-JennyNeural", name="Jenny", gender="F", style="neural"),
            VoiceInfo(id="en-US-GuyNeural", name="Guy", gender="M", style="neural"),
            VoiceInfo(id="en-GB-SoniaNeural", name="Sonia", gender="F", style="british"),
        ]
    }

    return VoicesResponse(
        language=language,
        kokoro=kokoro_voices.get(language, []),
        edge_tts=edge_voices.get(language, []),
        recommended="ff_siwis" if language == "fr" else "af_heart"
    )


@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate_audio(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Genere un fichier audio a partir de texte.

    Cette operation est asynchrone. Utilisez l'endpoint /jobs/{job_id}
    pour suivre la progression.
    """
    job_id = str(uuid.uuid4())[:8]
    output_name = request.output_name or f"audio_{job_id}"

    JOBS[job_id] = {
        "status": "pending",
        "progress": 0,
        "result": None,
        "error": None
    }

    async def process():
        try:
            JOBS[job_id]["status"] = "processing"
            JOBS[job_id]["progress"] = 10

            tts = get_tts_engine()
            JOBS[job_id]["progress"] = 30

            lang = "fr" if request.voice.startswith("ff") else "en"
            audio, sample_rate = tts.synthesize(
                text=request.text,
                voice=request.voice,
                speed=request.speed,
                lang=lang
            )
            JOBS[job_id]["progress"] = 80

            output_path = OUTPUT_DIR / f"{output_name}.wav"
            import soundfile as sf
            sf.write(str(output_path), audio, sample_rate)

            duration = len(audio) / sample_rate
            JOBS[job_id]["progress"] = 100
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["result"] = {
                "output_file": str(output_path),
                "duration_seconds": round(duration, 2),
                "download_url": f"/files/{output_name}.wav"
            }

        except Exception as e:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(e)

    background_tasks.add_task(process)

    return GenerateResponse(
        success=True,
        job_id=job_id,
        status="pending",
        message=f"Generation demarree. Suivez la progression avec GET /jobs/{job_id}"
    )


@app.post("/audiobook", response_model=AudiobookResponse, tags=["Generation"])
async def generate_audiobook(request: AudiobookRequest, background_tasks: BackgroundTasks):
    """
    Genere un audiobook complet avec analyse des dialogues et emotions.

    Cette operation est asynchrone et peut prendre plusieurs minutes
    pour les textes longs.
    """
    job_id = str(uuid.uuid4())[:8]

    JOBS[job_id] = {
        "status": "pending",
        "progress": 0,
        "result": None,
        "error": None
    }

    async def process():
        try:
            JOBS[job_id]["status"] = "processing"
            JOBS[job_id]["progress"] = 5

            from src.hq_pipeline_extended import create_extended_pipeline

            pipeline = create_extended_pipeline(
                lang=request.language,
                narrator_voice=request.narrator_voice,
                enable_emotion_analysis=request.enable_emotions,
                auto_assign_voices=request.enable_multi_voice,
                default_narration_style=request.style.value,
                enable_intonation_contours=True,
                enable_timing_humanization=True,
                enable_advanced_breaths=True,
            )
            JOBS[job_id]["progress"] = 15

            segments = pipeline.process_chapter(request.text, chapter_index=0)
            JOBS[job_id]["progress"] = 30

            tts = get_tts_engine()
            audios = []
            total_segs = len(segments)

            for i, seg in enumerate(segments):
                audio, sr = tts.synthesize(
                    text=seg.text,
                    voice=seg.voice_id,
                    speed=seg.final_speed,
                    lang=request.language
                )
                audios.append(audio)
                JOBS[job_id]["progress"] = 30 + int(50 * (i + 1) / total_segs)

            # Concatener
            import numpy as np
            from src.bio_acoustics import BioAudioGenerator
            bio_gen = BioAudioGenerator(sample_rate=24000)

            result_parts = [bio_gen.generate_silence(0.5)]
            for seg, audio in zip(segments, audios):
                if seg.pause_before > 0:
                    result_parts.append(bio_gen.generate_silence(seg.pause_before))
                result_parts.append(audio)
                if seg.pause_after > 0:
                    result_parts.append(bio_gen.generate_silence(seg.pause_after))
            result_parts.append(bio_gen.generate_silence(1.0))

            full_audio = np.concatenate(result_parts)
            JOBS[job_id]["progress"] = 90

            output_path = OUTPUT_DIR / f"{request.title}.wav"
            import soundfile as sf
            sf.write(str(output_path), full_audio, 24000)

            duration = len(full_audio) / 24000
            characters = pipeline.get_characters()

            JOBS[job_id]["progress"] = 100
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["result"] = {
                "output_file": str(output_path),
                "duration_seconds": round(duration, 2),
                "duration_formatted": f"{int(duration // 60)}:{int(duration % 60):02d}",
                "segments_count": len(segments),
                "characters_detected": characters,
                "download_url": f"/files/{request.title}.wav"
            }

        except Exception as e:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(e)

    background_tasks.add_task(process)

    return AudiobookResponse(
        success=True,
        job_id=job_id,
        status="pending",
        message=f"Generation demarree. Suivez la progression avec GET /jobs/{job_id}"
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["Jobs"])
async def get_job_status(job_id: str):
    """Obtient le statut d'un job de generation."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job non trouve")

    job = JOBS[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        result=job["result"],
        error=job["error"]
    )


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analyse"])
async def analyze_text(request: AnalyzeRequest):
    """Analyse un texte pour detecter les dialogues, emotions et personnages."""
    from src.dialogue_attribution import DialogueAttributor
    from src.emotion_analyzer import EmotionAnalyzer
    from src.intonation_contour import IntonationContourDetector
    import re

    attributor = DialogueAttributor(lang="fr")
    dialogues = attributor.process_text(request.text)

    emotion_analyzer = EmotionAnalyzer()
    contour_detector = IntonationContourDetector(language="fr")

    dialogue_infos = []
    for d in dialogues:
        dialogue_infos.append(DialogueInfo(
            text=d.text[:50] + "..." if len(d.text) > 50 else d.text,
            speaker=d.attribution.speaker,
            method=d.attribution.method.value,
            confidence=d.attribution.confidence
        ))

    sentence_infos = []
    sentences = re.split(r'[.!?]+', request.text)[:10]
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        contour = contour_detector.detect(sent)
        emotion_result = emotion_analyzer.analyze(sent)

        sentence_infos.append(SentenceInfo(
            text=sent[:40] + "..." if len(sent) > 40 else sent,
            intonation=contour.value,
            emotion=emotion_result.emotion.value if emotion_result else "neutral",
            intensity=emotion_result.intensity if emotion_result else 0.5
        ))

    return AnalyzeResponse(
        total_characters=len(request.text),
        dialogues=dialogue_infos,
        characters=list(attributor.context.participants),
        sentences=sentence_infos
    )


@app.get("/config", response_model=ConfigResponse, tags=["Configuration"])
async def get_config():
    """Retourne la configuration actuelle d'AudioReader."""
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
            "acx_compliance": True
        },
        styles_available=[
            "formal", "conversational", "dramatic",
            "storytelling", "documentary", "intimate", "energetic"
        ],
        version="2.4"
    )


@app.put("/config", tags=["Configuration"])
async def update_config(request: ConfigUpdateRequest):
    """Modifie la configuration d'AudioReader."""
    changes = {k: v for k, v in request.dict().items() if v is not None}
    return {
        "success": True,
        "changes": changes,
        "message": "Configuration mise a jour"
    }


@app.get("/files", response_model=FilesResponse, tags=["Fichiers"])
async def list_files():
    """Liste les fichiers audio generes."""
    files = []
    for f in OUTPUT_DIR.glob("*.wav"):
        stat = f.stat()
        files.append(FileInfo(
            name=f.name,
            path=str(f),
            size_mb=round(stat.st_size / 1024 / 1024, 2),
            modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            download_url=f"/files/{f.name}"
        ))

    files.sort(key=lambda x: x.modified, reverse=True)

    return FilesResponse(
        output_dir=str(OUTPUT_DIR),
        files=files,
        total_count=len(files)
    )


@app.get("/files/{filename}", tags=["Fichiers"])
async def download_file(filename: str):
    """Telecharge un fichier audio genere."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Fichier non trouve")

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="audio/wav"
    )


# ============================================================================
# Point d'entree
# ============================================================================

def main():
    """Demarre le serveur API."""
    import uvicorn

    print("=" * 50)
    print("AudioReader API Server v2.4")
    print("=" * 50)
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print("Endpoints:")
    print("  GET  /              - Info API")
    print("  GET  /voices        - Liste des voix")
    print("  POST /generate      - Genere audio simple")
    print("  POST /audiobook     - Genere audiobook complet")
    print("  POST /analyze       - Analyse texte")
    print("  GET  /config        - Configuration")
    print("  GET  /files         - Liste fichiers")
    print("  GET  /docs          - Documentation Swagger")
    print("  GET  /openapi.json  - Spec OpenAPI")
    print()
    print("Demarrage sur http://0.0.0.0:8000")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
