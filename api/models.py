"""Modèles Pydantic — requêtes, réponses, événements SSE."""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class NarrationStyle(str, Enum):
    formal = "formal"
    conversational = "conversational"
    dramatic = "dramatic"
    storytelling = "storytelling"
    documentary = "documentary"
    intimate = "intimate"
    energetic = "energetic"


class OutputFormat(str, Enum):
    wav = "wav"
    mp3 = "mp3"
    m4b = "m4b"


class LLMProvider(str, Enum):
    """Providers LLM pour l'amélioration du texte."""
    ollama = "ollama"
    openai = "openai"
    anthropic = "anthropic"
    gemini = "gemini"


# ── Voix ─────────────────────────────────────────────────────────────────────

class VoiceInfo(BaseModel):
    id: str
    name: str
    gender: str
    language: str
    engine: str  # kokoro | edge | cloned
    style: str = "neutral"
    preview_url: Optional[str] = None


class VoicePreviewRequest(BaseModel):
    voice_id: str
    text: str = "Bonjour, je suis une voix de synthèse."
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    language: str = "fr"


class VoiceCloneRequest(BaseModel):
    name: str
    language: str = "fr"


# ── Génération ───────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Texte à convertir")
    voice: str = Field(default="ff_siwis")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    language: str = "fr"
    output_name: Optional[str] = None


class AudiobookRequest(BaseModel):
    text: Optional[str] = None
    file_id: Optional[str] = None
    title: str = "audiobook"
    narrator_voice: str = "ff_siwis"
    style: NarrationStyle = NarrationStyle.storytelling
    enable_emotions: bool = True
    enable_multi_voice: bool = True
    language: str = "fr"
    output_format: OutputFormat = OutputFormat.wav
    enable_mastering: bool = False
    character_voices: Optional[Dict[str, str]] = None  # {personnage: voice_id}

    # v5.0: LLM Enhancement
    enable_llm_enhance: bool = False
    llm_provider: LLMProvider = LLMProvider.ollama
    llm_model: Optional[str] = None  # Auto-select if None

    # v5.0: Sound Effects
    enable_sound_effects: bool = False
    sound_effects_intensity: float = Field(default=0.3, ge=0.0, le=1.0)

    # v5.0: Advanced Options
    enable_subtitles: bool = False
    subtitle_format: str = "srt"  # srt, vtt, json

    # v5.0: Timing & Prosody
    enable_timing_humanization: bool = True
    pause_variation: float = Field(default=0.15, ge=0.0, le=0.5)  # 15% variation
    enable_intonation_contours: bool = True

    # v5.0: ACX Compliance
    enable_acx_compliance: bool = False
    acx_target_lufs: float = Field(default=-19.0, ge=-24.0, le=-14.0)


class PreviewRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = "ff_siwis"
    speed: float = 1.0
    language: str = "fr"
    duration: float = Field(default=30.0, le=60.0)


# ── Jobs ─────────────────────────────────────────────────────────────────────

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = 0
    phase: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


class JobCreatedResponse(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.pending
    message: str = ""


# ── Fichiers ─────────────────────────────────────────────────────────────────

class FileInfo(BaseModel):
    id: str
    name: str
    path: str
    size_mb: float
    mime_type: str
    created_at: str
    download_url: str


class UploadResponse(BaseModel):
    file_id: str
    original_name: str
    converted_path: Optional[str] = None
    mime_type: str
    chapters: Optional[List[str]] = None
    text_preview: Optional[str] = None


# ── Analyse ──────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: Optional[str] = None
    file_id: Optional[str] = None
    language: str = "fr"


class CharacterInfo(BaseModel):
    name: str
    gender: str
    dialogue_count: int
    suggested_voice: Optional[str] = None


class DialogueInfo(BaseModel):
    text: str
    speaker: str
    method: str
    confidence: float


class EmotionInfo(BaseModel):
    text: str
    emotion: str
    intensity: float
    intonation: str


class AnalysisResult(BaseModel):
    total_characters: int
    characters: List[CharacterInfo]
    dialogues: List[DialogueInfo]
    emotions: List[EmotionInfo]
    chapter_count: int = 0
    word_count: int = 0


# ── Projets ──────────────────────────────────────────────────────────────────

class ProjectCreate(BaseModel):
    name: str
    description: str = ""


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None


class ProjectInfo(BaseModel):
    id: str
    name: str
    description: str = ""
    settings: Dict[str, Any] = {}
    created_at: str = ""
    updated_at: str = ""
    files: List[str] = []


# ── Podcast ──────────────────────────────────────────────────────────────────

class PodcastStartRequest(BaseModel):
    audio_dir: str = "output"
    port: int = Field(default=8080, ge=1024, le=65535)
    title: str = "AudioReader Podcast"


class PodcastStatusResponse(BaseModel):
    running: bool
    url: Optional[str] = None
    port: Optional[int] = None
    qr_code_url: Optional[str] = None
    episode_count: int = 0


# ── Config / Health ──────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "4.0.0"
    engines: Dict[str, bool] = {}
    uptime_seconds: float = 0


class ConfigResponse(BaseModel):
    output_dir: str
    default_voice: str
    default_language: str
    features: Dict[str, bool]
    styles_available: List[str]
    version: str


# ── Événements SSE ───────────────────────────────────────────────────────────

class SSEEvent(BaseModel):
    event: str
    data: Dict[str, Any]
