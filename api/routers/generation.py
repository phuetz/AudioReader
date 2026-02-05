"""Endpoints de génération audio : TTS simple, audiobook, preview."""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks

from api.dependencies import OUTPUT_DIR, get_tts_engine, job_store
from api.errors import APIError, ErrorCode
from api.models import (
    AudiobookRequest,
    GenerateRequest,
    JobCreatedResponse,
    JobStatus,
    PreviewRequest,
)

router = APIRouter(prefix="/api/v2", tags=["Generation"])


@router.post("/generate", response_model=JobCreatedResponse)
async def generate_audio(request: GenerateRequest, background_tasks: BackgroundTasks):
    """TTS simple — retourne un job_id pour suivi SSE."""
    job_id = job_store.create("generate")

    async def process():
        try:
            job_store.update(job_id, status=JobStatus.processing, progress=10, phase="init")
            tts = get_tts_engine()
            job_store.update(job_id, progress=30, phase="synthesis")

            if job_store.is_cancelled(job_id):
                return

            audio, sample_rate = tts.synthesize(
                text=request.text,
                voice=request.voice,
                speed=request.speed,
                lang=request.language,
            )
            job_store.update(job_id, progress=80, phase="saving")

            output_name = request.output_name or f"audio_{job_id}"
            output_path = OUTPUT_DIR / f"{output_name}.wav"

            import soundfile as sf
            sf.write(str(output_path), audio, sample_rate)

            duration = len(audio) / sample_rate
            job_store.update(
                job_id,
                status=JobStatus.completed,
                progress=100,
                phase="done",
                result={
                    "output_file": f"{output_name}.wav",
                    "duration_seconds": round(duration, 2),
                    "download_url": f"/output/{output_name}.wav",
                },
            )
        except Exception as e:
            job_store.update(job_id, status=JobStatus.failed, error=str(e))

    background_tasks.add_task(process)
    return JobCreatedResponse(job_id=job_id, message="Génération démarrée")


@router.post("/audiobook", response_model=JobCreatedResponse)
async def generate_audiobook(request: AudiobookRequest, background_tasks: BackgroundTasks):
    """Pipeline HQ complet — retourne un job_id pour suivi SSE."""
    if not request.text and not request.file_id:
        raise APIError(ErrorCode.VALIDATION_ERROR, "text ou file_id requis")

    job_id = job_store.create("audiobook")

    async def process():
        try:
            text = request.text
            # Si file_id, lire le fichier converti
            if not text and request.file_id:
                from api.dependencies import file_store
                path = file_store.get_path(request.file_id)
                if not path:
                    job_store.update(job_id, status=JobStatus.failed, error="Fichier non trouvé")
                    return
                text = path.read_text(encoding="utf-8")

            job_store.update(job_id, status=JobStatus.processing, progress=5, phase="pipeline_init")

            from src.hq_pipeline_extended import create_extended_pipeline

            pipeline = create_extended_pipeline(
                lang=request.language,
                narrator_voice=request.narrator_voice,
                enable_emotion_analysis=request.enable_emotions,
                auto_assign_voices=request.enable_multi_voice,
                default_narration_style=request.style.value,
                enable_intonation_contours=request.enable_intonation_contours,
                enable_timing_humanization=request.enable_timing_humanization,
                enable_advanced_breaths=True,
                # v5.0: LLM Enhancement
                enable_llm_enhancer=request.enable_llm_enhance,
                llm_enhancer_provider=request.llm_provider.value,
                llm_enhancer_model=request.llm_model or "",
                # v5.0: Sound Effects
                enable_sound_effects=request.enable_sound_effects,
                sound_effects_intensity=request.sound_effects_intensity,
                # v5.0: Timing
                pause_variation_sigma=request.pause_variation,
                # v5.0: ACX Compliance
                enable_acx_compliance=request.enable_acx_compliance,
                acx_target_lufs=request.acx_target_lufs,
            )
            job_store.update(job_id, progress=15, phase="text_analysis")

            segments = pipeline.process_chapter(text, chapter_index=0)
            job_store.update(job_id, progress=30, phase="synthesis", segments_total=len(segments))

            if job_store.is_cancelled(job_id):
                return

            tts = get_tts_engine()
            audios = []
            total_segs = len(segments)

            for i, seg in enumerate(segments):
                if job_store.is_cancelled(job_id):
                    return
                audio, sr = tts.synthesize(
                    text=seg.text,
                    voice=seg.voice_id,
                    speed=seg.final_speed,
                    lang=request.language,
                )
                audios.append(audio)
                job_store.update(
                    job_id,
                    progress=30 + int(50 * (i + 1) / total_segs),
                    phase="synthesis",
                    segments_done=i + 1,
                )

            job_store.update(job_id, progress=85, phase="assembly")

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
            job_store.update(job_id, progress=90, phase="saving")

            output_path = OUTPUT_DIR / f"{request.title}.wav"
            import soundfile as sf
            sf.write(str(output_path), full_audio, 24000)

            duration = len(full_audio) / 24000
            characters = pipeline.get_characters()

            job_store.update(
                job_id,
                status=JobStatus.completed,
                progress=100,
                phase="done",
                result={
                    "output_file": f"{request.title}.wav",
                    "duration_seconds": round(duration, 2),
                    "duration_formatted": f"{int(duration // 60)}:{int(duration % 60):02d}",
                    "segments_count": len(segments),
                    "characters_detected": characters,
                    "download_url": f"/output/{request.title}.wav",
                },
            )
        except Exception as e:
            job_store.update(job_id, status=JobStatus.failed, error=str(e))

    background_tasks.add_task(process)
    return JobCreatedResponse(job_id=job_id, message="Génération audiobook démarrée")


@router.post("/preview", response_model=JobCreatedResponse)
async def generate_preview(request: PreviewRequest, background_tasks: BackgroundTasks):
    """Génère un preview 30s — retourne un job_id."""
    job_id = job_store.create("preview")

    async def process():
        try:
            job_store.update(job_id, status=JobStatus.processing, progress=10, phase="preview")

            from src.preview_generator import generate_quick_preview

            output_name = f"preview_{job_id}.wav"
            output_path = OUTPUT_DIR / output_name

            success, msg = generate_quick_preview(
                text=request.text,
                output_path=str(output_path),
                duration=request.duration,
            )

            if success:
                import soundfile as sf
                info = sf.info(str(output_path))
                job_store.update(
                    job_id,
                    status=JobStatus.completed,
                    progress=100,
                    phase="done",
                    result={
                        "output_file": output_name,
                        "duration_seconds": round(info.duration, 2),
                        "download_url": f"/output/{output_name}",
                    },
                )
            else:
                job_store.update(job_id, status=JobStatus.failed, error=msg)
        except Exception as e:
            job_store.update(job_id, status=JobStatus.failed, error=str(e))

    background_tasks.add_task(process)
    return JobCreatedResponse(job_id=job_id, message="Génération preview démarrée")
