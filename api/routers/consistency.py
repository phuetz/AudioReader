"""Détection d'incohérences — voix, personnages, émotions."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from api.dependencies import job_store
from api.errors import APIError, ErrorCode

router = APIRouter(prefix="/api/v2", tags=["Consistency"])


class ConsistencyIssue(BaseModel):
    type: str  # voice_change, attribution_conflict, emotion_jump
    severity: str  # info, warning, error
    segment_indices: List[int] = []
    description: str
    suggestion: str = ""


class ConsistencyReport(BaseModel):
    job_id: str
    score: float = 100.0  # 0-100, higher is better
    issues: List[ConsistencyIssue] = []
    total_segments: int = 0
    summary: str = ""


@router.post("/jobs/{job_id}/consistency")
async def analyze_consistency(job_id: str) -> ConsistencyReport:
    """Analyse les incohérences dans un job."""
    job = await job_store.get(job_id)
    if not job:
        raise APIError(ErrorCode.NOT_FOUND, f"Job {job_id} non trouvé", status_code=404)

    # Get segments from review store
    from api.routers.review import get_segments_store
    segments_store = get_segments_store()
    segments = segments_store.get(job_id, [])

    issues = []
    total = len(segments)

    if not segments:
        return ConsistencyReport(
            job_id=job_id,
            score=100.0,
            issues=[],
            total_segments=0,
            summary="Aucun segment disponible pour l'analyse.",
        )

    # Check 1: Voice consistency per speaker
    speaker_voices: dict[str, set[str]] = {}
    for i, seg in enumerate(segments):
        speaker = seg.get("speaker")
        voice = seg.get("voice_id", "")
        if speaker:
            speaker_voices.setdefault(speaker, set()).add(voice)
            if len(speaker_voices[speaker]) > 1:
                issues.append(ConsistencyIssue(
                    type="voice_change",
                    severity="warning",
                    segment_indices=[i],
                    description=f"{speaker} utilise des voix différentes: {', '.join(speaker_voices[speaker])}",
                    suggestion=f"Uniformiser la voix de {speaker} sur tous les segments.",
                ))

    # Check 2: Emotion continuity
    prev_emotion = None
    EMOTION_TRANSITIONS = {
        ("joy", "sadness"): "warning",
        ("fear", "joy"): "info",
        ("anger", "joy"): "info",
        ("neutral", "anger"): "info",
    }
    for i, seg in enumerate(segments):
        emotion = seg.get("emotion")
        if emotion and prev_emotion:
            pair = (prev_emotion, emotion)
            if pair in EMOTION_TRANSITIONS:
                severity = EMOTION_TRANSITIONS[pair]
                issues.append(ConsistencyIssue(
                    type="emotion_jump",
                    severity=severity,
                    segment_indices=[i - 1, i],
                    description=f"Transition émotionnelle brusque: {prev_emotion} → {emotion}",
                    suggestion="Vérifier si cette transition est intentionnelle.",
                ))
        prev_emotion = emotion

    # Check 3: Attribution consistency
    seen_speakers = set()
    for i, seg in enumerate(segments):
        speaker = seg.get("speaker")
        if speaker and speaker not in seen_speakers and i > 0:
            # New speaker appearing late might be misattribution
            if i > total * 0.8:
                issues.append(ConsistencyIssue(
                    type="attribution_conflict",
                    severity="info",
                    segment_indices=[i],
                    description=f"Nouveau personnage '{speaker}' apparaît tard dans le texte (segment {i}/{total})",
                    suggestion="Vérifier que ce n'est pas une erreur d'attribution.",
                ))
        if speaker:
            seen_speakers.add(speaker)

    # Calculate score
    penalty = sum(
        {"error": 10, "warning": 5, "info": 2}.get(issue.severity, 0)
        for issue in issues
    )
    score = max(0.0, 100.0 - penalty)

    summary_parts = []
    if not issues:
        summary_parts.append("Aucune incohérence détectée.")
    else:
        errors = sum(1 for i in issues if i.severity == "error")
        warnings = sum(1 for i in issues if i.severity == "warning")
        infos = sum(1 for i in issues if i.severity == "info")
        if errors:
            summary_parts.append(f"{errors} erreur(s)")
        if warnings:
            summary_parts.append(f"{warnings} avertissement(s)")
        if infos:
            summary_parts.append(f"{infos} info(s)")

    return ConsistencyReport(
        job_id=job_id,
        score=round(score, 1),
        issues=issues,
        total_segments=total,
        summary=", ".join(summary_parts) if summary_parts else "Analyse terminée.",
    )
