"""Endpoint d'analyse de texte : personnages, émotions, dialogues."""
from __future__ import annotations

import re
from typing import Optional

from fastapi import APIRouter

from api.dependencies import file_store
from api.errors import APIError, ErrorCode
from api.models import (
    AnalysisResult,
    AnalyzeRequest,
    CharacterInfo,
    DialogueInfo,
    EmotionInfo,
)

router = APIRouter(prefix="/api/v2", tags=["Analysis"])


@router.post("/analyze", response_model=AnalysisResult)
async def analyze_text(request: AnalyzeRequest):
    """Analyse un texte : détection personnages, émotions, dialogues."""
    text = request.text
    if not text and request.file_id:
        path = file_store.get_path(request.file_id)
        if not path:
            raise APIError(ErrorCode.NOT_FOUND, "Fichier non trouvé", status_code=404)
        text = path.read_text(encoding="utf-8")

    if not text:
        raise APIError(ErrorCode.VALIDATION_ERROR, "text ou file_id requis")

    # Dialogue attribution
    from src.dialogue_attribution import DialogueAttributor
    attributor = DialogueAttributor(lang=request.language)
    dialogues_raw = attributor.process_text(text)

    dialogues = [
        DialogueInfo(
            text=d.text[:80] + ("..." if len(d.text) > 80 else ""),
            speaker=d.attribution.speaker,
            method=d.attribution.method.value,
            confidence=round(d.attribution.confidence, 2),
        )
        for d in dialogues_raw
    ]

    # Personnages avec stats
    char_counts: dict[str, int] = {}
    for d in dialogues_raw:
        name = d.attribution.speaker
        char_counts[name] = char_counts.get(name, 0) + 1

    characters = []
    for name, count in sorted(char_counts.items(), key=lambda x: -x[1]):
        gender = "?"
        if hasattr(attributor, 'context') and hasattr(attributor.context, 'characters'):
            for c in getattr(attributor.context, 'characters', {}).values():
                if getattr(c, 'name', '') == name:
                    gender = getattr(c, 'gender', '?')
                    break
        characters.append(CharacterInfo(
            name=name,
            gender=gender,
            dialogue_count=count,
        ))

    # Émotions sur un échantillon
    from src.emotion_analyzer import EmotionAnalyzer
    from src.intonation_contour import IntonationContourDetector

    emotion_analyzer = EmotionAnalyzer()
    contour_detector = IntonationContourDetector(language=request.language)

    emotions = []
    sentences = re.split(r'[.!?]+', text)[:20]
    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 5:
            continue
        contour = contour_detector.detect(sent)
        emotion_result = emotion_analyzer.analyze(sent)
        emotions.append(EmotionInfo(
            text=sent[:60] + ("..." if len(sent) > 60 else ""),
            emotion=emotion_result.emotion.value if emotion_result else "neutral",
            intensity=round(emotion_result.intensity if emotion_result else 0.5, 2),
            intonation=contour.value,
        ))

    word_count = len(text.split())
    chapter_count = text.count("\n# ") + (1 if text.startswith("# ") else 0)

    return AnalysisResult(
        total_characters=len(text),
        characters=characters,
        dialogues=dialogues[:50],
        emotions=emotions,
        chapter_count=max(chapter_count, 1),
        word_count=word_count,
    )
