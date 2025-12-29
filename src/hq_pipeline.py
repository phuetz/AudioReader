"""
Pipeline haute qualite unifie pour audiobooks.

Integre tous les modules avances:
- Normalisation du texte (nombres, dates, abreviations)
- Detection de personnages et multi-voix
- Analyse emotionnelle et contexte narratif
- Continuite emotionnelle
- Synthese TTS avec prosodie adaptative
- Post-processing audio

Objectif: Qualite proche d'ElevenLabs avec Kokoro.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable
import json

# Import des modules
from .text_normalizer import TextNormalizer
from .character_detector import (
    CharacterDetector,
    VoiceAssigner,
    DialogueSegment,
    SpeakerType,
    Character
)
from .emotion_analyzer import (
    EmotionAnalyzer,
    EmotionAnalysis,
    Emotion,
    Intensity,
    ProsodyHints
)
from .narrative_context import (
    NarrativeContextDetector,
    NarrativeContext,
    NarrativeType
)
from .emotion_continuity import (
    EmotionContinuityManager,
    ChapterEmotionTracker,
    apply_emotion_continuity
)
from .text_processor import PronunciationCorrector, TextChunker
from .advanced_preprocessor import EnrichedSegment


@dataclass
class HQPipelineConfig:
    """Configuration complete du pipeline haute qualite."""

    # Langue
    lang: str = "fr"

    # Moteur TTS
    tts_engine: str = "auto"  # "auto", "kokoro", "edge-tts"

    # Voix
    narrator_voice: str = "ff_siwis"
    voice_mapping: dict[str, str] = field(default_factory=dict)
    auto_assign_voices: bool = True

    # Normalisation du texte
    normalize_numbers: bool = True
    normalize_dates: bool = True
    normalize_abbreviations: bool = True

    # Detection de personnages
    enable_character_detection: bool = True
    min_character_occurrences: int = 2

    # Analyse emotionnelle
    enable_emotion_analysis: bool = True
    enable_prosody_hints: bool = True
    emotion_smoothing: float = 0.3  # Lissage des transitions

    # Contexte narratif
    enable_narrative_context: bool = True
    adapt_speed_to_context: bool = True

    # Pauses
    sentence_pause: float = 0.3
    paragraph_pause: float = 0.8
    dialogue_pause: float = 0.4
    chapter_pause_start: float = 1.0
    chapter_pause_end: float = 2.0

    # Chunking
    max_chunk_size: int = 500

    # Corrections de prononciation
    enable_pronunciation_correction: bool = True
    custom_pronunciation: Optional[dict] = None

    # Post-processing audio
    enable_audio_enhancement: bool = True
    target_lufs: float = -19.0
    enable_deessing: bool = True
    enable_compression: bool = True

    # Crossfade
    crossfade_ms: int = 50

    def save(self, path: Path):
        """Sauvegarde la configuration."""
        data = {
            "lang": self.lang,
            "tts_engine": self.tts_engine,
            "narrator_voice": self.narrator_voice,
            "voice_mapping": self.voice_mapping,
            "auto_assign_voices": self.auto_assign_voices,
            "normalize_numbers": self.normalize_numbers,
            "normalize_dates": self.normalize_dates,
            "enable_character_detection": self.enable_character_detection,
            "enable_emotion_analysis": self.enable_emotion_analysis,
            "enable_narrative_context": self.enable_narrative_context,
            "emotion_smoothing": self.emotion_smoothing,
            "sentence_pause": self.sentence_pause,
            "paragraph_pause": self.paragraph_pause,
            "enable_audio_enhancement": self.enable_audio_enhancement,
            "target_lufs": self.target_lufs,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "HQPipelineConfig":
        """Charge la configuration."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class HQSegment:
    """
    Segment enrichi haute qualite.

    Contient toutes les metadonnees pour une synthese optimale.
    """
    text: str
    index: int

    # Locuteur
    speaker: str
    speaker_type: SpeakerType
    voice_id: str

    # Emotion
    emotion: Emotion
    intensity: Intensity
    prosody: ProsodyHints

    # Contexte narratif
    narrative_type: NarrativeType
    narrative_confidence: float

    # TTS
    tts_tags: list[str] = field(default_factory=list)

    # Timing
    pause_before: float = 0.0
    pause_after: float = 0.0

    # Metadata
    is_dialogue: bool = False
    is_internal_thought: bool = False
    chapter_index: int = 0

    # Vitesse finale calculee
    final_speed: float = 1.0


class HQPipeline:
    """
    Pipeline haute qualite pour audiobooks.

    Combine tous les modules pour produire des audiobooks
    de qualite professionnelle.
    """

    def __init__(self, config: Optional[HQPipelineConfig] = None):
        self.config = config or HQPipelineConfig()

        # Initialiser les composants
        self._init_components()

        # Etat
        self._characters: list[Character] = []
        self._voice_assignments: dict[str, str] = {}
        self._chapter_tracker = ChapterEmotionTracker()

    def _init_components(self):
        """Initialise tous les composants du pipeline."""
        # Normalisation
        self.text_normalizer = TextNormalizer(self.config.lang)

        # Prononciation
        self.pronunciation = PronunciationCorrector(
            lang=self.config.lang,
            custom_dict=self.config.custom_pronunciation
        )

        # Chunking
        self.chunker = TextChunker(max_chars=self.config.max_chunk_size)

        # Personnages
        self.character_detector = CharacterDetector(lang=self.config.lang)
        self.voice_assigner = VoiceAssigner(
            narrator_voice=self.config.narrator_voice,
            lang=self.config.lang,
            voice_mapping=self.config.voice_mapping
        )

        # Emotions
        self.emotion_analyzer = EmotionAnalyzer(lang=self.config.lang)
        self.emotion_continuity = EmotionContinuityManager(
            smoothing_factor=self.config.emotion_smoothing
        )

        # Contexte narratif
        self.narrative_detector = NarrativeContextDetector(lang=self.config.lang)

    def _normalize_text(self, text: str) -> str:
        """Applique toutes les normalisations de texte."""
        # Normalisation avancee (nombres, dates, etc.)
        if (self.config.normalize_numbers or
                self.config.normalize_dates or
                self.config.normalize_abbreviations):
            text = self.text_normalizer.normalize(text)

        # Corrections de prononciation
        if self.config.enable_pronunciation_correction:
            text = self.pronunciation.correct(text)

        return text

    def _calculate_final_speed(
        self,
        emotion_prosody: ProsodyHints,
        narrative_context: NarrativeContext
    ) -> float:
        """
        Calcule la vitesse finale en combinant tous les facteurs.

        Combine:
        - Vitesse de base
        - Ajustement emotionnel
        - Ajustement narratif
        """
        speed = 1.0

        # Ajustement emotionnel
        if self.config.enable_prosody_hints:
            speed *= emotion_prosody.speed

        # Ajustement narratif
        if self.config.enable_narrative_context and self.config.adapt_speed_to_context:
            speed *= narrative_context.suggested_speed

        # Limiter dans une plage raisonnable
        return max(0.7, min(1.3, speed))

    def _calculate_pauses(
        self,
        segment: DialogueSegment,
        narrative_context: NarrativeContext,
        emotion_prosody: ProsodyHints,
        next_segment: Optional[DialogueSegment]
    ) -> tuple[float, float]:
        """Calcule les pauses optimales avant/apres le segment."""
        pause_before = 0.0
        pause_after = self.config.sentence_pause

        # Pause du contexte narratif
        pause_before = max(pause_before, narrative_context.suggested_pause_before)
        pause_after = max(pause_after, narrative_context.suggested_pause_after)

        # Pause emotionnelle
        pause_before = max(pause_before, emotion_prosody.pause_before)
        pause_after = max(pause_after, emotion_prosody.pause_after)

        # Respiration
        if emotion_prosody.breath_before:
            pause_before = max(pause_before, 0.2)

        # Pause de dialogue
        if segment.speaker_type == SpeakerType.CHARACTER:
            pause_after = max(pause_after, self.config.dialogue_pause)

        # Pause entre personnages differents
        if next_segment and next_segment.speaker != segment.speaker:
            pause_after = max(pause_after, self.config.dialogue_pause)

        return pause_before, pause_after

    def process_chapter(
        self,
        text: str,
        chapter_index: int = 0,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> list[HQSegment]:
        """
        Traite un chapitre complet.

        Args:
            text: Texte du chapitre
            chapter_index: Index du chapitre
            progress_callback: Fonction de callback (step, total, message)

        Returns:
            Liste de segments haute qualite prets pour TTS
        """
        total_steps = 6
        step = 0

        # Etape 1: Normalisation du texte
        if progress_callback:
            progress_callback(step, total_steps, "Normalisation du texte")
        step += 1
        text = self._normalize_text(text)

        # Etape 2: Detection des personnages
        if progress_callback:
            progress_callback(step, total_steps, "Detection des personnages")
        step += 1

        if self.config.enable_character_detection:
            dialogue_segments = self.character_detector.detect_dialogue_segments(text)
            self._characters = self.character_detector.get_characters()
        else:
            dialogue_segments = [
                DialogueSegment(
                    text=text,
                    speaker="NARRATOR",
                    speaker_type=SpeakerType.NARRATOR,
                    index=0
                )
            ]

        # Etape 3: Attribution des voix
        if progress_callback:
            progress_callback(step, total_steps, "Attribution des voix")
        step += 1

        if self.config.auto_assign_voices:
            significant_chars = [
                c for c in self._characters
                if c.occurrence_count >= self.config.min_character_occurrences
            ]
            self._voice_assignments = self.voice_assigner.assign_voices_to_characters(
                significant_chars
            )

        # Etape 4: Analyse emotionnelle et contexte
        if progress_callback:
            progress_callback(step, total_steps, "Analyse emotionnelle")
        step += 1

        # Reset du gestionnaire de continuite pour ce chapitre
        self.emotion_continuity.reset()
        self._chapter_tracker = ChapterEmotionTracker()

        # Etape 5: Creation des segments HQ
        if progress_callback:
            progress_callback(step, total_steps, "Creation des segments")
        step += 1

        hq_segments = []

        for i, seg in enumerate(dialogue_segments):
            # Voix
            if seg.speaker == "NARRATOR":
                voice_id = self.config.narrator_voice
            elif seg.speaker in self._voice_assignments:
                voice_id = self._voice_assignments[seg.speaker]
            else:
                voice_id = self.voice_assigner.get_voice_for_segment(seg)

            # Analyse emotionnelle
            if self.config.enable_emotion_analysis:
                emotion_analysis = self.emotion_analyzer.analyze(seg.text)
                emotion = emotion_analysis.emotion
                intensity = emotion_analysis.intensity
                prosody = emotion_analysis.prosody
                tts_tags = emotion_analysis.tags
            else:
                emotion = Emotion.NEUTRAL
                intensity = Intensity.MEDIUM
                prosody = ProsodyHints()
                tts_tags = []

            # Contexte narratif
            if self.config.enable_narrative_context:
                narrative = self.narrative_detector.detect(seg.text)
            else:
                narrative = NarrativeContext(
                    type=NarrativeType.NARRATION,
                    confidence=1.0,
                    suggested_speed=1.0,
                    suggested_pause_before=0.0,
                    suggested_pause_after=0.3
                )

            # Continuite emotionnelle
            transition = self.emotion_continuity.process_emotion(emotion, intensity)
            prosody = self.emotion_continuity.smooth_prosody(
                prosody,
                hq_segments[-1].prosody if hq_segments else None
            )

            # Tracker de chapitre
            self._chapter_tracker.track(i, emotion, intensity)

            # Pauses
            next_seg = dialogue_segments[i + 1] if i + 1 < len(dialogue_segments) else None
            pause_before, pause_after = self._calculate_pauses(
                seg, narrative, prosody, next_seg
            )

            # Appliquer la pause de transition
            pause_before = max(pause_before, transition.pause_duration)

            # Vitesse finale
            final_speed = self._calculate_final_speed(prosody, narrative)
            final_speed *= transition.speed_adjustment

            # Creer le segment HQ
            hq_segment = HQSegment(
                text=seg.text,
                index=i,
                speaker=seg.speaker,
                speaker_type=seg.speaker_type,
                voice_id=voice_id,
                emotion=emotion,
                intensity=intensity,
                prosody=prosody,
                narrative_type=narrative.type,
                narrative_confidence=narrative.confidence,
                tts_tags=tts_tags,
                pause_before=pause_before,
                pause_after=pause_after,
                is_dialogue=(seg.speaker_type == SpeakerType.CHARACTER),
                is_internal_thought=narrative.is_internal_thought,
                chapter_index=chapter_index,
                final_speed=final_speed
            )
            hq_segments.append(hq_segment)

        # Etape 6: Chunking des segments longs
        if progress_callback:
            progress_callback(step, total_steps, "Chunking")
        step += 1

        result = []
        for seg in hq_segments:
            if len(seg.text) > self.config.max_chunk_size:
                chunks = self.chunker.chunk(seg.text)
                for j, chunk in enumerate(chunks):
                    chunked = HQSegment(
                        text=chunk.text,
                        index=len(result),
                        speaker=seg.speaker,
                        speaker_type=seg.speaker_type,
                        voice_id=seg.voice_id,
                        emotion=seg.emotion,
                        intensity=seg.intensity,
                        prosody=seg.prosody,
                        narrative_type=seg.narrative_type,
                        narrative_confidence=seg.narrative_confidence,
                        tts_tags=seg.tts_tags if j == 0 else [],
                        pause_before=seg.pause_before if j == 0 else 0.1,
                        pause_after=seg.pause_after if j == len(chunks) - 1 else 0.1,
                        is_dialogue=seg.is_dialogue,
                        is_internal_thought=seg.is_internal_thought,
                        chapter_index=seg.chapter_index,
                        final_speed=seg.final_speed
                    )
                    result.append(chunked)
            else:
                seg.index = len(result)
                result.append(seg)

        return result

    def get_characters(self) -> list[Character]:
        """Retourne les personnages detectes."""
        return self._characters

    def get_voice_assignments(self) -> dict[str, str]:
        """Retourne les assignations de voix."""
        return {"NARRATOR": self.config.narrator_voice, **self._voice_assignments}

    def get_chapter_analysis(self) -> dict:
        """Retourne l'analyse du chapitre traite."""
        return {
            "dominant_emotion": self._chapter_tracker.get_dominant_emotion().value,
            "tone": self._chapter_tracker.get_chapter_tone(),
            "max_intensity": self._chapter_tracker.max_intensity.value,
            "climax_count": len(self._chapter_tracker.climax_positions),
            "suggested_base_speed": self._chapter_tracker.get_suggested_base_speed(),
        }

    def reset(self):
        """Reinitialise le pipeline (nouveau livre)."""
        self.character_detector.reset()
        self.emotion_continuity.reset()
        self._characters = []
        self._voice_assignments = {}
        self._chapter_tracker = ChapterEmotionTracker()


def create_hq_pipeline(
    lang: str = "fr",
    narrator_voice: str = "ff_siwis",
    **kwargs
) -> HQPipeline:
    """
    Cree un pipeline haute qualite avec configuration par defaut.

    Args:
        lang: Code langue
        narrator_voice: Voix du narrateur
        **kwargs: Options supplementaires

    Returns:
        Pipeline configure
    """
    config = HQPipelineConfig(
        lang=lang,
        narrator_voice=narrator_voice,
        **kwargs
    )
    return HQPipeline(config)


if __name__ == "__main__":
    # Test du pipeline
    test_text = """
    Chapitre 1 : L'arrivee

    Le 15 mars 2024, Marie arriva a Paris. Il etait 14h30 et la temperature
    affichait 18°C.

    « Enfin ! » s'exclama-t-elle avec joie.

    Pierre l'attendait sur le quai, un bouquet de 12 roses a la main.

    « Tu m'as tellement manque », murmura-t-il tendrement.

    Soudain, un bruit terrible retentit. BOUM ! Une explosion au loin.
    Marie sursauta, terrifiee.

    « Qu'est-ce que c'etait ? » demanda-t-elle, le coeur battant.

    Pierre resta silencieux. Il se souvenait de ce jour, il y a 5 ans,
    ou tout avait change. Le souvenir etait encore vif dans sa memoire.
    """

    print("=== Test Pipeline Haute Qualite ===\n")

    pipeline = create_hq_pipeline(lang="fr", narrator_voice="ff_siwis")

    def progress(step, total, message):
        print(f"  [{step+1}/{total}] {message}")

    segments = pipeline.process_chapter(test_text, progress_callback=progress)

    print(f"\n=== Resultats ===")
    print(f"Segments: {len(segments)}")

    print(f"\n=== Personnages ===")
    for speaker, voice in pipeline.get_voice_assignments().items():
        print(f"  {speaker}: {voice}")

    print(f"\n=== Analyse du chapitre ===")
    analysis = pipeline.get_chapter_analysis()
    for key, value in analysis.items():
        print(f"  {key}: {value}")

    print(f"\n=== Premiers segments ===")
    for seg in segments[:5]:
        print(f"[{seg.speaker:10}] ({seg.emotion.value:10}/{seg.narrative_type.value:12}) "
              f"spd={seg.final_speed:.2f} | {seg.text[:40]}...")
