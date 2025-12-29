"""
Pipeline de preprocessing avance pour qualite ElevenLabs.

Integre:
- Detection de personnages et multi-voix
- Analyse emotionnelle et prosodie
- Chunking intelligent
- Annotations TTS
"""
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import json
import re

from .character_detector import (
    CharacterDetector,
    VoiceAssigner,
    DialogueSegment,
    SpeakerType,
    Character
)
from .emotion_analyzer import (
    EmotionAnalyzer,
    BreathPlacer,
    EmotionAnalysis,
    Emotion,
    Intensity,
    ProsodyHints
)
from .text_processor import PronunciationCorrector, TextChunker


@dataclass
class EnrichedSegment:
    """
    Segment de texte enrichi avec toutes les metadonnees.

    Contient toutes les informations necessaires pour une
    synthese vocale expressive multi-personnages.
    """
    text: str
    index: int

    # Personnage
    speaker: str
    speaker_type: SpeakerType
    voice_id: str

    # Emotion
    emotion: Emotion
    intensity: Intensity
    prosody: ProsodyHints

    # TTS
    tts_tags: list[str] = field(default_factory=list)
    emphasis_words: list[str] = field(default_factory=list)

    # Timing
    pause_before: float = 0.0
    pause_after: float = 0.0

    # Metadata
    is_dialogue: bool = False
    chapter_index: int = 0


@dataclass
class ProcessingConfig:
    """Configuration du pipeline de preprocessing."""
    # Langue
    lang: str = "fr"

    # Voix
    narrator_voice: str = "ff_siwis"
    voice_mapping: dict[str, str] = field(default_factory=dict)
    auto_assign_voices: bool = True

    # Emotions
    enable_emotion_analysis: bool = True
    enable_prosody_hints: bool = True
    enable_breath_pauses: bool = True

    # Chunking
    max_chunk_size: int = 500
    sentence_pause: float = 0.3
    paragraph_pause: float = 0.8
    dialogue_pause: float = 0.4

    # Corrections
    enable_pronunciation_correction: bool = True
    custom_pronunciation: Optional[dict] = None

    # Personnages
    enable_character_detection: bool = True
    min_character_occurrences: int = 2  # Ignorer personnages mentionnes 1 fois

    @classmethod
    def from_file(cls, path: Path) -> "ProcessingConfig":
        """Charge la config depuis un fichier JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)

    def to_file(self, path: Path):
        """Sauvegarde la config dans un fichier JSON."""
        data = {
            "lang": self.lang,
            "narrator_voice": self.narrator_voice,
            "voice_mapping": self.voice_mapping,
            "auto_assign_voices": self.auto_assign_voices,
            "enable_emotion_analysis": self.enable_emotion_analysis,
            "enable_prosody_hints": self.enable_prosody_hints,
            "enable_breath_pauses": self.enable_breath_pauses,
            "max_chunk_size": self.max_chunk_size,
            "sentence_pause": self.sentence_pause,
            "paragraph_pause": self.paragraph_pause,
            "dialogue_pause": self.dialogue_pause,
            "enable_pronunciation_correction": self.enable_pronunciation_correction,
            "enable_character_detection": self.enable_character_detection,
            "min_character_occurrences": self.min_character_occurrences,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


class AdvancedPreprocessor:
    """
    Pipeline complet de preprocessing pour audiobooks haute qualite.

    Etapes:
    1. Nettoyage et normalisation du texte
    2. Correction de prononciation
    3. Detection des personnages et dialogues
    4. Segmentation par locuteur
    5. Analyse emotionnelle par segment
    6. Attribution des voix
    7. Calcul des pauses et prosodie
    8. Generation des segments enrichis
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Args:
            config: Configuration du pipeline
        """
        self.config = config or ProcessingConfig()

        # Initialiser les composants
        self.pronunciation = PronunciationCorrector(
            lang=self.config.lang,
            custom_dict=self.config.custom_pronunciation
        )
        self.chunker = TextChunker(max_chars=self.config.max_chunk_size)
        self.character_detector = CharacterDetector(lang=self.config.lang)
        self.emotion_analyzer = EmotionAnalyzer(lang=self.config.lang)
        self.breath_placer = BreathPlacer()
        self.voice_assigner = VoiceAssigner(
            narrator_voice=self.config.narrator_voice,
            lang=self.config.lang,
            voice_mapping=self.config.voice_mapping
        )

        # Etat
        self._characters: list[Character] = []
        self._voice_assignments: dict[str, str] = {}

    def _clean_text(self, text: str) -> str:
        """Nettoie le texte brut."""
        # Normaliser les espaces
        text = re.sub(r'\s+', ' ', text)

        # Normaliser les guillemets
        text = text.replace('«', '"').replace('»', '"')
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace("'", "'")

        # Normaliser les tirets
        text = text.replace('—', ' - ').replace('–', ' - ')

        # Supprimer les caracteres Markdown
        text = re.sub(r'[*_~`#]', '', text)

        # Normaliser la ponctuation multiple
        text = re.sub(r'\.{4,}', '...', text)
        text = re.sub(r'!{3,}', '!!', text)
        text = re.sub(r'\?{3,}', '??', text)

        return text.strip()

    def _preprocess_text(self, text: str) -> str:
        """Preprocessing initial du texte."""
        text = self._clean_text(text)

        if self.config.enable_pronunciation_correction:
            text = self.pronunciation.correct(text)

        return text

    def _compute_segment_pause(
        self,
        segment: DialogueSegment,
        next_segment: Optional[DialogueSegment]
    ) -> tuple[float, float]:
        """Calcule les pauses avant/apres un segment."""
        pause_before = 0.0
        pause_after = self.config.sentence_pause

        # Pause de dialogue
        if segment.is_dialogue:
            pause_after = self.config.dialogue_pause

        # Pause entre personnages differents
        if next_segment and next_segment.speaker != segment.speaker:
            pause_after = max(pause_after, self.config.dialogue_pause)

        # Pause de transition narrateur -> personnage
        if (segment.speaker_type == SpeakerType.NARRATOR
                and next_segment
                and next_segment.speaker_type == SpeakerType.CHARACTER):
            pause_after = max(pause_after, 0.3)

        return pause_before, pause_after

    def process(
        self,
        text: str,
        chapter_index: int = 0
    ) -> list[EnrichedSegment]:
        """
        Traite un texte complet et retourne des segments enrichis.

        Args:
            text: Texte brut a traiter
            chapter_index: Index du chapitre (pour metadata)

        Returns:
            Liste de segments prets pour la synthese
        """
        # Etape 1: Preprocessing
        text = self._preprocess_text(text)

        # Etape 2: Detection personnages et segmentation
        if self.config.enable_character_detection:
            dialogue_segments = self.character_detector.detect_dialogue_segments(text)
            self._characters = self.character_detector.get_characters()
        else:
            # Tout comme narrateur
            dialogue_segments = [
                DialogueSegment(
                    text=text,
                    speaker="NARRATOR",
                    speaker_type=SpeakerType.NARRATOR,
                    index=0
                )
            ]
            self._characters = []

        # Etape 3: Attribution des voix
        if self.config.auto_assign_voices:
            # Filtrer personnages avec assez d'occurrences
            significant_chars = [
                c for c in self._characters
                if c.occurrence_count >= self.config.min_character_occurrences
            ]
            self._voice_assignments = self.voice_assigner.assign_voices_to_characters(
                significant_chars
            )

        # Etape 4: Enrichir chaque segment
        enriched_segments = []

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
                emphasis = emotion_analysis.emphasis_words
            else:
                emotion = Emotion.NEUTRAL
                intensity = Intensity.MEDIUM
                prosody = ProsodyHints()
                tts_tags = []
                emphasis = []

            # Pauses
            next_seg = dialogue_segments[i + 1] if i + 1 < len(dialogue_segments) else None
            pause_before, pause_after = self._compute_segment_pause(seg, next_seg)

            # Respiration
            if self.config.enable_breath_pauses:
                if prosody.breath_before:
                    pause_before = max(pause_before, 0.2)

            # Creer le segment enrichi
            enriched = EnrichedSegment(
                text=seg.text,
                index=i,
                speaker=seg.speaker,
                speaker_type=seg.speaker_type,
                voice_id=voice_id,
                emotion=emotion,
                intensity=intensity,
                prosody=prosody,
                tts_tags=tts_tags,
                emphasis_words=emphasis,
                pause_before=pause_before,
                pause_after=pause_after,
                is_dialogue=(seg.speaker_type == SpeakerType.CHARACTER),
                chapter_index=chapter_index
            )
            enriched_segments.append(enriched)

        return enriched_segments

    def process_with_chunking(
        self,
        text: str,
        chapter_index: int = 0
    ) -> list[EnrichedSegment]:
        """
        Traite le texte avec chunking pour longs segments.

        Utile pour eviter les depassements de phonemes dans le TTS.
        """
        # D'abord traiter normalement
        segments = self.process(text, chapter_index)

        # Puis chunker les segments trop longs
        result = []
        for seg in segments:
            if len(seg.text) > self.config.max_chunk_size:
                # Chunker ce segment
                chunks = self.chunker.chunk(seg.text)
                for j, chunk in enumerate(chunks):
                    # Copier les proprietes du segment original
                    chunked = EnrichedSegment(
                        text=chunk.text,
                        index=len(result),
                        speaker=seg.speaker,
                        speaker_type=seg.speaker_type,
                        voice_id=seg.voice_id,
                        emotion=seg.emotion,
                        intensity=seg.intensity,
                        prosody=seg.prosody,
                        tts_tags=seg.tts_tags if j == 0 else [],
                        emphasis_words=seg.emphasis_words,
                        pause_before=seg.pause_before if j == 0 else 0.1,
                        pause_after=seg.pause_after if j == len(chunks) - 1 else 0.1,
                        is_dialogue=seg.is_dialogue,
                        chapter_index=seg.chapter_index
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
        assignments = {"NARRATOR": self.config.narrator_voice}
        assignments.update(self._voice_assignments)
        return assignments

    def reset(self):
        """Reinitialise l'etat (pour nouveau livre)."""
        self.character_detector.reset()
        self._characters = []
        self._voice_assignments = {}


def create_default_config(
    lang: str = "fr",
    narrator_voice: str = "ff_siwis"
) -> ProcessingConfig:
    """Cree une configuration par defaut optimisee."""
    return ProcessingConfig(
        lang=lang,
        narrator_voice=narrator_voice,
        auto_assign_voices=True,
        enable_emotion_analysis=True,
        enable_prosody_hints=True,
        enable_breath_pauses=True,
        max_chunk_size=500,
        sentence_pause=0.3,
        paragraph_pause=0.8,
        dialogue_pause=0.4,
        enable_pronunciation_correction=True,
        enable_character_detection=True,
        min_character_occurrences=2
    )


def process_chapter(
    text: str,
    config: Optional[ProcessingConfig] = None,
    chapter_index: int = 0
) -> tuple[list[EnrichedSegment], dict[str, str]]:
    """
    Fonction utilitaire pour traiter un chapitre.

    Args:
        text: Texte du chapitre
        config: Configuration (optionnelle)
        chapter_index: Index du chapitre

    Returns:
        Tuple (segments enrichis, assignations de voix)
    """
    preprocessor = AdvancedPreprocessor(config)
    segments = preprocessor.process_with_chunking(text, chapter_index)
    assignments = preprocessor.get_voice_assignments()
    return segments, assignments


if __name__ == "__main__":
    # Test du pipeline complet
    test_text = """
    Chapitre 1

    Marie entra dans la piece sombre. Son coeur battait la chamade.

    « Pierre ? » appela-t-elle d'une voix tremblante.

    Silence. Puis soudain, une ombre bougea dans le coin.

    « Je suis la », repondit Pierre. « N'aie pas peur. »

    Marie poussa un soupir de soulagement. « Tu m'as fait une de ces peurs ! »
    s'exclama-t-elle.

    Pierre s'approcha d'elle avec un sourire. « Desole, je ne voulais pas
    t'effrayer. J'etais en train de chercher quelque chose. »

    « Quoi donc ? » demanda Marie, curieuse.

    « Ceci », dit-il en sortant un ecrin de sa poche.
    """

    print("=== Test du Pipeline Avance ===\n")

    config = create_default_config(lang="fr", narrator_voice="ff_siwis")
    segments, assignments = process_chapter(test_text, config)

    print("=== Personnages et Voix ===")
    for speaker, voice in assignments.items():
        print(f"  {speaker}: {voice}")

    print("\n=== Segments Enrichis ===")
    for seg in segments:
        emotion_str = f"{seg.emotion.value}/{seg.intensity.value}"
        prosody_str = f"spd={seg.prosody.speed:.2f}"
        print(f"[{seg.speaker:10}] ({seg.voice_id:12}) {emotion_str:20} | {seg.text[:40]}...")
        if seg.tts_tags:
            print(f"            Tags: {seg.tts_tags}")
        if seg.pause_after > 0.3:
            print(f"            Pause: {seg.pause_after:.2f}s")

    print(f"\n=== Statistiques ===")
    print(f"  Segments: {len(segments)}")
    print(f"  Personnages: {len(assignments) - 1}")  # -1 pour narrateur
