"""
Audio Tags avances pour TTS expressif.

Supporte les tags style ElevenLabs v3:
- Emotions: [excited], [sad], [angry], [whispers], [dramatic]
- Actions: [sigh], [laugh], [gasp], [clears throat]
- Effets sonores: [pause], [long pause]
- Styles: [sarcastic], [cheerful], [serious], [mysterious]
"""
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum


class TagCategory(Enum):
    """Categories de tags audio."""
    EMOTION = "emotion"
    ACTION = "action"
    PAUSE = "pause"
    STYLE = "style"
    SOUND = "sound"


@dataclass
class AudioTag:
    """Un tag audio avec ses proprietes."""
    name: str
    category: TagCategory
    tts_tag: Optional[str]  # Tag natif TTS si supporte
    speed_modifier: float = 1.0
    volume_modifier: float = 1.0
    pitch_modifier: float = 0.0
    pause_before: float = 0.0
    pause_after: float = 0.0
    description: str = ""


class AudioTagProcessor:
    """
    Processeur de tags audio avances.

    Transforme les tags en:
    - Tags TTS natifs quand supportes
    - Modifications de prosodie
    - Pauses contextuelles
    """

    # Definition des tags supportes
    TAGS = {
        # === EMOTIONS ===
        "excited": AudioTag(
            name="excited",
            category=TagCategory.EMOTION,
            tts_tag=None,
            speed_modifier=1.15,
            volume_modifier=1.1,
            pitch_modifier=0.3,
            description="Voix excitee et energique"
        ),
        "sad": AudioTag(
            name="sad",
            category=TagCategory.EMOTION,
            tts_tag=None,
            speed_modifier=0.85,
            volume_modifier=0.9,
            pitch_modifier=-0.2,
            pause_after=0.3,
            description="Voix triste et melancolique"
        ),
        "angry": AudioTag(
            name="angry",
            category=TagCategory.EMOTION,
            tts_tag=None,
            speed_modifier=1.1,
            volume_modifier=1.2,
            pitch_modifier=0.2,
            description="Voix en colere"
        ),
        "whispers": AudioTag(
            name="whispers",
            category=TagCategory.EMOTION,
            tts_tag=None,
            speed_modifier=0.9,
            volume_modifier=0.6,
            pitch_modifier=-0.1,
            description="Chuchotement"
        ),
        "fearful": AudioTag(
            name="fearful",
            category=TagCategory.EMOTION,
            tts_tag=None,
            speed_modifier=1.2,
            volume_modifier=0.95,
            pitch_modifier=0.4,
            pause_before=0.1,
            description="Voix apeuree"
        ),
        "surprised": AudioTag(
            name="surprised",
            category=TagCategory.EMOTION,
            tts_tag=None,
            speed_modifier=1.1,
            pitch_modifier=0.5,
            pause_before=0.2,
            description="Voix surprise"
        ),
        "tender": AudioTag(
            name="tender",
            category=TagCategory.EMOTION,
            tts_tag=None,
            speed_modifier=0.9,
            volume_modifier=0.85,
            pitch_modifier=-0.1,
            description="Voix tendre et douce"
        ),
        "dramatic": AudioTag(
            name="dramatic",
            category=TagCategory.EMOTION,
            tts_tag=None,
            speed_modifier=0.85,
            pause_before=0.4,
            pause_after=0.4,
            description="Ton dramatique"
        ),

        # === ACTIONS VOCALES ===
        "sigh": AudioTag(
            name="sigh",
            category=TagCategory.ACTION,
            tts_tag="[sigh]",
            pause_after=0.3,
            description="Soupir"
        ),
        "laugh": AudioTag(
            name="laugh",
            category=TagCategory.ACTION,
            tts_tag="[laugh]",
            pause_after=0.2,
            description="Rire"
        ),
        "chuckle": AudioTag(
            name="chuckle",
            category=TagCategory.ACTION,
            tts_tag="[chuckle]",
            pause_after=0.15,
            description="Petit rire"
        ),
        "gasp": AudioTag(
            name="gasp",
            category=TagCategory.ACTION,
            tts_tag="[gasp]",
            pause_before=0.1,
            pause_after=0.2,
            description="Halètement de surprise"
        ),
        "cough": AudioTag(
            name="cough",
            category=TagCategory.ACTION,
            tts_tag="[cough]",
            pause_after=0.2,
            description="Toux"
        ),
        "clears throat": AudioTag(
            name="clears throat",
            category=TagCategory.ACTION,
            tts_tag=None,
            pause_before=0.2,
            pause_after=0.3,
            description="Se racler la gorge"
        ),
        "sniff": AudioTag(
            name="sniff",
            category=TagCategory.ACTION,
            tts_tag=None,
            pause_after=0.1,
            description="Reniflement"
        ),
        "yawn": AudioTag(
            name="yawn",
            category=TagCategory.ACTION,
            tts_tag=None,
            speed_modifier=0.8,
            pause_after=0.3,
            description="Baillement"
        ),

        # === PAUSES ===
        "pause": AudioTag(
            name="pause",
            category=TagCategory.PAUSE,
            tts_tag=None,
            pause_after=0.5,
            description="Pause courte"
        ),
        "long pause": AudioTag(
            name="long pause",
            category=TagCategory.PAUSE,
            tts_tag=None,
            pause_after=1.0,
            description="Pause longue"
        ),
        "beat": AudioTag(
            name="beat",
            category=TagCategory.PAUSE,
            tts_tag=None,
            pause_after=0.3,
            description="Pause dramatique courte"
        ),
        "silence": AudioTag(
            name="silence",
            category=TagCategory.PAUSE,
            tts_tag=None,
            pause_after=1.5,
            description="Silence prolonge"
        ),

        # === STYLES ===
        "sarcastic": AudioTag(
            name="sarcastic",
            category=TagCategory.STYLE,
            tts_tag=None,
            speed_modifier=0.95,
            pitch_modifier=0.1,
            description="Ton sarcastique"
        ),
        "cheerful": AudioTag(
            name="cheerful",
            category=TagCategory.STYLE,
            tts_tag=None,
            speed_modifier=1.05,
            pitch_modifier=0.2,
            volume_modifier=1.05,
            description="Ton joyeux"
        ),
        "serious": AudioTag(
            name="serious",
            category=TagCategory.STYLE,
            tts_tag=None,
            speed_modifier=0.92,
            pitch_modifier=-0.1,
            description="Ton serieux"
        ),
        "mysterious": AudioTag(
            name="mysterious",
            category=TagCategory.STYLE,
            tts_tag=None,
            speed_modifier=0.88,
            volume_modifier=0.9,
            pause_before=0.3,
            description="Ton mysterieux"
        ),
        "narrator": AudioTag(
            name="narrator",
            category=TagCategory.STYLE,
            tts_tag=None,
            speed_modifier=0.95,
            description="Voix de narrateur"
        ),
        "announcer": AudioTag(
            name="announcer",
            category=TagCategory.STYLE,
            tts_tag=None,
            speed_modifier=1.0,
            volume_modifier=1.1,
            pitch_modifier=0.1,
            description="Voix d'annonceur"
        ),

        # === EFFETS SONORES (placeholder pour future integration) ===
        "footsteps": AudioTag(
            name="footsteps",
            category=TagCategory.SOUND,
            tts_tag=None,
            pause_after=0.5,
            description="Bruit de pas"
        ),
        "door": AudioTag(
            name="door",
            category=TagCategory.SOUND,
            tts_tag=None,
            pause_after=0.3,
            description="Bruit de porte"
        ),
        "thunder": AudioTag(
            name="thunder",
            category=TagCategory.SOUND,
            tts_tag=None,
            pause_before=0.2,
            pause_after=0.5,
            description="Tonnerre"
        ),
    }

    # Aliases pour tags courants
    ALIASES = {
        "whisper": "whispers",
        "laughs": "laugh",
        "laughing": "laugh",
        "sighs": "sigh",
        "sighing": "sigh",
        "gasps": "gasp",
        "gasping": "gasp",
        "coughs": "cough",
        "coughing": "cough",
        "chuckles": "chuckle",
        "chuckling": "chuckle",
        "yawns": "yawn",
        "yawning": "yawn",
        "sniffs": "sniff",
        "sniffing": "sniff",
        "happy": "cheerful",
        "joyful": "cheerful",
        "mad": "angry",
        "furious": "angry",
        "scared": "fearful",
        "afraid": "fearful",
        "terrified": "fearful",
        "soft": "tender",
        "gentle": "tender",
        "quiet": "whispers",
        "loudly": "announcer",
        "dramatically": "dramatic",
        "sarcastically": "sarcastic",
    }

    def __init__(self):
        # Pattern pour detecter les tags [xxx]
        self.tag_pattern = re.compile(r'\[([^\]]+)\]')

    def _resolve_alias(self, tag_name: str) -> str:
        """Resout les aliases vers le tag principal."""
        tag_lower = tag_name.lower().strip()
        return self.ALIASES.get(tag_lower, tag_lower)

    def get_tag(self, tag_name: str) -> Optional[AudioTag]:
        """Recupere un tag par son nom."""
        resolved = self._resolve_alias(tag_name)
        return self.TAGS.get(resolved)

    def extract_tags(self, text: str) -> List[Tuple[str, AudioTag, int, int]]:
        """
        Extrait tous les tags d'un texte.

        Returns:
            Liste de tuples (tag_text, AudioTag, start_pos, end_pos)
        """
        results = []

        for match in self.tag_pattern.finditer(text):
            tag_text = match.group(1)
            tag = self.get_tag(tag_text)

            if tag:
                results.append((
                    tag_text,
                    tag,
                    match.start(),
                    match.end()
                ))

        return results

    def process_text(self, text: str) -> Tuple[str, List[AudioTag]]:
        """
        Traite un texte et extrait les tags.

        Returns:
            Tuple (texte sans tags, liste des tags trouves)
        """
        tags_found = []

        # Extraire les tags
        for tag_text, tag, start, end in self.extract_tags(text):
            tags_found.append(tag)

        # Retirer les tags du texte (sauf les tags TTS natifs)
        processed_text = text
        for tag_text, tag, start, end in reversed(self.extract_tags(text)):
            if tag.tts_tag:
                # Remplacer par le tag TTS natif
                processed_text = (
                    processed_text[:start] +
                    tag.tts_tag +
                    processed_text[end:]
                )
            else:
                # Retirer le tag
                processed_text = (
                    processed_text[:start] +
                    processed_text[end:]
                )

        # Nettoyer les espaces multiples
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()

        return processed_text, tags_found

    def calculate_prosody_modifiers(
        self,
        tags: List[AudioTag]
    ) -> dict:
        """
        Calcule les modificateurs de prosodie combines.

        Combine les effets de tous les tags trouves.
        """
        speed = 1.0
        volume = 1.0
        pitch = 0.0
        pause_before = 0.0
        pause_after = 0.0

        for tag in tags:
            speed *= tag.speed_modifier
            volume *= tag.volume_modifier
            pitch += tag.pitch_modifier
            pause_before = max(pause_before, tag.pause_before)
            pause_after = max(pause_after, tag.pause_after)

        # Limiter les valeurs
        speed = max(0.5, min(1.5, speed))
        volume = max(0.3, min(1.5, volume))
        pitch = max(-1.0, min(1.0, pitch))

        return {
            "speed": speed,
            "volume": volume,
            "pitch": pitch,
            "pause_before": pause_before,
            "pause_after": pause_after,
        }

    def get_tts_tags(self, tags: List[AudioTag]) -> List[str]:
        """Retourne les tags TTS natifs a inserer."""
        return [tag.tts_tag for tag in tags if tag.tts_tag]

    @classmethod
    def list_available_tags(cls) -> None:
        """Affiche tous les tags disponibles."""
        print("\n=== Tags Audio Disponibles ===\n")

        current_category = None
        for name, tag in sorted(cls.TAGS.items(), key=lambda x: (x[1].category.value, x[0])):
            if tag.category != current_category:
                current_category = tag.category
                print(f"\n[{current_category.value.upper()}]")

            modifiers = []
            if tag.speed_modifier != 1.0:
                modifiers.append(f"speed:{tag.speed_modifier:.2f}")
            if tag.volume_modifier != 1.0:
                modifiers.append(f"vol:{tag.volume_modifier:.2f}")
            if tag.pitch_modifier != 0.0:
                modifiers.append(f"pitch:{tag.pitch_modifier:+.2f}")
            if tag.pause_after > 0:
                modifiers.append(f"pause:{tag.pause_after:.1f}s")

            mod_str = ", ".join(modifiers) if modifiers else "-"
            print(f"  [{name:15}] {tag.description:30} ({mod_str})")


@dataclass
class ProcessedSegment:
    """Segment traite avec tags audio."""
    text: str
    original_text: str
    tags: List[AudioTag]
    prosody: dict
    tts_tags: List[str]


def process_text_with_audio_tags(text: str) -> ProcessedSegment:
    """
    Fonction utilitaire pour traiter un texte avec tags audio.

    Args:
        text: Texte avec tags [xxx]

    Returns:
        ProcessedSegment avec toutes les informations
    """
    processor = AudioTagProcessor()
    processed_text, tags = processor.process_text(text)
    prosody = processor.calculate_prosody_modifiers(tags)
    tts_tags = processor.get_tts_tags(tags)

    return ProcessedSegment(
        text=processed_text,
        original_text=text,
        tags=tags,
        prosody=prosody,
        tts_tags=tts_tags
    )


if __name__ == "__main__":
    # Test
    AudioTagProcessor.list_available_tags()

    print("\n\n=== Test de traitement ===\n")

    test_texts = [
        "[whispers] Je dois te dire quelque chose...",
        "[excited] C'est incroyable ! [laugh]",
        "[dramatic] [pause] Et puis... tout a change.",
        "[sarcastic] Oh, quelle surprise...",
        "Il murmura [tender] je t'aime [long pause] pour toujours.",
    ]

    for text in test_texts:
        result = process_text_with_audio_tags(text)
        print(f"Original: {text}")
        print(f"Traite:   {result.text}")
        print(f"Tags:     {[t.name for t in result.tags]}")
        print(f"Prosodie: speed={result.prosody['speed']:.2f}, "
              f"pitch={result.prosody['pitch']:+.2f}")
        print()
