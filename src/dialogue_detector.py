"""
Détecteur de dialogues pour multi-voix.

Détecte les dialogues dans le texte et les sépare pour utiliser
des voix différentes (narrateur vs personnages).
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum


class SegmentType(Enum):
    """Type de segment de texte."""
    NARRATION = "narration"
    DIALOGUE = "dialogue"


@dataclass
class TextSegment:
    """Segment de texte avec son type."""
    text: str
    segment_type: SegmentType
    speaker: Optional[str] = None  # Pour futur: identification du personnage


def detect_dialogues(text: str) -> List[TextSegment]:
    """
    Détecte et sépare les dialogues du texte narratif.

    Supporte:
    - Guillemets français: « dialogue »
    - Guillemets anglais: "dialogue"
    - Tirets de dialogue: — dialogue ou - dialogue

    Args:
        text: Texte à analyser

    Returns:
        Liste de segments avec leur type (narration/dialogue)
    """
    segments = []

    # Pattern pour détecter les dialogues
    # Guillemets français « »
    # Guillemets anglais " "
    # Tirets en début de ligne (dialogue français classique)

    patterns = [
        # Guillemets français
        (r'«\s*([^»]+?)\s*»', 'guillemets_fr'),
        # Guillemets anglais doubles
        (r'"([^"]+?)"', 'guillemets_en'),
        # Guillemets typographiques
        (r'"([^"]+?)"', 'guillemets_typo'),
    ]

    # Combiner tous les patterns
    combined_pattern = r'(«[^»]+?»|"[^"]+?"|"[^"]+?")'

    # Trouver tous les dialogues et leurs positions
    matches = list(re.finditer(combined_pattern, text))

    if not matches:
        # Pas de dialogue détecté, tout est narration
        if text.strip():
            segments.append(TextSegment(
                text=text.strip(),
                segment_type=SegmentType.NARRATION
            ))
        return segments

    # Construire les segments alternés
    last_end = 0

    for match in matches:
        start, end = match.span()

        # Texte avant le dialogue (narration)
        if start > last_end:
            narration = text[last_end:start].strip()
            if narration:
                segments.append(TextSegment(
                    text=narration,
                    segment_type=SegmentType.NARRATION
                ))

        # Le dialogue lui-même
        dialogue_text = match.group(1)
        # Nettoyer les guillemets
        dialogue_clean = re.sub(r'^[«»"""\s]+|[«»"""\s]+$', '', dialogue_text).strip()

        if dialogue_clean:
            segments.append(TextSegment(
                text=dialogue_clean,
                segment_type=SegmentType.DIALOGUE
            ))

        last_end = end

    # Texte après le dernier dialogue
    if last_end < len(text):
        narration = text[last_end:].strip()
        if narration:
            segments.append(TextSegment(
                text=narration,
                segment_type=SegmentType.NARRATION
            ))

    return segments


def detect_dialogues_with_tirets(text: str) -> List[TextSegment]:
    """
    Version alternative qui gère aussi les tirets de dialogue.

    En français, les dialogues peuvent commencer par:
    - Un tiret cadratin (—)
    - Un tiret demi-cadratin (–)
    - Un tiret simple (-)
    """
    segments = []
    lines = text.split('\n')

    current_narration = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Vérifier si la ligne commence par un tiret de dialogue
        tiret_match = re.match(r'^[—–\-]\s*(.+)$', line)

        if tiret_match:
            # Sauvegarder la narration accumulée
            if current_narration:
                narration_text = ' '.join(current_narration)
                # Traiter la narration pour extraire les dialogues en guillemets
                sub_segments = detect_dialogues(narration_text)
                segments.extend(sub_segments)
                current_narration = []

            # Ajouter le dialogue
            dialogue_text = tiret_match.group(1).strip()
            if dialogue_text:
                segments.append(TextSegment(
                    text=dialogue_text,
                    segment_type=SegmentType.DIALOGUE
                ))
        else:
            # Accumuler la narration
            current_narration.append(line)

    # Traiter la narration restante
    if current_narration:
        narration_text = ' '.join(current_narration)
        sub_segments = detect_dialogues(narration_text)
        segments.extend(sub_segments)

    return segments


def merge_short_segments(segments: List[TextSegment], min_length: int = 10) -> List[TextSegment]:
    """
    Fusionne les segments très courts avec le précédent.
    Évite les changements de voix trop fréquents.
    """
    if len(segments) <= 1:
        return segments

    merged = [segments[0]]

    for seg in segments[1:]:
        if len(seg.text) < min_length and merged:
            # Fusionner avec le précédent
            prev = merged[-1]
            merged[-1] = TextSegment(
                text=prev.text + " " + seg.text,
                segment_type=prev.segment_type
            )
        else:
            merged.append(seg)

    return merged


def get_voice_for_segment(
    segment: TextSegment,
    narrator_voice: str,
    dialogue_voice: str
) -> str:
    """Retourne la voix à utiliser pour un segment."""
    if segment.segment_type == SegmentType.DIALOGUE:
        return dialogue_voice
    return narrator_voice


# Tests
if __name__ == "__main__":
    test_texts = [
        'Marie dit: "Bonjour Pierre!" Il répondit: "Salut!"',
        'Il faisait nuit. «Où es-tu?» demanda-t-elle. «Je suis là» répondit-il.',
        "— Bonjour!\n— Comment ça va?\nIl sourit.",
        'Le soleil brillait. "C\'est une belle journée" pensa-t-il en marchant.',
    ]

    print("=== Test du détecteur de dialogues ===\n")

    for i, text in enumerate(test_texts, 1):
        print(f"Test {i}: {text[:50]}...")
        segments = detect_dialogues(text)
        for seg in segments:
            print(f"  [{seg.segment_type.value:10}] {seg.text[:60]}...")
        print()
