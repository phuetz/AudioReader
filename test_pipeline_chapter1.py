#!/usr/bin/env python3
"""
Test du pipeline v2.3 avec le chapitre 1 de "Les Conquerants du Pognon".

Ce script teste:
- Detection des contours d'intonation (questions, exclamations, etc.)
- Humanisation du timing (variation des pauses)
- Detection des types de clauses (principale, subordonnee, incise)
- Micro-pauses avant mots d'emphase
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.hq_pipeline_extended import (
    ExtendedHQPipeline,
    ExtendedPipelineConfig,
    create_extended_pipeline
)
from src.intonation_contour import IntonationContour
from src.timing_humanizer import ClauseType


def load_chapter(path: str) -> str:
    """Charge le contenu d'un chapitre."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def test_pipeline_with_chapter():
    """Test complet du pipeline v2.3 avec le chapitre 1."""
    print("=" * 70)
    print("TEST PIPELINE v2.3 - Chapitre 1: Le Gamin de Saint-Denis")
    print("=" * 70)
    print()

    # Charger le chapitre
    chapter_path = "/home/patrice/claude/livre/Les_Conquerants_du_Pognon/tome-1-or-noir-v2/chapitres/chapitre-01.md"

    if not Path(chapter_path).exists():
        print(f"ERREUR: Fichier non trouve: {chapter_path}")
        return

    chapter_text = load_chapter(chapter_path)
    print(f"Chapitre charge: {len(chapter_text)} caracteres")
    print()

    # Creer le pipeline avec toutes les fonctionnalites v2.3
    print("Configuration du pipeline v2.3:")
    config = ExtendedPipelineConfig(
        lang="fr",
        narrator_voice="ff_siwis",
        # v2.3 features
        enable_advanced_breaths=True,
        enable_intonation_contours=True,
        intonation_strength=0.7,
        enable_timing_humanization=True,
        pause_variation_sigma=0.05,
        enable_emphasis_pauses=True,
        emphasis_pause_duration=0.05,
        # Autres features
        enable_audio_tags=True,
        enable_cache=False,  # Pas besoin de cache pour le test
        enable_parallel=False,
    )

    print(f"  - Respirations avancees: {config.enable_advanced_breaths}")
    print(f"  - Contours d'intonation: {config.enable_intonation_contours} (force: {config.intonation_strength})")
    print(f"  - Humanisation timing: {config.enable_timing_humanization} (sigma: {config.pause_variation_sigma})")
    print(f"  - Pauses d'emphase: {config.enable_emphasis_pauses} ({config.emphasis_pause_duration}s)")
    print()

    pipeline = ExtendedHQPipeline(config)

    # Traitement
    print("Traitement du chapitre...")
    print("-" * 70)

    def progress_callback(step, total, message):
        print(f"  [{step}/{total}] {message}")

    segments = pipeline.process_chapter(
        chapter_text,
        chapter_index=0,
        progress_callback=progress_callback
    )

    print(f"\n{len(segments)} segments generes")
    print()

    # Statistiques des contours d'intonation
    print("=" * 70)
    print("STATISTIQUES DES CONTOURS D'INTONATION")
    print("=" * 70)

    contour_counts = {}
    for seg in segments:
        contour = seg.intonation_contour
        name = contour.value if contour else "none"
        contour_counts[name] = contour_counts.get(name, 0) + 1

    for name, count in sorted(contour_counts.items(), key=lambda x: -x[1]):
        pct = count / len(segments) * 100
        bar = "#" * int(pct / 2)
        print(f"  {name:15} {count:4} ({pct:5.1f}%) {bar}")
    print()

    # Statistiques des types de clauses
    print("=" * 70)
    print("STATISTIQUES DES TYPES DE CLAUSES")
    print("=" * 70)

    clause_counts = {}
    for seg in segments:
        clause = seg.clause_type
        name = clause.value if clause else "none"
        clause_counts[name] = clause_counts.get(name, 0) + 1

    for name, count in sorted(clause_counts.items(), key=lambda x: -x[1]):
        pct = count / len(segments) * 100
        bar = "#" * int(pct / 2)
        print(f"  {name:15} {count:4} ({pct:5.1f}%) {bar}")
    print()

    # Statistiques des pauses humanisees
    print("=" * 70)
    print("STATISTIQUES DES PAUSES HUMANISEES")
    print("=" * 70)

    original_pauses = []
    humanized_pauses = []

    for seg in segments:
        if seg.pause_after > 0:
            original_pauses.append(seg.pause_after)
            humanized_pauses.append(seg.humanized_pause_after)

    if original_pauses:
        avg_original = sum(original_pauses) / len(original_pauses)
        avg_humanized = sum(humanized_pauses) / len(humanized_pauses)

        variations = [(h - o) / o * 100 for o, h in zip(original_pauses, humanized_pauses) if o > 0]
        avg_var = sum(variations) / len(variations) if variations else 0
        min_var = min(variations) if variations else 0
        max_var = max(variations) if variations else 0

        print(f"  Pauses analysees: {len(original_pauses)}")
        print(f"  Pause moyenne originale: {avg_original:.3f}s")
        print(f"  Pause moyenne humanisee: {avg_humanized:.3f}s")
        print(f"  Variation moyenne: {avg_var:+.1f}%")
        print(f"  Variation min/max: {min_var:+.1f}% / {max_var:+.1f}%")
    print()

    # Segments avec pauses d'emphase
    print("=" * 70)
    print("SEGMENTS AVEC PAUSES D'EMPHASE")
    print("=" * 70)

    emphasis_segments = [seg for seg in segments if seg.has_emphasis_pause]
    print(f"  {len(emphasis_segments)} segments avec micro-pauses d'emphase\n")

    for seg in emphasis_segments[:10]:  # Afficher les 10 premiers
        text_preview = seg.text[:60].replace('\n', ' ')
        if len(seg.text) > 60:
            text_preview += "..."
        print(f"  - {text_preview}")

    if len(emphasis_segments) > 10:
        print(f"  ... et {len(emphasis_segments) - 10} autres")
    print()

    # Exemples de segments par type de contour
    print("=" * 70)
    print("EXEMPLES DE SEGMENTS PAR CONTOUR D'INTONATION")
    print("=" * 70)

    for contour_type in IntonationContour:
        examples = [seg for seg in segments if seg.intonation_contour == contour_type][:3]
        if examples:
            print(f"\n  [{contour_type.value.upper()}]")
            for seg in examples:
                text_preview = seg.text[:70].replace('\n', ' ').strip()
                if len(seg.text) > 70:
                    text_preview += "..."
                print(f"    > {text_preview}")
    print()

    # Dialogues detectes
    print("=" * 70)
    print("DIALOGUES DETECTES (extraits)")
    print("=" * 70)

    dialogues = [seg for seg in segments if seg.is_dialogue]
    print(f"  {len(dialogues)} segments de dialogue\n")

    for seg in dialogues[:10]:
        speaker = seg.speaker if seg.speaker != "NARRATOR" else "?"
        text_preview = seg.text[:50].replace('\n', ' ')
        contour = seg.intonation_contour.value if seg.intonation_contour else "-"
        print(f"  [{speaker:12}] ({contour:12}) \"{text_preview}...\"")

    if len(dialogues) > 10:
        print(f"  ... et {len(dialogues) - 10} autres dialogues")
    print()

    # Stats finales du pipeline
    print("=" * 70)
    print("STATISTIQUES DU PIPELINE")
    print("=" * 70)

    stats = pipeline.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    # Personnages detectes
    print("=" * 70)
    print("PERSONNAGES DETECTES")
    print("=" * 70)

    characters = pipeline.get_characters()
    for char in characters[:10]:
        print(f"  - {char}")
    if len(characters) > 10:
        print(f"  ... et {len(characters) - 10} autres")
    print()

    print("=" * 70)
    print("TEST TERMINE")
    print("=" * 70)

    return segments


if __name__ == "__main__":
    segments = test_pipeline_with_chapter()
