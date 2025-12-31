#!/usr/bin/env python3
"""
Exemple d'utilisation du systeme multi-voix avance.

Ce script montre comment:
1. Detecter automatiquement les personnages
2. Assigner des voix differentes a chaque personnage
3. Analyser les emotions pour une lecture expressive
4. Generer un audiobook avec changement de voix

Usage:
    python example_multivoix.py input.md -o output/
    python example_multivoix.py input.md --config config.json
"""
import argparse
from pathlib import Path
import sys

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent))

from src.advanced_preprocessor import (
    AdvancedPreprocessor,
    ProcessingConfig,
    create_default_config
)
from src.tts_kokoro_engine import KokoroEngine
from src.character_detector import SpeakerType


def print_analysis(segments, assignments):
    """Affiche un resume de l'analyse."""
    print("\n" + "=" * 60)
    print("ANALYSE DU TEXTE")
    print("=" * 60)

    # Personnages et voix
    print("\n--- Personnages detectes ---")
    for speaker, voice in assignments.items():
        print(f"  {speaker:20} -> {voice}")

    # Stats par personnage
    print("\n--- Statistiques ---")
    speaker_counts = {}
    for seg in segments:
        speaker_counts[seg.speaker] = speaker_counts.get(seg.speaker, 0) + 1

    for speaker, count in sorted(speaker_counts.items(), key=lambda x: -x[1]):
        print(f"  {speaker:20}: {count} segments")

    # Emotions detectees
    print("\n--- Emotions detectees ---")
    emotion_counts = {}
    for seg in segments:
        emotion_counts[seg.emotion.value] = emotion_counts.get(seg.emotion.value, 0) + 1

    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {emotion:15}: {count}")

    print("\n" + "=" * 60)


def convert_with_multivoix(
    input_path: Path,
    output_dir: Path,
    config: ProcessingConfig,
    verbose: bool = True
):
    """
    Convertit un fichier texte en audio avec multi-voix.

    Args:
        input_path: Chemin du fichier d'entree (.md ou .txt)
        output_dir: Repertoire de sortie
        config: Configuration du preprocessing
        verbose: Afficher les details
    """
    # Lire le texte
    print(f"\nLecture de {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Preprocessing avance
    print("Analyse du texte (personnages, emotions)...")
    preprocessor = AdvancedPreprocessor(config)
    segments = preprocessor.process_with_chunking(text)
    assignments = preprocessor.get_voice_assignments()

    if verbose:
        print_analysis(segments, assignments)

    # Afficher quelques segments exemple
    if verbose:
        print("\n--- Premiers segments ---")
        for seg in segments[:5]:
            speaker = f"[{seg.speaker}]"
            emotion = f"({seg.emotion.value})"
            print(f"{speaker:15} {emotion:12} {seg.text[:50]}...")
        if len(segments) > 5:
            print(f"  ... et {len(segments) - 5} autres segments")

    # Initialiser le moteur TTS
    print("\nInitialisation du moteur Kokoro...")
    engine = KokoroEngine(
        voice=config.narrator_voice,
        speed=1.0,
        sentence_pause=config.sentence_pause,
        paragraph_pause=config.paragraph_pause
    )

    if not engine.is_available():
        print("ERREUR: Kokoro n'est pas disponible.")
        print("Installez-le avec: pip install kokoro-onnx soundfile")
        print("Et telechargez les modeles (voir README)")
        return False

    # Generer l'audio
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{input_path.stem}_multivoix.wav"

    print(f"\nGeneration de l'audio ({len(segments)} segments)...")

    def progress(index, total, segment):
        speaker = segment.speaker[:10]
        pct = (index + 1) / total * 100
        print(f"\r  [{pct:5.1f}%] {speaker:10} - {segment.text[:30]}...", end="", flush=True)

    success = engine.synthesize_enriched_segments(
        segments,
        output_file,
        progress_callback=progress
    )

    print()  # Nouvelle ligne apres la barre de progression

    if success:
        print(f"\nAudio genere: {output_file}")
        return True
    else:
        print("\nERREUR lors de la generation audio")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convertit un texte en audiobook multi-voix"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Fichier d'entree (.md, .txt)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("output"),
        help="Repertoire de sortie (defaut: output/)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Fichier de configuration JSON"
    )
    parser.add_argument(
        "--narrator-voice",
        default="ff_siwis",
        help="Voix du narrateur (defaut: ff_siwis)"
    )
    parser.add_argument(
        "--lang",
        choices=["fr", "en"],
        default="fr",
        help="Langue du texte (defaut: fr)"
    )
    parser.add_argument(
        "--no-emotions",
        action="store_true",
        help="Desactiver l'analyse emotionnelle"
    )
    parser.add_argument(
        "--no-characters",
        action="store_true",
        help="Desactiver la detection de personnages"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Mode silencieux"
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="Afficher les voix disponibles"
    )

    args = parser.parse_args()

    # Afficher les voix
    if args.list_voices:
        KokoroEngine.list_voices()
        return

    # Verifier le fichier d'entree
    if not args.input.exists():
        print(f"ERREUR: Fichier non trouve: {args.input}")
        sys.exit(1)

    # Charger ou creer la configuration
    if args.config and args.config.exists():
        config = ProcessingConfig.from_file(args.config)
        print(f"Configuration chargee depuis {args.config}")
    else:
        config = create_default_config(
            lang=args.lang,
            narrator_voice=args.narrator_voice
        )

    # Appliquer les options CLI
    if args.no_emotions:
        config.enable_emotion_analysis = False
    if args.no_characters:
        config.enable_character_detection = False

    # Lancer la conversion
    success = convert_with_multivoix(
        args.input,
        args.output,
        config,
        verbose=not args.quiet
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
