#!/usr/bin/env python3
"""
AudioReader - Convertit un livre Markdown en fichiers audio.

Usage:
    python audio_reader.py livre.md
    python audio_reader.py livre.md --language en --engine kokoro
    python audio_reader.py livre.md --output ./mon_audiobook
    python audio_reader.py --list-voices

Moteurs TTS:
    - MMS (Meta): Français natif de haute qualité (défaut pour fr)
    - Kokoro: Voix anglaises expressives (défaut pour en)
    - Edge: Microsoft Edge TTS (online, fallback)
"""
import argparse
import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from markdown_parser import parse_book, Chapter
from tts_engine import create_tts_engine, EngineType


# Moteurs et voix disponibles
ENGINES = {
    "auto": "Sélection automatique (MMS pour fr, Kokoro pour en)",
    "mms": "MMS-TTS (Meta) - Qualité native multilingue",
    "kokoro": "Kokoro - Voix expressives (anglais)",
    "edge": "Edge-TTS (Microsoft) - Online",
}

# Voix Kokoro disponibles
KOKORO_VOICES = {
    "ff_siwis": "Siwis - Femme française",
    "af_heart": "Heart - Femme américaine",
    "af_sarah": "Sarah - Femme américaine",
    "am_adam": "Adam - Homme américain",
    "bf_emma": "Emma - Femme britannique",
}

# Voix Edge-TTS (fallback)
EDGE_VOICES = {
    "fr-FR-DeniseNeural": "Denise - Femme (France)",
    "fr-FR-HenriNeural": "Henri - Homme (France)",
    "en-US-JennyNeural": "Jenny - Female (US)",
    "en-GB-SoniaNeural": "Sonia - Female (UK)",
}


def print_voices():
    """Affiche les moteurs et voix disponibles."""
    print("\n=== Moteurs TTS ===")
    print("-" * 50)
    for engine_id, description in ENGINES.items():
        print(f"  {engine_id:10} - {description}")

    print("\n=== Voix Kokoro ===")
    print("-" * 50)
    for voice_id, description in KOKORO_VOICES.items():
        print(f"  {voice_id:15} - {description}")

    print("\n=== Voix Edge-TTS (fallback) ===")
    print("-" * 50)
    for voice_id, description in EDGE_VOICES.items():
        print(f"  {voice_id:25} - {description}")
    print()


def print_progress(current: int, total: int, chapter_title: str):
    """Affiche la progression."""
    percent = (current / total) * 100
    bar_length = 30
    filled = int(bar_length * current / total)
    bar = "=" * filled + "-" * (bar_length - filled)
    print(f"\r[{bar}] {percent:5.1f}% - {chapter_title[:40]}", end="", flush=True)


def convert_book(
    input_file: Path,
    output_dir: Path,
    language: str,
    engine_type: str,
    voice: str,
    speed: float,
    header_level: int
):
    """Convertit un livre Markdown en fichiers audio."""

    print(f"\nLecture du fichier: {input_file}")

    # Parser le livre
    try:
        chapters = parse_book(input_file, header_level=header_level)
    except FileNotFoundError:
        print(f"Erreur: Fichier non trouvé - {input_file}")
        return False

    if not chapters:
        print("Erreur: Aucun chapitre trouvé dans le fichier.")
        return False

    print(f"Chapitres trouvés: {len(chapters)}")
    for ch in chapters:
        print(f"  {ch.number}. {ch.title}")

    # Créer le dossier de sortie
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDossier de sortie: {output_dir}")

    # Initialiser le moteur TTS unifié
    print(f"\nConfiguration TTS:")
    print(f"  Langue: {language}")
    print(f"  Moteur: {engine_type}")
    if voice:
        print(f"  Voix: {voice}")
    print(f"  Vitesse: {speed}x")

    engine = create_tts_engine(
        language=language,
        engine_type=engine_type,
        voice=voice,
        speed=speed
    )

    print(f"\nMoteur sélectionné: {engine.get_info().get('engine_type', 'unknown')}")

    if not engine.is_available():
        print("Erreur: Moteur TTS non disponible")
        return False

    print("\nConversion en cours...")
    print("-" * 50)

    success_count = 0
    for i, chapter in enumerate(chapters, 1):
        print_progress(i, len(chapters), chapter.title)

        # Nom du fichier de sortie
        filename = chapter.get_filename()
        output_path = output_dir / f"{filename}.wav"

        # Convertir en audio
        text = chapter.get_full_text()
        success = engine.synthesize(text, output_path)

        if success:
            success_count += 1

    print()  # Nouvelle ligne après la barre de progression
    print("-" * 50)
    print(f"\nTerminé! {success_count}/{len(chapters)} chapitres convertis.")
    print(f"Fichiers audio dans: {output_dir}")

    return success_count == len(chapters)


def main():
    parser = argparse.ArgumentParser(
        description="Convertit un livre Markdown en fichiers audio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python audio_reader.py mon_livre.md
  python audio_reader.py mon_livre.md --language en --engine kokoro
  python audio_reader.py mon_livre.md --output ./audiobook --speed 1.2
  python audio_reader.py --list-voices

Moteurs TTS (tous gratuits):
  - MMS: Meta Multilingual Speech - Français natif haute qualité
  - Kokoro: Voix expressives pour l'anglais
  - Edge: Microsoft Edge TTS (fallback, online)
        """
    )

    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        help="Fichier Markdown à convertir"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Dossier de sortie (défaut: ./output/<nom_livre>)"
    )

    parser.add_argument(
        "-l", "--language",
        default="fr",
        help="Langue (fr, en, de, es, etc.) - défaut: fr"
    )

    parser.add_argument(
        "-e", "--engine",
        default="auto",
        choices=["auto", "mms", "kokoro", "edge"],
        help="Moteur TTS (auto=sélection selon langue)"
    )

    parser.add_argument(
        "-v", "--voice",
        default="ff_siwis",
        help="Voix Kokoro (défaut: ff_siwis)"
    )

    parser.add_argument(
        "-s", "--speed",
        type=float,
        default=1.0,
        help="Vitesse de lecture (défaut: 1.0)"
    )

    parser.add_argument(
        "--header-level",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Niveau des headers pour les chapitres (1=#, 2=##, 3=###)"
    )

    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="Afficher les moteurs et voix disponibles"
    )

    args = parser.parse_args()

    # Afficher les voix
    if args.list_voices:
        print_voices()
        return 0

    # Vérifier le fichier d'entrée
    if not args.input_file:
        parser.print_help()
        return 1

    if not args.input_file.exists():
        print(f"Erreur: Fichier non trouvé - {args.input_file}")
        return 1

    # Dossier de sortie
    if args.output:
        output_dir = args.output
    else:
        output_dir = Path("output") / args.input_file.stem

    # Lancer la conversion
    success = convert_book(
        input_file=args.input_file,
        output_dir=output_dir,
        language=args.language,
        engine_type=args.engine,
        voice=args.voice,
        speed=args.speed,
        header_level=args.header_level
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
