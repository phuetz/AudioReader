#!/usr/bin/env python3
"""
AudioReader HQ - Convertisseur haute qualité pour publication professionnelle.

Moteurs supportés (par ordre de qualité):
  - Chatterbox: Bat ElevenLabs (63.75% préférence), contrôle émotionnel
  - Orpheus: Qualité niveau humain, 3B paramètres
  - F5-TTS: Meilleur voice cloning, nécessite audio référence

ATTENTION: Audible/ACX n'accepte PAS les voix AI!
Plateformes compatibles: Google Play Books, Findaway/Spotify, Kobo

Usage:
    python audio_reader_hq.py livre.md
    python audio_reader_hq.py livre.md --engine chatterbox
    python audio_reader_hq.py livre.md --engine f5 --reference-audio ma_voix.wav
    python audio_reader_hq.py livre.md --emotion happy --emotion-strength 0.7
    python audio_reader_hq.py --check-engines
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from markdown_parser import parse_book


def check_engines():
    """Vérifie les moteurs disponibles et affiche les instructions."""
    print("\n" + "=" * 70)
    print("   VÉRIFICATION DES MOTEURS HAUTE QUALITÉ")
    print("=" * 70 + "\n")

    engines_status = []

    # Chatterbox
    try:
        from chatterbox.tts import ChatterboxTTS
        engines_status.append(("Chatterbox", True, "Prêt à utiliser"))
    except ImportError:
        engines_status.append((
            "Chatterbox",
            False,
            "pip install chatterbox-tts torch torchaudio"
        ))

    # F5-TTS
    try:
        from f5_tts.api import F5TTS
        engines_status.append(("F5-TTS", True, "Prêt à utiliser"))
    except ImportError:
        engines_status.append((
            "F5-TTS",
            False,
            "git clone https://github.com/SWivid/F5-TTS && cd F5-TTS && pip install -e ."
        ))

    # Orpheus
    try:
        import orpheus
        engines_status.append(("Orpheus", True, "Prêt à utiliser"))
    except ImportError:
        engines_status.append((
            "Orpheus",
            False,
            "pip install orpheus-speech"
        ))

    # Edge-TTS (fallback)
    try:
        import edge_tts
        engines_status.append(("Edge-TTS (fallback)", True, "Prêt"))
    except ImportError:
        engines_status.append(("Edge-TTS", False, "pip install edge-tts"))

    # Affichage
    for name, available, info in engines_status:
        icon = "✓" if available else "✗"
        color_start = "\033[92m" if available else "\033[91m"
        color_end = "\033[0m"
        print(f"  {color_start}{icon}{color_end} {name:20} - {info}")

    print("\n" + "-" * 70)
    print("RECOMMANDATIONS:")
    print("-" * 70)
    print("""
  Pour publication professionnelle, installer Chatterbox:

    # Avec GPU NVIDIA (recommandé)
    pip install chatterbox-tts torch torchaudio --index-url https://download.pytorch.org/whl/cu118

    # CPU uniquement (plus lent)
    pip install chatterbox-tts torch torchaudio

  Requires: ~8GB VRAM (GPU) ou 16GB RAM (CPU)
""")

    print("-" * 70)
    print("ATTENTION - DISTRIBUTION:")
    print("-" * 70)
    print("""
  ✗ Audible/ACX: N'accepte PAS les voix AI tierces
  ✓ Google Play Books: Accepte (auto-narration intégrée)
  ✓ Findaway/Spotify: Accepte ElevenLabs (depuis 02/2025)
  ✓ Kobo: Accepte
  ✓ Vente directe: Aucune restriction
""")

    return any(s[1] for s in engines_status[:3])  # Au moins un HQ dispo


def print_progress(current: int, total: int, chapter_title: str, start_time: datetime):
    """Affiche la progression avec estimation temps restant."""
    percent = (current / total) * 100
    bar_length = 25
    filled = int(bar_length * current / total)
    bar = "█" * filled + "░" * (bar_length - filled)

    elapsed = (datetime.now() - start_time).total_seconds()
    if current > 0:
        eta = (elapsed / current) * (total - current)
        eta_str = f"ETA: {int(eta//60)}m{int(eta%60):02d}s"
    else:
        eta_str = "Calcul..."

    print(f"\r  [{bar}] {percent:5.1f}% | {eta_str} | {chapter_title[:30]:<30}", end="", flush=True)


def convert_book_hq(
    input_file: Path,
    output_dir: Path,
    engine: str,
    reference_audio: str,
    emotion: str,
    emotion_strength: float,
    speed: float,
    header_level: int,
    output_format: str
):
    """Convertit un livre avec moteur haute qualité."""

    print(f"\n{'=' * 70}")
    print(f"   AUDIO READER HQ - Conversion Haute Qualité")
    print(f"{'=' * 70}\n")

    # Parser le livre
    print(f"Fichier source: {input_file}")
    try:
        chapters = parse_book(input_file, header_level=header_level)
    except FileNotFoundError:
        print(f"ERREUR: Fichier non trouvé - {input_file}")
        return False

    if not chapters:
        print("ERREUR: Aucun chapitre trouvé.")
        return False

    print(f"Chapitres détectés: {len(chapters)}")
    total_chars = sum(len(ch.get_full_text()) for ch in chapters)
    print(f"Caractères totaux: {total_chars:,}")
    print()

    for ch in chapters:
        print(f"  {ch.number:2}. {ch.title} ({len(ch.get_full_text()):,} chars)")

    # Créer le dossier de sortie
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDossier de sortie: {output_dir}")

    # Sélectionner et configurer le moteur
    print(f"\nConfiguration:")
    print(f"  Moteur: {engine}")
    print(f"  Émotion: {emotion} (intensité: {emotion_strength})")
    print(f"  Vitesse: {speed}x")
    if reference_audio:
        print(f"  Audio référence: {reference_audio}")

    # Import du moteur HQ
    try:
        from tts_hq_engine import HQAudioReader, VoiceConfig
    except ImportError:
        print("\nERREUR: Module tts_hq_engine non trouvé")
        return False

    # Configuration voix
    voice_config = VoiceConfig(
        reference_audio=reference_audio,
        emotion=emotion,
        emotion_strength=emotion_strength,
        speed=speed
    )

    # Initialiser le reader
    try:
        reader = HQAudioReader(preferred_engine=engine, voice_config=voice_config)
    except Exception as e:
        print(f"\nERREUR initialisation moteur: {e}")
        return False

    print(f"\n{'─' * 70}")
    print("Conversion en cours...")
    print(f"{'─' * 70}\n")

    success_count = 0
    start_time = datetime.now()

    for i, chapter in enumerate(chapters, 1):
        print_progress(i, len(chapters), chapter.title, start_time)

        # Nom du fichier de sortie
        filename = chapter.get_filename()
        ext = ".wav" if output_format == "wav" else ".mp3"
        output_path = output_dir / f"{filename}{ext}"

        # Préparer le texte avec contexte
        text = chapter.get_full_text()

        # Ajouter pause naturelle entre titre et contenu
        text = text.replace("\n\n", "\n...\n", 1)

        # Convertir
        try:
            success = reader.synthesize_chapter_sync(text, output_path)
            if success:
                success_count += 1
        except Exception as e:
            print(f"\n  ! Erreur chapitre {i}: {e}")

    # Résumé
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n\n{'=' * 70}")
    print(f"   CONVERSION TERMINÉE")
    print(f"{'=' * 70}")
    print(f"""
  Résultat: {success_count}/{len(chapters)} chapitres convertis
  Durée: {int(elapsed//60)}m {int(elapsed%60):02d}s
  Sortie: {output_dir}
""")

    if success_count == len(chapters):
        print("  Status: ✓ Succès complet")
    else:
        print(f"  Status: ! {len(chapters) - success_count} échecs")

    # Rappel distribution
    print(f"""
{'─' * 70}
PROCHAINES ÉTAPES - DISTRIBUTION:
{'─' * 70}
  1. Vérifier la qualité audio des fichiers générés
  2. Post-traitement recommandé:
     - Normalisation à -20dB RMS (requis Audible specs)
     - Noise floor < -60dB
     - Ajout room tone 0.75s début / 2s fin

  Plateformes de distribution AI-friendly:
  • Google Play Books: https://play.google.com/books/publish
  • Findaway Voices: https://findawayvoices.com
  • Kobo Writing Life: https://www.kobo.com/writinglife
""")

    return success_count == len(chapters)


def main():
    parser = argparse.ArgumentParser(
        description="Convertisseur audiobook haute qualité pour publication professionnelle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Conversion basique (auto-détection meilleur moteur)
  python audio_reader_hq.py mon_livre.md

  # Avec Chatterbox et émotion
  python audio_reader_hq.py livre.md --engine chatterbox --emotion happy

  # Voice cloning avec F5-TTS
  python audio_reader_hq.py livre.md --engine f5 --reference-audio ma_voix.wav

  # Vérifier les moteurs disponibles
  python audio_reader_hq.py --check-engines

Tags émotionnels (dans le texte):
  Chatterbox: [laugh], [cough], [sigh], [gasp], [chuckle]
  Orpheus: <laugh>, <sigh>, <cough>, <groan>, <yawn>
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
        help="Dossier de sortie (défaut: ./output_hq/<nom_livre>)"
    )

    parser.add_argument(
        "-e", "--engine",
        choices=["auto", "chatterbox", "f5", "orpheus", "edge"],
        default="auto",
        help="Moteur TTS à utiliser (défaut: auto)"
    )

    parser.add_argument(
        "--reference-audio",
        type=str,
        default=None,
        help="Audio de référence pour voice cloning (F5-TTS, Chatterbox)"
    )

    parser.add_argument(
        "--emotion",
        choices=["neutral", "happy", "sad", "angry", "surprised", "whisper"],
        default="neutral",
        help="Émotion de base (défaut: neutral)"
    )

    parser.add_argument(
        "--emotion-strength",
        type=float,
        default=0.5,
        help="Intensité de l'émotion 0.0-1.0 (défaut: 0.5)"
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Vitesse de lecture 0.5-2.0 (défaut: 1.0)"
    )

    parser.add_argument(
        "--header-level",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Niveau des headers pour chapitres (1=#, 2=##)"
    )

    parser.add_argument(
        "--format",
        choices=["wav", "mp3"],
        default="wav",
        help="Format de sortie (défaut: wav pour qualité max)"
    )

    parser.add_argument(
        "--check-engines",
        action="store_true",
        help="Vérifier les moteurs HQ disponibles"
    )

    args = parser.parse_args()

    # Vérification moteurs
    if args.check_engines:
        check_engines()
        return 0

    # Vérifier fichier d'entrée
    if not args.input_file:
        parser.print_help()
        return 1

    if not args.input_file.exists():
        print(f"ERREUR: Fichier non trouvé - {args.input_file}")
        return 1

    # Dossier de sortie
    if args.output:
        output_dir = args.output
    else:
        output_dir = Path("output_hq") / args.input_file.stem

    # Conversion
    success = convert_book_hq(
        input_file=args.input_file,
        output_dir=output_dir,
        engine=args.engine,
        reference_audio=args.reference_audio,
        emotion=args.emotion,
        emotion_strength=args.emotion_strength,
        speed=args.speed,
        header_level=args.header_level,
        output_format=args.format
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
