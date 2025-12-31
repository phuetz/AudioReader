#!/usr/bin/env python3
"""
AudioReader Kokoro - Conversion haute qualité pour publication.

Kokoro-82M: Qualité proche d'ElevenLabs, 5x temps réel sur CPU.

Usage:
    python audio_reader_kokoro.py livre.md
    python audio_reader_kokoro.py livre.md --voice af_heart --lang en-us
    python audio_reader_kokoro.py --list-voices
    python audio_reader_kokoro.py --download-model
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from markdown_parser import parse_book
from tts_kokoro_engine import KokoroEngine, KOKORO_VOICES
from french_preprocessor import FrenchTextPreprocessor
from audiobook_packager import package_from_directory, AudiobookMetadata
from audio_postprocess import (
    convert_to_mp3, get_audio_duration, postprocess_audio, PostProcessConfig
)


def print_voices():
    """Affiche les voix disponibles."""
    print("\n" + "=" * 60)
    print("   VOIX KOKORO DISPONIBLES")
    print("=" * 60)

    langs = {}
    for vid, info in KOKORO_VOICES.items():
        lang = info["lang"]
        if lang not in langs:
            langs[lang] = []
        langs[lang].append((vid, info))

    for lang in sorted(langs.keys()):
        print(f"\n  [{lang.upper()}]")
        for vid, info in langs[lang]:
            gender = "♀" if info["gender"] == "F" else "♂"
            print(f"    {vid:15} {gender} {info['name']:12} - {info['desc']}")

    print()


def convert_book(
    input_file: Path,
    output_dir: Path,
    voice: str,
    speed: float,
    header_level: int,
    output_format: str,
    preprocess: bool = True,
    postprocess: bool = False
):
    """Convertit un livre avec Kokoro."""

    print("\n" + "=" * 70)
    print("   AUDIO READER KOKORO - Haute Qualité")
    print("=" * 70)

    # Initialiser le préprocesseur français
    # Basé sur la langue de la voix (pas juste le préfixe ff_)
    french_preprocessor = None
    voice_info = KOKORO_VOICES.get(voice, {})
    voice_lang = voice_info.get("lang", "")
    if preprocess and voice_lang.startswith("fr"):
        french_preprocessor = FrenchTextPreprocessor()
        print("Préprocesseur français: activé")

    # Vérifier le modèle
    engine = KokoroEngine(voice=voice, speed=speed)

    if not engine.is_available():
        print("\n⚠ Modèle Kokoro non trouvé!")
        print("\nTéléchargez-le avec:")
        print("  python audio_reader_kokoro.py --download-model")
        print("\nOu manuellement:")
        print('  curl -L -o kokoro-v1.0.onnx "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"')
        print('  curl -L -o voices-v1.0.bin "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"')
        return False

    # Parser le livre
    print(f"\nFichier: {input_file}")
    try:
        chapters = parse_book(input_file, header_level=header_level)
    except FileNotFoundError:
        print(f"ERREUR: Fichier non trouvé - {input_file}")
        return False

    if not chapters:
        print("ERREUR: Aucun chapitre trouvé.")
        return False

    total_chars = sum(len(ch.get_full_text()) for ch in chapters)
    print(f"Chapitres: {len(chapters)} | Caractères: {total_chars:,}")

    # Afficher chapitres
    for ch in chapters:
        print(f"  {ch.number}. {ch.title}")

    # Config
    voice_info = KOKORO_VOICES.get(voice, {"name": voice, "desc": ""})
    lang = engine.get_lang()

    print(f"\nVoix: {voice_info['name']} ({voice_info.get('desc', '')})")
    print(f"Langue: {lang}")
    print(f"Vitesse: {speed}x")

    # Créer dossier
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sortie: {output_dir}")

    # Conversion
    print("\n" + "-" * 70)
    print("Conversion en cours...")
    print("-" * 70)

    total_gen_time = 0
    total_audio_time = 0
    success_count = 0

    for chapter in chapters:
        text = chapter.get_full_text()
        filename = chapter.get_filename()

        # Appliquer le préprocesseur français si activé
        if french_preprocessor:
            text = french_preprocessor.process(text)

        # Toujours générer en WAV d'abord
        wav_path = output_dir / f"{filename}.wav"
        final_ext = ".wav" if output_format == "wav" else ".mp3"
        final_path = output_dir / f"{filename}{final_ext}"

        print(f"\n  [{chapter.number}/{len(chapters)}] {chapter.title}")
        print(f"      {len(text):,} caractères", end="", flush=True)

        start = time.time()
        success = engine.synthesize(text, wav_path)
        gen_time = time.time() - start

        if success:
            # Post-processing si activé (normalisation, fades)
            if postprocess:
                print(" [post]", end="", flush=True)
                config = PostProcessConfig(
                    normalize=True,
                    fade_in_ms=300,
                    fade_out_ms=500,
                    trim_silence=False  # Désactivé par défaut
                )
                processed_path = output_dir / f"{filename}_processed.wav"
                if postprocess_audio(str(wav_path), str(processed_path), config):
                    wav_path.unlink()  # Supprimer l'original
                    processed_path.rename(wav_path)  # Renommer

            # Conversion MP3 si demandé
            if output_format == "mp3":
                print(" [mp3]", end="", flush=True)
                if convert_to_mp3(str(wav_path), str(final_path)):
                    wav_path.unlink()  # Supprimer le WAV intermédiaire
                else:
                    print(" → ERREUR MP3")
                    continue

            success_count += 1
            total_gen_time += gen_time

            # Calculer durée audio (robuste avec ffprobe)
            audio_duration = get_audio_duration(str(final_path))
            total_audio_time += audio_duration

            ratio = audio_duration / gen_time if gen_time > 0 else 0
            print(f" → {audio_duration:.1f}s audio | {gen_time:.1f}s | {ratio:.1f}x RT")
        else:
            print(" → ERREUR")

    # Résumé
    print("\n" + "=" * 70)
    print("   RÉSULTAT")
    print("=" * 70)

    overall_ratio = total_audio_time / total_gen_time if total_gen_time > 0 else 0

    print(f"""
  Chapitres: {success_count}/{len(chapters)} convertis
  Audio:     {total_audio_time:.1f}s ({total_audio_time/60:.1f} min)
  Temps:     {total_gen_time:.1f}s
  Vitesse:   {overall_ratio:.1f}x temps réel

  Fichiers dans: {output_dir}/
""")

    # Lister fichiers
    total_size = 0
    for f in sorted(output_dir.glob(f"*{final_ext}")):
        size = f.stat().st_size / 1024 / 1024
        total_size += size
        print(f"    {f.name}")

    print(f"\n  Taille totale: {total_size:.1f} MB")

    # Guide post-production (seulement si pas déjà fait)
    if output_format == "wav" and not postprocess:
        print(f"""
{'─' * 70}
POST-PRODUCTION (ou relancer avec --postprocess):
{'─' * 70}
  Option automatique:
     python audio_reader_kokoro.py livre.md --postprocess

  Ou manuellement:
  1. Normaliser à -20dB LUFS:
     ffmpeg -i input.wav -af "loudnorm=I=-20:TP=-3:LRA=11" output.wav

  2. Convertir en MP3 192kbps:
     ffmpeg -i input.wav -b:a 192k -ar 44100 output.mp3

Distribution AI-friendly:
  • Google Play Books: https://play.google.com/books/publish
  • Findaway/Spotify: https://findawayvoices.com
  • Kobo: https://www.kobo.com/writinglife
""")

    return success_count == len(chapters)


def main():
    parser = argparse.ArgumentParser(
        description="Convertisseur audiobook haute qualité avec Kokoro-82M",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python audio_reader_kokoro.py livre.md
  python audio_reader_kokoro.py livre.md --voice af_heart
  python audio_reader_kokoro.py livre.md --speed 0.9
  python audio_reader_kokoro.py --list-voices
  python audio_reader_kokoro.py --download-model

Kokoro-82M:
  - 82 millions de paramètres
  - Qualité proche d'ElevenLabs
  - ~5x temps réel sur CPU
  - Voix française: ff_siwis
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
        help="Dossier de sortie"
    )

    parser.add_argument(
        "-v", "--voice",
        default="ff_siwis",
        help="Voix à utiliser (défaut: ff_siwis - française)"
    )

    parser.add_argument(
        "-s", "--speed",
        type=float,
        default=1.0,
        help="Vitesse de lecture 0.5-2.0 (défaut: 1.0)"
    )

    parser.add_argument(
        "--header-level",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Niveau des headers pour chapitres"
    )

    parser.add_argument(
        "--format",
        choices=["wav", "mp3"],
        default="wav",
        help="Format de sortie (défaut: wav)"
    )

    parser.add_argument(
        "--m4b",
        action="store_true",
        help="Créer un audiobook M4B avec chapitres"
    )

    parser.add_argument(
        "--author",
        default="",
        help="Auteur du livre (pour M4B)"
    )

    parser.add_argument(
        "--cover",
        type=Path,
        default=None,
        help="Image de couverture (pour M4B)"
    )

    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="Afficher les voix disponibles"
    )

    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Télécharger le modèle Kokoro"
    )

    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Désactiver le préprocesseur français (pour debug)"
    )

    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Appliquer post-production (normalisation -20 LUFS, fades)"
    )

    args = parser.parse_args()

    # Liste des voix
    if args.list_voices:
        print_voices()
        return 0

    # Téléchargement modèle
    if args.download_model:
        print("\nTéléchargement du modèle Kokoro-82M...")
        success = KokoroEngine.download_model()
        return 0 if success else 1

    # Conversion
    if not args.input_file:
        parser.print_help()
        return 1

    if not args.input_file.exists():
        print(f"ERREUR: Fichier non trouvé - {args.input_file}")
        return 1

    output_dir = args.output or Path("output_kokoro") / args.input_file.stem

    success = convert_book(
        input_file=args.input_file,
        output_dir=output_dir,
        voice=args.voice,
        speed=args.speed,
        header_level=args.header_level,
        output_format=args.format,
        preprocess=not args.no_preprocess,
        postprocess=args.postprocess
    )

    # Créer M4B si demandé
    if success and args.m4b:
        print("\n" + "=" * 70)
        print("   CRÉATION AUDIOBOOK M4B")
        print("=" * 70 + "\n")

        # Extraire le titre du nom de fichier
        title = args.input_file.stem.replace("_", " ").replace("-", " ").title()

        # Métadonnées
        voice_info = KOKORO_VOICES.get(args.voice, {})
        narrator = voice_info.get("name", "AI Voice")

        metadata = AudiobookMetadata(
            title=title,
            author=args.author or "Unknown",
            narrator=narrator,
            genre="Audiobook"
        )

        if args.cover and args.cover.exists():
            metadata.cover_image = str(args.cover)

        # Créer le M4B
        m4b_output = output_dir / f"{args.input_file.stem}.m4b"
        try:
            package_from_directory(
                input_dir=str(output_dir),
                output_file=str(m4b_output),
                metadata=metadata,
                pattern="*.wav"
            )
            print(f"\n✓ Audiobook M4B: {m4b_output}")
        except Exception as e:
            print(f"\n✗ Erreur création M4B: {e}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
