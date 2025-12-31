#!/usr/bin/env python3
"""
AudioReader - Convertisseur complet de livres en audiobooks.

Script principal intégrant toutes les fonctionnalités:
- Conversion haute qualité avec Kokoro-82M
- Traitement intelligent du texte (chunking, prononciation, émotions)
- Post-traitement audio professionnel
- Export M4B avec chapitres et métadonnées
- Support multi-fichiers et EPUB
- Voice blending et pauses intelligentes

Usage:
    python audioreader.py livre.md
    python audioreader.py mon_livre/                    # Répertoire avec chapitres
    python audioreader.py livre.epub                    # Fichier EPUB
    python audioreader.py livre.md --voice af_heart --author "Victor Hugo"
    python audioreader.py livre.md --voice "af_bella:60,am_adam:40"  # Voice blend
    python audioreader.py livre.md --format m4b --mastering
    python audioreader.py --gui
"""
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from markdown_parser import parse_book
from tts_kokoro_engine import KokoroEngine, KOKORO_VOICES
from text_processor import TextProcessor
from audiobook_builder import AudiobookBuilder, AudiobookMetadata


# Chemins par défaut
MODEL_PATH = Path("kokoro-v1.0.onnx")
VOICES_PATH = Path("voices-v1.0.bin")


def print_header():
    """Affiche l'en-tête."""
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║     █████╗ ██╗   ██╗██████╗ ██╗ ██████╗ ██████╗ ███████╗ █████╗      ║
║    ██╔══██╗██║   ██║██╔══██╗██║██╔═══██╗██╔══██╗██╔════╝██╔══██╗     ║
║    ███████║██║   ██║██║  ██║██║██║   ██║██████╔╝█████╗  ███████║     ║
║    ██╔══██║██║   ██║██║  ██║██║██║   ██║██╔══██╗██╔══╝  ██╔══██║     ║
║    ██║  ██║╚██████╔╝██████╔╝██║╚██████╔╝██║  ██║███████╗██║  ██║     ║
║    ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝     ║
║                                                                       ║
║            Convertisseur de livres en audiobooks                      ║
║                    Propulsé par Kokoro-82M                            ║
╚═══════════════════════════════════════════════════════════════════════╝
""")


def print_voices():
    """Affiche les voix disponibles."""
    print("\n  Voix disponibles:")
    print("  " + "-" * 50)

    current_lang = None
    for vid, info in sorted(KOKORO_VOICES.items(), key=lambda x: x[1]["lang"]):
        if info["lang"] != current_lang:
            current_lang = info["lang"]
            print(f"\n  [{current_lang.upper()}]")

        gender = "♀" if info["gender"] == "F" else "♂"
        print(f"    {vid:15} {gender} {info['name']:12} - {info['desc']}")

    print()


def check_model() -> bool:
    """Vérifie si le modèle est disponible."""
    if MODEL_PATH.exists() and VOICES_PATH.exists():
        return True

    print("\n  ⚠️  Modèle Kokoro non trouvé!")
    print("\n  Téléchargez-le avec:")
    print('    curl -L -o kokoro-v1.0.onnx "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"')
    print('    curl -L -o voices-v1.0.bin "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"')
    print()
    return False


def run_postprocess(audio_dir: Path, target_lufs: float = -20.0) -> Path:
    """Applique le post-traitement audio."""
    import subprocess

    mastered_dir = audio_dir / "mastered"
    mastered_dir.mkdir(exist_ok=True)

    for wav_file in sorted(audio_dir.glob("*.wav")):
        output_file = mastered_dir / wav_file.name

        # Normalisation loudness + EQ + limiter
        cmd = [
            "ffmpeg", "-y", "-i", str(wav_file),
            "-af", f"highpass=f=80,loudnorm=I={target_lufs}:TP=-3:LRA=11,alimiter=limit=-3dB",
            str(output_file)
        ]

        subprocess.run(cmd, capture_output=True)

    return mastered_dir


def convert_book(
    input_file: Path,
    output_dir: Path,
    voice: str = "ff_siwis",
    speed: float = 1.0,
    header_level: int = 1,
    apply_corrections: bool = True,
    mastering: bool = False,
    title: str = None,
    author: str = None,
    output_format: str = "mp3",
    smart_pauses: bool = True,
    sentence_pause: float = 0.3,
    paragraph_pause: float = 0.8,
    resume: bool = False
) -> bool:
    """
    Convertit un livre complet en audiobook.

    Supporte:
    - Fichier Markdown unique
    - Répertoire avec plusieurs fichiers .md
    - Fichier EPUB
    """
    # Déterminer le type d'entrée
    if input_file.is_dir():
        source_type = "répertoire"
    elif input_file.suffix.lower() == '.epub':
        source_type = "EPUB"
    else:
        source_type = "Markdown"

    print(f"\n  Source: {input_file} ({source_type})")

    # Parser le livre
    try:
        chapters = parse_book(input_file, header_level=header_level)
    except FileNotFoundError:
        print(f"  ❌ Fichier non trouvé")
        return False

    if not chapters:
        print("  ❌ Aucun chapitre trouvé")
        return False

    total_chars = sum(len(ch.get_full_text()) for ch in chapters)
    print(f"  Chapitres: {len(chapters)} | Caractères: {total_chars:,}")

    # Afficher chapitres
    for ch in chapters:
        print(f"    {ch.number:2}. {ch.title}")

    # Configuration
    voice_info = KOKORO_VOICES.get(voice, {"name": voice})
    print(f"\n  Voix: {voice_info.get('name', voice)}")
    print(f"  Vitesse: {speed}x")
    print(f"  Corrections: {'Oui' if apply_corrections else 'Non'}")
    print(f"  Mastering: {'Oui' if mastering else 'Non'}")

    # Créer dossiers
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = output_dir / "wav"
    wav_dir.mkdir(exist_ok=True)

    # Initialiser les composants
    engine = KokoroEngine(
        model_path=str(MODEL_PATH),
        voices_path=str(VOICES_PATH),
        voice=voice,
        speed=speed,
        sentence_pause=sentence_pause,
        paragraph_pause=paragraph_pause
    )

    processor = TextProcessor(lang="fr", engine="kokoro") if apply_corrections else None

    # Afficher info sur le voice blending
    if ',' in voice or ':' in voice:
        print(f"  Voice blend: {voice}")

    # Conversion
    print(f"\n  {'─' * 60}")
    print("  Conversion des chapitres...")
    print(f"  {'─' * 60}")

    start_time = time.time()
    total_audio_time = 0

    skipped = 0
    for i, chapter in enumerate(chapters, 1):
        text = chapter.get_full_text()

        # Traitement du texte
        if processor:
            text = processor.process_to_text(text)

        # Générer audio
        output_path = wav_dir / f"{chapter.get_filename()}.wav"

        # Skip si le fichier existe et resume est activé
        if resume and output_path.exists() and output_path.stat().st_size > 0:
            print(f"\n    [{i}/{len(chapters)}] {chapter.title} ✓ (déjà converti)")
            skipped += 1
            # Ajouter la durée au total
            import subprocess
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(output_path)],
                capture_output=True, text=True
            )
            try:
                total_audio_time += float(result.stdout.strip())
            except:
                pass
            continue

        print(f"\n    [{i}/{len(chapters)}] {chapter.title}")
        print(f"        {len(text):,} caractères", end="", flush=True)

        ch_start = time.time()
        success = engine.synthesize(text, output_path, add_smart_pauses=smart_pauses)
        ch_time = time.time() - ch_start

        if success:
            # Calculer durée audio
            import subprocess
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(output_path)],
                capture_output=True, text=True
            )
            try:
                audio_duration = float(result.stdout.strip())
                total_audio_time += audio_duration
                ratio = audio_duration / ch_time if ch_time > 0 else 0
                print(f" → {audio_duration:.1f}s | {ch_time:.1f}s | {ratio:.1f}x RT")
            except:
                print(f" → OK")
        else:
            print(" → ERREUR")

    total_time = time.time() - start_time

    # Post-traitement (mastering)
    if mastering:
        print(f"\n  {'─' * 60}")
        print("  Post-traitement audio...")
        wav_dir = run_postprocess(wav_dir)

    # Construire l'audiobook
    print(f"\n  {'─' * 60}")
    print("  Construction de l'audiobook...")

    metadata = AudiobookMetadata(
        title=title or input_file.stem,
        author=author or "Inconnu",
        narrator="Kokoro TTS"
    )

    builder = AudiobookBuilder(metadata)
    builder.add_chapters_from_dir(wav_dir, "*.wav")

    # Appliquer métadonnées
    builder.apply_metadata_to_all()

    # Export final
    if output_format == "m4b":
        final_path = output_dir / f"{metadata.title}.m4b"
        builder.build_m4b(final_path)
    elif output_format == "mp3-combined":
        final_path = output_dir / f"{metadata.title}.mp3"
        builder.build_combined_mp3(final_path)
    else:
        # Garder les fichiers séparés
        mp3_dir = output_dir / "mp3"
        mp3_dir.mkdir(exist_ok=True)
        for wav in wav_dir.glob("*.wav"):
            mp3_path = mp3_dir / f"{wav.stem}.mp3"
            import subprocess
            subprocess.run([
                "ffmpeg", "-y", "-i", str(wav),
                "-b:a", "192k", "-ar", "44100", "-ac", "1",
                str(mp3_path)
            ], capture_output=True)
        final_path = mp3_dir

    # Résumé
    print(f"\n  {'═' * 60}")
    print("  RÉSUMÉ")
    print(f"  {'═' * 60}")

    overall_ratio = total_audio_time / total_time if total_time > 0 else 0

    print(f"""
    Chapitres:     {len(chapters)}
    Caractères:    {total_chars:,}
    Durée audio:   {total_audio_time/60:.1f} minutes
    Temps total:   {total_time:.1f}s
    Performance:   {overall_ratio:.1f}x temps réel

    Sortie: {final_path}
    """)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="AudioReader - Convertisseur de livres en audiobooks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python audioreader.py livre.md
  python audioreader.py mon_livre/                        # Répertoire avec chapitres
  python audioreader.py livre.epub                        # Fichier EPUB
  python audioreader.py livre.md --voice af_heart --speed 0.9
  python audioreader.py livre.md --voice "af_bella:60,am_adam:40"  # Voice blend
  python audioreader.py livre.md --format m4b --mastering
  python audioreader.py livre.md --paragraph-pause 1.0    # Pauses plus longues
  python audioreader.py --list-voices
  python audioreader.py --gui

Formats d'entrée:
  .md       - Fichier Markdown avec headers pour chapitres
  .epub     - Fichier EPUB (extraction automatique)
  dossier/  - Répertoire avec fichiers .md (un par chapitre)

Formats de sortie:
  mp3       - Fichiers MP3 séparés par chapitre (défaut)
  mp3-combined - Un seul fichier MP3
  m4b       - Audiobook avec chapitres navigables

Voice blending:
  "af_bella:60,am_adam:40"  - 60% Bella, 40% Adam
  "af_bella,am_adam"        - 50% chaque voix
        """
    )

    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        help="Fichier (.md, .epub) ou répertoire contenant les chapitres"
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
        help="Voix (défaut: ff_siwis - française)"
    )

    parser.add_argument(
        "-s", "--speed",
        type=float,
        default=1.0,
        help="Vitesse 0.5-2.0 (défaut: 1.0)"
    )

    parser.add_argument(
        "--title",
        type=str,
        help="Titre de l'audiobook"
    )

    parser.add_argument(
        "--author",
        type=str,
        help="Auteur du livre"
    )

    parser.add_argument(
        "--header-level",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Niveau des headers pour chapitres"
    )

    parser.add_argument(
        "--no-corrections",
        action="store_true",
        help="Désactiver les corrections de prononciation"
    )

    parser.add_argument(
        "--mastering",
        action="store_true",
        help="Appliquer le post-traitement audio"
    )

    parser.add_argument(
        "--format",
        choices=["mp3", "mp3-combined", "m4b"],
        default="mp3",
        help="Format de sortie"
    )

    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="Afficher les voix disponibles"
    )

    parser.add_argument(
        "--no-smart-pauses",
        action="store_true",
        help="Désactiver les pauses intelligentes entre phrases/paragraphes"
    )

    parser.add_argument(
        "--sentence-pause",
        type=float,
        default=0.3,
        help="Durée de pause entre phrases (défaut: 0.3s)"
    )

    parser.add_argument(
        "--paragraph-pause",
        type=float,
        default=0.8,
        help="Durée de pause entre paragraphes (défaut: 0.8s)"
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Lancer l'interface graphique"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reprendre la conversion en sautant les chapitres déjà convertis"
    )

    args = parser.parse_args()

    print_header()

    # Liste des voix
    if args.list_voices:
        print_voices()
        return 0

    # Interface graphique
    if args.gui:
        print("  Lancement de l'interface graphique...")
        print("  Ouvrez http://localhost:7860 dans votre navigateur")
        print()
        from app import create_interface
        demo = create_interface()
        demo.launch(server_name="0.0.0.0", server_port=7860)
        return 0

    # Vérifier le fichier/répertoire d'entrée
    if not args.input_file:
        parser.print_help()
        return 1

    if not args.input_file.exists():
        print(f"  ❌ Fichier/répertoire non trouvé: {args.input_file}")
        return 1

    # Message informatif pour les répertoires
    if args.input_file.is_dir():
        md_files = list(args.input_file.glob("*.md"))
        print(f"\n  Répertoire détecté: {len(md_files)} fichiers .md trouvés")

    # Vérifier le modèle
    if not check_model():
        return 1

    # Dossier de sortie
    output_dir = args.output or Path("output") / args.input_file.stem

    # Conversion
    success = convert_book(
        input_file=args.input_file,
        output_dir=output_dir,
        voice=args.voice,
        speed=args.speed,
        header_level=args.header_level,
        apply_corrections=not args.no_corrections,
        mastering=args.mastering,
        title=args.title,
        author=args.author,
        output_format=args.format,
        smart_pauses=not args.no_smart_pauses,
        sentence_pause=args.sentence_pause,
        paragraph_pause=args.paragraph_pause,
        resume=args.resume
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
