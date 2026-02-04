#!/usr/bin/env python3
"""
AudioReader v4.0 - Convertit un livre Markdown en fichiers audio.

Usage:
    python audio_reader.py livre.md
    python audio_reader.py livre.md --language en --engine kokoro
    python audio_reader.py livre.md --output ./mon_audiobook
    python audio_reader.py livre.md --dry-run
    python audio_reader.py livre.md --resume
    python audio_reader.py --list-voices

Moteurs TTS:
    - Kokoro: Voix expressives, rapide (defaut)
    - MMS (Meta): Francais natif haute qualite
    - Chatterbox: Clonage voix, controle emotionnel
    - Dia: Multi-speakers natif
    - F5-TTS: Flow matching, CPU-friendly
    - XTTS-v2: Clonage haute qualite
    - Edge: Microsoft Edge TTS (online, fallback)
"""
import argparse
import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from markdown_parser import parse_book, Chapter
from tts_engine import create_tts_engine, EngineType

# Optional HQ imports
try:
    from src.hq_pipeline_extended import (
        ExtendedPipelineConfig,
        ExtendedHQPipeline,
        create_extended_pipeline
    )
    HAS_HQ = True
except ImportError:
    try:
        from hq_pipeline_extended import (
            ExtendedPipelineConfig,
            ExtendedHQPipeline,
            create_extended_pipeline
        )
        HAS_HQ = True
    except ImportError:
        HAS_HQ = False

# Optional config loader
try:
    from src.config_loader import load_config, merge_cli_args, AudioReaderConfig
    HAS_CONFIG = True
except ImportError:
    try:
        from config_loader import load_config, merge_cli_args, AudioReaderConfig
        HAS_CONFIG = True
    except ImportError:
        HAS_CONFIG = False

# Optional progress checkpoint
try:
    from src.progress_checkpoint import ProgressCheckpoint, Checkpoint
    HAS_CHECKPOINT = True
except ImportError:
    try:
        from progress_checkpoint import ProgressCheckpoint, Checkpoint
        HAS_CHECKPOINT = True
    except ImportError:
        HAS_CHECKPOINT = False

# Optional time estimator
try:
    from src.time_estimator import estimate_conversion_time, ConversionEstimate
    HAS_ESTIMATOR = True
except ImportError:
    try:
        from time_estimator import estimate_conversion_time, ConversionEstimate
        HAS_ESTIMATOR = True
    except ImportError:
        HAS_ESTIMATOR = False

# Optional rich progress
try:
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# Moteurs et voix disponibles
ENGINES = {
    "auto": "Selection automatique selon la langue",
    "kokoro": "Kokoro - Voix expressives, rapide (defaut)",
    "mms": "MMS-TTS (Meta) - Qualite native multilingue",
    "chatterbox": "Chatterbox - Clonage voix, controle emotionnel",
    "dia": "Dia 1.6B - Multi-speakers natif",
    "f5": "F5-TTS - Flow matching, CPU-friendly",
    "xtts": "XTTS-v2 - Clonage haute qualite",
    "edge": "Edge-TTS (Microsoft) - Online",
}

# Voix Kokoro disponibles
KOKORO_VOICES = {
    "ff_siwis": "Siwis - Femme francaise",
    "af_heart": "Heart - Femme americaine",
    "af_bella": "Bella - Femme americaine",
    "af_sarah": "Sarah - Femme americaine",
    "am_adam": "Adam - Homme americain",
    "am_michael": "Michael - Homme americain",
    "bf_emma": "Emma - Femme britannique",
    "bm_george": "George - Homme britannique",
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
    print("-" * 60)
    for engine_id, description in ENGINES.items():
        print(f"  {engine_id:12} - {description}")

    print("\n=== Voix Kokoro ===")
    print("-" * 60)
    for voice_id, description in KOKORO_VOICES.items():
        print(f"  {voice_id:15} - {description}")

    print("\n=== Voix Edge-TTS (fallback) ===")
    print("-" * 60)
    for voice_id, description in EDGE_VOICES.items():
        print(f"  {voice_id:25} - {description}")
    print()


class ProgressReporter:
    """Gere l'affichage de la progression (rich ou fallback)."""

    def __init__(self, total: int, use_rich: bool = True):
        self.total = total
        self.use_rich = use_rich and HAS_RICH
        self._progress = None
        self._task = None

    def __enter__(self):
        if self.use_rich:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
            )
            self._progress.__enter__()
            self._task = self._progress.add_task("Conversion", total=self.total)
        return self

    def __exit__(self, *args):
        if self._progress:
            self._progress.__exit__(*args)

    def update(self, current: int, chapter_title: str):
        if self.use_rich and self._progress:
            self._progress.update(self._task, completed=current, description=chapter_title[:40])
        else:
            percent = (current / self.total) * 100
            bar_length = 30
            filled = int(bar_length * current / self.total)
            bar = "=" * filled + "-" * (bar_length - filled)
            print(f"\r[{bar}] {percent:5.1f}% - {chapter_title[:40]}", end="", flush=True)

    def finish(self):
        if not self.use_rich:
            print()


import soundfile as sf
import numpy as np


def pipeline_synthesize_chapter(pipeline, text, output_path):
    """Synthetise un chapitre complet avec le pipeline HQ."""
    try:
        from src.hq_pipeline_extended import AudiobookGenerator
        import tempfile
        import os

        # 1. Initialiser le generateur
        generator = AudiobookGenerator(config=pipeline.config)
        generator.pipeline = pipeline

        # 2. Processus (Analyse -> Segments)
        segments = pipeline.process_chapter(text)

        # 3. Recuperer le moteur approprie
        is_xtts = pipeline.config.tts_engine == "xtts" or pipeline.config.enable_voice_cloning

        from src.tts_engine import create_tts_engine
        engine = create_tts_engine(
            language=pipeline.config.lang,
            engine_type="xtts" if is_xtts else "kokoro",
            voice=pipeline.config.narrator_voice
        )

        def synth_fn(t, v, s):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                success = engine.synthesize(
                    t, tmp_path, voice=v, speed=s,
                    speaker_wav=v if (is_xtts and os.path.exists(str(v))) else None
                )

                if success and os.path.exists(tmp_path):
                    audio, _ = sf.read(tmp_path)
                    return audio
                return np.array([], dtype=np.float32)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # Synthese des segments
        audios = pipeline.synthesize_segments(segments, synth_fn)

        # 4. Concatenation
        full_audio = generator._concatenate_with_pauses(segments, audios)

        # 5. Sauvegarde
        sf.write(str(output_path), full_audio, 24000)

        # 6. Mastering final
        if pipeline.config.enable_audio_enhancement:
            from src.audio_enhancer import AudioEnhancer
            enhancer = AudioEnhancer()
            if enhancer.is_available():
                mastered_path = output_path.with_name(f"{output_path.stem}_mastered.wav")
                if hasattr(pipeline.config, 'acx_target_lufs'):
                    enhancer.config.target_lufs = pipeline.config.acx_target_lufs

                success = enhancer.enhance_file(output_path, mastered_path)
                if success and mastered_path.exists():
                    import os as os_module
                    os_module.replace(mastered_path, output_path)

        return True
    except Exception as e:
        print(f"Erreur lors de la synthese HQ : {e}")
        import traceback
        traceback.print_exc()
        return False


def dry_run(input_file: Path, language: str, engine_type: str, hq: bool):
    """Execute une analyse sans synthese."""
    print(f"\n=== Mode Dry-Run (Analyse sans synthese) ===\n")
    print(f"Fichier: {input_file}")

    # Parser le livre
    try:
        chapters = parse_book(input_file, header_level=1)
    except FileNotFoundError:
        print(f"Erreur: Fichier non trouve - {input_file}")
        return False

    if not chapters:
        print("Erreur: Aucun chapitre trouve.")
        return False

    # Statistiques
    total_chars = sum(len(ch.get_full_text()) for ch in chapters)
    total_words = sum(len(ch.get_full_text().split()) for ch in chapters)

    print(f"\n--- Chapitres ({len(chapters)}) ---")
    for ch in chapters:
        text = ch.get_full_text()
        print(f"  {ch.number:2}. {ch.title[:40]:<40} ({len(text.split()):>5} mots)")

    # Detection de personnages
    try:
        from src.character_detector import CharacterDetector
        detector = CharacterDetector(lang=language)
        all_text = "\n".join(ch.get_full_text() for ch in chapters)
        detector.detect_dialogue_segments(all_text)
        characters = detector.get_characters()
        if characters:
            print(f"\n--- Personnages detectes ({len(characters)}) ---")
            for char in characters[:10]:
                print(f"  - {char.name} ({char.gender or '?'}): {char.occurrence_count} occurrences")
    except ImportError:
        pass

    # Estimation du temps
    if HAS_ESTIMATOR:
        estimate = estimate_conversion_time(all_text, engine=engine_type, hq=hq, chapter_count=len(chapters))
        print(f"\n--- Estimations ---")
        print(f"  Caracteres totaux: {estimate.total_chars:,}")
        print(f"  Mots totaux: {estimate.total_words:,}")
        print(f"  Duree audio estimee: {estimate.audio_duration_formatted}")
        print(f"  Temps de traitement estime: {estimate.processing_time_formatted}")
        print(f"  Taille fichier estimee: {estimate.estimated_file_size_mb:.1f} MB")
    else:
        print(f"\n--- Statistiques ---")
        print(f"  Caracteres totaux: {total_chars:,}")
        print(f"  Mots totaux: {total_words:,}")
        # Estimation simple
        audio_min = total_chars / 1000
        print(f"  Duree audio estimee: ~{int(audio_min)} min")

    print()
    return True


def convert_book(
    input_file: Path,
    output_dir: Path,
    language: str,
    engine_type: str,
    voice: str,
    speed: float,
    header_level: int,
    clone_path: Path = None,
    hq: bool = False,
    multivoice: bool = False,
    style: str = "storytelling",
    master: bool = False,
    use_cache: bool = True,
    resume: bool = False,
    ambiance: str = None,
    chapter_jingle: str = None,
    subtitles: str = None,
):
    """Convertit un livre Markdown en fichiers audio."""

    print(f"\nLecture du fichier: {input_file}")

    # Parser le livre
    try:
        chapters = parse_book(input_file, header_level=header_level)
    except FileNotFoundError:
        print(f"Erreur: Fichier non trouve - {input_file}")
        return False

    if not chapters:
        print("Erreur: Aucun chapitre trouve dans le fichier.")
        return False

    print(f"Chapitres trouves: {len(chapters)}")
    for ch in chapters:
        print(f"  {ch.number}. {ch.title}")

    # Creer le dossier de sortie
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDossier de sortie: {output_dir}")

    # Checkpoint pour la reprise
    checkpoint_mgr = None
    start_chapter = 0
    if HAS_CHECKPOINT and resume:
        checkpoint_mgr = ProgressCheckpoint(output_dir)
        book_hash = checkpoint_mgr.compute_book_hash(input_file)
        if checkpoint_mgr.can_resume(book_hash):
            start_chapter = checkpoint_mgr.get_resume_chapter(book_hash)
            print(f"Reprise au chapitre {start_chapter + 1} (checkpoint trouve)")
        else:
            # Creer un nouveau checkpoint
            cp = Checkpoint(
                book_hash=book_hash,
                book_path=str(input_file),
                output_dir=str(output_dir),
                last_completed_chapter=-1,
                total_chapters=len(chapters),
                engine=engine_type,
                hq=hq,
            )
            checkpoint_mgr.save(cp)

    # Si clonage demande, on force XTTS ou Chatterbox si auto
    if clone_path:
        if engine_type == "auto":
            # Preference Chatterbox > F5 > XTTS
            for eng in ["chatterbox", "f5", "xtts"]:
                try:
                    if eng == "chatterbox":
                        from src.tts_chatterbox_engine import ChatterboxEngine
                        if ChatterboxEngine.is_available():
                            engine_type = eng
                            break
                    elif eng == "f5":
                        from src.tts_f5_engine import F5Engine
                        if F5Engine.is_available():
                            engine_type = eng
                            break
                    else:
                        engine_type = "xtts"
                        break
                except ImportError:
                    continue
            print(f"Note: Moteur bascule sur {engine_type} pour le support du clonage.")

    # Initialiser le moteur ou pipeline
    print(f"\nConfiguration TTS:")
    print(f"  Langue: {language}")

    if hq and HAS_HQ:
        print(f"  Moteur: Pipeline HQ etendu (v4.0)")
        print(f"  Style: {style}")
        print(f"  Multi-voix: {'Oui' if multivoice else 'Non'}")
        print(f"  Mastering: {'Oui' if master else 'Non'}")
        if clone_path:
            print(f"  Clonage: {clone_path}")

        config = ExtendedPipelineConfig(
            lang=language,
            narrator_voice=voice,
            enable_dialogue_attribution=multivoice,
            auto_assign_voices=multivoice,
            default_narration_style=style,
            enable_acx_compliance=master,
            enable_audio_enhancement=master,
            enable_cache=use_cache,
            enable_voice_cloning=bool(clone_path)
        )
        pipeline = ExtendedHQPipeline(config)

        if clone_path and pipeline.cloning_manager:
            pipeline.cloning_manager.register_cloned_voice("narrator", clone_path)
            pipeline.config.narrator_voice = "narrator"

        engine = None
    else:
        if hq and not HAS_HQ:
            print("Warning: Pipeline HQ non disponible, utilisation du moteur standard.")

        print(f"  Moteur: {engine_type}")
        if voice:
            print(f"  Voix: {voice}")
        if clone_path:
            print(f"  Clonage: {clone_path}")
        print(f"  Vitesse: {speed}x")

        engine = create_tts_engine(
            language=language,
            engine_type=engine_type,
            voice=voice,
            speed=speed
        )
        pipeline = None

    if engine and not engine.is_available():
        print("Erreur: Moteur TTS non disponible")
        return False

    print("\nConversion en cours...")
    print("-" * 50)

    # Optionnel: generateur de jingles
    jingle_audio = None
    if chapter_jingle:
        try:
            from src.chapter_jingles import ChapterJingleGenerator
            jingle_gen = ChapterJingleGenerator()
            jingle_audio = jingle_gen.generate(chapter_jingle)
            print(f"  Jingle inter-chapitres: {chapter_jingle}")
        except ImportError:
            pass

    success_count = 0
    total_to_process = len(chapters) - start_chapter

    with ProgressReporter(total_to_process) as progress:
        for i, chapter in enumerate(chapters):
            if i < start_chapter:
                continue

            progress.update(i - start_chapter + 1, chapter.title)

            # Nom du fichier de sortie
            filename = chapter.get_filename()
            output_path = output_dir / f"{filename}.wav"

            # Convertir en audio
            text = chapter.get_full_text()

            if pipeline:
                success = pipeline_synthesize_chapter(pipeline, text, output_path)
            else:
                kw = {}
                if clone_path:
                    kw['speaker_wav'] = str(clone_path)
                success = engine.synthesize(text, output_path, **kw)

            # Post-traitement optionnel
            if success and output_path.exists():
                # Ambiance
                if ambiance:
                    try:
                        from src.ambiance_engine import AmbianceEngine
                        audio, sr = sf.read(str(output_path))
                        amb_engine = AmbianceEngine(sample_rate=sr)
                        audio = amb_engine.mix(audio, ambiance)
                        sf.write(str(output_path), audio, sr)
                    except ImportError:
                        pass

                # Sous-titres
                if subtitles:
                    try:
                        from src.subtitle_generator import SubtitleGenerator
                        sub_gen = SubtitleGenerator()
                        sub_path = output_path.with_suffix(f".{subtitles}")
                        if subtitles == "srt":
                            sub_gen.generate_srt(str(output_path), text, str(sub_path))
                        elif subtitles == "vtt":
                            sub_gen.generate_vtt(str(output_path), text, str(sub_path))
                    except ImportError:
                        pass

            if success:
                success_count += 1
                if checkpoint_mgr:
                    checkpoint_mgr.update_chapter(i)

        progress.finish()

    print("-" * 50)
    print(f"\nTermine! {success_count}/{total_to_process} chapitres convertis.")
    print(f"Fichiers audio dans: {output_dir}")

    # Nettoyer le checkpoint si tout est termine
    if checkpoint_mgr and success_count == total_to_process:
        checkpoint_mgr.clear()

    return success_count == total_to_process


def main():
    parser = argparse.ArgumentParser(
        description="AudioReader v4.0 - Convertit un livre Markdown en fichiers audio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python audio_reader.py mon_livre.md
  python audio_reader.py mon_livre.md --dry-run
  python audio_reader.py mon_livre.md --hq --master
  python audio_reader.py mon_livre.md --engine chatterbox --clone voix.wav
  python audio_reader.py mon_livre.md --resume
  python audio_reader.py --list-voices

Moteurs TTS (tous gratuits):
  - Kokoro: Voix expressives, rapide (defaut)
  - Chatterbox: Clonage voix avec controle emotionnel
  - Dia: Multi-speakers natif avec tags non-verbaux
  - F5-TTS: Flow matching, CPU-friendly
  - XTTS: Clonage haute qualite
  - MMS: Meta Multilingual Speech
  - Edge: Microsoft Edge TTS (online)
        """
    )

    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        help="Fichier Markdown a convertir"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Dossier de sortie (defaut: ./output/<nom_livre>)"
    )

    parser.add_argument(
        "-l", "--language",
        default="fr",
        help="Langue (fr, en, de, es, etc.) ou 'auto' pour detection"
    )

    parser.add_argument(
        "-e", "--engine",
        default="auto",
        choices=["auto", "kokoro", "mms", "chatterbox", "dia", "f5", "xtts", "edge"],
        help="Moteur TTS"
    )

    parser.add_argument(
        "--clone",
        type=Path,
        help="Fichier audio de reference pour le clonage de voix"
    )

    parser.add_argument(
        "-v", "--voice",
        default="ff_siwis",
        help="Voix (defaut: ff_siwis)"
    )

    parser.add_argument(
        "-s", "--speed",
        type=float,
        default=1.0,
        help="Vitesse de lecture (defaut: 1.0)"
    )

    parser.add_argument(
        "--header-level",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Niveau des headers pour les chapitres (1=#, 2=##, 3=###)"
    )

    # --- v4.0 New Options ---
    v4_group = parser.add_argument_group("v4.0 New Features")

    v4_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyse sans synthese: chapitres, mots, personnages, duree estimee"
    )

    v4_group.add_argument(
        "--resume",
        action="store_true",
        help="Reprendre une conversion interrompue"
    )

    v4_group.add_argument(
        "--subtitles",
        choices=["srt", "vtt"],
        help="Generer des sous-titres synchronises"
    )

    v4_group.add_argument(
        "--ambiance",
        choices=["library", "rain", "fireplace", "cafe", "forest"],
        help="Ajouter une ambiance sonore de fond"
    )

    v4_group.add_argument(
        "--chapter-jingle",
        choices=["chime", "page_turn", "orchestral", "minimal", "silence"],
        help="Jingle entre les chapitres"
    )

    # --- HQ Options ---
    hq_group = parser.add_argument_group("HQ Pipeline Options")

    hq_group.add_argument(
        "--hq",
        action="store_true",
        help="Utilise le pipeline Haute Qualite (plus lent, meilleur resultat)"
    )

    hq_group.add_argument(
        "--multivoice",
        action="store_true",
        help="Detection et attribution automatique des voix pour les dialogues"
    )

    hq_group.add_argument(
        "--style",
        choices=["storytelling", "formal", "conversational", "dramatic", "documentary", "intimate", "energetic"],
        default="storytelling",
        help="Style de narration"
    )

    hq_group.add_argument(
        "--master",
        action="store_true",
        help="Mastering audio final (conforme ACX/Audible)"
    )

    hq_group.add_argument(
        "--no-cache",
        action="store_false",
        dest="use_cache",
        default=True,
        help="Desactive le cache de synthese"
    )

    # --- Utility Options ---
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="Afficher les moteurs et voix disponibles"
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Lancer l'interface graphique (Gradio)"
    )

    parser.add_argument(
        "--api-v2",
        action="store_true",
        help="Lancer le serveur API v2 (FastAPI + React frontend)"
    )

    args = parser.parse_args()

    # Charger la configuration TOML si disponible
    if HAS_CONFIG:
        config = load_config()
        config = merge_cli_args(config, args)
    else:
        config = None

    # Lancer l'API v2
    if args.api_v2:
        print("Lancement de l'API v2...")
        from api import main as api_main
        api_main()
        return 0

    # Lancer le GUI
    if args.gui:
        print("Lancement de l'interface graphique...")
        from app import create_interface
        demo = create_interface()
        demo.launch()
        return 0

    # Afficher les voix
    if args.list_voices:
        print_voices()
        return 0

    # Verifier le fichier d'entree
    if not args.input_file:
        parser.print_help()
        return 1

    if not args.input_file.exists():
        print(f"Erreur: Fichier non trouve - {args.input_file}")
        return 1

    # Detection automatique de langue
    if args.language == "auto":
        try:
            from src.language_detector import AutoLanguageDetector
            detector = AutoLanguageDetector()
            text_sample = args.input_file.read_text(encoding="utf-8")[:5000]
            result = detector.detect(text_sample)
            args.language = result.language
            print(f"Langue detectee: {result.language} (confiance: {result.confidence:.0%})")
        except ImportError:
            args.language = "fr"

    # Mode dry-run
    if args.dry_run:
        success = dry_run(args.input_file, args.language, args.engine, args.hq)
        return 0 if success else 1

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
        header_level=args.header_level,
        clone_path=args.clone,
        hq=args.hq,
        multivoice=args.multivoice,
        style=args.style,
        master=args.master,
        use_cache=args.use_cache,
        resume=args.resume,
        ambiance=args.ambiance,
        chapter_jingle=args.chapter_jingle,
        subtitles=args.subtitles,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
