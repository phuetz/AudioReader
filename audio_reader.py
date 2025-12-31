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


import soundfile as sf
import numpy as np

def pipeline_synthesize_chapter(pipeline, text, output_path):
    """Synthétise un chapitre complet avec le pipeline HQ."""
    try:
        from src.hq_pipeline_extended import AudiobookGenerator
        import tempfile
        import os
        
        # 1. Initialiser le générateur
        generator = AudiobookGenerator(config=pipeline.config)
        generator.pipeline = pipeline
        
        # 2. Processus (Analyse -> Segments)
        segments = pipeline.process_chapter(text)
        
        # 3. Récupérer le moteur approprié
        # Si le mode XTTS est activé dans la config
        is_xtts = pipeline.config.tts_engine == "xtts" or pipeline.config.enable_voice_cloning
        
        from src.tts_engine import create_tts_engine
        engine = create_tts_engine(
            language=pipeline.config.lang,
            engine_type="xtts" if is_xtts else "kokoro",
            voice=pipeline.config.narrator_voice
        )
        
        def synth_fn(t, v, s):
            # Utilise un fichier temporaire pour récupérer l'audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                # Le moteur unifié gère le routage vers Kokoro ou XTTS
                # Si v est un chemin vers un WAV (clonage), XTTS le gérera via speaker_wav
                success = engine.synthesize(t, tmp_path, voice=v, speed=s, speaker_wav=v if (is_xtts and os.path.exists(str(v))) else None)
                
                if success and os.path.exists(tmp_path):
                    audio, _ = sf.read(tmp_path)
                    return audio
                return np.array([], dtype=np.float32)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            
        # Synthèse des segments
        audios = pipeline.synthesize_segments(segments, synth_fn)
        
        # 4. Concaténation
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
                    os.replace(mastered_path, output_path)

        return True
    except Exception as e:
        print(f"Erreur lors de la synthèse HQ : {e}")
        import traceback
        traceback.print_exc()
        return False

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
    use_cache: bool = True
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

    # Si clonage demandé, on force XTTS si auto
    if clone_path:
        if engine_type == "auto":
            engine_type = "xtts"
            print("Note: Moteur basculé sur XTTS pour le support du clonage.")
        elif engine_type != "xtts":
            print("Attention: L'argument --clone est ignoré si le moteur n'est pas XTTS.")

    # Initialiser le moteur ou pipeline
    print(f"\nConfiguration TTS:")
    print(f"  Langue: {language}")
    
    if hq and HAS_HQ:
        print(f"  Moteur: Pipeline HQ étendu (v2.4)")
        print(f"  Style: {style}")
        print(f"  Multi-voix: {'Oui' if multivoice else 'Non'}")
        print(f"  Mastering: {'Oui' if master else 'Non'}")
        if clone_path:
             print(f"  Clonage: {clone_path} (via HQ Voice Cloning)")
        
        config = ExtendedPipelineConfig(
            lang=language,
            narrator_voice=voice,
            enable_dialogue_attribution=multivoice,
            auto_assign_voices=multivoice,
            default_narration_style=style,
            enable_acx_compliance=master,
            enable_audio_enhancement=master,
            enable_cache=use_cache,
            # Activer le clonage si un fichier est fourni
            enable_voice_cloning=bool(clone_path)
        )
        pipeline = ExtendedHQPipeline(config)
        
        # Enregistrer la voix clonée dans le pipeline HQ
        if clone_path and pipeline.cloning_manager:
            pipeline.cloning_manager.register_cloned_voice("narrator", clone_path)
            # Assigner cette voix au narrateur
            pipeline.config.narrator_voice = "narrator"

        engine = None
    else:
        if hq and not HAS_HQ:
            print("⚠️ Pipeline HQ non disponible, utilisation du moteur standard.")
        
        print(f"  Moteur: {engine_type}")
        if voice:
            print(f"  Voix: {voice}")
        if clone_path and engine_type == "xtts":
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

    success_count = 0
    for i, chapter in enumerate(chapters, 1):
        print_progress(i, len(chapters), chapter.title)

        # Nom du fichier de sortie
        filename = chapter.get_filename()
        output_path = output_dir / f"{filename}.wav"

        # Convertir en audio
        text = chapter.get_full_text()
        
        if pipeline:
            success = pipeline_synthesize_chapter(pipeline, text, output_path)
        else:
            # Passer le fichier de clonage si nécessaire
            kw = {}
            if clone_path and engine_type == "xtts":
                kw['speaker_wav'] = str(clone_path)
                
            success = engine.synthesize(text, output_path, **kw)

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
        choices=["auto", "mms", "kokoro", "xtts", "edge"],
        help="Moteur TTS (auto, mms, kokoro, xtts, edge)"
    )

    parser.add_argument(
        "--clone",
        type=Path,
        help="Fichier audio de référence pour le clonage de voix (avec moteur XTTS)"
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

    # --- HQ Options ---
    hq_group = parser.add_argument_group("HQ Pipeline Options (v2.4)")
    
    hq_group.add_argument(
        "--hq",
        action="store_true",
        help="Utilise le pipeline Haute Qualité (plus lent, mais bien meilleur)"
    )

    hq_group.add_argument(
        "--multivoice",
        action="store_true",
        help="Active la détection et l'attribution automatique des voix pour les dialogues"
    )

    hq_group.add_argument(
        "--style",
        choices=["storytelling", "formal", "conversational", "dramatic"],
        default="storytelling",
        help="Style de narration (pour le mode HQ)"
    )

    hq_group.add_argument(
        "--master",
        action="store_true",
        help="Active le mastering audio final (conforme ACX/Audible)"
    )

    hq_group.add_argument(
        "--no-cache",
        action="store_false",
        dest="use_cache",
        default=True,
        help="Désactive le cache de synthèse"
    )

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

    args = parser.parse_args()

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
        header_level=args.header_level,
        clone_path=args.clone,
        hq=args.hq,
        multivoice=args.multivoice,
        style=args.style,
        master=args.master,
        use_cache=args.use_cache
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
