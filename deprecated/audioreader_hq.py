#!/usr/bin/env python3
"""
AudioReader HQ - Pipeline haute qualite pour audiobooks.

Genere des audiobooks de qualite professionnelle (proche ElevenLabs)
avec:
- Multi-voix automatique par personnage
- Prosodie emotionnelle adaptative
- Normalisation avancee du texte
- Post-processing audio broadcast

Usage:
    python audioreader_hq.py livre.md -o output/
    python audioreader_hq.py livre.epub --config config.json
    python audioreader_hq.py livre.md --analyze-only
"""
import argparse
from pathlib import Path
import sys
import time

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent))

from src.hq_pipeline import HQPipeline, HQPipelineConfig, create_hq_pipeline
from src.tts_kokoro_engine import KokoroEngine, KOKORO_VOICES
from src.tts_unified import UnifiedTTS, TTSEngine, TTSConfig, AVAILABLE_VOICES
from src.audio_enhancer import AudioEnhancer, AudioEnhancerConfig, enhance_audiobook
from src.markdown_parser import MarkdownBookParser, EPUBParser


def print_banner():
    """Affiche la banniere."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        AUDIOREADER HQ                                        ║
║                 Pipeline Haute Qualite pour Audiobooks                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Multi-voix | Emotions | Prosodie adaptative | Post-processing broadcast     ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


def print_section(title: str):
    """Affiche un titre de section."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def analyze_text(pipeline: HQPipeline, text: str, verbose: bool = True):
    """Analyse le texte et affiche les resultats."""
    print_section("ANALYSE DU TEXTE")

    segments = pipeline.process_chapter(text)

    if verbose:
        # Personnages
        print("\n--- Personnages detectes ---")
        assignments = pipeline.get_voice_assignments()
        for speaker, voice in assignments.items():
            char_segs = sum(1 for s in segments if s.speaker == speaker)
            print(f"  {speaker:20} -> {voice:15} ({char_segs} segments)")

        # Analyse du chapitre
        print("\n--- Analyse emotionnelle ---")
        analysis = pipeline.get_chapter_analysis()
        print(f"  Ton dominant:       {analysis['tone']}")
        print(f"  Emotion principale: {analysis['dominant_emotion']}")
        print(f"  Intensite max:      {analysis['max_intensity']}")
        print(f"  Nombre de climax:   {analysis['climax_count']}")
        print(f"  Vitesse suggeree:   {analysis['suggested_base_speed']:.2f}x")

        # Distribution des emotions
        print("\n--- Distribution des emotions ---")
        emotion_counts = {}
        for seg in segments:
            emotion_counts[seg.emotion.value] = emotion_counts.get(seg.emotion.value, 0) + 1
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1])[:5]:
            pct = count / len(segments) * 100
            bar = "█" * int(pct / 5)
            print(f"  {emotion:12}: {bar:20} {pct:5.1f}%")

        # Distribution des types narratifs
        print("\n--- Types narratifs ---")
        narrative_counts = {}
        for seg in segments:
            narrative_counts[seg.narrative_type.value] = narrative_counts.get(
                seg.narrative_type.value, 0) + 1
        for narrative, count in sorted(narrative_counts.items(), key=lambda x: -x[1]):
            pct = count / len(segments) * 100
            print(f"  {narrative:15}: {count:3} ({pct:5.1f}%)")

    return segments


def convert_to_audio(
    segments,
    tts: UnifiedTTS,
    output_path: Path,
    config: HQPipelineConfig,
    verbose: bool = True
):
    """Convertit les segments en audio avec UnifiedTTS."""
    import numpy as np
    import wave

    print_section("SYNTHESE VOCALE")

    # Determiner le moteur a utiliser
    engine_name = config.tts_engine
    if engine_name == "auto":
        engine_type = TTSEngine.AUTO
        engine_display = "AUTO (Edge-TTS FR / Kokoro EN)"
    elif engine_name == "edge-tts":
        engine_type = TTSEngine.EDGE_TTS
        engine_display = "Edge-TTS (Microsoft)"
    else:
        engine_type = TTSEngine.KOKORO
        engine_display = "Kokoro (offline)"

    if verbose:
        print(f"  Moteur: {engine_display}")
        print(f"  Langue: {config.lang}")
        print(f"  Segments: {len(segments)}")
        print()

    start_time = time.time()
    total = len(segments)
    all_audio = []
    sample_rate = 24000

    for i, seg in enumerate(segments):
        if verbose:
            pct = (i + 1) / total * 100
            speaker = seg.speaker[:10] if seg.speaker else "?"
            emotion = seg.emotion.value[:8] if hasattr(seg, 'emotion') else "?"
            print(f"\r  [{pct:5.1f}%] {speaker:10} ({emotion:8}) - "
                  f"{seg.text[:30]}...", end="", flush=True)

        # Pause avant
        if seg.pause_before > 0:
            pause_samples = int(sample_rate * seg.pause_before)
            all_audio.append(np.zeros(pause_samples, dtype=np.float32))

        # Determiner la voix selon le moteur
        voice = seg.voice_id
        if engine_type == TTSEngine.EDGE_TTS or (engine_type == TTSEngine.AUTO and config.lang == "fr"):
            # Mapper les voix Kokoro vers Edge-TTS pour le francais
            if voice == "ff_siwis" or voice.startswith("ff_"):
                voice = "fr-FR-DeniseNeural"
            elif voice.startswith("fm_") or voice == "narrator_male":
                voice = "fr-FR-HenriNeural"

        # Synthetiser
        try:
            audio, sr = tts.synthesize(
                seg.text,
                lang=config.lang,
                voice=voice,
                speed=seg.final_speed if hasattr(seg, 'final_speed') else 1.0,
                engine=engine_type,
                preprocess=True
            )
            sample_rate = sr
            all_audio.append(audio)
        except Exception as e:
            if verbose:
                print(f"\n  ERREUR segment {i}: {e}")
            continue

        # Pause apres
        if seg.pause_after > 0:
            pause_samples = int(sample_rate * seg.pause_after)
            all_audio.append(np.zeros(pause_samples, dtype=np.float32))

    if verbose:
        print()  # Nouvelle ligne

    # Concatener tous les segments
    if all_audio:
        final_audio = np.concatenate(all_audio)

        # Ecrire le fichier WAV
        with wave.open(str(output_path), 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16 bits
            wf.setframerate(sample_rate)
            # Convertir float32 [-1,1] en int16
            audio_int16 = (final_audio * 32767).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())

    elapsed = time.time() - start_time
    if verbose and all_audio:
        duration = len(final_audio) / sample_rate
        print(f"\n  Duree de synthese: {elapsed:.1f}s")
        print(f"  Duree audio: {duration/60:.1f} min")
        print(f"  Fichier genere: {output_path}")

    return len(all_audio) > 0


def enhance_audio(input_path: Path, output_path: Path, config: HQPipelineConfig):
    """Applique le post-processing audio."""
    print_section("POST-PROCESSING AUDIO")

    enhancer_config = AudioEnhancerConfig(
        target_lufs=config.target_lufs,
        deess_enabled=config.enable_deessing,
        compression_enabled=config.enable_compression,
    )

    return enhance_audiobook(input_path, output_path, enhancer_config, verbose=True)


def main():
    parser = argparse.ArgumentParser(
        description="AudioReader HQ - Audiobooks haute qualite"
    )

    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Fichier d'entree (.md, .txt, .epub)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("output_hq"),
        help="Repertoire de sortie"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Fichier de configuration JSON"
    )
    parser.add_argument(
        "-v", "--voice",
        default="ff_siwis",
        help="Voix du narrateur"
    )
    parser.add_argument(
        "--lang",
        choices=["fr", "en"],
        default="fr",
        help="Langue du texte"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Analyser sans generer l'audio"
    )
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Desactiver le post-processing audio"
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
        "-s", "--speed",
        type=float,
        default=1.0,
        help="Vitesse de base (0.7-1.3)"
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "kokoro", "edge-tts"],
        default="auto",
        help="Moteur TTS (auto=Edge-TTS FR/Kokoro EN)"
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="Afficher les voix disponibles"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Mode silencieux"
    )

    args = parser.parse_args()

    # Afficher les voix
    if args.list_voices:
        print_banner()
        print_section("VOIX DISPONIBLES")

        tts = UnifiedTTS()
        engines = tts.get_available_engines()

        print(f"\n  Moteurs disponibles: {', '.join([e.value for e in engines])}\n")

        # Voix françaises
        print("  === FRANÇAIS ===")
        for voice in tts.get_voices(lang="fr"):
            print(f"    [{voice.engine.value:8}] {voice.id:25} - {voice.name} ({voice.gender})")

        # Voix anglaises
        print("\n  === ANGLAIS ===")
        for voice in tts.get_voices(lang="en"):
            print(f"    [{voice.engine.value:8}] {voice.id:25} - {voice.name} ({voice.gender})")

        print("\n  Usage: --engine auto|kokoro|edge-tts --voice <voice_id>")
        return

    # Verifier le fichier d'entree
    if not args.input:
        parser.print_help()
        return

    if not args.input.exists():
        print(f"ERREUR: Fichier non trouve: {args.input}")
        sys.exit(1)

    if not args.quiet:
        print_banner()

    # Charger la configuration
    if args.config and args.config.exists():
        config = HQPipelineConfig.load(args.config)
        if not args.quiet:
            print(f"Configuration chargee: {args.config}")
    else:
        config = HQPipelineConfig(
            lang=args.lang,
            tts_engine=args.engine,
            narrator_voice=args.voice,
            enable_emotion_analysis=not args.no_emotions,
            enable_character_detection=not args.no_characters,
            enable_audio_enhancement=not args.no_enhance,
        )

    # Lire le fichier
    print_section("LECTURE DU FICHIER")
    print(f"  Source: {args.input}")

    if args.input.suffix.lower() == '.epub':
        epub_parser = EPUBParser()
        chapters = epub_parser.parse_file(args.input)
        text = "\n\n".join(ch.content for ch in chapters)
        print(f"  Chapitres: {len(chapters)}")
    else:
        with open(args.input, 'r', encoding='utf-8') as f:
            text = f.read()

    print(f"  Caracteres: {len(text):,}")
    print(f"  Mots: {len(text.split()):,}")

    # Creer le pipeline
    pipeline = create_hq_pipeline(
        lang=config.lang,
        narrator_voice=config.narrator_voice,
        enable_emotion_analysis=config.enable_emotion_analysis,
        enable_character_detection=config.enable_character_detection,
        enable_narrative_context=config.enable_narrative_context,
    )

    # Analyser
    segments = analyze_text(pipeline, text, verbose=not args.quiet)

    if args.analyze_only:
        print("\n[Mode analyse seule - pas de generation audio]")
        return

    # Initialiser UnifiedTTS
    tts_config = TTSConfig(
        speed=args.speed,
        use_french_preprocessor=(config.lang == "fr")
    )
    tts = UnifiedTTS(tts_config)

    # Verifier disponibilite
    available = tts.get_available_engines()
    if not available:
        print("\nERREUR: Aucun moteur TTS disponible.")
        print("Installez au moins un des moteurs:")
        print("  - Kokoro: pip install kokoro-onnx soundfile")
        print("  - Edge-TTS: pip install edge-tts pydub")
        sys.exit(1)

    if not args.quiet:
        engines_str = ", ".join([e.value for e in available])
        print(f"\n  Moteurs disponibles: {engines_str}")

    # Creer le repertoire de sortie
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generer l'audio
    raw_audio = output_dir / f"{args.input.stem}_raw.wav"
    success = convert_to_audio(
        segments,
        tts,
        raw_audio,
        config,
        verbose=not args.quiet
    )

    if not success:
        print("\nERREUR: Echec de la synthese vocale")
        sys.exit(1)

    # Post-processing
    if config.enable_audio_enhancement:
        final_audio = output_dir / f"{args.input.stem}_hq.mp3"
        success = enhance_audio(raw_audio, final_audio, config)

        if success:
            # Supprimer le fichier brut
            # raw_audio.unlink()  # Commenter pour garder le WAV
            print(f"\n  Audio final: {final_audio}")
        else:
            print("\n  Post-processing non disponible (ffmpeg manquant)")
            print(f"  Audio brut: {raw_audio}")
    else:
        print(f"\n  Audio: {raw_audio}")

    # Resume
    print_section("RESUME")
    print(f"  Moteur TTS: {config.tts_engine}")
    print(f"  Segments traites: {len(segments)}")
    print(f"  Personnages: {len(pipeline.get_voice_assignments())}")
    analysis = pipeline.get_chapter_analysis()
    print(f"  Ton: {analysis['tone']}")
    print(f"\n  Termine!")


if __name__ == "__main__":
    main()
