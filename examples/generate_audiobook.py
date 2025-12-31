#!/usr/bin/env python3
"""
Generateur d'audiobook complet pour Les Conquerants du Pognon.
"""
import sys
import time
import wave
import struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.hq_pipeline import HQPipeline, HQPipelineConfig, create_hq_pipeline
from src.tts_unified import UnifiedTTS, TTSEngine, TTSConfig
import numpy as np


def generate_chapter(chapter_path: Path, output_path: Path, tts: UnifiedTTS, config: HQPipelineConfig) -> bool:
    """Genere l'audio pour un chapitre."""
    # Lire le fichier
    with open(chapter_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Creer le pipeline
    pipeline = create_hq_pipeline(
        lang=config.lang,
        narrator_voice=config.narrator_voice,
        enable_emotion_analysis=True,
        enable_character_detection=True,
    )

    # Analyser
    segments = pipeline.process_chapter(text)

    # Determiner le moteur
    if config.tts_engine == "edge-tts":
        engine_type = TTSEngine.EDGE_TTS
    elif config.tts_engine == "kokoro":
        engine_type = TTSEngine.KOKORO
    else:
        engine_type = TTSEngine.AUTO

    # Generer l'audio
    all_audio = []
    sample_rate = 24000

    for seg in segments:
        # Pause avant
        if seg.pause_before > 0:
            pause_samples = int(sample_rate * seg.pause_before)
            all_audio.append(np.zeros(pause_samples, dtype=np.float32))

        # Voix
        voice = seg.voice_id
        if engine_type == TTSEngine.EDGE_TTS or (engine_type == TTSEngine.AUTO and config.lang == "fr"):
            if voice == "ff_siwis" or voice.startswith("ff_"):
                voice = "fr-FR-DeniseNeural"
            elif voice.startswith("fm_") or voice.startswith("af_") or voice.startswith("am_"):
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
            print(f"    Erreur: {e}")
            continue

        # Pause apres
        if seg.pause_after > 0:
            pause_samples = int(sample_rate * seg.pause_after)
            all_audio.append(np.zeros(pause_samples, dtype=np.float32))

    # Concatener et sauvegarder
    if all_audio:
        final_audio = np.concatenate(all_audio)

        with wave.open(str(output_path), 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            audio_int16 = (final_audio * 32767).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())

        return True

    return False


def concatenate_wav_files(wav_files: list, output_path: Path):
    """Concatene plusieurs fichiers WAV en un seul."""
    print(f"\nConcatenation de {len(wav_files)} fichiers...")

    all_audio = []
    sample_rate = 24000

    # Pause entre chapitres (3 secondes)
    chapter_pause = np.zeros(int(sample_rate * 3), dtype=np.float32)

    for i, wav_path in enumerate(wav_files):
        print(f"  [{i+1}/{len(wav_files)}] {wav_path.name}")

        with wave.open(str(wav_path), 'r') as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            sample_rate = wf.getframerate()

        all_audio.append(audio)

        # Ajouter pause entre chapitres (sauf dernier)
        if i < len(wav_files) - 1:
            all_audio.append(chapter_pause)

    # Concatener
    final_audio = np.concatenate(all_audio)

    # Sauvegarder
    with wave.open(str(output_path), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        audio_int16 = (final_audio * 32767).astype(np.int16)
        wf.writeframes(audio_int16.tobytes())

    duration_min = len(final_audio) / sample_rate / 60
    file_size_mb = output_path.stat().st_size / 1024 / 1024

    print(f"\nAudiobook genere: {output_path}")
    print(f"  Duree: {duration_min:.1f} min ({duration_min/60:.1f} h)")
    print(f"  Taille: {file_size_mb:.1f} Mo")


def main():
    # Configuration
    BOOK_DIR = Path("/home/patrice/claude/livre/Les_Conquerants_du_Pognon/tome-1-or-noir-v2/chapitres")
    OUTPUT_DIR = Path("/home/patrice/claude/AudioReader/audiobook_tome1")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Trouver tous les chapitres
    chapters = sorted(BOOK_DIR.glob("chapitre-*.md"))
    print(f"Chapitres trouves: {len(chapters)}")

    # Configuration TTS
    tts_config = TTSConfig(
        speed=1.0,
        use_french_preprocessor=True
    )
    tts = UnifiedTTS(tts_config)

    pipeline_config = HQPipelineConfig(
        lang="fr",
        tts_engine="edge-tts",
        narrator_voice="ff_siwis",
    )

    print(f"\nMoteur: Edge-TTS (Microsoft)")
    print(f"Estimation: ~50-60 minutes pour ~14 heures d'audio")
    print(f"Sortie: {OUTPUT_DIR}")
    print()

    start_time = time.time()
    generated_files = []

    for i, chapter_path in enumerate(chapters):
        chapter_name = chapter_path.stem
        output_path = OUTPUT_DIR / f"{chapter_name}.wav"

        print(f"[{i+1}/{len(chapters)}] {chapter_name}")

        if output_path.exists():
            print(f"  -> Deja genere, skip")
            generated_files.append(output_path)
            continue

        chapter_start = time.time()

        try:
            success = generate_chapter(chapter_path, output_path, tts, pipeline_config)

            if success:
                chapter_time = time.time() - chapter_start

                with wave.open(str(output_path), 'r') as wf:
                    duration = wf.getnframes() / wf.getframerate()

                print(f"  -> {duration/60:.1f} min en {chapter_time:.0f}s")
                generated_files.append(output_path)
            else:
                print(f"  -> ECHEC")
        except Exception as e:
            print(f"  -> ERREUR: {e}")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Generation terminee en {total_time/60:.1f} minutes")

    # Concatener tous les chapitres
    if len(generated_files) == len(chapters):
        final_output = OUTPUT_DIR / "tome1_or_noir_complet.wav"
        concatenate_wav_files(generated_files, final_output)
    else:
        print(f"\nATTENTION: Seulement {len(generated_files)}/{len(chapters)} chapitres generes")


if __name__ == "__main__":
    main()
