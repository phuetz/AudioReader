#!/usr/bin/env python3
"""
Test de comparaison : MMS (voix unique) vs Kokoro (multi-voix)
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from tts_engine import create_tts_engine
from character_detector import process_text_with_characters, CharacterDetector, VoiceAssigner
from tts_kokoro_engine import KokoroEngine

# Charger le chapitre 7
CHAPTER_FILE = Path("books/chapitre_test.md")
OUTPUT_DIR = Path("output/comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_chapter():
    """Lit le contenu du chapitre."""
    with open(CHAPTER_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    # Retirer le titre markdown
    lines = content.strip().split('\n')
    if lines[0].startswith('#'):
        lines = lines[1:]
    return '\n'.join(lines).strip()


def test_mms_single_voice():
    """Test 1: MMS-TTS avec une seule voix."""
    print("\n" + "="*60)
    print("TEST 1: MMS-TTS (voix unique française)")
    print("="*60)

    text = read_chapter()
    print(f"Texte: {len(text)} caractères")

    engine = create_tts_engine(language="fr", engine_type="mms")
    print(f"Moteur: {engine.get_info()}")

    output_file = OUTPUT_DIR / "chapitre7_mms_single.wav"

    print("\nGénération en cours...")
    success = engine.synthesize(text, output_file)

    if success and output_file.exists():
        size_mb = output_file.stat().st_size / (1024*1024)
        print(f"Fichier: {output_file} ({size_mb:.2f} MB)")
    else:
        print("ERREUR: Génération échouée")

    return success


def test_kokoro_multi_voice():
    """Test 2: Kokoro avec plusieurs voix par personnage."""
    print("\n" + "="*60)
    print("TEST 2: Kokoro (multi-voix par personnage)")
    print("="*60)

    text = read_chapter()

    # Détecter les personnages et assigner les voix
    print("\nAnalyse des personnages...")
    segments, voice_assignments = process_text_with_characters(
        text,
        narrator_voice="ff_siwis",  # Narrateur: femme française
        lang="fr",
        voice_mapping={
            "Marie": "ff_siwis",      # Marie: femme française
            "Pierre": "am_adam",       # Pierre: homme américain
            "Dubois": "am_michael",    # Commissaire: homme différent
            "NARRATOR": "ff_siwis",
        }
    )

    print(f"\nPersonnages détectés:")
    for char, voice in voice_assignments.items():
        print(f"  {char}: {voice}")

    print(f"\nSegments: {len(segments)}")

    # Initialiser Kokoro
    print("\nChargement de Kokoro...")

    # Générer chaque segment avec la voix appropriée
    all_audio = []
    sample_rate = 24000

    try:
        from kokoro import KPipeline
        pipeline = KPipeline(lang_code='a')  # 'a' = American English voices work

        for i, segment in enumerate(segments):
            voice = segment.voice_id or "ff_siwis"
            text_seg = segment.text.strip()

            if not text_seg:
                continue

            print(f"  [{i+1}/{len(segments)}] {segment.speaker[:15]:15} ({voice}): {text_seg[:40]}...")

            # Générer l'audio
            try:
                generator = pipeline(text_seg, voice=voice, speed=1.0)
                for _, _, audio in generator:
                    all_audio.append(audio)

                # Pause entre segments
                pause_samples = int(0.3 * sample_rate)
                all_audio.append(np.zeros(pause_samples, dtype=np.float32))

            except Exception as e:
                print(f"    ERREUR segment: {e}")
                continue

        # Concaténer et sauvegarder
        if all_audio:
            import soundfile as sf
            final_audio = np.concatenate(all_audio)

            # Normaliser
            max_val = np.max(np.abs(final_audio))
            if max_val > 0:
                final_audio = (final_audio / max_val * 0.9).astype(np.float32)

            output_file = OUTPUT_DIR / "chapitre7_kokoro_multi.wav"
            sf.write(str(output_file), final_audio, sample_rate)

            size_mb = output_file.stat().st_size / (1024*1024)
            duration = len(final_audio) / sample_rate
            print(f"\nFichier: {output_file}")
            print(f"  Taille: {size_mb:.2f} MB")
            print(f"  Durée: {duration:.1f}s")

            return True

    except Exception as e:
        print(f"ERREUR Kokoro: {e}")
        import traceback
        traceback.print_exc()

    return False


if __name__ == "__main__":
    print("="*60)
    print("COMPARAISON TTS: MMS vs Kokoro Multi-voix")
    print("="*60)
    print(f"Chapitre: {CHAPTER_FILE}")
    print(f"Sortie: {OUTPUT_DIR}")

    # Test 1: MMS
    success_mms = test_mms_single_voice()

    # Test 2: Kokoro multi-voix
    success_kokoro = test_kokoro_multi_voice()

    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)
    print(f"MMS (voix unique):     {'OK' if success_mms else 'ÉCHEC'}")
    print(f"Kokoro (multi-voix):   {'OK' if success_kokoro else 'ÉCHEC'}")
    print(f"\nFichiers dans: {OUTPUT_DIR}")
    print("  - chapitre7_mms_single.wav")
    print("  - chapitre7_kokoro_multi.wav")
