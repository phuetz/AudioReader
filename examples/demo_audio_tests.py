#!/usr/bin/env python3
"""
Tests audio reels - Generation de fichiers audio pour ecoute.

Ce script genere de vrais fichiers audio pour tester:
- Voix de base (male/female, fr/en)
- Emotions et tags audio
- Voice morphing (pitch, formants, etc.)
- Conversations multi-locuteurs
- Controle de prononciation

Usage:
    python demo_audio_tests.py [--all] [--voices] [--emotions] [--morphing] [--conversation] [--pronunciation]
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Ajouter le repertoire src au path
sys.path.insert(0, str(Path(__file__).parent))

# Creer le dossier de sortie
OUTPUT_DIR = Path(__file__).parent / "demo_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Chemins des fichiers modele
MODEL_PATH = Path(__file__).parent / "kokoro-v1.0.onnx"
VOICES_PATH = Path(__file__).parent / "voices-v1.0.bin"


def check_dependencies():
    """Verifie les dependances requises."""
    missing = []

    try:
        import kokoro_onnx
    except ImportError:
        missing.append("kokoro-onnx")

    try:
        import soundfile
    except ImportError:
        missing.append("soundfile")

    if missing:
        print(f"Dependances manquantes: {', '.join(missing)}")
        print("Installez-les avec: pip install " + " ".join(missing))
        return False

    if not MODEL_PATH.exists():
        print(f"Fichier modele manquant: {MODEL_PATH}")
        return False

    if not VOICES_PATH.exists():
        print(f"Fichier voix manquant: {VOICES_PATH}")
        return False

    return True


def save_audio(audio: np.ndarray, filename: str, sample_rate: int = 24000):
    """Sauvegarde l'audio dans un fichier."""
    import soundfile as sf

    filepath = OUTPUT_DIR / filename
    sf.write(str(filepath), audio, sample_rate)
    print(f"  -> Sauvegarde: {filepath}")
    return filepath


def create_tts():
    """Cree une instance du moteur TTS."""
    from kokoro_onnx import Kokoro
    return Kokoro(str(MODEL_PATH), str(VOICES_PATH))


def synthesize_text(kokoro, text: str, voice: str = "ff_siwis", speed: float = 1.0, lang: str = "fr-fr"):
    """Synthetise du texte en audio."""
    samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang=lang)
    return samples, sample_rate


def test_basic_voices():
    """Test des voix de base."""
    print("\n" + "="*60)
    print("TEST 1: Voix de base")
    print("="*60)

    kokoro = create_tts()

    # Voix francaises
    voices_fr = [
        ("ff_siwis", "fr-fr", "Bonjour, je suis une voix feminine francaise. Comment allez-vous aujourd'hui?"),
    ]

    # Voix anglaises
    voices_en = [
        ("af_bella", "en-us", "Hello, I am a female American voice. How are you today?"),
        ("af_heart", "en-us", "Hello, I am Heart, a warm and friendly voice."),
        ("am_adam", "en-us", "Hello, I am a male American voice. Nice to meet you."),
        ("bf_emma", "en-gb", "Hello, I am Emma, a British female voice."),
        ("bm_george", "en-gb", "Hello, I am George, a British male voice."),
    ]

    print("\nVoix francaise:")
    for voice_id, lang, text in voices_fr:
        print(f"  Voix: {voice_id}")
        try:
            audio, sr = synthesize_text(kokoro, text, voice=voice_id, lang=lang)
            save_audio(audio, f"voice_{voice_id}.wav", sr)
        except Exception as e:
            print(f"    Erreur: {e}")

    print("\nVoix anglaises:")
    for voice_id, lang, text in voices_en:
        print(f"  Voix: {voice_id}")
        try:
            audio, sr = synthesize_text(kokoro, text, voice=voice_id, lang=lang)
            save_audio(audio, f"voice_{voice_id}.wav", sr)
        except Exception as e:
            print(f"    Erreur: {e}")


def test_emotions():
    """Test des emotions et tags audio."""
    print("\n" + "="*60)
    print("TEST 2: Emotions et Tags Audio")
    print("="*60)

    from src.audio_tags import AudioTagProcessor, process_text_with_audio_tags

    kokoro = create_tts()
    processor = AudioTagProcessor()

    # Textes avec differentes emotions (tags en anglais)
    emotion_texts = [
        ("neutral", "Voici une phrase normale, sans emotion particuliere."),
        ("cheerful", "[cheerful] Oh quelle merveilleuse journee! Je suis tellement content!"),
        ("sad", "[sad] Helas, tout est fini maintenant. C'est vraiment dommage."),
        ("angry", "[angry] C'est inacceptable! Je ne tolererai pas ca!"),
        ("excited", "[excited] Incroyable! C'est la meilleure nouvelle de l'annee!"),
        ("whispers", "[whispers] C'est un secret, ne le dis a personne."),
        ("dramatic", "[dramatic] Et soudain, le silence. Tout s'arreta."),
        ("sarcastic", "[sarcastic] Oh bien sur, c'est vraiment une excellente idee."),
        ("mysterious", "[mysterious] Personne ne sait ce qui s'est passe cette nuit la."),
    ]

    print("\nGeneration des emotions:")
    for emotion_name, text in emotion_texts:
        print(f"\n  Emotion: {emotion_name}")
        print(f"  Texte: {text[:50]}...")

        try:
            # Traiter les tags
            result = process_text_with_audio_tags(text)
            cleaned_text = result.text
            prosody = result.prosody

            print(f"  Prosodie: speed={prosody['speed']:.2f}, pitch={prosody['pitch']:.2f}, volume={prosody['volume']:.2f}")

            # Generer l'audio avec la vitesse ajustee
            audio, sr = synthesize_text(
                kokoro,
                cleaned_text,
                voice="ff_siwis",
                speed=prosody["speed"],
                lang="fr-fr"
            )

            # Ajuster le volume
            audio = audio * prosody["volume"]
            audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
            save_audio(audio, f"emotion_{emotion_name}.wav", sr)

        except Exception as e:
            print(f"    Erreur: {e}")
            import traceback
            traceback.print_exc()


def test_voice_morphing():
    """Test du voice morphing."""
    print("\n" + "="*60)
    print("TEST 3: Voice Morphing")
    print("="*60)

    from src.voice_morphing import VoiceMorpher, VoicePresets, VoiceMorphSettings

    kokoro = create_tts()
    morpher = VoiceMorpher(sample_rate=24000)

    # Texte de base
    base_text = "Bonjour, ceci est un test de modification de voix. Ecoutez la difference."

    print("\nGeneration de l'audio de base...")
    try:
        base_audio, sr = synthesize_text(kokoro, base_text, voice="ff_siwis", lang="fr-fr")
        save_audio(base_audio, "morph_original.wav", sr)
    except Exception as e:
        print(f"  Erreur: {e}")
        return

    # Tester les presets
    presets_to_test = [
        "more_masculine",
        "more_feminine",
        "younger",
        "older",
        "whisper",
        "rough",
        "expressive",
    ]

    print("\nApplication des presets de morphing:")
    for preset_name in presets_to_test:
        print(f"\n  Preset: {preset_name}")
        settings = VoicePresets.get(preset_name)

        if settings:
            try:
                morphed = morpher.morph(base_audio.copy(), settings)
                save_audio(morphed.astype(np.float32), f"morph_{preset_name}.wav", sr)
            except Exception as e:
                print(f"    Erreur: {e}")

    # Test personnalise
    print("\n  Preset personnalise: voix de robot")
    custom_settings = VoiceMorphSettings(
        pitch_shift=-2.0,
        formant_shift=0.9,
        stability=1.0,
        roughness=0.1,
    )
    try:
        morphed = morpher.morph(base_audio.copy(), custom_settings)
        save_audio(morphed.astype(np.float32), "morph_custom_robot.wav", sr)
    except Exception as e:
        print(f"    Erreur: {e}")


def test_conversation():
    """Test de conversation multi-locuteurs."""
    print("\n" + "="*60)
    print("TEST 4: Conversation Multi-Locuteurs")
    print("="*60)

    from src.conversation_generator import ConversationGenerator

    kokoro = create_tts()
    generator = ConversationGenerator()

    # Script de conversation
    script = """
    MARIE: Bonjour Jean! Comment vas-tu aujourd'hui?
    JEAN: Tres bien merci! Et toi, Marie?
    MARIE: Super! J'ai une excellente nouvelle a t'annoncer.
    JEAN: Ah oui? Raconte-moi!
    MARIE: J'ai obtenu le poste dont je revais!
    JEAN: Felicitations! C'est fantastique!
    """

    # Configuration des speakers
    speaker_config = {
        "MARIE": {"gender": "female", "voice": "ff_siwis"},
        "JEAN": {"gender": "male", "voice": "am_adam"},
    }

    print("\nParsing du script...")
    conversation = generator.parse_script(script, speaker_config)

    print(f"  Speakers: {list(conversation.speakers.keys())}")
    print(f"  Lignes: {len(conversation.lines)}")

    # Generer chaque ligne
    audio_segments = []
    sample_rate = 24000
    pause_samples = int(sample_rate * 0.5)  # 500ms de pause entre les lignes
    silence = np.zeros(pause_samples, dtype=np.float32)

    # Mapping voix -> langue
    voice_lang = {
        "ff_siwis": "fr-fr",
        "am_adam": "en-us",
    }

    print("\nGeneration des lignes de dialogue:")
    for i, line in enumerate(conversation.lines):
        speaker = line.speaker
        text = line.text
        voice_id = speaker.voice_id
        lang = voice_lang.get(voice_id, "en-us")

        print(f"  {speaker.name}: {text[:40]}...")

        try:
            audio, sr = synthesize_text(kokoro, text, voice=voice_id, speed=speaker.speed, lang=lang)
            # Sauvegarder la ligne individuelle
            save_audio(audio, f"conv_line_{i+1}_{speaker.name}.wav", sr)
            audio_segments.append(audio)
            audio_segments.append(silence)
        except Exception as e:
            print(f"    Erreur: {e}")

    # Concatener pour la conversation complete
    if audio_segments:
        print("\nConcatenation de la conversation complete...")
        full_audio = np.concatenate(audio_segments)
        save_audio(full_audio, "conversation_complete.wav", sample_rate)


def test_pronunciation():
    """Test du controle de prononciation."""
    print("\n" + "="*60)
    print("TEST 5: Controle de Prononciation")
    print("="*60)

    from src.emotion_control import create_pronunciation_config

    kokoro = create_tts()
    pronunciation = create_pronunciation_config("fr")

    # Textes avec cas speciaux
    texts = [
        ("abbreviations", "M. Dupont et Mme Martin ont rendez-vous. Etc."),
        ("nombres", "C'est le 1er janvier, la 2eme fois."),
        ("anglicismes", "J'ai un meeting pour discuter du startup."),
        ("tech", "L'API utilise JavaScript et Python."),
    ]

    print("\nTest de prononciation:")
    for name, text in texts:
        print(f"\n  Test: {name}")
        print(f"  Original: {text}")

        # Appliquer les corrections
        corrected = pronunciation.process(text)
        print(f"  Corrige: {corrected}")

        try:
            audio, sr = synthesize_text(kokoro, corrected, voice="ff_siwis", lang="fr-fr")
            save_audio(audio, f"pronunciation_{name}.wav", sr)
        except Exception as e:
            print(f"    Erreur: {e}")


def test_speed_variations():
    """Test des variations de vitesse."""
    print("\n" + "="*60)
    print("TEST 6: Variations de Vitesse")
    print("="*60)

    kokoro = create_tts()

    text = "Ceci est un test de variation de vitesse."
    speeds = [0.7, 0.85, 1.0, 1.15, 1.3]

    print("\nGeneration a differentes vitesses:")
    for speed in speeds:
        print(f"  Vitesse: {speed}")
        try:
            audio, sr = synthesize_text(kokoro, text, voice="ff_siwis", speed=speed, lang="fr-fr")
            save_audio(audio, f"speed_{speed:.2f}.wav", sr)
        except Exception as e:
            print(f"    Erreur: {e}")


def test_long_text():
    """Test avec un texte plus long."""
    print("\n" + "="*60)
    print("TEST 7: Texte Long (Paragraphe)")
    print("="*60)

    kokoro = create_tts()

    long_text = """Il etait une fois, dans un petit village au coeur de la France,
une jeune fille nommee Amelie. Elle revait de voyager et de decouvrir le monde.
Un jour, elle trouva une vieille carte au grenier de sa maison.
Cette carte indiquait l'emplacement d'un tresor cache dans les montagnes lointaines."""

    # Nettoyer le texte
    long_text = " ".join(long_text.split())

    print("\nTexte:")
    print(f"  {long_text[:80]}...")
    print(f"  (Longueur: {len(long_text)} caracteres)")

    try:
        audio, sr = synthesize_text(kokoro, long_text, voice="ff_siwis", speed=1.0, lang="fr-fr")
        duration = len(audio) / sr
        print(f"  Duree audio: {duration:.1f}s")
        save_audio(audio, "long_text_paragraph.wav", sr)
    except Exception as e:
        print(f"  Erreur: {e}")


def test_english_text():
    """Test avec du texte anglais."""
    print("\n" + "="*60)
    print("TEST 8: Texte Anglais")
    print("="*60)

    kokoro = create_tts()

    texts = [
        ("af_bella", "en-us", "Hello everyone! Welcome to this audio demonstration. I hope you enjoy listening to these samples."),
        ("am_adam", "en-us", "Good morning! This is a test of the male American voice. The weather is beautiful today."),
        ("bf_emma", "en-gb", "Good afternoon! This is Emma speaking with a British accent. Lovely to meet you."),
    ]

    print("\nGeneration de texte anglais:")
    for voice_id, lang, text in texts:
        print(f"\n  Voix: {voice_id}")
        print(f"  Texte: {text[:50]}...")
        try:
            audio, sr = synthesize_text(kokoro, text, voice=voice_id, lang=lang)
            save_audio(audio, f"english_{voice_id}.wav", sr)
        except Exception as e:
            print(f"    Erreur: {e}")


def main():
    parser = argparse.ArgumentParser(description="Tests audio reels pour AudioReader")
    parser.add_argument("--all", action="store_true", help="Lancer tous les tests")
    parser.add_argument("--voices", action="store_true", help="Tester les voix de base")
    parser.add_argument("--emotions", action="store_true", help="Tester les emotions")
    parser.add_argument("--morphing", action="store_true", help="Tester le voice morphing")
    parser.add_argument("--conversation", action="store_true", help="Tester les conversations")
    parser.add_argument("--pronunciation", action="store_true", help="Tester la prononciation")
    parser.add_argument("--speed", action="store_true", help="Tester les vitesses")
    parser.add_argument("--long", action="store_true", help="Tester un texte long")
    parser.add_argument("--english", action="store_true", help="Tester l'anglais")

    args = parser.parse_args()

    # Si aucun argument, lancer tout
    if not any([args.all, args.voices, args.emotions, args.morphing,
                args.conversation, args.pronunciation, args.speed, args.long, args.english]):
        args.all = True

    print("="*60)
    print("TESTS AUDIO REELS - AudioReader")
    print("="*60)
    print(f"Dossier de sortie: {OUTPUT_DIR}")

    if not check_dependencies():
        return 1

    try:
        if args.all or args.voices:
            test_basic_voices()

        if args.all or args.emotions:
            test_emotions()

        if args.all or args.morphing:
            test_voice_morphing()

        if args.all or args.conversation:
            test_conversation()

        if args.all or args.pronunciation:
            test_pronunciation()

        if args.all or args.speed:
            test_speed_variations()

        if args.all or args.long:
            test_long_text()

        if args.all or args.english:
            test_english_text()

        print("\n" + "="*60)
        print("TESTS TERMINES")
        print("="*60)
        print(f"\nFichiers audio generes dans: {OUTPUT_DIR}")
        print("Utilisez un lecteur audio pour les ecouter.")

        # Lister les fichiers generes
        wav_files = list(OUTPUT_DIR.glob("*.wav"))
        if wav_files:
            print(f"\nFichiers generes ({len(wav_files)}):")
            for f in sorted(wav_files):
                size_kb = f.stat().st_size / 1024
                print(f"  - {f.name} ({size_kb:.1f} KB)")

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrompu par l'utilisateur.")
        return 1
    except Exception as e:
        print(f"\nErreur: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
