#!/usr/bin/env python3
"""
Démonstration du système TTS unifié.

Montre:
- Sélection automatique Edge-TTS (français) / Kokoro (anglais)
- Toutes les voix disponibles
- Préprocesseur français intégré
- Comparaison des moteurs
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.tts_unified import UnifiedTTS, TTSEngine, TTSConfig, create_tts

OUTPUT_DIR = Path("demo_output/unified_demo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def demo_auto_selection():
    """Démo de la sélection automatique par langue."""
    print("\n" + "="*60)
    print("1. SÉLECTION AUTOMATIQUE PAR LANGUE")
    print("="*60)

    tts = UnifiedTTS()

    tests = [
        # Français -> Edge-TTS (meilleure qualité)
        ("fr", "Bonjour! Je suis la voix française automatique. Comment allez-vous?"),
        ("fr", "C'était une belle journée ensoleillée. Le garçon a reçu sa leçon."),

        # Anglais -> Kokoro (rapide, offline)
        ("en", "Hello! I am the automatic English voice. How are you today?"),
        ("en", "The weather is beautiful. Let's go for a walk in the park."),
    ]

    import soundfile as sf

    for lang, text in tests:
        engine = "Edge-TTS" if lang == "fr" else "Kokoro"
        print(f"\n  [{lang.upper()}] ({engine})")
        print(f"  Texte: {text[:50]}...")

        try:
            audio, sr = tts.synthesize(text, lang=lang)
            filename = f"auto_{lang}_{hash(text) % 10000}.wav"
            sf.write(str(OUTPUT_DIR / filename), audio, sr)
            print(f"  -> {filename}")
        except Exception as e:
            print(f"  Erreur: {e}")


def demo_all_french_voices():
    """Démo de toutes les voix françaises."""
    print("\n" + "="*60)
    print("2. TOUTES LES VOIX FRANÇAISES")
    print("="*60)

    tts = UnifiedTTS()
    import soundfile as sf

    text = "Bonjour! Je suis une voix française. Comment trouvez-vous ma prononciation?"

    french_voices = tts.get_voices(lang="fr")

    print(f"\n  {len(french_voices)} voix françaises disponibles:\n")

    for voice in french_voices:
        print(f"  {voice.id} ({voice.engine.value})")
        print(f"    {voice.name} - {voice.gender} - {voice.description}")

        try:
            audio, sr = tts.synthesize(
                text,
                lang="fr",
                voice=voice.id,
                engine=voice.engine
            )
            filename = f"voice_fr_{voice.id.replace('-', '_')}.wav"
            sf.write(str(OUTPUT_DIR / filename), audio, sr)
            print(f"    -> {filename}")
        except Exception as e:
            print(f"    Erreur: {e}")

        print()


def demo_preprocessor():
    """Démo du préprocesseur français."""
    print("\n" + "="*60)
    print("3. PRÉPROCESSEUR FRANÇAIS")
    print("="*60)

    # Sans préprocesseur
    config_no_preproc = TTSConfig(use_french_preprocessor=False)
    tts_no_preproc = UnifiedTTS(config_no_preproc)

    # Avec préprocesseur
    tts_with_preproc = UnifiedTTS()

    import soundfile as sf

    tests = [
        ("abrev", "M. Dupont et Mme Martin ont rendez-vous à 15h."),
        ("nombres", "Il y avait 1500 personnes. C'est le 1er janvier."),
        ("anglicismes", "J'ai un meeting important, envoyez-moi un email."),
    ]

    print("\n  Comparaison avec/sans préprocesseur:\n")

    for name, text in tests:
        print(f"  [{name}] {text}")

        # Sans préprocesseur
        try:
            audio, sr = tts_no_preproc.synthesize(text, lang="fr", preprocess=False)
            sf.write(str(OUTPUT_DIR / f"preproc_{name}_without.wav"), audio, sr)
            print(f"    Sans préproc -> preproc_{name}_without.wav")
        except Exception as e:
            print(f"    Erreur sans préproc: {e}")

        # Avec préprocesseur
        try:
            audio, sr = tts_with_preproc.synthesize(text, lang="fr", preprocess=True)
            sf.write(str(OUTPUT_DIR / f"preproc_{name}_with.wav"), audio, sr)
            print(f"    Avec préproc -> preproc_{name}_with.wav")
        except Exception as e:
            print(f"    Erreur avec préproc: {e}")

        print()


def demo_compare_engines():
    """Comparaison directe des moteurs pour le français."""
    print("\n" + "="*60)
    print("4. COMPARAISON KOKORO vs EDGE-TTS (Français)")
    print("="*60)

    tts = UnifiedTTS()
    import soundfile as sf

    text = "C'était une belle journée ensoleillée. François était très heureux d'être là."

    print(f"\n  Texte: {text}\n")

    # Kokoro
    print("  [KOKORO] ff_siwis")
    try:
        audio, sr = tts.synthesize(text, lang="fr", engine=TTSEngine.KOKORO, voice="ff_siwis")
        sf.write(str(OUTPUT_DIR / "compare_kokoro_fr.wav"), audio, sr)
        print(f"    -> compare_kokoro_fr.wav")
    except Exception as e:
        print(f"    Erreur: {e}")

    # Edge-TTS femme
    print("  [EDGE-TTS] fr-FR-DeniseNeural (femme)")
    try:
        audio, sr = tts.synthesize(text, lang="fr", engine=TTSEngine.EDGE_TTS, voice="fr-FR-DeniseNeural")
        sf.write(str(OUTPUT_DIR / "compare_edge_denise.wav"), audio, sr)
        print(f"    -> compare_edge_denise.wav")
    except Exception as e:
        print(f"    Erreur: {e}")

    # Edge-TTS homme
    print("  [EDGE-TTS] fr-FR-HenriNeural (homme)")
    try:
        audio, sr = tts.synthesize(text, lang="fr", engine=TTSEngine.EDGE_TTS, voice="fr-FR-HenriNeural")
        sf.write(str(OUTPUT_DIR / "compare_edge_henri.wav"), audio, sr)
        print(f"    -> compare_edge_henri.wav")
    except Exception as e:
        print(f"    Erreur: {e}")


def demo_bilingual_text():
    """Démo avec texte bilingue."""
    print("\n" + "="*60)
    print("5. TEXTE BILINGUE (Narration)")
    print("="*60)

    tts = UnifiedTTS()
    import soundfile as sf

    # Simuler un audiobook avec passages en différentes langues
    segments = [
        ("fr", "Chapitre un. Le voyage commence."),
        ("fr", "Marie regarda par la fenêtre du train. Elle était excitée."),
        ("en", "\"Welcome to London!\" said the conductor."),
        ("fr", "Marie sourit. Son aventure anglaise commençait enfin."),
        ("en", "\"Thank you,\" she replied with a French accent."),
        ("fr", "Le train s'arrêta à la gare de King's Cross."),
    ]

    print("\n  Génération d'une narration bilingue:\n")

    all_audio = []
    pause = np.zeros(int(24000 * 0.5), dtype=np.float32)  # 0.5s pause

    for i, (lang, text) in enumerate(segments):
        engine = "Edge-TTS" if lang == "fr" else "Kokoro"
        print(f"  [{i+1}] [{lang.upper()}] ({engine}) {text[:40]}...")

        try:
            audio, sr = tts.synthesize(text, lang=lang)
            all_audio.append(audio)
            all_audio.append(pause)
        except Exception as e:
            print(f"      Erreur: {e}")

    # Concaténer
    if all_audio:
        full_audio = np.concatenate(all_audio)
        sf.write(str(OUTPUT_DIR / "bilingual_narration.wav"), full_audio, 24000)
        print(f"\n  -> bilingual_narration.wav ({len(full_audio)/24000:.1f}s)")


def demo_speed_variations():
    """Démo des variations de vitesse."""
    print("\n" + "="*60)
    print("6. VARIATIONS DE VITESSE")
    print("="*60)

    tts = UnifiedTTS()
    import soundfile as sf

    text = "Ceci est un test de variation de vitesse."
    speeds = [0.75, 1.0, 1.25, 1.5]

    print(f"\n  Texte: {text}\n")

    for speed in speeds:
        print(f"  Vitesse: {speed}x")
        try:
            audio, sr = tts.synthesize(text, lang="fr", speed=speed)
            sf.write(str(OUTPUT_DIR / f"speed_{speed:.2f}.wav"), audio, sr)
            print(f"    -> speed_{speed:.2f}.wav")
        except Exception as e:
            print(f"    Erreur: {e}")


def main():
    print("="*60)
    print("DÉMONSTRATION TTS UNIFIÉ")
    print("Kokoro (offline) + Edge-TTS (Microsoft)")
    print("="*60)
    print(f"\nDossier de sortie: {OUTPUT_DIR}")

    # 1. Sélection automatique
    demo_auto_selection()

    # 2. Toutes les voix françaises
    demo_all_french_voices()

    # 3. Préprocesseur français
    demo_preprocessor()

    # 4. Comparaison des moteurs
    demo_compare_engines()

    # 5. Texte bilingue
    demo_bilingual_text()

    # 6. Variations de vitesse
    demo_speed_variations()

    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)

    wav_files = list(OUTPUT_DIR.glob("*.wav"))
    print(f"\n  {len(wav_files)} fichiers audio générés")
    print(f"  Dossier: {OUTPUT_DIR}")

    print("\n  Fichiers clés à écouter:")
    print("    - compare_kokoro_fr.wav     (Kokoro français)")
    print("    - compare_edge_denise.wav   (Edge-TTS français femme)")
    print("    - compare_edge_henri.wav    (Edge-TTS français homme)")
    print("    - bilingual_narration.wav   (Narration FR/EN mixte)")

    print("\n  Pour écouter:")
    print(f"    cd {OUTPUT_DIR}")
    print("    aplay compare_edge_denise.wav")


if __name__ == "__main__":
    main()
