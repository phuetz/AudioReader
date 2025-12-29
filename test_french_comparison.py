#!/usr/bin/env python3
"""
Comparaison des 3 solutions TTS pour le français:
1. Kokoro brut (sans prétraitement)
2. Kokoro + préprocesseur français
3. Edge-TTS (Microsoft Azure)
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

OUTPUT_DIR = Path(__file__).parent / "demo_output" / "french_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Textes de test avec différents défis de prononciation française
TEST_TEXTS = [
    {
        "id": "01_accents",
        "text": "C'était une journée ensoleillée. Où êtes-vous allé cet été?",
        "desc": "Accents (é, è, ê, ù)"
    },
    {
        "id": "02_cedille",
        "text": "Le garçon français a reçu sa leçon. Ça commence bien.",
        "desc": "Cédille (ç)"
    },
    {
        "id": "03_nasales",
        "text": "Un bon vin blanc dans un restaurant charmant.",
        "desc": "Voyelles nasales (an, in, on, un)"
    },
    {
        "id": "04_liaisons",
        "text": "Les enfants ont appris leurs leçons. Ils adorent les histoires.",
        "desc": "Liaisons"
    },
    {
        "id": "05_abbreviations",
        "text": "M. Dupont et Mme Martin ont rendez-vous. Dr Bernard arrive à 15h.",
        "desc": "Abréviations"
    },
    {
        "id": "06_nombres",
        "text": "Il y avait 250 personnes. C'est le 1er janvier 2024.",
        "desc": "Nombres"
    },
    {
        "id": "07_anglicismes",
        "text": "J'ai un meeting important. Envoyez-moi un email sur mon smartphone.",
        "desc": "Anglicismes"
    },
    {
        "id": "08_dialogue",
        "text": "«Bonjour!» dit Marie. «Comment allez-vous aujourd'hui?»",
        "desc": "Dialogue avec guillemets"
    },
    {
        "id": "09_complexe",
        "text": "François était très heureux d'être à Noël. Sa sœur était là aussi.",
        "desc": "Texte complexe"
    },
    {
        "id": "10_long",
        "text": "Il était une fois, dans un petit village au cœur de la France, une jeune fille nommée Amélie. Elle rêvait de voyager et de découvrir le monde entier.",
        "desc": "Texte long narratif"
    },
]


def test_kokoro_raw():
    """Test Kokoro sans prétraitement."""
    print("\n" + "="*60)
    print("SOLUTION 1: Kokoro TTS (brut)")
    print("="*60)

    try:
        from kokoro_onnx import Kokoro
        import soundfile as sf

        kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

        for test in TEST_TEXTS:
            print(f"\n  [{test['id']}] {test['desc']}")
            print(f"  Texte: {test['text'][:50]}...")

            try:
                audio, sr = kokoro.create(
                    test['text'],
                    voice="ff_siwis",
                    speed=1.0,
                    lang="fr-fr"
                )
                output_file = OUTPUT_DIR / f"{test['id']}_kokoro_raw.wav"
                sf.write(str(output_file), audio, sr)
                print(f"  -> {output_file.name}")
            except Exception as e:
                print(f"  ERREUR: {e}")

        return True
    except Exception as e:
        print(f"ERREUR Kokoro: {e}")
        return False


def test_kokoro_preprocessed():
    """Test Kokoro avec préprocesseur français."""
    print("\n" + "="*60)
    print("SOLUTION 2: Kokoro TTS + Préprocesseur français")
    print("="*60)

    try:
        from kokoro_onnx import Kokoro
        import soundfile as sf
        from src.french_preprocessor import FrenchTextPreprocessor

        kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
        preprocessor = FrenchTextPreprocessor()

        for test in TEST_TEXTS:
            print(f"\n  [{test['id']}] {test['desc']}")

            # Prétraiter le texte
            processed_text = preprocessor.process(test['text'])
            print(f"  Original: {test['text'][:40]}...")
            print(f"  Traité:   {processed_text[:40]}...")

            try:
                audio, sr = kokoro.create(
                    processed_text,
                    voice="ff_siwis",
                    speed=1.0,
                    lang="fr-fr"
                )
                output_file = OUTPUT_DIR / f"{test['id']}_kokoro_preproc.wav"
                sf.write(str(output_file), audio, sr)
                print(f"  -> {output_file.name}")
            except Exception as e:
                print(f"  ERREUR: {e}")

        return True
    except Exception as e:
        print(f"ERREUR Kokoro+Preproc: {e}")
        return False


async def test_edge_tts():
    """Test Edge-TTS (Microsoft)."""
    print("\n" + "="*60)
    print("SOLUTION 3: Edge-TTS (Microsoft Azure)")
    print("="*60)

    try:
        import edge_tts

        # Voix françaises disponibles
        FRENCH_VOICES = [
            ("fr-FR-DeniseNeural", "Denise (femme)"),
            ("fr-FR-HenriNeural", "Henri (homme)"),
        ]

        print(f"\n  Voix utilisée: {FRENCH_VOICES[0][1]}")
        voice = FRENCH_VOICES[0][0]

        for test in TEST_TEXTS:
            print(f"\n  [{test['id']}] {test['desc']}")
            print(f"  Texte: {test['text'][:50]}...")

            try:
                output_file = OUTPUT_DIR / f"{test['id']}_edge_tts.wav"

                # Edge-TTS génère en MP3, on convertit en WAV
                mp3_file = OUTPUT_DIR / f"{test['id']}_edge_tts.mp3"

                communicate = edge_tts.Communicate(test['text'], voice)
                await communicate.save(str(mp3_file))

                # Convertir MP3 en WAV avec ffmpeg ou pydub
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_mp3(str(mp3_file))
                    audio.export(str(output_file), format="wav")
                    mp3_file.unlink()  # Supprimer le MP3
                    print(f"  -> {output_file.name}")
                except ImportError:
                    # Si pydub n'est pas disponible, garder le MP3
                    print(f"  -> {mp3_file.name} (MP3)")

            except Exception as e:
                print(f"  ERREUR: {e}")

        return True
    except ImportError:
        print("  edge-tts non installé. Installation...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "edge-tts", "-q"])
        return False
    except Exception as e:
        print(f"ERREUR Edge-TTS: {e}")
        return False


async def test_edge_tts_voices():
    """Liste les voix Edge-TTS disponibles pour le français."""
    print("\n  Voix françaises Edge-TTS disponibles:")
    try:
        import edge_tts
        voices = await edge_tts.list_voices()
        french_voices = [v for v in voices if v['Locale'].startswith('fr-')]
        for v in french_voices:
            print(f"    - {v['ShortName']}: {v['Gender']}")
    except Exception as e:
        print(f"    Erreur: {e}")


def generate_comparison_html():
    """Génère une page HTML pour comparer les résultats."""
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Comparaison TTS Français</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        audio { width: 200px; }
        .test-desc { font-size: 0.9em; color: #666; }
    </style>
</head>
<body>
    <h1>🎤 Comparaison TTS Français</h1>
    <p>Comparez la qualité de prononciation française entre les 3 solutions.</p>

    <table>
        <tr>
            <th>Test</th>
            <th>Texte</th>
            <th>Kokoro (brut)</th>
            <th>Kokoro + Préproc</th>
            <th>Edge-TTS</th>
        </tr>
"""

    for test in TEST_TEXTS:
        html += f"""        <tr>
            <td><strong>{test['id']}</strong><br><span class="test-desc">{test['desc']}</span></td>
            <td>{test['text']}</td>
            <td><audio controls src="{test['id']}_kokoro_raw.wav"></audio></td>
            <td><audio controls src="{test['id']}_kokoro_preproc.wav"></audio></td>
            <td><audio controls src="{test['id']}_edge_tts.wav"></audio></td>
        </tr>
"""

    html += """    </table>

    <h2>📊 Résumé</h2>
    <ul>
        <li><strong>Kokoro (brut)</strong>: Qualité variable, problèmes avec certains accents</li>
        <li><strong>Kokoro + Préprocesseur</strong>: Amélioration pour abréviations et nombres</li>
        <li><strong>Edge-TTS</strong>: Meilleure qualité française (Microsoft Azure)</li>
    </ul>
</body>
</html>
"""

    html_file = OUTPUT_DIR / "comparison.html"
    html_file.write_text(html, encoding='utf-8')
    print(f"\n  Page de comparaison: {html_file}")


async def main():
    print("="*60)
    print("COMPARAISON DES SOLUTIONS TTS FRANÇAIS")
    print("="*60)
    print(f"Dossier de sortie: {OUTPUT_DIR}")

    # Solution 1: Kokoro brut
    test_kokoro_raw()

    # Solution 2: Kokoro + préprocesseur
    test_kokoro_preprocessed()

    # Solution 3: Edge-TTS
    await test_edge_tts_voices()
    await test_edge_tts()

    # Générer la page de comparaison
    print("\n" + "="*60)
    print("GÉNÉRATION DE LA PAGE DE COMPARAISON")
    print("="*60)
    generate_comparison_html()

    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)

    wav_files = list(OUTPUT_DIR.glob("*.wav"))
    mp3_files = list(OUTPUT_DIR.glob("*.mp3"))

    print(f"\nFichiers générés: {len(wav_files)} WAV, {len(mp3_files)} MP3")
    print(f"\nPour comparer, ouvrez: {OUTPUT_DIR / 'comparison.html'}")
    print("\nOu écoutez directement:")
    print(f"  cd {OUTPUT_DIR}")
    print("  # Kokoro brut:")
    print("  aplay 01_accents_kokoro_raw.wav")
    print("  # Kokoro + préprocesseur:")
    print("  aplay 01_accents_kokoro_preproc.wav")
    print("  # Edge-TTS:")
    print("  aplay 01_accents_edge_tts.wav")


if __name__ == "__main__":
    asyncio.run(main())
