#!/usr/bin/env python3
"""Test Kokoro TTS haute qualité avec le livre exemple."""
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent / "src"))

from markdown_parser import parse_book
from kokoro_onnx import Kokoro
import soundfile as sf


# Voix disponibles Kokoro
VOICES = {
    # Français
    "ff_siwis": "Siwis - Femme (France)",
    # Anglais US - Femmes
    "af_heart": "Heart - Female (US)",
    "af_bella": "Bella - Female (US)",
    "af_nicole": "Nicole - Female (US)",
    "af_sky": "Sky - Female (US)",
    # Anglais US - Hommes
    "am_adam": "Adam - Male (US)",
    "am_michael": "Michael - Male (US)",
    # Anglais UK
    "bf_emma": "Emma - Female (UK)",
    "bm_george": "George - Male (UK)",
}


def main():
    print("=" * 70)
    print("   KOKORO TTS - TEST HAUTE QUALITÉ")
    print("=" * 70)

    # Vérifier fichiers modèle
    model_path = Path("kokoro-v1.0.onnx")
    voices_path = Path("voices-v1.0.bin")

    if not model_path.exists() or not voices_path.exists():
        print("\nERREUR: Fichiers modèle manquants.")
        print("Téléchargez-les avec:")
        print('  curl -L -o kokoro-v1.0.onnx "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"')
        print('  curl -L -o voices-v1.0.bin "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"')
        return 1

    # Charger le modèle
    print("\nChargement du modèle...")
    start = time.time()
    kokoro = Kokoro(str(model_path), str(voices_path))
    print(f"Modèle chargé en {time.time()-start:.1f}s")

    # Lire le livre
    print("\nLecture du livre exemple...")
    chapters = parse_book("books/exemple_livre.md")
    print(f"Chapitres: {len(chapters)}")

    total_chars = sum(len(ch.get_full_text()) for ch in chapters)
    print(f"Caractères: {total_chars:,}")

    # Créer dossier sortie
    output_dir = Path("output_kokoro")
    output_dir.mkdir(exist_ok=True)

    # Sélection voix
    voice = "ff_siwis"  # Voix française
    lang = "fr-fr"
    print(f"\nVoix: {VOICES.get(voice, voice)}")
    print(f"Langue: {lang}")

    print("\n" + "-" * 70)
    print("Conversion des chapitres...")
    print("-" * 70)

    total_gen_time = 0
    total_audio_time = 0

    for chapter in chapters:
        text = chapter.get_full_text()
        filename = chapter.get_filename()
        output_path = output_dir / f"{filename}.wav"

        print(f"\n  Chapitre {chapter.number}: {chapter.title}")
        print(f"    Caractères: {len(text):,}")

        start = time.time()

        # Générer audio
        samples, sr = kokoro.create(text, voice=voice, speed=1.0, lang=lang)

        gen_time = time.time() - start
        audio_duration = len(samples) / sr

        total_gen_time += gen_time
        total_audio_time += audio_duration

        # Sauvegarder
        sf.write(str(output_path), samples, sr)

        ratio = audio_duration / gen_time if gen_time > 0 else 0
        print(f"    Audio: {audio_duration:.1f}s | Génération: {gen_time:.1f}s | {ratio:.1f}x temps réel")

    # Résumé
    print("\n" + "=" * 70)
    print("   RÉSUMÉ")
    print("=" * 70)

    overall_ratio = total_audio_time / total_gen_time if total_gen_time > 0 else 0

    print(f"""
  Chapitres convertis: {len(chapters)}
  Caractères totaux:   {total_chars:,}

  Durée audio totale:  {total_audio_time:.1f}s ({total_audio_time/60:.1f} min)
  Temps de génération: {total_gen_time:.1f}s
  Performance:         {overall_ratio:.1f}x temps réel

  Dossier de sortie: {output_dir}/
""")

    # Lister fichiers
    print("  Fichiers générés:")
    total_size = 0
    for f in sorted(output_dir.glob("*.wav")):
        size = f.stat().st_size / 1024 / 1024
        total_size += size
        print(f"    {f.name:50} {size:.1f} MB")

    print(f"\n  Taille totale: {total_size:.1f} MB")

    # Info qualité
    print(f"""
{'=' * 70}
   INFORMATIONS QUALITÉ
{'=' * 70}

  Modèle: Kokoro-82M (82 millions de paramètres)
  Format: WAV 24kHz 16-bit mono

  Pour publication:
  - Convertir en MP3 192kbps pour distribution
  - Normaliser à -20dB RMS (standard Audible)
  - Ajouter room tone 0.75s début / 2s fin

  Commande FFmpeg pour conversion MP3:
    ffmpeg -i input.wav -b:a 192k -ar 44100 output.mp3
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
