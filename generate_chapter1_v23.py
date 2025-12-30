#!/usr/bin/env python3
"""
Génère l'audio du chapitre 1 avec le pipeline v2.3.

Fonctionnalités activées:
- Contours d'intonation (questions, exclamations, etc.)
- Humanisation du timing (variation des pauses)
- Pauses d'emphase avant mots importants
- Respirations avancées (bruit rose + formants)
- Moteur hybride MMS + Kokoro
"""
import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np

# Se placer dans le répertoire du script
os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

from src.hq_pipeline_extended import (
    ExtendedHQPipeline,
    ExtendedPipelineConfig,
)
from src.tts_hybrid_engine import HybridTTSEngine

# Configuration
CHAPTER_PATH = Path("/home/patrice/claude/livre/Les_Conquerants_du_Pognon/tome-1-or-noir-v2/chapitres/chapitre-01.md")
OUTPUT_DIR = Path("output/les_conquerants/tome1")
OUTPUT_FILE_WAV = OUTPUT_DIR / "chapitre-01-v23.wav"
OUTPUT_FILE_MP3 = OUTPUT_DIR / "chapitre-01-v23.mp3"

# Format de sortie: "wav" ou "mp3"
OUTPUT_FORMAT = "mp3"


def read_chapter(path: Path) -> str:
    """Lit le contenu du chapitre en nettoyant le Markdown."""
    text = path.read_text(encoding='utf-8')

    lines = text.split('\n')
    clean_lines = []

    for line in lines:
        # Ignorer les en-têtes Markdown (# titre)
        if line.strip().startswith('#'):
            continue
        # Ignorer les épigraphes (> *«...)
        if line.strip().startswith('>'):
            continue
        # Ignorer les séparateurs ---
        if line.strip() == '---':
            continue
        # Ignorer les lignes avec seulement des * ou _
        if line.strip() in ('*', '**', '_', '__', ''):
            continue

        clean_lines.append(line)

    # Joindre et nettoyer les espaces multiples
    text = '\n'.join(clean_lines)

    # Supprimer les lignes vides multiples
    import re
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def main():
    print("=" * 70)
    print("GÉNÉRATION CHAPITRE 1 - Pipeline v2.3 + Moteur Hybride")
    print("=" * 70)
    print(f"Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Créer le dossier de sortie
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Charger le chapitre
    print(f"Lecture: {CHAPTER_PATH.name}")
    chapter_text = read_chapter(CHAPTER_PATH)
    print(f"  {len(chapter_text)} caractères")
    print()

    # Créer le moteur hybride
    print("Chargement du moteur hybride (MMS + Kokoro)...")
    hybrid_engine = HybridTTSEngine(mms_language='fra')
    print()

    # Générer avec le moteur hybride standard
    print(f"Génération de l'audio (format: {OUTPUT_FORMAT})...")
    print("-" * 70)

    # Choisir le fichier de sortie
    output_file = OUTPUT_FILE_MP3 if OUTPUT_FORMAT == "mp3" else OUTPUT_FILE_WAV

    success = hybrid_engine.synthesize_chapter(
        chapter_text,
        output_file,
        output_format=OUTPUT_FORMAT,
    )

    if success:
        print()
        print("=" * 70)
        print("RÉSULTAT")
        print("=" * 70)
        print(f"Fichier: {output_file}")
        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"Taille: {size_mb:.1f} MB")
        print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("Échec de la génération")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
