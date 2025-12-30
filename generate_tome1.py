#!/usr/bin/env python3
"""
Génère tous les chapitres du Tome 1 des Conquérants du Pognon.
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# Se placer dans le répertoire du script
os.chdir(Path(__file__).parent)

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent))

from src.tts_hybrid_engine import create_hybrid_engine

# Configuration
CHAPTERS_DIR = Path("/home/patrice/claude/livre/Les_Conquerants_du_Pognon/tome-1-or-noir-v2/chapitres")
OUTPUT_DIR = Path("output/les_conquerants/tome1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Chapitres à générer (2-36, le 1 est déjà fait)
START_CHAPTER = 2
END_CHAPTER = 36


def read_chapter(chapter_path: Path) -> str:
    """Lit et nettoie le contenu d'un chapitre."""
    text = chapter_path.read_text(encoding='utf-8')
    lines = text.strip().split('\n')
    # Retirer le titre markdown
    if lines[0].startswith('#'):
        lines = lines[1:]
    return '\n'.join(lines).strip()


def main():
    print("=" * 60)
    print("GÉNÉRATION AUDIOBOOK - Les Conquérants du Pognon - Tome 1")
    print("=" * 60)
    print(f"Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Chapitres: {START_CHAPTER} à {END_CHAPTER}")
    print(f"Sortie: {OUTPUT_DIR}")
    print()

    # Créer le moteur une seule fois
    print("Chargement du moteur hybride...")
    engine = create_hybrid_engine('fr')
    print()

    success_count = 0
    total_duration = 0

    for chapter_num in range(START_CHAPTER, END_CHAPTER + 1):
        chapter_file = CHAPTERS_DIR / f"chapitre-{chapter_num:02d}.md"
        output_file = OUTPUT_DIR / f"chapitre-{chapter_num:02d}.wav"

        if not chapter_file.exists():
            print(f"⚠️  Chapitre {chapter_num:02d} non trouvé, ignoré")
            continue

        if output_file.exists():
            print(f"⏭️  Chapitre {chapter_num:02d} existe déjà, ignoré")
            success_count += 1
            continue

        print("=" * 60)
        print(f"CHAPITRE {chapter_num:02d} / {END_CHAPTER}")
        print("=" * 60)

        try:
            text = read_chapter(chapter_file)
            print(f"Fichier: {chapter_file.name} ({len(text)} caractères)")

            success = engine.synthesize_chapter(text, output_file)

            if success and output_file.exists():
                import soundfile as sf
                audio, sr = sf.read(str(output_file))
                duration = len(audio) / sr
                total_duration += duration
                success_count += 1
                print(f"\n✅ Chapitre {chapter_num:02d} terminé ({duration/60:.1f} min)")
            else:
                print(f"\n❌ Chapitre {chapter_num:02d} échoué")

        except Exception as e:
            print(f"\n❌ Erreur chapitre {chapter_num:02d}: {e}")
            import traceback
            traceback.print_exc()

        print()

    # Résumé final
    print("=" * 60)
    print("RÉSUMÉ FINAL")
    print("=" * 60)
    print(f"Chapitres générés: {success_count} / {END_CHAPTER - START_CHAPTER + 1}")
    print(f"Durée totale: {total_duration/60:.1f} min ({total_duration/3600:.1f} heures)")
    print(f"Fichiers dans: {OUTPUT_DIR}")
    print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
