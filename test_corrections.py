#!/usr/bin/env python3
"""
Outil de test des corrections de prononciation.

Usage:
    python test_corrections.py                    # Mode interactif
    python test_corrections.py "phrase à tester"  # Test rapide
    python test_corrections.py --add mot remplacement  # Ajouter correction
    python test_corrections.py --audio "phrase"   # Générer audio test
"""

import sys
import argparse
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent))

from src.french_preprocessor import FrenchTextPreprocessor


def test_phrase(preprocessor: FrenchTextPreprocessor, phrase: str) -> str:
    """Teste une phrase et affiche le résultat."""
    result = preprocessor.process(phrase)
    return result


def generate_audio(phrase: str, output_path: str = None):
    """Génère un fichier audio pour tester la prononciation."""
    try:
        from src.tts_kokoro_engine import KokoroEngine
        import soundfile as sf

        preprocessor = FrenchTextPreprocessor()
        processed = preprocessor.process(phrase)

        engine = KokoroEngine()
        if not engine.is_available():
            print("Erreur: Kokoro non disponible")
            return None

        output_path = output_path or "/tmp/test_correction.wav"
        success = engine.synthesize(processed, Path(output_path))

        if success:
            # Copier vers Windows si disponible
            windows_path = "/mnt/c/Users/Public/Music/test_correction.wav"
            try:
                import shutil
                shutil.copy(output_path, windows_path)
                return windows_path
            except:
                return output_path
        return None
    except ImportError as e:
        print(f"Erreur import: {e}")
        return None


def interactive_mode():
    """Mode interactif pour tester les corrections."""
    preprocessor = FrenchTextPreprocessor()
    info = preprocessor.get_corrections_info()

    print("=" * 60)
    print("  OUTIL DE TEST DES CORRECTIONS DE PRONONCIATION")
    print("=" * 60)
    print(f"\nDossier corrections: {info['corrections_dir']}")
    print(f"Corrections chargées:")
    print(f"  - Kokoro fixes: {info['kokoro_fixes_count']}")
    print(f"  - Pronunciation: {info['pronunciation_count']}")
    print(f"  - Accents: {info['accent_restore_count']}")
    print(f"  - Tech terms: {info['tech_terms_count']}")

    print("\nCommandes:")
    print("  <phrase>           - Tester une phrase")
    print("  +<mot>=<repl>      - Ajouter correction kokoro_fixes")
    print("  ++<mot>=<repl>     - Ajouter correction pronunciation")
    print("  audio <phrase>     - Générer audio test")
    print("  reload             - Recharger les corrections")
    print("  info               - Afficher les infos")
    print("  quit               - Quitter")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir!")
            break

        if not user_input:
            continue

        if user_input.lower() in ('quit', 'exit', 'q'):
            print("Au revoir!")
            break

        if user_input.lower() == 'reload':
            preprocessor.reload_corrections()
            print("Corrections rechargées!")
            continue

        if user_input.lower() == 'info':
            info = preprocessor.get_corrections_info()
            print(f"Kokoro fixes: {info['kokoro_fixes_count']}")
            print(f"Pronunciation: {info['pronunciation_count']}")
            continue

        if user_input.startswith('audio '):
            phrase = user_input[6:].strip()
            print(f"Génération audio pour: {phrase}")
            processed = preprocessor.process(phrase)
            print(f"Texte traité: {processed}")
            path = generate_audio(phrase)
            if path:
                print(f"Audio généré: {path}")
            continue

        if user_input.startswith('++'):
            # Ajouter correction pronunciation
            try:
                parts = user_input[2:].split('=', 1)
                if len(parts) == 2:
                    word, replacement = parts[0].strip(), parts[1].strip()
                    preprocessor.add_correction(word, replacement, "pronunciation")
                else:
                    print("Format: ++mot=remplacement")
            except Exception as e:
                print(f"Erreur: {e}")
            continue

        if user_input.startswith('+'):
            # Ajouter correction kokoro_fixes
            try:
                parts = user_input[1:].split('=', 1)
                if len(parts) == 2:
                    word, replacement = parts[0].strip(), parts[1].strip()
                    preprocessor.add_correction(word, replacement, "kokoro_fixes")
                else:
                    print("Format: +mot=remplacement")
            except Exception as e:
                print(f"Erreur: {e}")
            continue

        # Tester la phrase
        result = test_phrase(preprocessor, user_input)
        if result != user_input:
            print(f"Original:  {user_input}")
            print(f"Traité:    {result}")
        else:
            print(f"(inchangé) {result}")


def main():
    parser = argparse.ArgumentParser(description="Test des corrections de prononciation")
    parser.add_argument('phrase', nargs='*', help="Phrase à tester")
    parser.add_argument('--add', nargs=2, metavar=('MOT', 'REPL'), help="Ajouter correction")
    parser.add_argument('--audio', metavar='PHRASE', help="Générer audio test")
    parser.add_argument('--category', default='kokoro_fixes', help="Catégorie (kokoro_fixes ou pronunciation)")

    args = parser.parse_args()

    preprocessor = FrenchTextPreprocessor()

    if args.add:
        word, replacement = args.add
        preprocessor.add_correction(word, replacement, args.category)
        return

    if args.audio:
        path = generate_audio(args.audio)
        if path:
            print(f"Audio: {path}")
        return

    if args.phrase:
        phrase = ' '.join(args.phrase)
        result = test_phrase(preprocessor, phrase)
        print(f"Original: {phrase}")
        print(f"Traité:   {result}")
        return

    # Mode interactif
    interactive_mode()


if __name__ == "__main__":
    main()
