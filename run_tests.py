#!/usr/bin/env python3
"""
Script de lancement des tests AudioReader v2.1

Usage:
    python run_tests.py                 # Tous les tests
    python run_tests.py -v              # Mode verbose
    python run_tests.py -k audio_tags   # Tests audio_tags seulement
    python run_tests.py --fast          # Tests rapides seulement
    python run_tests.py --coverage      # Avec couverture de code
    python run_tests.py --module cache  # Tests d'un module specifique

Modules disponibles:
    - audio_tags
    - voice_morphing
    - voice_cloning
    - synthesis_cache
    - emotion_control
    - conversation_generator
    - hq_pipeline_extended
"""
import sys
import subprocess
import argparse
from pathlib import Path


# Repertoire racine du projet
ROOT_DIR = Path(__file__).parent
TESTS_DIR = ROOT_DIR / "tests"


# Mapping des modules vers leurs fichiers de test
MODULES = {
    "audio_tags": "test_audio_tags.py",
    "voice_morphing": "test_voice_morphing.py",
    "voice_cloning": "test_voice_cloning.py",
    "cache": "test_synthesis_cache.py",
    "synthesis_cache": "test_synthesis_cache.py",
    "emotion_control": "test_emotion_control.py",
    "emotion": "test_emotion_control.py",
    "conversation": "test_conversation_generator.py",
    "conversation_generator": "test_conversation_generator.py",
    "pipeline": "test_hq_pipeline_extended.py",
    "hq_pipeline": "test_hq_pipeline_extended.py",
    "hq_pipeline_extended": "test_hq_pipeline_extended.py",
}


def check_dependencies():
    """Verifie les dependances de test."""
    print("=== Verification des dependances ===\n")

    deps = {
        "pytest": False,
        "numpy": False,
        "soundfile": False,
        "librosa": False,
        "TTS": False,
    }

    for dep in deps:
        try:
            __import__(dep.lower() if dep != "TTS" else "TTS")
            deps[dep] = True
            print(f"  [OK] {dep}")
        except ImportError:
            print(f"  [--] {dep} (optionnel)")

    print()

    if not deps["pytest"]:
        print("ERREUR: pytest est requis. Installez avec: pip install pytest")
        return False

    if not deps["numpy"]:
        print("ERREUR: numpy est requis. Installez avec: pip install numpy")
        return False

    return True


def run_tests(args):
    """Execute les tests."""
    if not check_dependencies():
        return 1

    # Construire la commande pytest
    cmd = [sys.executable, "-m", "pytest"]

    # Repertoire de tests
    if args.module:
        module_name = args.module.lower()
        if module_name in MODULES:
            test_file = TESTS_DIR / MODULES[module_name]
            cmd.append(str(test_file))
        else:
            print(f"Module inconnu: {args.module}")
            print(f"Modules disponibles: {', '.join(MODULES.keys())}")
            return 1
    else:
        cmd.append(str(TESTS_DIR))

    # Options
    if args.verbose:
        cmd.append("-v")

    if args.very_verbose:
        cmd.append("-vv")

    if args.keyword:
        cmd.extend(["-k", args.keyword])

    if args.fast:
        cmd.extend(["-m", "not slow"])

    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])

    if args.failed_first:
        cmd.append("--ff")

    if args.exitfirst:
        cmd.append("-x")

    if args.show_locals:
        cmd.append("-l")

    if args.capture_no:
        cmd.append("-s")

    # Couleur
    cmd.append("--color=yes")

    print(f"=== Execution des tests ===\n")
    print(f"Commande: {' '.join(cmd)}\n")

    # Executer
    result = subprocess.run(cmd, cwd=ROOT_DIR)

    return result.returncode


def list_modules():
    """Affiche les modules disponibles."""
    print("\n=== Modules de test disponibles ===\n")

    # Grouper par fichier
    files = {}
    for name, file in MODULES.items():
        if file not in files:
            files[file] = []
        files[file].append(name)

    for file, names in sorted(files.items()):
        print(f"  {file}:")
        for name in sorted(names):
            print(f"    - {name}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Lance les tests AudioReader v2.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Mode verbose"
    )
    parser.add_argument(
        "-vv", "--very-verbose",
        action="store_true",
        help="Mode tres verbose"
    )
    parser.add_argument(
        "-k", "--keyword",
        help="Filtrer par mot-cle (ex: -k 'test_audio')"
    )
    parser.add_argument(
        "--module", "-m",
        help="Tester un module specifique (ex: --module audio_tags)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Exclure les tests lents"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generer un rapport de couverture"
    )
    parser.add_argument(
        "--failed-first", "-ff",
        action="store_true",
        help="Executer d'abord les tests echoues"
    )
    parser.add_argument(
        "-x", "--exitfirst",
        action="store_true",
        help="Arreter au premier echec"
    )
    parser.add_argument(
        "-l", "--show-locals",
        action="store_true",
        help="Afficher les variables locales en cas d'echec"
    )
    parser.add_argument(
        "-s", "--capture-no",
        action="store_true",
        help="Ne pas capturer stdout/stderr"
    )
    parser.add_argument(
        "--list-modules",
        action="store_true",
        help="Lister les modules disponibles"
    )

    args = parser.parse_args()

    if args.list_modules:
        list_modules()
        return 0

    return run_tests(args)


if __name__ == "__main__":
    sys.exit(main())
