#!/usr/bin/env python3
"""
Test d'intégration pour AudioReader.
Vérifie que tous les composants fonctionnent ensemble.
"""

import sys
import tempfile
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_french_preprocessor():
    """Test du préprocesseur français (version simplifiée)."""
    from french_preprocessor import FrenchTextPreprocessor

    p = FrenchTextPreprocessor()

    # Tests pour la version simplifiée (num2words, abréviations, acronymes)
    tests = [
        # Conversion heures
        ("19h30", "dix-neuf heures trente"),
        ("8h00", "huit heures"),
        # Abréviations (capitalisées)
        ("M. Dupont", "Monsieur Dupont"),
        ("Mme Martin", "Madame Martin"),
        ("Dr House", "Docteur House"),
        # Acronymes
        ("SNCF", "S.N.C.F"),
        ("TGV", "T.G.V"),
        # Mots normaux (inchangés)
        ("Bonjour", "Bonjour"),
        ("année", "année"),
        ("collègue", "collègue"),
    ]

    print("Test préprocesseur français:")
    failed = []
    for input_text, expected in tests:
        result = p.process(input_text)
        status = "✓" if result == expected else "✗"
        if result != expected:
            failed.append(f"{input_text} -> {result} (attendu: {expected})")
            print(f"  {status} {input_text} -> {result} (attendu: {expected})")
        else:
            print(f"  {status} {input_text} -> {result}")

    assert not failed, f"Échecs: {failed}"


def test_dialogue_detector():
    """Test du détecteur de dialogues."""
    from dialogue_detector import detect_dialogues, SegmentType

    print("\nTest détecteur de dialogues:")

    text = 'Marie dit: "Bonjour Pierre!" Il répondit: "Salut!"'
    segments = detect_dialogues(text)

    # Vérifier qu'on a des segments de chaque type
    has_dialogue = any(s.segment_type == SegmentType.DIALOGUE for s in segments)
    has_narration = any(s.segment_type == SegmentType.NARRATION for s in segments)

    print(f"  Texte: {text[:50]}...")
    print(f"  Segments trouvés: {len(segments)}")
    for s in segments:
        print(f"    [{s.segment_type.value}] {s.text[:40]}...")

    assert has_dialogue, "Aucun dialogue détecté"
    assert has_narration, "Aucune narration détectée"
    print(f"  ✓ Dialogues et narration détectés")


def test_audio_postprocess_config():
    """Test de la configuration du post-processeur."""
    from audio_postprocess import PostProcessConfig

    print("\nTest config post-processeur:")

    config = PostProcessConfig()

    # Vérifier les valeurs par défaut corrigées
    checks = [
        ("trim_silence désactivé", config.trim_silence == False),
        ("threshold -60dB", config.silence_threshold == -60.0),
        ("min_duration 1.0s", config.min_silence_duration == 1.0),
        ("normalize activé", config.normalize == True),
        ("target loudness -20 LUFS", config.target_loudness == -20.0),
    ]

    failed = []
    for name, check in checks:
        status = "✓" if check else "✗"
        print(f"  {status} {name}")
        if not check:
            failed.append(name)

    assert not failed, f"Échecs: {failed}"


def test_markdown_parser():
    """Test du parser Markdown."""
    from markdown_parser import parse_book

    print("\nTest parser Markdown:")

    content = """# Chapitre 1

Ceci est le premier chapitre.

# Chapitre 2

Ceci est le deuxième chapitre.
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(content)
        temp_path = f.name

    try:
        chapters = parse_book(temp_path)
        print(f"  Chapitres trouvés: {len(chapters)}")
        for ch in chapters:
            print(f"    {ch.number}. {ch.title}")

        assert len(chapters) == 2, f"Attendu 2 chapitres, obtenu {len(chapters)}"
        print(f"  ✓ 2 chapitres parsés")
    finally:
        Path(temp_path).unlink()


def test_kokoro_availability():
    """Test de la disponibilité de Kokoro."""
    try:
        from tts_kokoro_engine import KokoroEngine, KOKORO_VOICES

        print("\nTest Kokoro TTS:")
        print(f"  Voix disponibles: {len(KOKORO_VOICES)}")

        engine = KokoroEngine()
        available = engine.is_available()

        print(f"  {'✓' if available else '⚠'} Modèle {'disponible' if available else 'non installé'}")

        # Lister quelques voix
        for vid in list(KOKORO_VOICES.keys())[:3]:
            info = KOKORO_VOICES[vid]
            print(f"    - {vid}: {info['name']} ({info['lang']})")

        # Le test passe même si le modèle n'est pas installé (vérifie juste l'import)
        assert len(KOKORO_VOICES) > 0, "Aucune voix Kokoro définie"
    except ImportError as e:
        print(f"\n⚠ Kokoro non disponible: {e}")
        # Pas une erreur critique - le module peut ne pas être installé


def main():
    """Lance tous les tests."""
    print("=" * 60)
    print("   TESTS D'INTÉGRATION AUDIOREADER")
    print("=" * 60)

    results = []

    results.append(("Préprocesseur français", test_french_preprocessor()))
    results.append(("Détecteur de dialogues", test_dialogue_detector()))
    results.append(("Config post-processeur", test_audio_postprocess_config()))
    results.append(("Parser Markdown", test_markdown_parser()))
    results.append(("Kokoro TTS", test_kokoro_availability()))

    print("\n" + "=" * 60)
    print("   RÉSULTATS")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    print("\n" + ("✓ Tous les tests passent!" if all_passed else "✗ Certains tests ont échoué"))

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
