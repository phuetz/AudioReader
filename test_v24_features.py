#!/usr/bin/env python3
"""
Test des nouvelles fonctionnalites v2.4:
- Crossfade audio
- Preview rapide
- Interface corrections
- Moteur XTTS-v2
"""
import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent))


def test_crossfade():
    """Test du module crossfade."""
    print("\n=== Test Crossfade ===")
    try:
        import numpy as np
        from src.audio_crossfade import (
            AudioCrossfader, CrossfadeConfig,
            apply_crossfade_to_chapter
        )

        # Creer deux segments de test
        sr = 24000
        t = np.linspace(0, 0.5, int(sr * 0.5))
        segment1 = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        segment2 = (np.sin(2 * np.pi * 880 * t) * 0.5).astype(np.float32)

        # Test configuration
        config = CrossfadeConfig(
            crossfade_duration=0.05,
            curve_type='cosine'
        )
        crossfader = AudioCrossfader(config)

        # Test crossfade
        merged = crossfader.crossfade_segments(segment1, segment2, sr)

        print(f"  Segment1: {len(segment1)} samples")
        print(f"  Segment2: {len(segment2)} samples")
        print(f"  Merged:   {len(merged)} samples")

        # Verifier que le crossfade a reduit la longueur totale
        expected_reduction = int(0.05 * sr)
        actual_length = len(merged)
        expected_length = len(segment1) + len(segment2) - expected_reduction

        assert abs(actual_length - expected_length) < 10, "Crossfade length incorrect"
        print("  [OK] Crossfade fonctionne correctement")

        # Test apply_crossfade_to_chapter
        segments = [segment1, segment2, segment1]
        chapter = apply_crossfade_to_chapter(segments, sr, 50)
        print(f"  Chapter: {len(chapter)} samples ({len(chapter)/sr:.2f}s)")
        print("  [OK] apply_crossfade_to_chapter fonctionne")

        return True

    except Exception as e:
        print(f"  [ERREUR] {e}")
        return False


def test_preview_generator():
    """Test du generateur de preview."""
    print("\n=== Test Preview Generator ===")
    try:
        from src.preview_generator import PreviewGenerator, PreviewConfig

        # Texte de test
        test_text = """
        L'argent a une odeur. Celle de la sueur des autres.

        Victor le savait depuis son enfance a Saint-Denis, quand il observait
        les ouvriers de l'usine rentrer chez eux, le visage gris de fatigue.

        « Tu comprends, petit, » lui avait dit un jour son pere, « dans ce monde,
        y'a ceux qui comptent l'argent et ceux qui le gagnent. »

        Cette phrase l'avait marque a jamais. Soudain, il avait compris que
        la vie etait un jeu dont les regles etaient ecrites par les riches.

        Et il avait decide qu'un jour, c'est lui qui ecrirait les regles.
        """

        config = PreviewConfig(
            target_duration=30.0,
            target_chars=450,
            include_dialogue=True,
            include_emotional=True
        )

        generator = PreviewGenerator(config)
        preview = generator.extract_preview_text(test_text, 'fr')

        print(f"  Texte original: {len(test_text)} chars")
        print(f"  Preview extrait: {len(preview)} chars")
        print(f"  Extrait: {preview[:100]}...")

        # Verifier que le preview contient du dialogue
        has_dialogue = '«' in preview or '"' in preview
        print(f"  Contient dialogue: {has_dialogue}")

        assert len(preview) <= 500, "Preview trop long"
        assert len(preview) > 100, "Preview trop court"
        print("  [OK] Preview generator fonctionne")

        return True

    except Exception as e:
        print(f"  [ERREUR] {e}")
        return False


def test_corrections_manager():
    """Test du gestionnaire de corrections."""
    print("\n=== Test Corrections Manager ===")
    try:
        from src.corrections_ui import CorrectionManager
        import tempfile
        import json

        # Creer un fichier temporaire
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'version': '1.0',
                'corrections': {
                    'API': 'A P I',
                    'JSON': 'jason'
                }
            }, f)
            temp_path = Path(f.name)

        try:
            manager = CorrectionManager(temp_path)

            # Test chargement
            corrections = manager.get_all()
            print(f"  Corrections chargees: {len(corrections)}")
            assert len(corrections) == 2, "Nombre de corrections incorrect"

            # Test ajout
            manager.add('URL', 'U R L')
            corrections = manager.get_all()
            assert len(corrections) == 3, "Ajout echoue"
            print("  [OK] Ajout fonctionne")

            # Test application
            text = "L'API et le JSON sont importants"
            corrected = manager.apply(text)
            print(f"  Original:  {text}")
            print(f"  Corrige:   {corrected}")
            assert 'A P I' in corrected, "Correction API non appliquee"
            assert 'jason' in corrected, "Correction JSON non appliquee"
            print("  [OK] Application des corrections fonctionne")

            # Test recherche
            results = manager.search('API')
            assert len(results) == 1, "Recherche echouee"
            print("  [OK] Recherche fonctionne")

            # Test suppression
            manager.remove('URL')
            corrections = manager.get_all()
            assert len(corrections) == 2, "Suppression echouee"
            print("  [OK] Suppression fonctionne")

            return True

        finally:
            temp_path.unlink()

    except Exception as e:
        print(f"  [ERREUR] {e}")
        return False


def test_xtts_engine_import():
    """Test de l'import du moteur XTTS (sans charger le modele)."""
    print("\n=== Test XTTS Engine Import ===")
    try:
        from src.tts_xtts_engine import XTTSEngine, XTTSConfig, create_xtts_engine

        # Test configuration
        config = XTTSConfig(
            default_language="fr",
            use_gpu=False,
            temperature=0.7
        )

        print(f"  Language: {config.default_language}")
        print(f"  GPU: {config.use_gpu}")
        print(f"  Temperature: {config.temperature}")

        # Creer l'engine (sans charger le modele)
        engine = XTTSEngine(config)

        # Test list voices
        voices = engine.list_available_voices()
        print(f"  Voix par defaut: {voices['default']}")
        print(f"  Voix clonees: {voices['cloned']}")

        print("  [OK] Import XTTS fonctionne")
        print("  [INFO] Modele non charge (TTS library peut manquer)")

        return True

    except ImportError as e:
        print(f"  [INFO] TTS library non installee: {e}")
        print("  [INFO] Pour installer: pip install TTS torch torchaudio")
        return True  # Ce n'est pas une erreur

    except Exception as e:
        print(f"  [ERREUR] {e}")
        return False


def test_hybrid_crossfade_integration():
    """Test de l'integration crossfade dans le moteur hybride."""
    print("\n=== Test Integration Crossfade Hybrid ===")
    try:
        from src.tts_hybrid_engine import HybridTTSEngine, create_hybrid_engine

        # Verifier que les parametres existent
        engine = HybridTTSEngine(
            use_crossfade=True,
            crossfade_ms=50
        )

        print(f"  use_crossfade: {engine.use_crossfade}")
        print(f"  crossfade_ms: {engine.crossfade_ms}")

        # Test factory function
        engine2 = create_hybrid_engine(
            language="fr",
            use_crossfade=True,
            crossfade_ms=100
        )

        print(f"  Factory crossfade_ms: {engine2.crossfade_ms}")
        print("  [OK] Integration crossfade fonctionne")

        return True

    except Exception as e:
        print(f"  [ERREUR] {e}")
        return False


def main():
    """Execute tous les tests."""
    print("=" * 60)
    print("Tests des fonctionnalites v2.4")
    print("=" * 60)

    results = {
        'crossfade': test_crossfade(),
        'preview': test_preview_generator(),
        'corrections': test_corrections_manager(),
        'xtts_import': test_xtts_engine_import(),
        'hybrid_integration': test_hybrid_crossfade_integration(),
    }

    print("\n" + "=" * 60)
    print("RESUME")
    print("=" * 60)

    passed = 0
    failed = 0

    for name, result in results.items():
        status = "OK" if result else "FAIL"
        print(f"  {name:25} [{status}]")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passes, {failed} echecs")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
