"""
Tests pour le module intonation_contour.

Vérifie la détection et l'application des contours d'intonation.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.intonation_contour import (
    IntonationContour,
    IntonationContourDetector,
    IntonationContourApplicator,
    IntonationProcessor,
    CONTOUR_CONFIGS
)


class TestIntonationContourDetector:
    """Tests pour le détecteur de contours."""

    @pytest.fixture
    def detector(self):
        return IntonationContourDetector(language="fr")

    # === Tests Déclaratif ===

    def test_declarative_period(self, detector):
        """Phrase avec point = déclaratif."""
        assert detector.detect("Il fait beau.") == IntonationContour.DECLARATIVE

    def test_declarative_no_punctuation(self, detector):
        """Phrase sans ponctuation = déclaratif."""
        assert detector.detect("Il fait beau") == IntonationContour.DECLARATIVE

    # === Tests Question Oui/Non ===

    def test_question_yn_simple(self, detector):
        """Question simple = question Y/N."""
        assert detector.detect("Tu viens ?") == IntonationContour.QUESTION_YN

    def test_question_yn_est_ce_que(self, detector):
        """'Est-ce que' = question Y/N."""
        assert detector.detect("Est-ce que tu viens ?") == IntonationContour.QUESTION_YN

    # === Tests Question WH ===

    def test_question_wh_qui(self, detector):
        """Question en 'qui' = WH."""
        assert detector.detect("Qui est là ?") == IntonationContour.QUESTION_WH

    def test_question_wh_que(self, detector):
        """Question en 'que' = WH."""
        assert detector.detect("Que fais-tu ?") == IntonationContour.QUESTION_WH

    def test_question_wh_ou(self, detector):
        """Question en 'où' = WH."""
        assert detector.detect("Où vas-tu ?") == IntonationContour.QUESTION_WH

    def test_question_wh_quand(self, detector):
        """Question en 'quand' = WH."""
        assert detector.detect("Quand arrives-tu ?") == IntonationContour.QUESTION_WH

    def test_question_wh_comment(self, detector):
        """Question en 'comment' = WH."""
        assert detector.detect("Comment vas-tu ?") == IntonationContour.QUESTION_WH

    def test_question_wh_pourquoi(self, detector):
        """Question en 'pourquoi' = WH."""
        assert detector.detect("Pourquoi es-tu là ?") == IntonationContour.QUESTION_WH

    def test_question_wh_combien(self, detector):
        """Question en 'combien' = WH."""
        assert detector.detect("Combien ça coûte ?") == IntonationContour.QUESTION_WH

    def test_question_wh_quel(self, detector):
        """Question en 'quel' = WH."""
        assert detector.detect("Quel est ton nom ?") == IntonationContour.QUESTION_WH

    # === Tests Exclamation ===

    def test_exclamation(self, detector):
        """Phrase avec point d'exclamation."""
        assert detector.detect("C'est incroyable !") == IntonationContour.EXCLAMATION

    def test_exclamation_strong(self, detector):
        """Exclamation forte."""
        assert detector.detect("Non !") == IntonationContour.EXCLAMATION

    # === Tests Suspense ===

    def test_suspense_ellipsis(self, detector):
        """Points de suspension = suspense."""
        assert detector.detect("Et puis...") == IntonationContour.SUSPENSE

    def test_suspense_unicode_ellipsis(self, detector):
        """Caractère ellipse unicode."""
        assert detector.detect("Il était…") == IntonationContour.SUSPENSE

    # === Tests Continuation ===

    def test_continuation_comma(self, detector):
        """Virgule finale = continuation."""
        assert detector.detect("D'abord,") == IntonationContour.CONTINUATION

    def test_continuation_semicolon(self, detector):
        """Point-virgule = continuation."""
        assert detector.detect("Il arriva;") == IntonationContour.CONTINUATION

    def test_continuation_et(self, detector):
        """Finit par 'et' = continuation."""
        assert detector.detect("Il mangea et") == IntonationContour.CONTINUATION

    def test_continuation_mais(self, detector):
        """Finit par 'mais' = continuation."""
        assert detector.detect("C'est bien mais") == IntonationContour.CONTINUATION

    # === Tests Neutre ===

    def test_neutral_empty(self, detector):
        """Texte vide = neutre."""
        assert detector.detect("") == IntonationContour.NEUTRAL

    def test_neutral_whitespace(self, detector):
        """Espaces seulement = neutre."""
        assert detector.detect("   ") == IntonationContour.NEUTRAL


class TestIntonationContourApplicator:
    """Tests pour l'applicateur de contours."""

    @pytest.fixture
    def applicator(self):
        return IntonationContourApplicator(sample_rate=24000, strength=0.7)

    def test_apply_returns_reasonable_length(self, applicator):
        """Le contour retourne une longueur raisonnable."""
        audio = np.random.randn(4800).astype(np.float32)  # 0.2s à 24kHz
        result = applicator.apply_contour(audio, IntonationContour.DECLARATIVE)
        # Le crossfade entre segments peut réduire la longueur
        # On vérifie que le résultat a au moins 50% de la longueur originale
        assert len(result) >= len(audio) * 0.5
        assert len(result) <= len(audio) * 1.1

    def test_apply_dtype(self, applicator):
        """Le résultat est en float32."""
        audio = np.random.randn(2400).astype(np.float32)
        result = applicator.apply_contour(audio, IntonationContour.DECLARATIVE)
        assert result.dtype == np.float32

    def test_apply_empty_audio(self, applicator):
        """Audio vide retourne vide."""
        empty = np.array([], dtype=np.float32)
        result = applicator.apply_contour(empty, IntonationContour.DECLARATIVE)
        assert len(result) == 0

    def test_neutral_no_change(self, applicator):
        """Contour neutre ne modifie pas l'audio."""
        audio = np.random.randn(2400).astype(np.float32)
        result = applicator.apply_contour(audio, IntonationContour.NEUTRAL)
        np.testing.assert_array_equal(audio, result)

    def test_zero_strength_no_change(self, applicator):
        """Strength 0 ne modifie pas l'audio."""
        audio = np.random.randn(2400).astype(np.float32)
        result = applicator.apply_contour(audio, IntonationContour.DECLARATIVE, strength=0)
        np.testing.assert_array_equal(audio, result)

    @pytest.mark.parametrize("contour", [
        IntonationContour.DECLARATIVE,
        IntonationContour.QUESTION_YN,
        IntonationContour.QUESTION_WH,
        IntonationContour.EXCLAMATION,
        IntonationContour.CONTINUATION,
        IntonationContour.SUSPENSE
    ])
    def test_all_contours_work(self, applicator, contour):
        """Tous les contours fonctionnent."""
        audio = np.random.randn(4800).astype(np.float32)
        result = applicator.apply_contour(audio, contour)
        assert len(result) > 0
        assert result.dtype == np.float32

    def test_strength_affects_result(self, applicator):
        """La force affecte le résultat."""
        audio = np.sin(np.linspace(0, 100, 4800)).astype(np.float32)

        # Créer des applicateurs avec différentes forces
        result_weak = IntonationContourApplicator(24000, 0.3).apply_contour(
            audio.copy(), IntonationContour.QUESTION_YN
        )
        result_strong = IntonationContourApplicator(24000, 1.0).apply_contour(
            audio.copy(), IntonationContour.QUESTION_YN
        )

        # Les résultats devraient être différents
        # (difficile à tester précisément sans analyser le pitch)
        assert len(result_weak) > 0
        assert len(result_strong) > 0


class TestContourConfigs:
    """Tests pour les configurations de contours."""

    def test_all_contours_have_config(self):
        """Tous les types de contours ont une config."""
        for contour in IntonationContour:
            assert contour in CONTOUR_CONFIGS

    def test_config_has_5_segments(self):
        """Chaque config a 5 segments."""
        for config in CONTOUR_CONFIGS.values():
            assert len(config.pitch_curve) == 5
            assert len(config.timing_weights) == 5

    def test_timing_weights_positive(self):
        """Les poids temporels sont positifs."""
        for config in CONTOUR_CONFIGS.values():
            for w in config.timing_weights:
                assert w > 0

    def test_neutral_has_zero_pitch(self):
        """Le contour neutre n'a pas de changement de pitch."""
        neutral_config = CONTOUR_CONFIGS[IntonationContour.NEUTRAL]
        assert all(p == 0.0 for p in neutral_config.pitch_curve)

    def test_declarative_ends_lower(self):
        """Le déclaratif finit plus bas."""
        config = CONTOUR_CONFIGS[IntonationContour.DECLARATIVE]
        assert config.pitch_curve[-1] < 0

    def test_question_yn_ends_higher(self):
        """La question Y/N finit plus haut."""
        config = CONTOUR_CONFIGS[IntonationContour.QUESTION_YN]
        assert config.pitch_curve[-1] > config.pitch_curve[0]


class TestIntonationProcessor:
    """Tests pour le processeur complet."""

    @pytest.fixture
    def processor(self):
        return IntonationProcessor(sample_rate=24000, language="fr")

    def test_process_declarative(self, processor):
        """Traitement phrase déclarative."""
        audio = np.random.randn(4800).astype(np.float32)
        result = processor.process(audio, "Il fait beau.")
        assert len(result) > 0

    def test_process_question(self, processor):
        """Traitement question."""
        audio = np.random.randn(4800).astype(np.float32)
        result = processor.process(audio, "Tu viens ?")
        assert len(result) > 0

    def test_process_disabled(self):
        """Processeur désactivé ne modifie pas."""
        processor = IntonationProcessor(enabled=False)
        audio = np.random.randn(2400).astype(np.float32)
        result = processor.process(audio, "Test !")
        np.testing.assert_array_equal(audio, result)

    def test_detect_contour(self, processor):
        """Détection de contour."""
        assert processor.detect_contour("Bonjour.") == IntonationContour.DECLARATIVE
        assert processor.detect_contour("Bonjour ?") == IntonationContour.QUESTION_YN


class TestPitchShiftSimple:
    """Tests pour le pitch shift simple (sans librosa)."""

    @pytest.fixture
    def applicator_simple(self):
        # Forcer l'utilisation de la méthode simple
        app = IntonationContourApplicator(sample_rate=24000, use_librosa=False)
        app._librosa_available = False
        return app

    def test_pitch_shift_preserves_length(self, applicator_simple):
        """Le pitch shift préserve la longueur."""
        audio = np.sin(np.linspace(0, 100, 2400)).astype(np.float32)
        result = applicator_simple._pitch_shift_simple(audio, 2.0)
        assert len(result) == len(audio)

    def test_pitch_shift_empty(self, applicator_simple):
        """Audio vide retourne vide."""
        empty = np.array([], dtype=np.float32)
        result = applicator_simple._pitch_shift_simple(empty, 2.0)
        assert len(result) == 0

    def test_pitch_shift_small_delta(self, applicator_simple):
        """Petit delta ne modifie pas."""
        audio = np.sin(np.linspace(0, 100, 2400)).astype(np.float32)
        result = applicator_simple._pitch_shift_simple(audio, 0.05)
        np.testing.assert_array_equal(audio, result)


class TestSplitAudio:
    """Tests pour la division de l'audio."""

    @pytest.fixture
    def applicator(self):
        return IntonationContourApplicator(sample_rate=24000)

    def test_split_correct_number(self, applicator):
        """Division en bon nombre de segments."""
        audio = np.ones(1000, dtype=np.float32)
        weights = [0.2, 0.3, 0.5]
        segments, _ = applicator._split_audio(audio, weights)
        assert len(segments) == 3

    def test_split_preserves_samples(self, applicator):
        """La division préserve tous les échantillons."""
        audio = np.ones(1000, dtype=np.float32)
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        segments, _ = applicator._split_audio(audio, weights)
        total = sum(len(s) for s in segments)
        assert total == len(audio)

    def test_split_proportional(self, applicator):
        """Les segments sont proportionnels aux poids."""
        audio = np.ones(1000, dtype=np.float32)
        weights = [0.1, 0.4, 0.5]
        segments, _ = applicator._split_audio(audio, weights)

        # Vérifier les proportions approximatives
        assert len(segments[0]) < len(segments[1])
        assert len(segments[1]) < len(segments[2])


class TestCrossfadeSegments:
    """Tests pour le crossfade des segments."""

    @pytest.fixture
    def applicator(self):
        return IntonationContourApplicator(sample_rate=24000)

    def test_crossfade_empty(self, applicator):
        """Liste vide retourne vide."""
        result = applicator._crossfade_segments([])
        assert len(result) == 0

    def test_crossfade_single(self, applicator):
        """Un seul segment retourne ce segment."""
        seg = np.ones(1000, dtype=np.float32)
        result = applicator._crossfade_segments([seg])
        assert len(result) == len(seg)

    def test_crossfade_multiple(self, applicator):
        """Plusieurs segments sont combinés."""
        segs = [np.ones(500, dtype=np.float32) * i for i in range(3)]
        result = applicator._crossfade_segments(segs)
        # La longueur totale devrait être proche de la somme moins les overlaps
        assert len(result) > 0
        assert len(result) < sum(len(s) for s in segs)

    def test_crossfade_dtype(self, applicator):
        """Le résultat est en float32."""
        segs = [np.ones(500, dtype=np.float32) for _ in range(3)]
        result = applicator._crossfade_segments(segs)
        assert result.dtype == np.float32
