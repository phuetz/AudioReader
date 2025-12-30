"""
Tests pour le module timing_humanizer.

Vérifie la variation des pauses et l'humanisation du timing.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.timing_humanizer import (
    TimingHumanizer,
    TimingConfig,
    ClauseType,
    PauseCalculator,
    TextTimingProcessor,
    TimedSegment,
    EMPHASIS_WORDS_FR
)


class TestTimingConfig:
    """Tests pour TimingConfig."""

    def test_default_values(self):
        """Les valeurs par défaut sont raisonnables."""
        config = TimingConfig()
        assert config.pause_variation_sigma == 0.05
        assert config.pause_variation_min == 0.85
        assert config.pause_variation_max == 1.15
        assert config.enable_emphasis_pauses is True

    def test_custom_values(self):
        """Les valeurs personnalisées sont acceptées."""
        config = TimingConfig(
            pause_variation_sigma=0.1,
            emphasis_pause_duration=0.1
        )
        assert config.pause_variation_sigma == 0.1
        assert config.emphasis_pause_duration == 0.1


class TestTimingHumanizer:
    """Tests pour TimingHumanizer."""

    @pytest.fixture
    def humanizer(self):
        return TimingHumanizer()

    # === Tests humanize_pause ===

    def test_humanize_pause_positive(self, humanizer):
        """La pause reste positive."""
        for _ in range(100):
            result = humanizer.humanize_pause(0.5)
            assert result > 0

    def test_humanize_pause_within_bounds(self, humanizer):
        """La pause reste dans les limites."""
        base = 1.0
        config = humanizer.config

        for _ in range(100):
            result = humanizer.humanize_pause(base)
            assert result >= base * config.pause_variation_min
            assert result <= base * config.pause_variation_max

    def test_humanize_pause_zero(self, humanizer):
        """Pause 0 retourne 0."""
        assert humanizer.humanize_pause(0) == 0

    def test_humanize_pause_negative(self, humanizer):
        """Pause négative retourne négative."""
        result = humanizer.humanize_pause(-0.5)
        assert result == -0.5

    def test_humanize_pause_varies(self, humanizer):
        """Les résultats varient entre les appels."""
        results = [humanizer.humanize_pause(0.5) for _ in range(20)]
        unique = len(set(results))
        assert unique > 1  # Au moins 2 valeurs différentes

    # === Tests get_clause_type ===

    def test_clause_main_default(self, humanizer):
        """Texte normal = clause principale."""
        assert humanizer.get_clause_type("Il fait beau") == ClauseType.MAIN

    def test_clause_quote_guillemets(self, humanizer):
        """Texte avec guillemets = citation."""
        assert humanizer.get_clause_type('"Bonjour"') == ClauseType.QUOTE

    def test_clause_quote_chevrons(self, humanizer):
        """Texte avec chevrons = citation."""
        assert humanizer.get_clause_type("«Bonjour»") == ClauseType.QUOTE

    def test_clause_quote_tiret(self, humanizer):
        """Texte avec tiret dialogue = citation."""
        assert humanizer.get_clause_type("— Bonjour") == ClauseType.QUOTE

    def test_clause_parenthetical_parentheses(self, humanizer):
        """Texte entre parenthèses = incise."""
        assert humanizer.get_clause_type("(bien sûr)") == ClauseType.PARENTHETICAL

    def test_clause_subordinate_qui(self, humanizer):
        """Clause commençant par 'qui' = subordonnée."""
        assert humanizer.get_clause_type("qui est venu hier") == ClauseType.SUBORDINATE

    def test_clause_subordinate_parce_que(self, humanizer):
        """Clause commençant par 'parce que' = subordonnée."""
        assert humanizer.get_clause_type("parce que c'est vrai") == ClauseType.SUBORDINATE

    def test_clause_subordinate_quand(self, humanizer):
        """Clause commençant par 'quand' = subordonnée."""
        assert humanizer.get_clause_type("quand il arrive") == ClauseType.SUBORDINATE

    def test_clause_emphasis(self, humanizer):
        """Texte avec mot d'emphase = emphase."""
        assert humanizer.get_clause_type("c'est absolument vrai") == ClauseType.EMPHASIS

    # === Tests get_clause_speed_modifier ===

    def test_speed_main_is_one(self, humanizer):
        """Clause principale = vitesse 1.0."""
        assert humanizer.get_clause_speed_modifier(ClauseType.MAIN) == 1.0

    def test_speed_subordinate_faster(self, humanizer):
        """Clause subordonnée = légèrement plus rapide."""
        assert humanizer.get_clause_speed_modifier(ClauseType.SUBORDINATE) > 1.0

    def test_speed_parenthetical_fastest(self, humanizer):
        """Incise = plus rapide que subordonnée."""
        sub = humanizer.get_clause_speed_modifier(ClauseType.SUBORDINATE)
        par = humanizer.get_clause_speed_modifier(ClauseType.PARENTHETICAL)
        assert par > sub

    def test_speed_quote_slower(self, humanizer):
        """Citation = légèrement plus lente."""
        assert humanizer.get_clause_speed_modifier(ClauseType.QUOTE) < 1.0

    # === Tests add_emphasis_pauses ===

    def test_emphasis_pause_added(self, humanizer):
        """Une micro-pause est ajoutée avant mot d'emphase."""
        result = humanizer.add_emphasis_pauses("C'est vraiment beau")
        assert "[pause:" in result
        assert "vraiment" in result

    def test_emphasis_pause_multiple(self, humanizer):
        """Plusieurs micro-pauses pour plusieurs mots."""
        result = humanizer.add_emphasis_pauses("Jamais je n'aurais cru, mais enfin")
        # Devrait avoir des pauses avant "jamais" et "mais"
        count = result.count("[pause:")
        assert count >= 1  # Au moins une pause

    def test_emphasis_pause_no_change_without_words(self, humanizer):
        """Pas de changement si pas de mot d'emphase."""
        text = "Il mange une pomme"
        result = humanizer.add_emphasis_pauses(text)
        assert "[pause:" not in result

    def test_emphasis_pause_disabled(self):
        """Désactivation des pauses d'emphase."""
        config = TimingConfig(enable_emphasis_pauses=False)
        humanizer = TimingHumanizer(config)
        result = humanizer.add_emphasis_pauses("C'est vraiment beau")
        assert "[pause:" not in result

    # === Tests inter-phrase variation ===

    def test_inter_phrase_variation_varies(self, humanizer):
        """La variation inter-phrase change entre appels."""
        results = [humanizer.get_inter_phrase_variation() for _ in range(10)]
        unique = len(set(results))
        assert unique > 1

    def test_inter_phrase_variation_near_one(self, humanizer):
        """La variation reste proche de 1.0."""
        for _ in range(50):
            var = humanizer.get_inter_phrase_variation()
            assert 0.9 <= var <= 1.1

    def test_reset_phrase_counter(self, humanizer):
        """Reset du compteur de phrases."""
        # Avancer le compteur
        for _ in range(5):
            humanizer.get_inter_phrase_variation()

        # Reset
        humanizer.reset_phrase_counter()
        assert humanizer._phrase_counter == 0


class TestPauseCalculator:
    """Tests pour PauseCalculator."""

    @pytest.fixture
    def calculator(self):
        return PauseCalculator()

    def test_comma_pause(self, calculator):
        """Pause après virgule."""
        pause = calculator.calculate_pause(',')
        assert 0.1 <= pause <= 0.6  # Avec variation

    def test_period_pause(self, calculator):
        """Pause après point."""
        pause = calculator.calculate_pause('.')
        assert 0.3 <= pause <= 0.9  # Avec variation

    def test_period_longer_than_comma(self, calculator):
        """La pause après point est plus longue qu'après virgule."""
        # Tester sur la base
        assert calculator.base_pauses['.'] > calculator.base_pauses[',']

    def test_ellipsis_pause(self, calculator):
        """Pause après points de suspension."""
        pause = calculator.calculate_pause('...')
        assert pause > 0

    def test_unknown_punctuation(self, calculator):
        """Ponctuation inconnue = pause par défaut."""
        pause = calculator.calculate_pause('@')
        assert pause > 0

    def test_breath_pause(self, calculator):
        """Pause pour respiration."""
        pause = calculator.get_breath_pause()
        assert 0.05 <= pause <= 0.2


class TestTextTimingProcessor:
    """Tests pour TextTimingProcessor."""

    @pytest.fixture
    def processor(self):
        return TextTimingProcessor()

    def test_process_simple_sentence(self, processor):
        """Traitement d'une phrase simple."""
        segments = processor.process_text("Il fait beau.")
        assert len(segments) >= 1
        assert isinstance(segments[0], TimedSegment)

    def test_process_multiple_sentences(self, processor):
        """Traitement de plusieurs phrases."""
        segments = processor.process_text("Il fait beau. C'est l'été.")
        assert len(segments) >= 2

    def test_segment_has_timing(self, processor):
        """Les segments ont des informations de timing."""
        segments = processor.process_text("Bonjour!")
        seg = segments[0]

        assert hasattr(seg, 'text')
        assert hasattr(seg, 'pause_before')
        assert hasattr(seg, 'pause_after')
        assert hasattr(seg, 'speed_modifier')

    def test_segment_speed_modifier(self, processor):
        """Le modificateur de vitesse est proche de 1."""
        segments = processor.process_text("Bonjour.")
        for seg in segments:
            assert 0.8 <= seg.speed_modifier <= 1.2

    def test_emphasis_detection(self, processor):
        """Détection des pauses d'emphase."""
        segments = processor.process_text("C'est vraiment incroyable!")
        # Au moins un segment devrait avoir une emphase
        has_emphasis = any(seg.has_emphasis_pause for seg in segments)
        assert has_emphasis

    def test_empty_text(self, processor):
        """Texte vide retourne liste vide."""
        segments = processor.process_text("")
        assert segments == []

    def test_whitespace_text(self, processor):
        """Texte avec espaces seulement retourne liste vide."""
        segments = processor.process_text("   \n  ")
        assert segments == []


class TestEmphasisWords:
    """Tests pour la liste des mots d'emphase."""

    def test_emphasis_words_not_empty(self):
        """La liste n'est pas vide."""
        assert len(EMPHASIS_WORDS_FR) > 0

    def test_emphasis_words_lowercase(self):
        """Tous les mots sont en minuscules."""
        for word in EMPHASIS_WORDS_FR:
            assert word == word.lower()

    def test_common_emphasis_words_present(self):
        """Les mots d'emphase courants sont présents."""
        expected = ["jamais", "toujours", "vraiment", "soudain", "mais"]
        for word in expected:
            assert word in EMPHASIS_WORDS_FR


class TestClauseTypeDetection:
    """Tests supplémentaires pour la détection de clauses."""

    @pytest.fixture
    def humanizer(self):
        return TimingHumanizer()

    @pytest.mark.parametrize("marker", [
        "qui", "que", "dont", "où", "quand", "lorsque", "parce que",
        "si", "comme", "bien que", "afin que"
    ])
    def test_subordinate_markers(self, humanizer, marker):
        """Tous les marqueurs de subordonnée sont détectés."""
        text = f"{marker} cela arrive"
        result = humanizer.get_clause_type(text)
        assert result in [ClauseType.SUBORDINATE, ClauseType.EMPHASIS]

    def test_parenthetical_in_parens(self, humanizer):
        """Texte entre parenthèses = incise."""
        assert humanizer.get_clause_type("(dit-il)") == ClauseType.PARENTHETICAL

    def test_mixed_emphasis_and_subordinate(self, humanizer):
        """Priorité entre emphase et subordonnée."""
        # "qui vraiment" - commence par "qui" (subordonnée) mais contient "vraiment" (emphase)
        # Le premier test (quote) a priorité, puis parenthetical, puis subordinate
        text = "qui vraiment fait cela"
        result = humanizer.get_clause_type(text)
        # Devrait être SUBORDINATE car vérifié avant EMPHASIS
        assert result == ClauseType.SUBORDINATE
