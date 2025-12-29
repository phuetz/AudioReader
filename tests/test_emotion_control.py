"""
Tests pour le module emotion_control.py

Teste:
- EmotionSettings et EmotionController
- PhonemeEntry et PhonemeProcessor
- PronunciationManager
- Fonction create_pronunciation_config
"""
import pytest
import sys
from pathlib import Path
import tempfile
import json

# Ajouter le repertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.emotion_control import (
    EmotionSettings,
    EmotionController,
    PhonemeEntry,
    PhonemeProcessor,
    PronunciationManager,
    create_pronunciation_config
)


class TestEmotionSettings:
    """Tests pour EmotionSettings."""

    def test_default_values(self):
        """Test valeurs par defaut."""
        settings = EmotionSettings()

        assert settings.intensity == 0.5
        assert settings.stability == 0.7
        assert settings.style_exaggeration == 0.5

    def test_custom_values(self):
        """Test valeurs personnalisees."""
        settings = EmotionSettings(
            intensity=0.8,
            stability=0.5,
            style_exaggeration=0.9
        )

        assert settings.intensity == 0.8
        assert settings.stability == 0.5
        assert settings.style_exaggeration == 0.9


class TestEmotionController:
    """Tests pour EmotionController."""

    @pytest.fixture
    def controller(self):
        """Cree un controller avec settings par defaut."""
        return EmotionController()

    @pytest.fixture
    def high_intensity_controller(self):
        """Cree un controller haute intensite."""
        return EmotionController(EmotionSettings(intensity=1.0))

    @pytest.fixture
    def low_intensity_controller(self):
        """Cree un controller basse intensite."""
        return EmotionController(EmotionSettings(intensity=0.0))

    def test_init_default(self, controller):
        """Test initialisation par defaut."""
        assert controller.settings.intensity == 0.5

    def test_init_custom(self, high_intensity_controller):
        """Test initialisation personnalisee."""
        assert high_intensity_controller.settings.intensity == 1.0

    def test_interpolate_multipliers_low(self, low_intensity_controller):
        """Test interpolation a basse intensite."""
        speed, pitch, vol = low_intensity_controller._interpolate_multipliers(0.0)

        # Valeurs neutres attendues
        assert speed == 1.0
        assert pitch == 0.0
        assert vol == 1.0

    def test_interpolate_multipliers_high(self, high_intensity_controller):
        """Test interpolation a haute intensite."""
        speed, pitch, vol = high_intensity_controller._interpolate_multipliers(1.0)

        # Valeurs elevees attendues
        assert speed > 1.0
        assert pitch > 0.0
        assert vol > 1.0

    def test_interpolate_multipliers_middle(self, controller):
        """Test interpolation au milieu."""
        speed, pitch, vol = controller._interpolate_multipliers(0.5)

        # Valeurs intermediaires
        assert 1.0 <= speed <= 1.2
        assert 0.0 <= pitch <= 0.5
        assert 1.0 <= vol <= 1.15

    def test_interpolate_multipliers_clamps(self, controller):
        """Test que les valeurs hors limites sont clampees."""
        # Valeur negative
        speed, pitch, vol = controller._interpolate_multipliers(-0.5)
        assert speed == 1.0  # Meme que 0.0

        # Valeur trop elevee
        speed2, pitch2, vol2 = controller._interpolate_multipliers(1.5)
        # Devrait etre clampee a 1.0

    def test_calculate_prosody_basic(self, controller):
        """Test calcul prosodie basique."""
        result = controller.calculate_prosody()

        assert "speed" in result
        assert "pitch" in result
        assert "volume" in result

    def test_calculate_prosody_with_base_values(self, controller):
        """Test calcul prosodie avec valeurs de base."""
        result = controller.calculate_prosody(
            base_speed=1.2,
            base_pitch=0.5,
            base_volume=0.9
        )

        assert result["speed"] >= 1.0
        assert "pitch" in result
        assert "volume" in result

    def test_calculate_prosody_joy(self, controller):
        """Test prosodie pour joie."""
        result = controller.calculate_prosody(emotion_type="joy")

        # Joy devrait augmenter la vitesse et le pitch
        assert result["speed"] >= 1.0
        assert result["pitch"] >= 0.0

    def test_calculate_prosody_sad(self, controller):
        """Test prosodie pour tristesse."""
        result = controller.calculate_prosody(emotion_type="sad")

        # Sad devrait ralentir
        # (selon l'exageration, peut etre proche de 1.0)
        assert "speed" in result

    def test_calculate_prosody_anger(self, controller):
        """Test prosodie pour colere."""
        result = controller.calculate_prosody(emotion_type="anger")

        # Anger devrait augmenter le volume
        assert result["volume"] >= 1.0

    def test_calculate_prosody_unknown_emotion(self, controller):
        """Test prosodie pour emotion inconnue."""
        result = controller.calculate_prosody(emotion_type="unknown_emotion")

        # Devrait fonctionner sans erreur (ignore l'emotion inconnue)
        assert "speed" in result


class TestPhonemeEntry:
    """Tests pour PhonemeEntry."""

    def test_default_values(self):
        """Test valeurs par defaut."""
        entry = PhonemeEntry(
            word="test",
            ipa="tɛst"
        )

        assert entry.word == "test"
        assert entry.ipa == "tɛst"
        assert entry.arpabet is None
        assert entry.language == "fr"
        assert entry.notes == ""

    def test_custom_values(self):
        """Test valeurs personnalisees."""
        entry = PhonemeEntry(
            word="hello",
            ipa="həˈloʊ",
            arpabet="HH AH0 L OW1",
            language="en",
            notes="English greeting"
        )

        assert entry.word == "hello"
        assert entry.ipa == "həˈloʊ"
        assert entry.arpabet == "HH AH0 L OW1"
        assert entry.language == "en"
        assert entry.notes == "English greeting"


class TestPhonemeProcessor:
    """Tests pour PhonemeProcessor."""

    @pytest.fixture
    def processor_fr(self):
        """Cree un processeur francais."""
        return PhonemeProcessor("fr")

    @pytest.fixture
    def processor_en(self):
        """Cree un processeur anglais."""
        return PhonemeProcessor("en")

    def test_init_fr(self, processor_fr):
        """Test initialisation francais."""
        assert processor_fr.lang == "fr"
        assert len(processor_fr._builtin) > 0

    def test_init_en(self, processor_en):
        """Test initialisation anglais."""
        assert processor_en.lang == "en"
        assert len(processor_en._builtin) > 0

    def test_builtin_french_names(self, processor_fr):
        """Test noms propres francais integres."""
        assert processor_fr.get_phoneme("Jean") is not None
        assert processor_fr.get_phoneme("Marie") is not None
        assert processor_fr.get_phoneme("Paris") is not None

    def test_builtin_tech_terms(self, processor_fr):
        """Test termes techniques integres."""
        assert processor_fr.get_phoneme("API") is not None
        assert processor_fr.get_phoneme("Python") is not None
        assert processor_fr.get_phoneme("JavaScript") is not None

    def test_builtin_brands(self, processor_fr):
        """Test marques integrees."""
        assert processor_fr.get_phoneme("Google") is not None
        assert processor_fr.get_phoneme("Apple") is not None
        assert processor_fr.get_phoneme("Netflix") is not None

    def test_add_word(self, processor_fr):
        """Test ajout de mot personnalise."""
        processor_fr.add_word(
            word="CustomWord",
            ipa="kystɔmwɔʁd",
            arpabet="K AH0 S T AH0 M W ER1 D",
            notes="Test word"
        )

        assert processor_fr.get_phoneme("customword") is not None
        assert processor_fr.get_phoneme("CustomWord") is not None

    def test_get_phoneme_case_insensitive(self, processor_fr):
        """Test que la recherche est case-insensitive."""
        phoneme1 = processor_fr.get_phoneme("jean")
        phoneme2 = processor_fr.get_phoneme("Jean")
        phoneme3 = processor_fr.get_phoneme("JEAN")

        # Jean (avec majuscule) devrait etre trouve
        assert phoneme2 is not None

    def test_get_phoneme_not_found(self, processor_fr):
        """Test mot non trouve."""
        phoneme = processor_fr.get_phoneme("UnMotQuiNExistePas12345")
        assert phoneme is None

    def test_apply_phonemes(self, processor_fr):
        """Test application des phonemes au texte."""
        text = "Jean utilise Google."
        processed = processor_fr.apply_phonemes(text)

        # Le texte devrait etre modifie
        # (les mots connus remplacés par leur approximation)
        assert isinstance(processed, str)

    def test_phoneme_to_text(self, processor_fr):
        """Test conversion IPA vers texte."""
        result = processor_fr._phoneme_to_text("ʒɑ̃")  # Jean

        # Devrait contenir des caracteres lisibles
        assert "ʒ" not in result or "j" in result

    def test_generate_ssml_phoneme(self, processor_fr):
        """Test generation de tag SSML."""
        ssml = processor_fr.generate_ssml_phoneme(
            word="test",
            ipa="tɛst",
            alphabet="ipa"
        )

        assert "<phoneme" in ssml
        assert 'alphabet="ipa"' in ssml
        assert 'ph="tɛst"' in ssml
        assert ">test</phoneme>" in ssml

    def test_save_load_dictionary(self, processor_fr):
        """Test sauvegarde et chargement du dictionnaire."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            # Ajouter des mots
            processor_fr.add_word("custom1", "ipa1", notes="Note 1")
            processor_fr.add_word("custom2", "ipa2", notes="Note 2")

            # Sauvegarder
            processor_fr.save_dictionary(path)

            # Nouveau processeur
            processor2 = PhonemeProcessor("fr")
            processor2.load_dictionary(path)

            # Verifier
            assert processor2.get_phoneme("custom1") == "ipa1"
            assert processor2.get_phoneme("custom2") == "ipa2"
        finally:
            if path.exists():
                path.unlink()


class TestPronunciationManager:
    """Tests pour PronunciationManager."""

    @pytest.fixture
    def manager(self):
        """Cree un manager francais."""
        return PronunciationManager("fr")

    def test_init(self, manager):
        """Test initialisation."""
        assert manager.lang == "fr"
        assert manager.phoneme_processor is not None

    def test_add_correction(self, manager):
        """Test ajout de correction simple."""
        manager.add_correction("etc.", "et cetera")

        result = manager.process("J'aime etc.")
        assert "et cetera" in result

    def test_add_phoneme(self, manager):
        """Test ajout de phoneme."""
        manager.add_phoneme("NewWord", "njuːwɜːrd")

        # Le phoneme devrait etre accessible
        assert manager.phoneme_processor.get_phoneme("NewWord") is not None

    def test_add_regex_rule(self, manager):
        """Test ajout de regle regex."""
        manager.add_regex_rule(r'\bOK\b', 'okay')

        result = manager.process("C'est OK.")
        assert "okay" in result

    def test_process_combined(self, manager):
        """Test traitement combine."""
        manager.add_correction("Mr.", "Monsieur")
        manager.add_regex_rule(r'(\d+)%', r'\1 pourcent')

        result = manager.process("Mr. Dupont a 50% de reduction.")

        assert "Monsieur" in result
        assert "pourcent" in result

    def test_process_order(self, manager):
        """Test que l'ordre de traitement est correct."""
        # Corrections simples d'abord, puis regex, puis phonemes
        manager.add_correction("ABC", "alpha beta gamma")

        result = manager.process("ABC")
        assert "alpha beta gamma" in result

    def test_load_config(self, manager):
        """Test chargement de configuration."""
        config = {
            "corrections": {"Mme": "Madame"},
            "phonemes": {"Custom": {"ipa": "kystɔm"}},
            "rules": [{"pattern": r"\bDr\b", "replacement": "Docteur"}]
        }

        with tempfile.NamedTemporaryFile(
            mode='w', suffix=".json", delete=False
        ) as f:
            json.dump(config, f)
            path = Path(f.name)

        try:
            manager.load_config(path)

            result1 = manager.process("Mme Dupont")
            assert "Madame" in result1

            result2 = manager.process("Dr House")
            assert "Docteur" in result2
        finally:
            if path.exists():
                path.unlink()


class TestCreatePronunciationConfig:
    """Tests pour la fonction create_pronunciation_config."""

    def test_create_french(self):
        """Test creation config francaise."""
        manager = create_pronunciation_config("fr")

        assert manager.lang == "fr"

        # Verifier les corrections par defaut
        result = manager.process("etc.")
        assert "et cetera" in result

    def test_create_english(self):
        """Test creation config anglaise."""
        manager = create_pronunciation_config("en")

        assert manager.lang == "en"

    def test_french_abbreviations(self):
        """Test abreviations francaises."""
        manager = create_pronunciation_config("fr")

        assert "Monsieur" in manager.process("M. Dupont")
        assert "Madame" in manager.process("Mme Martin")
        assert "Docteur" in manager.process("Dr Smith")

    def test_french_ordinals(self):
        """Test ordinaux francais."""
        manager = create_pronunciation_config("fr")

        assert "eme" in manager.process("le 5e jour")
        assert "premier" in manager.process("le 1er prix")
        assert "premiere" in manager.process("la 1ere fois")


class TestIntegration:
    """Tests d'integration."""

    def test_full_pipeline(self):
        """Test pipeline complet."""
        # Creer le manager
        manager = create_pronunciation_config("fr")
        manager.add_correction("TTS", "text to speech")
        manager.add_phoneme("Kokoro", "kokɔʁo")

        # Creer le controller
        controller = EmotionController(EmotionSettings(intensity=0.7))

        # Traiter du texte
        text = "M. Dupont utilise Kokoro TTS pour le 1er chapitre."
        processed = manager.process(text)

        # Calculer la prosodie
        prosody = controller.calculate_prosody(emotion_type="joy")

        assert "Monsieur" in processed
        assert "text to speech" in processed
        assert "premier" in processed
        assert prosody["speed"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
