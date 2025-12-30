"""
Tests pour le module dynamic_voice.

Vérifie le mélange dynamique de voix basé sur les émotions.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dynamic_voice import DynamicVoiceManager, DynamicVoiceConfig
from src.emotion_analyzer import Emotion, Intensity


class TestDynamicVoiceConfig:
    """Tests pour DynamicVoiceConfig."""

    def test_default_values(self):
        """Les valeurs par défaut sont correctes."""
        config = DynamicVoiceConfig()
        assert config.blend_voice is None
        assert config.blend_weight == 0.0
        assert config.speed_mult == 1.0
        assert config.pitch_shift == 0.0
        assert config.energy_mult == 1.0

    def test_custom_values(self):
        """Les valeurs personnalisées sont acceptées."""
        config = DynamicVoiceConfig(
            blend_voice="am_adam",
            blend_weight=0.3,
            speed_mult=1.2,
            pitch_shift=-2.0,
            energy_mult=1.5
        )
        assert config.blend_voice == "am_adam"
        assert config.blend_weight == 0.3
        assert config.speed_mult == 1.2
        assert config.pitch_shift == -2.0
        assert config.energy_mult == 1.5


class TestDynamicVoiceManager:
    """Tests pour DynamicVoiceManager."""

    @pytest.fixture
    def manager(self):
        return DynamicVoiceManager()

    # === Tests get_voice_config ===

    def test_neutral_emotion_no_blend(self, manager):
        """Émotion neutre = pas de mélange."""
        result = manager.get_voice_config("ff_siwis", Emotion.NEUTRAL, Intensity.MEDIUM)
        assert result == "ff_siwis"

    def test_anger_blends_with_adam(self, manager):
        """Colère mélange avec am_adam (voix grave)."""
        result = manager.get_voice_config("ff_siwis", Emotion.ANGER, Intensity.HIGH)
        assert "ff_siwis" in result
        assert "am_adam" in result
        # Format: base:weight,target:weight
        assert ":" in result
        assert "," in result

    def test_sadness_blends_with_bella(self, manager):
        """Tristesse mélange avec af_bella (voix douce)."""
        result = manager.get_voice_config("ff_siwis", Emotion.SADNESS, Intensity.MEDIUM)
        assert "ff_siwis" in result
        assert "af_bella" in result

    def test_fear_blends_with_sky(self, manager):
        """Peur mélange avec af_sky (voix tremblante)."""
        result = manager.get_voice_config("ff_siwis", Emotion.FEAR, Intensity.HIGH)
        assert "ff_siwis" in result
        assert "af_sky" in result

    def test_joy_blends_with_nicole(self, manager):
        """Joie mélange avec af_nicole (voix vive)."""
        result = manager.get_voice_config("ff_siwis", Emotion.JOY, Intensity.MEDIUM)
        assert "ff_siwis" in result
        assert "af_nicole" in result

    def test_excitement_blends_like_joy(self, manager):
        """Excitation mélange comme la joie."""
        result = manager.get_voice_config("ff_siwis", Emotion.EXCITEMENT, Intensity.MEDIUM)
        assert "af_nicole" in result

    def test_suspense_blends_with_michael(self, manager):
        """Suspense mélange avec am_michael (voix calme)."""
        result = manager.get_voice_config("ff_siwis", Emotion.SUSPENSE, Intensity.MEDIUM)
        assert "ff_siwis" in result
        assert "am_michael" in result

    def test_base_voice_same_as_target_no_blend(self, manager):
        """Si la voix de base = cible, pas de mélange."""
        # am_adam est la cible pour ANGER
        result = manager.get_voice_config("am_adam", Emotion.ANGER, Intensity.HIGH)
        assert result == "am_adam"

    def test_intensity_affects_weight(self, manager):
        """L'intensité affecte le poids du mélange."""
        low = manager.get_voice_config("ff_siwis", Emotion.ANGER, Intensity.LOW)
        high = manager.get_voice_config("ff_siwis", Emotion.ANGER, Intensity.EXTREME)

        # Extraire les poids
        def extract_weight(s):
            # Format: "base:w1,target:w2"
            parts = s.split(",")
            return float(parts[1].split(":")[1])

        weight_low = extract_weight(low)
        weight_high = extract_weight(high)

        # Plus d'intensité = plus de poids sur la voix cible
        assert weight_high > weight_low

    def test_weights_sum_to_one(self, manager):
        """Les poids totalisent 1.0."""
        result = manager.get_voice_config("ff_siwis", Emotion.ANGER, Intensity.HIGH)

        # Extraire les poids
        parts = result.split(",")
        w1 = float(parts[0].split(":")[1])
        w2 = float(parts[1].split(":")[1])

        assert abs((w1 + w2) - 1.0) < 0.01

    def test_format_is_correct(self, manager):
        """Le format de sortie est correct pour Kokoro."""
        result = manager.get_voice_config("ff_siwis", Emotion.FEAR, Intensity.MEDIUM)

        # Doit être "voice1:weight1,voice2:weight2"
        parts = result.split(",")
        assert len(parts) == 2

        for part in parts:
            voice, weight = part.split(":")
            assert voice  # Non vide
            float(weight)  # Doit être un nombre

    # === Tests get_prosody_modifiers ===

    def test_neutral_prosody_default(self, manager):
        """Émotion neutre = prosodie par défaut."""
        mods = manager.get_prosody_modifiers(Emotion.NEUTRAL, Intensity.MEDIUM)
        assert mods["speed"] == 1.0
        assert mods["pitch"] == 0.0

    def test_anger_increases_speed(self, manager):
        """Colère augmente la vitesse."""
        mods = manager.get_prosody_modifiers(Emotion.ANGER, Intensity.HIGH)
        assert mods["speed"] > 1.0

    def test_sadness_decreases_speed(self, manager):
        """Tristesse diminue la vitesse."""
        mods = manager.get_prosody_modifiers(Emotion.SADNESS, Intensity.HIGH)
        assert mods["speed"] < 1.0

    def test_sadness_lowers_pitch(self, manager):
        """Tristesse baisse le pitch."""
        mods = manager.get_prosody_modifiers(Emotion.SADNESS, Intensity.HIGH)
        assert mods["pitch"] < 0

    def test_fear_increases_speed_and_pitch(self, manager):
        """Peur augmente vitesse et pitch."""
        mods = manager.get_prosody_modifiers(Emotion.FEAR, Intensity.HIGH)
        assert mods["speed"] > 1.0
        assert mods["pitch"] > 0

    def test_joy_increases_speed_and_pitch(self, manager):
        """Joie augmente vitesse et pitch."""
        mods = manager.get_prosody_modifiers(Emotion.JOY, Intensity.HIGH)
        assert mods["speed"] > 1.0
        assert mods["pitch"] > 0

    def test_suspense_decreases_speed(self, manager):
        """Suspense diminue la vitesse."""
        mods = manager.get_prosody_modifiers(Emotion.SUSPENSE, Intensity.HIGH)
        assert mods["speed"] < 1.0

    def test_intensity_affects_prosody(self, manager):
        """L'intensité affecte l'amplitude des modifications."""
        low = manager.get_prosody_modifiers(Emotion.FEAR, Intensity.LOW)
        high = manager.get_prosody_modifiers(Emotion.FEAR, Intensity.EXTREME)

        # Plus d'intensité = changements plus prononcés
        assert high["speed"] > low["speed"]
        assert high["pitch"] > low["pitch"]


class TestRefVoices:
    """Tests pour les voix de référence."""

    def test_all_ref_voices_defined(self):
        """Toutes les voix de référence sont définies."""
        manager = DynamicVoiceManager()
        expected_keys = ["deep_male", "soft_female", "bright_female", "shaky_female", "calm_male"]

        for key in expected_keys:
            assert key in manager.REF_VOICES
            assert manager.REF_VOICES[key]  # Non vide


class TestEdgeCases:
    """Tests pour les cas limites."""

    @pytest.fixture
    def manager(self):
        return DynamicVoiceManager()

    def test_unknown_emotion(self, manager):
        """Une émotion inconnue retourne la voix de base."""
        # Utiliser une émotion qui n'a pas de mélange défini
        result = manager.get_voice_config("ff_siwis", Emotion.SURPRISE, Intensity.HIGH)
        # SURPRISE n'est pas dans la logique, donc retourne la voix de base
        assert result == "ff_siwis"

    def test_unknown_intensity(self, manager):
        """Une intensité inconnue utilise la valeur par défaut."""
        # Passer None ou une valeur invalide (simulé via le fallback)
        mods = manager.get_prosody_modifiers(Emotion.ANGER, None)
        # Devrait utiliser le fallback (0.5 pour intensity, 2 pour level)
        assert "speed" in mods
        assert "pitch" in mods

    @pytest.mark.parametrize("emotion", [
        Emotion.ANGER, Emotion.SADNESS, Emotion.FEAR,
        Emotion.JOY, Emotion.SUSPENSE, Emotion.NEUTRAL
    ])
    def test_all_emotions_work(self, manager, emotion):
        """Toutes les émotions principales fonctionnent."""
        result = manager.get_voice_config("ff_siwis", emotion, Intensity.MEDIUM)
        assert result  # Non vide
        assert isinstance(result, str)

    @pytest.mark.parametrize("intensity", [
        Intensity.LOW, Intensity.MEDIUM, Intensity.HIGH, Intensity.EXTREME
    ])
    def test_all_intensities_work(self, manager, intensity):
        """Toutes les intensités fonctionnent."""
        result = manager.get_voice_config("ff_siwis", Emotion.ANGER, intensity)
        assert result
        mods = manager.get_prosody_modifiers(Emotion.ANGER, intensity)
        assert mods
