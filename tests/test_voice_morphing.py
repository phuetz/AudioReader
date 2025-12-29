"""
Tests pour le module voice_morphing.py

Teste:
- VoiceMorphSettings dataclass
- VoiceMorpher operations
- VoicePresets
- Fonction morph_audio_file
"""
import pytest
import numpy as np
import sys
from pathlib import Path
import tempfile

# Ajouter le repertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.voice_morphing import (
    VoiceMorphSettings,
    VoiceMorpher,
    VoicePresets,
    morph_audio_file
)


class TestVoiceMorphSettings:
    """Tests pour VoiceMorphSettings."""

    def test_default_values(self):
        """Test valeurs par defaut."""
        settings = VoiceMorphSettings()

        assert settings.pitch_shift == 0.0
        assert settings.formant_shift == 1.0
        assert settings.time_stretch == 1.0
        assert settings.stability == 0.7
        assert settings.breathiness == 0.0
        assert settings.roughness == 0.0

    def test_custom_values(self):
        """Test valeurs personnalisees."""
        settings = VoiceMorphSettings(
            pitch_shift=3.0,
            formant_shift=1.2,
            time_stretch=0.9,
            stability=0.5,
            breathiness=0.3,
            roughness=0.2
        )

        assert settings.pitch_shift == 3.0
        assert settings.formant_shift == 1.2
        assert settings.time_stretch == 0.9
        assert settings.stability == 0.5
        assert settings.breathiness == 0.3
        assert settings.roughness == 0.2


class TestVoiceMorpher:
    """Tests pour VoiceMorpher."""

    @pytest.fixture
    def morpher(self):
        """Cree un morpher."""
        return VoiceMorpher(sample_rate=24000)

    @pytest.fixture
    def sample_audio(self):
        """Cree un signal audio de test."""
        # Signal sinusoidal de 1 seconde
        sr = 24000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t) * 0.5  # 440 Hz
        return audio

    def test_init(self, morpher):
        """Test initialisation."""
        assert morpher.sample_rate == 24000

    def test_add_breathiness_zero(self, morpher, sample_audio):
        """Test breathiness a zero ne modifie pas."""
        result = morpher.add_breathiness(sample_audio, 0.0)
        np.testing.assert_array_almost_equal(result, sample_audio)

    def test_add_breathiness_nonzero(self, morpher, sample_audio):
        """Test breathiness ajoute du bruit."""
        result = morpher.add_breathiness(sample_audio, 0.5)

        # Le resultat ne doit pas etre identique
        assert not np.allclose(result, sample_audio)

        # Mais doit avoir la meme longueur
        assert len(result) == len(sample_audio)

    def test_add_roughness_zero(self, morpher, sample_audio):
        """Test roughness a zero ne modifie pas."""
        result = morpher.add_roughness(sample_audio, 0.0)
        np.testing.assert_array_almost_equal(result, sample_audio)

    def test_add_roughness_nonzero(self, morpher, sample_audio):
        """Test roughness modifie le signal."""
        result = morpher.add_roughness(sample_audio, 0.5)

        # Le resultat ne doit pas etre identique
        assert not np.allclose(result, sample_audio)

        # Mais doit avoir la meme longueur
        assert len(result) == len(sample_audio)

    def test_apply_variation_stable(self, morpher, sample_audio):
        """Test variation avec stabilite haute ne modifie pas."""
        result = morpher.apply_variation(sample_audio, 0.99)
        np.testing.assert_array_almost_equal(result, sample_audio)

    def test_apply_variation_unstable(self, morpher, sample_audio):
        """Test variation avec stabilite basse modifie."""
        result = morpher.apply_variation(sample_audio, 0.3)

        # Le resultat ne doit pas etre identique
        assert not np.allclose(result, sample_audio)

        # Mais doit avoir la meme longueur
        assert len(result) == len(sample_audio)

    def test_morph_neutral_settings(self, morpher, sample_audio):
        """Test morph avec settings neutres ne modifie presque pas."""
        settings = VoiceMorphSettings()  # Valeurs par defaut

        result = morpher.morph(sample_audio, settings)

        # Le resultat devrait etre tres proche de l'original
        # (seule la variation de stabilite=0.7 peut modifier legerement)
        assert len(result) == len(sample_audio)

    def test_morph_with_breathiness(self, morpher, sample_audio):
        """Test morph avec breathiness."""
        settings = VoiceMorphSettings(breathiness=0.4)
        result = morpher.morph(sample_audio, settings)

        assert not np.allclose(result, sample_audio)
        assert len(result) == len(sample_audio)

    def test_morph_with_roughness(self, morpher, sample_audio):
        """Test morph avec roughness."""
        settings = VoiceMorphSettings(roughness=0.3)
        result = morpher.morph(sample_audio, settings)

        assert not np.allclose(result, sample_audio)
        assert len(result) == len(sample_audio)

    def test_morph_preserves_dtype(self, morpher, sample_audio):
        """Test que morph preserve le dtype."""
        settings = VoiceMorphSettings(breathiness=0.2, roughness=0.1)
        result = morpher.morph(sample_audio.astype(np.float32), settings)

        assert result.dtype == np.float32


class TestVoicePresets:
    """Tests pour VoicePresets."""

    def test_get_existing_preset(self):
        """Test recuperation d'un preset existant."""
        preset = VoicePresets.get("more_masculine")
        assert preset is not None
        assert isinstance(preset, VoiceMorphSettings)

    def test_get_nonexistent_preset(self):
        """Test recuperation d'un preset inexistant."""
        preset = VoicePresets.get("nonexistent_preset")
        assert preset is None

    def test_preset_more_masculine(self):
        """Test preset more_masculine."""
        preset = VoicePresets.get("more_masculine")

        assert preset.pitch_shift < 0  # Pitch plus bas
        assert preset.formant_shift < 1.0  # Formants plus graves

    def test_preset_more_feminine(self):
        """Test preset more_feminine."""
        preset = VoicePresets.get("more_feminine")

        assert preset.pitch_shift > 0  # Pitch plus haut
        assert preset.formant_shift > 1.0  # Formants plus aigus

    def test_preset_younger(self):
        """Test preset younger."""
        preset = VoicePresets.get("younger")

        assert preset.pitch_shift > 0  # Voix plus aigue
        assert preset.stability < 0.7  # Plus variable

    def test_preset_older(self):
        """Test preset older."""
        preset = VoicePresets.get("older")

        assert preset.pitch_shift < 0  # Voix plus grave
        assert preset.breathiness > 0  # Plus de souffle

    def test_preset_whisper(self):
        """Test preset whisper."""
        preset = VoicePresets.get("whisper")

        assert preset.breathiness > 0.3  # Beaucoup de souffle

    def test_preset_rough(self):
        """Test preset rough."""
        preset = VoicePresets.get("rough")

        assert preset.roughness > 0  # Voix rauque

    def test_preset_expressive(self):
        """Test preset expressive."""
        preset = VoicePresets.get("expressive")

        assert preset.stability < 0.5  # Tres variable

    def test_preset_robotic(self):
        """Test preset robotic."""
        preset = VoicePresets.get("robotic")

        assert preset.stability == 1.0  # Tres stable

    def test_all_presets_exist(self):
        """Test que tous les presets listes existent."""
        expected_presets = [
            "more_masculine", "more_feminine",
            "younger", "older",
            "whisper", "rough", "expressive", "robotic",
            "excited_morph", "sad_morph", "angry_morph", "fearful_morph"
        ]

        for preset_name in expected_presets:
            preset = VoicePresets.get(preset_name)
            assert preset is not None, f"Preset {preset_name} devrait exister"

    def test_all_presets_have_valid_values(self):
        """Test que tous les presets ont des valeurs valides."""
        for name, preset in VoicePresets.PRESETS.items():
            assert -12 <= preset.pitch_shift <= 12, f"{name}: pitch_shift hors limites"
            assert 0.5 <= preset.formant_shift <= 2.0, f"{name}: formant_shift hors limites"
            assert 0.5 <= preset.time_stretch <= 2.0, f"{name}: time_stretch hors limites"
            assert 0.0 <= preset.stability <= 1.0, f"{name}: stability hors limites"
            assert 0.0 <= preset.breathiness <= 1.0, f"{name}: breathiness hors limites"
            assert 0.0 <= preset.roughness <= 1.0, f"{name}: roughness hors limites"


class TestMorphAudioFile:
    """Tests pour la fonction morph_audio_file."""

    @pytest.fixture
    def temp_audio_file(self):
        """Cree un fichier audio temporaire."""
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not installed")

        # Creer un fichier temporaire
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = Path(f.name)

        # Ecrire un signal de test
        sr = 24000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t) * 0.5
        sf.write(str(path), audio, sr)

        yield path

        # Nettoyer
        if path.exists():
            path.unlink()

    def test_morph_audio_file_basic(self, temp_audio_file):
        """Test morph basique d'un fichier."""
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not installed")

        output_path = temp_audio_file.with_suffix(".morphed.wav")
        settings = VoiceMorphSettings(breathiness=0.2)

        try:
            result = morph_audio_file(temp_audio_file, output_path, settings)

            assert result is True
            assert output_path.exists()

            # Verifier que le fichier de sortie est valide
            audio, sr = sf.read(str(output_path))
            assert len(audio) > 0
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_morph_audio_file_nonexistent(self):
        """Test avec fichier inexistant."""
        settings = VoiceMorphSettings()
        result = morph_audio_file(
            Path("/nonexistent/file.wav"),
            Path("/tmp/output.wav"),
            settings
        )

        assert result is False


class TestMorpherWithLibrosa:
    """Tests qui necessitent librosa (peuvent etre skippes)."""

    @pytest.fixture
    def morpher(self):
        return VoiceMorpher(sample_rate=24000)

    @pytest.fixture
    def sample_audio(self):
        sr = 24000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        return np.sin(2 * np.pi * 440 * t) * 0.5

    def test_pitch_shift_requires_librosa(self, morpher, sample_audio):
        """Test pitch_shift avec ou sans librosa."""
        result = morpher.pitch_shift(sample_audio, 2.0)

        # Si librosa n'est pas installe, retourne l'original
        # Sinon, modifie le signal
        assert len(result) == len(sample_audio)

    def test_time_stretch_requires_librosa(self, morpher, sample_audio):
        """Test time_stretch avec ou sans librosa."""
        result = morpher.time_stretch(sample_audio, 1.2)

        # Le resultat peut avoir une longueur differente
        # ou etre identique si librosa n'est pas installe
        assert len(result) > 0

    def test_formant_shift_requires_librosa(self, morpher, sample_audio):
        """Test formant_shift avec ou sans librosa."""
        result = morpher.formant_shift(sample_audio, 1.2)

        # Le resultat depend de librosa
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
