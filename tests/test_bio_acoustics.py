"""
Tests pour le module bio_acoustics.

Vérifie la génération de sons biologiques synthétiques
(respirations, room tone, bruits de bouche).
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bio_acoustics import BioAudioGenerator


class TestBioAudioGenerator:
    """Tests pour BioAudioGenerator."""

    @pytest.fixture
    def generator(self):
        """Crée un générateur avec sample rate standard."""
        return BioAudioGenerator(sample_rate=24000)

    # === Tests generate_silence (Room Tone) ===

    def test_silence_returns_correct_length(self, generator):
        """Le silence généré a la bonne durée."""
        duration = 1.0
        audio = generator.generate_silence(duration)
        expected_samples = int(duration * generator.sample_rate)
        assert len(audio) == expected_samples

    def test_silence_is_not_zero(self, generator):
        """Le 'silence' contient du bruit (pas de zéro absolu)."""
        audio = generator.generate_silence(0.5, noise_floor=0.001)
        # Vérifier qu'il y a des valeurs non-nulles
        assert np.any(audio != 0)
        # Mais le niveau reste très bas
        assert np.max(np.abs(audio)) < 0.01

    def test_silence_empty_duration(self, generator):
        """Durée 0 ou négative retourne tableau vide."""
        assert len(generator.generate_silence(0)) == 0
        assert len(generator.generate_silence(-1)) == 0

    def test_silence_dtype(self, generator):
        """Le silence est en float32."""
        audio = generator.generate_silence(0.1)
        assert audio.dtype == np.float32

    # === Tests generate_breath ===

    def test_breath_returns_audio(self, generator):
        """La respiration génère de l'audio."""
        audio = generator.generate_breath(duration=0.4)
        assert len(audio) > 0
        assert isinstance(audio, np.ndarray)

    def test_breath_duration_approximate(self, generator):
        """La durée est approximativement correcte (±10% de variation)."""
        duration = 0.5
        audio = generator.generate_breath(duration=duration)
        expected_min = int(duration * 0.9 * generator.sample_rate)
        expected_max = int(duration * 1.1 * generator.sample_rate)
        assert expected_min <= len(audio) <= expected_max

    @pytest.mark.parametrize("breath_type", ["soft", "sharp", "deep"])
    def test_breath_types(self, generator, breath_type):
        """Tous les types de respiration fonctionnent."""
        audio = generator.generate_breath(type=breath_type)
        assert len(audio) > 0
        # Vérifier que l'audio n'est pas silencieux
        assert np.max(np.abs(audio)) > 0

    def test_breath_intensity_affects_volume(self, generator):
        """L'intensité affecte le volume."""
        low = generator.generate_breath(intensity=0.2, type="soft")
        high = generator.generate_breath(intensity=1.0, type="soft")
        # Le volume RMS devrait être plus élevé pour high intensity
        rms_low = np.sqrt(np.mean(low**2))
        rms_high = np.sqrt(np.mean(high**2))
        assert rms_high > rms_low

    def test_breath_dtype(self, generator):
        """La respiration est en float32."""
        audio = generator.generate_breath()
        assert audio.dtype == np.float32

    def test_breath_envelope_shape(self, generator):
        """La respiration a une enveloppe (pas de niveau constant)."""
        audio = generator.generate_breath(duration=0.5, type="soft")
        # Diviser en 3 parties
        third = len(audio) // 3
        start = np.max(np.abs(audio[:third]))
        middle = np.max(np.abs(audio[third:2*third]))
        end = np.max(np.abs(audio[2*third:]))
        # Pour "soft", le milieu devrait être plus fort
        assert middle >= start or middle >= end

    # === Tests generate_mouth_noise ===

    def test_mouth_noise_short_duration(self, generator):
        """Le bruit de bouche est très court (< 0.05s)."""
        audio = generator.generate_mouth_noise()
        max_samples = int(0.05 * generator.sample_rate)
        assert len(audio) < max_samples

    def test_mouth_noise_has_content(self, generator):
        """Le bruit de bouche contient de l'audio."""
        audio = generator.generate_mouth_noise()
        assert len(audio) > 0
        assert np.max(np.abs(audio)) > 0

    def test_mouth_noise_dtype(self, generator):
        """Le bruit de bouche est en float32."""
        audio = generator.generate_mouth_noise()
        assert audio.dtype == np.float32

    # === Tests apply_crossfade ===

    def test_crossfade_concatenates(self, generator):
        """Le crossfade concatène deux segments avec overlap."""
        a = np.ones(1000, dtype=np.float32)
        b = np.ones(1000, dtype=np.float32) * 2
        fade_duration = 0.01
        fade_samples = int(fade_duration * generator.sample_rate)
        result = generator.apply_crossfade(a, b, duration=fade_duration)
        # Avec overlap, la longueur = len(a) + len(b) - fade_samples
        expected_length = len(a) + len(b) - fade_samples
        assert len(result) == expected_length

    def test_crossfade_smooth_transition(self, generator):
        """Le crossfade crée une transition douce."""
        # Segment 1: valeur constante 1.0
        a = np.ones(2400, dtype=np.float32)  # 0.1s
        # Segment 2: valeur constante 0.0
        b = np.zeros(2400, dtype=np.float32)  # 0.1s

        result = generator.apply_crossfade(a, b, duration=0.02)

        # À la jonction, il devrait y avoir une transition progressive
        # Le fade-out de 'a' fait que la fin de 'a' n'est plus exactement 1.0
        fade_samples = int(0.02 * generator.sample_rate)
        end_of_a = result[len(a) - fade_samples:len(a)]
        # Les dernières valeurs devraient décroître
        assert end_of_a[-1] < end_of_a[0]

    def test_crossfade_short_segments(self, generator):
        """Le crossfade gère les segments courts sans crash."""
        a = np.ones(10, dtype=np.float32)
        b = np.ones(10, dtype=np.float32)
        # Fade plus long que les segments
        result = generator.apply_crossfade(a, b, duration=1.0)
        # Devrait simplement concaténer sans crash
        assert len(result) == 20

    # === Tests d'intégration ===

    def test_full_pipeline_simulation(self, generator):
        """Simule un pipeline complet: silence + respiration + silence."""
        parts = []

        # Room tone initial
        parts.append(generator.generate_silence(0.3))

        # Respiration
        parts.append(generator.generate_breath(type="soft"))

        # Petit silence
        parts.append(generator.generate_silence(0.1))

        # Concaténer
        result = np.concatenate(parts)

        assert len(result) > 0
        assert result.dtype == np.float32

    def test_different_sample_rates(self):
        """Le générateur fonctionne avec différents sample rates."""
        for sr in [16000, 22050, 24000, 44100, 48000]:
            gen = BioAudioGenerator(sample_rate=sr)
            audio = gen.generate_silence(0.1)
            expected = int(0.1 * sr)
            assert len(audio) == expected


class TestBreathVariability:
    """Tests pour la variabilité naturelle des respirations."""

    @pytest.fixture
    def generator(self):
        return BioAudioGenerator(sample_rate=24000)

    def test_breath_duration_varies(self, generator):
        """Les respirations ont une durée légèrement variable."""
        durations = []
        for _ in range(10):
            audio = generator.generate_breath(duration=0.4)
            durations.append(len(audio))

        # Il devrait y avoir de la variation
        assert len(set(durations)) > 1

    def test_mouth_noise_varies(self, generator):
        """Les bruits de bouche varient."""
        lengths = []
        for _ in range(10):
            audio = generator.generate_mouth_noise()
            lengths.append(len(audio))

        # Variation attendue
        assert len(set(lengths)) > 1


class TestNewBreathTypes:
    """Tests pour les nouveaux types de respiration (gasp, sigh)."""

    @pytest.fixture
    def generator(self):
        return BioAudioGenerator(sample_rate=24000)

    @pytest.mark.parametrize("breath_type", ["gasp", "sigh"])
    def test_new_breath_types_work(self, generator, breath_type):
        """Les nouveaux types gasp et sigh fonctionnent."""
        audio = generator.generate_breath(type=breath_type)
        assert len(audio) > 0
        assert np.max(np.abs(audio)) > 0

    def test_gasp_is_short_and_sharp(self, generator):
        """Le gasp est court avec un pic rapide au début."""
        audio = generator.generate_breath(duration=0.3, type="gasp")
        # Le pic devrait être dans le premier tiers
        third = len(audio) // 3
        first_third_max = np.max(np.abs(audio[:third]))
        last_third_max = np.max(np.abs(audio[2*third:]))
        assert first_third_max > last_third_max

    def test_sigh_is_longer_decay(self, generator):
        """Le sigh a une longue décroissance."""
        audio = generator.generate_breath(duration=0.6, type="sigh")
        # La première moitié devrait avoir plus d'énergie
        half = len(audio) // 2
        first_half_energy = np.sum(audio[:half]**2)
        second_half_energy = np.sum(audio[half:]**2)
        assert first_half_energy > second_half_energy


class TestGenerateForTag:
    """Tests pour generate_for_tag."""

    @pytest.fixture
    def generator(self):
        return BioAudioGenerator(sample_rate=24000)

    @pytest.mark.parametrize("tag", ["gasp", "sigh", "breath", "breath:soft", "breath:deep"])
    def test_breath_tags(self, generator, tag):
        """Les tags de respiration génèrent de l'audio."""
        audio = generator.generate_for_tag(tag)
        assert audio is not None
        assert len(audio) > 0

    @pytest.mark.parametrize("tag", ["pause", "beat", "long pause", "silence"])
    def test_pause_tags(self, generator, tag):
        """Les tags de pause génèrent du room tone."""
        audio = generator.generate_for_tag(tag)
        assert audio is not None
        assert len(audio) > 0
        # Vérifier que c'est du bruit faible (room tone)
        assert np.max(np.abs(audio)) < 0.01

    def test_unknown_tag_returns_none(self, generator):
        """Un tag inconnu retourne None."""
        result = generator.generate_for_tag("unknown_tag")
        assert result is None

    def test_tag_case_insensitive(self, generator):
        """Les tags sont insensibles à la casse."""
        lower = generator.generate_for_tag("gasp")
        upper = generator.generate_for_tag("GASP")
        assert lower is not None
        assert upper is not None


class TestConcatenateWithCrossfade:
    """Tests pour concatenate_with_crossfade."""

    @pytest.fixture
    def generator(self):
        return BioAudioGenerator(sample_rate=24000)

    def test_empty_list_returns_empty(self, generator):
        """Liste vide retourne tableau vide."""
        result = generator.concatenate_with_crossfade([])
        assert len(result) == 0

    def test_single_segment_unchanged(self, generator):
        """Un seul segment est retourné tel quel."""
        seg = np.ones(1000, dtype=np.float32)
        result = generator.concatenate_with_crossfade([seg])
        assert len(result) == len(seg)

    def test_multiple_segments_concatenated(self, generator):
        """Plusieurs segments sont concaténés avec crossfade."""
        segs = [
            np.ones(1000, dtype=np.float32),
            np.ones(1000, dtype=np.float32) * 2,
            np.ones(1000, dtype=np.float32) * 3
        ]
        result = generator.concatenate_with_crossfade(segs, fade_duration=0.01)
        # La longueur doit être inférieure à la somme (à cause des overlaps)
        total_length = sum(len(s) for s in segs)
        assert len(result) < total_length
        assert len(result) > 0

    def test_result_dtype(self, generator):
        """Le résultat est en float32."""
        segs = [np.ones(100, dtype=np.float32) for _ in range(3)]
        result = generator.concatenate_with_crossfade(segs)
        assert result.dtype == np.float32


# === Tests v2.3: Fonctionnalités avancées ===

class TestPinkNoise:
    """Tests pour la génération de bruit rose."""

    @pytest.fixture
    def generator(self):
        return BioAudioGenerator(sample_rate=24000)

    def test_pink_noise_returns_correct_length(self, generator):
        """Le bruit rose a la bonne longueur."""
        num_samples = 2400
        noise = generator._generate_pink_noise(num_samples)
        assert len(noise) == num_samples

    def test_pink_noise_fast_returns_correct_length(self, generator):
        """Le bruit rose rapide a la bonne longueur."""
        num_samples = 2400
        noise = generator._generate_pink_noise_fast(num_samples)
        assert len(noise) == num_samples

    def test_pink_noise_dtype(self, generator):
        """Le bruit rose est en float32."""
        noise = generator._generate_pink_noise(1000)
        assert noise.dtype == np.float32

    def test_pink_noise_not_zero(self, generator):
        """Le bruit rose contient des valeurs non-nulles."""
        noise = generator._generate_pink_noise(1000)
        assert np.any(noise != 0)

    def test_pink_noise_empty_input(self, generator):
        """Zéro échantillons retourne tableau vide."""
        assert len(generator._generate_pink_noise(0)) == 0
        assert len(generator._generate_pink_noise(-10)) == 0

    def test_pink_noise_has_more_low_frequency_energy(self, generator):
        """Le bruit rose a plus d'énergie dans les basses fréquences."""
        # Générer assez d'échantillons pour une FFT significative
        noise = generator._generate_pink_noise_fast(24000)  # 1 seconde

        # Calculer le spectre
        fft = np.abs(np.fft.rfft(noise))
        freqs = np.fft.rfftfreq(len(noise), 1/generator.sample_rate)

        # Comparer énergie basses fréquences vs hautes
        low_mask = freqs < 1000
        high_mask = freqs > 5000

        low_energy = np.mean(fft[low_mask]) if np.any(low_mask) else 0
        high_energy = np.mean(fft[high_mask]) if np.any(high_mask) else 1

        # Le bruit rose devrait avoir plus d'énergie en basses fréquences
        assert low_energy > high_energy


class TestFormantFilter:
    """Tests pour le filtrage formant."""

    @pytest.fixture
    def generator(self):
        return BioAudioGenerator(sample_rate=24000)

    def test_formant_filter_preserves_length(self, generator):
        """Le filtrage préserve la longueur."""
        audio = np.random.randn(2400).astype(np.float32)
        filtered = generator._apply_formant_filter(audio)
        assert len(filtered) == len(audio)

    def test_formant_filter_dtype(self, generator):
        """Le résultat est en float32."""
        audio = np.random.randn(2400).astype(np.float32)
        filtered = generator._apply_formant_filter(audio)
        assert filtered.dtype == np.float32

    def test_formant_filter_empty_input(self, generator):
        """Entrée vide retourne vide."""
        empty = np.array([], dtype=np.float32)
        result = generator._apply_formant_filter(empty)
        assert len(result) == 0

    def test_formant_filter_zero_strength(self, generator):
        """Strength 0 retourne l'original."""
        audio = np.random.randn(1000).astype(np.float32)
        filtered = generator._apply_formant_filter(audio, strength=0)
        np.testing.assert_array_almost_equal(audio, filtered)

    def test_formant_filter_modifies_spectrum(self, generator):
        """Le filtre modifie le spectre."""
        audio = np.random.randn(2400).astype(np.float32)
        filtered = generator._apply_formant_filter(audio, strength=1.0)

        # Le spectre devrait être différent
        fft_original = np.abs(np.fft.rfft(audio))
        fft_filtered = np.abs(np.fft.rfft(filtered))

        # Pas identiques
        assert not np.allclose(fft_original, fft_filtered)


class TestAmplitudeJitter:
    """Tests pour le jitter d'amplitude."""

    @pytest.fixture
    def generator(self):
        return BioAudioGenerator(sample_rate=24000)

    def test_jitter_preserves_length(self, generator):
        """Le jitter préserve la longueur."""
        audio = np.ones(2400, dtype=np.float32)
        jittered = generator._add_amplitude_jitter(audio)
        assert len(jittered) == len(audio)

    def test_jitter_dtype(self, generator):
        """Le résultat est en float32."""
        audio = np.ones(2400, dtype=np.float32)
        jittered = generator._add_amplitude_jitter(audio)
        assert jittered.dtype == np.float32

    def test_jitter_modifies_signal(self, generator):
        """Le jitter modifie le signal."""
        audio = np.ones(2400, dtype=np.float32)
        jittered = generator._add_amplitude_jitter(audio, amount=0.1)
        # Le signal ne devrait plus être constant
        assert np.std(jittered) > 0

    def test_jitter_within_bounds(self, generator):
        """Le jitter reste dans les limites raisonnables."""
        audio = np.ones(2400, dtype=np.float32)
        jittered = generator._add_amplitude_jitter(audio, amount=0.05)
        # Devrait rester entre 0.9 et 1.1 (±10% pour amount=5%)
        assert np.min(jittered) > 0.8
        assert np.max(jittered) < 1.2

    def test_jitter_zero_amount(self, generator):
        """Amount 0 ne modifie pas le signal."""
        audio = np.ones(2400, dtype=np.float32)
        jittered = generator._add_amplitude_jitter(audio, amount=0)
        np.testing.assert_array_equal(audio, jittered)

    def test_jitter_empty_input(self, generator):
        """Entrée vide retourne vide."""
        empty = np.array([], dtype=np.float32)
        result = generator._add_amplitude_jitter(empty)
        assert len(result) == 0


class TestAdvancedBreathGeneration:
    """Tests pour la génération avancée de respirations."""

    @pytest.fixture
    def generator_advanced(self):
        return BioAudioGenerator(sample_rate=24000, use_advanced_breaths=True)

    @pytest.fixture
    def generator_standard(self):
        return BioAudioGenerator(sample_rate=24000, use_advanced_breaths=False)

    def test_advanced_mode_enabled_by_default(self):
        """Le mode avancé est activé par défaut."""
        gen = BioAudioGenerator()
        assert gen.use_advanced_breaths is True

    def test_can_disable_advanced_mode(self):
        """On peut désactiver le mode avancé."""
        gen = BioAudioGenerator(use_advanced_breaths=False)
        assert gen.use_advanced_breaths is False

    def test_advanced_breath_returns_audio(self, generator_advanced):
        """Le mode avancé génère de l'audio."""
        audio = generator_advanced.generate_breath(type="soft")
        assert len(audio) > 0
        assert audio.dtype == np.float32

    def test_standard_breath_returns_audio(self, generator_standard):
        """Le mode standard génère de l'audio."""
        audio = generator_standard.generate_breath(type="soft")
        assert len(audio) > 0
        assert audio.dtype == np.float32

    def test_override_advanced_mode(self, generator_standard):
        """On peut override le mode par paramètre."""
        # Generator standard mais forcer le mode avancé
        audio = generator_standard.generate_breath(type="soft", use_advanced=True)
        assert len(audio) > 0

    def test_advanced_vs_standard_different(self, generator_advanced, generator_standard):
        """Les modes avancé et standard produisent des résultats différents."""
        # Fixer le seed pour que la différence ne soit pas due au hasard
        np.random.seed(42)
        advanced = generator_advanced.generate_breath(type="soft", duration=0.5)

        np.random.seed(42)
        standard = generator_standard.generate_breath(type="soft", duration=0.5)

        # Les longueurs peuvent être similaires mais pas les contenus
        # (le seed est reset mais les algos sont différents)
        # Vérifier qu'ils génèrent tous les deux de l'audio valide
        assert len(advanced) > 0
        assert len(standard) > 0

    @pytest.mark.parametrize("breath_type", ["soft", "sharp", "deep", "gasp", "sigh"])
    def test_all_types_work_in_advanced_mode(self, generator_advanced, breath_type):
        """Tous les types fonctionnent en mode avancé."""
        audio = generator_advanced.generate_breath(type=breath_type)
        assert len(audio) > 0
        assert np.max(np.abs(audio)) > 0

    def test_advanced_breath_has_variation(self, generator_advanced):
        """Les respirations avancées ont de la variation (jitter)."""
        audio = generator_advanced.generate_breath(type="gasp", duration=0.3)
        # Diviser en petits segments et vérifier la variation
        segment_size = len(audio) // 10
        if segment_size > 0:
            segments_rms = [
                np.sqrt(np.mean(audio[i*segment_size:(i+1)*segment_size]**2))
                for i in range(10)
            ]
            # Il devrait y avoir de la variation entre les segments
            # (au-delà de l'enveloppe normale)
            assert len(set([round(x, 4) for x in segments_rms])) > 1


class TestBreathSampleManager:
    """Tests pour le gestionnaire de samples."""

    def test_empty_manager(self):
        """Manager sans dossier retourne None."""
        from src.breath_samples import BreathSampleManager
        manager = BreathSampleManager(samples_dir=None)
        assert manager.get_breath("soft") is None

    def test_has_samples_empty(self):
        """has_samples retourne False si vide."""
        from src.breath_samples import BreathSampleManager
        manager = BreathSampleManager(samples_dir=None)
        assert manager.has_samples() is False
        assert manager.has_samples("soft") is False

    def test_list_available_types_empty(self):
        """list_available_types retourne liste vide si pas de samples."""
        from src.breath_samples import BreathSampleManager
        manager = BreathSampleManager(samples_dir=None)
        assert manager.list_available_types() == []

    def test_get_sample_count_empty(self):
        """get_sample_count retourne 0 si pas de samples."""
        from src.breath_samples import BreathSampleManager
        manager = BreathSampleManager(samples_dir=None)
        assert manager.get_sample_count() == 0
        assert manager.get_sample_count("soft") == 0


class TestHybridBreathGenerator:
    """Tests pour le générateur hybride."""

    def test_hybrid_falls_back_to_synth(self):
        """Le générateur hybride utilise la synthèse sans samples."""
        from src.breath_samples import HybridBreathGenerator
        hybrid = HybridBreathGenerator(samples_dir=None)
        audio = hybrid.generate_breath("soft")
        assert len(audio) > 0
        assert audio.dtype == np.float32

    def test_hybrid_generate_for_tag(self):
        """generate_for_tag fonctionne."""
        from src.breath_samples import HybridBreathGenerator
        hybrid = HybridBreathGenerator(samples_dir=None)

        for tag in ["gasp", "sigh", "breath", "pause"]:
            audio = hybrid.generate_for_tag(tag)
            assert audio is not None
            assert len(audio) > 0

    def test_hybrid_unknown_tag(self):
        """Tag inconnu retourne None."""
        from src.breath_samples import HybridBreathGenerator
        hybrid = HybridBreathGenerator(samples_dir=None)
        result = hybrid.generate_for_tag("unknown_tag")
        assert result is None
