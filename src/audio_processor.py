"""
Pipeline de post-processing audio pour TTS.

Basé sur recherches scientifiques (arXiv 2023-2024):
- Débruitage spectral (noisereduce)
- EQ pour clarté vocale (boost 2-5kHz)
- Compression dynamique légère
- Normalisation LUFS (EBU R128)
- Voice enhancement

Usage:
    from src.audio_processor import AudioProcessor

    processor = AudioProcessor()
    enhanced = processor.process(audio, sample_rate=24000)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy import signal


@dataclass
class ProcessorConfig:
    """Configuration du post-processing audio."""
    # Débruitage
    denoise_enabled: bool = True
    denoise_stationary: bool = True
    denoise_prop_decrease: float = 0.75

    # EQ - Clarté vocale
    eq_enabled: bool = True
    highpass_freq: int = 80          # Coupe basses fréquences
    presence_freq: int = 3000        # Boost présence
    presence_gain_db: float = 2.0    # dB
    air_freq: int = 10000            # Boost "air"
    air_gain_db: float = 1.0         # dB

    # Compression
    compression_enabled: bool = True
    comp_threshold_db: float = -18.0
    comp_ratio: float = 3.0
    comp_attack_ms: float = 10.0
    comp_release_ms: float = 100.0

    # Normalisation
    normalize_enabled: bool = True
    target_lufs: float = -16.0       # Standard podcast/audiobook

    # De-essing (réduction sibilantes)
    deess_enabled: bool = False
    deess_freq: int = 6000
    deess_threshold_db: float = -20.0


class AudioProcessor:
    """
    Pipeline de post-processing pour améliorer la qualité TTS.

    Basé sur:
    - StyleTTS2 best practices
    - EBU R128 loudness standard
    - Broadcast audio processing techniques
    """

    def __init__(self, config: Optional[ProcessorConfig] = None):
        self.config = config or ProcessorConfig()

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 24000
    ) -> np.ndarray:
        """
        Applique le pipeline complet de post-processing.

        Args:
            audio: Signal audio (numpy array float32)
            sample_rate: Fréquence d'échantillonnage

        Returns:
            Audio amélioré
        """
        # S'assurer du bon format
        audio = audio.astype(np.float32)

        # 1. Débruitage
        if self.config.denoise_enabled:
            audio = self._denoise(audio, sample_rate)

        # 2. EQ
        if self.config.eq_enabled:
            audio = self._apply_eq(audio, sample_rate)

        # 3. Compression
        if self.config.compression_enabled:
            audio = self._compress(audio, sample_rate)

        # 4. Normalisation LUFS
        if self.config.normalize_enabled:
            audio = self._normalize_loudness(audio, sample_rate)

        # 5. Limiter final (éviter clipping)
        audio = self._limit(audio)

        return audio

    def _denoise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Réduit le bruit de fond."""
        try:
            import noisereduce as nr

            # Débruitage spectral
            audio_denoised = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=self.config.denoise_stationary,
                prop_decrease=self.config.denoise_prop_decrease,
                n_fft=2048,
                hop_length=512
            )
            return audio_denoised.astype(np.float32)
        except ImportError:
            print("Warning: noisereduce non installé, débruitage ignoré")
            return audio

    def _apply_eq(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Applique l'égalisation pour clarté vocale."""

        # 1. Highpass filter (couper basses fréquences)
        nyquist = sr / 2
        hp_freq = self.config.highpass_freq / nyquist
        if hp_freq < 1.0:
            b_hp, a_hp = signal.butter(2, hp_freq, btype='high')
            audio = signal.filtfilt(b_hp, a_hp, audio).astype(np.float32)

        # 2. Boost présence (2-5 kHz) - Peak EQ
        audio = self._apply_peak_eq(
            audio, sr,
            center_freq=self.config.presence_freq,
            gain_db=self.config.presence_gain_db,
            q=1.5
        )

        # 3. Boost air (10-16 kHz) - High shelf
        audio = self._apply_high_shelf(
            audio, sr,
            cutoff_freq=self.config.air_freq,
            gain_db=self.config.air_gain_db
        )

        return audio

    def _apply_peak_eq(
        self,
        audio: np.ndarray,
        sr: int,
        center_freq: float,
        gain_db: float,
        q: float = 1.0
    ) -> np.ndarray:
        """Applique un EQ peak (boost/cut à une fréquence)."""
        if abs(gain_db) < 0.1:
            return audio

        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * center_freq / sr
        alpha = np.sin(w0) / (2 * q)

        cos_w0 = np.cos(w0)

        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A

        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1.0, a1/a0, a2/a0])

        return signal.filtfilt(b, a, audio).astype(np.float32)

    def _apply_high_shelf(
        self,
        audio: np.ndarray,
        sr: int,
        cutoff_freq: float,
        gain_db: float
    ) -> np.ndarray:
        """Applique un high shelf filter."""
        if abs(gain_db) < 0.1:
            return audio

        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * cutoff_freq / sr

        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / 2 * np.sqrt(2)

        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha

        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1.0, a1/a0, a2/a0])

        return signal.filtfilt(b, a, audio).astype(np.float32)

    def _compress(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Applique une compression dynamique légère."""
        threshold = 10 ** (self.config.comp_threshold_db / 20)
        ratio = self.config.comp_ratio

        # Attack/release en samples
        attack_samples = int(self.config.comp_attack_ms * sr / 1000)
        release_samples = int(self.config.comp_release_ms * sr / 1000)

        # Coefficients smoothing
        attack_coef = np.exp(-1.0 / attack_samples) if attack_samples > 0 else 0.0
        release_coef = np.exp(-1.0 / release_samples) if release_samples > 0 else 0.0

        # Compression
        envelope = 0.0
        output = np.zeros_like(audio)

        for i, sample in enumerate(audio):
            input_level = abs(sample)

            # Envelope follower
            if input_level > envelope:
                envelope = attack_coef * envelope + (1 - attack_coef) * input_level
            else:
                envelope = release_coef * envelope + (1 - release_coef) * input_level

            # Gain reduction
            if envelope > threshold:
                gain_reduction = threshold + (envelope - threshold) / ratio
                gain = gain_reduction / envelope if envelope > 0 else 1.0
            else:
                gain = 1.0

            output[i] = sample * gain

        return output.astype(np.float32)

    def _normalize_loudness(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Normalise selon le standard LUFS."""
        try:
            import pyloudnorm as pyln

            meter = pyln.Meter(sr)

            # Mesurer loudness actuel
            current_loudness = meter.integrated_loudness(audio)

            # Si audio trop silencieux, éviter division par zéro
            if current_loudness < -70:
                return audio

            # Calculer gain nécessaire
            gain_db = self.config.target_lufs - current_loudness
            gain_linear = 10 ** (gain_db / 20)

            # Appliquer gain avec limite pour éviter clipping
            gain_linear = min(gain_linear, 10.0)  # Max +20dB

            return (audio * gain_linear).astype(np.float32)

        except ImportError:
            print("Warning: pyloudnorm non installé, normalisation simple")
            # Normalisation peak simple
            peak = np.max(np.abs(audio))
            if peak > 0:
                target_peak = 10 ** (self.config.target_lufs / 20 + 0.5)  # Approximation
                return (audio * target_peak / peak).astype(np.float32)
            return audio

    def _limit(self, audio: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
        """Limite les peaks pour éviter le clipping."""
        return np.clip(audio, -ceiling, ceiling).astype(np.float32)


class VoiceEnhancer:
    """
    Enhancement avancé pour voix synthétique.

    Techniques basées sur:
    - Resemble Enhance methodology
    - Speech enhancement research
    """

    def __init__(self):
        self._processor = AudioProcessor()

    def enhance(
        self,
        audio: np.ndarray,
        sample_rate: int = 24000,
        style: str = "broadcast"
    ) -> np.ndarray:
        """
        Améliore la qualité vocale.

        Args:
            audio: Signal audio
            sample_rate: Fréquence d'échantillonnage
            style: "broadcast" (podcast/audiobook), "natural", "bright"

        Returns:
            Audio amélioré
        """
        # Configuration selon style
        configs = {
            "broadcast": ProcessorConfig(
                denoise_enabled=True,
                eq_enabled=True,
                presence_gain_db=2.5,
                air_gain_db=1.5,
                compression_enabled=True,
                comp_threshold_db=-16.0,
                comp_ratio=2.5,
                normalize_enabled=True,
                target_lufs=-16.0
            ),
            "natural": ProcessorConfig(
                denoise_enabled=True,
                eq_enabled=True,
                presence_gain_db=1.5,
                air_gain_db=0.5,
                compression_enabled=True,
                comp_threshold_db=-20.0,
                comp_ratio=2.0,
                normalize_enabled=True,
                target_lufs=-18.0
            ),
            "bright": ProcessorConfig(
                denoise_enabled=True,
                eq_enabled=True,
                presence_gain_db=3.0,
                air_gain_db=2.5,
                compression_enabled=True,
                comp_threshold_db=-14.0,
                comp_ratio=3.0,
                normalize_enabled=True,
                target_lufs=-14.0
            )
        }

        config = configs.get(style, configs["broadcast"])
        processor = AudioProcessor(config)

        return processor.process(audio, sample_rate)


def enhance_tts_audio(
    audio: np.ndarray,
    sample_rate: int = 24000,
    style: str = "broadcast"
) -> np.ndarray:
    """
    Fonction utilitaire pour améliorer un audio TTS.

    Args:
        audio: Signal audio numpy
        sample_rate: Fréquence d'échantillonnage
        style: "broadcast", "natural", ou "bright"

    Returns:
        Audio amélioré
    """
    enhancer = VoiceEnhancer()
    return enhancer.enhance(audio, sample_rate, style)


if __name__ == "__main__":
    import soundfile as sf
    from pathlib import Path

    print("=== Test Audio Processor ===\n")

    # Charger un fichier test
    test_file = Path("output/01_Chapitre_7_v9.wav")

    if test_file.exists():
        audio, sr = sf.read(test_file)
        print(f"Fichier: {test_file}")
        print(f"Durée: {len(audio)/sr:.1f}s")
        print(f"Sample rate: {sr}")

        # Traiter
        print("\nTraitement...")
        enhanced = enhance_tts_audio(audio, sr, style="broadcast")

        # Sauvegarder
        output_file = test_file.with_name("01_Chapitre_7_v9_enhanced.wav")
        sf.write(output_file, enhanced, sr)
        print(f"Sauvegardé: {output_file}")
    else:
        print(f"Fichier test non trouvé: {test_file}")

        # Test avec signal synthétique
        print("\nTest avec signal synthétique...")
        sr = 24000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # Voix simulée (440Hz + harmoniques + bruit)
        audio = 0.3 * np.sin(2 * np.pi * 220 * t)
        audio += 0.15 * np.sin(2 * np.pi * 440 * t)
        audio += 0.1 * np.sin(2 * np.pi * 660 * t)
        audio += 0.02 * np.random.randn(len(t))  # Bruit
        audio = audio.astype(np.float32)

        # Traiter
        enhanced = enhance_tts_audio(audio, sr)

        print(f"Original - Peak: {np.max(np.abs(audio)):.3f}")
        print(f"Enhanced - Peak: {np.max(np.abs(enhanced)):.3f}")
