"""
Moteur d'effets sonores procéduraux pour audiobooks.

Génère des effets sonores contextuels (whoosh, impact, suspense, etc.)
pour améliorer l'immersion sans nécessiter de fichiers audio externes.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SoundEffectConfig:
    """Configuration d'un effet sonore."""
    type: str  # 'sweep', 'burst', 'drone', 'harmonic', 'fade'
    freq_start: float = 200.0
    freq_end: float = 800.0
    noise_type: str = 'pink'  # 'white', 'pink', 'brown'
    decay: float = 0.5
    sweep: bool = False
    modulation: float = 0.0
    shimmer: bool = False


# Effets prédéfinis
BUILTIN_EFFECTS = {
    'whoosh': SoundEffectConfig(
        type='sweep',
        freq_start=800,
        freq_end=3000,
        noise_type='pink',
        decay=0.3,
    ),
    'whoosh_reverse': SoundEffectConfig(
        type='sweep',
        freq_start=3000,
        freq_end=800,
        noise_type='pink',
        decay=0.3,
    ),
    'impact': SoundEffectConfig(
        type='burst',
        noise_type='brown',
        decay=0.15,
    ),
    'impact_soft': SoundEffectConfig(
        type='burst',
        noise_type='brown',
        decay=0.3,
    ),
    'magic': SoundEffectConfig(
        type='harmonic',
        freq_start=440,
        freq_end=880,
        sweep=True,
        shimmer=True,
        decay=0.8,
    ),
    'transition': SoundEffectConfig(
        type='fade',
        decay=0.5,
    ),
    'suspense': SoundEffectConfig(
        type='drone',
        freq_start=80,
        modulation=0.1,
        decay=2.0,
    ),
    'tension': SoundEffectConfig(
        type='drone',
        freq_start=120,
        modulation=0.2,
        decay=1.5,
    ),
    'reveal': SoundEffectConfig(
        type='harmonic',
        freq_start=220,
        freq_end=440,
        sweep=True,
        decay=0.6,
    ),
    'page_turn': SoundEffectConfig(
        type='burst',
        noise_type='white',
        decay=0.08,
        freq_start=2000,
        freq_end=8000,
    ),
    'chapter_start': SoundEffectConfig(
        type='harmonic',
        freq_start=330,  # Mi
        freq_end=440,    # La
        sweep=True,
        shimmer=True,
        decay=1.2,
    ),
    'chapter_end': SoundEffectConfig(
        type='harmonic',
        freq_start=440,  # La
        freq_end=330,    # Mi
        sweep=True,
        decay=1.5,
    ),
}


class SoundFXEngine:
    """
    Moteur de génération procédurale d'effets sonores.

    Génère des sons sans fichiers audio externes en utilisant
    des techniques de synthèse (bruit filtré, ondes, modulation).
    """

    def __init__(self, sample_rate: int = 24000):
        """
        Initialise le moteur d'effets sonores.

        Args:
            sample_rate: Taux d'échantillonnage (Hz)
        """
        self.sample_rate = sample_rate

    def generate(
        self,
        effect: str,
        intensity: float = 0.5,
        duration: float = 0.5,
    ) -> np.ndarray:
        """
        Génère un effet sonore.

        Args:
            effect: Nom de l'effet ('whoosh', 'impact', 'suspense', etc.)
            intensity: Intensité de l'effet (0.0-1.0)
            duration: Durée en secondes

        Returns:
            Array numpy avec l'audio généré
        """
        if effect not in BUILTIN_EFFECTS:
            # Effet inconnu, retourner du silence
            return np.zeros(int(duration * self.sample_rate), dtype=np.float32)

        config = BUILTIN_EFFECTS[effect]

        if config.type == 'sweep':
            audio = self._generate_sweep(config, intensity, duration)
        elif config.type == 'burst':
            audio = self._generate_burst(config, intensity, duration)
        elif config.type == 'drone':
            audio = self._generate_drone(config, intensity, duration)
        elif config.type == 'harmonic':
            audio = self._generate_harmonic(config, intensity, duration)
        elif config.type == 'fade':
            audio = self._generate_fade(config, intensity, duration)
        else:
            audio = np.zeros(int(duration * self.sample_rate), dtype=np.float32)

        return audio.astype(np.float32)

    def _generate_sweep(
        self,
        config: SoundEffectConfig,
        intensity: float,
        duration: float,
    ) -> np.ndarray:
        """
        Génère un effet sweep (whoosh).

        Bruit filtré avec balayage fréquentiel.
        """
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, 1, num_samples)

        # Générer le bruit de base
        noise = self._generate_noise(num_samples, config.noise_type)

        # Enveloppe (forme de cosinus carré pour un son doux)
        envelope = np.sin(np.pi * t) ** 2

        # Filtre passe-bande qui se déplace
        freq = config.freq_start + (config.freq_end - config.freq_start) * t
        bandwidth = 500  # Hz

        # Appliquer un filtre simple par multiplication fréquentielle
        filtered = self._apply_moving_bandpass(noise, freq, bandwidth)

        return filtered * envelope * intensity * 0.3

    def _generate_burst(
        self,
        config: SoundEffectConfig,
        intensity: float,
        duration: float,
    ) -> np.ndarray:
        """
        Génère un effet burst (impact).

        Bruit avec attaque rapide et decay exponentiel.
        """
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, 1, num_samples)

        # Bruit de base
        noise = self._generate_noise(num_samples, config.noise_type)

        # Enveloppe avec decay exponentiel
        attack_samples = int(0.01 * self.sample_rate)  # 10ms attack
        envelope = np.zeros(num_samples)
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        envelope[attack_samples:] = np.exp(-5 * t[attack_samples:] / config.decay)

        # Filtre passe-bas pour adoucir
        if config.noise_type == 'brown':
            filtered = self._lowpass_filter(noise, 1000)
        else:
            filtered = self._lowpass_filter(noise, 4000)

        return filtered * envelope * intensity * 0.4

    def _generate_drone(
        self,
        config: SoundEffectConfig,
        intensity: float,
        duration: float,
    ) -> np.ndarray:
        """
        Génère un effet drone (suspense).

        Onde basse fréquence avec modulation lente.
        """
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)

        # Fréquence de base avec légère modulation
        freq = config.freq_start
        modulation = 1 + config.modulation * np.sin(2 * np.pi * 0.5 * t)

        # Onde sinusoïdale principale
        wave = np.sin(2 * np.pi * freq * t * modulation)

        # Ajouter des harmoniques pour richesse
        wave += 0.3 * np.sin(2 * np.pi * freq * 2 * t * modulation)
        wave += 0.1 * np.sin(2 * np.pi * freq * 3 * t * modulation)

        # Enveloppe fade in/out
        fade_samples = int(0.2 * self.sample_rate)
        envelope = np.ones(num_samples)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

        return wave * envelope * intensity * 0.15

    def _generate_harmonic(
        self,
        config: SoundEffectConfig,
        intensity: float,
        duration: float,
    ) -> np.ndarray:
        """
        Génère un effet harmonique (magic, reveal).

        Sons musicaux avec harmoniques et shimmer optionnel.
        """
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)

        # Fréquence avec sweep optionnel
        if config.sweep:
            freq = np.linspace(config.freq_start, config.freq_end, num_samples)
        else:
            freq = np.full(num_samples, config.freq_start)

        # Générer les harmoniques
        wave = np.sin(2 * np.pi * np.cumsum(freq / self.sample_rate))
        wave += 0.5 * np.sin(2 * np.pi * np.cumsum(freq * 2 / self.sample_rate))
        wave += 0.25 * np.sin(2 * np.pi * np.cumsum(freq * 3 / self.sample_rate))

        # Shimmer (modulation d'amplitude rapide)
        if config.shimmer:
            shimmer_freq = 8  # Hz
            shimmer = 1 + 0.2 * np.sin(2 * np.pi * shimmer_freq * t)
            wave *= shimmer

        # Enveloppe ADSR simplifiée
        attack = int(0.05 * self.sample_rate)
        release = int(0.3 * self.sample_rate)
        envelope = np.ones(num_samples)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)

        return wave * envelope * intensity * 0.2

    def _generate_fade(
        self,
        config: SoundEffectConfig,
        intensity: float,
        duration: float,
    ) -> np.ndarray:
        """
        Génère un effet de transition (fade).

        Bruit très subtil pour transitions.
        """
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, 1, num_samples)

        noise = self._generate_noise(num_samples, 'pink')

        # Enveloppe triangulaire
        envelope = 1 - 2 * np.abs(t - 0.5)

        # Filtre très passe-bas
        filtered = self._lowpass_filter(noise, 500)

        return filtered * envelope * intensity * 0.05

    def _generate_noise(self, num_samples: int, noise_type: str) -> np.ndarray:
        """Génère du bruit de différents types."""
        if noise_type == 'white':
            return np.random.randn(num_samples)
        elif noise_type == 'pink':
            return self._pink_noise(num_samples)
        elif noise_type == 'brown':
            return self._brown_noise(num_samples)
        else:
            return np.random.randn(num_samples)

    def _pink_noise(self, num_samples: int) -> np.ndarray:
        """Génère du bruit rose (1/f spectrum)."""
        white = np.random.randn(num_samples)

        # Filtre IIR simple pour approximer le bruit rose
        b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
        a = np.array([1, -2.494956002, 2.017265875, -0.522189400])

        # Appliquer le filtre
        from scipy import signal
        try:
            pink = signal.lfilter(b, a, white)
        except Exception:
            # Fallback si scipy non disponible
            pink = white * 0.7
            for i in range(1, len(pink)):
                pink[i] = 0.7 * pink[i] + 0.3 * pink[i-1]

        return pink / (np.max(np.abs(pink)) + 1e-8)

    def _brown_noise(self, num_samples: int) -> np.ndarray:
        """Génère du bruit brun (intégration du bruit blanc)."""
        white = np.random.randn(num_samples)
        brown = np.cumsum(white)
        # Normaliser
        brown = brown - np.mean(brown)
        return brown / (np.max(np.abs(brown)) + 1e-8)

    def _lowpass_filter(self, signal: np.ndarray, cutoff: float) -> np.ndarray:
        """Applique un filtre passe-bas simple."""
        # Coefficient de lissage basé sur la fréquence de coupure
        rc = 1.0 / (2 * np.pi * cutoff)
        dt = 1.0 / self.sample_rate
        alpha = dt / (rc + dt)

        filtered = np.zeros_like(signal)
        filtered[0] = signal[0]
        for i in range(1, len(signal)):
            filtered[i] = alpha * signal[i] + (1 - alpha) * filtered[i-1]

        return filtered

    def _apply_moving_bandpass(
        self,
        signal: np.ndarray,
        center_freq: np.ndarray,
        bandwidth: float,
    ) -> np.ndarray:
        """
        Applique un filtre passe-bande qui se déplace.

        Note: Implémentation simplifiée pour la performance.
        """
        # Pour simplifier, on utilise une modulation en anneau
        t = np.arange(len(signal)) / self.sample_rate
        modulator = np.sin(2 * np.pi * center_freq * t)

        # Appliquer le filtre passe-bas après modulation
        filtered = self._lowpass_filter(signal * modulator, bandwidth)

        return filtered

    def get_available_effects(self) -> list[str]:
        """Retourne la liste des effets disponibles."""
        return list(BUILTIN_EFFECTS.keys())

    def get_effect_info(self, effect: str) -> Optional[dict]:
        """Retourne les informations sur un effet."""
        if effect not in BUILTIN_EFFECTS:
            return None

        config = BUILTIN_EFFECTS[effect]
        return {
            'name': effect,
            'type': config.type,
            'default_duration': config.decay,
        }


def apply_sound_effects_to_audio(
    audio: np.ndarray,
    sample_rate: int,
    effects_map: dict[float, tuple[str, float]],
) -> np.ndarray:
    """
    Applique des effets sonores à des positions spécifiques dans l'audio.

    Args:
        audio: Audio source
        sample_rate: Taux d'échantillonnage
        effects_map: Dict {timestamp_sec: (effect_name, intensity)}

    Returns:
        Audio avec effets mixés
    """
    engine = SoundFXEngine(sample_rate)
    result = audio.copy()

    for timestamp, (effect_name, intensity) in effects_map.items():
        # Générer l'effet
        fx = engine.generate(effect_name, intensity=intensity)

        # Position en samples
        start_sample = int(timestamp * sample_rate)

        # Mixer (attention aux limites)
        end_sample = min(start_sample + len(fx), len(result))
        fx_length = end_sample - start_sample

        if fx_length > 0:
            result[start_sample:end_sample] += fx[:fx_length]

    # Normaliser si nécessaire
    max_val = np.max(np.abs(result))
    if max_val > 1.0:
        result = result / max_val

    return result


if __name__ == "__main__":
    # Test du moteur d'effets sonores
    import soundfile as sf

    print("=== Test SoundFX Engine ===\n")

    engine = SoundFXEngine(sample_rate=24000)

    print("Effets disponibles:")
    for effect in engine.get_available_effects():
        info = engine.get_effect_info(effect)
        print(f"  - {effect}: {info['type']}")

    print("\nGénération des effets de test...")

    # Générer chaque effet
    all_audio = []
    silence = np.zeros(int(0.5 * 24000), dtype=np.float32)

    for effect in engine.get_available_effects():
        audio = engine.generate(effect, intensity=0.7, duration=1.0)
        all_audio.append(audio)
        all_audio.append(silence)
        print(f"  {effect}: {len(audio)} samples ({len(audio)/24000:.2f}s)")

    # Concaténer et sauvegarder
    full_audio = np.concatenate(all_audio)
    sf.write("test_soundfx.wav", full_audio, 24000)
    print(f"\nFichier de test créé: test_soundfx.wav ({len(full_audio)/24000:.2f}s)")
