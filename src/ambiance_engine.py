"""
Moteur de musique d'ambiance pour AudioReader.

Genere des fonds sonores proceduraux (bruit brun, pluie, etc.)
et les mixe avec la narration en utilisant du sidechain ducking.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple


@dataclass
class AmbiancePreset:
    """Preset d'ambiance sonore."""
    name: str
    description: str
    noise_type: str  # "brown", "pink", "white"
    base_volume: float = 0.05
    frequency_range: Tuple[float, float] = (20.0, 2000.0)
    modulation_speed: float = 0.1  # Hz
    modulation_depth: float = 0.3


# Presets disponibles
AMBIANCE_PRESETS: Dict[str, AmbiancePreset] = {
    "library": AmbiancePreset(
        name="library",
        description="Ambiance feutree de bibliotheque",
        noise_type="brown",
        base_volume=0.02,
        frequency_range=(20.0, 500.0),
        modulation_speed=0.05,
        modulation_depth=0.1,
    ),
    "rain": AmbiancePreset(
        name="rain",
        description="Pluie douce sur les vitres",
        noise_type="pink",
        base_volume=0.06,
        frequency_range=(200.0, 8000.0),
        modulation_speed=0.3,
        modulation_depth=0.4,
    ),
    "fireplace": AmbiancePreset(
        name="fireplace",
        description="Feu de cheminee crépitant",
        noise_type="brown",
        base_volume=0.04,
        frequency_range=(50.0, 3000.0),
        modulation_speed=0.8,
        modulation_depth=0.6,
    ),
    "cafe": AmbiancePreset(
        name="cafe",
        description="Ambiance de cafe parisien",
        noise_type="pink",
        base_volume=0.03,
        frequency_range=(100.0, 4000.0),
        modulation_speed=0.2,
        modulation_depth=0.3,
    ),
    "forest": AmbiancePreset(
        name="forest",
        description="Foret paisible avec vent leger",
        noise_type="pink",
        base_volume=0.04,
        frequency_range=(50.0, 6000.0),
        modulation_speed=0.15,
        modulation_depth=0.35,
    ),
}


class AmbianceEngine:
    """
    Moteur de generation d'ambiance sonore.

    Genere des sons d'ambiance proceduraux et les mixe
    avec la narration en utilisant le sidechain ducking.
    """

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate

    def generate_noise(self, noise_type: str, duration_s: float, seed: int = 42) -> np.ndarray:
        """Genere du bruit colore."""
        rng = np.random.RandomState(seed)
        n_samples = int(duration_s * self.sample_rate)

        if noise_type == "white":
            return rng.randn(n_samples).astype(np.float32)

        elif noise_type == "pink":
            # Approximation du bruit rose par filtrage
            white = rng.randn(n_samples).astype(np.float32)
            # Filtre passe-bas simple (moyenne mobile)
            kernel_size = 16
            kernel = np.ones(kernel_size) / kernel_size
            pink = np.convolve(white, kernel, mode="same")
            return pink.astype(np.float32)

        elif noise_type == "brown":
            # Bruit brun = integration du bruit blanc
            white = rng.randn(n_samples).astype(np.float32) * 0.02
            brown = np.cumsum(white)
            # Normaliser pour eviter la derive
            brown = brown - np.mean(brown)
            max_val = np.max(np.abs(brown))
            if max_val > 0:
                brown = brown / max_val
            return brown.astype(np.float32)

        return np.zeros(n_samples, dtype=np.float32)

    def apply_modulation(self, audio: np.ndarray, speed: float, depth: float) -> np.ndarray:
        """Applique une modulation d'amplitude lente."""
        n_samples = len(audio)
        t = np.arange(n_samples) / self.sample_rate
        modulation = 1.0 - depth * 0.5 * (1 + np.sin(2 * np.pi * speed * t))
        return (audio * modulation).astype(np.float32)

    def generate_ambiance(self, preset_name: str, duration_s: float) -> np.ndarray:
        """Genere une ambiance a partir d'un preset."""
        preset = AMBIANCE_PRESETS.get(preset_name)
        if preset is None:
            raise ValueError(f"Preset inconnu: {preset_name}. Disponibles: {list(AMBIANCE_PRESETS.keys())}")

        # Generer le bruit de base
        noise = self.generate_noise(preset.noise_type, duration_s)

        # Appliquer la modulation
        noise = self.apply_modulation(noise, preset.modulation_speed, preset.modulation_depth)

        # Appliquer le volume
        noise = noise * preset.base_volume

        return noise

    def mix(
        self,
        narration: np.ndarray,
        preset: str,
        duck_amount: float = 0.3,
        fade_in_s: float = 2.0,
        fade_out_s: float = 2.0,
    ) -> np.ndarray:
        """
        Mixe la narration avec l'ambiance en appliquant du sidechain ducking.

        Le volume de l'ambiance baisse quand la narration est active.

        Args:
            narration: Audio de narration
            preset: Nom du preset d'ambiance
            duck_amount: Reduction du volume d'ambiance pendant la parole (0-1)
            fade_in_s: Duree du fade-in de l'ambiance
            fade_out_s: Duree du fade-out de l'ambiance
        """
        duration_s = len(narration) / self.sample_rate
        ambiance = self.generate_ambiance(preset, duration_s)

        # Ajuster la longueur
        if len(ambiance) > len(narration):
            ambiance = ambiance[:len(narration)]
        elif len(ambiance) < len(narration):
            ambiance = np.pad(ambiance, (0, len(narration) - len(ambiance)))

        # Sidechain ducking : reduire l'ambiance quand la narration est active
        # Detecter l'activite vocale (envelope)
        envelope = np.abs(narration)
        # Lissage de l'envelope
        kernel_size = int(0.1 * self.sample_rate)  # 100ms
        if kernel_size > 0:
            kernel = np.ones(kernel_size) / kernel_size
            envelope = np.convolve(envelope, kernel, mode="same")

        # Normaliser l'envelope
        max_env = np.max(envelope)
        if max_env > 0:
            envelope = envelope / max_env

        # Appliquer le ducking
        duck_factor = 1.0 - duck_amount * envelope
        ambiance = ambiance * duck_factor

        # Fade in/out
        fade_in_samples = int(fade_in_s * self.sample_rate)
        fade_out_samples = int(fade_out_s * self.sample_rate)

        if fade_in_samples > 0 and fade_in_samples < len(ambiance):
            fade_in = np.linspace(0, 1, fade_in_samples)
            ambiance[:fade_in_samples] *= fade_in

        if fade_out_samples > 0 and fade_out_samples < len(ambiance):
            fade_out = np.linspace(1, 0, fade_out_samples)
            ambiance[-fade_out_samples:] *= fade_out

        # Mixer
        return (narration + ambiance).astype(np.float32)

    @staticmethod
    def list_presets() -> list:
        """Liste les presets disponibles."""
        return [
            {"name": p.name, "description": p.description}
            for p in AMBIANCE_PRESETS.values()
        ]
