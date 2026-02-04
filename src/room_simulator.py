"""
Simulation acoustique de piece pour AudioReader.

Ajoute de la reverb et du warmth pour simuler differents
environnements d'enregistrement.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class RoomPreset:
    """Preset de simulation de piece."""
    name: str
    description: str
    reverb_decay: float  # secondes
    reverb_mix: float    # 0-1, proportion de reverb
    warmth: float        # dB de boost basses frequences
    early_reflections: int  # nombre de reflexions precoces
    room_size: float     # 0-1 (petit -> grand)


ROOM_PRESETS: Dict[str, RoomPreset] = {
    "studio": RoomPreset(
        name="studio",
        description="Studio d'enregistrement professionnel (sec)",
        reverb_decay=0.2,
        reverb_mix=0.05,
        warmth=1.0,
        early_reflections=3,
        room_size=0.2,
    ),
    "living_room": RoomPreset(
        name="living_room",
        description="Salon confortable",
        reverb_decay=0.5,
        reverb_mix=0.12,
        warmth=2.0,
        early_reflections=5,
        room_size=0.4,
    ),
    "theater": RoomPreset(
        name="theater",
        description="Salle de theatre (spacieuse)",
        reverb_decay=1.2,
        reverb_mix=0.20,
        warmth=1.5,
        early_reflections=8,
        room_size=0.8,
    ),
    "intimate": RoomPreset(
        name="intimate",
        description="Piece intime et feutree",
        reverb_decay=0.3,
        reverb_mix=0.08,
        warmth=3.0,
        early_reflections=2,
        room_size=0.15,
    ),
}


class RoomSimulator:
    """
    Simule l'acoustique de differentes pieces.

    Utilise une reverb a convolution simplifiee et un EQ basses
    frequences pour simuler differents environnements.
    """

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate

    def _generate_impulse_response(self, preset: RoomPreset) -> np.ndarray:
        """Genere une reponse impulsionnelle synthetique."""
        duration = preset.reverb_decay
        n_samples = int(duration * self.sample_rate)

        # Impulsion initiale
        ir = np.zeros(n_samples, dtype=np.float32)
        ir[0] = 1.0

        # Reflexions precoces
        rng = np.random.RandomState(42)
        for i in range(preset.early_reflections):
            delay = int((i + 1) * preset.room_size * 0.01 * self.sample_rate)
            if delay < n_samples:
                amplitude = 0.6 ** (i + 1)
                ir[delay] += amplitude * (1 if rng.random() > 0.5 else -1)

        # Queue de reverb (decay exponentiel avec bruit)
        decay_env = np.exp(-3.0 * np.arange(n_samples) / n_samples)
        noise = rng.randn(n_samples).astype(np.float32) * 0.1
        ir += noise * decay_env

        # Normaliser
        max_val = np.max(np.abs(ir))
        if max_val > 0:
            ir = ir / max_val

        return ir

    def _apply_warmth(self, audio: np.ndarray, warmth_db: float) -> np.ndarray:
        """Applique un boost de basses frequences (warmth)."""
        if warmth_db <= 0:
            return audio

        # Filtre passe-bas simple pour extraire les basses
        cutoff = 300  # Hz
        dt = 1.0 / self.sample_rate
        rc = 1.0 / (2 * np.pi * cutoff)
        alpha = dt / (rc + dt)

        lows = np.zeros_like(audio)
        lows[0] = alpha * audio[0]
        for i in range(1, len(audio)):
            lows[i] = lows[i-1] + alpha * (audio[i] - lows[i-1])

        # Boost les basses
        gain = 10 ** (warmth_db / 20.0)
        boosted_lows = lows * (gain - 1.0)

        return (audio + boosted_lows).astype(np.float32)

    def process(
        self,
        audio: np.ndarray,
        preset_name: str = "studio",
    ) -> np.ndarray:
        """
        Applique la simulation de piece a l'audio.

        Args:
            audio: Signal audio d'entree
            preset_name: Nom du preset de piece

        Returns:
            Audio avec simulation de piece appliquee
        """
        preset = ROOM_PRESETS.get(preset_name)
        if preset is None:
            raise ValueError(f"Preset inconnu: {preset_name}. Disponibles: {list(ROOM_PRESETS.keys())}")

        # Generer la reponse impulsionnelle
        ir = self._generate_impulse_response(preset)

        # Convolution (reverb)
        reverb = np.convolve(audio, ir, mode="full")[:len(audio)]

        # Mixer dry/wet
        wet = preset.reverb_mix
        result = audio * (1 - wet) + reverb * wet

        # Appliquer le warmth
        result = self._apply_warmth(result, preset.warmth)

        # Normaliser pour eviter le clipping
        max_val = np.max(np.abs(result))
        if max_val > 0.95:
            result = result * (0.95 / max_val)

        return result.astype(np.float32)

    @staticmethod
    def list_presets() -> list:
        """Liste les presets disponibles."""
        return [
            {"name": p.name, "description": p.description}
            for p in ROOM_PRESETS.values()
        ]
