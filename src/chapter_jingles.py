"""
Jingles inter-chapitres pour AudioReader.

Genere ou charge des transitions sonores entre les chapitres.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict
from pathlib import Path


@dataclass
class JinglePreset:
    """Preset de jingle."""
    name: str
    description: str
    duration_s: float
    generated: bool = True  # True = synthetise, False = depuis fichier


JINGLE_PRESETS: Dict[str, JinglePreset] = {
    "chime": JinglePreset("chime", "Carillon doux (synthetise)", 2.0),
    "page_turn": JinglePreset("page_turn", "Bruit de page tournee", 1.0),
    "orchestral": JinglePreset("orchestral", "Transition orchestrale legere", 3.0),
    "minimal": JinglePreset("minimal", "Bip subtil et silence", 1.5),
    "silence": JinglePreset("silence", "Silence pur (separation)", 2.0),
}


class ChapterJingleGenerator:
    """
    Generateur de jingles inter-chapitres.

    Peut generer des jingles synthetiques ou charger
    des fichiers audio personnalises.
    """

    def __init__(self, sample_rate: int = 24000, custom_dir: Optional[Path] = None):
        self.sample_rate = sample_rate
        self.custom_dir = custom_dir

    def _generate_chime(self, duration_s: float = 2.0) -> np.ndarray:
        """Genere un carillon doux."""
        n_samples = int(duration_s * self.sample_rate)
        t = np.arange(n_samples) / self.sample_rate

        # Frequences harmoniques (accord majeur)
        freqs = [523.25, 659.25, 783.99]  # C5, E5, G5
        signal = np.zeros(n_samples, dtype=np.float32)

        for i, freq in enumerate(freqs):
            delay = i * 0.15  # Decalage temporel
            env_t = np.maximum(t - delay, 0)
            envelope = np.exp(-3.0 * env_t) * np.where(t >= delay, 1, 0)
            signal += 0.2 * envelope * np.sin(2 * np.pi * freq * t)

        # Fade out global
        fade_len = int(0.5 * self.sample_rate)
        if fade_len > 0 and fade_len < n_samples:
            fade = np.linspace(1, 0, fade_len)
            signal[-fade_len:] *= fade

        return signal

    def _generate_minimal(self, duration_s: float = 1.5) -> np.ndarray:
        """Genere un bip subtil suivi de silence."""
        n_samples = int(duration_s * self.sample_rate)
        signal = np.zeros(n_samples, dtype=np.float32)

        # Bip court
        bip_duration = 0.1
        bip_samples = int(bip_duration * self.sample_rate)
        t = np.arange(bip_samples) / self.sample_rate
        envelope = np.exp(-10 * t)
        bip = 0.15 * envelope * np.sin(2 * np.pi * 880 * t)  # A5
        signal[:bip_samples] = bip

        return signal

    def _generate_page_turn(self, duration_s: float = 1.0) -> np.ndarray:
        """Genere un son de page qui tourne (bruit filtre)."""
        n_samples = int(duration_s * self.sample_rate)

        # Bruit blanc avec envelope rapide
        rng = np.random.RandomState(42)
        noise = rng.randn(n_samples).astype(np.float32)

        # Envelope : attack rapide, decay moyen
        t = np.arange(n_samples) / self.sample_rate
        envelope = np.exp(-8 * t) * np.minimum(t * 50, 1.0)

        # Filtre passe-bande (approximation)
        kernel_size = 8
        kernel = np.ones(kernel_size) / kernel_size
        filtered = np.convolve(noise, kernel, mode="same")

        signal = 0.1 * envelope * filtered

        # Silence apres
        silence = np.zeros(int(0.3 * self.sample_rate), dtype=np.float32)
        signal = np.concatenate([signal, silence])[:n_samples]

        return signal.astype(np.float32)

    def _generate_orchestral(self, duration_s: float = 3.0) -> np.ndarray:
        """Genere une transition orchestrale legere."""
        n_samples = int(duration_s * self.sample_rate)
        t = np.arange(n_samples) / self.sample_rate

        # Accord avec violon synthetique (harmoniques riches)
        signal = np.zeros(n_samples, dtype=np.float32)
        base_freq = 261.63  # C4

        for harmonic in range(1, 6):
            freq = base_freq * harmonic
            amplitude = 0.15 / harmonic
            # Envelope douce
            envelope = np.sin(np.pi * t / duration_s) ** 2
            signal += amplitude * envelope * np.sin(2 * np.pi * freq * t)

        return signal

    def _generate_silence(self, duration_s: float = 2.0) -> np.ndarray:
        """Genere du silence pur."""
        return np.zeros(int(duration_s * self.sample_rate), dtype=np.float32)

    def generate(self, preset_name: str) -> np.ndarray:
        """
        Genere un jingle a partir d'un preset.

        Args:
            preset_name: Nom du preset

        Returns:
            Signal audio du jingle
        """
        # Verifier si un fichier personnalise existe
        if self.custom_dir:
            custom_file = self.custom_dir / f"{preset_name}.wav"
            if custom_file.exists():
                import soundfile as sf
                audio, sr = sf.read(str(custom_file))
                if sr != self.sample_rate:
                    # Resample simple
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                return audio.astype(np.float32)

        preset = JINGLE_PRESETS.get(preset_name)
        if preset is None:
            raise ValueError(f"Preset inconnu: {preset_name}. Disponibles: {list(JINGLE_PRESETS.keys())}")

        generators = {
            "chime": self._generate_chime,
            "page_turn": self._generate_page_turn,
            "orchestral": self._generate_orchestral,
            "minimal": self._generate_minimal,
            "silence": self._generate_silence,
        }

        gen_fn = generators.get(preset_name, self._generate_silence)
        return gen_fn(preset.duration_s)

    @staticmethod
    def list_presets() -> list:
        """Liste les presets disponibles."""
        return [
            {"name": p.name, "description": p.description, "duration_s": p.duration_s}
            for p in JINGLE_PRESETS.values()
        ]
