"""
Post-processing audio avance pour qualite broadcast.

Fonctionnalites:
- Normalisation de loudness (EBU R128)
- De-essing (reduction des sibilantes)
- EQ adaptatif pour voix
- Compression douce
- Limitation des pics
- Ajout de room tone (ambiance legere)
- Crossfade entre segments
"""
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import subprocess
import json


@dataclass
class AudioEnhancerConfig:
    """Configuration du post-processing audio."""
    # Normalisation
    target_lufs: float = -19.0      # Cible loudness (podcast: -16 a -19)
    true_peak_limit: float = -1.5   # Limite de peak en dB

    # EQ
    highpass_freq: int = 80         # Coupe-bas pour enlever les basses indesirables
    presence_boost: float = 2.0     # Boost presence (2-4 kHz) en dB
    air_boost: float = 1.5          # Boost "air" (10-16 kHz) en dB

    # De-essing
    deess_enabled: bool = True
    deess_freq: int = 6000          # Frequence centrale des sibilantes
    deess_threshold: float = -20.0  # Seuil en dB

    # Compression
    compression_enabled: bool = True
    comp_threshold: float = -18.0   # Seuil en dB
    comp_ratio: float = 3.0         # Ratio de compression
    comp_attack: float = 10.0       # Attack en ms
    comp_release: float = 100.0     # Release en ms

    # Room tone
    room_tone_enabled: bool = True
    room_tone_level: float = -50.0  # Niveau du bruit de fond en dB

    # Crossfade
    crossfade_duration: float = 0.05  # Duree du crossfade en secondes

    # Format de sortie
    output_sample_rate: int = 44100
    output_bitrate: str = "192k"


class AudioEnhancer:
    """
    Ameliore la qualite audio pour une qualite broadcast.

    Utilise ffmpeg pour le traitement audio.
    """

    def __init__(self, config: Optional[AudioEnhancerConfig] = None):
        self.config = config or AudioEnhancerConfig()
        self._ffmpeg_available = None

    def is_available(self) -> bool:
        """Verifie si ffmpeg est disponible."""
        if self._ffmpeg_available is not None:
            return self._ffmpeg_available

        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True
            )
            self._ffmpeg_available = result.returncode == 0
        except FileNotFoundError:
            self._ffmpeg_available = False

        return self._ffmpeg_available

    def _build_filter_chain(self) -> str:
        """Construit la chaine de filtres ffmpeg."""
        filters = []

        # 1. Highpass filter (enlever les basses)
        filters.append(f"highpass=f={self.config.highpass_freq}")

        # 2. De-essing
        if self.config.deess_enabled:
            # Utiliser un filtre de bande pour reduire les sibilantes
            filters.append(
                f"equalizer=f={self.config.deess_freq}:t=q:w=2:g=-4"
            )

        # 3. EQ presence et air
        if self.config.presence_boost > 0:
            filters.append(
                f"equalizer=f=3000:t=q:w=1.5:g={self.config.presence_boost}"
            )
        if self.config.air_boost > 0:
            filters.append(
                f"equalizer=f=12000:t=q:w=2:g={self.config.air_boost}"
            )

        # 4. Compression
        if self.config.compression_enabled:
            filters.append(
                f"acompressor="
                f"threshold={self.config.comp_threshold}dB:"
                f"ratio={self.config.comp_ratio}:"
                f"attack={self.config.comp_attack}:"
                f"release={self.config.comp_release}"
            )

        # 5. Normalisation loudness (EBU R128)
        filters.append(
            f"loudnorm="
            f"I={self.config.target_lufs}:"
            f"TP={self.config.true_peak_limit}:"
            f"LRA=11"
        )

        # 6. Limiter final
        filters.append(
            f"alimiter=limit={10 ** (self.config.true_peak_limit / 20)}"
        )

        return ",".join(filters)

    def enhance_file(
        self,
        input_path: Path,
        output_path: Path,
        verbose: bool = False
    ) -> bool:
        """
        Ameliore un fichier audio.

        Args:
            input_path: Fichier d'entree
            output_path: Fichier de sortie
            verbose: Afficher les details

        Returns:
            True si succes
        """
        if not self.is_available():
            print("ERREUR: ffmpeg n'est pas installe")
            return False

        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            print(f"ERREUR: Fichier non trouve: {input_path}")
            return False

        # Construire la commande ffmpeg
        filter_chain = self._build_filter_chain()

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-af", filter_chain,
            "-ar", str(self.config.output_sample_rate),
            "-b:a", self.config.output_bitrate,
            str(output_path)
        ]

        if verbose:
            print(f"Commande: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print(f"ERREUR ffmpeg: {result.stderr}")
                return False

            return True

        except Exception as e:
            print(f"ERREUR: {e}")
            return False

    def analyze_loudness(self, audio_path: Path) -> Optional[dict]:
        """
        Analyse le loudness d'un fichier audio.

        Returns:
            Dict avec integrated, true_peak, lra ou None si erreur
        """
        if not self.is_available():
            return None

        cmd = [
            "ffmpeg", "-i", str(audio_path),
            "-af", "loudnorm=print_format=json",
            "-f", "null", "-"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            # Parser la sortie JSON
            output = result.stderr
            json_start = output.rfind('{')
            json_end = output.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = output[json_start:json_end]
                data = json.loads(json_str)
                return {
                    "integrated": float(data.get("input_i", -99)),
                    "true_peak": float(data.get("input_tp", -99)),
                    "lra": float(data.get("input_lra", 0)),
                    "threshold": float(data.get("input_thresh", -99)),
                }

        except Exception as e:
            print(f"Erreur analyse: {e}")

        return None


class SegmentCrossfader:
    """
    Applique des crossfades entre segments audio.

    Evite les clics et pops entre segments.
    """

    def __init__(self, crossfade_ms: int = 50, sample_rate: int = 24000):
        self.crossfade_samples = int(crossfade_ms * sample_rate / 1000)
        self.sample_rate = sample_rate

    def _create_fade(self, length: int, fade_in: bool = True) -> np.ndarray:
        """Cree une courbe de fade."""
        t = np.linspace(0, np.pi / 2, length)
        if fade_in:
            return np.sin(t) ** 2
        return np.cos(t) ** 2

    def crossfade_segments(
        self,
        segments: List[np.ndarray]
    ) -> np.ndarray:
        """
        Applique des crossfades entre les segments.

        Args:
            segments: Liste de numpy arrays audio

        Returns:
            Audio concatene avec crossfades
        """
        if not segments:
            return np.array([], dtype=np.float32)

        if len(segments) == 1:
            return segments[0]

        result_parts = []

        for i, segment in enumerate(segments):
            if len(segment) == 0:
                continue

            # Fade in au debut (sauf premier segment)
            if i > 0 and len(segment) > self.crossfade_samples:
                fade_in = self._create_fade(self.crossfade_samples, fade_in=True)
                segment[:self.crossfade_samples] *= fade_in

            # Fade out a la fin (sauf dernier segment)
            if i < len(segments) - 1 and len(segment) > self.crossfade_samples:
                fade_out = self._create_fade(self.crossfade_samples, fade_in=False)
                segment[-self.crossfade_samples:] *= fade_out

            result_parts.append(segment)

        return np.concatenate(result_parts)

    def apply_fade_in(self, audio: np.ndarray, duration_ms: int = 50) -> np.ndarray:
        """Applique un fade-in au debut."""
        samples = int(duration_ms * self.sample_rate / 1000)
        if len(audio) < samples:
            return audio

        fade = self._create_fade(samples, fade_in=True)
        audio = audio.copy()
        audio[:samples] *= fade
        return audio

    def apply_fade_out(self, audio: np.ndarray, duration_ms: int = 100) -> np.ndarray:
        """Applique un fade-out a la fin."""
        samples = int(duration_ms * self.sample_rate / 1000)
        if len(audio) < samples:
            return audio

        fade = self._create_fade(samples, fade_in=False)
        audio = audio.copy()
        audio[-samples:] *= fade
        return audio


class RoomToneGenerator:
    """
    Genere un leger bruit de fond (room tone).

    Simule l'ambiance d'un studio d'enregistrement.
    """

    def __init__(self, sample_rate: int = 24000, level_db: float = -50):
        self.sample_rate = sample_rate
        self.level = 10 ** (level_db / 20)

    def generate(self, duration_seconds: float) -> np.ndarray:
        """Genere du room tone."""
        num_samples = int(duration_seconds * self.sample_rate)

        # Bruit rose (plus naturel que bruit blanc)
        white_noise = np.random.randn(num_samples)

        # Filtrer pour obtenir du bruit rose
        # Approximation simple: moyenne mobile
        window_size = 10
        pink_noise = np.convolve(
            white_noise,
            np.ones(window_size) / window_size,
            mode='same'
        )

        # Normaliser et appliquer le niveau
        pink_noise = pink_noise / np.max(np.abs(pink_noise)) * self.level

        return pink_noise.astype(np.float32)

    def add_to_audio(
        self,
        audio: np.ndarray,
        level_db: float = -50
    ) -> np.ndarray:
        """Ajoute du room tone a un audio existant."""
        duration = len(audio) / self.sample_rate
        room_tone = self.generate(duration)

        # Ajuster le niveau
        level = 10 ** (level_db / 20)
        room_tone = room_tone * level / self.level

        # Mixer
        return audio + room_tone[:len(audio)]


def enhance_audiobook(
    input_path: Path,
    output_path: Path,
    config: Optional[AudioEnhancerConfig] = None,
    verbose: bool = True
) -> bool:
    """
    Fonction utilitaire pour ameliorer un audiobook complet.

    Args:
        input_path: Fichier audio d'entree
        output_path: Fichier de sortie
        config: Configuration (optionnelle)
        verbose: Afficher les details

    Returns:
        True si succes
    """
    enhancer = AudioEnhancer(config)

    if not enhancer.is_available():
        print("ATTENTION: ffmpeg non disponible, pas d'amelioration audio")
        return False

    if verbose:
        print(f"Analyse du fichier source...")
        analysis = enhancer.analyze_loudness(input_path)
        if analysis:
            print(f"  Loudness: {analysis['integrated']:.1f} LUFS")
            print(f"  True Peak: {analysis['true_peak']:.1f} dB")
            print(f"  LRA: {analysis['lra']:.1f}")

    if verbose:
        print(f"Application des ameliorations...")

    success = enhancer.enhance_file(input_path, output_path, verbose)

    if success and verbose:
        print(f"Analyse du fichier ameliore...")
        analysis = enhancer.analyze_loudness(output_path)
        if analysis:
            print(f"  Loudness: {analysis['integrated']:.1f} LUFS")
            print(f"  True Peak: {analysis['true_peak']:.1f} dB")

    return success


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python audio_enhancer.py input.wav output.mp3")
        print("\nOptions de configuration:")
        print("  --lufs=-19     Target loudness")
        print("  --no-deess     Desactiver de-essing")
        print("  --no-compress  Desactiver compression")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    config = AudioEnhancerConfig()

    # Parser les options
    for arg in sys.argv[3:]:
        if arg.startswith("--lufs="):
            config.target_lufs = float(arg.split("=")[1])
        elif arg == "--no-deess":
            config.deess_enabled = False
        elif arg == "--no-compress":
            config.compression_enabled = False

    success = enhance_audiobook(input_file, output_file, config, verbose=True)
    sys.exit(0 if success else 1)
