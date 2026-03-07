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
        """Genere du room tone avec bruit rose (spectre 1/f)."""
        num_samples = int(duration_seconds * self.sample_rate)
        if num_samples <= 0:
            return np.zeros(0, dtype=np.float32)

        # Generer du vrai bruit rose via filtrage FFT (spectre 1/f)
        white_noise = np.random.randn(num_samples)
        fft = np.fft.rfft(white_noise)
        freqs = np.fft.rfftfreq(num_samples)

        # Filtre 1/f (bruit rose) — eviter division par zero
        freqs[0] = 1.0
        pink_filter = 1.0 / np.sqrt(freqs)
        pink_filter[0] = 0.0  # Pas de DC

        pink_noise = np.fft.irfft(fft * pink_filter, n=num_samples)

        # Normaliser et appliquer le niveau
        max_val = np.max(np.abs(pink_noise))
        if max_val > 0:
            pink_noise = pink_noise / max_val * self.level

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


class NativeAudioEnhancer:
    """
    Enhancer audio natif en numpy/scipy — pas besoin de ffmpeg.

    Pipeline de mastering broadcast:
    1. Highpass filter (retire rumble <80Hz)
    2. De-essing dynamique (reduit sibilantes 4-8kHz)
    3. EQ voix (presence + air)
    4. Compression douce (ratio 3:1)
    5. Normalisation loudness
    6. Brick-wall limiter
    """

    def __init__(self, config: Optional[AudioEnhancerConfig] = None):
        self.config = config or AudioEnhancerConfig()

    def enhance(self, audio: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
        """
        Applique le pipeline de mastering complet sur un array audio.

        Args:
            audio: Signal audio float32 [-1, 1]
            sample_rate: Taux d'echantillonnage

        Returns:
            Audio ameliore float32 [-1, 1]
        """
        if len(audio) == 0:
            return audio

        audio = audio.astype(np.float32).copy()

        # 0. Debruitage spectral (noisereduce si disponible, sinon gate maison)
        audio = self._denoise(audio, sample_rate)

        # 1. Highpass — retirer le rumble basse frequence
        audio = self._highpass(audio, sample_rate, self.config.highpass_freq)

        # 2. De-essing dynamique
        if self.config.deess_enabled:
            audio = self._deess(audio, sample_rate)

        # 3. EQ voix (presence + air)
        audio = self._eq_voice(audio, sample_rate)

        # 4. Compression douce
        if self.config.compression_enabled:
            audio = self._compress(audio, sample_rate)

        # 5. Normalisation loudness
        audio = self._normalize_loudness(audio)

        # 6. Brick-wall limiter
        audio = self._limiter(audio)

        return audio

    @staticmethod
    def _highpass(audio: np.ndarray, sr: int, cutoff: int) -> np.ndarray:
        """Filtre passe-haut butterworth 2nd ordre."""
        try:
            from scipy.signal import butter, sosfilt
            sos = butter(2, cutoff, btype='high', fs=sr, output='sos')
            return sosfilt(sos, audio).astype(np.float32)
        except ImportError:
            # Fallback simple: DC removal
            return (audio - np.mean(audio)).astype(np.float32)

    def _deess(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """De-essing dynamique — reduit les sibilantes sans affecter le reste."""
        try:
            from scipy.signal import butter, sosfilt
            # Isoler la bande sibilante (4-8 kHz)
            nyq = sr / 2
            low = min(4000 / nyq, 0.99)
            high = min(8000 / nyq, 0.99)
            if low >= high:
                return audio
            sos = butter(2, [low, high], btype='band', output='sos')
            sibilant = sosfilt(sos, audio)

            # Envelope follower
            window = int(sr * 0.01)  # 10ms
            if window < 1:
                return audio
            envelope = np.convolve(np.abs(sibilant), np.ones(window) / window, mode='same')

            # Seuil de de-essing
            threshold = 10 ** (self.config.deess_threshold / 20)

            # Gain reduction: attenuer seulement quand les sibilantes depassent le seuil
            gain = np.ones_like(audio)
            mask = envelope > threshold
            if np.any(mask):
                # Reduction progressive
                gain[mask] = threshold / np.maximum(envelope[mask], 1e-10)
                gain = np.clip(gain, 0.3, 1.0)  # Max 10dB de reduction

                # Lisser le gain pour eviter les artifacts
                smooth_window = int(sr * 0.005)
                if smooth_window > 1:
                    gain = np.convolve(gain, np.ones(smooth_window) / smooth_window, mode='same')

                # Appliquer uniquement sur la bande sibilante
                audio = audio - sibilant * (1 - gain)

            return audio.astype(np.float32)
        except ImportError:
            return audio

    def _eq_voice(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """EQ pour voix — boost presence (2-4kHz) et air (10-16kHz)."""
        try:
            from scipy.signal import butter, sosfilt

            # Presence boost (2-4 kHz)
            if self.config.presence_boost > 0:
                nyq = sr / 2
                low = min(2000 / nyq, 0.99)
                high = min(4000 / nyq, 0.99)
                if low < high:
                    sos = butter(2, [low, high], btype='band', output='sos')
                    presence = sosfilt(sos, audio)
                    gain = 10 ** (self.config.presence_boost / 20) - 1
                    audio = audio + presence * gain

            # Air boost (10-16 kHz) — seulement si sr > 22kHz
            if self.config.air_boost > 0 and sr > 22000:
                low = min(10000 / nyq, 0.99)
                high = min(16000 / nyq, 0.99)
                if low < high:
                    sos = butter(2, [low, high], btype='band', output='sos')
                    air = sosfilt(sos, audio)
                    gain = 10 ** (self.config.air_boost / 20) - 1
                    audio = audio + air * gain

            return audio.astype(np.float32)
        except ImportError:
            return audio

    def _compress(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Compression douce — reduit la dynamique sans ecraser le signal."""
        threshold_lin = 10 ** (self.config.comp_threshold / 20)
        ratio = self.config.comp_ratio
        attack_samples = max(1, int(self.config.comp_attack * sr / 1000))
        release_samples = max(1, int(self.config.comp_release * sr / 1000))

        # Envelope follower avec attack/release distincts
        envelope = np.zeros_like(audio)
        env_val = 0.0
        abs_audio = np.abs(audio)

        for i in range(len(audio)):
            sample = abs_audio[i]
            if sample > env_val:
                env_val += (sample - env_val) / attack_samples
            else:
                env_val += (sample - env_val) / release_samples
            envelope[i] = env_val

        # Calculer le gain de compression
        gain = np.ones_like(audio)
        mask = envelope > threshold_lin
        if np.any(mask):
            # Compression: above threshold, reduce by ratio
            above = envelope[mask] / threshold_lin
            compressed = threshold_lin * above ** (1 / ratio)
            gain[mask] = compressed / np.maximum(envelope[mask], 1e-10)

        # Appliquer makeup gain pour compenser
        makeup = threshold_lin ** (1 - 1 / ratio)
        audio = audio * gain / makeup

        return audio.astype(np.float32)

    @staticmethod
    def _normalize_loudness(audio: np.ndarray, target_db: float = -19.0) -> np.ndarray:
        """Normalise le loudness RMS vers la cible."""
        if len(audio) == 0:
            return audio

        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-10:
            return audio

        current_db = 20 * np.log10(rms)
        gain = 10 ** ((target_db - current_db) / 20)

        return (audio * gain).astype(np.float32)

    @staticmethod
    def _denoise(audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Debruitage spectral — utilise noisereduce si disponible,
        sinon fallback vers le noise gate maison.
        """
        try:
            import noisereduce as nr
            # noisereduce stationnaire: estime le bruit et le soustrait
            return nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=True,
                prop_decrease=0.75,  # Reduire 75% du bruit
                n_fft=2048,
                hop_length=512,
            ).astype(np.float32)
        except ImportError:
            # Fallback: noise gate spectral maison
            return NativeAudioEnhancer._spectral_noise_gate(audio, sr)

    @staticmethod
    def _spectral_noise_gate(
        audio: np.ndarray,
        sr: int,
        frame_size: int = 2048,
        hop_size: int = 512,
        noise_floor_db: float = -50.0,
        smoothing: float = 0.8,
    ) -> np.ndarray:
        """
        Noise gate spectral — reduit le bruit de fond sans affecter la voix.

        Estime le plancher de bruit sur les trames silencieuses,
        puis attenue les frequences proches du bruit.
        """
        if len(audio) < frame_size:
            return audio

        # STFT manuelle via numpy
        n_frames = 1 + (len(audio) - frame_size) // hop_size
        window = np.hanning(frame_size).astype(np.float32)
        frames = np.zeros((n_frames, frame_size), dtype=np.float32)

        for i in range(n_frames):
            start = i * hop_size
            frames[i] = audio[start:start + frame_size] * window

        spectra = np.fft.rfft(frames, axis=1)
        magnitudes = np.abs(spectra)

        # Estimer le plancher de bruit: moyenne des 10% de trames les plus calmes
        frame_energies = np.sum(magnitudes ** 2, axis=1)
        quiet_count = max(1, n_frames // 10)
        quiet_indices = np.argsort(frame_energies)[:quiet_count]
        noise_profile = np.mean(magnitudes[quiet_indices], axis=0)

        # Seuil en dB
        threshold_linear = 10 ** (noise_floor_db / 20)
        noise_profile = np.maximum(noise_profile, threshold_linear)

        # Appliquer le gate: attenuer les bins proches du bruit
        gain = np.ones_like(magnitudes)
        for i in range(n_frames):
            ratio = magnitudes[i] / (noise_profile + 1e-10)
            # Gain doux: 1.0 si bien au-dessus du bruit, 0 si en dessous
            frame_gain = np.clip((ratio - 1.0) / 2.0, 0.0, 1.0)
            # Lisser
            if i > 0:
                frame_gain = smoothing * gain[i - 1] + (1 - smoothing) * frame_gain
            gain[i] = frame_gain

        # Appliquer le gain spectral
        filtered_spectra = spectra * gain

        # ISTFT manuelle
        output = np.zeros(len(audio), dtype=np.float32)
        window_sum = np.zeros(len(audio), dtype=np.float32)

        for i in range(n_frames):
            start = i * hop_size
            frame = np.fft.irfft(filtered_spectra[i], n=frame_size).astype(np.float32)
            output[start:start + frame_size] += frame * window
            window_sum[start:start + frame_size] += window ** 2

        # Normaliser par la somme des fenetres
        nonzero = window_sum > 1e-8
        output[nonzero] /= window_sum[nonzero]

        return output

    def _limiter(self, audio: np.ndarray) -> np.ndarray:
        """Brick-wall limiter — empeche tout depassement."""
        limit = 10 ** (self.config.true_peak_limit / 20)
        return np.clip(audio, -limit, limit).astype(np.float32)


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
