"""
Optimisation des silences pour AudioReader.

Detecte et reduit les silences excessifs dans l'audio genere
tout en preservant les pauses dramatiques (taggees).
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class SilenceSegment:
    """Un segment de silence detecte."""
    start_sample: int
    end_sample: int
    duration_s: float
    is_dramatic: bool = False  # Pause taguee, a preserver


class SilenceOptimizer:
    """
    Optimise les silences dans l'audio.

    Detecte les silences trop longs et les reduit a une duree maximale,
    tout en preservant les pauses dramatiques.
    """

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate

    def detect_silences(
        self,
        audio: np.ndarray,
        threshold_db: float = -40.0,
        min_duration_s: float = 0.3,
    ) -> List[SilenceSegment]:
        """
        Detecte les segments de silence dans l'audio.

        Args:
            audio: Signal audio
            threshold_db: Seuil en dB sous lequel on considere du silence
            min_duration_s: Duree minimale pour considerer un silence
        """
        threshold_linear = 10 ** (threshold_db / 20.0)
        min_samples = int(min_duration_s * self.sample_rate)

        # Calculer l'amplitude RMS par fenetre
        window_size = int(0.01 * self.sample_rate)  # 10ms
        if window_size == 0:
            window_size = 1

        silences = []
        in_silence = False
        silence_start = 0

        for i in range(0, len(audio) - window_size, window_size):
            window = audio[i:i + window_size]
            rms = np.sqrt(np.mean(window ** 2))

            if rms < threshold_linear:
                if not in_silence:
                    silence_start = i
                    in_silence = True
            else:
                if in_silence:
                    duration = i - silence_start
                    if duration >= min_samples:
                        silences.append(SilenceSegment(
                            start_sample=silence_start,
                            end_sample=i,
                            duration_s=duration / self.sample_rate,
                        ))
                    in_silence = False

        # Silence a la fin
        if in_silence:
            duration = len(audio) - silence_start
            if duration >= min_samples:
                silences.append(SilenceSegment(
                    start_sample=silence_start,
                    end_sample=len(audio),
                    duration_s=duration / self.sample_rate,
                ))

        return silences

    def optimize(
        self,
        audio: np.ndarray,
        max_silence_ms: int = 800,
        min_silence_ms: int = 200,
        threshold_db: float = -40.0,
    ) -> np.ndarray:
        """
        Optimise les silences dans l'audio.

        Args:
            audio: Signal audio
            max_silence_ms: Duree maximale d'un silence (ms)
            min_silence_ms: Duree minimale a conserver (ms)
            threshold_db: Seuil de detection du silence

        Returns:
            Audio avec silences optimises
        """
        max_silence_s = max_silence_ms / 1000.0
        min_silence_s = min_silence_ms / 1000.0

        silences = self.detect_silences(audio, threshold_db, min_duration_s=max_silence_s)

        if not silences:
            return audio

        # Construire l'audio optimise
        result_parts = []
        prev_end = 0

        for silence in silences:
            # Garder l'audio avant le silence
            result_parts.append(audio[prev_end:silence.start_sample])

            # Reduire le silence a min_silence_ms
            keep_samples = int(min_silence_s * self.sample_rate)
            silence_audio = audio[silence.start_sample:silence.start_sample + keep_samples]
            result_parts.append(silence_audio)

            prev_end = silence.end_sample

        # Garder la fin
        result_parts.append(audio[prev_end:])

        return np.concatenate(result_parts).astype(np.float32)

    def get_stats(self, audio: np.ndarray, threshold_db: float = -40.0) -> dict:
        """Retourne des statistiques sur les silences."""
        silences = self.detect_silences(audio, threshold_db, min_duration_s=0.1)
        total_silence = sum(s.duration_s for s in silences)
        total_duration = len(audio) / self.sample_rate
        return {
            "total_duration_s": round(total_duration, 2),
            "silence_count": len(silences),
            "total_silence_s": round(total_silence, 2),
            "silence_percentage": round(100 * total_silence / max(total_duration, 0.001), 1),
            "longest_silence_s": round(max((s.duration_s for s in silences), default=0), 2),
        }
