"""
Crossfade et transitions audio entre segments.

Elimine les "coutures" audibles entre les chunks de texte
en appliquant des fondus enchaines intelligents.
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CrossfadeConfig:
    """Configuration du crossfade."""
    # Duree du crossfade en secondes
    crossfade_duration: float = 0.05  # 50ms par defaut
    # Duree minimale du segment pour appliquer le crossfade
    min_segment_duration: float = 0.1
    # Type de courbe: 'linear', 'cosine', 'exponential'
    curve_type: str = 'cosine'
    # Appliquer un fade in/out aux extremites
    apply_edge_fades: bool = True
    edge_fade_duration: float = 0.01  # 10ms


class AudioCrossfader:
    """
    Applique des crossfades entre segments audio pour des transitions fluides.
    """

    def __init__(self, config: Optional[CrossfadeConfig] = None):
        self.config = config or CrossfadeConfig()

    def _generate_fade_curve(self, length: int, fade_in: bool = True) -> np.ndarray:
        """
        Genere une courbe de fade.

        Args:
            length: Nombre d'echantillons
            fade_in: True pour fade in, False pour fade out

        Returns:
            Array de coefficients de 0 a 1
        """
        if length <= 0:
            return np.array([])

        t = np.linspace(0, 1, length)

        if self.config.curve_type == 'linear':
            curve = t
        elif self.config.curve_type == 'cosine':
            # Courbe en cosinus (plus naturelle)
            curve = (1 - np.cos(t * np.pi)) / 2
        elif self.config.curve_type == 'exponential':
            # Courbe exponentielle
            curve = t ** 2
        else:
            curve = t

        if not fade_in:
            curve = 1 - curve

        return curve.astype(np.float32)

    def apply_fade_in(self, audio: np.ndarray, sample_rate: int,
                      duration: Optional[float] = None) -> np.ndarray:
        """Applique un fade in au debut de l'audio."""
        if duration is None:
            duration = self.config.edge_fade_duration

        fade_samples = int(duration * sample_rate)
        fade_samples = min(fade_samples, len(audio))

        if fade_samples <= 0:
            return audio

        result = audio.copy()
        curve = self._generate_fade_curve(fade_samples, fade_in=True)
        result[:fade_samples] = result[:fade_samples] * curve

        return result

    def apply_fade_out(self, audio: np.ndarray, sample_rate: int,
                       duration: Optional[float] = None) -> np.ndarray:
        """Applique un fade out a la fin de l'audio."""
        if duration is None:
            duration = self.config.edge_fade_duration

        fade_samples = int(duration * sample_rate)
        fade_samples = min(fade_samples, len(audio))

        if fade_samples <= 0:
            return audio

        result = audio.copy()
        curve = self._generate_fade_curve(fade_samples, fade_in=False)
        result[-fade_samples:] = result[-fade_samples:] * curve

        return result

    def crossfade_segments(
        self,
        segment1: np.ndarray,
        segment2: np.ndarray,
        sample_rate: int,
        crossfade_duration: Optional[float] = None
    ) -> np.ndarray:
        """
        Fusionne deux segments avec un crossfade.

        Args:
            segment1: Premier segment audio
            segment2: Deuxieme segment audio
            sample_rate: Taux d'echantillonnage
            crossfade_duration: Duree du crossfade (optionnel)

        Returns:
            Audio fusionne avec crossfade
        """
        if crossfade_duration is None:
            crossfade_duration = self.config.crossfade_duration

        crossfade_samples = int(crossfade_duration * sample_rate)

        # Verifier que les segments sont assez longs
        min_samples = int(self.config.min_segment_duration * sample_rate)
        if len(segment1) < min_samples or len(segment2) < min_samples:
            # Pas de crossfade, simple concatenation
            return np.concatenate([segment1, segment2])

        # Limiter le crossfade a la taille des segments
        crossfade_samples = min(crossfade_samples, len(segment1), len(segment2))

        if crossfade_samples <= 0:
            return np.concatenate([segment1, segment2])

        # Creer les courbes de fade
        fade_out = self._generate_fade_curve(crossfade_samples, fade_in=False)
        fade_in = self._generate_fade_curve(crossfade_samples, fade_in=True)

        # Partie du segment1 avant le crossfade
        result_start = segment1[:-crossfade_samples]

        # Zone de crossfade
        crossfade_zone = (
            segment1[-crossfade_samples:] * fade_out +
            segment2[:crossfade_samples] * fade_in
        )

        # Partie du segment2 apres le crossfade
        result_end = segment2[crossfade_samples:]

        return np.concatenate([result_start, crossfade_zone, result_end])

    def crossfade_all(
        self,
        segments: List[np.ndarray],
        sample_rate: int,
        crossfade_duration: Optional[float] = None
    ) -> np.ndarray:
        """
        Fusionne une liste de segments avec crossfades.

        Args:
            segments: Liste de segments audio
            sample_rate: Taux d'echantillonnage
            crossfade_duration: Duree du crossfade

        Returns:
            Audio complet avec crossfades
        """
        if not segments:
            return np.array([], dtype=np.float32)

        if len(segments) == 1:
            result = segments[0]
        else:
            # Fusionner progressivement
            result = segments[0]
            for segment in segments[1:]:
                if len(segment) > 0:
                    result = self.crossfade_segments(
                        result, segment, sample_rate, crossfade_duration
                    )

        # Appliquer les fades aux extremites
        if self.config.apply_edge_fades and len(result) > 0:
            result = self.apply_fade_in(result, sample_rate)
            result = self.apply_fade_out(result, sample_rate)

        return result

    def smooth_silence_transitions(
        self,
        audio: np.ndarray,
        sample_rate: int,
        silence_threshold: float = 0.01,
        min_silence_duration: float = 0.1
    ) -> np.ndarray:
        """
        Lisse les transitions autour des silences pour eviter les clics.

        Args:
            audio: Audio a traiter
            sample_rate: Taux d'echantillonnage
            silence_threshold: Seuil de detection du silence
            min_silence_duration: Duree minimale d'un silence

        Returns:
            Audio avec transitions lissees
        """
        if len(audio) == 0:
            return audio

        result = audio.copy()
        min_silence_samples = int(min_silence_duration * sample_rate)
        fade_samples = int(0.01 * sample_rate)  # 10ms de fade

        # Detecter les zones de silence
        is_silent = np.abs(audio) < silence_threshold

        # Trouver les transitions
        transitions = np.diff(is_silent.astype(int))
        silence_starts = np.where(transitions == 1)[0]
        silence_ends = np.where(transitions == -1)[0]

        # Appliquer des micro-fades aux transitions
        for start in silence_starts:
            if start > fade_samples:
                fade_end = min(start + fade_samples, len(result))
                fade_curve = self._generate_fade_curve(fade_end - start, fade_in=False)
                result[start:fade_end] = result[start:fade_end] * fade_curve

        for end in silence_ends:
            if end < len(result) - fade_samples:
                fade_start = max(end - fade_samples, 0)
                fade_curve = self._generate_fade_curve(end - fade_start, fade_in=True)
                result[fade_start:end] = result[fade_start:end] * fade_curve

        return result


class SegmentAssembler:
    """
    Assemble des segments audio avec pauses et crossfades.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        crossfade_config: Optional[CrossfadeConfig] = None
    ):
        self.sample_rate = sample_rate
        self.crossfader = AudioCrossfader(crossfade_config)

    def create_silence(self, duration: float) -> np.ndarray:
        """Cree un segment de silence."""
        samples = int(duration * self.sample_rate)
        return np.zeros(samples, dtype=np.float32)

    def assemble(
        self,
        segments: List[Tuple[np.ndarray, float]],
        use_crossfade: bool = True
    ) -> np.ndarray:
        """
        Assemble des segments avec leurs pauses.

        Args:
            segments: Liste de tuples (audio, pause_after)
            use_crossfade: Utiliser le crossfade entre segments

        Returns:
            Audio assemble
        """
        if not segments:
            return np.array([], dtype=np.float32)

        all_parts = []

        for i, (audio, pause_after) in enumerate(segments):
            if len(audio) > 0:
                all_parts.append(audio)

            # Ajouter la pause (sauf apres le dernier segment)
            if pause_after > 0 and i < len(segments) - 1:
                silence = self.create_silence(pause_after)
                all_parts.append(silence)

        if not all_parts:
            return np.array([], dtype=np.float32)

        if use_crossfade:
            # Fusionner les segments audio (pas les silences) avec crossfade
            return self.crossfader.crossfade_all(all_parts, self.sample_rate)
        else:
            return np.concatenate(all_parts)

    def assemble_with_transitions(
        self,
        segments: List[np.ndarray],
        pauses: List[float],
        speaker_changes: Optional[List[bool]] = None
    ) -> np.ndarray:
        """
        Assemble avec gestion intelligente des transitions.

        Args:
            segments: Liste de segments audio
            pauses: Liste de durees de pause entre segments
            speaker_changes: Liste indiquant si le locuteur change

        Returns:
            Audio assemble
        """
        if not segments:
            return np.array([], dtype=np.float32)

        if speaker_changes is None:
            speaker_changes = [False] * len(segments)

        result_parts = []

        for i, segment in enumerate(segments):
            if len(segment) == 0:
                continue

            # Determiner le type de transition
            if i > 0:
                pause_duration = pauses[i - 1] if i - 1 < len(pauses) else 0.3
                is_speaker_change = speaker_changes[i] if i < len(speaker_changes) else False

                # Pause plus longue pour changement de locuteur
                if is_speaker_change:
                    pause_duration = max(pause_duration, 0.4)

                # Ajouter la pause
                if pause_duration > 0:
                    result_parts.append(self.create_silence(pause_duration))

            result_parts.append(segment)

        if not result_parts:
            return np.array([], dtype=np.float32)

        # Appliquer crossfade global
        return self.crossfader.crossfade_all(result_parts, self.sample_rate)


def apply_crossfade_to_chapter(
    audio_segments: List[np.ndarray],
    sample_rate: int = 24000,
    crossfade_ms: int = 50
) -> np.ndarray:
    """
    Fonction utilitaire pour appliquer le crossfade a un chapitre.

    Args:
        audio_segments: Liste de segments audio
        sample_rate: Taux d'echantillonnage
        crossfade_ms: Duree du crossfade en millisecondes

    Returns:
        Audio du chapitre avec crossfades
    """
    config = CrossfadeConfig(
        crossfade_duration=crossfade_ms / 1000,
        curve_type='cosine',
        apply_edge_fades=True
    )

    crossfader = AudioCrossfader(config)
    return crossfader.crossfade_all(audio_segments, sample_rate)


if __name__ == "__main__":
    # Test du crossfade
    import soundfile as sf

    # Creer deux segments de test (sinusoides)
    sr = 24000
    duration = 0.5

    t1 = np.linspace(0, duration, int(sr * duration))
    t2 = np.linspace(0, duration, int(sr * duration))

    segment1 = (np.sin(2 * np.pi * 440 * t1) * 0.5).astype(np.float32)  # La 440Hz
    segment2 = (np.sin(2 * np.pi * 880 * t2) * 0.5).astype(np.float32)  # La 880Hz

    # Sans crossfade
    no_crossfade = np.concatenate([segment1, segment2])

    # Avec crossfade
    crossfader = AudioCrossfader()
    with_crossfade = crossfader.crossfade_segments(segment1, segment2, sr)

    print(f"Sans crossfade: {len(no_crossfade)} samples")
    print(f"Avec crossfade: {len(with_crossfade)} samples")

    # Sauvegarder
    sf.write("output/test_no_crossfade.wav", no_crossfade, sr)
    sf.write("output/test_with_crossfade.wav", with_crossfade, sr)

    print("Fichiers de test generes dans output/")
