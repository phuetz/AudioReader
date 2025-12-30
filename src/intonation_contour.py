"""
Intonation Contour Module v2.3.

Détecte et applique les contours d'intonation phrase-level pour
une prosodie plus naturelle et expressive.

Les contours d'intonation définissent comment le pitch varie au cours
d'une phrase, donnant du sens et de l'émotion à la parole.
"""
import numpy as np
import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class IntonationContour(Enum):
    """Types de contours d'intonation."""

    DECLARATIVE = "declarative"      # Phrase déclarative: descente finale
    QUESTION_YN = "question_yn"       # Question oui/non: montée finale
    QUESTION_WH = "question_wh"       # Question en WH: pic puis descente
    EXCLAMATION = "exclamation"       # Exclamation: pic fort puis descente rapide
    CONTINUATION = "continuation"     # Continuation: légère montée
    SUSPENSE = "suspense"             # Suspense/ellipse: descente lente
    NEUTRAL = "neutral"               # Neutre: pas de modification


@dataclass
class ContourConfig:
    """Configuration pour un contour d'intonation."""

    pitch_curve: list[float]    # Changements de pitch en demi-tons par segment
    timing_weights: list[float]  # Poids temporels des segments (normalisés à 1)


# Configurations des contours (5 segments: début, début-milieu, milieu, milieu-fin, fin)
CONTOUR_CONFIGS = {
    IntonationContour.DECLARATIVE: ContourConfig(
        pitch_curve=[0.0, 0.0, 0.0, -1.0, -2.5],
        timing_weights=[0.15, 0.20, 0.30, 0.20, 0.15]
    ),
    IntonationContour.QUESTION_YN: ContourConfig(
        pitch_curve=[0.0, 0.0, 0.5, 2.0, 4.0],
        timing_weights=[0.20, 0.20, 0.25, 0.20, 0.15]
    ),
    IntonationContour.QUESTION_WH: ContourConfig(
        pitch_curve=[2.0, 1.0, 0.0, -0.5, -1.5],
        timing_weights=[0.15, 0.20, 0.30, 0.20, 0.15]
    ),
    IntonationContour.EXCLAMATION: ContourConfig(
        pitch_curve=[1.0, 2.5, 1.0, -1.0, -2.0],
        timing_weights=[0.10, 0.20, 0.30, 0.25, 0.15]
    ),
    IntonationContour.CONTINUATION: ContourConfig(
        pitch_curve=[0.0, 0.0, 0.0, 0.5, 1.0],
        timing_weights=[0.20, 0.20, 0.30, 0.15, 0.15]
    ),
    IntonationContour.SUSPENSE: ContourConfig(
        pitch_curve=[0.0, -0.5, -1.0, -1.5, -2.0],
        timing_weights=[0.15, 0.20, 0.30, 0.20, 0.15]
    ),
    IntonationContour.NEUTRAL: ContourConfig(
        pitch_curve=[0.0, 0.0, 0.0, 0.0, 0.0],
        timing_weights=[0.20, 0.20, 0.20, 0.20, 0.20]
    ),
}

# Mots interrogatifs français
WH_WORDS_FR = {
    "qui", "que", "quoi", "où", "quand", "comment", "pourquoi",
    "quel", "quelle", "quels", "quelles", "lequel", "laquelle",
    "combien", "duquel", "auquel"
}


class IntonationContourDetector:
    """Détecte le type de contour d'intonation à partir du texte."""

    def __init__(self, language: str = "fr"):
        """
        Initialise le détecteur.

        Args:
            language: Code de langue (fr, en)
        """
        self.language = language

    def detect(self, text: str) -> IntonationContour:
        """
        Détecte le contour d'intonation approprié pour le texte.

        Args:
            text: Texte de la phrase

        Returns:
            Type de contour détecté
        """
        text = text.strip()

        if not text:
            return IntonationContour.NEUTRAL

        # Vérifier la ponctuation finale
        if text.endswith('?'):
            if self._is_wh_question(text):
                return IntonationContour.QUESTION_WH
            return IntonationContour.QUESTION_YN

        if text.endswith('!'):
            return IntonationContour.EXCLAMATION

        if text.endswith('...') or text.endswith('…'):
            return IntonationContour.SUSPENSE

        # Vérifier les marqueurs de continuation
        if self._is_continuation(text):
            return IntonationContour.CONTINUATION

        # Par défaut: déclarative
        if text.endswith('.') or text[-1].isalnum():
            return IntonationContour.DECLARATIVE

        return IntonationContour.NEUTRAL

    def _is_wh_question(self, text: str) -> bool:
        """
        Vérifie si c'est une question en WH (qui, quoi, où, etc.).

        Args:
            text: Texte de la question

        Returns:
            True si question en WH
        """
        # Normaliser et extraire le premier mot
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        words = clean_text.split()

        if not words:
            return False

        # Le premier mot est-il un mot interrogatif?
        first_word = words[0]
        if first_word in WH_WORDS_FR:
            return True

        # Vérifier aussi "est-ce que" qui précède parfois un WH
        if len(words) >= 3 and words[:3] == ["est", "ce", "que"]:
            if len(words) > 3 and words[3] in WH_WORDS_FR:
                return True

        return False

    def _is_continuation(self, text: str) -> bool:
        """
        Vérifie si le texte indique une continuation.

        Args:
            text: Texte à analyser

        Returns:
            True si continuation détectée
        """
        # Finit par une virgule ou un point-virgule
        if text.endswith(',') or text.endswith(';'):
            return True

        # Finit par "et", "ou", "mais", "car", "donc"
        continuation_markers = ["et", "ou", "mais", "car", "donc", "puis", "alors"]
        words = text.lower().split()

        if words and words[-1].rstrip('.,;:') in continuation_markers:
            return True

        return False


class IntonationContourApplicator:
    """Applique les contours d'intonation à l'audio."""

    def __init__(
        self,
        sample_rate: int = 24000,
        strength: float = 0.7,
        use_librosa: bool = True
    ):
        """
        Initialise l'applicateur.

        Args:
            sample_rate: Taux d'échantillonnage
            strength: Force du contour (0.0 à 1.0)
            use_librosa: Utiliser librosa pour le pitch shift (sinon, approximation simple)
        """
        self.sample_rate = sample_rate
        self.strength = strength
        self.use_librosa = use_librosa

        # Vérifier la disponibilité de librosa
        self._librosa_available = False
        if use_librosa:
            try:
                import librosa
                self._librosa_available = True
            except ImportError:
                logger.warning("librosa non disponible, utilisation de l'approximation simple")

    def apply_contour(
        self,
        audio: np.ndarray,
        contour: IntonationContour,
        strength: float = None
    ) -> np.ndarray:
        """
        Applique un contour d'intonation à l'audio.

        Args:
            audio: Signal audio (numpy array)
            contour: Type de contour à appliquer
            strength: Force du contour (override)

        Returns:
            Audio avec le contour appliqué
        """
        if len(audio) == 0:
            return audio

        if contour == IntonationContour.NEUTRAL:
            return audio

        strength = strength if strength is not None else self.strength
        if strength <= 0:
            return audio

        config = CONTOUR_CONFIGS.get(contour, CONTOUR_CONFIGS[IntonationContour.NEUTRAL])

        # Diviser l'audio en segments selon les poids temporels
        segments, boundaries = self._split_audio(audio, config.timing_weights)

        # Appliquer le pitch shift à chaque segment
        shifted_segments = []
        for seg, pitch_delta in zip(segments, config.pitch_curve):
            # Appliquer strength au pitch delta
            actual_delta = pitch_delta * strength

            if abs(actual_delta) < 0.1:
                # Pas de changement significatif
                shifted_segments.append(seg)
            else:
                shifted = self._pitch_shift(seg, actual_delta)
                shifted_segments.append(shifted)

        # Recombiner avec crossfade
        result = self._crossfade_segments(shifted_segments)

        return result.astype(np.float32)

    def _split_audio(
        self,
        audio: np.ndarray,
        weights: list[float]
    ) -> tuple[list[np.ndarray], list[int]]:
        """
        Divise l'audio en segments selon les poids.

        Args:
            audio: Signal audio
            weights: Poids relatifs des segments

        Returns:
            Tuple (liste de segments, liste des frontières)
        """
        total_length = len(audio)

        # Normaliser les poids
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Calculer les frontières
        boundaries = [0]
        for i, w in enumerate(normalized_weights[:-1]):
            next_boundary = int(boundaries[-1] + w * total_length)
            boundaries.append(next_boundary)
        boundaries.append(total_length)

        # Extraire les segments
        segments = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            segments.append(audio[start:end])

        return segments, boundaries

    def _pitch_shift(self, audio: np.ndarray, semitones: float) -> np.ndarray:
        """
        Applique un pitch shift au signal.

        Args:
            audio: Signal audio
            semitones: Décalage en demi-tons

        Returns:
            Signal avec pitch modifié
        """
        if len(audio) == 0:
            return audio

        if self._librosa_available:
            try:
                import librosa
                # librosa.effects.pitch_shift attend n_steps (semitones)
                shifted = librosa.effects.pitch_shift(
                    audio.astype(np.float32),
                    sr=self.sample_rate,
                    n_steps=semitones
                )
                return shifted.astype(np.float32)
            except Exception as e:
                logger.warning(f"Erreur librosa pitch_shift: {e}, fallback")

        # Fallback: approximation simple via resampling
        return self._pitch_shift_simple(audio, semitones)

    def _pitch_shift_simple(self, audio: np.ndarray, semitones: float) -> np.ndarray:
        """
        Pitch shift approximatif sans librosa.

        Utilise un resampling simple qui modifie aussi la durée,
        puis un time-stretch pour compenser.

        Args:
            audio: Signal audio
            semitones: Décalage en demi-tons

        Returns:
            Signal avec pitch approximativement modifié
        """
        if len(audio) == 0 or abs(semitones) < 0.1:
            return audio

        # Facteur de pitch (2^(semitones/12))
        factor = 2.0 ** (semitones / 12.0)

        # Resample pour changer le pitch (change aussi la durée)
        original_length = len(audio)
        new_length = int(original_length / factor)

        if new_length <= 0:
            return audio

        # Interpolation pour le resampling
        x_original = np.linspace(0, 1, original_length)
        x_new = np.linspace(0, 1, new_length)
        resampled = np.interp(x_new, x_original, audio)

        # Ré-interpoler pour revenir à la longueur originale (time-stretch)
        x_resampled = np.linspace(0, 1, len(resampled))
        x_final = np.linspace(0, 1, original_length)
        result = np.interp(x_final, x_resampled, resampled)

        return result.astype(np.float32)

    def _crossfade_segments(
        self,
        segments: list[np.ndarray],
        fade_samples: int = None
    ) -> np.ndarray:
        """
        Recombine les segments avec crossfade.

        Args:
            segments: Liste de segments audio
            fade_samples: Nombre d'échantillons pour le crossfade

        Returns:
            Audio combiné
        """
        if not segments:
            return np.array([], dtype=np.float32)

        if len(segments) == 1:
            return segments[0].astype(np.float32)

        # Calculer la taille du fade (10ms par défaut)
        if fade_samples is None:
            fade_samples = int(0.01 * self.sample_rate)

        result = segments[0].copy()

        for seg in segments[1:]:
            if len(seg) == 0:
                continue

            # Ajuster le fade si les segments sont trop courts
            actual_fade = min(fade_samples, len(result) // 4, len(seg) // 4)

            if actual_fade < 2:
                # Pas assez d'échantillons pour un fade, concaténer directement
                result = np.concatenate([result, seg])
            else:
                # Créer le crossfade
                t = np.linspace(0, np.pi / 2, actual_fade)
                fade_out = np.cos(t) ** 2
                fade_in = np.sin(t) ** 2

                # Zone de crossfade
                overlap = (result[-actual_fade:] * fade_out +
                           seg[:actual_fade] * fade_in)

                # Combiner
                result = np.concatenate([
                    result[:-actual_fade],
                    overlap,
                    seg[actual_fade:]
                ])

        return result.astype(np.float32)


class IntonationProcessor:
    """Processeur d'intonation complet (détection + application)."""

    def __init__(
        self,
        sample_rate: int = 24000,
        language: str = "fr",
        strength: float = 0.7,
        enabled: bool = True
    ):
        """
        Initialise le processeur.

        Args:
            sample_rate: Taux d'échantillonnage
            language: Code de langue
            strength: Force des contours (0.0 à 1.0)
            enabled: Activer le traitement
        """
        self.enabled = enabled
        self.strength = strength

        self.detector = IntonationContourDetector(language)
        self.applicator = IntonationContourApplicator(sample_rate, strength)

    def process(
        self,
        audio: np.ndarray,
        text: str,
        strength: float = None
    ) -> np.ndarray:
        """
        Détecte et applique le contour approprié.

        Args:
            audio: Signal audio
            text: Texte correspondant
            strength: Force du contour (override)

        Returns:
            Audio avec contour appliqué
        """
        if not self.enabled or len(audio) == 0:
            return audio

        # Détecter le contour
        contour = self.detector.detect(text)

        # Appliquer
        return self.applicator.apply_contour(
            audio,
            contour,
            strength=strength or self.strength
        )

    def detect_contour(self, text: str) -> IntonationContour:
        """Détecte le contour sans appliquer."""
        return self.detector.detect(text)
