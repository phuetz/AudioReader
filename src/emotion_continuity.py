"""
Gestion de la continuite emotionnelle entre segments.

Fonctionnalites:
- Lissage des transitions emotionnelles
- Detection des changements brusques
- Ajustement progressif de la prosodie
- Coherence inter-chapitres
"""
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum

from .emotion_analyzer import Emotion, Intensity, ProsodyHints


@dataclass
class EmotionState:
    """Etat emotionnel courant."""
    emotion: Emotion
    intensity: Intensity
    momentum: float = 0.0  # -1 (decroissant) a +1 (croissant)
    duration: int = 0      # Nombre de segments avec cette emotion


class TransitionType(Enum):
    """Types de transitions emotionnelles."""
    SMOOTH = "smooth"           # Transition douce
    GRADUAL = "gradual"         # Transition progressive
    SUDDEN = "sudden"           # Changement brusque (voulu)
    DRAMATIC = "dramatic"       # Climax dramatique
    RESOLUTION = "resolution"   # Retour au calme


@dataclass
class EmotionTransition:
    """Transition entre deux etats emotionnels."""
    from_emotion: Emotion
    to_emotion: Emotion
    transition_type: TransitionType
    pause_duration: float       # Pause suggeree
    speed_adjustment: float     # Ajustement de vitesse


class EmotionContinuityManager:
    """
    Gere la continuite emotionnelle entre segments.

    Objectifs:
    - Eviter les changements de ton trop brusques
    - Creer des crescendos/decrescendos naturels
    - Maintenir la coherence narrative
    """

    # Compatibilite entre emotions (transitions naturelles)
    EMOTION_COMPATIBILITY = {
        Emotion.NEUTRAL: [Emotion.JOY, Emotion.SADNESS, Emotion.SUSPENSE],
        Emotion.JOY: [Emotion.EXCITEMENT, Emotion.TENDERNESS, Emotion.NEUTRAL],
        Emotion.SADNESS: [Emotion.TENDERNESS, Emotion.NEUTRAL, Emotion.ANGER],
        Emotion.ANGER: [Emotion.FEAR, Emotion.SADNESS, Emotion.NEUTRAL],
        Emotion.FEAR: [Emotion.SUSPENSE, Emotion.ANGER, Emotion.NEUTRAL],
        Emotion.SURPRISE: [Emotion.JOY, Emotion.FEAR, Emotion.ANGER],
        Emotion.EXCITEMENT: [Emotion.JOY, Emotion.SURPRISE, Emotion.SUSPENSE],
        Emotion.TENDERNESS: [Emotion.JOY, Emotion.SADNESS, Emotion.NEUTRAL],
        Emotion.SUSPENSE: [Emotion.FEAR, Emotion.SURPRISE, Emotion.EXCITEMENT],
        Emotion.DISGUST: [Emotion.ANGER, Emotion.FEAR, Emotion.NEUTRAL],
        Emotion.IRONY: [Emotion.NEUTRAL, Emotion.JOY, Emotion.ANGER],
    }

    # Poids emotionnel (pour calcul d'intensite)
    EMOTION_WEIGHT = {
        Emotion.NEUTRAL: 0.0,
        Emotion.JOY: 0.6,
        Emotion.SADNESS: 0.5,
        Emotion.ANGER: 0.8,
        Emotion.FEAR: 0.7,
        Emotion.SURPRISE: 0.6,
        Emotion.EXCITEMENT: 0.9,
        Emotion.TENDERNESS: 0.4,
        Emotion.SUSPENSE: 0.7,
        Emotion.DISGUST: 0.6,
        Emotion.IRONY: 0.3,
    }

    def __init__(self, smoothing_factor: float = 0.3):
        """
        Args:
            smoothing_factor: Facteur de lissage (0=pas de lissage, 1=lissage max)
        """
        self.smoothing_factor = smoothing_factor
        self._state_history: List[EmotionState] = []
        self._current_state: Optional[EmotionState] = None

    def _calculate_emotional_distance(
        self,
        from_emotion: Emotion,
        to_emotion: Emotion
    ) -> float:
        """Calcule la distance emotionnelle entre deux emotions."""
        if from_emotion == to_emotion:
            return 0.0

        # Transition naturelle
        if to_emotion in self.EMOTION_COMPATIBILITY.get(from_emotion, []):
            return 0.3

        # Transition neutre (via neutral)
        if from_emotion == Emotion.NEUTRAL or to_emotion == Emotion.NEUTRAL:
            return 0.5

        # Transition opposee
        opposites = {
            Emotion.JOY: Emotion.SADNESS,
            Emotion.ANGER: Emotion.TENDERNESS,
            Emotion.FEAR: Emotion.JOY,
            Emotion.EXCITEMENT: Emotion.SADNESS,
        }
        if opposites.get(from_emotion) == to_emotion:
            return 1.0
        if opposites.get(to_emotion) == from_emotion:
            return 1.0

        return 0.7

    def _determine_transition_type(
        self,
        from_state: EmotionState,
        to_emotion: Emotion,
        to_intensity: Intensity
    ) -> TransitionType:
        """Determine le type de transition appropriee."""
        distance = self._calculate_emotional_distance(
            from_state.emotion,
            to_emotion
        )

        # Retour au calme apres emotion forte
        if (from_state.intensity in [Intensity.HIGH, Intensity.EXTREME]
                and to_intensity in [Intensity.LOW, Intensity.MEDIUM]
                and to_emotion in [Emotion.NEUTRAL, Emotion.TENDERNESS]):
            return TransitionType.RESOLUTION

        # Climax dramatique
        if (to_intensity == Intensity.EXTREME
                and to_emotion in [Emotion.FEAR, Emotion.ANGER, Emotion.EXCITEMENT]):
            return TransitionType.DRAMATIC

        # Changement brusque
        if distance > 0.8:
            return TransitionType.SUDDEN

        # Transition progressive
        if distance > 0.4:
            return TransitionType.GRADUAL

        return TransitionType.SMOOTH

    def _calculate_transition_pause(
        self,
        transition_type: TransitionType,
        distance: float
    ) -> float:
        """Calcule la pause appropriee pour la transition."""
        base_pauses = {
            TransitionType.SMOOTH: 0.2,
            TransitionType.GRADUAL: 0.4,
            TransitionType.SUDDEN: 0.6,
            TransitionType.DRAMATIC: 0.8,
            TransitionType.RESOLUTION: 0.5,
        }
        return base_pauses.get(transition_type, 0.3) * (1 + distance * 0.5)

    def _calculate_speed_adjustment(
        self,
        transition_type: TransitionType,
        to_emotion: Emotion,
        to_intensity: Intensity
    ) -> float:
        """Calcule l'ajustement de vitesse pour la transition."""
        # Ajustement de base selon le type
        if transition_type == TransitionType.DRAMATIC:
            return 0.95  # Legerement plus lent pour le drame
        elif transition_type == TransitionType.RESOLUTION:
            return 0.9   # Plus lent pour la resolution
        elif transition_type == TransitionType.SUDDEN:
            return 1.0   # Pas de changement (le contraste est voulu)

        # Ajustement selon l'emotion cible
        emotion_speeds = {
            Emotion.EXCITEMENT: 1.1,
            Emotion.ANGER: 1.05,
            Emotion.SADNESS: 0.92,
            Emotion.TENDERNESS: 0.95,
            Emotion.SUSPENSE: 0.88,
        }
        return emotion_speeds.get(to_emotion, 1.0)

    def process_emotion(
        self,
        emotion: Emotion,
        intensity: Intensity
    ) -> EmotionTransition:
        """
        Traite une nouvelle emotion et retourne la transition.

        Args:
            emotion: Emotion detectee
            intensity: Intensite de l'emotion

        Returns:
            EmotionTransition avec suggestions de prosodie
        """
        if self._current_state is None:
            # Premier segment
            self._current_state = EmotionState(
                emotion=emotion,
                intensity=intensity,
                duration=1
            )
            return EmotionTransition(
                from_emotion=Emotion.NEUTRAL,
                to_emotion=emotion,
                transition_type=TransitionType.SMOOTH,
                pause_duration=0.3,
                speed_adjustment=1.0
            )

        # Calculer la transition
        transition_type = self._determine_transition_type(
            self._current_state,
            emotion,
            intensity
        )

        distance = self._calculate_emotional_distance(
            self._current_state.emotion,
            emotion
        )

        pause = self._calculate_transition_pause(transition_type, distance)
        speed = self._calculate_speed_adjustment(transition_type, emotion, intensity)

        transition = EmotionTransition(
            from_emotion=self._current_state.emotion,
            to_emotion=emotion,
            transition_type=transition_type,
            pause_duration=pause,
            speed_adjustment=speed
        )

        # Mettre a jour l'etat
        self._state_history.append(self._current_state)

        if emotion == self._current_state.emotion:
            self._current_state.duration += 1
            # Ajuster le momentum
            if intensity.value > self._current_state.intensity.value:
                self._current_state.momentum = min(1.0, self._current_state.momentum + 0.2)
            elif intensity.value < self._current_state.intensity.value:
                self._current_state.momentum = max(-1.0, self._current_state.momentum - 0.2)
            self._current_state.intensity = intensity
        else:
            self._current_state = EmotionState(
                emotion=emotion,
                intensity=intensity,
                duration=1
            )

        return transition

    def smooth_prosody(
        self,
        current_prosody: ProsodyHints,
        previous_prosody: Optional[ProsodyHints]
    ) -> ProsodyHints:
        """
        Lisse la prosodie pour eviter les changements brusques.

        Args:
            current_prosody: Prosodie calculee pour le segment actuel
            previous_prosody: Prosodie du segment precedent

        Returns:
            Prosodie lissee
        """
        if previous_prosody is None:
            return current_prosody

        factor = self.smoothing_factor

        # Lisser la vitesse
        smoothed_speed = (
            current_prosody.speed * (1 - factor) +
            previous_prosody.speed * factor
        )

        # Lisser le pitch
        smoothed_pitch = (
            current_prosody.pitch * (1 - factor) +
            previous_prosody.pitch * factor
        )

        # Lisser le volume
        smoothed_volume = (
            current_prosody.volume * (1 - factor) +
            previous_prosody.volume * factor
        )

        return ProsodyHints(
            speed=smoothed_speed,
            pitch=smoothed_pitch,
            volume=smoothed_volume,
            pause_before=current_prosody.pause_before,
            pause_after=current_prosody.pause_after,
            breath_before=current_prosody.breath_before
        )

    def get_emotional_arc(self) -> List[EmotionState]:
        """Retourne l'arc emotionnel complet (historique)."""
        result = list(self._state_history)
        if self._current_state:
            result.append(self._current_state)
        return result

    def reset(self):
        """Reinitialise le manager (nouveau chapitre)."""
        self._state_history.clear()
        self._current_state = None


class ChapterEmotionTracker:
    """
    Suit les emotions au niveau du chapitre.

    Permet de:
    - Identifier les climax emotionnels
    - Detecter le ton general du chapitre
    - Ajuster la prosodie globale
    """

    def __init__(self):
        self.emotions_count: dict[Emotion, int] = {}
        self.max_intensity: Intensity = Intensity.LOW
        self.climax_positions: List[int] = []
        self.segment_count: int = 0

    def track(
        self,
        segment_index: int,
        emotion: Emotion,
        intensity: Intensity
    ):
        """Enregistre l'emotion d'un segment."""
        self.emotions_count[emotion] = self.emotions_count.get(emotion, 0) + 1
        self.segment_count += 1

        # Detecter les climax
        if intensity == Intensity.EXTREME:
            self.climax_positions.append(segment_index)

        # Mettre a jour l'intensite max
        if intensity.value > self.max_intensity.value:
            self.max_intensity = intensity

    def get_dominant_emotion(self) -> Emotion:
        """Retourne l'emotion dominante du chapitre."""
        if not self.emotions_count:
            return Emotion.NEUTRAL

        return max(self.emotions_count, key=self.emotions_count.get)

    def get_chapter_tone(self) -> str:
        """Determine le ton general du chapitre."""
        dominant = self.get_dominant_emotion()

        tone_mapping = {
            Emotion.JOY: "joyeux",
            Emotion.SADNESS: "melancolique",
            Emotion.ANGER: "tendu",
            Emotion.FEAR: "angoissant",
            Emotion.SUSPENSE: "haletant",
            Emotion.TENDERNESS: "tendre",
            Emotion.EXCITEMENT: "exaltant",
            Emotion.NEUTRAL: "neutre",
        }

        return tone_mapping.get(dominant, "mixte")

    def get_suggested_base_speed(self) -> float:
        """Suggere une vitesse de base pour le chapitre."""
        dominant = self.get_dominant_emotion()

        speed_mapping = {
            Emotion.EXCITEMENT: 1.05,
            Emotion.SUSPENSE: 0.95,
            Emotion.SADNESS: 0.92,
            Emotion.TENDERNESS: 0.95,
            Emotion.ANGER: 1.02,
        }

        return speed_mapping.get(dominant, 1.0)


def apply_emotion_continuity(
    segments: list,
    smoothing_factor: float = 0.3
) -> list:
    """
    Applique la continuite emotionnelle a une liste de segments.

    Args:
        segments: Liste de segments avec emotions
        smoothing_factor: Facteur de lissage

    Returns:
        Segments avec prosodie ajustee
    """
    manager = EmotionContinuityManager(smoothing_factor)
    previous_prosody = None

    for segment in segments:
        # Traiter la transition
        transition = manager.process_emotion(
            segment.emotion,
            segment.intensity
        )

        # Ajuster la pause avant selon la transition
        if transition.pause_duration > segment.pause_before:
            segment.pause_before = transition.pause_duration

        # Lisser la prosodie
        if hasattr(segment, 'prosody') and segment.prosody:
            segment.prosody = manager.smooth_prosody(
                segment.prosody,
                previous_prosody
            )
            # Appliquer l'ajustement de vitesse de transition
            segment.prosody.speed *= transition.speed_adjustment
            previous_prosody = segment.prosody

    return segments


if __name__ == "__main__":
    # Test
    from .emotion_analyzer import Emotion, Intensity

    print("=== Test continuite emotionnelle ===\n")

    manager = EmotionContinuityManager()

    # Simuler une sequence emotionnelle
    sequence = [
        (Emotion.NEUTRAL, Intensity.LOW),
        (Emotion.SUSPENSE, Intensity.MEDIUM),
        (Emotion.SUSPENSE, Intensity.HIGH),
        (Emotion.FEAR, Intensity.EXTREME),  # Climax
        (Emotion.FEAR, Intensity.HIGH),
        (Emotion.NEUTRAL, Intensity.MEDIUM),  # Resolution
        (Emotion.TENDERNESS, Intensity.MEDIUM),
    ]

    print("Sequence emotionnelle:")
    for i, (emotion, intensity) in enumerate(sequence):
        transition = manager.process_emotion(emotion, intensity)
        print(f"  {i+1}. {emotion.value:12} ({intensity.value:6}) "
              f"-> {transition.transition_type.value:10} "
              f"pause={transition.pause_duration:.2f}s "
              f"speed={transition.speed_adjustment:.2f}")

    print("\nArc emotionnel:")
    for state in manager.get_emotional_arc():
        print(f"  {state.emotion.value}: {state.duration} segments, "
              f"momentum={state.momentum:.2f}")
