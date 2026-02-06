"""
Détection d'incohérences pour AudioReader.

Analyse les segments d'un audiobook pour détecter:
- Changements de voix pour un même personnage
- Attributions contradictoires de dialogues
- Sauts d'émotion brusques
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class ConsistencyIssue:
    """Un problème de cohérence détecté."""
    type: str  # voice_change, attribution_conflict, emotion_jump
    severity: str  # info, warning, error
    segment_indices: List[int] = field(default_factory=list)
    description: str = ""
    suggestion: str = ""


@dataclass
class ConsistencyReport:
    """Rapport de cohérence."""
    score: float = 100.0  # 0-100
    issues: List[ConsistencyIssue] = field(default_factory=list)
    total_segments: int = 0


class ConsistencyChecker:
    """
    Vérifie la cohérence d'un ensemble de segments audio.

    Détecte les incohérences de voix, d'attribution et d'émotion.
    """

    # Transitions émotionnelles suspectes
    SUSPICIOUS_TRANSITIONS = {
        ("joy", "sadness"): "warning",
        ("sadness", "joy"): "info",
        ("fear", "joy"): "info",
        ("anger", "joy"): "info",
        ("joy", "anger"): "warning",
        ("neutral", "anger"): "info",
        ("calm", "fear"): "warning",
    }

    def check_all(self, segments: List[dict]) -> ConsistencyReport:
        """
        Analyse complète de cohérence.

        Args:
            segments: Liste de dicts avec clés 'speaker', 'voice_id', 'emotion', etc.

        Returns:
            ConsistencyReport
        """
        issues = []
        issues.extend(self.check_voice_consistency(segments))
        issues.extend(self.check_emotion_continuity(segments))
        issues.extend(self.check_attribution_conflicts(segments))

        # Calcul du score
        penalty = sum(
            {"error": 10, "warning": 5, "info": 2}.get(i.severity, 0)
            for i in issues
        )
        score = max(0.0, 100.0 - penalty)

        return ConsistencyReport(
            score=round(score, 1),
            issues=issues,
            total_segments=len(segments),
        )

    def check_voice_consistency(self, segments: List[dict]) -> List[ConsistencyIssue]:
        """Vérifie que chaque personnage garde la même voix."""
        issues = []
        speaker_voices: Dict[str, Dict[str, List[int]]] = {}

        for i, seg in enumerate(segments):
            speaker = seg.get("speaker")
            voice = seg.get("voice_id", "")
            if not speaker or not voice:
                continue

            if speaker not in speaker_voices:
                speaker_voices[speaker] = {}
            speaker_voices[speaker].setdefault(voice, []).append(i)

        for speaker, voices in speaker_voices.items():
            if len(voices) > 1:
                # Ce personnage a plusieurs voix
                all_indices = []
                for indices in voices.values():
                    all_indices.extend(indices)
                voice_list = ", ".join(voices.keys())
                issues.append(ConsistencyIssue(
                    type="voice_change",
                    severity="warning",
                    segment_indices=sorted(all_indices),
                    description=(
                        f"{speaker} utilise des voix différentes: {voice_list}"
                    ),
                    suggestion=(
                        f"Uniformiser la voix de {speaker} sur tous les segments."
                    ),
                ))

        return issues

    def check_emotion_continuity(self, segments: List[dict]) -> List[ConsistencyIssue]:
        """Détecte les sauts d'émotion brusques."""
        issues = []
        prev_emotion: Optional[str] = None

        for i, seg in enumerate(segments):
            emotion = seg.get("emotion")
            if emotion and prev_emotion:
                pair = (prev_emotion, emotion)
                if pair in self.SUSPICIOUS_TRANSITIONS:
                    severity = self.SUSPICIOUS_TRANSITIONS[pair]
                    issues.append(ConsistencyIssue(
                        type="emotion_jump",
                        severity=severity,
                        segment_indices=[i - 1, i],
                        description=(
                            f"Transition émotionnelle brusque: "
                            f"{prev_emotion} → {emotion}"
                        ),
                        suggestion=(
                            "Vérifier si cette transition est intentionnelle "
                            "ou si un segment intermédiaire manque."
                        ),
                    ))
            if emotion:
                prev_emotion = emotion

        return issues

    def check_attribution_conflicts(self, segments: List[dict]) -> List[ConsistencyIssue]:
        """Détecte les attributions contradictoires."""
        issues = []
        seen_speakers: Set[str] = set()
        total = len(segments)

        for i, seg in enumerate(segments):
            speaker = seg.get("speaker")
            if not speaker:
                continue

            # Nouveau personnage apparaissant très tard
            if speaker not in seen_speakers and i > total * 0.8 and total > 10:
                issues.append(ConsistencyIssue(
                    type="attribution_conflict",
                    severity="info",
                    segment_indices=[i],
                    description=(
                        f"Nouveau personnage '{speaker}' apparaît tard "
                        f"(segment {i + 1}/{total})"
                    ),
                    suggestion=(
                        "Vérifier que ce n'est pas une erreur d'attribution."
                    ),
                ))

            seen_speakers.add(speaker)

        # Vérifier les alternances rapides improbables
        if total >= 4:
            for i in range(2, total):
                s0 = segments[i - 2].get("speaker")
                s1 = segments[i - 1].get("speaker")
                s2 = segments[i].get("speaker")
                if s0 and s1 and s2 and s0 == s2 and s0 != s1:
                    # A-B-A pattern, normal pour un dialogue
                    pass
                elif (
                    s0 and s1 and s2
                    and s0 != s1 and s1 != s2 and s0 != s2
                    and i + 1 < total
                ):
                    s3 = segments[i + 1].get("speaker") if i + 1 < total else None
                    if s3 and s3 != s0 and s3 != s1 and s3 != s2:
                        # 4 personnages différents consécutifs
                        issues.append(ConsistencyIssue(
                            type="attribution_conflict",
                            severity="info",
                            segment_indices=[i - 2, i - 1, i, i + 1],
                            description=(
                                f"4 personnages différents consécutifs "
                                f"({s0}, {s1}, {s2}, {s3})"
                            ),
                            suggestion=(
                                "Vérifier les attributions — "
                                "rare d'avoir 4 locuteurs consécutifs différents."
                            ),
                        ))

        return issues
