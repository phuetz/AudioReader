"""
Chargeur de corrections depuis fichiers JSON.

Charge et applique les corrections de transcription depuis des glossaires JSON
comme celui créé par Lisa.

Usage:
    from corrections_loader import load_corrections, apply_corrections

    corrections = load_corrections("glossaire_corrections_v1.json")
    text = apply_corrections(text, corrections)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Callable, Union
from dataclasses import dataclass


@dataclass
class CorrectionRule:
    """Règle de correction."""
    id: str
    pattern: str
    replacement: str
    confidence: str  # high, medium, low
    action: str  # replace, flag
    flags: List[str]
    section: str = ""
    notes: str = ""
    alternatives: List[str] = None

    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []


class CorrectionEngine:
    """Moteur de corrections basé sur des règles JSON."""

    def __init__(self, rules: List[CorrectionRule] = None):
        self.rules = rules or []
        self._compiled_patterns: Dict[str, re.Pattern] = {}

    def _compile_pattern(self, rule: CorrectionRule) -> re.Pattern:
        """Compile le pattern regex d'une règle."""
        if rule.id not in self._compiled_patterns:
            flags = 0
            if "i" in rule.flags:
                flags |= re.IGNORECASE
            if "m" in rule.flags:
                flags |= re.MULTILINE
            self._compiled_patterns[rule.id] = re.compile(rule.pattern, flags)
        return self._compiled_patterns[rule.id]

    def apply(
        self,
        text: str,
        confidence_levels: List[str] = None,
        actions: List[str] = None
    ) -> str:
        """
        Applique les corrections au texte.

        Args:
            text: Texte à corriger
            confidence_levels: Niveaux de confiance à appliquer (défaut: ["high"])
            actions: Actions à appliquer (défaut: ["replace"])

        Returns:
            Texte corrigé
        """
        if confidence_levels is None:
            confidence_levels = ["high"]
        if actions is None:
            actions = ["replace"]

        for rule in self.rules:
            if rule.confidence not in confidence_levels:
                continue
            if rule.action not in actions:
                continue

            pattern = self._compile_pattern(rule)
            text = pattern.sub(rule.replacement, text)

        return text

    def get_stats(self) -> Dict:
        """Retourne des statistiques sur les règles."""
        stats = {
            "total": len(self.rules),
            "by_confidence": {},
            "by_action": {},
            "by_section": {}
        }

        for rule in self.rules:
            # Par confiance
            if rule.confidence not in stats["by_confidence"]:
                stats["by_confidence"][rule.confidence] = 0
            stats["by_confidence"][rule.confidence] += 1

            # Par action
            if rule.action not in stats["by_action"]:
                stats["by_action"][rule.action] = 0
            stats["by_action"][rule.action] += 1

            # Par section
            if rule.section not in stats["by_section"]:
                stats["by_section"][rule.section] = 0
            stats["by_section"][rule.section] += 1

        return stats


def load_corrections(path: Union[str, Path]) -> CorrectionEngine:
    """
    Charge un fichier de corrections JSON.

    Args:
        path: Chemin vers le fichier JSON

    Returns:
        CorrectionEngine configuré
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier de corrections non trouvé: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    rules = []
    for rule_data in data.get("rules", []):
        rule = CorrectionRule(
            id=rule_data.get("id", ""),
            pattern=rule_data.get("pattern", ""),
            replacement=rule_data.get("replacement", ""),
            confidence=rule_data.get("confidence", "low"),
            action=rule_data.get("action", "flag"),
            flags=rule_data.get("flags", []),
            section=rule_data.get("section", ""),
            notes=rule_data.get("notes", ""),
            alternatives=rule_data.get("alternatives", [])
        )
        rules.append(rule)

    return CorrectionEngine(rules)


def apply_corrections(
    text: str,
    engine: CorrectionEngine,
    confidence_levels: List[str] = None,
    actions: List[str] = None
) -> str:
    """
    Applique les corrections au texte.

    Args:
        text: Texte à corriger
        engine: Moteur de corrections
        confidence_levels: Niveaux à appliquer (défaut: ["high"])
        actions: Actions à appliquer (défaut: ["replace"])

    Returns:
        Texte corrigé
    """
    return engine.apply(text, confidence_levels, actions)


def create_correction_func(
    json_path: Union[str, Path],
    confidence_levels: List[str] = None,
    include_medium: bool = False
) -> Callable[[str], str]:
    """
    Crée une fonction de correction pour le moteur hybride TTS.

    Args:
        json_path: Chemin vers le fichier JSON
        confidence_levels: Niveaux de confiance à appliquer
        include_medium: Inclure les corrections "medium" (défaut: False)

    Returns:
        Fonction de correction (text) -> text
    """
    engine = load_corrections(json_path)

    if confidence_levels is None:
        confidence_levels = ["high"]
        if include_medium:
            confidence_levels.append("medium")

    def correction_func(text: str) -> str:
        return engine.apply(text, confidence_levels, ["replace"])

    return correction_func


# Chemin par défaut vers le glossaire Lisa
DEFAULT_GLOSSARY_PATH = Path(__file__).parent.parent / "Lisa" / "glossaire_corrections_v1.json"


def get_default_corrections() -> Optional[CorrectionEngine]:
    """Charge les corrections par défaut si disponibles."""
    if DEFAULT_GLOSSARY_PATH.exists():
        return load_corrections(DEFAULT_GLOSSARY_PATH)
    return None


# Fonction de correction par défaut pour le moteur hybride
def apply_default_corrections(text: str) -> str:
    """Applique les corrections par défaut (high confidence)."""
    engine = get_default_corrections()
    if engine:
        return engine.apply(text, ["high"], ["replace"])
    return text


if __name__ == "__main__":
    # Test
    print("=== Test du chargeur de corrections ===\n")

    if DEFAULT_GLOSSARY_PATH.exists():
        engine = load_corrections(DEFAULT_GLOSSARY_PATH)
        stats = engine.get_stats()

        print(f"Fichier: {DEFAULT_GLOSSARY_PATH.name}")
        print(f"Total règles: {stats['total']}")
        print(f"\nPar confiance:")
        for level, count in stats["by_confidence"].items():
            print(f"  {level}: {count}")
        print(f"\nPar action:")
        for action, count in stats["by_action"].items():
            print(f"  {action}: {count}")

        # Test de corrections
        print("\n=== Tests ===\n")
        tests = [
            "Il avait 20 France dans sa poche.",
            "Elle fumait des Cogoises.",
            "Son Loyer hum était cher.",
            "Il a gagné Nu for fune.",
            "Les yeux soulignés de flou.",
            "Direction Du bail pour les affaires.",
        ]

        for t in tests:
            corrected = engine.apply(t, ["high"], ["replace"])
            if t != corrected:
                print(f"AVANT: {t}")
                print(f"APRÈS: {corrected}")
                print()
    else:
        print(f"Fichier non trouvé: {DEFAULT_GLOSSARY_PATH}")
