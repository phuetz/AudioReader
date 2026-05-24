"""
Selecteur intelligent de moteur TTS.

Choisit automatiquement le meilleur moteur en fonction du contexte:
- Longueur du texte
- Presence de dialogues
- Besoin de clonage
- Ressources disponibles
"""
from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class SelectionContext:
    """Contexte pour la selection du moteur."""
    word_count: int = 0
    has_dialogues: bool = False
    has_clone_voice: bool = False
    language: str = "fr"
    prefer_quality: bool = False
    prefer_speed: bool = False


class EngineSelector:
    """
    Selecteur intelligent qui choisit le meilleur moteur TTS
    en fonction du contexte de la conversion.
    """

    # Priorites par scenario
    RULES = [
        # (condition_fn, engine, raison)
    ]

    def __init__(self):
        self._available_cache: Dict[str, bool] = {}

    def _check_available(self, engine: str) -> bool:
        """Verifie si un moteur est disponible (avec cache)."""
        if engine in self._available_cache:
            return self._available_cache[engine]

        available = False
        try:
            if engine == "kokoro":
                from src.tts_kokoro_engine import KokoroEngine
                available = True
            elif engine == "chatterbox":
                from src.tts_chatterbox_engine import ChatterboxEngine
                available = ChatterboxEngine.is_available()
            elif engine == "dia":
                from src.tts_dia_engine import DiaEngine
                available = DiaEngine.is_available()
            elif engine == "f5":
                from src.tts_f5_engine import F5Engine
                available = F5Engine.is_available()
            elif engine == "qwen3":
                from src.tts_qwen3_engine import Qwen3Engine
                available = Qwen3Engine().is_available()
            elif engine == "voxtral":
                from src.tts_voxtral_engine import VoxtralEngine
                available = VoxtralEngine().is_available()
            elif engine == "xtts":
                from src.tts_xtts_engine import XTTSEngine
                available = True
            elif engine == "edge":
                import edge_tts
                available = True
            elif engine == "mms":
                available = True
        except ImportError:
            available = False

        self._available_cache[engine] = available
        return available

    def select(self, context: SelectionContext) -> str:
        """
        Selectionne le meilleur moteur pour le contexte donne.

        Regles de selection:
        1. Clonage voix -> Chatterbox (qualite) > F5 > XTTS
        2. Dialogues multi-speakers -> Dia (natif) > Kokoro
        3. Court (<5000 mots) -> Kokoro (vitesse)
        4. Long (>20000 mots) -> Kokoro (coherence)
        5. Defaut -> Kokoro
        """
        # Clonage de voix
        if context.has_clone_voice:
            for eng in ["chatterbox", "qwen3", "voxtral", "f5", "xtts"]:
                if self._check_available(eng):
                    return eng

        # Dialogues multi-speakers
        if context.has_dialogues and not context.prefer_speed:
            if self._check_available("dia"):
                return "dia"

        # Qualite preferee
        if context.prefer_quality:
            for eng in ["chatterbox", "qwen3", "voxtral", "f5", "kokoro"]:
                if self._check_available(eng):
                    return eng

        # Defaut: Kokoro (rapide et fiable)
        if self._check_available("kokoro"):
            return "kokoro"

        # Fallbacks
        for eng in ["mms", "edge"]:
            if self._check_available(eng):
                return eng

        return "kokoro"  # Retourne kokoro meme si non disponible

    def get_recommendation(self, context: SelectionContext) -> Dict:
        """Retourne une recommandation detaillee."""
        selected = self.select(context)
        reasons = []

        if context.has_clone_voice:
            reasons.append("Clonage de voix demande")
        if context.has_dialogues:
            reasons.append("Dialogues multi-speakers detectes")
        if context.word_count < 5000:
            reasons.append("Texte court (vitesse prioritaire)")
        elif context.word_count > 20000:
            reasons.append("Texte long (coherence prioritaire)")

        return {
            "engine": selected,
            "available": self._check_available(selected),
            "reasons": reasons,
            "context": {
                "word_count": context.word_count,
                "has_dialogues": context.has_dialogues,
                "has_clone_voice": context.has_clone_voice,
            },
        }
