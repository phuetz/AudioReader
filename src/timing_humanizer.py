"""
Timing Humanizer Module v2.3.

Ajoute des micro-variations de timing pour éviter la régularité mécanique
et donner un rythme plus naturel à la synthèse vocale.

Fonctionnalités:
- Variation gaussienne des durées de pause
- Rythme basé sur la structure syntaxique
- Micro-pauses avant les mots importants
- Alternance légère rapide/lent entre phrases
"""
import re
import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ClauseType(Enum):
    """Types de clauses syntaxiques."""

    MAIN = "main"                    # Clause principale
    SUBORDINATE = "subordinate"      # Clause subordonnée
    PARENTHETICAL = "parenthetical"  # Incise (entre virgules/tirets)
    QUOTE = "quote"                  # Citation/dialogue
    EMPHASIS = "emphasis"            # Contient un mot d'emphase


@dataclass
class TimingConfig:
    """Configuration du timing humanisé."""

    # Variation des pauses
    pause_variation_sigma: float = 0.05   # Écart-type de la variation (5%)
    pause_variation_min: float = 0.85     # Minimum relatif (85%)
    pause_variation_max: float = 1.15     # Maximum relatif (115%)

    # Rythme par type de clause
    main_clause_speed: float = 1.0        # Vitesse clause principale
    subordinate_speed: float = 1.05       # Légèrement plus rapide
    parenthetical_speed: float = 1.08     # Encore plus rapide
    quote_speed: float = 0.95             # Légèrement plus lent (emphase)

    # Micro-pauses
    enable_emphasis_pauses: bool = True
    emphasis_pause_duration: float = 0.05  # 50ms avant mots importants

    # Variation inter-phrase
    enable_inter_phrase_variation: bool = True
    inter_phrase_variation: float = 0.03   # ±3% entre phrases


# Mots déclencheurs de micro-pauses (emphase)
EMPHASIS_WORDS_FR = {
    # Adverbes d'intensité
    "jamais", "toujours", "absolument", "vraiment", "totalement",
    "complètement", "parfaitement", "exactement", "certainement",
    # Marqueurs temporels forts
    "soudain", "soudainement", "brusquement", "subitement",
    "enfin", "finalement", "désormais",
    # Connecteurs forts
    "mais", "cependant", "pourtant", "néanmoins", "toutefois",
    "or", "donc", "ainsi", "alors",
    # Mots émotionnels
    "terrible", "incroyable", "extraordinaire", "magnifique",
    "horrible", "merveilleux", "épouvantable",
    # Négations fortes
    "rien", "personne", "aucun", "aucune", "nul", "nulle",
    # Autres emphases
    "même", "seul", "unique", "premier", "dernier",
}

# Marqueurs de clause subordonnée (français)
SUBORDINATE_MARKERS_FR = {
    "qui", "que", "dont", "où", "lequel", "laquelle", "lesquels", "lesquelles",
    "quand", "lorsque", "pendant que", "tandis que", "alors que",
    "parce que", "puisque", "comme", "si", "quoique", "bien que",
    "afin que", "pour que", "avant que", "après que", "depuis que",
    "à condition que", "pourvu que", "sans que", "de sorte que",
}


class TimingHumanizer:
    """Humanise le timing de la synthèse vocale."""

    def __init__(self, config: TimingConfig = None, language: str = "fr"):
        """
        Initialise l'humaniseur de timing.

        Args:
            config: Configuration du timing
            language: Code de langue
        """
        self.config = config or TimingConfig()
        self.language = language

        # Compteur pour l'alternance inter-phrase
        self._phrase_counter = 0

    def humanize_pause(self, base_pause: float) -> float:
        """
        Ajoute une variation gaussienne à une durée de pause.

        Args:
            base_pause: Durée de base en secondes

        Returns:
            Durée modifiée avec variation naturelle
        """
        if base_pause <= 0:
            return base_pause

        # Variation gaussienne centrée sur 1.0
        variation = random.gauss(1.0, self.config.pause_variation_sigma)

        # Limiter aux bornes configurées
        variation = max(self.config.pause_variation_min, variation)
        variation = min(self.config.pause_variation_max, variation)

        return base_pause * variation

    def get_clause_type(self, text: str) -> ClauseType:
        """
        Détecte le type de clause syntaxique.

        Args:
            text: Texte de la clause

        Returns:
            Type de clause détecté
        """
        text_lower = text.lower().strip()

        # Vérifier si c'est un dialogue/citation
        if text.startswith('"') or text.startswith('«') or text.startswith('—'):
            return ClauseType.QUOTE

        # Vérifier si c'est une incise (entre virgules, parenthèses, tirets)
        if self._is_parenthetical(text):
            return ClauseType.PARENTHETICAL

        # Vérifier si c'est une subordonnée
        if self._starts_with_subordinate_marker(text_lower):
            return ClauseType.SUBORDINATE

        # Vérifier si contient un mot d'emphase
        if self._contains_emphasis_word(text_lower):
            return ClauseType.EMPHASIS

        # Par défaut: clause principale
        return ClauseType.MAIN

    def _is_parenthetical(self, text: str) -> bool:
        """Vérifie si le texte est une incise."""
        # Entre parenthèses
        text = text.strip()

        if text.startswith('(') and text.endswith(')'):
            return True

        # Entre tirets longs (incise typographique)
        if text.startswith('—') and text.endswith('—'):
            return True
        if text.startswith('–') and text.endswith('–'):
            return True

        return False

    def _starts_with_subordinate_marker(self, text: str) -> bool:
        """Vérifie si le texte commence par un marqueur de subordonnée."""
        words = text.split()
        if not words:
            return False

        first_word = words[0].strip('.,;:!?')

        # Vérifier les marqueurs simples
        if first_word in SUBORDINATE_MARKERS_FR:
            return True

        # Vérifier les marqueurs composés
        text_start = ' '.join(words[:3]).lower()
        for marker in SUBORDINATE_MARKERS_FR:
            if ' ' in marker and text_start.startswith(marker):
                return True

        return False

    def _contains_emphasis_word(self, text: str) -> bool:
        """Vérifie si le texte contient un mot d'emphase."""
        words = set(re.findall(r'\b\w+\b', text.lower()))
        return bool(words & EMPHASIS_WORDS_FR)

    def get_clause_speed_modifier(self, clause_type: ClauseType) -> float:
        """
        Retourne le modificateur de vitesse pour un type de clause.

        Args:
            clause_type: Type de clause

        Returns:
            Modificateur de vitesse (1.0 = normal)
        """
        speed_map = {
            ClauseType.MAIN: self.config.main_clause_speed,
            ClauseType.SUBORDINATE: self.config.subordinate_speed,
            ClauseType.PARENTHETICAL: self.config.parenthetical_speed,
            ClauseType.QUOTE: self.config.quote_speed,
            ClauseType.EMPHASIS: 0.97,  # Légèrement plus lent pour l'emphase
        }
        return speed_map.get(clause_type, 1.0)

    def add_emphasis_pauses(self, text: str) -> str:
        """
        Ajoute des micro-pauses avant les mots d'emphase.

        Args:
            text: Texte original

        Returns:
            Texte avec marqueurs de pause ajoutés
        """
        if not self.config.enable_emphasis_pauses:
            return text

        pause_tag = f"[pause:{self.config.emphasis_pause_duration}]"

        # Tokeniser le texte en mots et espaces
        tokens = re.split(r'(\s+)', text)
        result_tokens = []

        for i, token in enumerate(tokens):
            # Vérifier si c'est un mot d'emphase
            word_lower = token.lower().strip('.,;:!?"\'"«»')

            if word_lower in EMPHASIS_WORDS_FR:
                # Ajouter une pause avant (si pas déjà une pause)
                if result_tokens and not result_tokens[-1].strip().endswith(']'):
                    result_tokens.append(f' {pause_tag}')

            result_tokens.append(token)

        return ''.join(result_tokens)

    def get_inter_phrase_variation(self) -> float:
        """
        Retourne une variation pour éviter la monotonie inter-phrases.

        Alterne légèrement entre phrases plus rapides et plus lentes.

        Returns:
            Modificateur de vitesse
        """
        if not self.config.enable_inter_phrase_variation:
            return 1.0

        self._phrase_counter += 1

        # Variation sinusoïdale douce
        import math
        base_variation = math.sin(self._phrase_counter * 0.7) * self.config.inter_phrase_variation

        # Ajouter un peu d'aléatoire
        random_component = random.uniform(-0.01, 0.01)

        return 1.0 + base_variation + random_component

    def reset_phrase_counter(self):
        """Réinitialise le compteur de phrases."""
        self._phrase_counter = 0


class PauseCalculator:
    """Calcule les durées de pause optimales."""

    def __init__(
        self,
        humanizer: TimingHumanizer = None,
        base_comma_pause: float = 0.3,
        base_period_pause: float = 0.6,
        base_paragraph_pause: float = 1.0
    ):
        """
        Initialise le calculateur de pauses.

        Args:
            humanizer: Humaniseur de timing (optionnel)
            base_comma_pause: Pause après virgule (secondes)
            base_period_pause: Pause après point (secondes)
            base_paragraph_pause: Pause entre paragraphes (secondes)
        """
        self.humanizer = humanizer or TimingHumanizer()
        self.base_pauses = {
            ',': base_comma_pause,
            ';': base_comma_pause * 1.2,
            ':': base_comma_pause * 1.1,
            '.': base_period_pause,
            '!': base_period_pause * 1.1,
            '?': base_period_pause * 1.1,
            '...': base_period_pause * 1.5,
            '…': base_period_pause * 1.5,
            '\n\n': base_paragraph_pause,
        }

    def calculate_pause(self, punctuation: str, context: str = "") -> float:
        """
        Calcule la durée de pause appropriée.

        Args:
            punctuation: Ponctuation déclenchant la pause
            context: Contexte textuel (pour analyse)

        Returns:
            Durée de pause en secondes
        """
        # Obtenir la pause de base
        base = self.base_pauses.get(punctuation, 0.2)

        # Appliquer la variation humaine
        humanized = self.humanizer.humanize_pause(base)

        # Ajuster selon le contexte si fourni
        if context:
            clause_type = self.humanizer.get_clause_type(context)
            speed_mod = self.humanizer.get_clause_speed_modifier(clause_type)
            # Pause plus courte si vitesse plus rapide
            humanized = humanized / speed_mod

        return humanized

    def get_breath_pause(self, text_before: str = "", text_after: str = "") -> float:
        """
        Calcule la pause pour une respiration.

        Args:
            text_before: Texte avant la respiration
            text_after: Texte après la respiration

        Returns:
            Durée de pause en secondes
        """
        # Pause de base pour respiration
        base = 0.1

        # Ajouter variation
        return self.humanizer.humanize_pause(base)


@dataclass
class TimedSegment:
    """Segment de texte avec timing."""

    text: str
    pause_before: float = 0.0
    pause_after: float = 0.0
    speed_modifier: float = 1.0
    has_emphasis_pause: bool = False


class TextTimingProcessor:
    """Processeur complet pour le timing du texte."""

    def __init__(
        self,
        humanizer: TimingHumanizer = None,
        pause_calculator: PauseCalculator = None
    ):
        """
        Initialise le processeur.

        Args:
            humanizer: Humaniseur de timing
            pause_calculator: Calculateur de pauses
        """
        self.humanizer = humanizer or TimingHumanizer()
        self.pause_calculator = pause_calculator or PauseCalculator(self.humanizer)

    def process_text(self, text: str) -> list[TimedSegment]:
        """
        Traite un texte et retourne des segments timés.

        Args:
            text: Texte à traiter

        Returns:
            Liste de segments avec timing
        """
        # Diviser en phrases/clauses
        segments = self._split_into_segments(text)

        timed_segments = []
        for i, seg_text in enumerate(segments):
            if not seg_text.strip():
                continue

            # Détecter le type de clause
            clause_type = self.humanizer.get_clause_type(seg_text)

            # Calculer le modificateur de vitesse
            speed_mod = self.humanizer.get_clause_speed_modifier(clause_type)

            # Ajouter variation inter-phrase
            speed_mod *= self.humanizer.get_inter_phrase_variation()

            # Calculer les pauses
            pause_before = 0.0
            pause_after = 0.0

            # Pause après selon la ponctuation finale
            if seg_text.strip():
                last_char = seg_text.strip()[-1]
                if last_char in self.pause_calculator.base_pauses:
                    pause_after = self.pause_calculator.calculate_pause(
                        last_char, seg_text
                    )

            # Ajouter micro-pauses d'emphase si nécessaire
            processed_text = self.humanizer.add_emphasis_pauses(seg_text)
            has_emphasis = processed_text != seg_text

            timed_segments.append(TimedSegment(
                text=processed_text,
                pause_before=pause_before,
                pause_after=pause_after,
                speed_modifier=speed_mod,
                has_emphasis_pause=has_emphasis
            ))

        return timed_segments

    def _split_into_segments(self, text: str) -> list[str]:
        """
        Divise le texte en segments (phrases/clauses).

        Args:
            text: Texte à diviser

        Returns:
            Liste de segments
        """
        # Diviser sur les ponctuations majeures
        pattern = r'([.!?;]|\n\n)'
        parts = re.split(pattern, text)

        segments = []
        current = ""

        for part in parts:
            if re.match(pattern, part):
                # C'est une ponctuation, l'ajouter au segment courant
                current += part
                if current.strip():
                    segments.append(current)
                current = ""
            else:
                current = part

        # Ajouter le reste
        if current.strip():
            segments.append(current)

        return segments
