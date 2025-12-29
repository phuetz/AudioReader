"""
Detection du contexte narratif pour prosodie adaptee.

Fonctionnalites:
- Detection du type de narration (action, description, introspection)
- Gestion des pensees interieures
- Detection des flashbacks/souvenirs
- Support des onomatopees
- Variations de rythme selon le contexte
"""
import re
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class NarrativeType(Enum):
    """Types de contexte narratif."""
    DIALOGUE = "dialogue"           # Paroles prononcees
    NARRATION = "narration"         # Narration standard
    ACTION = "action"               # Scenes d'action rapides
    DESCRIPTION = "description"     # Descriptions detaillees
    INTROSPECTION = "introspection" # Pensees interieures
    FLASHBACK = "flashback"         # Souvenirs, retours en arriere
    LETTER = "letter"               # Lettres, messages ecrits
    QUOTE = "quote"                 # Citations
    ONOMATOPOEIA = "onomatopoeia"   # Onomatopees


@dataclass
class NarrativeContext:
    """Contexte narratif d'un segment."""
    type: NarrativeType
    confidence: float  # 0-1
    suggested_speed: float  # Multiplicateur de vitesse
    suggested_pause_before: float
    suggested_pause_after: float
    is_internal_thought: bool = False
    is_emphasized: bool = False


class NarrativeContextDetector:
    """
    Detecte le contexte narratif pour adapter la prosodie.

    Strategie:
    - Analyse lexicale et syntaxique
    - Detection de patterns specifiques
    - Ajustement du rythme selon le contexte
    """

    # Verbes d'action (scenes rapides)
    ACTION_VERBS_FR = {
        "courir", "sauter", "frapper", "lancer", "attraper", "bondir",
        "esquiver", "fuir", "poursuivre", "se jeter", "s'elancer",
        "degainer", "tirer", "exploser", "s'ecraser", "percuter",
        "foncer", "sprinter", "plonger", "rouler", "se relever"
    }

    ACTION_VERBS_EN = {
        "run", "jump", "hit", "throw", "catch", "leap", "dodge",
        "flee", "chase", "dive", "sprint", "crash", "explode",
        "strike", "punch", "kick", "shoot", "grab", "rush"
    }

    # Verbes de description (scenes lentes)
    DESCRIPTION_VERBS_FR = {
        "etait", "semblait", "paraissait", "ressemblait", "avait l'air",
        "se trouvait", "s'etendait", "dominait", "surplombait",
        "ornait", "decorait", "baignait", "flottait"
    }

    DESCRIPTION_VERBS_EN = {
        "was", "seemed", "appeared", "looked", "stood",
        "lay", "hung", "stretched", "loomed", "dominated",
        "decorated", "adorned", "bathed", "floated"
    }

    # Verbes d'introspection
    INTROSPECTION_VERBS_FR = {
        "pensait", "songeait", "reflechissait", "se demandait",
        "se souvenait", "revoyait", "imaginait", "revait",
        "se rappelait", "comprenait", "realisait", "savait"
    }

    INTROSPECTION_VERBS_EN = {
        "thought", "wondered", "pondered", "reflected",
        "remembered", "recalled", "imagined", "dreamed",
        "realized", "understood", "knew", "felt"
    }

    # Marqueurs de pensees interieures
    THOUGHT_MARKERS_FR = [
        r"pensa-t-(?:il|elle)",
        r"se dit-(?:il|elle)",
        r"songea-t-(?:il|elle)",
        r"se demanda-t-(?:il|elle)",
        r"il se dit",
        r"elle se dit",
    ]

    THOUGHT_MARKERS_EN = [
        r"(?:he|she) thought",
        r"(?:he|she) wondered",
        r"(?:he|she) realized",
        r"(?:he|she) knew",
    ]

    # Marqueurs de flashback
    FLASHBACK_MARKERS_FR = [
        r"il y a (?:longtemps|des annees|des mois)",
        r"autrefois",
        r"jadis",
        r"a cette epoque",
        r"en ce temps-la",
        r"il se souvenait",
        r"le souvenir",
        r"dans sa memoire",
    ]

    FLASHBACK_MARKERS_EN = [
        r"years ago",
        r"long ago",
        r"back then",
        r"once upon a time",
        r"(?:he|she) remembered",
        r"the memory",
        r"in (?:his|her) mind",
    ]

    # Onomatopees courantes
    ONOMATOPOEIA_FR = {
        # Sons
        "bang", "boum", "paf", "vlan", "clac", "crac", "plouf",
        "splash", "pschitt", "vroum", "bip", "ding", "dong",
        "tic-tac", "toc-toc", "pan", "pif", "pouf", "badaboum",
        # Voix animales
        "miaou", "ouaf", "cocorico", "meuh", "beeeh", "coin-coin",
        "cui-cui", "croâ", "hou-hou", "grr", "sss",
        # Expressions
        "ouf", "pfff", "bof", "hum", "euh", "ah", "oh", "aie",
        "ouille", "oups", "chut", "psst", "hep",
        # Rires
        "ha ha", "hi hi", "ho ho", "hé hé",
    }

    ONOMATOPOEIA_EN = {
        "bang", "boom", "crash", "splash", "whoosh", "zoom",
        "beep", "ding", "dong", "tick-tock", "knock-knock",
        "pow", "wham", "zap", "buzz", "hiss", "roar",
        "meow", "woof", "moo", "baa", "quack", "chirp",
        "oops", "shh", "psst", "hey", "wow", "ouch",
        "ha ha", "hee hee", "ho ho",
    }

    def __init__(self, lang: str = "fr"):
        self.lang = lang

        if lang == "fr":
            self.action_verbs = self.ACTION_VERBS_FR
            self.description_verbs = self.DESCRIPTION_VERBS_FR
            self.introspection_verbs = self.INTROSPECTION_VERBS_FR
            self.thought_markers = self.THOUGHT_MARKERS_FR
            self.flashback_markers = self.FLASHBACK_MARKERS_FR
            self.onomatopoeia = self.ONOMATOPOEIA_FR
        else:
            self.action_verbs = self.ACTION_VERBS_EN
            self.description_verbs = self.DESCRIPTION_VERBS_EN
            self.introspection_verbs = self.INTROSPECTION_VERBS_EN
            self.thought_markers = self.THOUGHT_MARKERS_EN
            self.flashback_markers = self.FLASHBACK_MARKERS_EN
            self.onomatopoeia = self.ONOMATOPOEIA_EN

    def _is_dialogue(self, text: str) -> bool:
        """Detecte si le texte est un dialogue."""
        return bool(re.search(r'[«»"""]|^[\s]*[-–—]', text))

    def _is_onomatopoeia(self, text: str) -> tuple[bool, float]:
        """Detecte si le texte contient des onomatopees."""
        text_lower = text.lower()
        found = 0
        for ono in self.onomatopoeia:
            if ono in text_lower:
                found += 1

        if found > 0:
            # Si plus de la moitie du texte est des onomatopees
            words = text.split()
            ratio = found / max(len(words), 1)
            return ratio > 0.3, ratio

        return False, 0.0

    def _detect_action(self, text: str) -> float:
        """Score de detection d'action (0-1)."""
        score = 0
        text_lower = text.lower()

        # Verbes d'action
        for verb in self.action_verbs:
            if verb in text_lower:
                score += 1

        # Phrases courtes = action
        sentences = re.split(r'[.!?]', text)
        short_sentences = sum(1 for s in sentences if len(s.split()) < 8)
        if short_sentences > len(sentences) / 2:
            score += 1

        # Ponctuation exclamative
        if text.count('!') > 1:
            score += 0.5

        # Points de suspension (suspense)
        if '...' in text:
            score += 0.3

        return min(score / 3, 1.0)

    def _detect_description(self, text: str) -> float:
        """Score de detection de description (0-1)."""
        score = 0
        text_lower = text.lower()

        # Verbes de description
        for verb in self.description_verbs:
            if verb in text_lower:
                score += 1

        # Phrases longues = description
        sentences = re.split(r'[.!?]', text)
        long_sentences = sum(1 for s in sentences if len(s.split()) > 15)
        if long_sentences > len(sentences) / 2:
            score += 1

        # Adjectifs multiples (pattern: adj, adj et adj)
        adj_pattern = r'\b\w+(?:e|eux|euse|ant|ent)(?:s)?\b.*\b\w+(?:e|eux|euse|ant|ent)(?:s)?\b'
        if re.search(adj_pattern, text_lower):
            score += 0.5

        # Absence d'action directe
        if not any(verb in text_lower for verb in self.action_verbs):
            score += 0.3

        return min(score / 3, 1.0)

    def _detect_introspection(self, text: str) -> tuple[float, bool]:
        """Score de detection d'introspection et pensees internes."""
        score = 0
        is_thought = False
        text_lower = text.lower()

        # Verbes d'introspection
        for verb in self.introspection_verbs:
            if verb in text_lower:
                score += 1

        # Marqueurs de pensees
        for marker in self.thought_markers:
            if re.search(marker, text_lower):
                score += 1
                is_thought = True

        # Texte en italique (souvent pensees)
        if re.search(r'\*[^*]+\*|_[^_]+_', text):
            score += 0.5
            is_thought = True

        # Questions rhetoriques
        if re.search(r'\?(?!\s*[»"])', text) and not self._is_dialogue(text):
            score += 0.5

        return min(score / 3, 1.0), is_thought

    def _detect_flashback(self, text: str) -> float:
        """Score de detection de flashback."""
        score = 0
        text_lower = text.lower()

        for marker in self.flashback_markers:
            if re.search(marker, text_lower):
                score += 1

        # Temps du passe compose ou imparfait (francais)
        if self.lang == "fr":
            if re.search(r'\bavait\b.*\b\w+é\b', text_lower):
                score += 0.3

        return min(score / 2, 1.0)

    def _detect_letter_or_quote(self, text: str) -> tuple[bool, NarrativeType]:
        """Detecte les lettres ou citations."""
        # Lettres
        letter_markers = [
            r"cher(?:e)?\s+\w+",
            r"mon(?:sieur|ami|amour)",
            r"cordialement",
            r"sincèrement",
            r"je vous (?:prie|écris)",
        ]
        for marker in letter_markers:
            if re.search(marker, text.lower()):
                return True, NarrativeType.LETTER

        # Citations
        if text.strip().startswith('"') or text.strip().startswith('«'):
            if re.search(r'(?:disait|écrivait|affirmait)', text.lower()):
                return True, NarrativeType.QUOTE

        return False, NarrativeType.NARRATION

    def detect(self, text: str) -> NarrativeContext:
        """
        Detecte le contexte narratif du texte.

        Returns:
            NarrativeContext avec type, confiance et suggestions prosodiques
        """
        # Verifier le dialogue en premier
        if self._is_dialogue(text):
            return NarrativeContext(
                type=NarrativeType.DIALOGUE,
                confidence=0.9,
                suggested_speed=1.0,
                suggested_pause_before=0.2,
                suggested_pause_after=0.3
            )

        # Onomatopees
        is_ono, ono_score = self._is_onomatopoeia(text)
        if is_ono:
            return NarrativeContext(
                type=NarrativeType.ONOMATOPOEIA,
                confidence=ono_score,
                suggested_speed=1.1,  # Leger accent
                suggested_pause_before=0.1,
                suggested_pause_after=0.2,
                is_emphasized=True
            )

        # Lettre ou citation
        is_special, special_type = self._detect_letter_or_quote(text)
        if is_special:
            return NarrativeContext(
                type=special_type,
                confidence=0.8,
                suggested_speed=0.95,  # Plus lent, plus formel
                suggested_pause_before=0.5,
                suggested_pause_after=0.5
            )

        # Calculer les scores
        action_score = self._detect_action(text)
        description_score = self._detect_description(text)
        introspection_score, is_thought = self._detect_introspection(text)
        flashback_score = self._detect_flashback(text)

        # Determiner le type dominant
        scores = {
            NarrativeType.ACTION: action_score,
            NarrativeType.DESCRIPTION: description_score,
            NarrativeType.INTROSPECTION: introspection_score,
            NarrativeType.FLASHBACK: flashback_score,
        }

        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]

        # Si aucun score significatif, c'est de la narration standard
        if max_score < 0.3:
            return NarrativeContext(
                type=NarrativeType.NARRATION,
                confidence=0.7,
                suggested_speed=1.0,
                suggested_pause_before=0.1,
                suggested_pause_after=0.3
            )

        # Ajuster la prosodie selon le type
        if max_type == NarrativeType.ACTION:
            return NarrativeContext(
                type=NarrativeType.ACTION,
                confidence=max_score,
                suggested_speed=1.15,  # Plus rapide
                suggested_pause_before=0.1,
                suggested_pause_after=0.2  # Pauses courtes
            )

        elif max_type == NarrativeType.DESCRIPTION:
            return NarrativeContext(
                type=NarrativeType.DESCRIPTION,
                confidence=max_score,
                suggested_speed=0.9,  # Plus lent
                suggested_pause_before=0.2,
                suggested_pause_after=0.4  # Pauses plus longues
            )

        elif max_type == NarrativeType.INTROSPECTION:
            return NarrativeContext(
                type=NarrativeType.INTROSPECTION,
                confidence=max_score,
                suggested_speed=0.92,  # Legerement plus lent
                suggested_pause_before=0.3,
                suggested_pause_after=0.4,
                is_internal_thought=is_thought
            )

        elif max_type == NarrativeType.FLASHBACK:
            return NarrativeContext(
                type=NarrativeType.FLASHBACK,
                confidence=max_score,
                suggested_speed=0.88,  # Plus lent, reveur
                suggested_pause_before=0.5,  # Pause avant transition
                suggested_pause_after=0.5
            )

        return NarrativeContext(
            type=NarrativeType.NARRATION,
            confidence=0.5,
            suggested_speed=1.0,
            suggested_pause_before=0.1,
            suggested_pause_after=0.3
        )


class OnomatopoeiaProcessor:
    """
    Traite les onomatopees pour une meilleure synthese.

    Strategies:
    - Repetitions expressives
    - Pauses dramatiques
    - Tags TTS speciaux
    """

    # Mapping onomatopees -> effets
    EFFECTS = {
        # Explosions/impacts
        "bang": {"repeat": False, "pause_after": 0.3, "emphasis": True},
        "boum": {"repeat": False, "pause_after": 0.4, "emphasis": True},
        "crash": {"repeat": False, "pause_after": 0.3, "emphasis": True},
        "paf": {"repeat": False, "pause_after": 0.2, "emphasis": True},

        # Sons continus
        "tic-tac": {"repeat": True, "pause_after": 0.1, "emphasis": False},
        "vroum": {"repeat": False, "pause_after": 0.2, "emphasis": False},

        # Silence/attention
        "chut": {"repeat": False, "pause_after": 0.5, "emphasis": False, "whisper": True},
        "psst": {"repeat": False, "pause_after": 0.3, "emphasis": False, "whisper": True},

        # Rires
        "ha ha": {"repeat": True, "pause_after": 0.2, "emphasis": False, "tag": "[laugh]"},
        "hi hi": {"repeat": True, "pause_after": 0.2, "emphasis": False, "tag": "[chuckle]"},

        # Douleur/surprise
        "aie": {"repeat": False, "pause_after": 0.2, "emphasis": True, "tag": "[gasp]"},
        "ouch": {"repeat": False, "pause_after": 0.2, "emphasis": True, "tag": "[gasp]"},
        "ouf": {"repeat": False, "pause_after": 0.3, "emphasis": False, "tag": "[sigh]"},
    }

    def process(self, text: str) -> tuple[str, list[str]]:
        """
        Traite les onomatopees dans le texte.

        Returns:
            Tuple (texte traite, liste de tags TTS)
        """
        tags = []

        for ono, effects in self.EFFECTS.items():
            if ono.lower() in text.lower():
                # Ajouter le tag si present
                if "tag" in effects:
                    tags.append(effects["tag"])

        return text, tags


def detect_narrative_context(text: str, lang: str = "fr") -> NarrativeContext:
    """
    Fonction utilitaire pour detecter le contexte narratif.

    Args:
        text: Texte a analyser
        lang: Code langue

    Returns:
        NarrativeContext avec suggestions prosodiques
    """
    detector = NarrativeContextDetector(lang)
    return detector.detect(text)


if __name__ == "__main__":
    # Tests
    test_texts = [
        # Action
        "Il bondit par-dessus la barriere et sprinta vers la sortie. Bang ! La porte claqua.",

        # Description
        "La vieille maison se dressait au sommet de la colline, ses murs de pierre grise couverts de lierre. Les fenetres etroites semblaient observer le vallon.",

        # Introspection
        "Marie se demandait si elle avait fait le bon choix. Tant d'annees perdues, pensa-t-elle avec amertume.",

        # Flashback
        "Il y a longtemps, dans sa jeunesse, il avait connu ce meme sentiment. Le souvenir revenait, intact.",

        # Dialogue
        "« Ou vas-tu ? » demanda-t-elle.",

        # Onomatopee
        "Tic-tac, tic-tac. L'horloge egreynait les secondes. Soudain, BOUM !",
    ]

    print("=== Test detection contexte narratif ===\n")
    detector = NarrativeContextDetector("fr")

    for text in test_texts:
        context = detector.detect(text)
        print(f"Texte: {text[:60]}...")
        print(f"  Type: {context.type.value}")
        print(f"  Confiance: {context.confidence:.2f}")
        print(f"  Vitesse suggeree: {context.suggested_speed:.2f}")
        print(f"  Pause avant/apres: {context.suggested_pause_before:.2f}s / {context.suggested_pause_after:.2f}s")
        if context.is_internal_thought:
            print(f"  [Pensee interieure]")
        print()
