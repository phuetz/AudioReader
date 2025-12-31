"""
Attribution de Dialogue Automatique v2.4.

Identifie automatiquement QUI parle dans les dialogues:
- Analyse des verbes de parole: "dit Victor", "repondit Marie"
- Suivi du contexte: alternance entre personnages
- Detection des indices: pronoms, noms propres
- Gestion des dialogues sans attribution explicite

Strategies:
1. Attribution explicite: "dit X", "s'exclama Y"
2. Attribution contextuelle: alternance, pronoms
3. Attribution par inference: derniere personne mentionnee
"""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from collections import defaultdict


class AttributionMethod(Enum):
    """Methode utilisee pour attribuer le dialogue."""
    EXPLICIT = "explicit"        # "dit Victor"
    ALTERNATION = "alternation"  # Alternance entre 2 personnages
    PRONOUN = "pronoun"          # "dit-il", "murmura-t-elle"
    CONTEXT = "context"          # Derniere personne mentionnee
    INFERENCE = "inference"      # Deduction logique
    UNKNOWN = "unknown"          # Non attribuable


@dataclass
class DialogueAttribution:
    """Attribution d'un dialogue a un personnage."""
    speaker: str
    method: AttributionMethod
    confidence: float  # 0.0 - 1.0
    gender: Optional[str] = None  # "M", "F", None


@dataclass
class AttributedDialogue:
    """Dialogue avec attribution."""
    text: str
    attribution: DialogueAttribution
    is_dialogue: bool = True
    original_text: str = ""  # Texte complet avec narration


@dataclass
class ConversationContext:
    """Contexte de conversation pour l'attribution."""
    participants: List[str] = field(default_factory=list)
    last_speaker: Optional[str] = None
    speaker_order: List[str] = field(default_factory=list)
    gender_map: Dict[str, str] = field(default_factory=dict)


class DialogueAttributor:
    """
    Attribue automatiquement les dialogues aux personnages.

    Utilise plusieurs strategies:
    1. Detection explicite via verbes de parole
    2. Analyse des pronoms (il/elle)
    3. Suivi de l'alternance dans les conversations
    4. Inference par contexte narratif
    """

    # Verbes de parole francais
    SPEECH_VERBS_FR = {
        # Verbes neutres
        "dire", "dit", "disait", "avait dit",
        # Questions
        "demander", "demanda", "demandait",
        "questionner", "questionna",
        # Reponses
        "repondre", "repondit", "repondait",
        "retorquer", "retorqua",
        "repliquer", "repliqua",
        # Emotions
        "s'exclamer", "s'exclama",
        "s'ecrier", "s'ecria",
        "crier", "cria", "criait",
        "hurler", "hurla",
        "murmurer", "murmura", "murmurait",
        "chuchoter", "chuchota",
        "soupirer", "soupira",
        "gemir", "gemit",
        "grommeler", "grommela",
        "marmonner", "marmonna",
        "begayer", "begaya",
        # Ajouts
        "ajouter", "ajouta", "ajoutait",
        "continuer", "continua",
        "reprendre", "reprit",
        "poursuivre", "poursuivit",
        "conclure", "conclut",
        # Autres
        "annoncer", "annoncera", "annoncait",
        "declarer", "declara",
        "affirmer", "affirma",
        "expliquer", "expliqua",
        "preciser", "precisa",
        "confirmer", "confirma",
        "admettre", "admit",
        "avouer", "avoua",
        "protester", "protesta",
        "insister", "insista",
        "supplier", "supplia",
        "ordonner", "ordonna",
        "suggerer", "suggera",
        "proposer", "proposa",
        "remarquer", "remarqua",
        "observer", "observa",
        "commenter", "commenta",
        "lancer", "lancera",  # "lancera-t-il"
        "couper", "coupa",  # "coupa-t-elle"
        "interrompre", "interrompit",
    }

    # Patterns pour extraction
    PATTERNS = {
        # "dit Victor" / "repondit Marie"
        "verb_name": r'\b(' + '|'.join(SPEECH_VERBS_FR) + r')\s+([A-ZÀÂÄÉÈÊËÏÎÔÙÛÜÇ][a-zàâäéèêëïîôùûüç]+)',

        # "Victor dit" / "Marie murmura"
        "name_verb": r'([A-ZÀÂÄÉÈÊËÏÎÔÙÛÜÇ][a-zàâäéèêëïîôùûüç]+)\s+(' + '|'.join(SPEECH_VERBS_FR) + r')',

        # "dit-il" / "murmura-t-elle"
        "verb_pronoun": r'\b(' + '|'.join(SPEECH_VERBS_FR) + r')-t?-(il|elle|on)',

        # "il dit" / "elle repondit"
        "pronoun_verb": r'\b(il|elle)\s+(' + '|'.join(SPEECH_VERBS_FR) + r')',
    }

    # Noms communs francais (pour filtrer les faux positifs)
    COMMON_WORDS_FR = {
        "le", "la", "les", "un", "une", "des", "ce", "cette", "ces",
        "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses",
        "notre", "nos", "votre", "vos", "leur", "leurs",
        "qui", "que", "quoi", "dont", "ou", "mais", "donc", "car",
        "puis", "alors", "ainsi", "bien", "mal", "tout", "tous",
    }

    def __init__(self, lang: str = "fr"):
        """
        Initialise l'attributeur.

        Args:
            lang: Code langue
        """
        self.lang = lang
        self.context = ConversationContext()
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile les patterns regex."""
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE | re.UNICODE)
            for name, pattern in self.PATTERNS.items()
        }

    def reset_context(self):
        """Reinitialise le contexte de conversation."""
        self.context = ConversationContext()

    def _extract_explicit_speaker(self, text: str) -> Optional[Tuple[str, str, float]]:
        """
        Extrait le locuteur explicitement mentionne.

        Returns:
            Tuple (speaker, gender, confidence) ou None
        """
        # Pattern "verbe Nom"
        match = self.compiled_patterns["verb_name"].search(text)
        if match:
            name = match.group(2)
            if name.lower() not in self.COMMON_WORDS_FR:
                return (name, self._guess_gender(name), 0.95)

        # Pattern "Nom verbe"
        match = self.compiled_patterns["name_verb"].search(text)
        if match:
            name = match.group(1)
            if name.lower() not in self.COMMON_WORDS_FR:
                return (name, self._guess_gender(name), 0.9)

        return None

    def _extract_pronoun_speaker(self, text: str) -> Optional[Tuple[str, str, float]]:
        """
        Extrait le genre du locuteur via pronom.

        Returns:
            Tuple (placeholder, gender, confidence) ou None
        """
        # Pattern "verbe-t-il/elle"
        match = self.compiled_patterns["verb_pronoun"].search(text)
        if match:
            pronoun = match.group(2).lower()
            gender = "M" if pronoun == "il" else "F"
            return (f"[{gender}]", gender, 0.8)

        # Pattern "il/elle verbe"
        match = self.compiled_patterns["pronoun_verb"].search(text)
        if match:
            pronoun = match.group(1).lower()
            gender = "M" if pronoun == "il" else "F"
            return (f"[{gender}]", gender, 0.75)

        return None

    def _guess_gender(self, name: str) -> Optional[str]:
        """Devine le genre d'un prenom (heuristique simple)."""
        # Prenoms feminins courants terminant par 'e', 'a', 'ie'
        female_endings = ["e", "a", "ie", "ine", "elle", "ette", "ise", "aine"]
        male_endings = ["o", "us", "is", "el", "en", "on", "er"]

        # Prenoms explicites
        female_names = {
            "marie", "sophie", "claire", "julie", "anne", "sarah",
            "emma", "lea", "chloe", "manon", "camille", "lucie"
        }
        male_names = {
            "victor", "pierre", "jean", "paul", "marc", "louis",
            "thomas", "nicolas", "antoine", "julien", "lucas", "hugo",
            "kamel", "crawford", "momo", "paulo"
        }

        name_lower = name.lower()

        if name_lower in female_names:
            return "F"
        if name_lower in male_names:
            return "M"

        for ending in female_endings:
            if name_lower.endswith(ending):
                return "F"

        for ending in male_endings:
            if name_lower.endswith(ending):
                return "M"

        return None

    def _find_speaker_by_gender(self, gender: str) -> Optional[str]:
        """Trouve un personnage par genre dans le contexte."""
        for name, g in self.context.gender_map.items():
            if g == gender:
                return name
        return None

    def _get_alternating_speaker(self) -> Optional[str]:
        """Retourne le locuteur suivant dans l'alternance."""
        if len(self.context.participants) < 2:
            return None

        if self.context.last_speaker is None:
            return self.context.participants[0]

        # Trouver le prochain dans l'alternance
        try:
            idx = self.context.participants.index(self.context.last_speaker)
            next_idx = (idx + 1) % len(self.context.participants)
            return self.context.participants[next_idx]
        except ValueError:
            return self.context.participants[0]

    def attribute_dialogue(
        self,
        dialogue_text: str,
        surrounding_text: str = ""
    ) -> DialogueAttribution:
        """
        Attribue un dialogue a un personnage.

        Args:
            dialogue_text: Texte du dialogue
            surrounding_text: Contexte narratif autour du dialogue

        Returns:
            DialogueAttribution avec le locuteur identifie
        """
        # 1. Chercher attribution explicite
        combined_text = f"{surrounding_text} {dialogue_text}"

        explicit = self._extract_explicit_speaker(combined_text)
        if explicit:
            speaker, gender, confidence = explicit
            self._update_context(speaker, gender)
            return DialogueAttribution(
                speaker=speaker,
                method=AttributionMethod.EXPLICIT,
                confidence=confidence,
                gender=gender
            )

        # 2. Chercher attribution par pronom
        pronoun_result = self._extract_pronoun_speaker(combined_text)
        if pronoun_result:
            placeholder, gender, confidence = pronoun_result

            # Chercher un personnage correspondant au genre
            speaker = self._find_speaker_by_gender(gender)
            if speaker:
                self._update_context(speaker, gender)
                return DialogueAttribution(
                    speaker=speaker,
                    method=AttributionMethod.PRONOUN,
                    confidence=confidence * 0.9,
                    gender=gender
                )

            # Sinon, utiliser le placeholder
            return DialogueAttribution(
                speaker=placeholder,
                method=AttributionMethod.PRONOUN,
                confidence=confidence * 0.7,
                gender=gender
            )

        # 3. Alternance si conversation etablie
        if len(self.context.participants) >= 2:
            alternating = self._get_alternating_speaker()
            if alternating:
                gender = self.context.gender_map.get(alternating)
                self._update_context(alternating, gender)
                return DialogueAttribution(
                    speaker=alternating,
                    method=AttributionMethod.ALTERNATION,
                    confidence=0.6,
                    gender=gender
                )

        # 4. Contexte: dernier personnage mentionne
        if self.context.last_speaker:
            return DialogueAttribution(
                speaker=self.context.last_speaker,
                method=AttributionMethod.CONTEXT,
                confidence=0.4,
                gender=self.context.gender_map.get(self.context.last_speaker)
            )

        # 5. Inconnu
        return DialogueAttribution(
            speaker="UNKNOWN",
            method=AttributionMethod.UNKNOWN,
            confidence=0.0,
            gender=None
        )

    def _update_context(self, speaker: str, gender: Optional[str]):
        """Met a jour le contexte de conversation."""
        if speaker not in self.context.participants:
            self.context.participants.append(speaker)

        self.context.last_speaker = speaker
        self.context.speaker_order.append(speaker)

        if gender:
            self.context.gender_map[speaker] = gender

    def process_text(
        self,
        text: str
    ) -> List[AttributedDialogue]:
        """
        Traite un texte complet et attribue tous les dialogues.

        Args:
            text: Texte avec dialogues

        Returns:
            Liste de dialogues attribues
        """
        results = []

        # Pattern pour detecter les dialogues
        # Guillemets francais, anglais, tirets
        dialogue_pattern = r'(«\s*[^»]+\s*»|"[^"]+"|"[^"]+")'

        # Trouver tous les dialogues et leur contexte
        last_end = 0
        for match in re.finditer(dialogue_pattern, text):
            start, end = match.span()
            dialogue_raw = match.group(1)

            # Extraire le texte du dialogue (sans guillemets)
            dialogue_text = re.sub(r'^[«»"""\s]+|[«»"""\s]+$', '', dialogue_raw).strip()

            # Contexte: texte avant et apres le dialogue
            context_before = text[max(0, start - 100):start]
            context_after = text[end:min(len(text), end + 100)]
            surrounding = f"{context_before} {context_after}"

            # Attribuer
            attribution = self.attribute_dialogue(dialogue_text, surrounding)

            results.append(AttributedDialogue(
                text=dialogue_text,
                attribution=attribution,
                is_dialogue=True,
                original_text=dialogue_raw
            ))

            last_end = end

        return results

    def register_character(
        self,
        name: str,
        gender: Optional[str] = None
    ):
        """
        Enregistre un personnage dans le contexte.

        Args:
            name: Nom du personnage
            gender: Genre (M/F)
        """
        if name not in self.context.participants:
            self.context.participants.append(name)

        if gender:
            self.context.gender_map[name] = gender

    def get_conversation_stats(self) -> Dict:
        """Retourne les statistiques de conversation."""
        speaker_counts = defaultdict(int)
        for speaker in self.context.speaker_order:
            speaker_counts[speaker] += 1

        return {
            "participants": list(self.context.participants),
            "total_dialogues": len(self.context.speaker_order),
            "speaker_counts": dict(speaker_counts),
            "gender_map": self.context.gender_map.copy()
        }


def attribute_dialogues_in_text(
    text: str,
    known_characters: Optional[Dict[str, str]] = None
) -> List[AttributedDialogue]:
    """
    Fonction utilitaire pour attribuer les dialogues.

    Args:
        text: Texte a analyser
        known_characters: Dict {nom: genre} des personnages connus

    Returns:
        Liste de dialogues attribues
    """
    attributor = DialogueAttributor()

    # Enregistrer les personnages connus
    if known_characters:
        for name, gender in known_characters.items():
            attributor.register_character(name, gender)

    return attributor.process_text(text)


if __name__ == "__main__":
    print("=== Test Attribution de Dialogue ===\n")

    test_text = """
    Victor entra dans la piece. Il regarda Kamel avec mefiance.

    « Tu es en retard, » dit Kamel avec un sourire froid.

    « J'avais des affaires a regler, » repondit Victor en haussant les epaules.

    Marie apparut dans l'encadrement de la porte.

    « Messieurs, le diner est servi, » annoncera-t-elle.

    « On arrive, » lancera-t-il.

    « Je n'ai pas faim, » murmura Victor.

    « Tant pis pour toi, » repondit Kamel.
    """

    # Personnages connus
    known = {
        "Victor": "M",
        "Kamel": "M",
        "Marie": "F"
    }

    attributor = DialogueAttributor()
    for name, gender in known.items():
        attributor.register_character(name, gender)

    results = attributor.process_text(test_text)

    print("Dialogues attribues:\n")
    for i, dialogue in enumerate(results, 1):
        attr = dialogue.attribution
        print(f"{i}. [{attr.speaker:10}] ({attr.method.value:11}, conf={attr.confidence:.2f})")
        print(f"   \"{dialogue.text[:50]}...\"")
        print()

    print("\nStatistiques:")
    stats = attributor.get_conversation_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
