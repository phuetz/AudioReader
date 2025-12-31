"""
Détection automatique des personnages et attribution de voix.

Fonctionnalités:
- Détection des dialogues (guillemets, tirets)
- Identification des personnages par pattern "dit X", "répondit Y"
- Attribution automatique de voix différentes par personnage
- Support du narrateur vs personnages
"""
import re
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class SpeakerType(Enum):
    """Type de locuteur."""
    NARRATOR = "narrator"
    CHARACTER = "character"
    UNKNOWN = "unknown"


@dataclass
class DialogueSegment:
    """Un segment de dialogue ou narration."""
    text: str
    speaker: str  # Nom du personnage ou "NARRATOR"
    speaker_type: SpeakerType
    voice_id: Optional[str] = None
    emotion: Optional[str] = None
    index: int = 0


@dataclass
class Character:
    """Un personnage détecté dans le texte."""
    name: str
    aliases: list[str] = field(default_factory=list)
    gender: Optional[str] = None  # "M", "F", ou None
    voice_id: Optional[str] = None
    occurrence_count: int = 0

    def matches(self, name: str) -> bool:
        """Vérifie si le nom correspond à ce personnage."""
        name_lower = name.lower().strip()
        if self.name.lower() == name_lower:
            return True
        return any(alias.lower() == name_lower for alias in self.aliases)


class CharacterDetector:
    """
    Détecte les personnages et segmente le texte par locuteur.

    Patterns de détection:
    - Dialogues entre guillemets: "texte" dit Marie
    - Dialogues avec tirets: — Texte, dit Marie.
    - Verbes de parole: dit, répondit, murmura, cria, demanda, etc.
    """

    # Verbes de parole français
    SPEECH_VERBS_FR = [
        "dit", "dis", "disait", "dirent",
        "répondit", "répond", "répondait", "répondirent",
        "demanda", "demande", "demandait", "demandèrent",
        "murmura", "murmure", "murmurait", "murmurèrent",
        "cria", "crie", "criait", "crièrent",
        "chuchota", "chuchote", "chuchotait", "chuchotèrent",
        "s'exclama", "s'exclame", "s'exclamait", "s'exclamèrent",
        "hurla", "hurle", "hurlait", "hurlèrent",
        "soupira", "soupire", "soupirait", "soupirèrent",
        "grogna", "grogne", "grognait", "grognèrent",
        "annonça", "annonce", "annonçait", "annoncèrent",
        "expliqua", "explique", "expliquait", "expliquèrent",
        "ajouta", "ajoute", "ajoutait", "ajoutèrent",
        "continua", "continue", "continuait", "continuèrent",
        "reprit", "reprend", "reprenait", "reprirent",
        "interrompit", "interrompt", "interrompait",
        "lança", "lance", "lançait", "lancèrent",
        "souffla", "souffle", "soufflait",
        "ricana", "ricane", "ricanait",
        "grommela", "grommèle", "grommelait",
        "bégaya", "bégaie", "bégayait",
        "balbutia", "balbutie", "balbutiait",
        "acquiesça", "acquiesce", "acquiesçait",
        "protesta", "proteste", "protestait",
        "objecta", "objecte", "objectait",
        "confirma", "confirme", "confirmait",
        "avoua", "avoue", "avouait",
        "confessa", "confesse", "confessait",
        "révéla", "révèle", "révélait",
        "supplia", "supplie", "suppliait",
        "implora", "implore", "implorait",
        "ordonna", "ordonne", "ordonnait",
        "commanda", "commande", "commandait",
        "interrogea", "interroge", "interrogeait",
        "questionna", "questionne", "questionnait",
        "proposa", "propose", "proposait",
        "suggéra", "suggère", "suggérait",
        "remarqua", "remarque", "remarquait",
        "observa", "observe", "observait",
        "nota", "note", "notait",
        "constata", "constate", "constatait",
        "admit", "admet", "admettait",
        "concéda", "concède", "concédait",
        "rétorqua", "rétorque", "rétorquait",
        "répliqua", "réplique", "répliquait",
        "marmonna", "marmonne", "marmonnait",
        "bredouilla", "bredouille", "bredouillait",
    ]

    # Verbes de parole anglais
    SPEECH_VERBS_EN = [
        "said", "says", "say",
        "replied", "replies", "reply",
        "asked", "asks", "ask",
        "whispered", "whispers", "whisper",
        "shouted", "shouts", "shout",
        "yelled", "yells", "yell",
        "screamed", "screams", "scream",
        "murmured", "murmurs", "murmur",
        "muttered", "mutters", "mutter",
        "exclaimed", "exclaims", "exclaim",
        "answered", "answers", "answer",
        "responded", "responds", "respond",
        "added", "adds", "add",
        "continued", "continues", "continue",
        "explained", "explains", "explain",
        "announced", "announces", "announce",
        "declared", "declares", "declare",
        "stated", "states", "state",
        "suggested", "suggests", "suggest",
        "wondered", "wonders", "wonder",
        "thought", "thinks", "think",
        "sighed", "sighs", "sigh",
        "groaned", "groans", "groan",
        "moaned", "moans", "moan",
        "laughed", "laughs", "laugh",
        "chuckled", "chuckles", "chuckle",
        "giggled", "giggles", "giggle",
        "sobbed", "sobs", "sob",
        "cried", "cries", "cry",
        "pleaded", "pleads", "plead",
        "begged", "begs", "beg",
        "demanded", "demands", "demand",
        "ordered", "orders", "order",
        "commanded", "commands", "command",
        "insisted", "insists", "insist",
        "admitted", "admits", "admit",
        "confessed", "confesses", "confess",
        "revealed", "reveals", "reveal",
        "interrupted", "interrupts", "interrupt",
        "interjected", "interjects", "interject",
        "protested", "protests", "protest",
        "objected", "objects", "object",
        "agreed", "agrees", "agree",
        "disagreed", "disagrees", "disagree",
        "confirmed", "confirms", "confirm",
        "denied", "denies", "deny",
        "stammered", "stammers", "stammer",
        "stuttered", "stutters", "stutter",
        "mumbled", "mumbles", "mumble",
        "growled", "growls", "growl",
        "snapped", "snaps", "snap",
        "hissed", "hisses", "hiss",
        "barked", "barks", "bark",
        "roared", "roars", "roar",
        "bellowed", "bellows", "bellow",
        "called", "calls", "call",
        "inquired", "inquires", "inquire",
        "queried", "queries", "query",
        "questioned", "questions", "question",
        "remarked", "remarks", "remark",
        "observed", "observes", "observe",
        "noted", "notes", "note",
        "commented", "comments", "comment",
        "mentioned", "mentions", "mention",
        "reported", "reports", "report",
    ]

    # Prénoms communs pour détection de genre
    FEMALE_NAMES_FR = {
        "marie", "jeanne", "anne", "sophie", "claire", "julie", "emma",
        "lucie", "alice", "léa", "camille", "charlotte", "isabelle",
        "catherine", "christine", "nathalie", "valérie", "sylvie",
        "marguerite", "hélène", "élise", "louise", "mathilde", "amélie"
    }

    MALE_NAMES_FR = {
        "jean", "pierre", "paul", "jacques", "michel", "philippe", "henri",
        "louis", "françois", "marc", "charles", "nicolas", "antoine",
        "thomas", "julien", "alexandre", "guillaume", "olivier", "éric",
        "bernard", "alain", "laurent", "patrick", "christophe", "maxime"
    }

    FEMALE_NAMES_EN = {
        "mary", "jane", "anne", "sarah", "emma", "alice", "lucy", "emily",
        "sophie", "claire", "elizabeth", "margaret", "catherine", "helen",
        "rose", "grace", "olivia", "charlotte", "isabella", "victoria"
    }

    MALE_NAMES_EN = {
        "john", "james", "william", "michael", "david", "robert", "richard",
        "thomas", "charles", "george", "edward", "henry", "peter", "paul",
        "mark", "daniel", "matthew", "andrew", "joseph", "christopher"
    }

    def __init__(self, lang: str = "fr"):
        """
        Args:
            lang: Code langue ("fr" ou "en")
        """
        self.lang = lang
        self.characters: dict[str, Character] = {}
        self.speech_verbs = (
            self.SPEECH_VERBS_FR if lang == "fr" else self.SPEECH_VERBS_EN
        )
        self.female_names = (
            self.FEMALE_NAMES_FR if lang == "fr" else self.FEMALE_NAMES_EN
        )
        self.male_names = (
            self.MALE_NAMES_FR if lang == "fr" else self.MALE_NAMES_EN
        )

        # Construire le pattern de verbes de parole
        verbs_pattern = "|".join(re.escape(v) for v in self.speech_verbs)
        self.speech_verb_pattern = re.compile(
            rf'\b({verbs_pattern})\b',
            re.IGNORECASE
        )

    def _guess_gender(self, name: str) -> Optional[str]:
        """Devine le genre à partir du prénom."""
        name_lower = name.lower().split()[0]  # Premier mot seulement

        if name_lower in self.female_names:
            return "F"
        elif name_lower in self.male_names:
            return "M"

        # Heuristiques pour le français
        if self.lang == "fr":
            if name_lower.endswith(("ette", "elle", "ine", "ienne", "euse")):
                return "F"
            elif name_lower.endswith(("eur", "ien", "ois")):
                return "M"

        return None

    # Mots à ignorer après le nom du personnage
    STOP_WORDS_FR = {
        "avec", "en", "d'une", "d'un", "sans", "pour", "dans", "sur",
        "mais", "et", "ou", "donc", "car", "ni", "que", "qui",
        "tout", "toute", "tous", "toutes", "très", "plus", "moins",
        "avant", "après", "pendant", "depuis", "vers", "chez"
    }

    def _extract_speaker_name(self, text: str) -> Optional[str]:
        """
        Extrait le nom du locuteur à partir du contexte.

        Patterns recherchés:
        - "texte" dit Marie
        - dit Marie
        - Marie dit:
        - , dit Kamel avec un sourire -> Kamel (pas "Kamel avec")
        """
        # Pattern: verbe + nom propre (un seul mot avec majuscule)
        for verb in self.speech_verbs:
            # Pattern: "verbe Nom" (un seul mot)
            pattern = rf'\b{re.escape(verb)}\s+([A-ZÀÂÄÉÈÊËÏÎÔÙÛÜÇ][a-zàâäéèêëïîôùûüç]+)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Ignorer si c'est un stop word
                if name.lower() not in self.STOP_WORDS_FR:
                    return name

            # Pattern: "Nom verbe" (un seul mot)
            pattern = rf'([A-ZÀÂÄÉÈÊËÏÎÔÙÛÜÇ][a-zàâäéèêëïîôùûüç]+)\s+{re.escape(verb)}\b'
            match = re.search(pattern, text)
            if match:
                name = match.group(1).strip()
                if name.lower() not in self.STOP_WORDS_FR:
                    return name

        return None

    def _register_character(self, name: str) -> Character:
        """Enregistre ou met à jour un personnage."""
        # Chercher si le personnage existe déjà
        name_key = name.lower().strip()

        for char_name, char in self.characters.items():
            if char.matches(name):
                char.occurrence_count += 1
                return char

        # Nouveau personnage
        gender = self._guess_gender(name)
        char = Character(
            name=name.strip(),
            gender=gender,
            occurrence_count=1
        )
        self.characters[name_key] = char
        return char

    def detect_dialogue_segments(self, text: str) -> list[DialogueSegment]:
        """
        Segmente le texte en parties de dialogue et narration.

        Détecte:
        - Dialogues entre guillemets: « », " ", " "
        - Dialogues avec tirets: — ou -
        - Narration entre dialogues
        """
        segments = []

        # Pattern pour dialogues français (guillemets)
        # Capture: guillemet ouvrant, dialogue, guillemet fermant, contexte après
        dialogue_pattern = re.compile(
            r'([«"""])'                      # Guillemet ouvrant
            r'([^»"""]+)'                    # Contenu du dialogue
            r'([»"""])'                      # Guillemet fermant
            r'([^«"""]*?(?=[«"""]|$))',      # Contexte jusqu'au prochain dialogue
            re.DOTALL
        )

        # Pattern pour dialogues avec tirets (style français)
        dash_dialogue_pattern = re.compile(
            r'^[\s]*[—–-]\s*(.+?)(?=\n[\s]*[—–-]|\n\n|$)',
            re.MULTILINE
        )

        last_speaker = "NARRATOR"
        current_pos = 0
        segment_index = 0

        # Collecter d'abord tous les dialogues avec guillemets
        for match in dialogue_pattern.finditer(text):
            start = match.start()

            # Ajouter la narration avant le dialogue
            if start > current_pos:
                narration = text[current_pos:start].strip()
                if narration:
                    segments.append(DialogueSegment(
                        text=narration,
                        speaker="NARRATOR",
                        speaker_type=SpeakerType.NARRATOR,
                        index=segment_index
                    ))
                    segment_index += 1

            dialogue_text = match.group(2).strip()
            context_after = match.group(4).strip() if match.group(4) else ""

            # Chercher le locuteur dans le contexte
            speaker_name = self._extract_speaker_name(context_after)
            if not speaker_name:
                # Essayer dans le dialogue lui-même (rare mais possible)
                speaker_name = self._extract_speaker_name(dialogue_text)

            if speaker_name:
                char = self._register_character(speaker_name)
                speaker = char.name
                speaker_type = SpeakerType.CHARACTER
                last_speaker = speaker
            else:
                # Garder le dernier locuteur connu si pas d'indication
                speaker = last_speaker if last_speaker != "NARRATOR" else "UNKNOWN"
                speaker_type = SpeakerType.UNKNOWN if speaker == "UNKNOWN" else SpeakerType.CHARACTER

            segments.append(DialogueSegment(
                text=dialogue_text,
                speaker=speaker,
                speaker_type=speaker_type,
                index=segment_index
            ))
            segment_index += 1

            # Ajouter le contexte après comme narration (s'il contient autre chose que le verbe)
            if context_after:
                # Retirer la partie "dit X" du contexte
                clean_context = self.speech_verb_pattern.sub("", context_after)
                clean_context = re.sub(r'^\s*[A-ZÀÂÄÉÈÊËÏÎÔÙÛÜÇ][a-zàâäéèêëïîôùûüç]+\.?\s*', '', clean_context)
                clean_context = clean_context.strip(' ,.')

                if clean_context and len(clean_context) > 10:
                    segments.append(DialogueSegment(
                        text=clean_context,
                        speaker="NARRATOR",
                        speaker_type=SpeakerType.NARRATOR,
                        index=segment_index
                    ))
                    segment_index += 1

            current_pos = match.end()

        # Ajouter le reste comme narration
        if current_pos < len(text):
            remaining = text[current_pos:].strip()
            if remaining:
                segments.append(DialogueSegment(
                    text=remaining,
                    speaker="NARRATOR",
                    speaker_type=SpeakerType.NARRATOR,
                    index=segment_index
                ))

        # Si aucun dialogue trouvé, traiter tout comme narration
        if not segments:
            segments.append(DialogueSegment(
                text=text.strip(),
                speaker="NARRATOR",
                speaker_type=SpeakerType.NARRATOR,
                index=0
            ))

        return segments

    def get_characters(self) -> list[Character]:
        """Retourne la liste des personnages détectés, triés par occurrences."""
        return sorted(
            self.characters.values(),
            key=lambda c: c.occurrence_count,
            reverse=True
        )

    def reset(self):
        """Réinitialise la liste des personnages."""
        self.characters.clear()


class VoiceAssigner:
    """
    Assigne automatiquement des voix aux personnages.

    Stratégie:
    - Narrateur: voix principale (configurable)
    - Personnages: voix différente selon genre et ordre d'apparition
    - Rotation des voix pour éviter les répétitions
    """

    # Voix par défaut par genre et langue
    DEFAULT_VOICES = {
        "fr": {
            "narrator": "ff_siwis",
            "female": ["ff_siwis"],
            "male": ["am_adam", "am_michael", "am_eric"],
            "neutral": ["af_bella", "af_nicole"]
        },
        "en-us": {
            "narrator": "af_heart",
            "female": ["af_bella", "af_nicole", "af_nova", "af_sarah"],
            "male": ["am_adam", "am_michael", "am_eric"],
            "neutral": ["af_sky"]
        },
        "en-gb": {
            "narrator": "bf_emma",
            "female": ["bf_emma", "bf_isabella"],
            "male": ["bm_george", "bm_lewis"],
            "neutral": ["bf_emma"]
        }
    }

    def __init__(
        self,
        narrator_voice: str = "ff_siwis",
        lang: str = "fr",
        voice_mapping: Optional[dict[str, str]] = None
    ):
        """
        Args:
            narrator_voice: Voix pour le narrateur
            lang: Code langue
            voice_mapping: Mapping personnalisé personnage -> voix
        """
        self.narrator_voice = narrator_voice
        self.lang = lang
        self.voice_mapping = voice_mapping or {}
        self._voice_assignments: dict[str, str] = {}
        self._used_female_voices: list[str] = []
        self._used_male_voices: list[str] = []

        # Charger les voix par défaut
        lang_key = lang if lang in self.DEFAULT_VOICES else "en-us"
        self.available_voices = self.DEFAULT_VOICES[lang_key]

    def assign_voice(self, character: Character) -> str:
        """Assigne une voix à un personnage."""
        # Vérifier si déjà assigné
        if character.name in self._voice_assignments:
            return self._voice_assignments[character.name]

        # Vérifier le mapping manuel
        if character.name in self.voice_mapping:
            voice = self.voice_mapping[character.name]
            self._voice_assignments[character.name] = voice
            return voice

        # Sélection automatique selon le genre
        if character.gender == "F":
            voices = self.available_voices["female"]
            used = self._used_female_voices
        elif character.gender == "M":
            voices = self.available_voices["male"]
            used = self._used_male_voices
        else:
            voices = self.available_voices["neutral"]
            used = self._used_female_voices  # Partager avec female

        # Choisir une voix non utilisée si possible
        for voice in voices:
            if voice not in used:
                used.append(voice)
                self._voice_assignments[character.name] = voice
                return voice

        # Si toutes utilisées, recycler
        voice = voices[len(used) % len(voices)]
        self._voice_assignments[character.name] = voice
        return voice

    def get_voice_for_segment(self, segment: DialogueSegment) -> str:
        """Retourne la voix à utiliser pour un segment."""
        if segment.speaker_type == SpeakerType.NARRATOR:
            return self.narrator_voice

        if segment.speaker in self._voice_assignments:
            return self._voice_assignments[segment.speaker]

        # Chercher le personnage correspondant
        # (normalement déjà assigné lors de l'analyse)
        return self.narrator_voice  # Fallback

    def assign_voices_to_characters(
        self,
        characters: list[Character]
    ) -> dict[str, str]:
        """Assigne des voix à tous les personnages."""
        for char in characters:
            self.assign_voice(char)
        return self._voice_assignments.copy()

    def get_assignments(self) -> dict[str, str]:
        """Retourne toutes les assignations voix -> personnage."""
        return self._voice_assignments.copy()


def process_text_with_characters(
    text: str,
    narrator_voice: str = "ff_siwis",
    lang: str = "fr",
    voice_mapping: Optional[dict[str, str]] = None
) -> tuple[list[DialogueSegment], dict[str, str]]:
    """
    Traite un texte et assigne des voix aux personnages.

    Args:
        text: Texte à analyser
        narrator_voice: Voix pour le narrateur
        lang: Code langue
        voice_mapping: Mapping personnalisé personnage -> voix

    Returns:
        Tuple (segments avec voix, mapping personnage -> voix)
    """
    # Détecter les personnages et segmenter
    detector = CharacterDetector(lang=lang)
    segments = detector.detect_dialogue_segments(text)
    characters = detector.get_characters()

    # Assigner les voix
    assigner = VoiceAssigner(
        narrator_voice=narrator_voice,
        lang=lang,
        voice_mapping=voice_mapping
    )
    voice_assignments = assigner.assign_voices_to_characters(characters)

    # Mettre à jour les segments avec les voix
    for segment in segments:
        segment.voice_id = assigner.get_voice_for_segment(segment)

    return segments, voice_assignments


if __name__ == "__main__":
    # Test avec un texte exemple
    test_text = """
    Marie entra dans la pièce. Elle regarda autour d'elle.

    « Où est Pierre ? » demanda Marie.

    « Je suis ici », répondit Pierre depuis le fond de la pièce.

    Marie s'approcha de lui. « Tu m'as fait peur ! » s'exclama-t-elle.

    « Désolé », murmura Pierre avec un sourire.

    Ils restèrent silencieux un moment.
    """

    segments, assignments = process_text_with_characters(test_text)

    print("=== Segments détectés ===")
    for seg in segments:
        print(f"[{seg.speaker_type.value:10}] {seg.speaker:15} | {seg.voice_id:12} | {seg.text[:50]}...")

    print("\n=== Assignations de voix ===")
    for char, voice in assignments.items():
        print(f"  {char}: {voice}")
