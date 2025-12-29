"""
Analyse emotionnelle avancee pour TTS expressif.

Fonctionnalites:
- Analyse de sentiment par phrase
- Detection d'intensite dramatique
- Annotation prosodique (vitesse, pitch, volume)
- Tags emotionnels pour TTS
- Micro-pauses naturelles (respiration)
"""
import re
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class Emotion(Enum):
    """Emotions detectables."""
    NEUTRAL = "neutral"
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    EXCITEMENT = "excitement"
    TENDERNESS = "tenderness"
    SUSPENSE = "suspense"
    IRONY = "irony"


class Intensity(Enum):
    """Niveaux d'intensite emotionnelle."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class ProsodyHints:
    """Indications prosodiques pour le TTS."""
    speed: float = 1.0      # 0.7-1.3 (lent-rapide)
    pitch: float = 0.0      # -2 a +2 (grave-aigu)
    volume: float = 1.0     # 0.5-1.5 (bas-fort)
    pause_before: float = 0.0
    pause_after: float = 0.0
    breath_before: bool = False  # Micro-pause respiration


@dataclass
class EmotionAnalysis:
    """Resultat d'analyse emotionnelle."""
    text: str
    emotion: Emotion
    intensity: Intensity
    prosody: ProsodyHints
    tags: list[str]  # Tags TTS a inserer
    emphasis_words: list[str]  # Mots a accentuer


class EmotionAnalyzer:
    """
    Analyseur emotionnel avance pour texte narratif.

    Utilise des heuristiques linguistiques pour detecter:
    - Emotions via lexique et patterns
    - Intensite via ponctuation et repetitions
    - Context narratif (dialogue vs narration)
    """

    # Lexique emotionnel francais
    EMOTION_LEXICON_FR = {
        Emotion.JOY: {
            "keywords": [
                "heureux", "heureuse", "joie", "bonheur", "content", "contente",
                "ravi", "ravie", "enchante", "enchantee", "merveilleux", "magnifique",
                "formidable", "genial", "super", "fantastique", "excellent",
                "extraordinaire", "parfait", "sublime", "delicieux", "sourire",
                "rire", "eclat", "rayonnant", "epanoui", "comble", "jubiler"
            ],
            "patterns": [r"!\s*$", r":-?\)", r"sourire|rire|rigoler"]
        },
        Emotion.SADNESS: {
            "keywords": [
                "triste", "malheureux", "malheureuse", "chagrin", "peine",
                "desespoir", "melancolie", "nostalgie", "larme", "pleurer",
                "sanglot", "deuil", "affliction", "accable", "abattu",
                "deprime", "morose", "sombre", "desole", "navrant",
                "lamentable", "pitoyable", "regret", "remords"
            ],
            "patterns": [r"helas", r"malheureusement", r":-?\("]
        },
        Emotion.ANGER: {
            "keywords": [
                "colere", "furieux", "furieuse", "rage", "enerve", "enervee",
                "agace", "agacee", "irrite", "irritee", "exaspere", "excede",
                "outrage", "indigne", "revolte", "furibond", "enrage",
                "courroux", "emporte", "violent", "brutal", "detester", "hair"
            ],
            "patterns": [r"!{2,}", r"comment oses?-tu", r"assez\s*!"]
        },
        Emotion.FEAR: {
            "keywords": [
                "peur", "terreur", "effroi", "angoisse", "anxiete", "crainte",
                "frayeur", "epouvante", "terrifie", "terrifiee", "apeure",
                "trembler", "frissonner", "palir", "glacer", "paralyse",
                "petrifie", "horrifie", "panique", "affolement"
            ],
            "patterns": [r"mon dieu", r"au secours", r"a l'aide"]
        },
        Emotion.SURPRISE: {
            "keywords": [
                "surpris", "surprise", "etonne", "etonnee", "stupefait",
                "sidere", "abasourdi", "ebahi", "interdit", "decontenance",
                "incroyable", "inimaginable", "inattendu", "soudain"
            ],
            "patterns": [r"\?!", r"!\?", r"quoi\s*[?!]", r"comment\s*[?!]"]
        },
        Emotion.DISGUST: {
            "keywords": [
                "degout", "degoute", "repugnant", "ecoeurer", "repulsion",
                "abject", "ignoble", "immonde", "infect", "nauseabond",
                "repugnance", "aversion", "horreur"
            ],
            "patterns": [r"beurk", r"pouah", r"quelle horreur"]
        },
        Emotion.EXCITEMENT: {
            "keywords": [
                "excite", "excitee", "enthousiaste", "impatient", "impatiente",
                "febrile", "ardent", "passionne", "enflamme", "exalte",
                "transporte", "galvanise", "electrise"
            ],
            "patterns": [r"!{2,}", r"vite", r"enfin\s*!"]
        },
        Emotion.TENDERNESS: {
            "keywords": [
                "tendre", "tendresse", "doux", "douce", "affection", "amour",
                "caresse", "cajoler", "cherir", "adorer", "aime", "aimee",
                "precieux", "cher", "chere", "bien-aime", "attachement"
            ],
            "patterns": [r"mon (cher|amour|coeur)", r"ma (chere|cherie)"]
        },
        Emotion.SUSPENSE: {
            "keywords": [
                "soudain", "tout a coup", "brusquement", "subitement",
                "silence", "immobile", "fige", "retenir son souffle",
                "attendre", "guetter", "epier", "mysterieux", "etrange"
            ],
            "patterns": [r"\.\.\.$", r"—$", r"puis,?\s*$"]
        },
        Emotion.IRONY: {
            "keywords": [
                "ironique", "sarcastique", "moqueur", "railleur", "cynique",
                "persiflage", "derision"
            ],
            "patterns": [r"bien sur", r"evidemment", r"naturellement"]
        }
    }

    # Lexique emotionnel anglais
    EMOTION_LEXICON_EN = {
        Emotion.JOY: {
            "keywords": [
                "happy", "joyful", "delighted", "pleased", "glad", "cheerful",
                "elated", "ecstatic", "thrilled", "wonderful", "amazing",
                "fantastic", "excellent", "perfect", "brilliant", "smile",
                "laugh", "grin", "beam", "radiant"
            ],
            "patterns": [r"!\s*$", r":-?\)", r"smil|laugh|grin"]
        },
        Emotion.SADNESS: {
            "keywords": [
                "sad", "unhappy", "miserable", "depressed", "melancholy",
                "grief", "sorrow", "tears", "cry", "weep", "sob",
                "heartbroken", "devastated", "gloomy", "mournful"
            ],
            "patterns": [r"alas", r"unfortunately", r":-?\("]
        },
        Emotion.ANGER: {
            "keywords": [
                "angry", "furious", "enraged", "mad", "outraged", "irritated",
                "annoyed", "exasperated", "infuriated", "livid", "seething",
                "hate", "detest", "loathe"
            ],
            "patterns": [r"!{2,}", r"how dare", r"enough\s*!"]
        },
        Emotion.FEAR: {
            "keywords": [
                "afraid", "scared", "terrified", "frightened", "fearful",
                "anxious", "nervous", "worried", "panicked", "horrified",
                "trembl", "shiver", "pale", "frozen"
            ],
            "patterns": [r"oh god", r"help", r"oh no"]
        },
        Emotion.SURPRISE: {
            "keywords": [
                "surprised", "astonished", "amazed", "shocked", "stunned",
                "startled", "bewildered", "dumbfounded", "incredible",
                "unbelievable", "unexpected", "sudden"
            ],
            "patterns": [r"\?!", r"!\?", r"what\s*[?!]", r"how\s*[?!]"]
        },
        Emotion.EXCITEMENT: {
            "keywords": [
                "excited", "thrilled", "eager", "enthusiastic", "passionate",
                "energetic", "animated", "exhilarated"
            ],
            "patterns": [r"!{2,}", r"finally\s*!", r"yes\s*!"]
        },
        Emotion.TENDERNESS: {
            "keywords": [
                "tender", "gentle", "soft", "loving", "affectionate",
                "caring", "sweet", "dear", "darling", "beloved", "cherish"
            ],
            "patterns": [r"my (dear|love|darling)", r"sweetheart"]
        },
        Emotion.SUSPENSE: {
            "keywords": [
                "suddenly", "abruptly", "silence", "still", "frozen",
                "waiting", "watching", "mysterious", "strange", "eerie"
            ],
            "patterns": [r"\.\.\.$", r"—$", r"then,?\s*$"]
        }
    }

    # Modificateurs d'intensite
    INTENSITY_BOOSTERS_FR = [
        "tres", "vraiment", "absolument", "completement", "totalement",
        "extremement", "incroyablement", "terriblement", "affreusement",
        "horriblement", "tellement", "si", "tant", "trop"
    ]

    INTENSITY_BOOSTERS_EN = [
        "very", "really", "absolutely", "completely", "totally",
        "extremely", "incredibly", "terribly", "awfully", "so", "such"
    ]

    # Tags TTS supportes
    TTS_TAGS = {
        "laugh": "[laugh]",
        "sigh": "[sigh]",
        "gasp": "[gasp]",
        "cry": "[cry]",
        "whisper": "[whisper]",
        "shout": "[shout]",
    }

    def __init__(self, lang: str = "fr"):
        """
        Args:
            lang: Code langue ("fr" ou "en")
        """
        self.lang = lang
        self.lexicon = (
            self.EMOTION_LEXICON_FR if lang == "fr"
            else self.EMOTION_LEXICON_EN
        )
        self.intensity_boosters = (
            self.INTENSITY_BOOSTERS_FR if lang == "fr"
            else self.INTENSITY_BOOSTERS_EN
        )

    def _normalize_text(self, text: str) -> str:
        """Normalise le texte pour l'analyse."""
        # Retirer les accents pour la comparaison (francais)
        import unicodedata
        normalized = unicodedata.normalize('NFD', text.lower())
        return ''.join(c for c in normalized if not unicodedata.combining(c))

    def _detect_emotion(self, text: str) -> tuple[Emotion, float]:
        """
        Detecte l'emotion dominante dans le texte.

        Returns:
            Tuple (emotion, score de confiance 0-1)
        """
        normalized = self._normalize_text(text)
        scores: dict[Emotion, float] = {e: 0.0 for e in Emotion}

        for emotion, data in self.lexicon.items():
            # Score des mots-cles
            for keyword in data["keywords"]:
                keyword_norm = self._normalize_text(keyword)
                if keyword_norm in normalized:
                    scores[emotion] += 1.0
                    # Bonus si le mot est au debut ou fin
                    if normalized.startswith(keyword_norm) or normalized.endswith(keyword_norm):
                        scores[emotion] += 0.3

            # Score des patterns
            for pattern in data["patterns"]:
                if re.search(pattern, text, re.IGNORECASE):
                    scores[emotion] += 0.5

        # Trouver l'emotion dominante
        max_score = max(scores.values())
        if max_score == 0:
            return Emotion.NEUTRAL, 0.0

        dominant = max(scores.items(), key=lambda x: x[1])
        confidence = min(dominant[1] / 3.0, 1.0)  # Normaliser

        return dominant[0], confidence

    def _detect_intensity(self, text: str) -> Intensity:
        """Detecte l'intensite emotionnelle."""
        score = 0

        # Ponctuation multiple
        if re.search(r'[!?]{2,}', text):
            score += 2
        elif re.search(r'[!?]', text):
            score += 1

        # Points de suspension (suspense/hesitation)
        if '...' in text:
            score += 0.5

        # Majuscules (cri)
        words = text.split()
        caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
        if caps_words > 0:
            score += caps_words

        # Modificateurs d'intensite
        normalized = self._normalize_text(text)
        for booster in self.intensity_boosters:
            if booster in normalized:
                score += 1

        # Repetitions
        if re.search(r'\b(\w+)\b.*\b\1\b', text, re.IGNORECASE):
            score += 0.5

        # Determiner le niveau
        if score >= 4:
            return Intensity.EXTREME
        elif score >= 2:
            return Intensity.HIGH
        elif score >= 1:
            return Intensity.MEDIUM
        return Intensity.LOW

    def _compute_prosody(
        self,
        emotion: Emotion,
        intensity: Intensity
    ) -> ProsodyHints:
        """Calcule les indications prosodiques."""
        prosody = ProsodyHints()

        # Intensite -> multiplicateur
        intensity_mult = {
            Intensity.LOW: 0.5,
            Intensity.MEDIUM: 1.0,
            Intensity.HIGH: 1.5,
            Intensity.EXTREME: 2.0
        }[intensity]

        # Ajustements par emotion
        if emotion == Emotion.JOY:
            prosody.speed = 1.0 + (0.1 * intensity_mult)
            prosody.pitch = 0.5 * intensity_mult
            prosody.volume = 1.0 + (0.1 * intensity_mult)

        elif emotion == Emotion.SADNESS:
            prosody.speed = 1.0 - (0.1 * intensity_mult)
            prosody.pitch = -0.3 * intensity_mult
            prosody.volume = 1.0 - (0.1 * intensity_mult)
            prosody.pause_after = 0.2 * intensity_mult

        elif emotion == Emotion.ANGER:
            prosody.speed = 1.0 + (0.15 * intensity_mult)
            prosody.pitch = 0.3 * intensity_mult
            prosody.volume = 1.0 + (0.2 * intensity_mult)

        elif emotion == Emotion.FEAR:
            prosody.speed = 1.0 + (0.2 * intensity_mult)
            prosody.pitch = 0.4 * intensity_mult
            prosody.breath_before = intensity in [Intensity.HIGH, Intensity.EXTREME]

        elif emotion == Emotion.SURPRISE:
            prosody.speed = 1.1
            prosody.pitch = 0.5 * intensity_mult
            prosody.pause_before = 0.1

        elif emotion == Emotion.EXCITEMENT:
            prosody.speed = 1.15
            prosody.pitch = 0.4 * intensity_mult
            prosody.volume = 1.1

        elif emotion == Emotion.TENDERNESS:
            prosody.speed = 0.9
            prosody.pitch = -0.1
            prosody.volume = 0.9

        elif emotion == Emotion.SUSPENSE:
            prosody.speed = 0.85
            prosody.pause_before = 0.3
            prosody.pause_after = 0.5

        elif emotion == Emotion.IRONY:
            prosody.speed = 0.95
            # L'ironie est difficile a rendre avec prosodie seule

        # Limiter les valeurs
        prosody.speed = max(0.7, min(1.3, prosody.speed))
        prosody.pitch = max(-2.0, min(2.0, prosody.pitch))
        prosody.volume = max(0.5, min(1.5, prosody.volume))

        return prosody

    def _detect_tts_tags(self, text: str) -> list[str]:
        """Detecte les tags TTS a inserer."""
        tags = []

        # Rires
        if re.search(r'\b(ha\s*ha|hi\s*hi|rire|laugh)\b', text, re.IGNORECASE):
            tags.append(self.TTS_TAGS["laugh"])

        # Soupirs
        if re.search(r'\b(soupir|sigh)\b', text, re.IGNORECASE):
            tags.append(self.TTS_TAGS["sigh"])

        # Halètements (peur, surprise)
        if re.search(r'\b(haleter|gasp|souffle coupe)\b', text, re.IGNORECASE):
            tags.append(self.TTS_TAGS["gasp"])

        return tags

    def _detect_emphasis_words(self, text: str) -> list[str]:
        """Detecte les mots a accentuer."""
        emphasis = []

        # Mots en majuscules
        for word in re.findall(r'\b[A-Z]{2,}\b', text):
            emphasis.append(word)

        # Mots en italique (si Markdown)
        for match in re.finditer(r'\*([^*]+)\*|_([^_]+)_', text):
            word = match.group(1) or match.group(2)
            emphasis.append(word)

        # Mots entre guillemets simples (emphasis)
        for match in re.finditer(r"'([^']+)'", text):
            emphasis.append(match.group(1))

        return emphasis

    def analyze(self, text: str) -> EmotionAnalysis:
        """
        Analyse complete d'un segment de texte.

        Returns:
            EmotionAnalysis avec emotion, intensite, prosodie et tags
        """
        emotion, confidence = self._detect_emotion(text)
        intensity = self._detect_intensity(text)
        prosody = self._compute_prosody(emotion, intensity)
        tags = self._detect_tts_tags(text)
        emphasis = self._detect_emphasis_words(text)

        return EmotionAnalysis(
            text=text,
            emotion=emotion,
            intensity=intensity,
            prosody=prosody,
            tags=tags,
            emphasis_words=emphasis
        )

    def analyze_sentences(self, text: str) -> list[EmotionAnalysis]:
        """Analyse chaque phrase separement."""
        # Decouper en phrases
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [self.analyze(s) for s in sentences if s.strip()]


class BreathPlacer:
    """
    Ajoute des micro-pauses de respiration naturelles.

    Simule les pauses respiratoires d'un narrateur humain:
    - Avant les phrases longues
    - Apres les virgules dans des phrases complexes
    - Aux transitions emotionnelles
    """

    # Duree des respirations en secondes
    BREATH_SHORT = 0.15
    BREATH_MEDIUM = 0.25
    BREATH_LONG = 0.4

    def __init__(self, words_per_breath: int = 25):
        """
        Args:
            words_per_breath: Nombre moyen de mots avant respiration
        """
        self.words_per_breath = words_per_breath

    def add_breath_pauses(self, text: str) -> str:
        """
        Ajoute des marqueurs de respiration au texte.

        Les marqueurs sont des pauses silencieuses: [pause:0.2]
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []

        word_count = 0
        for sentence in sentences:
            words = len(sentence.split())

            # Respiration avant phrase longue
            if words > 15 and word_count > 10:
                result.append(f"[pause:{self.BREATH_MEDIUM}]")
                word_count = 0

            result.append(sentence)
            word_count += words

            # Respiration periodique
            if word_count >= self.words_per_breath:
                word_count = 0

        return " ".join(result)

    def insert_breath_markers(
        self,
        analyses: list[EmotionAnalysis]
    ) -> list[EmotionAnalysis]:
        """
        Ajoute des indications de respiration aux analyses.

        Modifie prosody.breath_before selon le contexte.
        """
        for i, analysis in enumerate(analyses):
            # Respiration avant changement d'emotion forte
            if i > 0:
                prev = analyses[i - 1]
                if (analysis.intensity in [Intensity.HIGH, Intensity.EXTREME]
                        and prev.emotion != analysis.emotion):
                    analysis.prosody.breath_before = True

            # Respiration avant suspense
            if analysis.emotion == Emotion.SUSPENSE:
                analysis.prosody.breath_before = True

        return analyses


def analyze_text_emotions(
    text: str,
    lang: str = "fr",
    add_breath: bool = True
) -> list[EmotionAnalysis]:
    """
    Analyse les emotions d'un texte complet.

    Args:
        text: Texte a analyser
        lang: Code langue
        add_breath: Ajouter des pauses de respiration

    Returns:
        Liste d'analyses par phrase
    """
    analyzer = EmotionAnalyzer(lang=lang)
    analyses = analyzer.analyze_sentences(text)

    if add_breath:
        placer = BreathPlacer()
        analyses = placer.insert_breath_markers(analyses)

    return analyses


if __name__ == "__main__":
    # Test
    test_text = """
    Marie etait si heureuse ! Elle avait enfin reussi.

    Mais soudain, un bruit etrange la fit sursauter. Qu'est-ce que c'etait ?

    « Au secours ! » cria-t-elle, terrifiee.

    Pierre accourut. « Calme-toi, ce n'est rien », murmura-t-il tendrement.
    """

    print("=== Analyse emotionnelle ===\n")
    analyses = analyze_text_emotions(test_text)

    for analysis in analyses:
        print(f"Texte: {analysis.text[:50]}...")
        print(f"  Emotion: {analysis.emotion.value}")
        print(f"  Intensite: {analysis.intensity.value}")
        print(f"  Prosodie: speed={analysis.prosody.speed:.2f}, "
              f"pitch={analysis.prosody.pitch:.2f}, "
              f"volume={analysis.prosody.volume:.2f}")
        if analysis.prosody.breath_before:
            print(f"  [Respiration avant]")
        print()
