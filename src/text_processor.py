"""
Traitement avancé du texte pour TTS.

Fonctionnalités:
- Chunking intelligent (découpage aux frontières naturelles)
- Correction de prononciation
- Tags émotionnels et prosodiques
- Nettoyage et normalisation
"""
import re
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import json


@dataclass
class TextChunk:
    """Un segment de texte optimisé pour TTS."""
    text: str
    index: int
    is_dialogue: bool = False
    emotion: Optional[str] = None
    pause_before: float = 0.0  # secondes
    pause_after: float = 0.0


class PronunciationCorrector:
    """
    Corrige la prononciation des mots techniques et étrangers.

    Basé sur la recherche: les TTS ont des difficultés avec les acronymes,
    noms propres, et termes techniques non présents dans leurs données
    d'entraînement.
    """

    # Dictionnaire par défaut (français)
    DEFAULT_CORRECTIONS_FR = {
        # Technologie
        "API": "A P I",
        "APIs": "A P I S",
        "URL": "U R L",
        "URLs": "U R L S",
        "HTML": "H T M L",
        "CSS": "C S S",
        "JSON": "jason",
        "SQL": "S Q L",
        "GitHub": "Guite Hub",
        "ChatGPT": "Tchatte Dji Pi Ti",
        "GPT": "Dji Pi Ti",
        "AI": "A I",
        "ML": "M L",
        "CPU": "C P U",
        "GPU": "G P U",
        "RAM": "ramme",
        "SSD": "S S D",
        "USB": "U S B",
        "WiFi": "Wi Fi",
        "Bluetooth": "Blou tousse",
        "iPhone": "aïe faune",
        "iPad": "aïe pad",
        "macOS": "mac O S",
        "iOS": "aïe O S",
        "Linux": "Linuxe",
        "Windows": "Ouine dose",
        "Python": "Païtone",
        "JavaScript": "Java scripte",

        # Entreprises
        "Google": "Gougueule",
        "Microsoft": "Maïcro softe",
        "Amazon": "Amazone",
        "Netflix": "Nète flixe",
        "Spotify": "Spoti faï",
        "Tesla": "Tèsla",

        # Expressions anglaises courantes
        "email": "i-mèle",
        "emails": "i-mèles",
        "newsletter": "niouze lèteur",
        "deadline": "dèd laïne",
        "feedback": "fid bak",
        "hashtag": "ache tague",
        "startup": "starte eup",
        "podcast": "pod kaste",
        "streaming": "stri mingue",
        "online": "one laïne",
        "offline": "offe laïne",

        # Ponctuation spéciale
        "...": " ",
        "—": ", ",
        "–": ", ",

        # Nombres et unités
        "km/h": "kilomètres heure",
        "m/s": "mètres par seconde",
        "°C": "degrés Celsius",
        "°F": "degrés Fahrenheit",
        "%": " pourcent",
        "€": " euros",
        "$": " dollars",
        "£": " livres",
    }

    DEFAULT_CORRECTIONS_EN = {
        # Technology
        "API": "A P I",
        "APIs": "A P I s",
        "URL": "U R L",
        "SQL": "sequel",
        "JSON": "jay-son",
        "GitHub": "git hub",
        "ChatGPT": "chat G P T",
        "iOS": "eye O S",
        "macOS": "mac O S",

        # Common abbreviations
        "etc.": "et cetera",
        "e.g.": "for example",
        "i.e.": "that is",
        "vs.": "versus",
        "Dr.": "Doctor",
        "Mr.": "Mister",
        "Mrs.": "Missus",
        "Ms.": "Miz",
    }

    def __init__(self, lang: str = "fr", custom_dict: Optional[dict] = None):
        """
        Args:
            lang: Code langue ("fr", "en")
            custom_dict: Dictionnaire personnalisé de corrections
        """
        self.lang = lang

        if lang == "fr":
            self.corrections = self.DEFAULT_CORRECTIONS_FR.copy()
        else:
            self.corrections = self.DEFAULT_CORRECTIONS_EN.copy()

        if custom_dict:
            self.corrections.update(custom_dict)

    def load_custom_dict(self, filepath: Path):
        """Charge un dictionnaire personnalisé depuis un fichier JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            custom = json.load(f)
            self.corrections.update(custom)

    def correct(self, text: str) -> str:
        """Applique les corrections de prononciation."""
        for word, replacement in self.corrections.items():
            escaped = re.escape(word)

            # Utiliser des frontieres de mots pour les patterns alphanumeriques
            # pour eviter de remplacer "ai" dans "mais", "avait", etc.
            if word[0].isalnum() and word[-1].isalnum():
                pattern = re.compile(rf'\b{escaped}\b', re.IGNORECASE)
            else:
                # Pour les symboles (%, €, ...) pas de frontiere
                pattern = re.compile(escaped, re.IGNORECASE)

            text = pattern.sub(replacement, text)

        return text


class EmotionTagger:
    """
    Ajoute des tags émotionnels au texte pour TTS expressif.

    Basé sur la recherche de Pethe et al. (2025) sur la prédiction
    de prosodie à partir du texte.
    """

    # Patterns pour détecter les émotions
    EMOTION_PATTERNS = {
        "excited": [
            r'!{2,}',  # !!
            r'\b(incroyable|fantastique|génial|super|wow|hourra)\b',
            r'\b(amazing|awesome|incredible|fantastic|wonderful)\b',
        ],
        "sad": [
            r'\b(triste|malheureux|désolé|hélas|malheureusement)\b',
            r'\b(sad|sorry|unfortunately|alas|grief)\b',
        ],
        "angry": [
            r'\b(furieux|énervé|colère|rage)\b',
            r'\b(angry|furious|rage|mad)\b',
        ],
        "whisper": [
            r'\b(chuchot|murmur|secret|silence)\b',
            r'\b(whisper|murmur|quiet|secret)\b',
        ],
        "laugh": [
            r'\b(ha\s*ha|hi\s*hi|ho\s*ho|rire|rigol)\b',
            r'\b(haha|hehe|lol|laugh)\b',
        ],
    }

    # Tags supportés par différents moteurs
    KOKORO_TAGS = {
        "laugh": "[laugh]",
        "sigh": "[sigh]",
        "cough": "[cough]",
        "gasp": "[gasp]",
        "chuckle": "[chuckle]",
    }

    CHATTERBOX_TAGS = {
        "laugh": "[laugh]",
        "sigh": "[sigh]",
        "cough": "[cough]",
        "gasp": "[gasp]",
        "chuckle": "[chuckle]",
        "groan": "[groan]",
    }

    def __init__(self, engine: str = "kokoro"):
        """
        Args:
            engine: Moteur TTS ("kokoro", "chatterbox", "orpheus")
        """
        self.engine = engine
        if engine == "chatterbox":
            self.tags = self.CHATTERBOX_TAGS
        else:
            self.tags = self.KOKORO_TAGS

    def detect_emotion(self, text: str) -> Optional[str]:
        """Détecte l'émotion dominante dans le texte."""
        for emotion, patterns in self.EMOTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return emotion
        return None

    def add_tags(self, text: str) -> str:
        """Ajoute des tags émotionnels au texte."""
        # Détecter les rires explicites
        text = re.sub(
            r'\b(ha\s*ha\s*ha|hi\s*hi\s*hi)\b',
            self.tags.get("laugh", "") + " ",
            text,
            flags=re.IGNORECASE
        )

        # Détecter les soupirs
        text = re.sub(
            r'\b(soupir|sigh)\b',
            self.tags.get("sigh", ""),
            text,
            flags=re.IGNORECASE
        )

        return text


class TextChunker:
    """
    Découpe intelligente du texte pour TTS optimal.

    Basé sur les recommandations de Deepgram et la recherche
    sur le chunking pour la synthèse vocale.

    Références:
    - Deepgram: Text Chunking for TTS Optimization
    - Microsoft SpeechT5: Long text chunking strategies
    """

    def __init__(
        self,
        max_chars: int = 500,
        min_chars: int = 50,
        preserve_paragraphs: bool = True
    ):
        """
        Args:
            max_chars: Taille maximale d'un chunk
            min_chars: Taille minimale (évite chunks trop courts)
            preserve_paragraphs: Respecter les frontières de paragraphes
        """
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.preserve_paragraphs = preserve_paragraphs

    def chunk(self, text: str) -> list[TextChunk]:
        """
        Découpe le texte en chunks optimisés pour TTS.

        Stratégie (par ordre de priorité):
        1. Frontières de paragraphes
        2. Frontières de phrases
        3. Frontières de clauses (virgules, etc.)
        4. Découpage par caractères (dernier recours)
        """
        chunks = []

        # Étape 1: Séparer par paragraphes
        if self.preserve_paragraphs:
            paragraphs = re.split(r'\n\s*\n', text)
        else:
            paragraphs = [text]

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Si le paragraphe est assez court, le garder tel quel
            if len(para) <= self.max_chars:
                chunks.append(para)
                continue

            # Étape 2: Découper par phrases
            sentences = self._split_sentences(para)

            current_chunk = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Si la phrase seule est trop longue, la découper
                if len(sentence) > self.max_chars:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""

                    # Découper la phrase longue
                    sub_chunks = self._split_long_sentence(sentence)
                    chunks.extend(sub_chunks)
                    continue

                # Essayer d'ajouter au chunk courant
                test_chunk = current_chunk + " " + sentence if current_chunk else sentence

                if len(test_chunk) <= self.max_chars:
                    current_chunk = test_chunk
                else:
                    # Sauvegarder le chunk courant et commencer un nouveau
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence

            # Ajouter le dernier chunk
            if current_chunk:
                chunks.append(current_chunk)

        # Convertir en objets TextChunk
        result = []
        for i, text in enumerate(chunks):
            is_dialogue = self._is_dialogue(text)
            result.append(TextChunk(
                text=text.strip(),
                index=i,
                is_dialogue=is_dialogue,
                pause_after=0.3 if is_dialogue else 0.1
            ))

        return result

    def _split_sentences(self, text: str) -> list[str]:
        """Découpe en phrases."""
        # Pattern pour fin de phrase
        pattern = r'(?<=[.!?])\s+'
        sentences = re.split(pattern, text)
        return [s for s in sentences if s.strip()]

    def _split_long_sentence(self, sentence: str) -> list[str]:
        """Découpe une phrase trop longue."""
        chunks = []

        # Essayer de découper aux clauses (;:,)
        parts = re.split(r'([;:])\s*', sentence)

        current = ""
        for part in parts:
            if part in [';', ':']:
                current += part
                continue

            test = current + " " + part if current else part

            if len(test) <= self.max_chars:
                current = test
            else:
                if current:
                    chunks.append(current)

                # Si la partie est encore trop longue, découper aux virgules
                if len(part) > self.max_chars:
                    sub_parts = self._split_at_commas(part)
                    chunks.extend(sub_parts[:-1])
                    current = sub_parts[-1] if sub_parts else ""
                else:
                    current = part

        if current:
            chunks.append(current)

        return chunks

    def _split_at_commas(self, text: str) -> list[str]:
        """Découpe aux virgules."""
        parts = re.split(r',\s*', text)

        chunks = []
        current = ""

        for part in parts:
            test = current + ", " + part if current else part

            if len(test) <= self.max_chars:
                current = test
            else:
                if current:
                    chunks.append(current)
                current = part

        if current:
            chunks.append(current)

        return chunks

    def _is_dialogue(self, text: str) -> bool:
        """Détecte si le texte est un dialogue."""
        # Guillemets ou tirets de dialogue
        return bool(re.search(r'[«»"""]|^[\s]*[-–—]', text))


class TextProcessor:
    """
    Processeur de texte complet pour audiobooks.

    Combine toutes les fonctionnalités:
    - Nettoyage et normalisation
    - Correction prononciation
    - Tags émotionnels
    - Chunking intelligent
    """

    def __init__(
        self,
        lang: str = "fr",
        engine: str = "kokoro",
        max_chunk_size: int = 500,
        custom_pronunciation: Optional[dict] = None
    ):
        self.lang = lang
        self.pronunciation = PronunciationCorrector(lang, custom_pronunciation)
        self.emotion = EmotionTagger(engine)
        self.chunker = TextChunker(max_chars=max_chunk_size)

    def process(self, text: str) -> list[TextChunk]:
        """
        Traite le texte complet et retourne des chunks optimisés.
        """
        # Étape 1: Nettoyage de base
        text = self._clean_text(text)

        # Étape 2: Correction prononciation
        text = self.pronunciation.correct(text)

        # Étape 3: Tags émotionnels
        text = self.emotion.add_tags(text)

        # Étape 4: Chunking
        chunks = self.chunker.chunk(text)

        # Étape 5: Détecter émotions par chunk
        for chunk in chunks:
            chunk.emotion = self.emotion.detect_emotion(chunk.text)

        return chunks

    def _clean_text(self, text: str) -> str:
        """Nettoie le texte pour TTS."""
        # Normaliser les espaces
        text = re.sub(r'\s+', ' ', text)

        # Normaliser les guillemets
        text = text.replace('«', '"').replace('»', '"')
        text = text.replace('"', '"').replace('"', '"')

        # Normaliser les tirets
        text = text.replace('—', ' - ').replace('–', ' - ')

        # Supprimer les caractères spéciaux problématiques
        text = re.sub(r'[*_~`#]', '', text)

        # Normaliser la ponctuation multiple
        text = re.sub(r'\.{4,}', '...', text)
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)

        return text.strip()

    def process_to_text(self, text: str) -> str:
        """Traite et retourne le texte complet (sans chunking)."""
        text = self._clean_text(text)
        text = self.pronunciation.correct(text)
        text = self.emotion.add_tags(text)
        return text


# Fonction utilitaire
def process_for_tts(
    text: str,
    lang: str = "fr",
    engine: str = "kokoro",
    chunk: bool = True
) -> str | list[TextChunk]:
    """
    Fonction simple pour traiter du texte pour TTS.

    Args:
        text: Texte à traiter
        lang: Code langue
        engine: Moteur TTS cible
        chunk: Si True, retourne des chunks, sinon texte complet

    Returns:
        Liste de TextChunk ou texte traité
    """
    processor = TextProcessor(lang=lang, engine=engine)

    if chunk:
        return processor.process(text)
    else:
        return processor.process_to_text(text)
