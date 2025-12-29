"""
Controle d'exageration emotionnelle et phonemes personnalises.

Fonctionnalites:
- Controle de l'intensite emotionnelle (0.0 a 1.0)
- Phonemes personnalises (IPA et ARPAbet)
- Dictionnaire de prononciation etendu
- Gestion des noms propres et mots etrangers
"""
import re
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import json


@dataclass
class EmotionSettings:
    """Parametres de controle emotionnel."""
    intensity: float = 0.5          # 0.0 = neutre, 1.0 = tres expressif
    stability: float = 0.7          # 0.0 = variable, 1.0 = stable
    style_exaggeration: float = 0.5 # Exageration du style


class EmotionController:
    """
    Controleur d'exageration emotionnelle.

    Permet d'ajuster l'expressivite de la voix.
    """

    # Multiplicateurs de prosodie par niveau d'intensite
    INTENSITY_MULTIPLIERS = {
        # (speed_mult, pitch_mult, volume_mult)
        0.0: (1.0, 0.0, 1.0),      # Neutre
        0.25: (1.02, 0.1, 1.02),   # Leger
        0.5: (1.05, 0.2, 1.05),    # Modere
        0.75: (1.1, 0.35, 1.1),    # Fort
        1.0: (1.2, 0.5, 1.15),     # Maximum
    }

    def __init__(self, settings: Optional[EmotionSettings] = None):
        self.settings = settings or EmotionSettings()

    def _interpolate_multipliers(
        self,
        intensity: float
    ) -> Tuple[float, float, float]:
        """Interpole les multiplicateurs pour une intensite donnee."""
        intensity = max(0.0, min(1.0, intensity))

        # Trouver les bornes
        keys = sorted(self.INTENSITY_MULTIPLIERS.keys())
        lower_key = 0.0
        upper_key = 1.0

        for k in keys:
            if k <= intensity:
                lower_key = k
            if k >= intensity:
                upper_key = k
                break

        if lower_key == upper_key:
            return self.INTENSITY_MULTIPLIERS[lower_key]

        # Interpolation lineaire
        ratio = (intensity - lower_key) / (upper_key - lower_key)
        lower = self.INTENSITY_MULTIPLIERS[lower_key]
        upper = self.INTENSITY_MULTIPLIERS[upper_key]

        return (
            lower[0] + (upper[0] - lower[0]) * ratio,
            lower[1] + (upper[1] - lower[1]) * ratio,
            lower[2] + (upper[2] - lower[2]) * ratio,
        )

    def calculate_prosody(
        self,
        base_speed: float = 1.0,
        base_pitch: float = 0.0,
        base_volume: float = 1.0,
        emotion_type: Optional[str] = None
    ) -> dict:
        """
        Calcule la prosodie avec l'exageration emotionnelle.

        Args:
            base_speed: Vitesse de base
            base_pitch: Pitch de base
            base_volume: Volume de base
            emotion_type: Type d'emotion (joy, sad, anger, etc.)

        Returns:
            Dict avec speed, pitch, volume ajustes
        """
        speed_mult, pitch_mult, vol_mult = self._interpolate_multipliers(
            self.settings.intensity
        )

        # Appliquer l'exageration
        exag = self.settings.style_exaggeration

        # Ajustements specifiques par emotion
        emotion_adjustments = {
            "joy": (0.05, 0.2, 0.05),
            "sad": (-0.08, -0.15, -0.05),
            "anger": (0.1, 0.1, 0.15),
            "fear": (0.15, 0.25, 0.0),
            "surprise": (0.05, 0.3, 0.05),
            "tender": (-0.05, -0.1, -0.05),
            "excitement": (0.12, 0.25, 0.1),
        }

        if emotion_type and emotion_type in emotion_adjustments:
            adj = emotion_adjustments[emotion_type]
            speed_mult += adj[0] * exag
            pitch_mult += adj[1] * exag
            vol_mult += adj[2] * exag

        return {
            "speed": base_speed * speed_mult,
            "pitch": base_pitch + pitch_mult,
            "volume": base_volume * vol_mult,
        }


@dataclass
class PhonemeEntry:
    """Entree de dictionnaire de phonemes."""
    word: str
    ipa: str                        # Transcription IPA
    arpabet: Optional[str] = None   # Transcription ARPAbet (optionnel)
    language: str = "fr"
    notes: str = ""


class PhonemeProcessor:
    """
    Processeur de phonemes personnalises.

    Supporte:
    - IPA (International Phonetic Alphabet)
    - ARPAbet (pour compatibilite)
    - Dictionnaire personnalise
    """

    # Dictionnaire de prononciation francais integre
    BUILTIN_FR = {
        # Noms propres courants
        "Jean": "ʒɑ̃",
        "Jacques": "ʒak",
        "Pierre": "pjɛʁ",
        "Marie": "maʁi",
        "François": "fʁɑ̃swa",
        "Dupont": "dypɔ̃",
        "Durand": "dyʁɑ̃",
        "Paris": "paʁi",
        "Lyon": "ljɔ̃",
        "Marseille": "maʁsɛj",

        # Mots anglais courants
        "email": "imɛl",
        "smartphone": "smaʁtfon",
        "startup": "staʁtœp",
        "meeting": "mitiŋ",
        "weekend": "wikɛnd",
        "shopping": "ʃɔpiŋ",
        "hashtag": "aʃtag",
        "selfie": "sɛlfi",
        "podcast": "pɔdkast",
        "streaming": "stʁimiŋ",

        # Termes techniques
        "API": "a pe i",
        "HTML": "aʃ te ɛm ɛl",
        "CSS": "se ɛs ɛs",
        "JavaScript": "ʒavaskʁipt",
        "Python": "pitɔn",
        "Linux": "linuks",
        "GitHub": "gitœb",

        # Marques
        "Google": "gugl",
        "Apple": "apœl",
        "Microsoft": "majkʁosɔft",
        "Amazon": "amazon",
        "Netflix": "nɛtfliks",
        "Spotify": "spɔtifaj",
        "Tesla": "tɛsla",
        "iPhone": "ajfon",
        "iPad": "ajpad",
    }

    # Dictionnaire anglais integre
    BUILTIN_EN = {
        # Noms propres
        "Jean": "ʒɑn",
        "Pierre": "piˈɛr",

        # Termes francais en anglais
        "croissant": "kwɑˈsɑnt",
        "café": "kæˈfeɪ",
        "fiancé": "fiˈɑnseɪ",
        "résumé": "ˈrezjʊmeɪ",

        # Marques
        "Porsche": "ˈpɔrʃə",
        "Hermès": "ɛrˈmɛz",
        "Givenchy": "ʒiˈvɑnʃi",
    }

    def __init__(self, lang: str = "fr"):
        self.lang = lang
        self._custom_dict: Dict[str, PhonemeEntry] = {}

        # Charger le dictionnaire integre
        if lang == "fr":
            self._builtin = self.BUILTIN_FR
        else:
            self._builtin = self.BUILTIN_EN

    def add_word(
        self,
        word: str,
        ipa: str,
        arpabet: Optional[str] = None,
        notes: str = ""
    ):
        """Ajoute un mot au dictionnaire personnalise."""
        self._custom_dict[word.lower()] = PhonemeEntry(
            word=word,
            ipa=ipa,
            arpabet=arpabet,
            language=self.lang,
            notes=notes
        )

    def load_dictionary(self, path: Path):
        """
        Charge un dictionnaire depuis un fichier JSON.

        Format attendu:
        {
            "words": {
                "mot": {"ipa": "...", "arpabet": "...", "notes": "..."}
            }
        }
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for word, entry in data.get("words", {}).items():
            self.add_word(
                word=word,
                ipa=entry.get("ipa", ""),
                arpabet=entry.get("arpabet"),
                notes=entry.get("notes", "")
            )

    def save_dictionary(self, path: Path):
        """Sauvegarde le dictionnaire personnalise."""
        data = {
            "words": {
                entry.word: {
                    "ipa": entry.ipa,
                    "arpabet": entry.arpabet,
                    "notes": entry.notes,
                }
                for entry in self._custom_dict.values()
            }
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_phoneme(self, word: str) -> Optional[str]:
        """Recupere la transcription phonetique d'un mot."""
        word_lower = word.lower()

        # Dictionnaire personnalise d'abord
        if word_lower in self._custom_dict:
            return self._custom_dict[word_lower].ipa

        # Puis dictionnaire integre
        if word in self._builtin:
            return self._builtin[word]

        if word_lower in self._builtin:
            return self._builtin[word_lower]

        return None

    def apply_phonemes(self, text: str) -> str:
        """
        Applique les phonemes au texte.

        Remplace les mots connus par leur transcription phonetique.
        """
        # Trouver tous les mots
        words = re.findall(r'\b[\w]+\b', text)

        for word in words:
            phoneme = self.get_phoneme(word)
            if phoneme:
                # Remplacer par une version prononçable
                # (pas de tags SSML car Kokoro ne les supporte pas)
                # On utilise une approximation
                replacement = self._phoneme_to_text(phoneme)
                if replacement != word:
                    text = re.sub(
                        rf'\b{re.escape(word)}\b',
                        replacement,
                        text,
                        count=1
                    )

        return text

    def _phoneme_to_text(self, ipa: str) -> str:
        """
        Convertit une transcription IPA en texte lisible.

        Approximation pour les TTS qui ne supportent pas IPA.
        """
        # Mapping simplifie IPA -> texte francais
        ipa_to_text = {
            'ʒ': 'j',
            'ɑ̃': 'an',
            'ɛ̃': 'in',
            'ɔ̃': 'on',
            'œ̃': 'un',
            'ʁ': 'r',
            'ɛ': 'è',
            'ø': 'eu',
            'œ': 'eu',
            'y': 'u',
            'ɥ': 'u',
            'ŋ': 'ng',
            'ʃ': 'ch',
            'ɲ': 'gn',
        }

        result = ipa
        for ipa_char, text_char in ipa_to_text.items():
            result = result.replace(ipa_char, text_char)

        return result

    def generate_ssml_phoneme(
        self,
        word: str,
        ipa: str,
        alphabet: str = "ipa"
    ) -> str:
        """
        Genere un tag SSML pour un phoneme.

        Note: Kokoro ne supporte pas SSML, mais utile pour d'autres moteurs.
        """
        return f'<phoneme alphabet="{alphabet}" ph="{ipa}">{word}</phoneme>'


class PronunciationManager:
    """
    Gestionnaire de prononciation complet.

    Combine:
    - Dictionnaire de corrections simples
    - Phonemes IPA pour cas complexes
    - Regles de prononciation automatiques
    """

    def __init__(self, lang: str = "fr"):
        self.lang = lang
        self.phoneme_processor = PhonemeProcessor(lang)

        # Corrections simples (mot -> remplacement)
        self._simple_corrections: Dict[str, str] = {}

        # Regles regex
        self._regex_rules: List[Tuple[str, str]] = []

    def add_correction(self, word: str, replacement: str):
        """Ajoute une correction simple."""
        self._simple_corrections[word.lower()] = replacement

    def add_phoneme(
        self,
        word: str,
        ipa: str,
        arpabet: Optional[str] = None
    ):
        """Ajoute un phoneme."""
        self.phoneme_processor.add_word(word, ipa, arpabet)

    def add_regex_rule(self, pattern: str, replacement: str):
        """Ajoute une regle regex."""
        self._regex_rules.append((pattern, replacement))

    def process(self, text: str) -> str:
        """Applique toutes les corrections."""
        # 1. Corrections simples
        for word, replacement in self._simple_corrections.items():
            # Pour les mots avec ponctuation (ex: "etc.", "M."), ne pas utiliser \b a la fin
            escaped = re.escape(word)
            if word.endswith('.') or word.endswith('°'):
                pattern = rf'\b{escaped}(?=\s|$|[,;:!?])'
            else:
                pattern = rf'\b{escaped}\b'
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 2. Regles regex
        for pattern, replacement in self._regex_rules:
            text = re.sub(pattern, replacement, text)

        # 3. Phonemes (si necessaire)
        text = self.phoneme_processor.apply_phonemes(text)

        return text

    def load_config(self, path: Path):
        """
        Charge une configuration de prononciation.

        Format JSON:
        {
            "corrections": {"mot": "remplacement"},
            "phonemes": {"mot": {"ipa": "..."}},
            "rules": [{"pattern": "...", "replacement": "..."}]
        }
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for word, repl in data.get("corrections", {}).items():
            self.add_correction(word, repl)

        for word, entry in data.get("phonemes", {}).items():
            self.add_phoneme(
                word,
                entry.get("ipa", ""),
                entry.get("arpabet")
            )

        for rule in data.get("rules", []):
            self.add_regex_rule(
                rule.get("pattern", ""),
                rule.get("replacement", "")
            )


def create_pronunciation_config(lang: str = "fr") -> PronunciationManager:
    """Cree un gestionnaire de prononciation avec config par defaut."""
    manager = PronunciationManager(lang)

    if lang == "fr":
        # Ajouts specifiques au francais
        manager.add_correction("etc.", "et cetera")
        manager.add_correction("M.", "Monsieur")
        manager.add_correction("Mme", "Madame")
        manager.add_correction("Dr", "Docteur")
        manager.add_correction("n°", "numero")

        # Nombres en lettres pour certains cas
        manager.add_regex_rule(r'\b(\d+)e\b', r'\1eme')
        manager.add_regex_rule(r'\b1er\b', 'premier')
        manager.add_regex_rule(r'\b1ere\b', 'premiere')

    return manager


if __name__ == "__main__":
    print("=== Test Controle Emotionnel ===\n")

    controller = EmotionController(EmotionSettings(intensity=0.8))
    result = controller.calculate_prosody(
        base_speed=1.0,
        emotion_type="joy"
    )
    print(f"Joy (intensity=0.8): {result}")

    print("\n=== Test Phonemes ===\n")

    processor = PhonemeProcessor("fr")
    print("Phoneme 'Jean':", processor.get_phoneme("Jean"))
    print("Phoneme 'Google':", processor.get_phoneme("Google"))

    text = "Jean utilise Google et JavaScript."
    processed = processor.apply_phonemes(text)
    print(f"\nOriginal: {text}")
    print(f"Traite: {processed}")
