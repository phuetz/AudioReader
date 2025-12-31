"""
Controle Mot-par-Mot (SSML-like) v2.4.

Permet d'appliquer des modifications prosodiques au niveau du mot:
- Emphase: mots importants mis en valeur
- Pitch: variation de hauteur locale
- Vitesse: acceleration/ralentissement local
- Pauses: micro-pauses avant/apres

Syntaxe supportee:
- <em>mot</em> ou *mot* : emphase
- <slow>texte</slow> : ralentissement
- <fast>texte</fast> : acceleration
- <pitch high>texte</pitch> : pitch eleve
- <pitch low>texte</pitch> : pitch grave
- <pause/> ou [pause] : micro-pause
- <whisper>texte</whisper> : chuchotement
- <loud>texte</loud> : plus fort
"""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np


class WordModificationType(Enum):
    """Types de modifications de mots."""
    NORMAL = "normal"
    EMPHASIS = "emphasis"
    SLOW = "slow"
    FAST = "fast"
    PITCH_HIGH = "pitch_high"
    PITCH_LOW = "pitch_low"
    WHISPER = "whisper"
    LOUD = "loud"
    PAUSE_BEFORE = "pause_before"
    PAUSE_AFTER = "pause_after"


@dataclass
class WordProsody:
    """Parametres prosodiques pour un mot ou groupe de mots."""
    text: str
    speed_multiplier: float = 1.0
    pitch_shift: float = 0.0  # En demi-tons
    volume_multiplier: float = 1.0
    pause_before: float = 0.0  # Secondes
    pause_after: float = 0.0
    is_emphasis: bool = False
    is_whisper: bool = False


@dataclass
class ProcessedText:
    """Texte traite avec annotations prosodiques."""
    clean_text: str  # Texte sans balises
    word_prosodies: List[WordProsody] = field(default_factory=list)
    global_modifications: Dict[str, float] = field(default_factory=dict)


class WordLevelController:
    """
    Controleur de prosodie au niveau du mot.

    Parse le texte avec balises et genere des instructions prosodiques.
    """

    # Patterns de balises supportees
    TAG_PATTERNS = {
        # Emphase
        r'<em>(.*?)</em>': WordModificationType.EMPHASIS,
        r'\*([^*]+)\*': WordModificationType.EMPHASIS,
        r'_([^_]+)_': WordModificationType.EMPHASIS,

        # Vitesse
        r'<slow>(.*?)</slow>': WordModificationType.SLOW,
        r'<fast>(.*?)</fast>': WordModificationType.FAST,

        # Pitch
        r'<pitch\s+high>(.*?)</pitch>': WordModificationType.PITCH_HIGH,
        r'<pitch\s+low>(.*?)</pitch>': WordModificationType.PITCH_LOW,
        r'<high>(.*?)</high>': WordModificationType.PITCH_HIGH,
        r'<low>(.*?)</low>': WordModificationType.PITCH_LOW,

        # Volume
        r'<whisper>(.*?)</whisper>': WordModificationType.WHISPER,
        r'<loud>(.*?)</loud>': WordModificationType.LOUD,
        r'<soft>(.*?)</soft>': WordModificationType.WHISPER,

        # Pauses
        r'<pause\s*/?>': WordModificationType.PAUSE_BEFORE,
        r'\[pause\]': WordModificationType.PAUSE_BEFORE,
        r'\[pause:(\d+\.?\d*)\]': WordModificationType.PAUSE_BEFORE,
    }

    # Valeurs par defaut pour chaque modification
    MODIFICATION_VALUES = {
        WordModificationType.EMPHASIS: {
            "speed_multiplier": 0.92,
            "pitch_shift": 0.5,
            "volume_multiplier": 1.1,
            "pause_before": 0.05,
        },
        WordModificationType.SLOW: {
            "speed_multiplier": 0.8,
        },
        WordModificationType.FAST: {
            "speed_multiplier": 1.2,
        },
        WordModificationType.PITCH_HIGH: {
            "pitch_shift": 2.0,
        },
        WordModificationType.PITCH_LOW: {
            "pitch_shift": -2.0,
        },
        WordModificationType.WHISPER: {
            "volume_multiplier": 0.6,
            "speed_multiplier": 0.9,
            "is_whisper": True,
        },
        WordModificationType.LOUD: {
            "volume_multiplier": 1.3,
            "pitch_shift": 0.3,
        },
        WordModificationType.PAUSE_BEFORE: {
            "pause_before": 0.15,
        },
    }

    # Mots francais a accentuer automatiquement
    AUTO_EMPHASIS_WORDS_FR = {
        # Adverbes d'intensite
        "jamais", "toujours", "absolument", "vraiment", "totalement",
        "completement", "parfaitement", "exactement", "certainement",
        # Marqueurs temporels forts
        "soudain", "soudainement", "brusquement", "subitement",
        "enfin", "finalement", "desormais",
        # Connecteurs forts
        "mais", "cependant", "pourtant", "neanmoins", "toutefois",
        "or", "donc", "ainsi", "alors",
        # Negations fortes
        "rien", "personne", "aucun", "aucune", "jamais",
        # Autres emphases
        "meme", "seul", "unique", "premier", "dernier",
        "terrible", "incroyable", "extraordinaire",
    }

    def __init__(
        self,
        auto_emphasis: bool = True,
        emphasis_strength: float = 1.0,
        lang: str = "fr"
    ):
        """
        Initialise le controleur.

        Args:
            auto_emphasis: Detecter automatiquement les mots a accentuer
            emphasis_strength: Force de l'emphase (0-2)
            lang: Code langue
        """
        self.auto_emphasis = auto_emphasis
        self.emphasis_strength = emphasis_strength
        self.lang = lang

    def process_text(self, text: str) -> ProcessedText:
        """
        Traite un texte avec balises prosodiques.

        Args:
            text: Texte avec balises optionnelles

        Returns:
            ProcessedText avec texte nettoye et annotations
        """
        # Collecter toutes les modifications
        modifications: List[Tuple[int, int, str, WordModificationType, Optional[str]]] = []

        # Parser les balises
        for pattern, mod_type in self.TAG_PATTERNS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                start, end = match.span()

                # Extraire le contenu (groupe 1 si present)
                if match.lastindex and match.lastindex >= 1:
                    content = match.group(1)
                else:
                    content = None

                modifications.append((start, end, match.group(0), mod_type, content))

        # Trier par position
        modifications.sort(key=lambda x: x[0])

        # Construire le texte nettoye et les prosodies
        clean_parts = []
        word_prosodies = []
        last_end = 0

        for start, end, original, mod_type, content in modifications:
            # Ajouter le texte avant la balise
            if start > last_end:
                before_text = text[last_end:start]
                clean_parts.append(before_text)

                # Traiter les mots normaux (avec auto-emphase si active)
                self._process_normal_text(before_text, word_prosodies)

            # Traiter le contenu de la balise
            if content:
                clean_parts.append(content)
                prosody = self._create_prosody_for_modification(content, mod_type)
                word_prosodies.append(prosody)
            elif mod_type == WordModificationType.PAUSE_BEFORE:
                # Pause sans contenu
                pause_duration = 0.15
                # Verifier si une duree est specifiee
                pause_match = re.search(r':(\d+\.?\d*)', original)
                if pause_match:
                    pause_duration = float(pause_match.group(1))

                # Ajouter la pause au dernier mot si existant
                if word_prosodies:
                    word_prosodies[-1].pause_after += pause_duration

            last_end = end

        # Ajouter le texte restant
        if last_end < len(text):
            remaining = text[last_end:]
            clean_parts.append(remaining)
            self._process_normal_text(remaining, word_prosodies)

        clean_text = "".join(clean_parts)

        return ProcessedText(
            clean_text=clean_text,
            word_prosodies=word_prosodies
        )

    def _process_normal_text(
        self,
        text: str,
        prosodies: List[WordProsody]
    ):
        """
        Traite du texte normal (sans balises).

        Applique l'auto-emphase si activee.
        """
        words = re.findall(r'\S+', text)

        for word in words:
            # Nettoyer le mot pour la comparaison
            clean_word = re.sub(r'[^\w]', '', word.lower())

            is_emphasis = False
            if self.auto_emphasis and clean_word in self.AUTO_EMPHASIS_WORDS_FR:
                is_emphasis = True

            if is_emphasis:
                prosody = self._create_prosody_for_modification(
                    word,
                    WordModificationType.EMPHASIS
                )
            else:
                prosody = WordProsody(text=word)

            prosodies.append(prosody)

    def _create_prosody_for_modification(
        self,
        text: str,
        mod_type: WordModificationType
    ) -> WordProsody:
        """
        Cree un objet WordProsody pour une modification.

        Args:
            text: Texte du mot/groupe
            mod_type: Type de modification

        Returns:
            WordProsody configure
        """
        values = self.MODIFICATION_VALUES.get(mod_type, {})

        # Appliquer le facteur d'emphase
        speed = values.get("speed_multiplier", 1.0)
        pitch = values.get("pitch_shift", 0.0) * self.emphasis_strength
        volume = values.get("volume_multiplier", 1.0)
        pause_before = values.get("pause_before", 0.0)
        pause_after = values.get("pause_after", 0.0)
        is_whisper = values.get("is_whisper", False)

        return WordProsody(
            text=text,
            speed_multiplier=speed,
            pitch_shift=pitch,
            volume_multiplier=volume,
            pause_before=pause_before,
            pause_after=pause_after,
            is_emphasis=(mod_type == WordModificationType.EMPHASIS),
            is_whisper=is_whisper
        )

    def add_emphasis_to_text(
        self,
        text: str,
        words_to_emphasize: List[str]
    ) -> str:
        """
        Ajoute des balises d'emphase autour de mots specifiques.

        Args:
            text: Texte original
            words_to_emphasize: Mots a mettre en emphase

        Returns:
            Texte avec balises d'emphase
        """
        for word in words_to_emphasize:
            # Pattern pour trouver le mot (insensible a la casse)
            pattern = rf'\b({re.escape(word)})\b'
            text = re.sub(
                pattern,
                r'<em>\1</em>',
                text,
                flags=re.IGNORECASE
            )

        return text

    def generate_prosody_curve(
        self,
        processed: ProcessedText,
        total_duration: float,
        sample_rate: int = 24000
    ) -> Dict[str, np.ndarray]:
        """
        Genere des courbes de prosodie pour l'application post-synthese.

        Args:
            processed: Texte traite
            total_duration: Duree totale de l'audio (secondes)
            sample_rate: Taux d'echantillonnage

        Returns:
            Dict avec courbes de pitch, volume, speed
        """
        total_samples = int(total_duration * sample_rate)

        # Initialiser les courbes
        pitch_curve = np.zeros(total_samples)
        volume_curve = np.ones(total_samples)
        speed_curve = np.ones(total_samples)

        # Estimer la position de chaque mot
        total_chars = sum(len(p.text) for p in processed.word_prosodies)
        if total_chars == 0:
            return {
                "pitch": pitch_curve,
                "volume": volume_curve,
                "speed": speed_curve
            }

        current_pos = 0
        for prosody in processed.word_prosodies:
            word_len = len(prosody.text)
            word_ratio = word_len / total_chars

            # Position dans l'audio
            start_sample = int(current_pos / total_chars * total_samples)
            end_sample = int((current_pos + word_len) / total_chars * total_samples)

            # Appliquer les modifications
            if start_sample < end_sample:
                pitch_curve[start_sample:end_sample] = prosody.pitch_shift
                volume_curve[start_sample:end_sample] = prosody.volume_multiplier
                speed_curve[start_sample:end_sample] = prosody.speed_multiplier

            current_pos += word_len

        return {
            "pitch": pitch_curve,
            "volume": volume_curve,
            "speed": speed_curve
        }


class TextMarkerInserter:
    """
    Insere des marqueurs prosodiques dans le texte.

    Utile pour preparer le texte avant synthese.
    """

    def insert_emphasis_markers(
        self,
        text: str,
        emphasis_words: List[str]
    ) -> str:
        """Insere des marqueurs d'emphase."""
        for word in emphasis_words:
            pattern = rf'\b({re.escape(word)})\b'
            text = re.sub(pattern, r'<em>\1</em>', text, flags=re.IGNORECASE)
        return text

    def insert_pause_markers(
        self,
        text: str,
        pause_triggers: List[str] = None
    ) -> str:
        """Insere des pauses avant certains mots."""
        if pause_triggers is None:
            pause_triggers = ["mais", "cependant", "soudain", "alors"]

        for word in pause_triggers:
            pattern = rf'\b({re.escape(word)})\b'
            text = re.sub(pattern, r'[pause:0.1] \1', text, flags=re.IGNORECASE)
        return text

    def insert_speed_markers(
        self,
        text: str,
        fast_triggers: List[str] = None,
        slow_triggers: List[str] = None
    ) -> str:
        """Insere des marqueurs de vitesse."""
        if fast_triggers:
            for word in fast_triggers:
                pattern = rf'\b({re.escape(word)})\b'
                text = re.sub(pattern, r'<fast>\1</fast>', text, flags=re.IGNORECASE)

        if slow_triggers:
            for word in slow_triggers:
                pattern = rf'\b({re.escape(word)})\b'
                text = re.sub(pattern, r'<slow>\1</slow>', text, flags=re.IGNORECASE)

        return text


def process_text_with_word_control(
    text: str,
    auto_emphasis: bool = True,
    emphasis_strength: float = 1.0,
    lang: str = "fr"
) -> ProcessedText:
    """
    Fonction utilitaire pour traiter un texte avec controle mot-par-mot.

    Args:
        text: Texte a traiter
        auto_emphasis: Activer l'auto-emphase
        emphasis_strength: Force de l'emphase
        lang: Code langue

    Returns:
        ProcessedText avec annotations
    """
    controller = WordLevelController(
        auto_emphasis=auto_emphasis,
        emphasis_strength=emphasis_strength,
        lang=lang
    )
    return controller.process_text(text)


if __name__ == "__main__":
    print("=== Test Controle Mot-par-Mot ===\n")

    # Texte de test avec balises
    test_text = """
    Il etait <em>absolument</em> certain que quelque chose allait se passer.

    « <whisper>Ecoute...</whisper> » murmura-t-elle. [pause:0.3]

    Soudain, un bruit <loud>ENORME</loud> retentit!

    <slow>Le temps semblait s'etre arrete.</slow>

    C'etait *vraiment* incroyable.
    """

    controller = WordLevelController(auto_emphasis=True)
    result = controller.process_text(test_text)

    print("Texte nettoye:")
    print(result.clean_text)

    print("\nMots avec modifications:")
    for prosody in result.word_prosodies:
        if prosody.speed_multiplier != 1.0 or prosody.pitch_shift != 0.0 or prosody.is_emphasis:
            print(f"  '{prosody.text}': speed={prosody.speed_multiplier:.2f}, "
                  f"pitch={prosody.pitch_shift:.1f}, emphasis={prosody.is_emphasis}")
