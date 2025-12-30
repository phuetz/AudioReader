"""
Generateur de preview audio rapide.

Permet de generer un extrait de 30 secondes pour tester
les voix et les parametres avant de lancer la generation complete.
"""
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
import re


@dataclass
class PreviewConfig:
    """Configuration du preview."""
    # Duree cible du preview en secondes
    target_duration: float = 30.0
    # Nombre de caracteres approximatif (15 chars/sec en moyenne)
    target_chars: int = 450
    # Inclure le debut du texte
    include_start: bool = True
    # Inclure un dialogue si possible
    include_dialogue: bool = True
    # Inclure une partie emotionnelle si possible
    include_emotional: bool = True


class PreviewGenerator:
    """
    Genere des previews audio rapides pour tester les parametres.
    """

    # Patterns pour detecter les dialogues
    DIALOGUE_PATTERNS = [
        r'[«"].*?[»"]',  # Guillemets
        r'[-–—]\s*.+',   # Tirets de dialogue
    ]

    # Mots indicateurs d'emotion
    EMOTIONAL_WORDS = {
        'fr': ['soudain', 'hurla', 'cria', 'murmura', 'sanglota', 'rit',
               'incroyable', 'terrible', 'merveilleux', 'effrayant'],
        'en': ['suddenly', 'shouted', 'screamed', 'whispered', 'sobbed',
               'laughed', 'incredible', 'terrible', 'wonderful', 'frightening']
    }

    def __init__(self, config: Optional[PreviewConfig] = None):
        self.config = config or PreviewConfig()

    def extract_preview_text(
        self,
        text: str,
        lang: str = 'fr'
    ) -> str:
        """
        Extrait un texte representatif pour le preview.

        Args:
            text: Texte complet
            lang: Langue du texte

        Returns:
            Texte extrait pour le preview
        """
        paragraphs = self._split_paragraphs(text)

        if not paragraphs:
            return text[:self.config.target_chars]

        selected = []
        total_chars = 0

        # 1. Toujours inclure le debut
        if self.config.include_start and paragraphs:
            first_para = paragraphs[0]
            selected.append(('start', first_para))
            total_chars += len(first_para)

        # 2. Chercher un dialogue
        if self.config.include_dialogue:
            dialogue_para = self._find_dialogue_paragraph(paragraphs[1:])
            if dialogue_para and total_chars + len(dialogue_para) < self.config.target_chars:
                selected.append(('dialogue', dialogue_para))
                total_chars += len(dialogue_para)

        # 3. Chercher un passage emotionnel
        if self.config.include_emotional:
            emotional_para = self._find_emotional_paragraph(paragraphs[1:], lang)
            if emotional_para and total_chars + len(emotional_para) < self.config.target_chars:
                # Eviter les doublons
                if emotional_para not in [p for _, p in selected]:
                    selected.append(('emotional', emotional_para))
                    total_chars += len(emotional_para)

        # 4. Completer avec des paragraphes suivants si necessaire
        if total_chars < self.config.target_chars * 0.7:
            for para in paragraphs[1:]:
                if para not in [p for _, p in selected]:
                    if total_chars + len(para) < self.config.target_chars:
                        selected.append(('filler', para))
                        total_chars += len(para)
                    else:
                        break

        # Assembler le preview
        preview_parts = [p for _, p in selected]
        preview_text = '\n\n'.join(preview_parts)

        # Tronquer si trop long
        if len(preview_text) > self.config.target_chars:
            preview_text = self._smart_truncate(preview_text, self.config.target_chars)

        return preview_text

    def _split_paragraphs(self, text: str) -> List[str]:
        """Decoupe le texte en paragraphes."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _find_dialogue_paragraph(self, paragraphs: List[str]) -> Optional[str]:
        """Trouve un paragraphe contenant un dialogue."""
        for para in paragraphs:
            for pattern in self.DIALOGUE_PATTERNS:
                if re.search(pattern, para):
                    return para
        return None

    def _find_emotional_paragraph(
        self,
        paragraphs: List[str],
        lang: str
    ) -> Optional[str]:
        """Trouve un paragraphe emotionnel."""
        emotional_words = self.EMOTIONAL_WORDS.get(lang, self.EMOTIONAL_WORDS['en'])

        best_para = None
        best_score = 0

        for para in paragraphs:
            para_lower = para.lower()
            score = sum(1 for word in emotional_words if word in para_lower)

            # Bonus pour les points d'exclamation
            score += para.count('!') * 0.5
            # Bonus pour les points de suspension
            score += para.count('...') * 0.3

            if score > best_score:
                best_score = score
                best_para = para

        return best_para if best_score > 0 else None

    def _smart_truncate(self, text: str, max_chars: int) -> str:
        """Tronque intelligemment a la fin d'une phrase."""
        if len(text) <= max_chars:
            return text

        # Chercher la fin de phrase la plus proche
        truncated = text[:max_chars]

        # Chercher le dernier point, point d'exclamation ou point d'interrogation
        last_sentence_end = max(
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?')
        )

        if last_sentence_end > max_chars * 0.5:
            return truncated[:last_sentence_end + 1]

        return truncated

    def generate_preview(
        self,
        text: str,
        output_path: Path,
        synthesize_fn,
        lang: str = 'fr',
        **synth_kwargs
    ) -> Tuple[bool, float]:
        """
        Genere un preview audio.

        Args:
            text: Texte complet
            output_path: Chemin de sortie
            synthesize_fn: Fonction de synthese (text, output_path) -> bool
            lang: Langue du texte
            **synth_kwargs: Arguments supplementaires pour la synthese

        Returns:
            Tuple (succes, duree_secondes)
        """
        # Extraire le texte du preview
        preview_text = self.extract_preview_text(text, lang)

        print(f"Preview: {len(preview_text)} caracteres extraits")
        print(f"Extrait: {preview_text[:100]}...")

        # Generer l'audio
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        success = synthesize_fn(preview_text, output_path, **synth_kwargs)

        if success and output_path.exists():
            import soundfile as sf
            audio, sr = sf.read(str(output_path))
            duration = len(audio) / sr
            return True, duration

        return False, 0.0


class QuickPreview:
    """
    Interface simplifiee pour generer des previews rapides.
    """

    def __init__(self, tts_engine=None):
        """
        Args:
            tts_engine: Moteur TTS a utiliser (optionnel)
        """
        self.tts_engine = tts_engine
        self.generator = PreviewGenerator()

    def set_engine(self, engine):
        """Definit le moteur TTS."""
        self.tts_engine = engine

    def generate(
        self,
        text: str,
        output_path: str = "output/preview.wav",
        duration: float = 30.0,
        voice: Optional[str] = None,
        lang: str = 'fr'
    ) -> Tuple[bool, str]:
        """
        Genere un preview rapide.

        Args:
            text: Texte source
            output_path: Chemin de sortie
            duration: Duree cible en secondes
            voice: Voix a utiliser
            lang: Langue

        Returns:
            Tuple (succes, message)
        """
        if self.tts_engine is None:
            return False, "Aucun moteur TTS configure"

        # Configurer la duree
        self.generator.config.target_duration = duration
        self.generator.config.target_chars = int(duration * 15)

        output_path = Path(output_path)

        # Fonction de synthese
        def synth_fn(text, path, **kwargs):
            if hasattr(self.tts_engine, 'synthesize_chapter'):
                return self.tts_engine.synthesize_chapter(text, path)
            elif hasattr(self.tts_engine, 'synthesize'):
                return self.tts_engine.synthesize(text, path)
            else:
                return False

        success, actual_duration = self.generator.generate_preview(
            text, output_path, synth_fn, lang
        )

        if success:
            return True, f"Preview genere: {output_path} ({actual_duration:.1f}s)"
        else:
            return False, "Echec de la generation"


def generate_quick_preview(
    text: str,
    output_path: str = "output/preview.wav",
    engine_type: str = "hybrid",
    voice: Optional[str] = None,
    lang: str = "fr",
    duration: float = 30.0
) -> Tuple[bool, str]:
    """
    Fonction utilitaire pour generer un preview rapidement.

    Args:
        text: Texte source
        output_path: Chemin de sortie
        engine_type: Type de moteur ("hybrid", "kokoro", "mms")
        voice: Voix a utiliser
        lang: Langue
        duration: Duree cible

    Returns:
        Tuple (succes, message)
    """
    # Creer le moteur selon le type
    if engine_type == "hybrid":
        from .tts_hybrid_engine import HybridTTSEngine
        lang_map = {"fr": "fra", "en": "eng", "de": "deu"}
        engine = HybridTTSEngine(mms_language=lang_map.get(lang, lang))
    elif engine_type == "kokoro":
        from .tts_kokoro_engine import KokoroTTSEngine
        engine = KokoroTTSEngine(voice=voice or "ff_siwis")
    elif engine_type == "mms":
        from .tts_mms_engine import MMSTTSEngine
        engine = MMSTTSEngine(language=lang)
    else:
        return False, f"Type de moteur inconnu: {engine_type}"

    preview = QuickPreview(engine)
    return preview.generate(text, output_path, duration, voice, lang)


if __name__ == "__main__":
    # Test
    test_text = """
    L'argent a une odeur. Celle de la sueur des autres.

    Victor le savait depuis son enfance a Saint-Denis, quand il observait
    les ouvriers de l'usine rentrer chez eux, le visage gris de fatigue.

    « Tu comprends, petit, » lui avait dit un jour son pere, « dans ce monde,
    y'a ceux qui comptent l'argent et ceux qui le gagnent. »

    Cette phrase l'avait marque a jamais. Soudain, il avait compris que
    la vie etait un jeu dont les regles etaient ecrites par les riches.

    Et il avait decide qu'un jour, c'est lui qui ecrirait les regles.
    """

    generator = PreviewGenerator()
    preview = generator.extract_preview_text(test_text, 'fr')

    print("=== PREVIEW EXTRAIT ===")
    print(preview)
    print(f"\nLongueur: {len(preview)} caracteres")
