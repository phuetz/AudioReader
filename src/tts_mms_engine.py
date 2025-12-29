"""
Moteur TTS utilisant MMS-TTS de Meta (Massively Multilingual Speech).

MMS-TTS supporte 1000+ langues avec une qualité native.
Pour le français: facebook/mms-tts-fra

Avantages:
- Qualité native française (entraîné sur données françaises)
- Pas besoin de preprocessing complexe
- Rapide sur CPU
- Open-source (CC BY-NC 4.0)

Usage:
    from src.tts_mms_engine import MMSTTSEngine

    engine = MMSTTSEngine(language="fra")
    engine.synthesize("Bonjour le monde!", "output.wav")
"""

import numpy as np
from pathlib import Path
from typing import Optional, List
import re


# Langues supportées par MMS-TTS (codes ISO 639-3)
MMS_LANGUAGES = {
    "français": "fra",
    "french": "fra",
    "fr": "fra",
    "anglais": "eng",
    "english": "eng",
    "en": "eng",
    "allemand": "deu",
    "german": "deu",
    "de": "deu",
    "espagnol": "spa",
    "spanish": "spa",
    "es": "spa",
    "italien": "ita",
    "italian": "ita",
    "it": "ita",
    "portugais": "por",
    "portuguese": "por",
    "pt": "por",
}


class MMSTTSEngine:
    """
    Moteur TTS basé sur MMS (Meta Multilingual Speech).

    Utilise les modèles facebook/mms-tts-* pour 1000+ langues.
    Recommandé pour le français car entraîné sur données natives.
    """

    def __init__(
        self,
        language: str = "fra",
        speed: float = 1.0,
        sentence_pause: float = 0.4,
        paragraph_pause: float = 0.8
    ):
        """
        Initialise le moteur MMS-TTS.

        Args:
            language: Code langue ISO 639-3 (fra, eng, deu, spa, etc.)
                      ou nom commun (français, french, fr)
            speed: Vitesse de lecture (non supporté actuellement)
            sentence_pause: Pause entre phrases (secondes)
            paragraph_pause: Pause entre paragraphes (secondes)
        """
        # Normaliser le code langue
        lang_lower = language.lower()
        self.language = MMS_LANGUAGES.get(lang_lower, lang_lower)

        self.model_name = f"facebook/mms-tts-{self.language}"
        self.speed = speed
        self.sentence_pause = sentence_pause
        self.paragraph_pause = paragraph_pause

        self._model = None
        self._tokenizer = None
        self._available = None
        self._sample_rate = 16000  # MMS génère à 16kHz

    @property
    def sample_rate(self) -> int:
        """Retourne le sample rate du modèle."""
        return self._sample_rate

    def is_available(self) -> bool:
        """Vérifie si MMS-TTS est disponible."""
        if self._available is not None:
            return self._available

        try:
            from transformers import VitsModel, AutoTokenizer
            import torch
            self._available = True
        except ImportError:
            self._available = False

        return self._available

    def _load_model(self):
        """Charge le modèle MMS-TTS."""
        if self._model is not None:
            return

        from transformers import VitsModel, AutoTokenizer
        import torch

        print(f"Chargement MMS-TTS ({self.language})...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = VitsModel.from_pretrained(self.model_name)
        self._sample_rate = self._model.config.sampling_rate

        # Utiliser GPU si disponible
        if torch.cuda.is_available():
            self._model = self._model.cuda()
            print("  (GPU activé)")

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: Optional[str] = None,  # Ignoré (MMS n'a qu'une voix par langue)
        speed: Optional[float] = None
    ) -> bool:
        """
        Synthétise du texte en audio.

        Args:
            text: Texte à synthétiser
            output_path: Chemin du fichier de sortie (.wav)
            voice: Ignoré (MMS n'a qu'une voix par langue)
            speed: Ignoré (non supporté par MMS)

        Returns:
            True si succès
        """
        import torch
        import soundfile as sf

        try:
            self._load_model()

            # Découper en segments
            segments = self._split_text(text)
            all_audio = []

            print(f"Génération de {len(segments)} segments...")

            for i, (segment, is_paragraph_end) in enumerate(segments):
                if not segment.strip():
                    continue

                # Tokenizer
                inputs = self._tokenizer(segment, return_tensors="pt")

                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                # Générer
                with torch.no_grad():
                    output = self._model(**inputs).waveform

                audio = output.squeeze().cpu().numpy()
                all_audio.append(audio)

                # Ajouter pause appropriée
                if is_paragraph_end:
                    pause_duration = self.paragraph_pause
                else:
                    pause_duration = self.sentence_pause

                pause = np.zeros(int(pause_duration * self._sample_rate))
                all_audio.append(pause)

            # Concaténer
            if not all_audio:
                print("Aucun audio généré")
                return False

            final_audio = np.concatenate(all_audio)

            # Normaliser le volume (éviter clipping)
            max_val = np.max(np.abs(final_audio))
            if max_val > 0:
                final_audio = (final_audio / max_val * 0.9).astype(np.float32)

            # Sauvegarder
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), final_audio, self._sample_rate)

            duration = len(final_audio) / self._sample_rate
            print(f"✅ Audio généré: {duration:.1f}s")

            return True

        except Exception as e:
            print(f"Erreur MMS-TTS: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _split_text(self, text: str, max_chars: int = 250) -> List[tuple]:
        """
        Découpe le texte en segments gérables.

        Returns:
            Liste de tuples (segment, is_paragraph_end)
        """
        # Découper par paragraphes d'abord
        paragraphs = text.split('\n\n')

        results = []

        for para_idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Découper par phrases
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)

            for sent_idx, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Si phrase trop longue, découper par virgules
                if len(sentence) > max_chars:
                    parts = sentence.split(', ')
                    current = ""
                    for part in parts:
                        if len(current) + len(part) < max_chars:
                            current += ", " + part if current else part
                        else:
                            if current:
                                results.append((current.strip(), False))
                            current = part
                    if current:
                        is_para_end = (sent_idx == len(sentences) - 1 and
                                      para_idx < len(paragraphs) - 1)
                        results.append((current.strip(), is_para_end))
                else:
                    is_para_end = (sent_idx == len(sentences) - 1 and
                                  para_idx < len(paragraphs) - 1)
                    results.append((sentence, is_para_end))

        return results

    def get_info(self) -> dict:
        """Retourne des informations sur le moteur."""
        return {
            "engine": "MMS-TTS",
            "provider": "Meta (Facebook)",
            "model": self.model_name,
            "language": self.language,
            "sample_rate": self._sample_rate,
            "supports_voice_cloning": False,
            "supports_emotions": False,
        }


def get_available_languages() -> dict:
    """Retourne les langues courantes supportées."""
    return {
        "fra": "Français",
        "eng": "English",
        "deu": "Deutsch",
        "spa": "Español",
        "ita": "Italiano",
        "por": "Português",
        "nld": "Nederlands",
        "pol": "Polski",
        "rus": "Русский",
        "jpn": "日本語",
        "cmn": "中文",
        "kor": "한국어",
    }


if __name__ == "__main__":
    # Test
    print("=== Test MMS-TTS Engine ===\n")

    engine = MMSTTSEngine(language="fra")

    if not engine.is_available():
        print("MMS-TTS non disponible")
        exit(1)

    print(f"Info: {engine.get_info()}\n")

    text = """Chapitre 1 : Le Début

    Il était une fois, dans un pays lointain, un jeune homme qui rêvait d'aventure.

    Chaque matin, il regardait par la fenêtre et imaginait les merveilles du monde."""

    output = Path("output/test_mms_engine.wav")
    success = engine.synthesize(text, output)

    if success:
        print(f"\nTest réussi: {output}")
