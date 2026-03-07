"""
Moteur TTS Parler (Hugging Face) — Synthese vocale multilangue haute qualite.

Parler-TTS est un modele leger qui genere de la parole naturelle a partir
d'une description textuelle du style vocal (genre, pitch, vitesse, etc.).

Avantages:
- Haute qualite pour le francais (multilingual v1.1)
- Controle fin du style via description textuelle
- Modele leger (~880M params)
- Pas besoin de sample audio pour le clonage

Necessite: pip install git+https://github.com/huggingface/parler-tts.git
GPU recommande (CUDA) pour performances optimales.
"""
import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ParlerConfig:
    """Configuration du moteur Parler TTS."""
    model_name: str = "parler-tts/parler-tts-mini-multilingual-v1.1"
    default_language: str = "fr"
    use_gpu: bool = True
    speed: float = 1.0
    sample_rate: int = 44100  # Parler sort en 44.1kHz
    target_sample_rate: int = 24000  # Resample pour compatibilite AudioReader


# Descriptions vocales predefinies pour le francais
FRENCH_VOICE_DESCRIPTIONS = {
    "narrator_female": (
        "A female speaker delivers a calm, clear narration with moderate speed "
        "and natural intonation. The recording is of very high quality, with "
        "the speaker's voice sounding warm and close up."
    ),
    "narrator_male": (
        "A male speaker delivers a steady, professional narration with moderate "
        "speed and a deep, resonant voice. The recording is of very high quality, "
        "with clear audio and close-up microphone placement."
    ),
    "expressive_female": (
        "A female speaker delivers a slightly expressive and animated speech "
        "with moderate speed and varied pitch. The recording is of very high "
        "quality, with the speaker's voice sounding clear and very close up."
    ),
    "expressive_male": (
        "A male speaker delivers an expressive and engaging speech with varied "
        "pace and emotional depth. The recording is of very high quality, "
        "with a warm, close-up sound."
    ),
    "soft_female": (
        "A female speaker delivers a soft, gentle speech with slow pace and "
        "low volume. The recording is of very high quality, with an intimate "
        "and close-up feel."
    ),
    "energetic_male": (
        "A male speaker delivers an energetic and fast-paced speech with high "
        "pitch variation and enthusiasm. The recording is of very high quality, "
        "with clear, close-up audio."
    ),
    "dramatic": (
        "A speaker delivers a dramatic and intense narration with slow, "
        "deliberate pacing and deep emotional resonance. The recording is "
        "of very high quality with a close-up, immersive feel."
    ),
    "whisper": (
        "A speaker delivers a very soft, whispery speech with slow pace "
        "and very low volume. The recording is of very high quality, "
        "extremely close to the microphone."
    ),
}

# Mapping emotion -> description override
EMOTION_DESCRIPTIONS = {
    "joy": "with a joyful, bright tone and slightly faster pace",
    "sadness": "with a melancholic, subdued tone and slower pace",
    "anger": "with an intense, forceful tone and faster pace",
    "fear": "with a trembling, anxious tone and uneven pace",
    "surprise": "with an excited, high-pitched tone",
    "tenderness": "with a warm, gentle and caring tone",
    "suspense": "with a tense, measured and deliberate tone",
}


class ParlerEngine:
    """
    Moteur TTS utilisant Parler-TTS (Hugging Face).

    Genere de la parole naturelle en francais a partir d'une description
    du style vocal. Pas besoin d'echantillon audio de reference.
    """

    def __init__(self, config: Optional[ParlerConfig] = None):
        self.config = config or ParlerConfig()
        self._model = None
        self._tokenizer = None
        self._description_tokenizer = None
        self._available = None
        self._device = None

    def is_available(self) -> bool:
        """Verifie si Parler TTS est disponible."""
        if self._available is not None:
            return self._available

        try:
            import torch
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
            self._available = True
        except ImportError:
            self._available = False

        return self._available

    def _load_model(self):
        """Charge le modele Parler TTS."""
        if self._model is not None:
            return

        if not self.is_available():
            raise RuntimeError(
                "Parler TTS non disponible. "
                "Installez: pip install git+https://github.com/huggingface/parler-tts.git"
            )

        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        # Determiner le device
        if self.config.use_gpu and torch.cuda.is_available():
            self._device = "cuda:0"
        else:
            self._device = "cpu"

        logger.info(f"Chargement de Parler TTS ({self.config.model_name}) sur {self._device}...")
        print(f"Chargement de Parler TTS sur {self._device}...")

        self._model = ParlerTTSForConditionalGeneration.from_pretrained(
            self.config.model_name
        ).to(self._device)

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._description_tokenizer = AutoTokenizer.from_pretrained(
            self._model.config.text_encoder._name_or_path
        )

        logger.info("Parler TTS charge avec succes")

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample l'audio si necessaire."""
        if orig_sr == target_sr:
            return audio

        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Fallback: interpolation lineaire
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def synthesize_array(
        self,
        text: str,
        voice_description: Optional[str] = None,
        emotion: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        """
        Synthetise du texte et retourne un array numpy.

        Args:
            text: Texte a synthetiser
            voice_description: Description du style vocal (ou nom d'un preset)
            emotion: Emotion a appliquer (joy, sadness, anger, etc.)

        Returns:
            Tuple (audio_array, sample_rate)
        """
        import torch

        self._load_model()

        # Construire la description
        description = self._build_description(voice_description, emotion)

        # Tokeniser
        input_ids = self._description_tokenizer(
            description, return_tensors="pt"
        ).input_ids.to(self._device)

        prompt_input_ids = self._tokenizer(
            text, return_tensors="pt"
        ).input_ids.to(self._device)

        # Generer
        with torch.no_grad():
            generation = self._model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids
            )

        # Convertir en numpy
        audio = generation.cpu().numpy().squeeze()
        if audio.ndim == 0:
            return np.array([], dtype=np.float32), self.config.target_sample_rate

        audio = audio.astype(np.float32)

        # Normaliser
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95

        # Resample vers le sample rate cible
        model_sr = self._model.config.sampling_rate
        audio = self._resample(audio, model_sr, self.config.target_sample_rate)

        return audio, self.config.target_sample_rate

    def synthesize(
        self,
        text: str,
        output_path: str,
        voice_description: Optional[str] = None,
        emotion: Optional[str] = None,
    ) -> bool:
        """
        Synthetise du texte et sauvegarde en fichier audio.

        Args:
            text: Texte a synthetiser
            output_path: Chemin du fichier de sortie
            voice_description: Description du style vocal
            emotion: Emotion a appliquer

        Returns:
            True si succes
        """
        try:
            import soundfile as sf

            audio, sr = self.synthesize_array(text, voice_description, emotion)

            if len(audio) == 0:
                logger.warning("Audio vide genere")
                return False

            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output), audio, sr)

            return True

        except Exception as e:
            logger.error(f"Erreur Parler TTS: {e}")
            return False

    def synthesize_chapter(
        self,
        text: str,
        output_path: str,
        voice_description: Optional[str] = None,
        emotion: Optional[str] = None,
        max_chunk_chars: int = 300,
    ) -> bool:
        """
        Synthetise un chapitre complet avec decoupage automatique.

        Args:
            text: Texte du chapitre
            output_path: Chemin de sortie
            voice_description: Description vocale
            emotion: Emotion
            max_chunk_chars: Taille max des chunks

        Returns:
            True si succes
        """
        try:
            import soundfile as sf

            chunks = self._split_text(text, max_chunk_chars)
            all_audio = []

            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue

                print(f"  [{i+1}/{len(chunks)}] {chunk[:50]}...")

                audio, sr = self.synthesize_array(chunk, voice_description, emotion)
                if len(audio) > 0:
                    all_audio.append(audio)
                    # Pause inter-phrase
                    pause = np.zeros(int(0.3 * sr), dtype=np.float32)
                    all_audio.append(pause)

            if not all_audio:
                return False

            final = np.concatenate(all_audio)

            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output), final, self.config.target_sample_rate)

            return True

        except Exception as e:
            logger.error(f"Erreur Parler chapter: {e}")
            return False

    def _build_description(
        self,
        voice_description: Optional[str] = None,
        emotion: Optional[str] = None,
    ) -> str:
        """Construit la description vocale pour Parler."""
        # Si c'est un nom de preset, utiliser la description correspondante
        if voice_description and voice_description in FRENCH_VOICE_DESCRIPTIONS:
            desc = FRENCH_VOICE_DESCRIPTIONS[voice_description]
        elif voice_description:
            desc = voice_description
        else:
            # Description par defaut
            desc = FRENCH_VOICE_DESCRIPTIONS["narrator_female"]

        # Ajouter l'emotion si specifiee
        if emotion and emotion in EMOTION_DESCRIPTIONS:
            emotion_suffix = EMOTION_DESCRIPTIONS[emotion]
            # Inserer avant le dernier point
            if desc.endswith("."):
                desc = desc[:-1] + f", {emotion_suffix}."
            else:
                desc = f"{desc}, {emotion_suffix}."

        return desc

    def _split_text(self, text: str, max_chars: int = 300) -> list[str]:
        """Decoupe le texte en chunks pour la synthese."""
        import re

        # Decouper sur les phrases
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) > max_chars and current:
                chunks.append(current.strip())
                current = sentence
            else:
                current = f"{current} {sentence}" if current else sentence

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def get_info(self) -> dict:
        """Retourne les informations sur le moteur."""
        return {
            "name": "Parler TTS",
            "model": self.config.model_name,
            "available": self.is_available(),
            "languages": ["fr", "en", "es", "pt", "pl", "de", "it", "nl"],
            "features": [
                "description-based voice control",
                "multilingual",
                "high quality French",
                "no reference audio needed",
            ],
            "presets": list(FRENCH_VOICE_DESCRIPTIONS.keys()),
        }

    @staticmethod
    def list_presets():
        """Affiche les presets vocaux disponibles."""
        print("\n=== Presets vocaux Parler TTS ===\n")
        for name, desc in FRENCH_VOICE_DESCRIPTIONS.items():
            print(f"  {name:20}: {desc[:80]}...")
