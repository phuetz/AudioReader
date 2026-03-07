"""
Moteur TTS Orpheus (Canopy Labs) — Speech emotionnel basee sur Llama.

Orpheus TTS genere de la parole avec emotion et intonation naturelles,
superieure a beaucoup de modeles closed-source. Utilise SNAC audio codec.

Tags emotionnels natifs: <laugh>, <chuckle>, <sigh>, <cough>,
                         <sniffle>, <groan>, <yawn>, <gasp>

Necessite: pip install orpheus-speech (ou vllm + snac + transformers)
GPU CUDA requis.
"""
import re
import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class OrpheusConfig:
    """Configuration du moteur Orpheus TTS."""
    model_name: str = "canopylabs/orpheus-tts-0.1-finetune-prod"
    max_tokens: int = 2048
    default_voice: str = "tara"  # Voix par defaut parmi les 8 disponibles
    sample_rate: int = 24000
    use_streaming: bool = False  # Streaming pour latence reduite


# Voix Orpheus disponibles
ORPHEUS_VOICES = {
    "tara": {"gender": "F", "desc": "Female, warm and natural"},
    "leah": {"gender": "F", "desc": "Female, clear and bright"},
    "jess": {"gender": "F", "desc": "Female, young and dynamic"},
    "leo": {"gender": "M", "desc": "Male, warm and deep"},
    "dan": {"gender": "M", "desc": "Male, clear and professional"},
    "mia": {"gender": "F", "desc": "Female, soft and intimate"},
    "zac": {"gender": "M", "desc": "Male, young and energetic"},
    "zoe": {"gender": "F", "desc": "Female, mature and authoritative"},
}

# Mapping des tags AudioReader vers tags Orpheus natifs
AUDIOREADER_TO_ORPHEUS_TAGS = {
    "[laugh]": "<laugh>",
    "[rire]": "<laugh>",
    "[chuckle]": "<chuckle>",
    "[sigh]": "<sigh>",
    "[gasp]": "<gasp>",
    "[cough]": "<cough>",
    "[yawn]": "<yawn>",
    "[groan]": "<groan>",
    "[sniffle]": "<sniffle>",
    # Tags sans equivalent direct - supprimes
    "[whispers]": "",
    "[murmure]": "",
    "[excited]": "",
    "[enthousiaste]": "",
    "[pause]": " ... ",
}


class OrpheusEngine:
    """
    Moteur TTS Orpheus avec emotion naturelle.

    Orpheus est base sur Llama-3B et genere du speech avec:
    - Intonation et emotion naturelles
    - Tags emotionnels inline (<laugh>, <sigh>, etc.)
    - 8 voix pre-entrainées
    - Streaming a ~200ms de latence
    """

    def __init__(self, config: Optional[OrpheusConfig] = None):
        self.config = config or OrpheusConfig()
        self._model = None
        self._available = None
        self.sample_rate = self.config.sample_rate

    def is_available(self) -> bool:
        """Verifie si orpheus-speech est installe."""
        if self._available is not None:
            return self._available

        try:
            from orpheus_tts import OrpheusModel
            self._available = True
        except ImportError:
            self._available = False
            logger.info(
                "Orpheus non disponible. Installer avec: pip install orpheus-speech"
            )

        return self._available

    def _load_model(self):
        """Charge le modele Orpheus (lazy loading)."""
        if self._model is not None:
            return self._model

        from orpheus_tts import OrpheusModel
        self._model = OrpheusModel(
            model_name=self.config.model_name,
            max_model_len=self.config.max_tokens,
        )
        logger.info("Orpheus TTS charge: %s", self.config.model_name)
        return self._model

    def _convert_tags(self, text: str) -> str:
        """Convertit les tags AudioReader en tags Orpheus natifs."""
        for tag, replacement in AUDIOREADER_TO_ORPHEUS_TAGS.items():
            text = text.replace(tag, replacement)

        # Nettoyer les tags restants non supportes
        text = re.sub(r'\[[a-z_]+(?::[0-9.]+)?\]', '', text)
        text = re.sub(r'\s{2,}', ' ', text).strip()
        return text

    def _select_voice(self, gender: str = "female", voice: Optional[str] = None) -> str:
        """Selectionne une voix Orpheus."""
        if voice and voice in ORPHEUS_VOICES:
            return voice

        # Selectionner par genre
        target_g = "F" if gender.lower() in ("female", "f") else "M"
        for name, info in ORPHEUS_VOICES.items():
            if info["gender"] == target_g:
                return name

        return self.config.default_voice

    def synthesize(
        self,
        text: str,
        output_path=None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        lang: str = "en",
        **kwargs,
    ) -> "bool | Tuple[np.ndarray, int]":
        """
        Synthetise du texte en audio.

        Args:
            text: Texte a synthetiser (peut contenir des tags emotionnels)
            output_path: Chemin de sortie (si None, retourne array)
            voice: Nom de la voix Orpheus
            speed: Vitesse de lecture
            lang: Langue (Orpheus = anglais uniquement)

        Returns:
            Si output_path: bool (succes)
            Si pas output_path: Tuple (audio_array, sample_rate)
        """
        try:
            model = self._load_model()

            # Convertir les tags
            text = self._convert_tags(text)
            if not text:
                empty = np.zeros(1, dtype=np.float32)
                return (empty, self.sample_rate) if output_path is None else True

            # Selectionner la voix
            voice_name = self._select_voice(voice=voice)

            # Generer l'audio
            if self.config.use_streaming:
                audio_chunks = model.generate_speech(
                    prompt=text,
                    voice=voice_name,
                )
                # Collecter les chunks de streaming
                all_audio = []
                for chunk in audio_chunks:
                    all_audio.append(np.array(chunk, dtype=np.float32))
                audio = np.concatenate(all_audio) if all_audio else np.zeros(1, dtype=np.float32)
            else:
                result = model.generate_speech(
                    prompt=text,
                    voice=voice_name,
                )
                # Le resultat peut etre un generateur ou un array
                if hasattr(result, '__iter__') and not isinstance(result, np.ndarray):
                    all_audio = []
                    for chunk in result:
                        all_audio.append(np.array(chunk, dtype=np.float32))
                    audio = np.concatenate(all_audio) if all_audio else np.zeros(1, dtype=np.float32)
                else:
                    audio = np.array(result, dtype=np.float32)

            # Normaliser
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            max_val = np.abs(audio).max()
            if max_val > 0 and max_val > 1.0:
                audio = audio / max_val

            # Appliquer la vitesse
            spd = speed or 1.0
            if abs(spd - 1.0) > 0.01:
                try:
                    import librosa
                    audio = librosa.effects.time_stretch(audio, rate=spd)
                except ImportError:
                    logger.warning("librosa non disponible, vitesse non appliquee")

            sr = self.sample_rate

            if output_path:
                import soundfile as sf
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(output_path), audio, sr)
                return True
            else:
                return audio, sr

        except Exception as e:
            logger.error("Erreur Orpheus: %s", e)
            if output_path:
                return False
            raise

    def synthesize_array(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        lang: str = "en",
    ) -> Tuple[np.ndarray, int]:
        """Synthetise et retourne un array (compatible UnifiedTTS)."""
        return self.synthesize(text=text, output_path=None, voice=voice, speed=speed, lang=lang)

    def get_voices(self) -> dict:
        """Retourne les voix disponibles."""
        return ORPHEUS_VOICES.copy()

    def get_info(self) -> dict:
        """Retourne les informations sur le moteur."""
        return {
            "engine": "orpheus",
            "version": "3b-0.1",
            "provider": "Canopy Labs",
            "available": self.is_available(),
            "supports_cloning": False,
            "supports_emotion_tags": True,
            "supported_tags": ["<laugh>", "<chuckle>", "<sigh>", "<cough>",
                               "<sniffle>", "<groan>", "<yawn>", "<gasp>"],
            "voices": list(ORPHEUS_VOICES.keys()),
            "language": "en",
        }
