"""
Moteur TTS Chatterbox (Resemble AI) — SoTA open-source.

Bat ElevenLabs en tests aveugles (63.75% preference).
Supporte 3 variantes:
- Chatterbox original: anglais, controle d'emotion (exaggeration)
- Chatterbox Turbo: plus rapide, tags paralinguistiques ([laugh], [cough]...)
- Chatterbox Multilingual: 23 langues dont le francais

Necessite: pip install chatterbox-tts torch torchaudio
GPU recommande (CUDA) pour performances optimales.
"""
import re
import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ChatterboxConfig:
    """Configuration du moteur Chatterbox."""
    default_language: str = "fr"
    use_gpu: bool = True
    exaggeration: float = 0.5  # Intensite emotionnelle (0=monotone, 1=tres expressif)
    cfg_weight: float = 0.5  # Guidance (baisser a 0.3 si voix de reference rapide)
    speed: float = 1.0
    sample_rate: int = 24000
    use_turbo: bool = False  # Turbo = plus rapide, tags paralinguistiques natifs
    use_multilingual: bool = True  # Multilingual = 23 langues


# Mapping des tags AudioReader vers tags Chatterbox Turbo natifs
AUDIOREADER_TO_CHATTERBOX_TAGS = {
    "[laugh]": "[laugh]",
    "[rire]": "[laugh]",
    "[chuckle]": "[chuckle]",
    "[cough]": "[cough]",
    "[sigh]": "[sigh]",
    "[gasp]": "[gasp]",
}

# Tags qui ajustent l'exaggeration plutot que d'etre convertis
EMOTION_TAG_ADJUSTMENTS = {
    "[whispers]": -0.3,
    "[murmure]": -0.3,
    "[excited]": 0.3,
    "[enthousiaste]": 0.3,
    "[angry]": 0.4,
    "[colere]": 0.4,
    "[sad]": 0.2,
    "[triste]": 0.2,
}


class ChatterboxEngine:
    """
    Moteur TTS Chatterbox avec clonage zero-shot et controle d'emotion.

    Chatterbox de Resemble AI offre:
    - Clonage vocal avec quelques secondes de reference audio
    - Controle d'exageration emotionnelle unique (0=plat, 1=expressif)
    - Tags paralinguistiques natifs (mode Turbo)
    - Support de 23 langues (mode Multilingual)
    """

    def __init__(self, config: Optional[ChatterboxConfig] = None):
        self.config = config or ChatterboxConfig()
        self._model = None
        self._available = None
        self._cloned_voices = {}  # nom -> chemin fichier audio reference
        self.sample_rate = self.config.sample_rate

    def is_available(self) -> bool:
        """Verifie si chatterbox-tts est installe."""
        if self._available is not None:
            return self._available

        try:
            import torch
            from chatterbox.tts import ChatterboxTTS
            self._available = True
        except ImportError:
            self._available = False
            logger.info(
                "Chatterbox non disponible. Installer avec: pip install chatterbox-tts"
            )

        return self._available

    def _load_model(self):
        """Charge le modele Chatterbox (lazy loading)."""
        if self._model is not None:
            return self._model

        import torch

        device = "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
        if self.config.use_gpu and not torch.cuda.is_available():
            logger.warning("CUDA non disponible, fallback vers CPU")

        if self.config.use_turbo:
            try:
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                self._model = ChatterboxTurboTTS.from_pretrained(device=device)
                logger.info("Chatterbox Turbo charge sur %s", device)
            except ImportError:
                from chatterbox.tts import ChatterboxTTS
                self._model = ChatterboxTTS.from_pretrained(device=device)
                logger.info("Chatterbox (fallback standard) charge sur %s", device)
        elif self.config.use_multilingual:
            try:
                from chatterbox.tts_multilingual import ChatterboxMultilingualTTS
                self._model = ChatterboxMultilingualTTS.from_pretrained(device=device)
                logger.info("Chatterbox Multilingual charge sur %s", device)
            except ImportError:
                from chatterbox.tts import ChatterboxTTS
                self._model = ChatterboxTTS.from_pretrained(device=device)
                logger.info("Chatterbox (fallback standard) charge sur %s", device)
        else:
            from chatterbox.tts import ChatterboxTTS
            self._model = ChatterboxTTS.from_pretrained(device=device)
            logger.info("Chatterbox charge sur %s", device)

        return self._model

    def register_voice(self, name: str, audio_path: str):
        """
        Enregistre une voix de reference pour le clonage.

        Args:
            name: Identifiant de la voix
            audio_path: Chemin vers un fichier audio (min 6 secondes recommande)
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier audio non trouve: {audio_path}")
        self._cloned_voices[name] = str(path)
        logger.info("Voix '%s' enregistree: %s", name, audio_path)

    def _process_tags(self, text: str) -> Tuple[str, float]:
        """
        Convertit les tags AudioReader en format Chatterbox.

        Returns:
            Tuple (texte_modifie, ajustement_exaggeration)
        """
        exag_adjust = 0.0

        # Convertir les tags paralinguistiques (mode Turbo uniquement)
        if self.config.use_turbo:
            for tag, replacement in AUDIOREADER_TO_CHATTERBOX_TAGS.items():
                text = text.replace(tag, replacement)

        # Calculer ajustement d'exaggeration selon les tags emotionnels
        for tag, adjust in EMOTION_TAG_ADJUSTMENTS.items():
            if tag in text:
                exag_adjust = adjust
                text = text.replace(tag, "")

        # Nettoyer les tags non supportes restants
        text = re.sub(r'\[[a-z_]+(?::[0-9.]+)?\]', '', text)
        text = re.sub(r'\s{2,}', ' ', text).strip()

        return text, exag_adjust

    def synthesize(
        self,
        text: str,
        output_path=None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        speaker_wav: Optional[str] = None,
        lang: str = "fr",
        **kwargs
    ) -> "bool | Tuple[np.ndarray, int]":
        """
        Synthetise du texte en audio.

        Args:
            text: Texte a synthetiser
            output_path: Chemin du fichier de sortie (si None, retourne array)
            voice: Nom de voix clonee enregistree
            speed: Vitesse de lecture
            exaggeration: Controle d'exageration emotionnelle (0.0-1.0)
            cfg_weight: Poids de guidance
            speaker_wav: Fichier audio de reference pour le clonage
            lang: Langue du texte

        Returns:
            Si output_path: bool (succes)
            Si pas output_path: Tuple (audio_array, sample_rate)
        """
        try:
            import torch

            model = self._load_model()

            # Traiter les tags
            text, exag_adjust = self._process_tags(text)
            if not text:
                empty = np.zeros(1, dtype=np.float32)
                return (empty, self.sample_rate) if output_path is None else True

            spd = speed or self.config.speed
            exag = exaggeration if exaggeration is not None else self.config.exaggeration
            exag = max(0.0, min(1.0, exag + exag_adjust))
            cfg = cfg_weight if cfg_weight is not None else self.config.cfg_weight

            # Resoudre la reference vocale
            audio_prompt_path = None
            if speaker_wav and Path(speaker_wav).exists():
                audio_prompt_path = speaker_wav
            elif voice and voice in self._cloned_voices:
                audio_prompt_path = self._cloned_voices[voice]
            elif voice and Path(voice).exists():
                audio_prompt_path = voice

            # Construire les kwargs de generation
            gen_kwargs = {"text": text}
            if audio_prompt_path:
                gen_kwargs["audio_prompt_path"] = audio_prompt_path

            # Parametres selon le type de modele
            if not self.config.use_turbo:
                gen_kwargs["exaggeration"] = exag
                gen_kwargs["cfg_weight"] = cfg

            wav = model.generate(**gen_kwargs)

            # Convertir en numpy
            if isinstance(wav, torch.Tensor):
                audio = wav.squeeze().cpu().numpy()
            else:
                audio = np.array(wav, dtype=np.float32)

            # Normaliser en float32 [-1, 1]
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio = audio / max_val

            # Appliquer la vitesse si != 1.0
            if abs(spd - 1.0) > 0.01:
                try:
                    import librosa
                    audio = librosa.effects.time_stretch(audio, rate=spd)
                except ImportError:
                    logger.warning("librosa non disponible, vitesse non appliquee")

            # Sauvegarder ou retourner
            sr = getattr(model, 'sr', self.sample_rate)
            if output_path:
                import soundfile as sf
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(output_path), audio, sr)
                return True
            else:
                return audio, sr

        except Exception as e:
            logger.error("Erreur Chatterbox: %s", e)
            if output_path:
                return False
            raise

    def synthesize_array(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        exaggeration: Optional[float] = None,
        speaker_wav: Optional[str] = None,
        lang: str = "fr",
    ) -> Tuple[np.ndarray, int]:
        """
        Synthetise et retourne un array numpy (compatible avec UnifiedTTS).

        Returns:
            Tuple (audio_array float32 [-1,1], sample_rate)
        """
        result = self.synthesize(
            text=text,
            output_path=None,
            voice=voice,
            speed=speed,
            exaggeration=exaggeration,
            speaker_wav=speaker_wav,
            lang=lang,
        )
        return result

    def synthesize_chapter(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        exaggeration: Optional[float] = None,
        max_chunk_chars: int = 500,
    ) -> bool:
        """
        Synthetise un chapitre complet avec decoupe automatique.

        Args:
            text: Texte du chapitre
            output_path: Chemin de sortie (.wav)
            voice: Reference vocale pour le clonage
            speaker_wav: Fichier audio de reference directement
            exaggeration: Intensite emotionnelle
            max_chunk_chars: Taille max des chunks de texte

        Returns:
            True si succes
        """
        try:
            import soundfile as sf

            chunks = self._split_text(text, max_chunk_chars)
            all_audio = []
            sr = self.sample_rate

            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue

                logger.info("Chatterbox: chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))
                audio, sr = self.synthesize_array(
                    chunk,
                    voice=voice,
                    speaker_wav=speaker_wav,
                    exaggeration=exaggeration,
                )
                all_audio.append(audio)

                # Pause inter-phrase (50ms de silence)
                pause = np.zeros(int(sr * 0.05), dtype=np.float32)
                all_audio.append(pause)

            if not all_audio:
                return False

            final_audio = np.concatenate(all_audio)

            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output), final_audio, sr)

            logger.info("Chapitre sauvegarde: %s (%.1fs)", output_path, len(final_audio) / sr)
            return True

        except Exception as e:
            logger.error("Erreur synthese chapitre: %s", e)
            return False

    @staticmethod
    def _split_text(text: str, max_chars: int = 500) -> List[str]:
        """Decoupe le texte en chunks aux frontieres de phrases."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) + 1 > max_chars and current:
                chunks.append(current.strip())
                current = sentence
            else:
                current = current + " " + sentence if current else sentence

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def get_info(self) -> dict:
        """Retourne les informations sur le moteur."""
        return {
            "engine": "chatterbox",
            "version": "2.0",
            "provider": "Resemble AI",
            "supports_cloning": True,
            "supports_exaggeration": True,
            "turbo": self.config.use_turbo,
            "multilingual": self.config.use_multilingual,
            "available": self.is_available(),
            "cloned_voices": list(self._cloned_voices.keys()),
            "features": [
                "voice_cloning",
                "emotion_control",
                "paralinguistic_tags" if self.config.use_turbo else "exaggeration_control",
                "multilingual_23_languages" if self.config.use_multilingual else "english_only",
            ],
        }
