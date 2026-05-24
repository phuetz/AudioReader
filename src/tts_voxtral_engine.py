"""
Moteur TTS Voxtral (Mistral AI) — Synthese vocale multilangue via API.

Voxtral supporte 2 modes de deploiement:
- Cloud: API Mistral hebergee (necessite MISTRAL_API_KEY)
- Local: Serveur vLLM auto-heberge (necessite vllm + vllm-omni)

Fonctionnalites:
- 9 langues: anglais, francais, allemand, espagnol, italien, portugais,
  neerlandais, hindi, arabe
- Clonage vocal zero-shot a partir de 2-3 secondes d'audio
- 20 voix preconfigurees
- Streaming a ~90ms de latence (mode cloud)
- Style vocal via "voice-as-instruction" (reference audio)

Cloud: pip install mistralai
Local: pip install httpx soundfile
"""
import io
import os
import re
import base64
import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict

logger = logging.getLogger(__name__)


@dataclass
class VoxtralConfig:
    """Configuration du moteur Voxtral."""
    # Mode de deploiement: "cloud" (API Mistral) ou "local" (vLLM)
    mode: str = "cloud"
    # Modele cloud
    model: str = "voxtral-mini-tts-2603"
    # Modele local (vLLM)
    local_model: str = "mistralai/Voxtral-4B-TTS-2603"
    # Cle API Mistral (cloud) — auto-lue depuis MISTRAL_API_KEY
    api_key: Optional[str] = None
    # URL de base du serveur vLLM (local)
    base_url: str = "http://localhost:8000"
    default_language: str = "fr"
    # Format audio de sortie: wav, mp3, pcm, flac, opus
    response_format: str = "wav"
    # Voix par defaut
    default_voice: str = "casual_male"
    speed: float = 1.0
    sample_rate: int = 24000  # Voxtral sort nativement en 24kHz
    # Timeout requete en secondes
    timeout: float = 120.0

    def __post_init__(self):
        # Auto-detection cle API
        if self.api_key is None:
            self.api_key = os.environ.get("MISTRAL_API_KEY")
        # Auto-detection URL locale
        env_url = os.environ.get("VOXTRAL_BASE_URL")
        if env_url:
            self.base_url = env_url
            if self.mode == "cloud":
                self.mode = "local"


# Mapping codes langue AudioReader -> codes Voxtral
LANG_MAP = {
    "fr": "fr",
    "en": "en",
    "en-us": "en",
    "en-gb": "en",
    "de": "de",
    "es": "es",
    "it": "it",
    "pt": "pt",
    "nl": "nl",
    "hi": "hi",
    "ar": "ar",
}

# Voix preconfigurees connues
VOXTRAL_VOICES = {
    "casual_male": {"gender": "male", "desc": "Casual male voice"},
}


class VoxtralEngine:
    """
    Moteur TTS Voxtral (Mistral AI) avec support cloud et local.

    Voxtral offre:
    - 9 langues dont le francais
    - Clonage vocal zero-shot en 2-3 secondes d'audio
    - 20 voix preconfigurees
    - Mode cloud (API Mistral) ou local (serveur vLLM)
    - Controle de style via "voice-as-instruction" (reference audio)
    """

    def __init__(self, config: Optional[VoxtralConfig] = None):
        self.config = config or VoxtralConfig()
        self._client = None
        self._available = None
        self._cloned_voices: Dict[str, str] = {}  # nom -> audio base64
        self.sample_rate = self.config.sample_rate

    def is_available(self) -> bool:
        """Verifie si le moteur est disponible."""
        if self._available is not None:
            return self._available

        if self.config.mode == "cloud":
            try:
                from mistralai import Mistral  # noqa: F401
                self._available = bool(self.config.api_key)
                if not self._available:
                    logger.info(
                        "Voxtral cloud: MISTRAL_API_KEY non definie. "
                        "Definir la variable d'environnement ou passer api_key dans VoxtralConfig."
                    )
            except ImportError:
                self._available = False
                logger.info(
                    "Voxtral cloud non disponible. Installer avec: pip install mistralai"
                )
        else:
            # Local: verifier httpx
            try:
                import httpx  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
                logger.info(
                    "Voxtral local non disponible. Installer avec: pip install httpx"
                )

        return self._available

    def _load_client(self):
        """Initialise le client API (lazy loading)."""
        if self._client is not None:
            return self._client

        if self.config.mode == "cloud":
            from mistralai import Mistral
            self._client = Mistral(api_key=self.config.api_key)
            logger.info("Client Mistral API initialise (cloud)")
        else:
            import httpx
            self._client = httpx.Client(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
            logger.info("Client HTTP initialise (local: %s)", self.config.base_url)

        return self._client

    def _resolve_language(self, lang: str) -> str:
        """Convertit un code langue AudioReader en code Voxtral."""
        lang_base = lang.split("-")[0].lower()
        return LANG_MAP.get(lang_base, LANG_MAP.get(lang.lower(), "fr"))

    def _process_tags(self, text: str) -> str:
        """
        Nettoie les tags AudioReader du texte.

        Voxtral utilise "voice-as-instruction" (reference audio) pour le style,
        pas des tags textuels. Tous les tags sont supprimes.
        """
        # Supprimer tous les tags AudioReader
        text = re.sub(r'\[[a-z_]+(?::[0-9.]+)?\]', '', text)
        text = re.sub(r'\s{2,}', ' ', text).strip()
        return text

    def _audio_to_base64(self, audio_path: str) -> str:
        """Encode un fichier audio en base64."""
        return base64.b64encode(Path(audio_path).read_bytes()).decode()

    def register_voice(self, name: str, audio_path: str, ref_text: str = ""):
        """
        Enregistre une voix de reference pour le clonage.

        Args:
            name: Identifiant de la voix
            audio_path: Chemin vers un fichier audio (min 2-3 secondes)
            ref_text: Non utilise par Voxtral (conserve pour compatibilite)
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier audio non trouve: {audio_path}")
        self._cloned_voices[name] = self._audio_to_base64(audio_path)
        logger.info("Voix '%s' enregistree: %s", name, audio_path)

    def synthesize(
        self,
        text: str,
        output_path=None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        speaker_wav: Optional[str] = None,
        lang: str = "fr",
        **kwargs
    ) -> "bool | Tuple[np.ndarray, int]":
        """
        Synthetise du texte en audio via l'API Voxtral.

        Args:
            text: Texte a synthetiser
            output_path: Chemin du fichier de sortie (si None, retourne array)
            voice: Nom de voix preconfiguree ou voix clonee enregistree
            speed: Vitesse de lecture
            speaker_wav: Fichier audio de reference pour clonage direct
            lang: Code langue

        Returns:
            Si output_path: bool (succes)
            Si pas output_path: Tuple (audio_array, sample_rate)
        """
        try:
            # Nettoyer les tags
            text = self._process_tags(text)
            if not text:
                empty = np.zeros(1, dtype=np.float32)
                return (empty, self.sample_rate) if output_path is None else True

            # Resoudre la reference audio pour le clonage
            ref_audio_b64 = None
            if speaker_wav and Path(speaker_wav).exists():
                ref_audio_b64 = self._audio_to_base64(speaker_wav)
            elif voice and voice in self._cloned_voices:
                ref_audio_b64 = self._cloned_voices[voice]
            elif voice and Path(voice).exists():
                ref_audio_b64 = self._audio_to_base64(voice)

            # Determiner la voix preset (si pas de clonage)
            voice_id = None
            if not ref_audio_b64:
                voice_id = voice if voice else self.config.default_voice

            # Synthetiser selon le mode
            if self.config.mode == "cloud":
                audio_bytes = self._synthesize_cloud(text, voice_id, ref_audio_b64, lang)
            else:
                audio_bytes = self._synthesize_local(text, voice_id, ref_audio_b64, lang)

            # Convertir en numpy
            audio, sr = self._bytes_to_numpy(audio_bytes)

            # Appliquer la vitesse
            spd = speed or self.config.speed
            if abs(spd - 1.0) > 0.01:
                audio = self._apply_speed(audio, spd)

            # Sauvegarder ou retourner
            if output_path:
                import soundfile as sf
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(output_path), audio, sr)
                return True
            else:
                return audio, sr

        except Exception as e:
            logger.error("Erreur Voxtral: %s", e)
            if output_path:
                return False
            raise

    def _synthesize_cloud(
        self,
        text: str,
        voice_id: Optional[str],
        ref_audio_b64: Optional[str],
        lang: str,
    ) -> bytes:
        """Synthetise via l'API Mistral cloud."""
        client = self._load_client()
        language = self._resolve_language(lang)

        gen_kwargs = {
            "model": self.config.model,
            "input": text,
            "response_format": self.config.response_format,
        }

        if ref_audio_b64:
            gen_kwargs["ref_audio"] = ref_audio_b64
        elif voice_id:
            gen_kwargs["voice_id"] = voice_id

        response = client.audio.speech.complete(**gen_kwargs)
        return base64.b64decode(response.audio_data)

    def _synthesize_local(
        self,
        text: str,
        voice_id: Optional[str],
        ref_audio_b64: Optional[str],
        lang: str,
    ) -> bytes:
        """Synthetise via un serveur vLLM local."""
        client = self._load_client()

        payload = {
            "input": text,
            "model": self.config.local_model,
            "response_format": self.config.response_format,
            "voice": voice_id or self.config.default_voice,
        }

        if ref_audio_b64:
            payload["ref_audio"] = ref_audio_b64

        response = client.post("/v1/audio/speech", json=payload)
        response.raise_for_status()
        return response.content

    def _bytes_to_numpy(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Convertit des bytes audio en array numpy float32."""
        import soundfile as sf

        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        # Normaliser [-1, 1]
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        return audio, sr

    @staticmethod
    def _apply_speed(audio: np.ndarray, speed: float) -> np.ndarray:
        """Applique un changement de vitesse."""
        try:
            import librosa
            return librosa.effects.time_stretch(audio, rate=speed)
        except ImportError:
            logger.warning("librosa non disponible, vitesse non appliquee")
            return audio

    def synthesize_array(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        speaker_wav: Optional[str] = None,
        lang: str = "fr",
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        Synthetise et retourne un array numpy (compatible avec UnifiedTTS).

        Returns:
            Tuple (audio_array float32 [-1,1], sample_rate)
        """
        return self.synthesize(
            text=text,
            output_path=None,
            voice=voice,
            speed=speed,
            speaker_wav=speaker_wav,
            lang=lang,
        )

    def synthesize_chapter(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        lang: str = "fr",
        max_chunk_chars: int = 500,
    ) -> bool:
        """
        Synthetise un chapitre complet avec decoupe automatique.

        Args:
            text: Texte du chapitre
            output_path: Chemin de sortie (.wav)
            voice: Voix preconfiguree ou voix clonee
            speaker_wav: Fichier audio de reference directement
            lang: Code langue
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

                logger.info("Voxtral: chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))

                audio, sr = self.synthesize_array(
                    chunk,
                    voice=voice,
                    speaker_wav=speaker_wav,
                    lang=lang,
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
            logger.error("Erreur synthese chapitre Voxtral: %s", e)
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
            "engine": "voxtral",
            "version": "1.0",
            "provider": "Mistral AI",
            "mode": self.config.mode,
            "model": self.config.model if self.config.mode == "cloud" else self.config.local_model,
            "supports_cloning": True,
            "supports_streaming": self.config.mode == "cloud",
            "available": self.is_available(),
            "cloned_voices": list(self._cloned_voices.keys()),
            "languages": list(set(LANG_MAP.values())),
            "features": [
                "9_languages",
                "voice_cloning_2s",
                "voice_as_instruction",
                "cloud_api" if self.config.mode == "cloud" else "local_vllm",
                "20_preset_voices",
            ],
        }
