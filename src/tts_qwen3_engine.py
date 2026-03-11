"""
Moteur TTS Qwen3 (Alibaba Qwen) — Synthese vocale multilangue haute qualite.

Qwen3-TTS supporte 3 modes:
- CustomVoice: 9 voix pre-entrainees avec controle de style via instruct
- VoiceDesign: Creation de voix par description textuelle
- Base: Clonage vocal a partir de 3 secondes d'audio

Supporte 10 langues: chinois, anglais, francais, allemand, japonais,
coreen, russe, portugais, espagnol, italien.

Necessite: pip install -U qwen-tts torch
GPU recommande (CUDA) avec bfloat16/float16.
Optionnel: pip install -U flash-attn --no-build-isolation
"""
import re
import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict

logger = logging.getLogger(__name__)


@dataclass
class Qwen3Config:
    """Configuration du moteur Qwen3-TTS."""
    # Mode: "custom_voice", "voice_design", "base"
    mode: str = "custom_voice"
    # Modele par mode (overridable)
    model_name: Optional[str] = None
    # Taille du modele: "1.7B" ou "0.6B"
    model_size: str = "1.7B"
    default_language: str = "fr"
    use_gpu: bool = True
    # Precision: "bfloat16", "float16"
    dtype: str = "bfloat16"
    # FlashAttention 2 (reduit la memoire GPU)
    use_flash_attention: bool = True
    speed: float = 1.0
    sample_rate: int = 24000  # Target sample rate pour compatibilite AudioReader
    # Voix par defaut pour CustomVoice
    default_speaker: str = "Vivian"


# Modeles par mode et taille
MODEL_NAMES = {
    ("custom_voice", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    ("custom_voice", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    ("voice_design", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    ("base", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    ("base", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}

# Voix pre-entrainees (mode CustomVoice)
QWEN3_SPEAKERS = {
    "Vivian": {"lang": "zh", "gender": "female", "desc": "Chinese, bright, young female"},
    "Serena": {"lang": "zh", "gender": "female", "desc": "Chinese, warm, gentle female"},
    "Uncle_Fu": {"lang": "zh", "gender": "male", "desc": "Chinese, seasoned male"},
    "Dylan": {"lang": "zh", "gender": "male", "desc": "Beijing dialect male"},
    "Eric": {"lang": "zh", "gender": "male", "desc": "Sichuan dialect male"},
    "Ryan": {"lang": "en", "gender": "male", "desc": "English male"},
    "Aiden": {"lang": "en", "gender": "male", "desc": "American English male"},
    "Ono_Anna": {"lang": "ja", "gender": "female", "desc": "Japanese female"},
    "Sohee": {"lang": "ko", "gender": "female", "desc": "Korean female"},
}

# Mapping codes langue AudioReader -> noms Qwen3-TTS
LANG_MAP = {
    "fr": "French",
    "en": "English",
    "en-us": "English",
    "en-gb": "English",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
}

# Mapping emotions AudioReader -> instructions Qwen3
EMOTION_TO_INSTRUCT = {
    "joy": "Very happy and joyful.",
    "sadness": "Sad and melancholic.",
    "anger": "Angry and intense.",
    "fear": "Fearful and anxious.",
    "surprise": "Surprised and excited.",
    "tenderness": "Warm and tender.",
    "suspense": "Tense and suspenseful.",
    "irony": "Ironic and sarcastic.",
    "neutral": "",
}

# Mapping tags AudioReader -> instructions Qwen3
TAG_TO_INSTRUCT = {
    "[whispers]": "Whispering softly.",
    "[murmure]": "Whispering softly.",
    "[excited]": "Very excited and energetic.",
    "[enthousiaste]": "Very excited and energetic.",
    "[angry]": "Angry and forceful.",
    "[colere]": "Angry and forceful.",
    "[sad]": "Sad and subdued.",
    "[triste]": "Sad and subdued.",
    "[laugh]": "Laughing happily.",
    "[rire]": "Laughing happily.",
    "[sigh]": "Sighing deeply.",
    "[gasp]": "Gasping in shock.",
}


class Qwen3Engine:
    """
    Moteur TTS Qwen3 avec 3 modes: voix personnalisees, design vocal, clonage.

    Qwen3-TTS d'Alibaba offre:
    - 10 langues dont le francais
    - 9 voix pre-entrainees avec controle de style (mode CustomVoice)
    - Creation de voix par description (mode VoiceDesign)
    - Clonage vocal en 3 secondes (mode Base)
    - Streaming a 97ms de latence
    """

    def __init__(self, config: Optional[Qwen3Config] = None):
        self.config = config or Qwen3Config()
        self._model = None
        self._available = None
        self._cloned_voices: Dict[str, dict] = {}  # nom -> {audio_path, ref_text}
        self.sample_rate = self.config.sample_rate

    def is_available(self) -> bool:
        """Verifie si qwen-tts est installe."""
        if self._available is not None:
            return self._available

        try:
            import torch
            from qwen_tts import Qwen3TTSModel
            self._available = True
        except ImportError:
            self._available = False
            logger.info(
                "Qwen3-TTS non disponible. Installer avec: pip install -U qwen-tts"
            )

        return self._available

    def _get_model_name(self) -> str:
        """Determine le nom du modele a charger."""
        if self.config.model_name:
            return self.config.model_name
        key = (self.config.mode, self.config.model_size)
        if key not in MODEL_NAMES:
            raise ValueError(
                f"Combinaison mode={self.config.mode} / size={self.config.model_size} "
                f"non disponible. Options: {list(MODEL_NAMES.keys())}"
            )
        return MODEL_NAMES[key]

    def _load_model(self):
        """Charge le modele Qwen3-TTS (lazy loading)."""
        if self._model is not None:
            return self._model

        import torch
        from qwen_tts import Qwen3TTSModel

        model_name = self._get_model_name()

        # Determiner le device
        if self.config.use_gpu and torch.cuda.is_available():
            device_map = "cuda:0"
        else:
            device_map = "cpu"
            if self.config.use_gpu:
                logger.warning("CUDA non disponible, fallback vers CPU")

        # Precision
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        # FlashAttention 2
        kwargs = {}
        if self.config.use_flash_attention:
            try:
                import flash_attn  # noqa: F401
                kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                logger.info("flash-attn non installe, utilisation de l'attention standard")

        logger.info("Chargement de Qwen3-TTS (%s) sur %s...", model_name, device_map)
        print(f"Chargement de Qwen3-TTS ({model_name}) sur {device_map}...")

        self._model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=dtype,
            **kwargs,
        )

        logger.info("Qwen3-TTS charge avec succes (mode=%s)", self.config.mode)
        return self._model

    def _resolve_language(self, lang: str) -> str:
        """Convertit un code langue AudioReader en nom Qwen3."""
        lang_base = lang.split("-")[0].lower()
        return LANG_MAP.get(lang_base, LANG_MAP.get(lang.lower(), "French"))

    def _resolve_speaker(self, voice: Optional[str], gender: str = "female") -> str:
        """Selectionne un speaker pour le mode CustomVoice."""
        if voice and voice in QWEN3_SPEAKERS:
            return voice

        # Chercher par genre
        for name, info in QWEN3_SPEAKERS.items():
            if info["gender"] == gender:
                return name

        return self.config.default_speaker

    def _process_tags(self, text: str) -> Tuple[str, str]:
        """
        Extrait les tags AudioReader et construit une instruction Qwen3.

        Returns:
            Tuple (texte_nettoye, instruction)
        """
        instruct_parts = []

        for tag, instruct in TAG_TO_INSTRUCT.items():
            if tag in text:
                instruct_parts.append(instruct)
                text = text.replace(tag, "")

        # Nettoyer les tags restants
        text = re.sub(r'\[[a-z_]+(?::[0-9.]+)?\]', '', text)
        text = re.sub(r'\s{2,}', ' ', text).strip()

        instruct = " ".join(instruct_parts)
        return text, instruct

    def register_voice(self, name: str, audio_path: str, ref_text: str = ""):
        """
        Enregistre une voix de reference pour le clonage (mode Base).

        Args:
            name: Identifiant de la voix
            audio_path: Chemin vers un fichier audio (min 3 secondes)
            ref_text: Transcription du texte de reference
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier audio non trouve: {audio_path}")
        self._cloned_voices[name] = {
            "audio_path": str(path),
            "ref_text": ref_text,
        }
        logger.info("Voix '%s' enregistree: %s", name, audio_path)

    def synthesize(
        self,
        text: str,
        output_path=None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        emotion: Optional[str] = None,
        instruct: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        ref_text: Optional[str] = None,
        voice_description: Optional[str] = None,
        lang: str = "fr",
        gender: str = "female",
        **kwargs
    ) -> "bool | Tuple[np.ndarray, int]":
        """
        Synthetise du texte en audio.

        Args:
            text: Texte a synthetiser
            output_path: Chemin du fichier de sortie (si None, retourne array)
            voice: Nom de speaker (CustomVoice) ou voix clonee (Base)
            speed: Vitesse de lecture
            emotion: Emotion AudioReader (joy, sadness, anger, etc.)
            instruct: Instruction de style directe pour Qwen3
            speaker_wav: Fichier audio de reference pour clonage
            ref_text: Transcription de l'audio de reference
            voice_description: Description vocale pour mode VoiceDesign
            lang: Code langue
            gender: Genre pour selection automatique de voix

        Returns:
            Si output_path: bool (succes)
            Si pas output_path: Tuple (audio_array, sample_rate)
        """
        try:
            model = self._load_model()
            language = self._resolve_language(lang)

            # Traiter les tags
            text, tag_instruct = self._process_tags(text)
            if not text:
                empty = np.zeros(1, dtype=np.float32)
                return (empty, self.sample_rate) if output_path is None else True

            # Construire l'instruction finale
            final_instruct = self._build_instruct(instruct, emotion, tag_instruct)

            # Generer selon le mode
            if self.config.mode == "voice_design":
                wavs, sr = self._generate_voice_design(
                    model, text, language, voice_description or final_instruct
                )
            elif self.config.mode == "base":
                wavs, sr = self._generate_voice_clone(
                    model, text, language, voice, speaker_wav, ref_text
                )
            else:
                # custom_voice (default)
                speaker = self._resolve_speaker(voice, gender)
                gen_kwargs = {
                    "text": text,
                    "language": language,
                    "speaker": speaker,
                }
                if final_instruct:
                    gen_kwargs["instruct"] = final_instruct
                wavs, sr = model.generate_custom_voice(**gen_kwargs)

            # Convertir en numpy float32
            audio = self._wavs_to_numpy(wavs)

            # Resample si necessaire
            if sr != self.sample_rate:
                audio = self._resample(audio, sr, self.sample_rate)
                sr = self.sample_rate

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
            logger.error("Erreur Qwen3-TTS: %s", e)
            if output_path:
                return False
            raise

    def _generate_voice_design(self, model, text, language, description):
        """Genere via le mode VoiceDesign."""
        gen_kwargs = {
            "text": text,
            "language": language,
        }
        if description:
            gen_kwargs["instruct"] = description
        return model.generate_voice_design(**gen_kwargs)

    def _generate_voice_clone(self, model, text, language, voice, speaker_wav, ref_text):
        """Genere via le mode Base (clonage)."""
        # Resoudre la reference audio
        ref_audio = None
        ref_transcript = ref_text or ""

        if speaker_wav and Path(speaker_wav).exists():
            ref_audio = speaker_wav
        elif voice and voice in self._cloned_voices:
            clone_info = self._cloned_voices[voice]
            ref_audio = clone_info["audio_path"]
            ref_transcript = ref_transcript or clone_info["ref_text"]
        elif voice and Path(voice).exists():
            ref_audio = voice

        if not ref_audio:
            raise ValueError(
                "Mode base (clonage) necessite un audio de reference. "
                "Utilisez register_voice() ou speaker_wav=..."
            )

        return model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_transcript,
        )

    def _build_instruct(
        self,
        instruct: Optional[str],
        emotion: Optional[str],
        tag_instruct: str,
    ) -> str:
        """Construit l'instruction finale a partir des sources."""
        parts = []

        if instruct:
            parts.append(instruct)
        elif emotion and emotion in EMOTION_TO_INSTRUCT:
            emotion_instr = EMOTION_TO_INSTRUCT[emotion]
            if emotion_instr:
                parts.append(emotion_instr)

        if tag_instruct:
            parts.append(tag_instruct)

        return " ".join(parts)

    def _wavs_to_numpy(self, wavs) -> np.ndarray:
        """Convertit la sortie Qwen3 en numpy float32."""
        import torch

        if isinstance(wavs, list):
            wav = wavs[0]
        else:
            wav = wavs

        if isinstance(wav, torch.Tensor):
            audio = wav.squeeze().cpu().numpy()
        else:
            audio = np.array(wav, dtype=np.float32)

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normaliser [-1, 1]
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        return audio

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample l'audio vers le sample rate cible."""
        if orig_sr == target_sr:
            return audio
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

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
        emotion: Optional[str] = None,
        instruct: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        lang: str = "fr",
        gender: str = "female",
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
            emotion=emotion,
            instruct=instruct,
            speaker_wav=speaker_wav,
            lang=lang,
            gender=gender,
        )

    def synthesize_chapter(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        emotion: Optional[str] = None,
        instruct: Optional[str] = None,
        lang: str = "fr",
        gender: str = "female",
        max_chunk_chars: int = 500,
    ) -> bool:
        """
        Synthetise un chapitre complet avec decoupe automatique.

        Args:
            text: Texte du chapitre
            output_path: Chemin de sortie (.wav)
            voice: Speaker (CustomVoice) ou voix clonee (Base)
            speaker_wav: Fichier audio de reference directement
            emotion: Emotion a appliquer
            instruct: Instruction de style
            lang: Code langue
            gender: Genre pour selection auto de voix
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

                logger.info("Qwen3: chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))

                audio, sr = self.synthesize_array(
                    chunk,
                    voice=voice,
                    speaker_wav=speaker_wav,
                    emotion=emotion,
                    instruct=instruct,
                    lang=lang,
                    gender=gender,
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
            logger.error("Erreur synthese chapitre Qwen3: %s", e)
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
            "engine": "qwen3",
            "version": "1.0",
            "provider": "Alibaba Qwen",
            "mode": self.config.mode,
            "model_size": self.config.model_size,
            "model_name": self._get_model_name(),
            "supports_cloning": self.config.mode == "base",
            "supports_voice_design": self.config.mode == "voice_design",
            "supports_instruct": True,
            "available": self.is_available(),
            "cloned_voices": list(self._cloned_voices.keys()),
            "speakers": list(QWEN3_SPEAKERS.keys()) if self.config.mode == "custom_voice" else [],
            "languages": list(LANG_MAP.values()),
            "features": [
                "10_languages",
                "style_control_via_instruct",
                "streaming_97ms_latency",
                "3s_voice_cloning" if self.config.mode == "base" else "preset_voices",
                "flash_attention_2" if self.config.use_flash_attention else "standard_attention",
            ],
        }

    @staticmethod
    def list_speakers():
        """Affiche les speakers disponibles (mode CustomVoice)."""
        print("\n=== Speakers Qwen3-TTS (CustomVoice) ===\n")
        for name, info in QWEN3_SPEAKERS.items():
            print(f"  {name:15} ({info['gender']:6}) {info['lang']:4} - {info['desc']}")
