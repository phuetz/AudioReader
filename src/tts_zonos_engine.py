"""
Moteur TTS Zonos (Zyphra) - Synthèse vocale expressive et clonage.

Zonos offre:
- Contrôle de la prosodie et des émotions fine-grained.
- Clonage de voix de haute qualité à partir d'échantillons courts (5-30s).
- Licence Apache 2.0 (commercial-friendly).

Prérequis:
    pip install zonos torch torchaudio
    Et installer 'espeak-ng' sur le système.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np


@dataclass
class ZonosConfig:
    """Configuration du moteur Zonos."""
    model_name: str = "Zyphra/Zonos-v0.1-hybrid" # ou "Zyphra/Zonos-v0.1-transformer"
    use_gpu: bool = True
    speed: float = 1.0
    language: str = "fr"
    sample_rate: int = 44100

class ZonosEngine:
    """
    Moteur de synthèse vocale Zonos par Zyphra.
    """
    def __init__(self, config: Optional[ZonosConfig] = None):
        self.config = config or ZonosConfig()
        self._model = None
        self._device = None
        self._cloned_voices: Dict[str, str] = {} # voice_id -> speaker_embedding / audio_path
        self.sample_rate = self.config.sample_rate

    @staticmethod
    def is_available() -> bool:
        """Vérifie si la bibliothèque zonos est installée."""
        try:
            import torch
            import zonos
            return True
        except ImportError:
            return False

    def _load_model(self):
        """Charge le modèle Zonos en mémoire."""
        if self._model is not None:
            return self._model

        try:
            import torch
            from zonos.model import Zonos

            # Déterminer le device (CUDA requis idéalement pour 1.6B)
            if self.config.use_gpu and torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"

            print(f"Chargement du modèle Zonos ({self.config.model_name}) sur {self._device}...")
            self._model = Zonos.from_pretrained(self.config.model_name, device=self._device)
            print("Modèle Zonos chargé avec succès.")
            return self._model
        except ImportError:
            raise ImportError(
                "Zonos n'est pas installé. Veuillez l'installer avec :\n"
                "pip install zonos"
            )
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement de Zonos : {e}")

    def register_voice(self, voice_id: str, audio_path: Union[str, Path]) -> bool:
        """Enregistre un échantillon audio de référence pour le clonage."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            print(f"Fichier audio de référence non trouvé : {audio_path}")
            return False
        self._cloned_voices[voice_id] = str(audio_path)
        return True

    def synthesize(
        self,
        text: str,
        output_path: Union[str, Path],
        voice_id: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        language: Optional[str] = None,
        speed: Optional[float] = None,
        emotion: Optional[str] = None, # "happiness", "anger", "sadness", "fear", "surprise", etc.
        pitch: Optional[float] = None,
        **kwargs
    ) -> bool:
        """
        Synthétise du texte en un fichier audio en utilisant Zonos.
        """
        try:
            import soundfile as sf
            import torch
            import torchaudio
            from zonos.conditioning import make_cond_dict

            model = self._load_model()
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            lang_code = language or self.config.language
            # Zonos utilise des codes de langue de type "fr-fr" ou "en-us"
            if len(lang_code) == 2:
                lang_code = f"{lang_code}-{lang_code}"

            spd = speed or self.config.speed

            # 1. Préparation du Speaker Embedding (Clonage)
            ref_path = None
            if voice_id and voice_id in self._cloned_voices:
                ref_path = self._cloned_voices[voice_id]
            elif speaker_wav:
                ref_path = speaker_wav

            if ref_path and os.path.exists(ref_path):
                # Charger l'audio de référence
                wav_ref, sr_ref = torchaudio.load(ref_path)
                # S'assurer que l'audio de référence est envoyé sur le bon device
                wav_ref = wav_ref.to(self._device)
                speaker_emb = model.make_speaker_embedding(wav_ref, sr_ref)
            else:
                # Si aucune voix de référence, utiliser un embedding neutre (zéro)
                # ou le comportement par défaut de zonos
                speaker_emb = None

            # 2. Préparation des conditionnements (émotions, pitch, langue, etc.)
            cond_args = {
                "text": text,
                "language": lang_code,
            }

            if speaker_emb is not None:
                cond_args["speaker"] = speaker_emb

            # Émotions supportées par Zonos
            if emotion:
                # zonos gère les émotions via des vecteurs ou dicts de conditionnement spécifiques
                cond_args["emotion"] = emotion

            cond_dict = make_cond_dict(**cond_args)
            conditioning = model.prepare_conditioning(cond_dict)

            # 3. Génération de l'audio
            print(f"Zonos TTS génération : {text[:40]}...")
            codes = model.generate(conditioning, speed_coefficient=spd)
            wavs = model.autoencoder.decode(codes).cpu()

            # wavs est un tenseur [batch, channels, samples], on extrait le premier batch
            audio_data = wavs[0].numpy().squeeze()

            # Sauvegarder
            sf.write(str(output_path), audio_data, self.config.sample_rate)
            return output_path.exists()

        except Exception as e:
            print(f"Erreur de synthèse avec Zonos : {e}")
            return False

    def get_info(self) -> dict:
        return {
            "engine": "ZonosEngine",
            "model": self.config.model_name,
            "supports_cloning": True,
            "expressive_control": True,
            "licence": "Apache 2.0"
        }
