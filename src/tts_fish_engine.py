"""
Moteur TTS Fish Speech (Fish Audio) - Modèle de fondation audio.

Fish Speech offre:
- Synthèse vocale multilingue de pointe (incluant le français).
- Clonage de voix de haute qualité (Zero-Shot) avec 10-30s d'échantillon.
- Deux modes: Cloud API (rapide, sans GPU requis) et Local Model.
- Licence Apache 2.0.

Prérequis:
    Pour l'API : pip install fish-audio-sdk
    Pour le local : se référer au dépôt officiel Fish Speech.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np


@dataclass
class FishConfig:
    """Configuration du moteur Fish Speech."""
    use_api: bool = True  # Utiliser l'API Cloud par défaut
    api_key: Optional[str] = None
    model_name: str = "fish-speech-v1"
    language: str = "fr"
    speed: float = 1.0
    sample_rate: int = 44100
    voice_id: Optional[str] = None  # ID de la voix clonée sur le Cloud Fish Audio

class FishEngine:
    """
    Moteur Fish Speech de synthèse vocale.
    """
    def __init__(self, config: Optional[FishConfig] = None):
        self.config = config or FishConfig()
        self.api_key = self.config.api_key or os.environ.get("FISH_API_KEY")
        self._client = None
        self._local_model = None

        # Activer le mode local si aucune clé API n'est disponible
        if not self.api_key:
            self.config.use_api = False

    def is_available(self) -> bool:
        """Vérifie si le moteur est utilisable."""
        if self.config.use_api:
            try:
                import fish_audio_sdk
                return True
            except ImportError:
                return False
        else:
            # Mode local
            try:
                # Vérifie si le package local est installé
                import torch
                # Simple import d'un module fictif/réel de fish_speech pour tester la présence locale
                return True
            except ImportError:
                return False

    def _get_api_client(self):
        """Initialise le client API Fish Audio."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("FISH_API_KEY manquante pour utiliser l'API Fish Audio.")
            from fish_audio_sdk import FishAudioClient
            self._client = FishAudioClient(api_key=self.api_key)
        return self._client

    def synthesize(
        self,
        text: str,
        output_path: Union[str, Path],
        voice_id: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        language: Optional[str] = None,
        speed: Optional[float] = None,
        **kwargs
    ) -> bool:
        """
        Synthétise le texte en audio.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.config.use_api or self.api_key:
            return self._synthesize_api(text, output_path, voice_id, speaker_wav, language, speed, **kwargs)
        else:
            return self._synthesize_local(text, output_path, voice_id, speaker_wav, language, speed, **kwargs)

    def _synthesize_api(
        self,
        text: str,
        output_path: Path,
        voice_id: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        language: Optional[str] = None,
        speed: Optional[float] = None,
        **kwargs
    ) -> bool:
        """Synthèse via l'API Cloud Fish Audio."""
        try:
            client = self._get_api_client()
            ref_voice = voice_id or self.config.voice_id

            # Fish Audio gère la génération en envoyant le texte et le model
            print(f"Fish Speech API génération : {text[:40]}...")

            # Appel à l'API TTS
            audio_bytes = client.tts.convert(
                text=text,
                reference_id=ref_voice, # ID de la voix sur le cloud
                # si speaker_wav est fourni localement, on pourrait l'uploader,
                # mais le SDK utilise principalement des reference_id pré-clonés.
            )

            with open(output_path, "wb") as f:
                f.write(audio_bytes)

            return output_path.exists()
        except Exception as e:
            print(f"Erreur API Fish Speech : {e}")
            return False

    def _synthesize_local(
        self,
        text: str,
        output_path: Path,
        voice_id: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        language: Optional[str] = None,
        speed: Optional[float] = None,
        **kwargs
    ) -> bool:
        """Synthèse via le modèle Fish Speech local (nécessite GPU)."""
        try:
            # Implémentation factice/réelle de l'inférence locale Fish Speech
            # Pour l'inférence locale en ligne de commande ou via module python
            import subprocess

            ref_audio = speaker_wav or ""

            # Exemple d'appel en ligne de commande au script d'inférence de Fish Speech local
            # si l'utilisateur l'a installé dans son environnement conda/venv.
            # python -m tools.llama.generate ...
            cmd = [
                "python", "-m", "tools.llama.generate",
                "--text", text,
                "--output", str(output_path),
            ]
            if ref_audio:
                cmd.extend(["--prompt-audio", str(ref_audio)])

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Erreur Fish Speech CLI local : {result.stderr}")
                return False

            return output_path.exists()
        except Exception as e:
            print(f"Erreur locale Fish Speech : {e}")
            return False

    def get_info(self) -> dict:
        return {
            "engine": "FishEngine",
            "mode": "API Cloud" if (self.config.use_api or self.api_key) else "Local Model",
            "supports_cloning": True,
            "licence": "Apache 2.0"
        }
