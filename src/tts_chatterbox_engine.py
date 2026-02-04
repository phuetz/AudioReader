"""
Moteur TTS Chatterbox (Resemble AI).

Clonage de voix zero-shot avec controle d'exageration emotionnelle.
Necessite: pip install chatterbox-tts torch torchaudio
"""
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ChatterboxConfig:
    """Configuration du moteur Chatterbox."""
    default_language: str = "en"
    use_gpu: bool = False
    temperature: float = 0.7
    speed: float = 1.0
    exaggeration: float = 0.5  # 0.0-1.0, controle emotionnel unique a Chatterbox
    sample_rate: int = 24000


class ChatterboxEngine:
    """
    Moteur TTS Chatterbox avec clonage zero-shot.

    Chatterbox de Resemble AI offre un clonage de voix de haute qualite
    avec un parametre unique de controle d'exageration emotionnelle.
    """

    def __init__(self, config: Optional[ChatterboxConfig] = None):
        self.config = config or ChatterboxConfig()
        self._model = None
        self._available = None
        self.sample_rate = self.config.sample_rate

    @staticmethod
    def is_available() -> bool:
        """Verifie si chatterbox-tts est installe."""
        try:
            import chatterbox.tts
            return True
        except ImportError:
            return False

    def _load_model(self):
        """Charge le modele Chatterbox."""
        if self._model is None:
            import torch
            from chatterbox.tts import ChatterboxTTS
            device = "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
            self._model = ChatterboxTTS.from_pretrained(device=device)
        return self._model

    def synthesize(
        self,
        text: str,
        output_path,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        exaggeration: Optional[float] = None,
        speaker_wav: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Synthetise du texte en audio.

        Args:
            text: Texte a synthetiser
            output_path: Chemin du fichier de sortie
            voice: Non utilise (Chatterbox utilise speaker_wav)
            speed: Vitesse de lecture
            exaggeration: Controle d'exageration emotionnelle (0.0-1.0)
            speaker_wav: Fichier audio de reference pour le clonage
        """
        try:
            import torch
            import soundfile as sf

            model = self._load_model()
            spd = speed or self.config.speed
            exag = exaggeration if exaggeration is not None else self.config.exaggeration

            wav = model.generate(
                text,
                audio_prompt_path=speaker_wav,
                exaggeration=exag,
            )

            # Convertir en numpy
            if isinstance(wav, torch.Tensor):
                audio = wav.squeeze().cpu().numpy()
            else:
                audio = np.array(wav, dtype=np.float32)

            # Appliquer la vitesse si != 1.0
            if abs(spd - 1.0) > 0.01:
                import librosa
                audio = librosa.effects.time_stretch(audio, rate=spd)

            sf.write(str(output_path), audio, self.sample_rate)
            return True

        except Exception as e:
            print(f"Erreur Chatterbox: {e}")
            return False

    def get_info(self) -> dict:
        return {
            "engine": "chatterbox",
            "version": "1.0",
            "supports_cloning": True,
            "supports_exaggeration": True,
        }
