"""
Moteur TTS F5-TTS (Flow Matching).

Clonage de voix avec ~10s d'audio de reference.
Non-autoregressif, CPU-friendly.
Necessite: pip install f5-tts torch torchaudio
"""
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class F5Config:
    """Configuration du moteur F5-TTS."""
    default_language: str = "fr"
    use_gpu: bool = False
    speed: float = 1.0
    sample_rate: int = 24000


class F5Engine:
    """
    Moteur F5-TTS base sur le flow matching.

    F5-TTS est non-autoregressif et CPU-friendly.
    Il necessite environ 10 secondes d'audio de reference pour le clonage.
    """

    def __init__(self, config: Optional[F5Config] = None):
        self.config = config or F5Config()
        self._model = None
        self._available = None
        self.sample_rate = self.config.sample_rate

    @staticmethod
    def is_available() -> bool:
        """Verifie si f5-tts est installe."""
        try:
            from f5_tts.api import F5TTS
            return True
        except ImportError:
            return False

    def _load_model(self):
        """Charge le modele F5-TTS."""
        if self._model is None:
            from f5_tts.api import F5TTS
            device = "cuda" if self.config.use_gpu else "cpu"
            self._model = F5TTS(device=device)
        return self._model

    def synthesize(
        self,
        text: str,
        output_path,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Synthetise du texte en audio.

        Args:
            text: Texte a synthetiser
            output_path: Chemin du fichier de sortie
            ref_audio: Fichier audio de reference pour le clonage (~10s)
            ref_text: Transcription de l'audio de reference
            speed: Vitesse de lecture
        """
        try:
            import soundfile as sf

            model = self._load_model()
            spd = speed or self.config.speed

            wav, sr, _ = model.infer(
                ref_file=ref_audio or "",
                ref_text=ref_text or "",
                gen_text=text,
                speed=spd,
            )

            audio = np.array(wav, dtype=np.float32)
            self.sample_rate = sr

            sf.write(str(output_path), audio, sr)
            return True

        except Exception as e:
            print(f"Erreur F5-TTS: {e}")
            return False

    def get_info(self) -> dict:
        return {
            "engine": "f5",
            "version": "1.0",
            "supports_cloning": True,
            "non_autoregressive": True,
            "cpu_friendly": True,
        }
