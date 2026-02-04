"""
Moteur TTS Dia 1.6B (Nari Labs).

Multi-speakers natif avec tags non-verbaux.
Necessite: pip install dia-tts torch torchaudio
"""
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List


# Mapping des audio tags AudioReader -> tags Dia natifs
DIA_TAG_MAPPING = {
    "laugh": "(laughs)",
    "chuckle": "(laughs)",
    "cough": "(coughs)",
    "gasp": "(gasps)",
    "sigh": "(sighs)",
    "clears throat": "(clears throat)",
}


@dataclass
class DiaConfig:
    """Configuration du moteur Dia."""
    use_gpu: bool = False
    speed: float = 1.0
    sample_rate: int = 44100  # Dia produit a 44.1kHz


class DiaEngine:
    """
    Moteur TTS Dia 1.6B avec multi-speakers natif.

    Dia supporte nativement:
    - Deux speakers via [S1] et [S2] tags
    - Tags non-verbaux: (laughs), (coughs), (gasps), (sighs)
    """

    def __init__(self, config: Optional[DiaConfig] = None):
        self.config = config or DiaConfig()
        self._model = None
        self._available = None
        self.sample_rate = self.config.sample_rate

    @staticmethod
    def is_available() -> bool:
        """Verifie si dia-tts est installe."""
        try:
            from dia.model import Dia
            return True
        except ImportError:
            return False

    def _load_model(self):
        """Charge le modele Dia."""
        if self._model is None:
            from dia.model import Dia
            device = "cuda" if self.config.use_gpu else "cpu"
            self._model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float32")
        return self._model

    @staticmethod
    def convert_audio_tags_to_dia(text: str) -> str:
        """Convertit les tags AudioReader en tags Dia natifs."""
        import re
        for ar_tag, dia_tag in DIA_TAG_MAPPING.items():
            text = re.sub(
                rf'\[{re.escape(ar_tag)}\]',
                dia_tag,
                text,
                flags=re.IGNORECASE,
            )
        return text

    @staticmethod
    def format_multispeaker(text: str, speakers: Optional[Dict[str, str]] = None) -> str:
        """
        Formate le texte pour multi-speakers Dia.

        Si le texte contient deja [S1]/[S2], le retourne tel quel.
        Sinon, prefixe avec [S1].
        """
        if "[S1]" in text or "[S2]" in text:
            return text
        return f"[S1] {text}"

    def synthesize(
        self,
        text: str,
        output_path,
        voice: Optional[str] = None,
        voices: Optional[Dict[str, str]] = None,
        speed: Optional[float] = None,
        **kwargs
    ) -> bool:
        """
        Synthetise du texte en audio.

        Args:
            text: Texte a synthetiser (peut contenir [S1]/[S2] et tags non-verbaux)
            output_path: Chemin du fichier de sortie
            voices: Mapping speaker -> config (optionnel)
            speed: Vitesse de lecture
        """
        try:
            import soundfile as sf

            model = self._load_model()

            # Convertir les tags AudioReader en tags Dia
            processed_text = self.convert_audio_tags_to_dia(text)
            processed_text = self.format_multispeaker(processed_text)

            # Generer l'audio
            output = model.generate(
                processed_text,
                use_torch_compile=False,
                verbose=False,
            )

            audio = np.array(output, dtype=np.float32)

            # Appliquer la vitesse
            spd = speed or self.config.speed
            if abs(spd - 1.0) > 0.01:
                import librosa
                audio = librosa.effects.time_stretch(audio, rate=spd)

            sf.write(str(output_path), audio, self.sample_rate)
            return True

        except Exception as e:
            print(f"Erreur Dia: {e}")
            return False

    def get_info(self) -> dict:
        return {
            "engine": "dia",
            "version": "1.6B",
            "supports_multispeaker": True,
            "supports_nonverbal_tags": True,
            "supported_tags": list(DIA_TAG_MAPPING.values()),
        }
