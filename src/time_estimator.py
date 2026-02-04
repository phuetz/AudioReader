"""
Estimation du temps de conversion pour AudioReader.

Fournit des estimations de temps de traitement et de duree audio finale
basees sur les statistiques du texte et le moteur TTS utilise.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict


# Constantes d'estimation
CHARS_PER_MINUTE_AUDIO = 1000  # ~1000 chars = 1 min d'audio parle
BYTES_PER_SECOND_WAV = 48000   # 24kHz mono 16-bit = 48 KB/s

# Facteurs de temps CPU par moteur (relatif a Kokoro = 1.0)
ENGINE_SPEED_FACTORS = {
    "kokoro": 1.0,
    "mms": 1.2,
    "edge": 0.8,  # reseau, rapide mais latence
    "xtts": 3.0,  # clonage = plus lent
    "chatterbox": 2.5,
    "dia": 2.0,
    "f5": 1.8,
    "auto": 1.0,
}

# Overhead du pipeline HQ (multiplicateur)
HQ_OVERHEAD = 1.5


@dataclass
class ConversionEstimate:
    """Resultat d'une estimation de conversion."""
    total_chars: int = 0
    total_words: int = 0
    chapter_count: int = 0
    estimated_audio_duration_s: float = 0.0
    estimated_processing_time_s: float = 0.0
    estimated_file_size_mb: float = 0.0
    engine: str = "kokoro"
    hq_mode: bool = False

    @property
    def audio_duration_formatted(self) -> str:
        """Duree audio au format HH:MM:SS."""
        s = int(self.estimated_audio_duration_s)
        h, remainder = divmod(s, 3600)
        m, sec = divmod(remainder, 60)
        if h > 0:
            return f"{h}h{m:02d}m{sec:02d}s"
        return f"{m}m{sec:02d}s"

    @property
    def processing_time_formatted(self) -> str:
        """Temps de traitement au format HH:MM:SS."""
        s = int(self.estimated_processing_time_s)
        h, remainder = divmod(s, 3600)
        m, sec = divmod(remainder, 60)
        if h > 0:
            return f"{h}h{m:02d}m{sec:02d}s"
        return f"{m}m{sec:02d}s"

    def to_dict(self) -> Dict:
        """Convertit en dictionnaire."""
        return {
            "total_chars": self.total_chars,
            "total_words": self.total_words,
            "chapter_count": self.chapter_count,
            "estimated_audio_duration_s": round(self.estimated_audio_duration_s, 1),
            "estimated_audio_duration": self.audio_duration_formatted,
            "estimated_processing_time_s": round(self.estimated_processing_time_s, 1),
            "estimated_processing_time": self.processing_time_formatted,
            "estimated_file_size_mb": round(self.estimated_file_size_mb, 1),
            "engine": self.engine,
            "hq_mode": self.hq_mode,
        }


def estimate_conversion_time(
    text: str,
    engine: str = "kokoro",
    hq: bool = False,
    chapter_count: int = 1,
) -> ConversionEstimate:
    """
    Estime le temps de conversion pour un texte donne.

    Args:
        text: Texte complet a convertir
        engine: Moteur TTS utilise
        hq: Mode haute qualite active
        chapter_count: Nombre de chapitres

    Returns:
        ConversionEstimate avec toutes les estimations
    """
    total_chars = len(text)
    total_words = len(text.split())

    # Duree audio estimee
    audio_duration_s = (total_chars / CHARS_PER_MINUTE_AUDIO) * 60

    # Taille fichier estimee (WAV non compresse)
    file_size_bytes = audio_duration_s * BYTES_PER_SECOND_WAV
    file_size_mb = file_size_bytes / (1024 * 1024)

    # Temps de traitement CPU
    # Base : ~12s CPU pour 1 min d'audio avec Kokoro
    base_ratio = 12.0 / 60.0  # 0.2x temps reel
    speed_factor = ENGINE_SPEED_FACTORS.get(engine, 1.0)
    processing_time_s = audio_duration_s * base_ratio * speed_factor

    if hq:
        processing_time_s *= HQ_OVERHEAD

    return ConversionEstimate(
        total_chars=total_chars,
        total_words=total_words,
        chapter_count=chapter_count,
        estimated_audio_duration_s=audio_duration_s,
        estimated_processing_time_s=processing_time_s,
        estimated_file_size_mb=file_size_mb,
        engine=engine,
        hq_mode=hq,
    )
