"""
Generateur de sous-titres synchronises pour AudioReader.

Genere des fichiers SRT, VTT et JSON avec timestamps
synchronises a l'audio genere.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class SubtitleEntry:
    """Une entree de sous-titre."""
    index: int
    start_s: float
    end_s: float
    text: str


@dataclass
class WordTimestamp:
    """Timestamp pour un mot individuel."""
    word: str
    start_s: float
    end_s: float
    confidence: float = 1.0


# Constante d'estimation de debit de parole
CHARS_PER_SECOND = 15  # ~15 caracteres par seconde en parole normale


class SubtitleGenerator:
    """
    Generateur de sous-titres synchronises.

    Utilise Whisper pour le forced alignment si disponible,
    sinon estimation basee sur le nombre de caracteres.
    """

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self._whisper_available = None

    def _check_whisper(self) -> bool:
        """Verifie si Whisper est disponible."""
        if self._whisper_available is None:
            try:
                import whisper
                self._whisper_available = True
            except ImportError:
                self._whisper_available = False
        return self._whisper_available

    def _split_into_sentences(self, text: str) -> List[str]:
        """Decoupe le texte en phrases."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _estimate_timestamps(self, text: str, audio_duration_s: float) -> List[SubtitleEntry]:
        """
        Estime les timestamps par phrase basees sur le ratio caracteres/temps.

        Fallback quand Whisper n'est pas disponible.
        """
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        # Calculer le ratio chars/seconde effectif
        total_chars = sum(len(s) for s in sentences)
        if total_chars == 0:
            return []

        ratio = audio_duration_s / total_chars

        entries = []
        current_time = 0.0

        for i, sentence in enumerate(sentences):
            duration = len(sentence) * ratio
            entries.append(SubtitleEntry(
                index=i + 1,
                start_s=round(current_time, 3),
                end_s=round(current_time + duration, 3),
                text=sentence,
            ))
            current_time += duration

        return entries

    def _whisper_align(self, audio_path: str, text: str) -> List[SubtitleEntry]:
        """Aligne le texte avec l'audio via Whisper."""
        try:
            import whisper

            model = whisper.load_model("base")
            result = model.transcribe(
                audio_path,
                language="fr",
                word_timestamps=True,
            )

            entries = []
            for i, segment in enumerate(result.get("segments", [])):
                entries.append(SubtitleEntry(
                    index=i + 1,
                    start_s=round(segment["start"], 3),
                    end_s=round(segment["end"], 3),
                    text=segment["text"].strip(),
                ))

            return entries

        except Exception:
            return []

    def _whisper_word_timestamps(self, audio_path: str) -> List[WordTimestamp]:
        """Obtient les timestamps mot par mot via Whisper."""
        try:
            import whisper

            model = whisper.load_model("base")
            result = model.transcribe(
                audio_path,
                language="fr",
                word_timestamps=True,
            )

            words = []
            for segment in result.get("segments", []):
                for word_info in segment.get("words", []):
                    words.append(WordTimestamp(
                        word=word_info["word"].strip(),
                        start_s=round(word_info["start"], 3),
                        end_s=round(word_info["end"], 3),
                        confidence=round(word_info.get("probability", 1.0), 3),
                    ))

            return words

        except Exception:
            return []

    def _estimate_word_timestamps(self, text: str, audio_duration_s: float) -> List[WordTimestamp]:
        """Estime les timestamps mot par mot (fallback)."""
        words = text.split()
        if not words:
            return []

        total_chars = sum(len(w) for w in words)
        if total_chars == 0:
            return []

        ratio = audio_duration_s / total_chars
        timestamps = []
        current_time = 0.0

        for word in words:
            duration = len(word) * ratio
            timestamps.append(WordTimestamp(
                word=word,
                start_s=round(current_time, 3),
                end_s=round(current_time + duration, 3),
                confidence=0.5,  # Estimation
            ))
            current_time += duration

        return timestamps

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Formate le temps en format SRT (HH:MM:SS,mmm)."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def _format_vtt_time(seconds: float) -> str:
        """Formate le temps en format VTT (HH:MM:SS.mmm)."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    def generate_srt(
        self,
        audio_path: str,
        text: str,
        output_path: str,
        audio_duration_s: Optional[float] = None,
    ) -> bool:
        """
        Genere un fichier SRT.

        Args:
            audio_path: Chemin du fichier audio
            text: Texte source
            output_path: Chemin du fichier SRT de sortie
            audio_duration_s: Duree audio (pour le fallback sans Whisper)
        """
        if self._check_whisper():
            entries = self._whisper_align(audio_path, text)
        else:
            duration = audio_duration_s or self._get_audio_duration(audio_path)
            entries = self._estimate_timestamps(text, duration)

        if not entries:
            return False

        lines = []
        for entry in entries:
            lines.append(str(entry.index))
            lines.append(
                f"{self._format_srt_time(entry.start_s)} --> {self._format_srt_time(entry.end_s)}"
            )
            lines.append(entry.text)
            lines.append("")

        Path(output_path).write_text("\n".join(lines), encoding="utf-8")
        return True

    def generate_vtt(
        self,
        audio_path: str,
        text: str,
        output_path: str,
        audio_duration_s: Optional[float] = None,
    ) -> bool:
        """Genere un fichier WebVTT."""
        if self._check_whisper():
            entries = self._whisper_align(audio_path, text)
        else:
            duration = audio_duration_s or self._get_audio_duration(audio_path)
            entries = self._estimate_timestamps(text, duration)

        if not entries:
            return False

        lines = ["WEBVTT", ""]
        for entry in entries:
            lines.append(
                f"{self._format_vtt_time(entry.start_s)} --> {self._format_vtt_time(entry.end_s)}"
            )
            lines.append(entry.text)
            lines.append("")

        Path(output_path).write_text("\n".join(lines), encoding="utf-8")
        return True

    def generate_word_level_json(
        self,
        audio_path: str,
        text: str,
        output_path: str,
        audio_duration_s: Optional[float] = None,
    ) -> bool:
        """Genere un fichier JSON avec timestamps par mot."""
        import json

        if self._check_whisper():
            words = self._whisper_word_timestamps(audio_path)
        else:
            duration = audio_duration_s or self._get_audio_duration(audio_path)
            words = self._estimate_word_timestamps(text, duration)

        if not words:
            return False

        data = {
            "words": [
                {
                    "word": w.word,
                    "start": w.start_s,
                    "end": w.end_s,
                    "confidence": w.confidence,
                }
                for w in words
            ]
        }

        Path(output_path).write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return True

    def generate_with_whisper(
        self,
        audio_path: str,
        output_path: str,
        format: str = "srt",
        language: str = "fr",
    ) -> bool:
        """
        Genere des sous-titres en utilisant Whisper pour l'alignement.

        Fallback automatique sur l'estimation si Whisper n'est pas disponible.

        Args:
            audio_path: Chemin du fichier audio
            output_path: Chemin du fichier de sortie
            format: Format de sortie ('srt', 'vtt', 'json')
            language: Code langue

        Returns:
            True si generation reussie
        """
        if not self._check_whisper():
            # Fallback: lire le texte n'est pas possible ici,
            # utiliser Whisper en mode transcription pure
            return False

        try:
            import whisper

            model = whisper.load_model("base")
            result = model.transcribe(
                audio_path,
                language=language,
                word_timestamps=True,
            )

            segments = result.get("segments", [])
            if not segments:
                return False

            if format == "json":
                import json
                words = []
                for seg in segments:
                    for w in seg.get("words", []):
                        words.append({
                            "word": w["word"].strip(),
                            "start": round(w["start"], 3),
                            "end": round(w["end"], 3),
                            "confidence": round(w.get("probability", 1.0), 3),
                        })
                Path(output_path).write_text(
                    json.dumps({"words": words}, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            elif format == "vtt":
                lines = ["WEBVTT", ""]
                for i, seg in enumerate(segments):
                    lines.append(
                        f"{self._format_vtt_time(seg['start'])} --> {self._format_vtt_time(seg['end'])}"
                    )
                    lines.append(seg["text"].strip())
                    lines.append("")
                Path(output_path).write_text("\n".join(lines), encoding="utf-8")
            else:  # srt
                lines = []
                for i, seg in enumerate(segments):
                    lines.append(str(i + 1))
                    lines.append(
                        f"{self._format_srt_time(seg['start'])} --> {self._format_srt_time(seg['end'])}"
                    )
                    lines.append(seg["text"].strip())
                    lines.append("")
                Path(output_path).write_text("\n".join(lines), encoding="utf-8")

            return True

        except Exception:
            return False

    def _get_audio_duration(self, audio_path: str) -> float:
        """Obtient la duree d'un fichier audio."""
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            return info.duration
        except Exception:
            return 60.0  # Fallback: 1 minute
