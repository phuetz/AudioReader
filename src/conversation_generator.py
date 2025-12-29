"""
Generateur de conversations multi-speakers.

Fonctionnalites:
- Parsing de scripts de dialogue
- Attribution automatique de voix
- Generation coordonnee multi-personnages
- Export avec timing pour montage
"""
import re
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from enum import Enum
import numpy as np


class SpeakerGender(Enum):
    """Genre du personnage."""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


@dataclass
class Speaker:
    """Un personnage avec sa voix."""
    name: str
    voice_id: str
    gender: SpeakerGender = SpeakerGender.NEUTRAL
    speed: float = 1.0
    pitch_shift: float = 0.0
    description: str = ""
    color: str = "#FFFFFF"  # Pour visualisation


@dataclass
class DialogueLine:
    """Une ligne de dialogue."""
    speaker: Speaker
    text: str
    emotion: Optional[str] = None
    direction: str = ""  # Indication scenique
    pause_before: float = 0.0
    pause_after: float = 0.3


@dataclass
class Conversation:
    """Une conversation complete."""
    title: str
    speakers: Dict[str, Speaker]
    lines: List[DialogueLine]
    metadata: dict = field(default_factory=dict)


class DialogueParser:
    """
    Parse differents formats de dialogue.

    Formats supportes:
    - Format script: "PERSONNAGE: dialogue"
    - Format markdown: "**Personnage:** dialogue"
    - Format theatre: "PERSONNAGE. - dialogue"
    - Format JSON structure
    """

    # Patterns pour differents formats
    PATTERNS = {
        "script": re.compile(r'^([A-Z][A-Z\s]+):\s*(.+)$'),
        "script_lower": re.compile(r'^([A-Za-z][A-Za-z\s]+):\s*(.+)$'),
        "markdown": re.compile(r'^\*\*([^*:]+):?\*\*:?\s*(.+)$'),
        "theatre": re.compile(r'^([A-Z][A-Z\s]+)\.\s*[-–]\s*(.+)$'),
        "parentheses": re.compile(r'^\(([^)]+)\)\s*(.+)$'),
    }

    # Pattern pour les indications sceniques
    DIRECTION_PATTERN = re.compile(r'\[([^\]]+)\]|\(([^)]+)\)')

    # Pattern pour les emotions
    EMOTION_PATTERN = re.compile(
        r'\[(excited|sad|angry|whispers|fearful|tender|dramatic|'
        r'sarcastic|cheerful|serious|mysterious)\]',
        re.IGNORECASE
    )

    def __init__(self):
        self._detected_format = None

    def detect_format(self, text: str) -> str:
        """Detecte le format du dialogue."""
        lines = text.strip().split('\n')

        for line in lines[:20]:  # Check first 20 lines
            line = line.strip()
            if not line:
                continue

            for format_name, pattern in self.PATTERNS.items():
                if pattern.match(line):
                    return format_name

        return "unknown"

    def parse(self, text: str, default_speakers: Dict[str, Speaker] = None) -> List[Tuple[str, str, str]]:
        """
        Parse le texte en lignes de dialogue.

        Returns:
            Liste de tuples (speaker_name, dialogue, direction)
        """
        format_type = self.detect_format(text)
        self._detected_format = format_type

        if format_type == "unknown":
            # Essayer de parser comme texte narratif avec guillemets
            return self._parse_narrative(text)

        lines = text.strip().split('\n')
        dialogues = []
        current_speaker = None
        current_text = []

        for line in lines:
            line = line.strip()
            if not line:
                if current_speaker and current_text:
                    dialogues.append((
                        current_speaker,
                        ' '.join(current_text),
                        ""
                    ))
                    current_text = []
                continue

            # Essayer de matcher un nouveau speaker
            matched = False
            for pattern in self.PATTERNS.values():
                match = pattern.match(line)
                if match:
                    # Sauvegarder le precedent
                    if current_speaker and current_text:
                        dialogues.append((
                            current_speaker,
                            ' '.join(current_text),
                            ""
                        ))

                    current_speaker = match.group(1).strip()
                    current_text = [match.group(2).strip()]
                    matched = True
                    break

            if not matched and current_speaker:
                # Continuation du dialogue
                current_text.append(line)

        # Dernier dialogue
        if current_speaker and current_text:
            dialogues.append((
                current_speaker,
                ' '.join(current_text),
                ""
            ))

        return dialogues

    def _parse_narrative(self, text: str) -> List[Tuple[str, str, str]]:
        """Parse un texte narratif avec guillemets."""
        # Pattern pour dialogues entre guillemets
        quote_patterns = [
            r'[«"]([^»"]+)[»"]',  # Guillemets francais et anglais
            r'"([^"]+)"',         # Double quotes
            r"'([^']+)'",         # Single quotes
        ]

        dialogues = []
        remaining = text

        for pattern in quote_patterns:
            for match in re.finditer(pattern, remaining):
                dialogue = match.group(1).strip()
                if len(dialogue) > 5:  # Ignorer les tres courts
                    dialogues.append(("SPEAKER", dialogue, ""))

        return dialogues if dialogues else [("NARRATOR", text, "")]

    def extract_direction(self, text: str) -> Tuple[str, str]:
        """
        Extrait les indications sceniques du texte.

        Returns:
            Tuple (texte_nettoye, indication)
        """
        directions = []

        for match in self.DIRECTION_PATTERN.finditer(text):
            direction = match.group(1) or match.group(2)
            if direction and not self.EMOTION_PATTERN.match(f"[{direction}]"):
                directions.append(direction)

        cleaned = self.DIRECTION_PATTERN.sub('', text).strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)

        return cleaned, '; '.join(directions)

    def extract_emotion(self, text: str) -> Tuple[str, Optional[str]]:
        """
        Extrait l'emotion du texte.

        Returns:
            Tuple (texte_nettoye, emotion)
        """
        match = self.EMOTION_PATTERN.search(text)
        if match:
            emotion = match.group(1).lower()
            cleaned = self.EMOTION_PATTERN.sub('', text).strip()
            return cleaned, emotion
        return text, None


class VoicePool:
    """Pool de voix disponibles pour assignation automatique."""

    # Voix Kokoro par defaut
    KOKORO_VOICES = {
        "male_fr": ["fm_hugo", "fm_florian"],
        "female_fr": ["ff_siwis", "ff_sophie"],
        "male_en": ["am_michael", "am_adam"],
        "female_en": ["af_sarah", "af_sky"],
    }

    def __init__(self, voices: Dict[str, List[str]] = None):
        self.voices = voices or self.KOKORO_VOICES
        self._assigned: Dict[str, str] = {}
        self._usage_count: Dict[str, int] = {}

    def assign_voice(
        self,
        speaker_name: str,
        gender: SpeakerGender = SpeakerGender.NEUTRAL,
        language: str = "fr"
    ) -> str:
        """
        Assigne une voix a un speaker.

        Evite de reutiliser la meme voix pour differents speakers.
        """
        if speaker_name in self._assigned:
            return self._assigned[speaker_name]

        # Determiner le pool de voix
        if gender == SpeakerGender.MALE:
            pool_key = f"male_{language}"
        elif gender == SpeakerGender.FEMALE:
            pool_key = f"female_{language}"
        else:
            # Alterner entre male et female
            assigned_count = len(self._assigned)
            pool_key = f"{'male' if assigned_count % 2 == 0 else 'female'}_{language}"

        pool = self.voices.get(pool_key, list(self.voices.values())[0])

        # Trouver la voix la moins utilisee
        min_usage = float('inf')
        best_voice = pool[0]

        for voice in pool:
            usage = self._usage_count.get(voice, 0)
            if usage < min_usage:
                min_usage = usage
                best_voice = voice

        # Assigner
        self._assigned[speaker_name] = best_voice
        self._usage_count[best_voice] = self._usage_count.get(best_voice, 0) + 1

        return best_voice

    def get_assignment(self) -> Dict[str, str]:
        """Retourne les assignations courantes."""
        return self._assigned.copy()

    def reset(self):
        """Reinitialise les assignations."""
        self._assigned.clear()
        self._usage_count.clear()


class ConversationGenerator:
    """
    Generateur de conversations TTS multi-speakers.

    Pipeline complet:
    1. Parsing du script
    2. Assignation des voix
    3. Generation TTS coordonnee
    4. Assemblage avec timing
    """

    def __init__(
        self,
        voice_pool: VoicePool = None,
        default_language: str = "fr"
    ):
        self.parser = DialogueParser()
        self.voice_pool = voice_pool or VoicePool()
        self.default_language = default_language

    def parse_script(
        self,
        script: str,
        speaker_config: Dict[str, dict] = None
    ) -> Conversation:
        """
        Parse un script de dialogue.

        Args:
            script: Texte du script
            speaker_config: Config optionnelle des speakers
                {
                    "Jean": {"gender": "male", "voice": "fm_hugo"},
                    "Marie": {"gender": "female"}
                }

        Returns:
            Conversation object
        """
        # Parser le dialogue
        raw_dialogues = self.parser.parse(script)

        # Construire les speakers
        speakers: Dict[str, Speaker] = {}
        speaker_config = speaker_config or {}

        for speaker_name, _, _ in raw_dialogues:
            if speaker_name not in speakers:
                config = speaker_config.get(speaker_name, {})

                # Determiner le genre
                gender_str = config.get("gender", "neutral")
                gender = SpeakerGender(gender_str) if gender_str in ["male", "female", "neutral"] else SpeakerGender.NEUTRAL

                # Assigner la voix
                voice_id = config.get("voice")
                if not voice_id:
                    voice_id = self.voice_pool.assign_voice(
                        speaker_name,
                        gender,
                        self.default_language
                    )

                speakers[speaker_name] = Speaker(
                    name=speaker_name,
                    voice_id=voice_id,
                    gender=gender,
                    speed=config.get("speed", 1.0),
                    pitch_shift=config.get("pitch_shift", 0.0),
                    description=config.get("description", "")
                )

        # Construire les lignes
        lines: List[DialogueLine] = []

        for speaker_name, text, _ in raw_dialogues:
            # Extraire emotion d'abord (avant direction qui enleve les [])
            text, emotion = self.parser.extract_emotion(text)
            # Puis extraire direction
            text, direction = self.parser.extract_direction(text)

            lines.append(DialogueLine(
                speaker=speakers[speaker_name],
                text=text,
                emotion=emotion,
                direction=direction
            ))

        return Conversation(
            title="Conversation",
            speakers=speakers,
            lines=lines
        )

    def generate_audio(
        self,
        conversation: Conversation,
        synthesize_fn,
        output_dir: Path,
        gap_between_lines: float = 0.3
    ) -> List[dict]:
        """
        Genere l'audio pour une conversation.

        Args:
            conversation: Conversation a generer
            synthesize_fn: Fonction de synthese (text, voice_id, speed) -> audio
            output_dir: Dossier de sortie
            gap_between_lines: Pause entre les lignes

        Returns:
            Liste des segments avec timing
        """
        import soundfile as sf

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        segments = []
        current_time = 0.0
        sample_rate = 24000  # Kokoro default

        for i, line in enumerate(conversation.lines):
            # Pause avant
            pause_before = line.pause_before or 0.0
            current_time += pause_before

            # Generer l'audio
            audio = synthesize_fn(
                text=line.text,
                voice_id=line.speaker.voice_id,
                speed=line.speaker.speed
            )

            # Sauvegarder le segment
            segment_path = output_dir / f"segment_{i:04d}_{line.speaker.name}.wav"
            sf.write(str(segment_path), audio, sample_rate)

            duration = len(audio) / sample_rate

            segments.append({
                "index": i,
                "speaker": line.speaker.name,
                "voice_id": line.speaker.voice_id,
                "text": line.text,
                "emotion": line.emotion,
                "direction": line.direction,
                "start_time": current_time,
                "duration": duration,
                "end_time": current_time + duration,
                "audio_path": str(segment_path),
            })

            current_time += duration + (line.pause_after or gap_between_lines)

        return segments

    def assemble_audio(
        self,
        segments: List[dict],
        output_path: Path,
        sample_rate: int = 24000,
        crossfade_ms: int = 50
    ) -> Path:
        """
        Assemble tous les segments en un fichier final.

        Args:
            segments: Liste des segments avec timing
            output_path: Chemin de sortie
            sample_rate: Frequence d'echantillonnage
            crossfade_ms: Duree du crossfade en ms

        Returns:
            Chemin du fichier assemble
        """
        import soundfile as sf

        if not segments:
            return output_path

        # Calculer la duree totale
        total_duration = max(seg["end_time"] for seg in segments)
        total_samples = int(total_duration * sample_rate) + sample_rate  # +1s margin

        # Buffer de sortie
        output = np.zeros(total_samples, dtype=np.float32)

        # Crossfade samples
        crossfade_samples = int(crossfade_ms * sample_rate / 1000)

        for seg in segments:
            # Charger l'audio
            audio, _ = sf.read(seg["audio_path"])

            # Position de debut
            start_sample = int(seg["start_time"] * sample_rate)
            end_sample = start_sample + len(audio)

            if end_sample > len(output):
                # Etendre le buffer si necessaire
                output = np.pad(output, (0, end_sample - len(output)))

            # Appliquer avec crossfade
            if crossfade_samples > 0 and start_sample > crossfade_samples:
                # Fade in
                fade_in = np.linspace(0, 1, min(crossfade_samples, len(audio)))
                audio[:len(fade_in)] *= fade_in

            # Mixer (additionner, pas remplacer)
            output[start_sample:end_sample] += audio

        # Normaliser
        max_val = np.abs(output).max()
        if max_val > 0.95:
            output = output * 0.95 / max_val

        # Sauvegarder
        sf.write(str(output_path), output, sample_rate)

        return output_path

    def export_timeline(
        self,
        segments: List[dict],
        output_path: Path,
        format: str = "json"
    ):
        """
        Exporte la timeline pour montage externe.

        Formats: json, srt, csv
        """
        output_path = Path(output_path)

        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=2, ensure_ascii=False)

        elif format == "srt":
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, seg in enumerate(segments):
                    start = self._format_srt_time(seg["start_time"])
                    end = self._format_srt_time(seg["end_time"])
                    f.write(f"{i + 1}\n")
                    f.write(f"{start} --> {end}\n")
                    f.write(f"[{seg['speaker']}] {seg['text']}\n\n")

        elif format == "csv":
            import csv
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "index", "speaker", "text", "start_time", "duration", "emotion"
                ])
                writer.writeheader()
                for seg in segments:
                    writer.writerow({
                        "index": seg["index"],
                        "speaker": seg["speaker"],
                        "text": seg["text"],
                        "start_time": f"{seg['start_time']:.3f}",
                        "duration": f"{seg['duration']:.3f}",
                        "emotion": seg.get("emotion", ""),
                    })

    def _format_srt_time(self, seconds: float) -> str:
        """Formate un temps en format SRT."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def generate_conversation_from_script(
    script: str,
    output_path: Path,
    speaker_config: Dict[str, dict] = None,
    synthesize_fn = None
) -> dict:
    """
    Fonction utilitaire pour generer une conversation.

    Usage:
        result = generate_conversation_from_script(
            script=\"\"\"
            JEAN: Bonjour Marie, comment vas-tu?
            MARIE: Tres bien, merci! Et toi?
            JEAN: [excited] Super! J'ai une grande nouvelle!
            \"\"\",
            output_path=Path("conversation.wav"),
            speaker_config={
                "JEAN": {"gender": "male"},
                "MARIE": {"gender": "female"}
            }
        )
    """
    generator = ConversationGenerator()

    # Parser
    conversation = generator.parse_script(script, speaker_config)

    if synthesize_fn is None:
        # Retourner juste la conversation parsee
        return {
            "conversation": conversation,
            "speakers": {name: sp.voice_id for name, sp in conversation.speakers.items()},
            "lines_count": len(conversation.lines),
        }

    # Generer
    output_dir = output_path.parent / f"{output_path.stem}_segments"
    segments = generator.generate_audio(
        conversation,
        synthesize_fn,
        output_dir
    )

    # Assembler
    generator.assemble_audio(segments, output_path)

    # Exporter timeline
    timeline_path = output_path.with_suffix('.json')
    generator.export_timeline(segments, timeline_path)

    return {
        "output_path": str(output_path),
        "timeline_path": str(timeline_path),
        "segments_count": len(segments),
        "total_duration": segments[-1]["end_time"] if segments else 0,
        "speakers": {name: sp.voice_id for name, sp in conversation.speakers.items()},
    }


class PodcastGenerator(ConversationGenerator):
    """
    Generateur specialise pour podcasts.

    Ajoute:
    - Intro/outro musicaux
    - Jingles entre sections
    - Normalisation broadcast
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intro_path: Optional[Path] = None
        self.outro_path: Optional[Path] = None
        self.jingle_path: Optional[Path] = None

    def set_audio_elements(
        self,
        intro: Path = None,
        outro: Path = None,
        jingle: Path = None
    ):
        """Configure les elements audio."""
        self.intro_path = intro
        self.outro_path = outro
        self.jingle_path = jingle

    def generate_podcast(
        self,
        sections: List[Conversation],
        output_path: Path,
        synthesize_fn,
        add_jingles: bool = True
    ) -> Path:
        """
        Genere un podcast complet avec plusieurs sections.

        Args:
            sections: Liste de conversations (une par section)
            output_path: Fichier de sortie
            synthesize_fn: Fonction de synthese
            add_jingles: Ajouter des jingles entre sections

        Returns:
            Chemin du podcast genere
        """
        import soundfile as sf

        all_segments = []
        current_time = 0.0
        sample_rate = 24000

        # Intro
        if self.intro_path and self.intro_path.exists():
            intro_audio, sr = sf.read(str(self.intro_path))
            current_time += len(intro_audio) / sr + 0.5

        # Sections
        for i, conversation in enumerate(sections):
            # Jingle entre sections
            if i > 0 and add_jingles and self.jingle_path:
                jingle_audio, sr = sf.read(str(self.jingle_path))
                current_time += len(jingle_audio) / sr + 0.3

            # Generer la section
            temp_dir = output_path.parent / f"section_{i}"
            segments = self.generate_audio(
                conversation,
                synthesize_fn,
                temp_dir
            )

            # Ajuster les timings
            for seg in segments:
                seg["start_time"] += current_time
                seg["end_time"] += current_time

            all_segments.extend(segments)

            if segments:
                current_time = segments[-1]["end_time"] + 1.0

        # Assembler
        self.assemble_audio(all_segments, output_path)

        return output_path


if __name__ == "__main__":
    print("=== Test Conversation Generator ===\n")

    # Test parsing
    script = """
    JEAN: Bonjour Marie! Comment ca va aujourd'hui?
    MARIE: [cheerful] Ca va tres bien, merci! Et toi?
    JEAN: Super! [excited] J'ai une grande nouvelle a t'annoncer!
    MARIE: Ah oui? Dis-moi tout!
    JEAN: [dramatic] Je vais... me marier!
    MARIE: [surprised] Oh mon dieu! Felicitations!
    """

    generator = ConversationGenerator()
    conversation = generator.parse_script(
        script,
        speaker_config={
            "JEAN": {"gender": "male"},
            "MARIE": {"gender": "female"}
        }
    )

    print("Speakers detectes:")
    for name, speaker in conversation.speakers.items():
        print(f"  {name}: {speaker.voice_id} ({speaker.gender.value})")

    print(f"\nNombre de lignes: {len(conversation.lines)}")

    print("\nDialogues:")
    for line in conversation.lines:
        emotion_str = f" [{line.emotion}]" if line.emotion else ""
        print(f"  {line.speaker.name}{emotion_str}: {line.text[:50]}...")
