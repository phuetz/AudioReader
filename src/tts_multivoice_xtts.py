"""
Moteur TTS multi-voix basé sur XTTS-v2.

Utilise le clonage de voix XTTS-v2 avec différents samples
pour créer des voix distinctes pour chaque personnage.
"""
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np
import soundfile as sf

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.character_detector import (
    CharacterDetector, VoiceAssigner, DialogueSegment,
    SpeakerType, Character, process_text_with_characters
)


@dataclass
class VoiceSample:
    """Un échantillon de voix pour le clonage."""
    path: Path
    gender: str  # "M", "F", "N" (neutral)
    name: str
    duration: float = 0.0


@dataclass
class MultiVoiceConfig:
    """Configuration du moteur multi-voix."""
    samples_dir: Path = Path("voice_samples")
    default_narrator_voice: str = "narrator_sample_01"
    default_male_voice: str = "male_extended"
    default_female_voice: str = "female_narrator_01"
    language: str = "fr"
    use_gpu: bool = False
    # Mapping personnage -> sample de voix
    character_voices: Dict[str, str] = field(default_factory=dict)


class MultiVoiceXTTSEngine:
    """
    Moteur TTS multi-voix utilisant XTTS-v2.

    Permet de générer de l'audio avec différentes voix pour
    le narrateur et les personnages.
    """

    def __init__(self, config: Optional[MultiVoiceConfig] = None):
        self.config = config or MultiVoiceConfig()
        self.tts = None
        self.voices: Dict[str, VoiceSample] = {}
        self._load_voices()

    def _load_voices(self):
        """Charge les samples de voix disponibles."""
        samples_dir = self.config.samples_dir

        for subdir, gender in [("narrator", "N"), ("male", "M"), ("female", "F")]:
            voice_dir = samples_dir / subdir
            if not voice_dir.exists():
                continue

            for wav_file in voice_dir.glob("*.wav"):
                try:
                    audio, sr = sf.read(str(wav_file))
                    duration = len(audio) / sr

                    voice = VoiceSample(
                        path=wav_file,
                        gender=gender,
                        name=wav_file.stem,
                        duration=duration
                    )
                    self.voices[wav_file.stem] = voice
                    print(f"  Voix chargée: {wav_file.stem} ({gender}, {duration:.1f}s)")
                except Exception as e:
                    print(f"  Erreur chargement {wav_file}: {e}")

    def _init_tts(self):
        """Initialise le modèle XTTS-v2."""
        if self.tts is not None:
            return

        try:
            from TTS.api import TTS
            print("Chargement XTTS-v2...")
            self.tts = TTS(
                "tts_models/multilingual/multi-dataset/xtts_v2",
                gpu=self.config.use_gpu
            )
            print("XTTS-v2 prêt!")
        except Exception as e:
            print(f"Erreur chargement XTTS-v2: {e}")
            raise

    def get_voice_sample_path(self, voice_name: str) -> Optional[Path]:
        """Retourne le chemin du sample pour une voix donnée."""
        if voice_name in self.voices:
            return self.voices[voice_name].path

        # Chercher par pattern partiel
        for name, voice in self.voices.items():
            if voice_name.lower() in name.lower():
                return voice.path

        return None

    def get_voice_for_character(self, character: Character) -> Path:
        """Détermine le sample de voix pour un personnage."""
        # Vérifier le mapping manuel
        if character.name in self.config.character_voices:
            voice_name = self.config.character_voices[character.name]
            path = self.get_voice_sample_path(voice_name)
            if path:
                return path

        # Sélection automatique par genre
        if character.gender == "F":
            path = self.get_voice_sample_path(self.config.default_female_voice)
        elif character.gender == "M":
            path = self.get_voice_sample_path(self.config.default_male_voice)
        else:
            # Genre inconnu -> utiliser narrateur
            path = self.get_voice_sample_path(self.config.default_narrator_voice)

        return path or self.get_voice_sample_path(self.config.default_narrator_voice)

    def get_voice_for_segment(self, segment: DialogueSegment,
                              voice_assignments: Dict[str, Path]) -> Path:
        """Retourne le sample de voix pour un segment."""
        if segment.speaker_type == SpeakerType.NARRATOR:
            return self.get_voice_sample_path(self.config.default_narrator_voice)

        if segment.speaker in voice_assignments:
            return voice_assignments[segment.speaker]

        # Fallback vers narrateur
        return self.get_voice_sample_path(self.config.default_narrator_voice)

    def synthesize_segment(self, text: str, speaker_wav: Path) -> np.ndarray:
        """Synthétise un segment avec une voix spécifique."""
        self._init_tts()

        # Générer dans un fichier temporaire
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            self.tts.tts_to_file(
                text=text,
                file_path=temp_path,
                speaker_wav=str(speaker_wav),
                language=self.config.language
            )

            audio, sr = sf.read(temp_path)
            return audio, sr
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def synthesize_multivoice(
        self,
        text: str,
        output_path: str,
        character_mapping: Optional[Dict[str, str]] = None
    ) -> Dict:
        """
        Synthétise un texte avec plusieurs voix.

        Args:
            text: Texte à synthétiser
            output_path: Chemin du fichier de sortie
            character_mapping: Mapping personnage -> nom de voix

        Returns:
            Dictionnaire avec les informations de génération
        """
        self._init_tts()

        # Mettre à jour le mapping des personnages
        if character_mapping:
            self.config.character_voices.update(character_mapping)

        # Détecter les personnages et segmenter
        print("Analyse du texte...")
        detector = CharacterDetector(lang=self.config.language[:2])
        segments = detector.detect_dialogue_segments(text)
        characters = detector.get_characters()

        print(f"  Segments détectés: {len(segments)}")
        print(f"  Personnages: {[c.name for c in characters]}")

        # Assigner les voix aux personnages
        voice_assignments = {}
        for char in characters:
            voice_path = self.get_voice_for_character(char)
            voice_assignments[char.name] = voice_path
            print(f"  {char.name} ({char.gender or '?'}) -> {voice_path.name if voice_path else 'default'}")

        # Générer l'audio pour chaque segment
        print("\nGénération audio...")
        all_audio = []
        sample_rate = 24000

        for i, segment in enumerate(segments):
            speaker_wav = self.get_voice_for_segment(segment, voice_assignments)

            if not speaker_wav or not speaker_wav.exists():
                speaker_wav = self.get_voice_sample_path(self.config.default_narrator_voice)

            print(f"  [{i+1}/{len(segments)}] {segment.speaker_type.value}: "
                  f"{segment.speaker[:15]:15} | {segment.text[:40]}...")

            try:
                audio, sr = self.synthesize_segment(segment.text, speaker_wav)
                sample_rate = sr
                all_audio.append(audio)

                # Ajouter une petite pause entre segments
                pause = np.zeros(int(sr * 0.3))  # 300ms
                all_audio.append(pause)

            except Exception as e:
                print(f"    Erreur: {e}")
                # Continuer avec le segment suivant

        # Concatener et sauvegarder
        if all_audio:
            final_audio = np.concatenate(all_audio)
            sf.write(output_path, final_audio, sample_rate)

            duration = len(final_audio) / sample_rate
            print(f"\nAudio généré: {output_path}")
            print(f"  Durée: {duration/60:.1f} minutes")
            print(f"  Segments: {len(segments)}")

            return {
                "output_path": output_path,
                "duration": duration,
                "segments": len(segments),
                "characters": [c.name for c in characters],
                "voice_assignments": {k: str(v) for k, v in voice_assignments.items()}
            }
        else:
            raise RuntimeError("Aucun audio généré")

    def list_voices(self) -> Dict[str, List[str]]:
        """Liste les voix disponibles par catégorie."""
        result = {"narrator": [], "male": [], "female": []}

        for name, voice in self.voices.items():
            if voice.gender == "N":
                result["narrator"].append(name)
            elif voice.gender == "M":
                result["male"].append(name)
            elif voice.gender == "F":
                result["female"].append(name)

        return result


def create_multivoice_engine(
    samples_dir: str = "voice_samples",
    language: str = "fr",
    use_gpu: bool = False,
    character_voices: Optional[Dict[str, str]] = None
) -> MultiVoiceXTTSEngine:
    """Factory function pour créer un moteur multi-voix."""
    config = MultiVoiceConfig(
        samples_dir=Path(samples_dir),
        language=language,
        use_gpu=use_gpu,
        character_voices=character_voices or {}
    )
    return MultiVoiceXTTSEngine(config)


# Test
if __name__ == "__main__":
    # Texte de test avec dialogues
    test_text = """
    Victor entra dans la pièce enfumée. L'odeur du tabac lui piqua les yeux.

    « Tu es en retard, » dit Kamel avec un sourire froid.

    « J'avais des affaires à régler, » répondit Victor en s'asseyant.

    Marie apparut dans l'encadrement de la porte.

    « Messieurs, le dîner est servi, » annonça-t-elle d'une voix douce.

    Kamel se leva lentement. « On continue cette conversation plus tard. »
    """

    print("=== Test MultiVoice XTTS Engine ===\n")

    # Créer le moteur
    engine = create_multivoice_engine()

    print("\nVoix disponibles:")
    for category, voices in engine.list_voices().items():
        print(f"  {category}: {voices}")

    # Générer l'audio
    print("\n" + "="*50)
    result = engine.synthesize_multivoice(
        test_text,
        "output/test_multivoice.wav",
        character_mapping={
            "Kamel": "male_extended",
            "Marie": "female_narrator_01"
        }
    )

    print("\nRésultat:")
    for k, v in result.items():
        print(f"  {k}: {v}")
