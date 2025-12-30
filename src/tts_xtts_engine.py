"""
Moteur TTS XTTS-v2 (Coqui) - Voix plus naturelles.

XTTS-v2 offre:
- Clonage de voix avec 6 secondes d'audio
- Prosodie plus naturelle
- Support multilingue natif
- Emotions plus expressives

Prerequis:
    pip install TTS torch torchaudio
"""
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union
from dataclasses import dataclass
import tempfile
import warnings

# Supprimer les warnings de torch
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class XTTSConfig:
    """Configuration du moteur XTTS."""
    # Modele a utiliser
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    # Utiliser le GPU si disponible
    use_gpu: bool = True
    # Langue par defaut
    default_language: str = "fr"
    # Vitesse de parole (0.5 a 2.0)
    speed: float = 1.0
    # Temperature pour la generation (0.1 a 1.0, plus bas = plus stable)
    temperature: float = 0.7
    # Repetition penalty
    repetition_penalty: float = 2.0
    # Top-k sampling
    top_k: int = 50
    # Top-p sampling
    top_p: float = 0.85
    # Activer le streaming
    enable_streaming: bool = False


class XTTSEngine:
    """
    Moteur TTS base sur XTTS-v2 de Coqui.

    Plus naturel que Kokoro, avec support du clonage de voix.
    """

    # Voix de reference par defaut (incluses avec XTTS)
    DEFAULT_SPEAKERS = {
        'fr_female': 'Claribel Dervla',
        'fr_male': 'Damien Black',
        'en_female': 'Sofia Hellen',
        'en_male': 'Edmund Lewis',
    }

    def __init__(self, config: Optional[XTTSConfig] = None):
        self.config = config or XTTSConfig()
        self._model = None
        self._device = None
        self._cloned_voices: Dict[str, str] = {}  # voice_id -> audio_path

    def _load_model(self):
        """Charge le modele XTTS."""
        if self._model is not None:
            return

        try:
            from TTS.api import TTS
            import torch

            print("Chargement du modele XTTS-v2...")

            # Determiner le device
            if self.config.use_gpu and torch.cuda.is_available():
                self._device = "cuda"
                print("  Utilisation du GPU")
            else:
                self._device = "cpu"
                print("  Utilisation du CPU")

            # Charger le modele
            self._model = TTS(self.config.model_name).to(self._device)

            print("  Modele XTTS-v2 charge")

        except ImportError:
            raise ImportError(
                "TTS n'est pas installe. Installation:\n"
                "pip install TTS torch torchaudio"
            )
        except Exception as e:
            raise RuntimeError(f"Erreur chargement XTTS: {e}")

    def register_voice(
        self,
        voice_id: str,
        audio_path: Union[str, Path],
        description: str = ""
    ) -> bool:
        """
        Enregistre une voix clonee.

        Args:
            voice_id: Identifiant de la voix
            audio_path: Chemin vers l'audio de reference (min 6s)
            description: Description de la voix

        Returns:
            True si l'enregistrement a reussi
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            print(f"Fichier audio non trouve: {audio_path}")
            return False

        self._cloned_voices[voice_id] = str(audio_path)
        print(f"Voix enregistree: {voice_id} -> {audio_path}")
        return True

    def synthesize(
        self,
        text: str,
        output_path: Union[str, Path],
        voice_id: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        language: Optional[str] = None,
        speed: Optional[float] = None
    ) -> bool:
        """
        Synthetise du texte en audio.

        Args:
            text: Texte a synthetiser
            output_path: Chemin de sortie
            voice_id: ID d'une voix enregistree
            speaker_wav: Chemin vers un audio de reference
            language: Code langue (fr, en, de, es, etc.)
            speed: Vitesse de parole

        Returns:
            True si la synthese a reussi
        """
        self._load_model()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        language = language or self.config.default_language
        speed = speed or self.config.speed

        # Determiner l'audio de reference
        if voice_id and voice_id in self._cloned_voices:
            speaker_wav = self._cloned_voices[voice_id]
        elif speaker_wav is None:
            # Utiliser une voix par defaut
            speaker_key = f"{language}_female"
            if speaker_key in self.DEFAULT_SPEAKERS:
                speaker = self.DEFAULT_SPEAKERS[speaker_key]
            else:
                speaker = list(self.DEFAULT_SPEAKERS.values())[0]
            speaker_wav = None  # Utiliser le speaker name

        try:
            # Generer l'audio
            if speaker_wav:
                # Clonage de voix
                self._model.tts_to_file(
                    text=text,
                    file_path=str(output_path),
                    speaker_wav=speaker_wav,
                    language=language,
                    speed=speed
                )
            else:
                # Voix par defaut du modele
                self._model.tts_to_file(
                    text=text,
                    file_path=str(output_path),
                    language=language,
                    speed=speed
                )

            return output_path.exists()

        except Exception as e:
            print(f"Erreur synthese XTTS: {e}")
            return False

    def synthesize_segments(
        self,
        segments: List[Dict],
        output_dir: Path,
        default_voice: Optional[str] = None
    ) -> List[Path]:
        """
        Synthetise plusieurs segments.

        Args:
            segments: Liste de dicts avec 'text', 'voice_id' (optionnel)
            output_dir: Dossier de sortie
            default_voice: Voix par defaut

        Returns:
            Liste des chemins audio generes
        """
        self._load_model()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_files = []

        for i, segment in enumerate(segments):
            text = segment.get('text', '')
            voice_id = segment.get('voice_id', default_voice)
            language = segment.get('language', self.config.default_language)

            if not text.strip():
                continue

            output_path = output_dir / f"segment_{i:04d}.wav"

            print(f"  [{i+1}/{len(segments)}] {text[:40]}...")

            if self.synthesize(text, output_path, voice_id=voice_id, language=language):
                audio_files.append(output_path)

        return audio_files

    def synthesize_chapter(
        self,
        text: str,
        output_path: Union[str, Path],
        voice_id: Optional[str] = None,
        language: Optional[str] = None,
        chunk_size: int = 250
    ) -> bool:
        """
        Synthetise un chapitre complet.

        Args:
            text: Texte du chapitre
            output_path: Chemin de sortie
            voice_id: Voix a utiliser
            language: Langue
            chunk_size: Taille max des chunks

        Returns:
            True si la synthese a reussi
        """
        import soundfile as sf
        from .audio_crossfade import apply_crossfade_to_chapter

        self._load_model()

        output_path = Path(output_path)
        language = language or self.config.default_language

        # Decouper le texte en chunks
        chunks = self._split_text(text, chunk_size)

        if not chunks:
            return False

        print(f"Synthese XTTS: {len(chunks)} segments")

        # Generer chaque chunk
        audio_segments = []
        sample_rate = 24000  # XTTS genere en 24kHz

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            for i, chunk in enumerate(chunks):
                temp_path = temp_dir / f"chunk_{i:04d}.wav"

                print(f"  [{i+1}/{len(chunks)}] {chunk[:40]}...")

                if self.synthesize(chunk, temp_path, voice_id=voice_id, language=language):
                    audio, sr = sf.read(str(temp_path))
                    audio_segments.append(audio.astype(np.float32))
                    sample_rate = sr

            if not audio_segments:
                return False

            # Appliquer le crossfade
            print("Application du crossfade...")
            final_audio = apply_crossfade_to_chapter(audio_segments, sample_rate)

            # Normaliser
            max_val = np.max(np.abs(final_audio))
            if max_val > 0:
                final_audio = (final_audio / max_val * 0.9).astype(np.float32)

            # Sauvegarder
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), final_audio, sample_rate)

            duration = len(final_audio) / sample_rate
            print(f"Audio genere: {output_path} ({duration:.1f}s)")

            return True

    def _split_text(self, text: str, max_chars: int = 250) -> List[str]:
        """Decoupe le texte en chunks."""
        import re

        # Separer par phrases
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) < max_chars:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def list_available_voices(self) -> Dict[str, List[str]]:
        """Liste les voix disponibles."""
        voices = {
            'default': list(self.DEFAULT_SPEAKERS.keys()),
            'cloned': list(self._cloned_voices.keys())
        }
        return voices


def create_xtts_engine(
    language: str = "fr",
    use_gpu: bool = True
) -> XTTSEngine:
    """
    Cree un moteur XTTS configure.

    Args:
        language: Langue par defaut
        use_gpu: Utiliser le GPU

    Returns:
        Instance XTTSEngine
    """
    config = XTTSConfig(
        default_language=language,
        use_gpu=use_gpu
    )
    return XTTSEngine(config)


# Test
if __name__ == "__main__":
    print("Test du moteur XTTS-v2")
    print("=" * 50)

    try:
        engine = create_xtts_engine("fr")

        test_text = """
        L'argent a une odeur. Celle de la sueur des autres.
        Victor le savait depuis son enfance a Saint-Denis.
        """

        output = Path("output/test_xtts.wav")
        success = engine.synthesize(test_text, output)

        if success:
            print(f"Audio genere: {output}")
        else:
            print("Echec de la generation")

    except ImportError as e:
        print(f"Dependance manquante: {e}")
    except Exception as e:
        print(f"Erreur: {e}")
