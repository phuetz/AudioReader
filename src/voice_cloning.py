"""
Voice Cloning avec XTTS-v2.

Permet de cloner une voix a partir d'un echantillon audio (6+ secondes).
Supporte le clonage multilingue.

Prerequis:
    pip install TTS torch torchaudio
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import hashlib
import json


@dataclass
class ClonedVoice:
    """Voix clonee."""
    name: str
    source_audio: Path
    embedding_path: Optional[Path] = None
    language: str = "fr"
    description: str = ""
    duration_seconds: float = 0.0


class VoiceCloner:
    """
    Clonage de voix avec XTTS-v2.

    XTTS-v2 permet de cloner une voix avec seulement 6 secondes
    d'audio de reference.
    """

    MIN_AUDIO_DURATION = 6.0  # Minimum 6 secondes
    RECOMMENDED_DURATION = 30.0  # 30 secondes pour meilleure qualite

    def __init__(self, cache_dir: Path = Path(".voice_cache")):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._tts = None
        self._available = None

    def is_available(self) -> bool:
        """Verifie si XTTS est disponible."""
        if self._available is not None:
            return self._available

        try:
            from TTS.api import TTS
            self._available = True
        except ImportError:
            self._available = False

        return self._available

    def _load_model(self):
        """Charge le modele XTTS-v2."""
        if self._tts is not None:
            return self._tts

        if not self.is_available():
            raise RuntimeError("XTTS non disponible. Installez avec: pip install TTS")

        from TTS.api import TTS
        print("Chargement du modele XTTS-v2...")
        self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        return self._tts

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Retourne la duree d'un fichier audio."""
        try:
            import soundfile as sf
            info = sf.info(str(audio_path))
            return info.duration
        except Exception:
            return 0.0

    def _compute_audio_hash(self, audio_path: Path) -> str:
        """Calcule un hash du fichier audio."""
        hasher = hashlib.md5()
        with open(audio_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()[:12]

    def clone_voice(
        self,
        audio_path: Path,
        name: str,
        language: str = "fr",
        description: str = ""
    ) -> Optional[ClonedVoice]:
        """
        Clone une voix a partir d'un fichier audio.

        Args:
            audio_path: Chemin vers l'audio de reference (6+ sec)
            name: Nom pour cette voix clonee
            language: Code langue (fr, en, es, de, etc.)
            description: Description optionnelle

        Returns:
            ClonedVoice ou None si erreur
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            print(f"Erreur: Fichier non trouve: {audio_path}")
            return None

        # Verifier la duree
        duration = self._get_audio_duration(audio_path)
        if duration < self.MIN_AUDIO_DURATION:
            print(f"Erreur: Audio trop court ({duration:.1f}s). "
                  f"Minimum: {self.MIN_AUDIO_DURATION}s")
            return None

        if duration < self.RECOMMENDED_DURATION:
            print(f"Note: Audio de {duration:.1f}s. "
                  f"Recommande: {self.RECOMMENDED_DURATION}s pour meilleure qualite.")

        # Creer l'objet voix clonee
        voice = ClonedVoice(
            name=name,
            source_audio=audio_path,
            language=language,
            description=description,
            duration_seconds=duration
        )

        # Sauvegarder les metadonnees
        meta_path = self.cache_dir / f"{name}_meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                "name": name,
                "source_audio": str(audio_path),
                "language": language,
                "description": description,
                "duration_seconds": duration,
                "audio_hash": self._compute_audio_hash(audio_path),
            }, f, indent=2)

        print(f"Voix '{name}' enregistree ({duration:.1f}s)")
        return voice

    def synthesize(
        self,
        text: str,
        voice: ClonedVoice,
        output_path: Path,
        speed: float = 1.0
    ) -> bool:
        """
        Synthetise du texte avec une voix clonee.

        Args:
            text: Texte a synthetiser
            voice: Voix clonee a utiliser
            output_path: Chemin de sortie
            speed: Vitesse de lecture

        Returns:
            True si succes
        """
        if not self.is_available():
            print("XTTS non disponible")
            return False

        try:
            tts = self._load_model()

            # Synthetiser
            tts.tts_to_file(
                text=text,
                speaker_wav=str(voice.source_audio),
                language=voice.language,
                file_path=str(output_path),
                speed=speed
            )

            return True

        except Exception as e:
            print(f"Erreur synthese: {e}")
            return False

    def list_cloned_voices(self) -> List[ClonedVoice]:
        """Liste toutes les voix clonees en cache."""
        voices = []

        for meta_path in self.cache_dir.glob("*_meta.json"):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                voice = ClonedVoice(
                    name=data["name"],
                    source_audio=Path(data["source_audio"]),
                    language=data.get("language", "fr"),
                    description=data.get("description", ""),
                    duration_seconds=data.get("duration_seconds", 0.0)
                )
                voices.append(voice)
            except Exception:
                continue

        return voices

    def get_voice(self, name: str) -> Optional[ClonedVoice]:
        """Recupere une voix clonee par son nom."""
        meta_path = self.cache_dir / f"{name}_meta.json"

        if not meta_path.exists():
            return None

        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return ClonedVoice(
                name=data["name"],
                source_audio=Path(data["source_audio"]),
                language=data.get("language", "fr"),
                description=data.get("description", ""),
                duration_seconds=data.get("duration_seconds", 0.0)
            )
        except Exception:
            return None


class VoiceCloningManager:
    """
    Gestionnaire de voix clonees pour AudioReader.

    Integre les voix clonees avec les voix Kokoro.
    """

    def __init__(self, cache_dir: Path = Path(".voice_cache")):
        self.cloner = VoiceCloner(cache_dir)
        self._voice_registry: dict[str, ClonedVoice] = {}

    def register_cloned_voice(
        self,
        audio_path: Path,
        voice_id: str,
        language: str = "fr",
        description: str = ""
    ) -> bool:
        """
        Enregistre une nouvelle voix clonee.

        Args:
            audio_path: Audio de reference
            voice_id: ID unique pour cette voix (ex: "cloned_marie")
            language: Langue
            description: Description

        Returns:
            True si succes
        """
        voice = self.cloner.clone_voice(
            audio_path=audio_path,
            name=voice_id,
            language=language,
            description=description
        )

        if voice:
            self._voice_registry[voice_id] = voice
            return True
        return False

    def is_cloned_voice(self, voice_id: str) -> bool:
        """Verifie si une voix est une voix clonee."""
        return voice_id.startswith("cloned_") or voice_id in self._voice_registry

    def get_cloned_voice(self, voice_id: str) -> Optional[ClonedVoice]:
        """Recupere une voix clonee."""
        if voice_id in self._voice_registry:
            return self._voice_registry[voice_id]

        # Essayer de charger depuis le cache
        voice = self.cloner.get_voice(voice_id)
        if voice:
            self._voice_registry[voice_id] = voice
        return voice

    def synthesize_with_cloned_voice(
        self,
        text: str,
        voice_id: str,
        output_path: Path,
        speed: float = 1.0
    ) -> bool:
        """
        Synthetise avec une voix clonee.

        Args:
            text: Texte
            voice_id: ID de la voix clonee
            output_path: Sortie
            speed: Vitesse

        Returns:
            True si succes
        """
        voice = self.get_cloned_voice(voice_id)
        if not voice:
            print(f"Voix clonee non trouvee: {voice_id}")
            return False

        return self.cloner.synthesize(
            text=text,
            voice=voice,
            output_path=output_path,
            speed=speed
        )

    def list_all_voices(self) -> dict:
        """Liste toutes les voix (clonees et Kokoro)."""
        from .tts_kokoro_engine import KOKORO_VOICES

        result = {
            "kokoro": list(KOKORO_VOICES.keys()),
            "cloned": [v.name for v in self.cloner.list_cloned_voices()]
        }
        return result


def clone_voice_from_file(
    audio_path: str,
    voice_name: str,
    language: str = "fr"
) -> bool:
    """
    Fonction utilitaire pour cloner une voix.

    Usage:
        clone_voice_from_file("sample.wav", "ma_voix", "fr")
    """
    manager = VoiceCloningManager()
    return manager.register_cloned_voice(
        audio_path=Path(audio_path),
        voice_id=f"cloned_{voice_name}",
        language=language
    )


if __name__ == "__main__":
    print("=== Voice Cloning avec XTTS-v2 ===")
    print()
    print("Installation:")
    print("  pip install TTS torch torchaudio")
    print()
    print("Usage:")
    print("  from voice_cloning import clone_voice_from_file")
    print("  clone_voice_from_file('sample.wav', 'ma_voix', 'fr')")
    print()

    cloner = VoiceCloner()
    if cloner.is_available():
        print("XTTS-v2 est disponible!")
    else:
        print("XTTS-v2 n'est pas installe.")
        print("Installez avec: pip install TTS torch torchaudio")
