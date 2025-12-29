"""
Moteurs TTS Haute Qualité pour production audiobook professionnelle.

Modèles supportés:
- Chatterbox (ResembleAI) - Bat ElevenLabs, contrôle émotionnel
- F5-TTS - Meilleur voice cloning
- Orpheus TTS - Qualité humaine, streaming temps réel
"""
import asyncio
import os
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import warnings


@dataclass
class VoiceConfig:
    """Configuration de la voix pour la synthèse."""
    reference_audio: Optional[str] = None  # Chemin vers audio de référence pour clonage
    emotion: str = "neutral"  # neutral, happy, sad, angry, surprised
    emotion_strength: float = 0.5  # 0.0 à 1.0
    speed: float = 1.0  # 0.5 à 2.0

    # Tags paralinguistiques (Chatterbox)
    # Utiliser dans le texte: [laugh], [cough], [sigh], [gasp], [chuckle]


class HighQualityTTSEngine(ABC):
    """Interface pour les moteurs TTS haute qualité."""

    @abstractmethod
    def synthesize(self, text: str, output_path: Path, voice_config: VoiceConfig = None) -> bool:
        """Synthétise le texte en audio haute qualité."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Vérifie si le moteur est disponible."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Retourne le nom du moteur."""
        pass


class ChatterboxEngine(HighQualityTTSEngine):
    """
    Chatterbox TTS par ResembleAI.

    - Bat ElevenLabs dans les tests aveugles (63.75%)
    - Contrôle émotionnel avec exaggeration
    - Tags paralinguistiques: [laugh], [cough], [sigh], etc.
    - Voice cloning zero-shot

    Installation: pip install chatterbox-tts torch torchaudio
    Requires: ~8GB VRAM (GPU) ou CPU (plus lent)
    """

    def __init__(self, device: str = "auto"):
        """
        Args:
            device: "cuda", "cpu", ou "auto" (détection automatique)
        """
        self.device = device
        self._model = None
        self._available = None

    def _get_device(self):
        """Détermine le device à utiliser."""
        if self.device != "auto":
            return self.device

        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load_model(self):
        """Charge le modèle Chatterbox."""
        if self._model is not None:
            return self._model

        try:
            from chatterbox.tts import ChatterboxTTS
            import torch

            device = self._get_device()
            print(f"Chargement de Chatterbox sur {device}...")

            self._model = ChatterboxTTS.from_pretrained(device=device)
            return self._model

        except Exception as e:
            print(f"Erreur chargement Chatterbox: {e}")
            return None

    def is_available(self) -> bool:
        """Vérifie si Chatterbox est installé."""
        if self._available is not None:
            return self._available

        try:
            import chatterbox
            import torch
            import torchaudio
            self._available = True
        except ImportError:
            self._available = False

        return self._available

    def get_name(self) -> str:
        return "Chatterbox (ResembleAI)"

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice_config: VoiceConfig = None
    ) -> bool:
        """
        Synthétise le texte avec Chatterbox.

        Supporte les tags dans le texte:
        - [laugh], [chuckle] - rires
        - [cough], [sigh], [gasp] - sons
        """
        if not self.is_available():
            print("Chatterbox non disponible. Installer: pip install chatterbox-tts torch torchaudio")
            return False

        try:
            import torchaudio

            model = self._load_model()
            if model is None:
                return False

            voice_config = voice_config or VoiceConfig()

            # Paramètres de génération
            kwargs = {
                "exaggeration": voice_config.emotion_strength,
            }

            # Voice cloning si audio de référence fourni
            if voice_config.reference_audio and Path(voice_config.reference_audio).exists():
                audio_prompt, sr = torchaudio.load(voice_config.reference_audio)
                kwargs["audio_prompt"] = audio_prompt

            # Génération
            output = model.generate(text, **kwargs)

            # Sauvegarde
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Chatterbox output à 24kHz
            torchaudio.save(str(output_path), output.cpu(), 24000)

            return True

        except Exception as e:
            print(f"Erreur Chatterbox: {e}")
            return False


class F5TTSEngine(HighQualityTTSEngine):
    """
    F5-TTS - Meilleur voice cloning open source.

    - 335M paramètres
    - Clonage voix zero-shot exceptionnel
    - Support Anglais + Chinois (+ autres en dev)

    Installation:
        git clone https://github.com/SWivid/F5-TTS.git
        cd F5-TTS && pip install -e .

    Requires: ~6.4GB VRAM
    """

    def __init__(self, model_type: str = "F5-TTS"):
        """
        Args:
            model_type: "F5-TTS" ou "E2-TTS"
        """
        self.model_type = model_type
        self._model = None
        self._available = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available

        try:
            from f5_tts.api import F5TTS
            self._available = True
        except ImportError:
            self._available = False

        return self._available

    def get_name(self) -> str:
        return f"F5-TTS ({self.model_type})"

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from f5_tts.api import F5TTS

            print(f"Chargement de {self.model_type}...")
            self._model = F5TTS(model_type=self.model_type)
            return self._model

        except Exception as e:
            print(f"Erreur chargement F5-TTS: {e}")
            return None

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice_config: VoiceConfig = None
    ) -> bool:
        """
        Synthétise avec F5-TTS.

        IMPORTANT: Nécessite un audio de référence pour le clonage de voix.
        """
        if not self.is_available():
            print("F5-TTS non disponible. Voir: https://github.com/SWivid/F5-TTS")
            return False

        try:
            model = self._load_model()
            if model is None:
                return False

            voice_config = voice_config or VoiceConfig()

            if not voice_config.reference_audio:
                print("F5-TTS nécessite un audio de référence pour le clonage")
                return False

            # Génération avec clonage
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            model.infer(
                ref_file=voice_config.reference_audio,
                ref_text="",  # Transcription auto si vide
                gen_text=text,
                file_wave=str(output_path),
                speed=voice_config.speed
            )

            return True

        except Exception as e:
            print(f"Erreur F5-TTS: {e}")
            return False


class OrpheusTTSEngine(HighQualityTTSEngine):
    """
    Orpheus TTS - Qualité niveau humain.

    - 3B paramètres (versions 1B, 400M, 150M disponibles)
    - Contrôle émotion avec tags: <laugh>, <sigh>, <cough>
    - Streaming temps réel (~200ms latence)
    - Multilingue

    Installation: pip install orpheus-speech
    Requires: Variable selon taille modèle
    """

    def __init__(self, model_size: str = "medium"):
        """
        Args:
            model_size: "small" (150M), "medium" (400M), "large" (1B), "xlarge" (3B)
        """
        self.model_size = model_size
        self._model = None
        self._available = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available

        try:
            # Vérifier si orpheus est installé
            import importlib.util
            spec = importlib.util.find_spec("orpheus")
            self._available = spec is not None
        except:
            self._available = False

        return self._available

    def get_name(self) -> str:
        return f"Orpheus TTS ({self.model_size})"

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice_config: VoiceConfig = None
    ) -> bool:
        """
        Synthétise avec Orpheus TTS.

        Tags d'émotion supportés dans le texte:
        - <laugh>, <chuckle>
        - <sigh>, <cough>, <sniffle>
        - <groan>, <yawn>
        """
        if not self.is_available():
            print("Orpheus TTS non disponible. Voir: https://github.com/canopyai/Orpheus-TTS")
            return False

        try:
            from orpheus import OrpheusModel
            import soundfile as sf

            if self._model is None:
                print(f"Chargement Orpheus ({self.model_size})...")
                self._model = OrpheusModel(size=self.model_size)

            voice_config = voice_config or VoiceConfig()

            # Génération
            audio = self._model.generate(
                text,
                speaker=voice_config.reference_audio,
                speed=voice_config.speed
            )

            # Sauvegarde
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), audio, 24000)

            return True

        except Exception as e:
            print(f"Erreur Orpheus: {e}")
            return False


class HQAudioReader:
    """
    Convertisseur de livres avec TTS haute qualité.

    Sélectionne automatiquement le meilleur moteur disponible.
    Priorité: Chatterbox > Orpheus > F5-TTS > Edge-TTS (fallback)
    """

    def __init__(
        self,
        preferred_engine: str = "auto",
        voice_config: VoiceConfig = None
    ):
        """
        Args:
            preferred_engine: "chatterbox", "f5", "orpheus", "edge", ou "auto"
            voice_config: Configuration de la voix
        """
        self.voice_config = voice_config or VoiceConfig()
        self.engine = self._select_engine(preferred_engine)

    def _select_engine(self, preferred: str) -> HighQualityTTSEngine:
        """Sélectionne le moteur TTS."""
        engines = {
            "chatterbox": ChatterboxEngine,
            "f5": F5TTSEngine,
            "orpheus": OrpheusTTSEngine,
        }

        if preferred != "auto" and preferred in engines:
            engine = engines[preferred]()
            if engine.is_available():
                print(f"Moteur sélectionné: {engine.get_name()}")
                return engine
            else:
                print(f"{preferred} non disponible, recherche alternative...")

        # Sélection automatique par priorité
        priority = ["chatterbox", "orpheus", "f5"]

        for name in priority:
            engine = engines[name]()
            if engine.is_available():
                print(f"Moteur sélectionné: {engine.get_name()}")
                return engine

        # Fallback vers edge-tts
        print("Aucun moteur HQ disponible, fallback vers edge-tts")
        from .tts_engine import EdgeTTSEngine
        return EdgeTTSEngine()

    async def synthesize_chapter(
        self,
        text: str,
        output_path: Path,
        chapter_context: str = ""
    ) -> bool:
        """
        Synthétise un chapitre complet.

        Args:
            text: Texte du chapitre
            output_path: Chemin de sortie
            chapter_context: Contexte du chapitre (pour cohérence)
        """
        # Pour les engines HQ, on utilise la méthode sync
        if hasattr(self.engine, 'synthesize'):
            return self.engine.synthesize(text, output_path, self.voice_config)

        # Fallback async pour edge-tts
        return await self.engine.synthesize(text, output_path)

    def synthesize_chapter_sync(
        self,
        text: str,
        output_path: Path
    ) -> bool:
        """Version synchrone."""
        return asyncio.run(self.synthesize_chapter(text, output_path))


def get_installation_instructions() -> str:
    """Retourne les instructions d'installation pour chaque moteur."""
    return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    INSTALLATION MOTEURS HAUTE QUALITÉ                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  1. CHATTERBOX (Recommandé - Bat ElevenLabs)                                 ║
║     pip install chatterbox-tts torch torchaudio                              ║
║     Requires: ~8GB VRAM (GPU) ou CPU                                         ║
║     https://github.com/resemble-ai/chatterbox                                ║
║                                                                              ║
║  2. F5-TTS (Meilleur voice cloning)                                          ║
║     git clone https://github.com/SWivid/F5-TTS.git                           ║
║     cd F5-TTS && pip install -e .                                            ║
║     Requires: ~6.4GB VRAM                                                    ║
║                                                                              ║
║  3. ORPHEUS TTS (Qualité humaine)                                            ║
║     pip install orpheus-speech                                               ║
║     https://github.com/canopyai/Orpheus-TTS                                  ║
║                                                                              ║
║  4. TORTOISE TTS (Excellent mais très lent)                                  ║
║     pip install tortoise-tts                                                 ║
║     https://github.com/neonbjb/tortoise-tts                                  ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PLATEFORMES ACCEPTANT L'AI (pas Audible/ACX!)                               ║
║  • Google Play Books (auto-narration intégrée)                               ║
║  • Findaway Voices / Spotify (ElevenLabs depuis 02/2025)                     ║
║  • Kobo                                                                      ║
║  • Vente directe sur votre site                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    print(get_installation_instructions())

    # Test disponibilité
    print("\nVérification des moteurs disponibles:")
    for name, cls in [
        ("Chatterbox", ChatterboxEngine),
        ("F5-TTS", F5TTSEngine),
        ("Orpheus", OrpheusTTSEngine),
    ]:
        engine = cls()
        status = "✓ Disponible" if engine.is_available() else "✗ Non installé"
        print(f"  {name}: {status}")
