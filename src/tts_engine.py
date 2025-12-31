"""
Moteur TTS unifié pour AudioReader.

Sélectionne automatiquement le meilleur moteur selon la langue:
- Français (fr): MMS-TTS (Meta) - qualité native française
- Anglais (en): Kokoro - voix expressives
- Autres: MMS-TTS (1000+ langues supportées)

Fallback: Edge-TTS (online) ou pyttsx3 (offline)

Usage:
    from src.tts_engine import create_tts_engine

    # Création automatique selon langue
    engine = create_tts_engine(language="fr")
    engine.synthesize("Bonjour!", "output.wav")
"""

import asyncio
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from enum import Enum


class EngineType(Enum):
    """Types de moteurs TTS disponibles."""
    KOKORO = "kokoro"
    MMS = "mms"
    XTTS = "xtts"
    EDGE = "edge"
    PYTTSX3 = "pyttsx3"
    AUTO = "auto"


# Mapping langue -> meilleur moteur
LANGUAGE_ENGINE_MAP = {
    # Français -> MMS (meilleure qualité native)
    "fr": EngineType.MMS,
    "fra": EngineType.MMS,
    "french": EngineType.MMS,
    "français": EngineType.MMS,

    # Anglais -> Kokoro (voix expressives)
    "en": EngineType.KOKORO,
    "eng": EngineType.KOKORO,
    "english": EngineType.KOKORO,

    # Autres langues -> MMS (large couverture)
    "de": EngineType.MMS,
    "es": EngineType.MMS,
    "it": EngineType.MMS,
    "pt": EngineType.MMS,
}


class TTSEngineBase(ABC):
    """Interface abstraite pour les moteurs TTS."""

    @abstractmethod
    def synthesize(self, text: str, output_path: Path, **kwargs) -> bool:
        """Synthétise le texte en fichier audio."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Vérifie si le moteur est disponible."""
        pass

    def get_info(self) -> dict:
        """Retourne des informations sur le moteur."""
        return {"engine": self.__class__.__name__}


class UnifiedTTSEngine:
    """
    Moteur TTS unifié avec sélection automatique.

    Choisit le meilleur moteur selon la langue:
    - Français: MMS-TTS (Meta)
    - Anglais: Kokoro
    - Autres: MMS-TTS
    """

    def __init__(
        self,
        engine_type: str = "auto",
        language: str = "fr",
        voice: str = "ff_siwis",
        speed: float = 1.0,
        **kwargs
    ):
        """
        Initialise le moteur TTS.

        Args:
            engine_type: "auto", "kokoro", "mms", "xtts", "edge"
            language: Code langue (fr, en, de, etc.)
            voice: Voix (pour Kokoro uniquement)
            speed: Vitesse de lecture
        """
        self.language = language.lower()
        self.voice = voice
        self.speed = speed
        self._kwargs = kwargs

        # Déterminer le type de moteur
        if engine_type == "auto":
            self._engine_type = self._select_engine(self.language)
        else:
            self._engine_type = EngineType(engine_type)

        # Créer le moteur
        self._engine = None
        self._create_engine()

    def _select_engine(self, language: str) -> EngineType:
        """Sélectionne le meilleur moteur pour la langue."""
        return LANGUAGE_ENGINE_MAP.get(language, EngineType.MMS)

    def _create_engine(self):
        """Crée l'instance du moteur."""
        # Helper pour import robuste (relatif ou absolu)
        def robust_import(module_name, class_name):
            try:
                module = __import__(module_name, fromlist=[class_name])
                return getattr(module, class_name)
            except ImportError:
                # Essayer import relatif (si package)
                try:
                    module = __import__(f".{module_name}", fromlist=[class_name], package="src")
                    return getattr(module, class_name)
                except ImportError:
                    # Essayer import direct (si src/ est dans sys.path)
                    try:
                        # Cas spécifique où src est dans le path
                        import sys
                        if "src" not in sys.modules:
                            # Tenter un import sans prefixe
                            module = __import__(module_name, fromlist=[class_name])
                            return getattr(module, class_name)
                    except:
                        pass
                    raise

        if self._engine_type == EngineType.KOKORO:
            try:
                KokoroEngine = robust_import("tts_kokoro_engine", "KokoroEngine")
                self._engine = KokoroEngine(
                    voice=self.voice,
                    speed=self.speed,
                    **self._kwargs
                )
            except Exception as e:
                print(f"Kokoro non disponible: {e}, fallback MMS")
                self._engine_type = EngineType.MMS
                self._create_engine()

        elif self._engine_type == EngineType.MMS:
            try:
                MMSTTSEngine = robust_import("tts_mms_engine", "MMSTTSEngine")
                lang_map = {"fr": "fra", "en": "eng", "de": "deu",
                           "es": "spa", "it": "ita", "pt": "por"}
                mms_lang = lang_map.get(self.language, self.language)
                self._engine = MMSTTSEngine(
                    language=mms_lang,
                    speed=self.speed,
                    **self._kwargs
                )
            except Exception as e:
                print(f"MMS non disponible: {e}, fallback Edge")
                self._engine_type = EngineType.EDGE
                self._create_engine()

        elif self._engine_type == EngineType.XTTS:
            try:
                XTTSEngine = robust_import("tts_xtts_engine", "XTTSEngine")
                XTTSConfig = robust_import("tts_xtts_engine", "XTTSConfig")
                
                config = XTTSConfig(
                    default_language=self.language,
                    speed=self.speed
                )
                self._engine = XTTSEngine(config)
            except Exception as e:
                print(f"XTTS non disponible: {e}, fallback MMS")
                self._engine_type = EngineType.MMS
                self._create_engine()

        elif self._engine_type == EngineType.EDGE:
            self._engine = EdgeTTSEngine(voice=self.voice)

        else:
            self._engine = Pyttsx3Engine()

    def is_available(self) -> bool:
        """Vérifie si le moteur est disponible."""
        if self._engine is None:
            return False
        return self._engine.is_available()

    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        speaker_wav: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Synthétise du texte en audio.

        Args:
            text: Texte à synthétiser
            output_path: Chemin du fichier de sortie
            voice: Voix (optionnel)
            speed: Vitesse (optionnel)
            speaker_wav: Chemin vers un fichier audio pour clonage (XTTS)

        Returns:
            True si succès
        """
        if self._engine is None:
            print("Aucun moteur TTS disponible")
            return False

        # Si le moteur supporte le clonage (XTTS), passer speaker_wav
        if self._engine_type == EngineType.XTTS:
             if speaker_wav:
                 kwargs['speaker_wav'] = speaker_wav
             # XTTS utilise voice_id au lieu de voice
             if voice:
                 kwargs['voice_id'] = voice
             
             # Appeler sans l'argument 'voice' qui n'existe pas dans XTTS
             return self._engine.synthesize(
                text=text,
                output_path=Path(output_path),
                speed=speed,
                **kwargs
            )

        return self._engine.synthesize(
            text=text,
            output_path=Path(output_path),
            voice=voice,
            speed=speed,
            **kwargs
        )

    def get_info(self) -> dict:
        """Retourne des informations sur le moteur."""
        info = {
            "wrapper": "UnifiedTTSEngine",
            "engine_type": self._engine_type.value,
            "language": self.language,
        }
        if self._engine and hasattr(self._engine, 'get_info'):
            info.update(self._engine.get_info())
        return info

    @property
    def sample_rate(self) -> int:
        """Retourne le sample rate du moteur."""
        if self._engine and hasattr(self._engine, 'sample_rate'):
            return self._engine.sample_rate
        return 24000


# === Anciens moteurs (Edge-TTS, pyttsx3) ===

class EdgeTTSEngine(TTSEngineBase):
    """Moteur TTS utilisant Microsoft Edge TTS (gratuit, online)."""

    FRENCH_VOICES = {
        "fr-FR-DeniseNeural": "Femme (France)",
        "fr-FR-HenriNeural": "Homme (France)",
        "fr-CA-SylvieNeural": "Femme (Canada)",
    }

    def __init__(self, voice: str = "fr-FR-DeniseNeural", rate: str = "+0%"):
        self.voice = voice
        self.rate = rate

    def is_available(self) -> bool:
        try:
            import edge_tts
            return True
        except ImportError:
            return False

    def synthesize(self, text: str, output_path: Path, **kwargs) -> bool:
        try:
            import edge_tts
            async def _synth():
                communicate = edge_tts.Communicate(text=text, voice=self.voice, rate=self.rate)
                await communicate.save(str(output_path))
            asyncio.run(_synth())
            return True
        except Exception as e:
            print(f"Erreur Edge-TTS: {e}")
            return False


class Pyttsx3Engine(TTSEngineBase):
    """Moteur TTS utilisant pyttsx3 (offline, qualité basique)."""

    def __init__(self, rate: int = 150):
        self.rate = rate
        self._engine = None

    def is_available(self) -> bool:
        try:
            import pyttsx3
            return True
        except ImportError:
            return False

    def synthesize(self, text: str, output_path: Path, **kwargs) -> bool:
        try:
            import pyttsx3
            if self._engine is None:
                self._engine = pyttsx3.init()
                self._engine.setProperty('rate', self.rate)
            self._engine.save_to_file(text, str(output_path))
            self._engine.runAndWait()
            return True
        except Exception as e:
            print(f"Erreur pyttsx3: {e}")
            return False


# === Factory function ===

def create_tts_engine(
    language: str = "fr",
    engine_type: str = "auto",
    **kwargs
) -> UnifiedTTSEngine:
    """
    Crée un moteur TTS optimisé pour la langue.

    Args:
        language: Code langue (fr, en, de, es, etc.)
        engine_type: "auto", "kokoro", "mms", "edge"
        **kwargs: Arguments supplémentaires

    Returns:
        Instance UnifiedTTSEngine

    Examples:
        # Français (utilise MMS automatiquement)
        engine = create_tts_engine("fr")

        # Anglais (utilise Kokoro automatiquement)
        engine = create_tts_engine("en")

        # Forcer un moteur spécifique
        engine = create_tts_engine("fr", engine_type="kokoro")
    """
    return UnifiedTTSEngine(
        engine_type=engine_type,
        language=language,
        **kwargs
    )


# === Alias pour compatibilité ===
TTSEngine = UnifiedTTSEngine


if __name__ == "__main__":
    print("=== Test Moteur TTS Unifié ===\n")

    # Test français (MMS)
    print("1. Test Français (auto -> MMS)")
    print("-" * 40)
    engine_fr = create_tts_engine("fr")
    print(f"Info: {engine_fr.get_info()}")

    if engine_fr.is_available():
        text_fr = "Bonjour! Ceci est un test en français."
        engine_fr.synthesize(text_fr, Path("output/test_unified_fr.wav"))

    print("\n✅ Test terminé")
