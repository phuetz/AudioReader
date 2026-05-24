"""
Wrapper TTS unifié - Kokoro + Edge-TTS.

Sélection automatique du meilleur moteur selon la langue:
- Français: Edge-TTS (Microsoft) - meilleure qualité
- Anglais: Kokoro - rapide et offline
- Autres: Configurable

Usage:
    from src.tts_unified import UnifiedTTS

    tts = UnifiedTTS()
    audio = tts.synthesize("Bonjour le monde!", lang="fr")
    audio = tts.synthesize("Hello world!", lang="en")
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class TTSEngine(Enum):
    """Moteurs TTS disponibles."""
    KOKORO = "kokoro"
    EDGE_TTS = "edge_tts"
    CHATTERBOX = "chatterbox"  # Resemble AI — bat ElevenLabs
    ORPHEUS = "orpheus"  # Canopy Labs — emotion naturelle
    PARLER = "parler"  # Hugging Face — description-based, multilingual
    QWEN3 = "qwen3"  # Alibaba Qwen — 10 langues, clonage 3s, instruct
    VOXTRAL = "voxtral"  # Mistral AI — cloud/local, 9 langues, clonage 2s
    AUTO = "auto"  # Selection automatique selon la langue


@dataclass
class TTSVoice:
    """Configuration d'une voix TTS."""
    id: str
    name: str
    engine: TTSEngine
    lang: str
    gender: str  # "male" ou "female"
    description: str = ""


@dataclass
class TTSConfig:
    """Configuration du système TTS unifié."""
    # Moteur par défaut
    default_engine: TTSEngine = TTSEngine.AUTO

    # Préférences par langue (pour mode AUTO)
    lang_preferences: Dict[str, TTSEngine] = field(default_factory=lambda: {
        "fr": TTSEngine.EDGE_TTS,  # Français -> Edge-TTS (meilleure qualité)
        "en": TTSEngine.KOKORO,    # Anglais -> Kokoro (rapide, offline)
    })

    # Voix par défaut par langue
    default_voices: Dict[str, str] = field(default_factory=lambda: {
        "fr": "fr-FR-DeniseNeural",      # Edge-TTS français femme
        "fr-male": "fr-FR-HenriNeural",  # Edge-TTS français homme
        "en": "af_bella",                 # Kokoro anglais femme
        "en-male": "am_adam",             # Kokoro anglais homme
    })

    # Vitesse par défaut
    speed: float = 1.0

    # Utiliser le préprocesseur français
    use_french_preprocessor: bool = True

    # Sample rate de sortie
    sample_rate: int = 24000


# Voix disponibles
AVAILABLE_VOICES: Dict[str, TTSVoice] = {
    # === KOKORO VOICES ===
    # Français
    "ff_siwis": TTSVoice("ff_siwis", "Siwis", TTSEngine.KOKORO, "fr", "female", "Kokoro French Female"),

    # Anglais US - Femmes
    "af_bella": TTSVoice("af_bella", "Bella", TTSEngine.KOKORO, "en-us", "female", "Soft American"),
    "af_heart": TTSVoice("af_heart", "Heart", TTSEngine.KOKORO, "en-us", "female", "Warm American"),
    "af_nicole": TTSVoice("af_nicole", "Nicole", TTSEngine.KOKORO, "en-us", "female", "Clear American"),
    "af_sky": TTSVoice("af_sky", "Sky", TTSEngine.KOKORO, "en-us", "female", "Young American"),

    # Anglais US - Hommes
    "am_adam": TTSVoice("am_adam", "Adam", TTSEngine.KOKORO, "en-us", "male", "Deep American"),
    "am_michael": TTSVoice("am_michael", "Michael", TTSEngine.KOKORO, "en-us", "male", "Warm American"),

    # Anglais UK
    "bf_emma": TTSVoice("bf_emma", "Emma", TTSEngine.KOKORO, "en-gb", "female", "British Female"),
    "bm_george": TTSVoice("bm_george", "George", TTSEngine.KOKORO, "en-gb", "male", "British Male"),

    # === EDGE-TTS VOICES ===
    # Français France
    "fr-FR-DeniseNeural": TTSVoice("fr-FR-DeniseNeural", "Denise", TTSEngine.EDGE_TTS, "fr", "female", "Microsoft French Female"),
    "fr-FR-HenriNeural": TTSVoice("fr-FR-HenriNeural", "Henri", TTSEngine.EDGE_TTS, "fr", "male", "Microsoft French Male"),
    "fr-FR-EloiseNeural": TTSVoice("fr-FR-EloiseNeural", "Eloise", TTSEngine.EDGE_TTS, "fr", "female", "Microsoft French Female (Child)"),

    # Français Canada
    "fr-CA-SylvieNeural": TTSVoice("fr-CA-SylvieNeural", "Sylvie", TTSEngine.EDGE_TTS, "fr-ca", "female", "Microsoft Quebec Female"),
    "fr-CA-JeanNeural": TTSVoice("fr-CA-JeanNeural", "Jean", TTSEngine.EDGE_TTS, "fr-ca", "male", "Microsoft Quebec Male"),

    # Français Belgique
    "fr-BE-CharlineNeural": TTSVoice("fr-BE-CharlineNeural", "Charline", TTSEngine.EDGE_TTS, "fr-be", "female", "Microsoft Belgian Female"),

    # Anglais US (Edge-TTS)
    "en-US-JennyNeural": TTSVoice("en-US-JennyNeural", "Jenny", TTSEngine.EDGE_TTS, "en-us", "female", "Microsoft US Female"),
    "en-US-GuyNeural": TTSVoice("en-US-GuyNeural", "Guy", TTSEngine.EDGE_TTS, "en-us", "male", "Microsoft US Male"),

    # Anglais UK (Edge-TTS)
    "en-GB-SoniaNeural": TTSVoice("en-GB-SoniaNeural", "Sonia", TTSEngine.EDGE_TTS, "en-gb", "female", "Microsoft UK Female"),
    "en-GB-RyanNeural": TTSVoice("en-GB-RyanNeural", "Ryan", TTSEngine.EDGE_TTS, "en-gb", "male", "Microsoft UK Male"),
}


class KokoroTTSBackend:
    """Backend Kokoro TTS."""

    def __init__(self, model_path: str = "kokoro-v1.0.onnx", voices_path: str = "voices-v1.0.bin"):
        self.model_path = Path(model_path)
        self.voices_path = Path(voices_path)
        self._kokoro = None
        self._available = None

    def is_available(self) -> bool:
        """Vérifie si Kokoro est disponible."""
        if self._available is not None:
            return self._available

        try:
            import kokoro_onnx
            self._available = self.model_path.exists() and self.voices_path.exists()
        except ImportError:
            self._available = False

        return self._available

    def _load(self):
        """Charge le modèle Kokoro."""
        if self._kokoro is None:
            from kokoro_onnx import Kokoro
            self._kokoro = Kokoro(str(self.model_path), str(self.voices_path))
        return self._kokoro

    def synthesize(
        self,
        text: str,
        voice: str = "ff_siwis",
        speed: float = 1.0,
        lang: str = "fr-fr"
    ) -> Tuple[np.ndarray, int]:
        """
        Synthétise du texte en audio.

        Returns:
            Tuple (audio_array, sample_rate)
        """
        kokoro = self._load()

        # Mapper la langue
        lang_map = {
            "fr": "fr-fr",
            "en": "en-us",
            "en-us": "en-us",
            "en-gb": "en-gb",
        }
        lang = lang_map.get(lang, lang)

        samples, sr = kokoro.create(text, voice=voice, speed=speed, lang=lang)
        return samples, sr


class EdgeTTSBackend:
    """Backend Edge-TTS (Microsoft)."""

    def __init__(self):
        self._available = None

    def is_available(self) -> bool:
        """Vérifie si Edge-TTS est disponible."""
        if self._available is not None:
            return self._available

        try:
            import edge_tts
            self._available = True
        except ImportError:
            self._available = False

        return self._available

    def synthesize(
        self,
        text: str,
        voice: str = "fr-FR-DeniseNeural",
        speed: float = 1.0
    ) -> Tuple[np.ndarray, int]:
        """
        Synthétise du texte en audio.

        Returns:
            Tuple (audio_array, sample_rate)
        """
        import edge_tts
        from pydub import AudioSegment
        import io

        # Convertir speed en format Edge-TTS (+X% ou -X%)
        speed_percent = int((speed - 1.0) * 100)
        if speed_percent >= 0:
            rate = f"+{speed_percent}%"
        else:
            rate = f"{speed_percent}%"

        # Créer la communication
        communicate = edge_tts.Communicate(text, voice, rate=rate)

        # Synthétiser dans un buffer
        async def _synthesize():
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            return audio_data

        # Exécuter async (thread-safe)
        try:
            loop = asyncio.get_running_loop()
            # On est dans un contexte async — lancer dans un nouveau thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(lambda: asyncio.run(_synthesize()))
                audio_bytes = future.result(timeout=120)
        except RuntimeError:
            # Pas de boucle en cours, utiliser asyncio.run()
            audio_bytes = asyncio.run(_synthesize())

        # Convertir MP3 en numpy array
        audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))

        # Convertir en mono si nécessaire
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)

        # Normaliser à 24kHz pour cohérence avec Kokoro
        audio_segment = audio_segment.set_frame_rate(24000)

        # Convertir en numpy array
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        # Normaliser selon la profondeur de bits réelle
        bit_depth = audio_segment.sample_width * 8
        samples = samples / (2 ** (bit_depth - 1))  # int16->32768, int24->8388608, etc.

        return samples, 24000

    async def synthesize_async(
        self,
        text: str,
        voice: str = "fr-FR-DeniseNeural",
        speed: float = 1.0
    ) -> Tuple[np.ndarray, int]:
        """Version asynchrone de synthesize."""
        import edge_tts
        from pydub import AudioSegment
        import io

        speed_percent = int((speed - 1.0) * 100)
        rate = f"+{speed_percent}%" if speed_percent >= 0 else f"{speed_percent}%"

        communicate = edge_tts.Communicate(text, voice, rate=rate)

        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]

        audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
        if audio_segment.channels > 1:
            audio_segment = audio_segment.set_channels(1)
        audio_segment = audio_segment.set_frame_rate(24000)

        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        samples = samples / 32768.0

        return samples, 24000


class UnifiedTTS:
    """
    Système TTS unifié avec sélection automatique du meilleur moteur.

    Usage:
        tts = UnifiedTTS()

        # Sélection automatique (Edge-TTS pour FR, Kokoro pour EN)
        audio, sr = tts.synthesize("Bonjour!", lang="fr")
        audio, sr = tts.synthesize("Hello!", lang="en")

        # Forcer un moteur spécifique
        audio, sr = tts.synthesize("Bonjour!", engine=TTSEngine.KOKORO)

        # Voix spécifique
        audio, sr = tts.synthesize("Bonjour!", voice="fr-FR-HenriNeural")
    """

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()

        # Initialiser les backends
        self._kokoro = KokoroTTSBackend()
        self._edge_tts = EdgeTTSBackend()
        self._chatterbox = None  # Lazy loading (GPU lourd)
        self._orpheus = None  # Lazy loading (GPU lourd)
        self._parler = None  # Lazy loading (GPU lourd)
        self._qwen3 = None  # Lazy loading (GPU lourd)
        self._voxtral = None  # Lazy loading (API-based)

        # Préprocesseur français
        self._french_preprocessor = None
        if self.config.use_french_preprocessor:
            try:
                from src.french_preprocessor import FrenchTextPreprocessor
                self._french_preprocessor = FrenchTextPreprocessor()
            except ImportError:
                pass

    def _get_chatterbox(self):
        """Retourne le backend Chatterbox (lazy loading)."""
        if self._chatterbox is None:
            try:
                from src.tts_chatterbox_engine import ChatterboxEngine
                self._chatterbox = ChatterboxEngine()
            except ImportError:
                try:
                    from tts_chatterbox_engine import ChatterboxEngine
                    self._chatterbox = ChatterboxEngine()
                except ImportError:
                    return None
        return self._chatterbox

    def _get_orpheus(self):
        """Retourne le backend Orpheus (lazy loading)."""
        if self._orpheus is None:
            try:
                from src.tts_orpheus_engine import OrpheusEngine
                self._orpheus = OrpheusEngine()
            except ImportError:
                try:
                    from tts_orpheus_engine import OrpheusEngine
                    self._orpheus = OrpheusEngine()
                except ImportError:
                    return None
        return self._orpheus

    def _get_parler(self):
        """Retourne le backend Parler TTS (lazy loading)."""
        if self._parler is None:
            try:
                from src.tts_parler_engine import ParlerEngine
                self._parler = ParlerEngine()
            except ImportError:
                try:
                    from tts_parler_engine import ParlerEngine
                    self._parler = ParlerEngine()
                except ImportError:
                    return None
        return self._parler

    def _get_qwen3(self):
        """Retourne le backend Qwen3-TTS (lazy loading)."""
        if self._qwen3 is None:
            try:
                from src.tts_qwen3_engine import Qwen3Engine
                self._qwen3 = Qwen3Engine()
            except ImportError:
                try:
                    from tts_qwen3_engine import Qwen3Engine
                    self._qwen3 = Qwen3Engine()
                except ImportError:
                    return None
        return self._qwen3

    def _get_voxtral(self):
        """Retourne le backend Voxtral (lazy loading)."""
        if self._voxtral is None:
            try:
                from src.tts_voxtral_engine import VoxtralEngine
                self._voxtral = VoxtralEngine()
            except ImportError:
                try:
                    from tts_voxtral_engine import VoxtralEngine
                    self._voxtral = VoxtralEngine()
                except ImportError:
                    return None
        return self._voxtral

    def get_available_engines(self) -> List[TTSEngine]:
        """Retourne la liste des moteurs disponibles."""
        engines = []
        if self._kokoro.is_available():
            engines.append(TTSEngine.KOKORO)
        if self._edge_tts.is_available():
            engines.append(TTSEngine.EDGE_TTS)
        cb = self._get_chatterbox()
        if cb and cb.is_available():
            engines.append(TTSEngine.CHATTERBOX)
        orph = self._get_orpheus()
        if orph and orph.is_available():
            engines.append(TTSEngine.ORPHEUS)
        parler = self._get_parler()
        if parler and parler.is_available():
            engines.append(TTSEngine.PARLER)
        qwen3 = self._get_qwen3()
        if qwen3 and qwen3.is_available():
            engines.append(TTSEngine.QWEN3)
        voxtral = self._get_voxtral()
        if voxtral and voxtral.is_available():
            engines.append(TTSEngine.VOXTRAL)
        return engines

    def get_voices(self, lang: Optional[str] = None, engine: Optional[TTSEngine] = None) -> List[TTSVoice]:
        """
        Retourne les voix disponibles.

        Args:
            lang: Filtrer par langue (ex: "fr", "en")
            engine: Filtrer par moteur
        """
        voices = []
        for voice in AVAILABLE_VOICES.values():
            # Filtrer par langue
            if lang and not voice.lang.startswith(lang):
                continue

            # Filtrer par moteur
            if engine and voice.engine != engine:
                continue

            # Vérifier disponibilité
            if voice.engine == TTSEngine.KOKORO and not self._kokoro.is_available():
                continue
            if voice.engine == TTSEngine.EDGE_TTS and not self._edge_tts.is_available():
                continue

            voices.append(voice)

        return voices

    def _select_engine(self, lang: str, engine: Optional[TTSEngine] = None) -> TTSEngine:
        """Sélectionne le meilleur moteur pour une langue."""
        if engine and engine != TTSEngine.AUTO:
            return engine

        # Mode AUTO: utiliser les préférences par langue
        lang_base = lang.split("-")[0]  # "fr-FR" -> "fr"

        preferred = self.config.lang_preferences.get(lang_base, TTSEngine.KOKORO)

        # Vérifier disponibilité du moteur prefere
        if preferred == TTSEngine.CHATTERBOX:
            cb = self._get_chatterbox()
            if cb and cb.is_available():
                return TTSEngine.CHATTERBOX
        if preferred == TTSEngine.ORPHEUS:
            orph = self._get_orpheus()
            if orph and orph.is_available():
                return TTSEngine.ORPHEUS
        if preferred == TTSEngine.PARLER:
            parler = self._get_parler()
            if parler and parler.is_available():
                return TTSEngine.PARLER
        if preferred == TTSEngine.QWEN3:
            qwen3 = self._get_qwen3()
            if qwen3 and qwen3.is_available():
                return TTSEngine.QWEN3
        if preferred == TTSEngine.VOXTRAL:
            voxtral = self._get_voxtral()
            if voxtral and voxtral.is_available():
                return TTSEngine.VOXTRAL
        if preferred == TTSEngine.EDGE_TTS and self._edge_tts.is_available():
            return TTSEngine.EDGE_TTS
        if preferred == TTSEngine.KOKORO and self._kokoro.is_available():
            return TTSEngine.KOKORO

        # Fallback: Chatterbox > Kokoro > Edge-TTS > Orpheus
        cb = self._get_chatterbox()
        if cb and cb.is_available():
            return TTSEngine.CHATTERBOX
        if self._kokoro.is_available():
            return TTSEngine.KOKORO
        if self._edge_tts.is_available():
            return TTSEngine.EDGE_TTS
        orph = self._get_orpheus()
        if orph and orph.is_available():
            return TTSEngine.ORPHEUS
        parler = self._get_parler()
        if parler and parler.is_available():
            return TTSEngine.PARLER
        qwen3 = self._get_qwen3()
        if qwen3 and qwen3.is_available():
            return TTSEngine.QWEN3
        voxtral = self._get_voxtral()
        if voxtral and voxtral.is_available():
            return TTSEngine.VOXTRAL

        raise RuntimeError("Aucun moteur TTS disponible")

    def _select_voice(self, lang: str, gender: str = "female", engine: TTSEngine = None) -> str:
        """Sélectionne la meilleure voix pour une langue."""
        lang_base = lang.split("-")[0]

        # Chercher dans les voix par défaut
        key = f"{lang_base}-{gender}" if gender == "male" else lang_base
        if key in self.config.default_voices:
            return self.config.default_voices[key]
        if lang_base in self.config.default_voices:
            return self.config.default_voices[lang_base]

        # Chercher une voix compatible
        for voice in self.get_voices(lang=lang_base, engine=engine):
            if voice.gender == gender:
                return voice.id

        # Fallback
        voices = self.get_voices(lang=lang_base, engine=engine)
        if voices:
            return voices[0].id

        # Dernier recours
        return "ff_siwis" if engine == TTSEngine.KOKORO else "fr-FR-DeniseNeural"

    def _preprocess_text(self, text: str, lang: str) -> str:
        """Prétraite le texte selon la langue."""
        lang_base = lang.split("-")[0]

        if lang_base == "fr":
            # Initialiser à la volée si pas fait (lazy loading)
            if self._french_preprocessor is None and self.config.use_french_preprocessor:
                try:
                    from src.french_preprocessor import FrenchTextPreprocessor
                    self._french_preprocessor = FrenchTextPreprocessor()
                except ImportError as e:
                    print(f"Warning: French preprocessor not available: {e}")

            if self._french_preprocessor:
                return self._french_preprocessor.process(text)

        return text

    def synthesize(
        self,
        text: str,
        lang: str = "fr",
        voice: Optional[str] = None,
        gender: str = "female",
        speed: Optional[float] = None,
        engine: Optional[TTSEngine] = None,
        preprocess: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Synthétise du texte en audio.

        Args:
            text: Texte à synthétiser
            lang: Code langue ("fr", "en", "fr-FR", etc.)
            voice: ID de voix spécifique (optionnel)
            gender: "female" ou "male" (si voice non spécifié)
            speed: Vitesse de lecture (1.0 = normal)
            engine: Moteur à utiliser (AUTO par défaut)
            preprocess: Appliquer le prétraitement de texte

        Returns:
            Tuple (audio_array, sample_rate)
        """
        speed = speed or self.config.speed

        # Sélectionner le moteur
        selected_engine = self._select_engine(lang, engine)

        # Sélectionner la voix
        if not voice:
            voice = self._select_voice(lang, gender, selected_engine)

        # Prétraiter le texte
        if preprocess:
            text = self._preprocess_text(text, lang)

        # Synthétiser
        if selected_engine == TTSEngine.KOKORO:
            return self._kokoro.synthesize(text, voice=voice, speed=speed, lang=lang)
        elif selected_engine == TTSEngine.CHATTERBOX:
            cb = self._get_chatterbox()
            return cb.synthesize_array(text, speed=speed, lang=lang)
        elif selected_engine == TTSEngine.ORPHEUS:
            orph = self._get_orpheus()
            return orph.synthesize_array(text, voice=voice, speed=speed, lang=lang)
        elif selected_engine == TTSEngine.PARLER:
            parler = self._get_parler()
            return parler.synthesize_array(text, voice_description=voice)
        elif selected_engine == TTSEngine.QWEN3:
            qwen3 = self._get_qwen3()
            return qwen3.synthesize_array(text, voice=voice, speed=speed, lang=lang)
        elif selected_engine == TTSEngine.VOXTRAL:
            voxtral = self._get_voxtral()
            return voxtral.synthesize_array(text, voice=voice, speed=speed, lang=lang)
        else:
            return self._edge_tts.synthesize(text, voice=voice, speed=speed)

    async def synthesize_async(
        self,
        text: str,
        lang: str = "fr",
        voice: Optional[str] = None,
        gender: str = "female",
        speed: Optional[float] = None,
        engine: Optional[TTSEngine] = None,
        preprocess: bool = True
    ) -> Tuple[np.ndarray, int]:
        """Version asynchrone de synthesize (pour Edge-TTS)."""
        speed = speed or self.config.speed
        selected_engine = self._select_engine(lang, engine)

        if not voice:
            voice = self._select_voice(lang, gender, selected_engine)

        if preprocess:
            text = self._preprocess_text(text, lang)

        if selected_engine == TTSEngine.EDGE_TTS:
            return await self._edge_tts.synthesize_async(text, voice=voice, speed=speed)
        elif selected_engine == TTSEngine.CHATTERBOX:
            cb = self._get_chatterbox()
            return cb.synthesize_array(text, speed=speed, lang=lang)
        elif selected_engine == TTSEngine.ORPHEUS:
            orph = self._get_orpheus()
            return orph.synthesize_array(text, voice=voice, speed=speed, lang=lang)
        elif selected_engine == TTSEngine.PARLER:
            parler = self._get_parler()
            return parler.synthesize_array(text, voice_description=voice)
        elif selected_engine == TTSEngine.QWEN3:
            qwen3 = self._get_qwen3()
            return qwen3.synthesize_array(text, voice=voice, speed=speed, lang=lang)
        elif selected_engine == TTSEngine.VOXTRAL:
            voxtral = self._get_voxtral()
            return voxtral.synthesize_array(text, voice=voice, speed=speed, lang=lang)
        else:
            # Kokoro est synchrone
            return self._kokoro.synthesize(text, voice=voice, speed=speed, lang=lang)

    def synthesize_to_file(
        self,
        text: str,
        output_path: Path,
        **kwargs
    ) -> bool:
        """
        Synthétise et sauvegarde dans un fichier.

        Args:
            text: Texte à synthétiser
            output_path: Chemin du fichier de sortie (.wav)
            **kwargs: Arguments passés à synthesize()

        Returns:
            True si succès
        """
        try:
            import soundfile as sf

            audio, sr = self.synthesize(text, **kwargs)

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), audio, sr)

            return True
        except Exception as e:
            print(f"Erreur synthèse: {e}")
            return False

    def list_voices(self, lang: Optional[str] = None):
        """Affiche les voix disponibles."""
        print("\n🎤 Voix TTS disponibles:\n")

        current_engine = None
        for voice in self.get_voices(lang=lang):
            if voice.engine != current_engine:
                current_engine = voice.engine
                print(f"\n  [{current_engine.value.upper()}]")

            print(f"    {voice.id:25} - {voice.name:10} ({voice.gender}) {voice.description}")


def create_tts(
    french_engine: TTSEngine = TTSEngine.EDGE_TTS,
    english_engine: TTSEngine = TTSEngine.KOKORO
) -> UnifiedTTS:
    """
    Crée une instance TTS configurée.

    Args:
        french_engine: Moteur pour le français
        english_engine: Moteur pour l'anglais
    """
    config = TTSConfig(
        default_engine=TTSEngine.AUTO,
        lang_preferences={
            "fr": french_engine,
            "en": english_engine,
        }
    )
    return UnifiedTTS(config)


# Tests
if __name__ == "__main__":
    import soundfile as sf

    print("="*60)
    print("TEST DU SYSTÈME TTS UNIFIÉ")
    print("="*60)

    tts = UnifiedTTS()

    # Afficher les moteurs disponibles
    print(f"\nMoteurs disponibles: {[e.value for e in tts.get_available_engines()]}")

    # Afficher les voix
    tts.list_voices()

    # Tests de synthèse
    print("\n" + "="*60)
    print("TESTS DE SYNTHÈSE")
    print("="*60)

    tests = [
        ("fr", "Bonjour! Comment allez-vous aujourd'hui? C'était une belle journée."),
        ("fr", "M. Dupont a rendez-vous à 14h avec Mme Martin."),
        ("en", "Hello! How are you today? The weather is beautiful."),
    ]

    output_dir = Path("demo_output/unified_tts")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (lang, text) in enumerate(tests):
        print(f"\n[{i+1}] Langue: {lang}")
        print(f"    Texte: {text[:50]}...")

        try:
            audio, sr = tts.synthesize(text, lang=lang)
            output_file = output_dir / f"test_{i+1}_{lang}.wav"
            sf.write(str(output_file), audio, sr)
            print(f"    -> {output_file}")
        except Exception as e:
            print(f"    Erreur: {e}")

    print(f"\nFichiers générés dans: {output_dir}")
