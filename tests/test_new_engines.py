"""
Tests pour les nouveaux moteurs TTS (Chatterbox, Orpheus) et NativeAudioEnhancer.
"""
import numpy as np
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ajouter le repertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))


# === Tests Chatterbox Engine ===

class TestChatterboxConfig:
    def test_default_config(self):
        from src.tts_chatterbox_engine import ChatterboxConfig
        config = ChatterboxConfig()
        assert config.default_language == "fr"
        assert config.use_gpu is True
        assert 0 <= config.exaggeration <= 1
        assert config.sample_rate == 24000

    def test_custom_config(self):
        from src.tts_chatterbox_engine import ChatterboxConfig
        config = ChatterboxConfig(
            default_language="en",
            use_gpu=False,
            exaggeration=0.8,
            use_turbo=True,
        )
        assert config.default_language == "en"
        assert config.use_turbo is True


class TestChatterboxEngine:
    def test_import(self):
        from src.tts_chatterbox_engine import ChatterboxEngine
        engine = ChatterboxEngine()
        assert engine is not None
        assert engine.sample_rate == 24000

    def test_is_available_without_lib(self):
        from src.tts_chatterbox_engine import ChatterboxEngine
        engine = ChatterboxEngine()
        # Sans la lib installee, devrait retourner False sans crasher
        result = engine.is_available()
        assert isinstance(result, bool)

    def test_tag_processing(self):
        from src.tts_chatterbox_engine import ChatterboxEngine
        engine = ChatterboxEngine()

        text, adjust = engine._process_tags("[whispers] Bonjour le monde")
        assert "whispers" not in text
        assert adjust < 0  # Whisper = exaggeration basse

        text, adjust = engine._process_tags("[angry] Je suis furieux!")
        assert "angry" not in text
        assert adjust > 0  # Anger = exaggeration haute

    def test_tag_processing_neutral(self):
        from src.tts_chatterbox_engine import ChatterboxEngine
        engine = ChatterboxEngine()

        text, adjust = engine._process_tags("Bonjour, comment allez-vous?")
        assert text == "Bonjour, comment allez-vous?"
        assert adjust == 0.0

    def test_split_text(self):
        from src.tts_chatterbox_engine import ChatterboxEngine
        text = "Premiere phrase. Deuxieme phrase. Troisieme phrase."
        chunks = ChatterboxEngine._split_text(text, max_chars=30)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 60  # Avec marge pour la phrase

    def test_split_text_short(self):
        from src.tts_chatterbox_engine import ChatterboxEngine
        text = "Court."
        chunks = ChatterboxEngine._split_text(text)
        assert len(chunks) == 1

    def test_register_voice_missing_file(self):
        from src.tts_chatterbox_engine import ChatterboxEngine
        engine = ChatterboxEngine()
        with pytest.raises(FileNotFoundError):
            engine.register_voice("test", "/nonexistent/file.wav")

    def test_get_info(self):
        from src.tts_chatterbox_engine import ChatterboxEngine
        engine = ChatterboxEngine()
        info = engine.get_info()
        assert info["engine"] == "chatterbox"
        assert "voice_cloning" in info["features"]
        assert "emotion_control" in info["features"]


# === Tests Orpheus Engine ===

class TestOrpheusEngine:
    def test_import(self):
        from src.tts_orpheus_engine import OrpheusEngine
        engine = OrpheusEngine()
        assert engine is not None
        assert engine.sample_rate == 24000

    def test_is_available_without_lib(self):
        from src.tts_orpheus_engine import OrpheusEngine
        engine = OrpheusEngine()
        result = engine.is_available()
        assert isinstance(result, bool)

    def test_convert_tags(self):
        from src.tts_orpheus_engine import OrpheusEngine
        engine = OrpheusEngine()

        result = engine._convert_tags("[laugh] Ha ha! [sigh] C'est triste.")
        assert "<laugh>" in result
        assert "<sigh>" in result
        assert "[laugh]" not in result

    def test_convert_tags_cleanup(self):
        from src.tts_orpheus_engine import OrpheusEngine
        engine = OrpheusEngine()

        result = engine._convert_tags("[whispers] Bonjour [unknown_tag]")
        assert "[" not in result  # Tous les tags nettoyes

    def test_select_voice(self):
        from src.tts_orpheus_engine import OrpheusEngine
        engine = OrpheusEngine()

        voice_f = engine._select_voice(gender="female")
        assert voice_f in ["tara", "leah", "jess", "mia", "zoe"]

        voice_m = engine._select_voice(gender="male")
        assert voice_m in ["leo", "dan", "zac"]

        voice_specific = engine._select_voice(voice="dan")
        assert voice_specific == "dan"

    def test_voices_dict(self):
        from src.tts_orpheus_engine import ORPHEUS_VOICES
        assert len(ORPHEUS_VOICES) == 8
        for name, info in ORPHEUS_VOICES.items():
            assert "gender" in info
            assert info["gender"] in ("M", "F")

    def test_get_info(self):
        from src.tts_orpheus_engine import OrpheusEngine
        engine = OrpheusEngine()
        info = engine.get_info()
        assert info["engine"] == "orpheus"
        assert info["supports_emotion_tags"] is True
        assert "<laugh>" in info["supported_tags"]


# === Tests NativeAudioEnhancer ===

class TestNativeAudioEnhancer:
    def _make_audio(self, duration=1.0, sr=24000, freq=440):
        """Genere un signal sinusoidal de test."""
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        return np.sin(2 * np.pi * freq * t).astype(np.float32)

    def test_import(self):
        from src.audio_enhancer import NativeAudioEnhancer
        enhancer = NativeAudioEnhancer()
        assert enhancer is not None

    def test_enhance_sine(self):
        from src.audio_enhancer import NativeAudioEnhancer
        enhancer = NativeAudioEnhancer()
        audio = self._make_audio()
        result = enhancer.enhance(audio, 24000)

        assert result.dtype == np.float32
        assert len(result) == len(audio)
        assert np.max(np.abs(result)) <= 1.0  # Pas de clipping

    def test_enhance_empty(self):
        from src.audio_enhancer import NativeAudioEnhancer
        enhancer = NativeAudioEnhancer()
        audio = np.array([], dtype=np.float32)
        result = enhancer.enhance(audio, 24000)
        assert len(result) == 0

    def test_enhance_silence(self):
        from src.audio_enhancer import NativeAudioEnhancer
        enhancer = NativeAudioEnhancer()
        audio = np.zeros(24000, dtype=np.float32)
        result = enhancer.enhance(audio, 24000)
        assert len(result) == 24000
        # Le resultat doit rester tres silencieux
        assert np.max(np.abs(result)) < 0.01

    def test_highpass_removes_dc(self):
        from src.audio_enhancer import NativeAudioEnhancer
        enhancer = NativeAudioEnhancer()
        # Signal avec DC offset
        audio = self._make_audio() + 0.5
        result = enhancer._highpass(audio, 24000, 80)
        # Le DC offset devrait etre reduit
        assert abs(np.mean(result)) < abs(np.mean(audio))

    def test_limiter_prevents_clipping(self):
        from src.audio_enhancer import NativeAudioEnhancer, AudioEnhancerConfig
        config = AudioEnhancerConfig(true_peak_limit=-3.0)
        enhancer = NativeAudioEnhancer(config)

        audio = self._make_audio() * 2.0  # Signal qui clip
        result = enhancer._limiter(audio)
        limit = 10 ** (-3.0 / 20)
        assert np.max(np.abs(result)) <= limit + 1e-6

    def test_normalize_loudness(self):
        from src.audio_enhancer import NativeAudioEnhancer
        enhancer = NativeAudioEnhancer()
        audio = self._make_audio() * 0.01  # Signal tres faible
        result = enhancer._normalize_loudness(audio, target_db=-19.0)
        # Le signal doit etre plus fort
        assert np.sqrt(np.mean(result ** 2)) > np.sqrt(np.mean(audio ** 2))

    def test_full_pipeline_no_artifacts(self):
        """Verifie que le pipeline complet ne produit pas de NaN ou Inf."""
        from src.audio_enhancer import NativeAudioEnhancer
        enhancer = NativeAudioEnhancer()

        # Signal mixte: voix simulee + bruit
        audio = self._make_audio(duration=2.0) * 0.3
        noise = np.random.randn(len(audio)).astype(np.float32) * 0.01
        audio = audio + noise

        result = enhancer.enhance(audio, 24000)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert result.dtype == np.float32


# === Tests RoomTone ameliore ===

class TestRoomToneGenerator:
    def test_pink_noise_spectrum(self):
        """Verifie que le bruit rose a un spectre 1/f."""
        from src.audio_enhancer import RoomToneGenerator
        gen = RoomToneGenerator(sample_rate=24000, level_db=-50)
        noise = gen.generate(2.0)

        assert len(noise) == 48000
        assert noise.dtype == np.float32

        # Verifier le spectre: les basses frequences doivent etre plus fortes
        fft = np.abs(np.fft.rfft(noise))
        n = len(fft)
        low_energy = np.mean(fft[1:n // 10] ** 2)
        high_energy = np.mean(fft[n // 2:] ** 2)
        # Le bruit rose doit avoir plus d'energie dans les basses
        assert low_energy > high_energy

    def test_zero_duration(self):
        from src.audio_enhancer import RoomToneGenerator
        gen = RoomToneGenerator()
        noise = gen.generate(0.0)
        assert len(noise) == 0


# === Tests integration TTSEngine enum ===

class TestUnifiedTTSEngineEnum:
    def test_new_engines_in_enum(self):
        from src.tts_unified import TTSEngine
        assert TTSEngine.CHATTERBOX.value == "chatterbox"
        assert TTSEngine.ORPHEUS.value == "orpheus"
        assert TTSEngine.PARLER.value == "parler"

    def test_unified_tts_creates(self):
        from src.tts_unified import UnifiedTTS
        tts = UnifiedTTS()
        engines = tts.get_available_engines()
        # Au minimum Kokoro devrait etre disponible
        assert isinstance(engines, list)


# === Tests Parler Engine ===

class TestParlerConfig:
    def test_default_config(self):
        from src.tts_parler_engine import ParlerConfig
        config = ParlerConfig()
        assert config.default_language == "fr"
        assert config.target_sample_rate == 24000
        assert config.use_gpu is True

    def test_custom_config(self):
        from src.tts_parler_engine import ParlerConfig
        config = ParlerConfig(
            model_name="custom/model",
            use_gpu=False,
            target_sample_rate=16000,
        )
        assert config.model_name == "custom/model"
        assert config.use_gpu is False
        assert config.target_sample_rate == 16000


class TestParlerEngine:
    def test_import(self):
        from src.tts_parler_engine import ParlerEngine
        engine = ParlerEngine()
        assert engine is not None

    def test_voice_descriptions(self):
        from src.tts_parler_engine import FRENCH_VOICE_DESCRIPTIONS
        assert "narrator_female" in FRENCH_VOICE_DESCRIPTIONS
        assert "narrator_male" in FRENCH_VOICE_DESCRIPTIONS
        assert "expressive_female" in FRENCH_VOICE_DESCRIPTIONS
        assert "whisper" in FRENCH_VOICE_DESCRIPTIONS
        # Toutes les descriptions doivent etre non vides
        for name, desc in FRENCH_VOICE_DESCRIPTIONS.items():
            assert len(desc) > 20, f"Description trop courte pour {name}"

    def test_emotion_descriptions(self):
        from src.tts_parler_engine import EMOTION_DESCRIPTIONS
        assert "joy" in EMOTION_DESCRIPTIONS
        assert "sadness" in EMOTION_DESCRIPTIONS
        assert "anger" in EMOTION_DESCRIPTIONS
        assert "fear" in EMOTION_DESCRIPTIONS

    def test_build_description_default(self):
        from src.tts_parler_engine import ParlerEngine, FRENCH_VOICE_DESCRIPTIONS
        engine = ParlerEngine()
        desc = engine._build_description()
        assert desc == FRENCH_VOICE_DESCRIPTIONS["narrator_female"]

    def test_build_description_preset(self):
        from src.tts_parler_engine import ParlerEngine, FRENCH_VOICE_DESCRIPTIONS
        engine = ParlerEngine()
        desc = engine._build_description("narrator_male")
        assert desc == FRENCH_VOICE_DESCRIPTIONS["narrator_male"]

    def test_build_description_custom(self):
        from src.tts_parler_engine import ParlerEngine
        engine = ParlerEngine()
        custom = "A custom voice description."
        desc = engine._build_description(custom)
        assert desc == custom

    def test_build_description_with_emotion(self):
        from src.tts_parler_engine import ParlerEngine
        engine = ParlerEngine()
        desc = engine._build_description("narrator_female", emotion="joy")
        assert "joyful" in desc
        assert "bright" in desc

    def test_split_text(self):
        from src.tts_parler_engine import ParlerEngine
        engine = ParlerEngine()
        text = "Premiere phrase. Deuxieme phrase. Troisieme phrase."
        chunks = engine._split_text(text, max_chars=40)
        assert len(chunks) >= 2
        # Toutes les phrases doivent etre presentes
        full = " ".join(chunks)
        assert "Premiere" in full
        assert "Troisieme" in full

    def test_split_text_short(self):
        from src.tts_parler_engine import ParlerEngine
        engine = ParlerEngine()
        text = "Court texte."
        chunks = engine._split_text(text, max_chars=300)
        assert len(chunks) == 1
        assert chunks[0] == "Court texte."

    def test_get_info(self):
        from src.tts_parler_engine import ParlerEngine
        engine = ParlerEngine()
        info = engine.get_info()
        assert info["name"] == "Parler TTS"
        assert "fr" in info["languages"]
        assert "en" in info["languages"]
        assert len(info["presets"]) > 0

    def test_is_available_without_lib(self):
        from src.tts_parler_engine import ParlerEngine
        engine = ParlerEngine()
        # Ne devrait pas planter meme si parler_tts n'est pas installe
        result = engine.is_available()
        assert isinstance(result, bool)


# === Tests bug fixes ===

class TestBugFixes:
    def test_tens_fr_values(self):
        """Verifie que TENS_FR a les bonnes valeurs pour 70, 80, 90."""
        from src.text_normalizer import NumberToWords
        n2w = NumberToWords("fr")
        assert n2w.convert(70) == "soixante-dix"
        assert n2w.convert(71) == "soixante-onze"
        assert n2w.convert(80) == "quatre-vingts"
        assert n2w.convert(90) == "quatre-vingt-dix"
        assert n2w.convert(91) == "quatre-vingt-onze"
        assert n2w.convert(99) == "quatre-vingt-dix-neuf"

    def test_narrative_context_word_boundary(self):
        """Verifie que la detection utilise des frontieres de mot."""
        from src.narrative_context import NarrativeContextDetector
        detector = NarrativeContextDetector(lang="fr")
        # "courir" ne doit PAS matcher dans "recourir"
        score = detector._detect_action("Il a du recourir a la justice.")
        # Le score ne devrait pas etre augmente par le faux match
        assert score < 0.5

    def test_llm_enhancer_json_extraction(self):
        """Verifie que l'extraction JSON est robuste."""
        from src.llm_enhancer import LLMEnhancer
        # Test avec code block
        result = LLMEnhancer._extract_json_str('Voici: ```json\n{"key": "value"}\n```')
        assert '"key"' in result

        # Test sans code block
        result = LLMEnhancer._extract_json_str('Response: {"key": "value"} fin')
        assert '"key"' in result

        # Test plain JSON
        result = LLMEnhancer._extract_json_str('{"key": "value"}')
        assert '"key"' in result

    def test_crossfade_empty_dtype(self):
        """Verifie que les curves vides ont le bon dtype."""
        from src.audio_crossfade import AudioCrossfader
        crossfader = AudioCrossfader()
        curve = crossfader._generate_fade_curve(0, fade_in=True)
        assert curve.dtype == np.float32


# === Tests Voice Designer ===

class TestVoiceDesigner:
    def test_import(self):
        from src.voice_designer import VoiceDesigner, CHARACTER_PRESETS
        assert len(CHARACTER_PRESETS) > 0

    def test_presets_have_required_keys(self):
        from src.voice_designer import CHARACTER_PRESETS
        for name, preset in CHARACTER_PRESETS.items():
            assert "base" in preset, f"Preset {name} manque 'base'"
            assert "description" in preset, f"Preset {name} manque 'description'"
            # Poids positifs
            for voice, weight in preset["base"].items():
                assert weight > 0, f"Poids negatif pour {voice} dans {name}"

    def test_presets_weights_valid(self):
        from src.voice_designer import CHARACTER_PRESETS
        for name, preset in CHARACTER_PRESETS.items():
            total = sum(preset["base"].values())
            assert abs(total - 1.0) < 0.01, f"Poids ne totalisent pas 1.0 pour {name}: {total}"

    def test_voice_design_dataclass(self):
        from src.voice_designer import VoiceDesign
        design = VoiceDesign(
            name="test",
            description="Test voice",
            tensor=np.zeros((510, 1, 256), dtype=np.float32),
            source_voices={"af_bella": 1.0},
        )
        assert design.name == "test"
        assert design.tensor.shape == (510, 1, 256)

    def test_character_variety(self):
        """Verifie qu'il y a des presets pour differents types."""
        from src.voice_designer import CHARACTER_PRESETS
        names = list(CHARACTER_PRESETS.keys())
        # Au moins un preset femme et un homme
        has_femme = any("femme" in n for n in names)
        has_homme = any("homme" in n for n in names)
        has_enfant = any("enfant" in n for n in names)
        has_narrateur = any("narrateur" in n for n in names)
        assert has_femme and has_homme and has_enfant and has_narrateur


# === Tests NativeAudioEnhancer denoise ===

class TestDenoiseEnhancer:
    def test_denoise_fallback(self):
        """Sans noisereduce, utilise le noise gate maison."""
        from src.audio_enhancer import NativeAudioEnhancer
        enhancer = NativeAudioEnhancer()
        # Signal avec bruit
        sr = 24000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        clean = np.sin(2 * np.pi * 440 * t) * 0.5
        noise = np.random.randn(sr).astype(np.float32) * 0.01
        noisy = clean + noise
        # Le denoise ne devrait pas planter
        result = enhancer._denoise(noisy, sr)
        assert len(result) == len(noisy)
        assert result.dtype == np.float32
