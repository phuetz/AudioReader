"""
Tests pour le module hq_pipeline_extended.py

Teste:
- ExtendedPipelineConfig
- ExtendedHQSegment
- ExtendedHQPipeline
- AudiobookGenerator
"""
import pytest
import numpy as np
import sys
from pathlib import Path
import tempfile
import json

# Ajouter le repertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Ces imports peuvent echouer si les modules de base ne sont pas disponibles
try:
    from src.hq_pipeline_extended import (
        ExtendedPipelineConfig,
        ExtendedHQSegment,
        ExtendedHQPipeline,
        create_extended_pipeline,
        AudiobookGenerator
    )
    from src.audio_tags import AudioTag, TagCategory
    from src.voice_morphing import VoiceMorphSettings
    from src.emotion_analyzer import Emotion, Intensity, ProsodyHints
    from src.narrative_context import NarrativeType
    from src.character_detector import SpeakerType
    HQ_AVAILABLE = True
except ImportError as e:
    HQ_AVAILABLE = False
    IMPORT_ERROR = str(e)


# Skip tous les tests si les modules ne sont pas disponibles
pytestmark = pytest.mark.skipif(
    not HQ_AVAILABLE,
    reason=f"HQ pipeline modules not available: {IMPORT_ERROR if not HQ_AVAILABLE else ''}"
)


class TestExtendedPipelineConfig:
    """Tests pour ExtendedPipelineConfig."""

    def test_default_values(self):
        """Test valeurs par defaut."""
        config = ExtendedPipelineConfig()

        # Valeurs de base heritees
        assert config.lang == "fr"
        assert config.narrator_voice == "ff_siwis"

        # Nouvelles valeurs
        assert config.enable_audio_tags is True
        assert config.enable_voice_morphing is False
        assert config.enable_voice_cloning is False
        assert config.enable_cache is True
        assert config.enable_parallel is True
        assert config.num_workers == 4
        assert config.emotion_intensity == 0.5
        assert config.enable_phoneme_processing is True

    def test_custom_values(self):
        """Test valeurs personnalisees."""
        config = ExtendedPipelineConfig(
            lang="en",
            narrator_voice="af_heart",
            enable_audio_tags=False,
            enable_voice_morphing=True,
            num_workers=8,
            emotion_intensity=0.8,
            custom_phonemes={"API": "a pi ai"}
        )

        assert config.lang == "en"
        assert config.narrator_voice == "af_heart"
        assert config.enable_audio_tags is False
        assert config.enable_voice_morphing is True
        assert config.num_workers == 8
        assert config.emotion_intensity == 0.8
        assert "API" in config.custom_phonemes

    def test_save_config(self):
        """Test sauvegarde configuration."""
        config = ExtendedPipelineConfig(
            enable_audio_tags=True,
            num_workers=6
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            config.save(path)

            assert path.exists()

            with open(path) as f:
                data = json.load(f)

            assert data["enable_audio_tags"] is True
            assert data["num_workers"] == 6
        finally:
            if path.exists():
                path.unlink()


class TestExtendedHQSegment:
    """Tests pour ExtendedHQSegment."""

    @pytest.fixture
    def sample_segment(self):
        """Cree un segment de test."""
        return ExtendedHQSegment(
            text="Bonjour !",
            index=0,
            speaker="NARRATOR",
            speaker_type=SpeakerType.NARRATOR,
            voice_id="ff_siwis",
            emotion=Emotion.NEUTRAL,
            intensity=Intensity.MEDIUM,
            prosody=ProsodyHints(),
            narrative_type=NarrativeType.NARRATION,
            narrative_confidence=0.9,
            tts_tags=[],
            pause_before=0.0,
            pause_after=0.3,
            is_dialogue=False,
            is_internal_thought=False,
            chapter_index=0,
            final_speed=1.0
        )

    def test_default_extended_values(self, sample_segment):
        """Test valeurs etendues par defaut."""
        assert sample_segment.audio_tags == []
        assert sample_segment.audio_tag_prosody == {}
        assert sample_segment.morph_settings is None
        assert sample_segment.is_cloned_voice is False
        assert sample_segment.cloned_voice_name is None
        assert sample_segment.cache_key is None
        assert sample_segment.from_cache is False
        assert sample_segment.emotion_exaggeration == 0.5

    def test_with_audio_tags(self):
        """Test segment avec audio tags."""
        tag = AudioTag(
            name="excited",
            category=TagCategory.EMOTION,
            tts_tag=None,
            speed_modifier=1.1
        )

        segment = ExtendedHQSegment(
            text="C'est genial !",
            index=0,
            speaker="MARIE",
            speaker_type=SpeakerType.CHARACTER,
            voice_id="af_bella",
            emotion=Emotion.JOY,
            intensity=Intensity.HIGH,
            prosody=ProsodyHints(),
            narrative_type=NarrativeType.DIALOGUE,
            narrative_confidence=1.0,
            audio_tags=[tag],
            audio_tag_prosody={"speed": 1.1},
            final_speed=1.1
        )

        assert len(segment.audio_tags) == 1
        assert segment.audio_tags[0].name == "excited"
        assert segment.audio_tag_prosody["speed"] == 1.1

    def test_with_morph_settings(self):
        """Test segment avec settings de morphing."""
        morph = VoiceMorphSettings(pitch_shift=2.0)

        segment = ExtendedHQSegment(
            text="Test",
            index=0,
            speaker="NARRATOR",
            speaker_type=SpeakerType.NARRATOR,
            voice_id="ff_siwis",
            emotion=Emotion.NEUTRAL,
            intensity=Intensity.MEDIUM,
            prosody=ProsodyHints(),
            narrative_type=NarrativeType.NARRATION,
            narrative_confidence=1.0,
            morph_settings=morph,
            final_speed=1.0
        )

        assert segment.morph_settings is not None
        assert segment.morph_settings.pitch_shift == 2.0


class TestCreateExtendedPipeline:
    """Tests pour la fonction create_extended_pipeline."""

    def test_basic_creation(self):
        """Test creation basique."""
        pipeline = create_extended_pipeline()

        assert pipeline is not None
        assert pipeline.config.lang == "fr"

    def test_with_custom_params(self):
        """Test creation avec parametres."""
        pipeline = create_extended_pipeline(
            lang="en",
            narrator_voice="af_heart",
            enable_audio_tags=True,
            num_workers=2
        )

        assert pipeline.config.lang == "en"
        assert pipeline.config.narrator_voice == "af_heart"
        assert pipeline.config.enable_audio_tags is True
        assert pipeline.config.num_workers == 2

    def test_enable_all_features(self):
        """Test activation de toutes les fonctionnalites."""
        pipeline = create_extended_pipeline(enable_all_features=True)

        assert pipeline.config.enable_audio_tags is True
        assert pipeline.config.enable_voice_morphing is True
        assert pipeline.config.enable_cache is True
        assert pipeline.config.enable_parallel is True
        assert pipeline.config.enable_phoneme_processing is True


class TestExtendedHQPipeline:
    """Tests pour ExtendedHQPipeline."""

    @pytest.fixture
    def pipeline(self):
        """Cree un pipeline pour les tests."""
        config = ExtendedPipelineConfig(
            enable_cache=False,  # Desactiver pour les tests
            enable_parallel=False,  # Desactiver pour les tests
            enable_voice_cloning=False
        )
        return ExtendedHQPipeline(config)

    def test_init(self, pipeline):
        """Test initialisation."""
        assert pipeline.base_pipeline is not None
        assert pipeline.tag_processor is not None
        assert pipeline.voice_morpher is not None
        assert pipeline.emotion_controller is not None

    def test_init_with_cache(self):
        """Test initialisation avec cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExtendedPipelineConfig(
                enable_cache=True,
                cache_dir=Path(tmpdir)
            )
            pipeline = ExtendedHQPipeline(config)

            assert pipeline.cache is not None

    def test_process_chapter_basic(self, pipeline):
        """Test traitement chapitre basique."""
        text = "Bonjour tout le monde. Comment allez-vous ?"

        segments = pipeline.process_chapter(text)

        assert len(segments) > 0
        assert all(isinstance(s, ExtendedHQSegment) for s in segments)

    def test_process_chapter_with_tags(self, pipeline):
        """Test traitement avec audio tags."""
        text = "[excited] C'est incroyable ! [pause] Vraiment ?"

        segments = pipeline.process_chapter(text)

        assert len(segments) > 0
        # Au moins un segment devrait avoir des tags
        all_tags = [tag for s in segments for tag in s.audio_tags]
        assert len(all_tags) >= 1

    def test_process_chapter_with_progress(self, pipeline):
        """Test avec callback de progression."""
        text = "Court texte de test."
        progress_calls = []

        def progress(step, total, msg):
            progress_calls.append((step, total, msg))

        segments = pipeline.process_chapter(text, progress_callback=progress)

        assert len(progress_calls) > 0
        # Le dernier appel devrait indiquer "Termine"
        assert "Termine" in progress_calls[-1][2]

    def test_get_stats(self, pipeline):
        """Test recuperation des stats."""
        stats = pipeline.get_stats()

        assert "segments_processed" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats

    def test_get_characters(self, pipeline):
        """Test recuperation des personnages."""
        text = '''Marie dit "Bonjour !". Pierre repondit "Salut !"'''
        pipeline.process_chapter(text)

        characters = pipeline.get_characters()
        # Devrait detecter les personnages si le texte est assez long

    def test_get_voice_assignments(self, pipeline):
        """Test recuperation des assignations de voix."""
        text = "Texte simple."
        pipeline.process_chapter(text)

        assignments = pipeline.get_voice_assignments()

        assert "NARRATOR" in assignments

    def test_reset(self, pipeline):
        """Test reinitialisation."""
        text = "Premier texte."
        pipeline.process_chapter(text)

        pipeline.reset()

        stats = pipeline.get_stats()
        assert stats["segments_processed"] == 0


class TestAudiobookGenerator:
    """Tests pour AudiobookGenerator."""

    @pytest.fixture
    def generator(self):
        """Cree un generateur sans moteur TTS."""
        config = ExtendedPipelineConfig(
            enable_cache=False,
            enable_parallel=False
        )
        return AudiobookGenerator(config=config, tts_engine=None)

    def test_init(self, generator):
        """Test initialisation."""
        assert generator.pipeline is not None
        assert generator.tts_engine is None

    def test_generate_audiobook_without_tts(self, generator):
        """Test generation sans moteur TTS (analyse seule)."""
        chapters = [
            "Chapitre 1: Introduction. Bonjour !",
            "Chapitre 2: Suite. Au revoir !"
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = generator.generate_audiobook(
                chapters=chapters,
                output_dir=Path(tmpdir),
                title="Test Audiobook"
            )

            assert result["title"] == "Test Audiobook"
            assert len(result["chapters"]) == 2
            assert result["total_segments"] > 0


class TestIntegration:
    """Tests d'integration."""

    def test_full_pipeline_workflow(self):
        """Test workflow complet du pipeline."""
        # Configuration
        config = ExtendedPipelineConfig(
            lang="fr",
            narrator_voice="ff_siwis",
            enable_audio_tags=True,
            enable_phoneme_processing=True,
            enable_cache=False,
            enable_parallel=False,
            custom_phonemes={"TTS": "te te esse"}
        )

        # Pipeline
        pipeline = ExtendedHQPipeline(config)

        # Texte de test
        text = """
        [dramatic] Il etait une fois...

        Marie s'exclama : « [excited] C'est le TTS le plus avance ! »

        [pause] Pierre hocha la tete. [whispers] « Je suis d'accord. »

        Et ils vecurent heureux.
        """

        # Traitement
        segments = pipeline.process_chapter(text)

        # Verifications
        assert len(segments) > 0

        # Verifier les types
        for seg in segments:
            assert isinstance(seg, ExtendedHQSegment)
            assert seg.voice_id is not None
            assert seg.final_speed > 0

        # Verifier les stats
        stats = pipeline.get_stats()
        assert stats["segments_processed"] > 0

    def test_morph_settings_assignment(self):
        """Test assignation des settings de morphing."""
        config = ExtendedPipelineConfig(
            enable_voice_morphing=True,
            default_morph_preset="more_feminine",
            character_morph_presets={
                "Pierre": "more_masculine"
            },
            enable_cache=False,
            enable_parallel=False
        )

        pipeline = ExtendedHQPipeline(config)

        # Verifier que le preset par defaut est recupere
        settings = pipeline._get_morph_settings("NARRATOR")
        assert settings is not None

        # Verifier le preset specifique
        settings_pierre = pipeline._get_morph_settings("Pierre")
        assert settings_pierre is not None
        assert settings_pierre.pitch_shift < 0  # more_masculine


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
