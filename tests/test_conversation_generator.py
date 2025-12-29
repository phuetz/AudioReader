"""
Tests pour le module conversation_generator.py

Teste:
- Speaker et DialogueLine
- DialogueParser
- VoicePool
- ConversationGenerator
- PodcastGenerator
"""
import pytest
import numpy as np
import sys
from pathlib import Path
import tempfile
import json

# Ajouter le repertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.conversation_generator import (
    SpeakerGender,
    Speaker,
    DialogueLine,
    Conversation,
    DialogueParser,
    VoicePool,
    ConversationGenerator,
    PodcastGenerator,
    generate_conversation_from_script
)


class TestSpeaker:
    """Tests pour Speaker."""

    def test_default_values(self):
        """Test valeurs par defaut."""
        speaker = Speaker(
            name="Test",
            voice_id="ff_siwis"
        )

        assert speaker.name == "Test"
        assert speaker.voice_id == "ff_siwis"
        assert speaker.gender == SpeakerGender.NEUTRAL
        assert speaker.speed == 1.0
        assert speaker.pitch_shift == 0.0
        assert speaker.description == ""
        assert speaker.color == "#FFFFFF"

    def test_custom_values(self):
        """Test valeurs personnalisees."""
        speaker = Speaker(
            name="Marie",
            voice_id="af_bella",
            gender=SpeakerGender.FEMALE,
            speed=1.1,
            pitch_shift=0.5,
            description="Personnage principal",
            color="#FF0000"
        )

        assert speaker.name == "Marie"
        assert speaker.gender == SpeakerGender.FEMALE
        assert speaker.speed == 1.1


class TestDialogueLine:
    """Tests pour DialogueLine."""

    @pytest.fixture
    def speaker(self):
        return Speaker(name="Test", voice_id="ff_siwis")

    def test_default_values(self, speaker):
        """Test valeurs par defaut."""
        line = DialogueLine(
            speaker=speaker,
            text="Bonjour !"
        )

        assert line.speaker == speaker
        assert line.text == "Bonjour !"
        assert line.emotion is None
        assert line.direction == ""
        assert line.pause_before == 0.0
        assert line.pause_after == 0.3

    def test_with_emotion(self, speaker):
        """Test avec emotion."""
        line = DialogueLine(
            speaker=speaker,
            text="C'est genial !",
            emotion="excited"
        )

        assert line.emotion == "excited"


class TestDialogueParser:
    """Tests pour DialogueParser."""

    @pytest.fixture
    def parser(self):
        return DialogueParser()

    def test_detect_format_script(self, parser):
        """Test detection format script."""
        text = "JEAN: Bonjour !\nMARIE: Salut !"
        fmt = parser.detect_format(text)
        assert fmt in ["script", "script_lower"]

    def test_detect_format_markdown(self, parser):
        """Test detection format markdown."""
        text = "**Jean:** Bonjour !\n**Marie:** Salut !"
        fmt = parser.detect_format(text)
        assert fmt == "markdown"

    def test_detect_format_theatre(self, parser):
        """Test detection format theatre."""
        text = "JEAN. - Bonjour !\nMARIE. - Salut !"
        fmt = parser.detect_format(text)
        assert fmt == "theatre"

    def test_detect_format_unknown(self, parser):
        """Test detection format inconnu."""
        text = "Juste du texte sans format particulier."
        fmt = parser.detect_format(text)
        assert fmt == "unknown"

    def test_parse_script_format(self, parser):
        """Test parsing format script."""
        text = """
        JEAN: Bonjour Marie !
        MARIE: Salut Jean !
        JEAN: Comment vas-tu ?
        """
        dialogues = parser.parse(text)

        assert len(dialogues) == 3
        assert dialogues[0][0] == "JEAN"
        assert "Bonjour Marie" in dialogues[0][1]
        assert dialogues[1][0] == "MARIE"

    def test_parse_markdown_format(self, parser):
        """Test parsing format markdown."""
        text = """
        **Jean:** Bonjour !
        **Marie:** Salut !
        """
        dialogues = parser.parse(text)

        assert len(dialogues) == 2
        assert dialogues[0][0] == "Jean"
        assert dialogues[1][0] == "Marie"

    def test_parse_multiline_dialogue(self, parser):
        """Test parsing dialogue multiligne."""
        text = """
        JEAN: Ceci est un long dialogue
        qui continue sur plusieurs lignes.
        MARIE: Et voici la reponse.
        """
        dialogues = parser.parse(text)

        assert len(dialogues) == 2
        assert "plusieurs lignes" in dialogues[0][1]

    def test_extract_direction(self, parser):
        """Test extraction des indications sceniques."""
        text = "Il regarde [pensif] par la fenetre."
        cleaned, direction = parser.extract_direction(text)

        assert "[pensif]" not in cleaned
        assert "pensif" in direction

    def test_extract_direction_parentheses(self, parser):
        """Test extraction avec parentheses."""
        text = "Elle sourit (doucement) et dit merci."
        cleaned, direction = parser.extract_direction(text)

        assert "(doucement)" not in cleaned
        assert "doucement" in direction

    def test_extract_emotion(self, parser):
        """Test extraction des emotions."""
        text = "[excited] C'est genial !"
        cleaned, emotion = parser.extract_emotion(text)

        assert "[excited]" not in cleaned
        assert emotion == "excited"

    def test_extract_emotion_none(self, parser):
        """Test sans emotion."""
        text = "Texte normal sans emotion."
        cleaned, emotion = parser.extract_emotion(text)

        assert cleaned == text
        assert emotion is None

    def test_parse_narrative_with_quotes(self, parser):
        """Test parsing narratif avec guillemets."""
        text = '''Marie dit « Bonjour ! » et partit.'''
        dialogues = parser.parse(text)

        assert len(dialogues) >= 1


class TestVoicePool:
    """Tests pour VoicePool."""

    @pytest.fixture
    def pool(self):
        return VoicePool()

    def test_init_default_voices(self, pool):
        """Test initialisation avec voix par defaut."""
        assert "male_fr" in pool.voices
        assert "female_fr" in pool.voices
        assert "male_en" in pool.voices
        assert "female_en" in pool.voices

    def test_assign_voice_male(self, pool):
        """Test assignation voix masculine."""
        voice = pool.assign_voice("Pierre", SpeakerGender.MALE, "fr")

        assert voice in pool.voices["male_fr"]

    def test_assign_voice_female(self, pool):
        """Test assignation voix feminine."""
        voice = pool.assign_voice("Marie", SpeakerGender.FEMALE, "fr")

        assert voice in pool.voices["female_fr"]

    def test_assign_voice_neutral_alternates(self, pool):
        """Test que neutral alterne entre male et female."""
        voice1 = pool.assign_voice("Speaker1", SpeakerGender.NEUTRAL, "fr")
        voice2 = pool.assign_voice("Speaker2", SpeakerGender.NEUTRAL, "fr")

        # Devrait alterner
        male_voices = pool.voices["male_fr"]
        female_voices = pool.voices["female_fr"]

        assert (voice1 in male_voices and voice2 in female_voices) or \
               (voice1 in female_voices and voice2 in male_voices)

    def test_assign_voice_same_speaker(self, pool):
        """Test que le meme speaker garde la meme voix."""
        voice1 = pool.assign_voice("Jean", SpeakerGender.MALE, "fr")
        voice2 = pool.assign_voice("Jean", SpeakerGender.MALE, "fr")

        assert voice1 == voice2

    def test_assign_voice_avoids_reuse(self, pool):
        """Test que les voix sont distribuees."""
        # Assigner plusieurs speakers males
        voices = []
        for i in range(4):
            voice = pool.assign_voice(f"Male{i}", SpeakerGender.MALE, "fr")
            voices.append(voice)

        # Les voix devraient etre distribuees (pas toutes identiques)
        # si assez de voix sont disponibles

    def test_get_assignment(self, pool):
        """Test recuperation des assignations."""
        pool.assign_voice("Jean", SpeakerGender.MALE, "fr")
        pool.assign_voice("Marie", SpeakerGender.FEMALE, "fr")

        assignments = pool.get_assignment()

        assert "Jean" in assignments
        assert "Marie" in assignments

    def test_reset(self, pool):
        """Test reinitialisation."""
        pool.assign_voice("Jean", SpeakerGender.MALE, "fr")
        pool.reset()

        assignments = pool.get_assignment()
        assert len(assignments) == 0


class TestConversationGenerator:
    """Tests pour ConversationGenerator."""

    @pytest.fixture
    def generator(self):
        return ConversationGenerator()

    def test_init(self, generator):
        """Test initialisation."""
        assert generator.parser is not None
        assert generator.voice_pool is not None

    def test_parse_script_basic(self, generator):
        """Test parsing script basique."""
        script = """
        JEAN: Bonjour !
        MARIE: Salut !
        """
        conversation = generator.parse_script(script)

        assert len(conversation.speakers) == 2
        assert "JEAN" in conversation.speakers
        assert "MARIE" in conversation.speakers
        assert len(conversation.lines) == 2

    def test_parse_script_with_config(self, generator):
        """Test parsing avec configuration."""
        script = """
        JEAN: Bonjour !
        MARIE: Salut !
        """
        config = {
            "JEAN": {"gender": "male", "voice": "fm_hugo"},
            "MARIE": {"gender": "female", "voice": "ff_siwis"}
        }

        conversation = generator.parse_script(script, config)

        assert conversation.speakers["JEAN"].voice_id == "fm_hugo"
        assert conversation.speakers["MARIE"].voice_id == "ff_siwis"

    def test_parse_script_with_emotions(self, generator):
        """Test parsing avec emotions."""
        script = """
        JEAN: [excited] C'est genial !
        MARIE: [sad] Vraiment ?
        """
        conversation = generator.parse_script(script)

        assert conversation.lines[0].emotion == "excited"
        assert conversation.lines[1].emotion == "sad"

    def test_parse_script_auto_assigns_voices(self, generator):
        """Test assignation automatique des voix."""
        script = """
        ALICE: Hello
        BOB: World
        """
        conversation = generator.parse_script(script)

        # Chaque speaker devrait avoir une voix
        assert conversation.speakers["ALICE"].voice_id is not None
        assert conversation.speakers["BOB"].voice_id is not None

    def test_export_timeline_json(self, generator):
        """Test export timeline JSON."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            segments = [
                {
                    "index": 0,
                    "speaker": "JEAN",
                    "text": "Bonjour",
                    "start_time": 0.0,
                    "end_time": 1.0,
                },
                {
                    "index": 1,
                    "speaker": "MARIE",
                    "text": "Salut",
                    "start_time": 1.5,
                    "end_time": 2.5,
                }
            ]

            generator.export_timeline(segments, path, format="json")

            assert path.exists()

            with open(path) as f:
                data = json.load(f)

            assert len(data) == 2
        finally:
            if path.exists():
                path.unlink()

    def test_export_timeline_srt(self, generator):
        """Test export timeline SRT."""
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
            path = Path(f.name)

        try:
            segments = [
                {
                    "index": 0,
                    "speaker": "JEAN",
                    "text": "Bonjour",
                    "start_time": 0.0,
                    "end_time": 1.0,
                }
            ]

            generator.export_timeline(segments, path, format="srt")

            assert path.exists()

            with open(path) as f:
                content = f.read()

            assert "JEAN" in content
            assert "Bonjour" in content
            assert "-->" in content
        finally:
            if path.exists():
                path.unlink()

    def test_export_timeline_csv(self, generator):
        """Test export timeline CSV."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)

        try:
            segments = [
                {
                    "index": 0,
                    "speaker": "JEAN",
                    "text": "Bonjour",
                    "start_time": 0.0,
                    "duration": 1.0,
                    "emotion": "neutral",
                }
            ]

            generator.export_timeline(segments, path, format="csv")

            assert path.exists()

            with open(path) as f:
                content = f.read()

            assert "speaker" in content
            assert "JEAN" in content
        finally:
            if path.exists():
                path.unlink()

    def test_format_srt_time(self, generator):
        """Test formatage temps SRT."""
        result = generator._format_srt_time(3661.5)  # 1h 1m 1s 500ms

        assert "01:" in result  # 1 heure
        assert "01:" in result  # 1 minute
        assert "01," in result  # 1 seconde
        assert "500" in result  # 500 ms


class TestGenerateConversationFromScript:
    """Tests pour la fonction utilitaire."""

    def test_basic_generation(self):
        """Test generation basique."""
        script = """
        JEAN: Bonjour !
        MARIE: Salut !
        """

        result = generate_conversation_from_script(
            script,
            output_path=Path("dummy.wav"),
            speaker_config={
                "JEAN": {"gender": "male"},
                "MARIE": {"gender": "female"}
            }
        )

        assert "conversation" in result
        assert "speakers" in result
        assert "lines_count" in result
        assert result["lines_count"] == 2


class TestPodcastGenerator:
    """Tests pour PodcastGenerator."""

    @pytest.fixture
    def generator(self):
        return PodcastGenerator()

    def test_init(self, generator):
        """Test initialisation."""
        assert generator.intro_path is None
        assert generator.outro_path is None
        assert generator.jingle_path is None

    def test_set_audio_elements(self, generator):
        """Test configuration des elements audio."""
        generator.set_audio_elements(
            intro=Path("intro.wav"),
            outro=Path("outro.wav"),
            jingle=Path("jingle.wav")
        )

        assert generator.intro_path == Path("intro.wav")
        assert generator.outro_path == Path("outro.wav")
        assert generator.jingle_path == Path("jingle.wav")


class TestConversation:
    """Tests pour Conversation."""

    def test_create_conversation(self):
        """Test creation conversation."""
        speakers = {
            "JEAN": Speaker(name="JEAN", voice_id="fm_hugo"),
            "MARIE": Speaker(name="MARIE", voice_id="ff_siwis")
        }

        lines = [
            DialogueLine(
                speaker=speakers["JEAN"],
                text="Bonjour !"
            ),
            DialogueLine(
                speaker=speakers["MARIE"],
                text="Salut !"
            )
        ]

        conversation = Conversation(
            title="Test Conversation",
            speakers=speakers,
            lines=lines,
            metadata={"author": "Test"}
        )

        assert conversation.title == "Test Conversation"
        assert len(conversation.speakers) == 2
        assert len(conversation.lines) == 2
        assert conversation.metadata["author"] == "Test"


class TestIntegration:
    """Tests d'integration."""

    def test_full_workflow(self):
        """Test workflow complet."""
        script = """
        NARRATEUR: Il etait une fois...
        JEAN: [excited] Bonjour Marie !
        MARIE: [cheerful] Salut Jean, comment vas-tu ?
        JEAN: [pause] Tres bien, merci.
        NARRATEUR: Et ils partirent ensemble.
        """

        generator = ConversationGenerator()
        conversation = generator.parse_script(
            script,
            speaker_config={
                "NARRATEUR": {"gender": "neutral"},
                "JEAN": {"gender": "male"},
                "MARIE": {"gender": "female"}
            }
        )

        # Verifier la structure
        assert len(conversation.speakers) == 3
        assert len(conversation.lines) == 5

        # Verifier les emotions
        emotions = [line.emotion for line in conversation.lines]
        assert "excited" in emotions
        assert "cheerful" in emotions

        # Verifier les voix assignees
        for speaker in conversation.speakers.values():
            assert speaker.voice_id is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
