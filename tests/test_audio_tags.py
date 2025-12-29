"""
Tests pour le module audio_tags.py

Teste:
- Detection des tags
- Extraction et parsing
- Calcul des modificateurs de prosodie
- Aliases
- Tags TTS natifs
"""
import pytest
import sys
from pathlib import Path

# Ajouter le repertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_tags import (
    AudioTagProcessor,
    AudioTag,
    TagCategory,
    ProcessedSegment,
    process_text_with_audio_tags
)


class TestAudioTagProcessor:
    """Tests pour AudioTagProcessor."""

    @pytest.fixture
    def processor(self):
        """Cree un processeur de tags."""
        return AudioTagProcessor()

    def test_get_tag_exists(self, processor):
        """Test recuperation d'un tag existant."""
        tag = processor.get_tag("excited")
        assert tag is not None
        assert tag.name == "excited"
        assert tag.category == TagCategory.EMOTION

    def test_get_tag_not_exists(self, processor):
        """Test recuperation d'un tag inexistant."""
        tag = processor.get_tag("nonexistent_tag")
        assert tag is None

    def test_get_tag_alias(self, processor):
        """Test resolution des aliases."""
        # "whisper" est un alias de "whispers"
        tag = processor.get_tag("whisper")
        assert tag is not None
        assert tag.name == "whispers"

        # "happy" est un alias de "cheerful"
        tag = processor.get_tag("happy")
        assert tag is not None
        assert tag.name == "cheerful"

        # "mad" est un alias de "angry"
        tag = processor.get_tag("mad")
        assert tag is not None
        assert tag.name == "angry"

    def test_extract_tags_single(self, processor):
        """Test extraction d'un seul tag."""
        text = "[excited] C'est genial !"
        tags = processor.extract_tags(text)

        assert len(tags) == 1
        assert tags[0][0] == "excited"
        assert tags[0][1].name == "excited"

    def test_extract_tags_multiple(self, processor):
        """Test extraction de plusieurs tags."""
        text = "[whispers] Je te dis un secret [pause] vraiment important"
        tags = processor.extract_tags(text)

        assert len(tags) == 2
        assert tags[0][0] == "whispers"
        assert tags[1][0] == "pause"

    def test_extract_tags_none(self, processor):
        """Test texte sans tags."""
        text = "Texte normal sans aucun tag"
        tags = processor.extract_tags(text)
        assert len(tags) == 0

    def test_extract_tags_with_unknown(self, processor):
        """Test avec tags inconnus (ignores)."""
        text = "[unknown_tag] Texte [excited] suite"
        tags = processor.extract_tags(text)

        # Seul [excited] doit etre detecte
        assert len(tags) == 1
        assert tags[0][0] == "excited"

    def test_process_text_removes_tags(self, processor):
        """Test que process_text retire les tags."""
        text = "[excited] C'est genial !"
        processed, tags = processor.process_text(text)

        assert "[excited]" not in processed
        assert "C'est genial !" in processed
        assert len(tags) == 1

    def test_process_text_keeps_tts_tags(self, processor):
        """Test que les tags TTS natifs sont remplaces."""
        text = "[laugh] Ha ha ha !"
        processed, tags = processor.process_text(text)

        # Le tag [laugh] a un tts_tag natif
        laugh_tag = processor.get_tag("laugh")
        if laugh_tag.tts_tag:
            assert laugh_tag.tts_tag in processed

    def test_process_text_cleans_whitespace(self, processor):
        """Test nettoyage des espaces multiples."""
        text = "[pause]    Beaucoup    d'espaces    [pause]"
        processed, _ = processor.process_text(text)

        assert "    " not in processed
        assert processed == "Beaucoup d'espaces"

    def test_calculate_prosody_single_tag(self, processor):
        """Test calcul prosodie avec un seul tag."""
        excited = processor.get_tag("excited")
        prosody = processor.calculate_prosody_modifiers([excited])

        assert prosody["speed"] > 1.0  # excited accelere
        assert prosody["pitch"] > 0    # excited augmente le pitch
        assert prosody["volume"] > 1.0  # excited augmente le volume

    def test_calculate_prosody_multiple_tags(self, processor):
        """Test calcul prosodie avec plusieurs tags."""
        excited = processor.get_tag("excited")
        whispers = processor.get_tag("whispers")

        prosody = processor.calculate_prosody_modifiers([excited, whispers])

        # Les effets se combinent
        assert "speed" in prosody
        assert "volume" in prosody
        assert "pitch" in prosody

    def test_calculate_prosody_limits(self, processor):
        """Test que les valeurs sont limitees."""
        # Empiler plusieurs tags pour forcer les limites
        tags = [processor.get_tag("excited")] * 10
        prosody = processor.calculate_prosody_modifiers(tags)

        # Les valeurs doivent etre dans les limites
        assert 0.5 <= prosody["speed"] <= 1.5
        assert 0.3 <= prosody["volume"] <= 1.5
        assert -1.0 <= prosody["pitch"] <= 1.0

    def test_get_tts_tags(self, processor):
        """Test recuperation des tags TTS natifs."""
        sigh = processor.get_tag("sigh")
        laugh = processor.get_tag("laugh")
        excited = processor.get_tag("excited")

        tts_tags = processor.get_tts_tags([sigh, laugh, excited])

        # sigh et laugh ont des tts_tags, pas excited
        assert len(tts_tags) >= 1  # Au moins un tag TTS


class TestAudioTagCategories:
    """Tests pour les categories de tags."""

    @pytest.fixture
    def processor(self):
        return AudioTagProcessor()

    def test_emotion_tags(self, processor):
        """Test que les tags emotion sont bien categorises."""
        emotion_tags = ["excited", "sad", "angry", "whispers", "fearful", "tender", "dramatic"]

        for tag_name in emotion_tags:
            tag = processor.get_tag(tag_name)
            assert tag is not None, f"Tag {tag_name} devrait exister"
            assert tag.category == TagCategory.EMOTION, f"{tag_name} devrait etre EMOTION"

    def test_action_tags(self, processor):
        """Test que les tags action sont bien categorises."""
        action_tags = ["sigh", "laugh", "chuckle", "gasp", "cough", "yawn"]

        for tag_name in action_tags:
            tag = processor.get_tag(tag_name)
            assert tag is not None, f"Tag {tag_name} devrait exister"
            assert tag.category == TagCategory.ACTION, f"{tag_name} devrait etre ACTION"

    def test_pause_tags(self, processor):
        """Test que les tags pause sont bien categorises."""
        pause_tags = ["pause", "long pause", "beat", "silence"]

        for tag_name in pause_tags:
            tag = processor.get_tag(tag_name)
            assert tag is not None, f"Tag {tag_name} devrait exister"
            assert tag.category == TagCategory.PAUSE, f"{tag_name} devrait etre PAUSE"

    def test_style_tags(self, processor):
        """Test que les tags style sont bien categorises."""
        style_tags = ["sarcastic", "cheerful", "serious", "mysterious", "narrator"]

        for tag_name in style_tags:
            tag = processor.get_tag(tag_name)
            assert tag is not None, f"Tag {tag_name} devrait exister"
            assert tag.category == TagCategory.STYLE, f"{tag_name} devrait etre STYLE"


class TestProcessTextWithAudioTags:
    """Tests pour la fonction utilitaire."""

    def test_basic_processing(self):
        """Test traitement basique."""
        text = "[excited] Wow, c'est genial !"
        result = process_text_with_audio_tags(text)

        assert isinstance(result, ProcessedSegment)
        assert result.original_text == text
        assert "[excited]" not in result.text
        assert len(result.tags) == 1

    def test_prosody_in_result(self):
        """Test que la prosodie est calculee."""
        text = "[whispers] Secret..."
        result = process_text_with_audio_tags(text)

        assert "speed" in result.prosody
        assert "volume" in result.prosody
        assert "pitch" in result.prosody

    def test_complex_text(self):
        """Test texte complexe avec plusieurs elements."""
        text = "[dramatic] [pause] Et puis... [whispers] tout a change. [long pause]"
        result = process_text_with_audio_tags(text)

        assert len(result.tags) >= 3
        assert result.prosody["pause_after"] > 0


class TestTagEffects:
    """Tests pour les effets specifiques des tags."""

    @pytest.fixture
    def processor(self):
        return AudioTagProcessor()

    def test_excited_increases_speed(self, processor):
        """Test que excited accelere."""
        tag = processor.get_tag("excited")
        assert tag.speed_modifier > 1.0

    def test_sad_slows_down(self, processor):
        """Test que sad ralentit."""
        tag = processor.get_tag("sad")
        assert tag.speed_modifier < 1.0

    def test_whispers_reduces_volume(self, processor):
        """Test que whispers reduit le volume."""
        tag = processor.get_tag("whispers")
        assert tag.volume_modifier < 1.0

    def test_angry_increases_volume(self, processor):
        """Test que angry augmente le volume."""
        tag = processor.get_tag("angry")
        assert tag.volume_modifier > 1.0

    def test_pause_has_pause_after(self, processor):
        """Test que pause a une pause apres."""
        tag = processor.get_tag("pause")
        assert tag.pause_after > 0

    def test_long_pause_longer_than_pause(self, processor):
        """Test que long pause est plus long que pause."""
        pause = processor.get_tag("pause")
        long_pause = processor.get_tag("long pause")
        assert long_pause.pause_after > pause.pause_after

    def test_dramatic_has_pauses_before_and_after(self, processor):
        """Test que dramatic a des pauses avant et apres."""
        tag = processor.get_tag("dramatic")
        assert tag.pause_before > 0
        assert tag.pause_after > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
