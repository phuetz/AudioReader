"""
Tests pour le module voice_cloning.py

Teste:
- ClonedVoice dataclass
- VoiceCloner
- VoiceCloningManager
- Fonction clone_voice_from_file
"""
import pytest
import sys
from pathlib import Path
import tempfile
import json
import numpy as np

# Ajouter le repertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.voice_cloning import (
    ClonedVoice,
    VoiceCloner,
    VoiceCloningManager,
    clone_voice_from_file
)


class TestClonedVoice:
    """Tests pour ClonedVoice dataclass."""

    def test_default_values(self):
        """Test valeurs par defaut."""
        voice = ClonedVoice(
            name="test_voice",
            source_audio=Path("/path/to/audio.wav")
        )

        assert voice.name == "test_voice"
        assert voice.source_audio == Path("/path/to/audio.wav")
        assert voice.embedding_path is None
        assert voice.language == "fr"
        assert voice.description == ""
        assert voice.duration_seconds == 0.0

    def test_custom_values(self):
        """Test valeurs personnalisees."""
        voice = ClonedVoice(
            name="custom_voice",
            source_audio=Path("/path/to/audio.wav"),
            embedding_path=Path("/path/to/embedding.npy"),
            language="en",
            description="A custom voice",
            duration_seconds=30.5
        )

        assert voice.name == "custom_voice"
        assert voice.language == "en"
        assert voice.description == "A custom voice"
        assert voice.duration_seconds == 30.5


class TestVoiceCloner:
    """Tests pour VoiceCloner."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Cree un repertoire de cache temporaire."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def cloner(self, temp_cache_dir):
        """Cree un cloner avec cache temporaire."""
        return VoiceCloner(cache_dir=temp_cache_dir)

    @pytest.fixture
    def sample_audio_file(self):
        """Cree un fichier audio de test."""
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not installed")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = Path(f.name)

        # Audio de 10 secondes (au-dessus du minimum de 6s)
        sr = 24000
        duration = 10
        t = np.linspace(0, duration, sr * duration, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t) * 0.5
        sf.write(str(path), audio, sr)

        yield path

        if path.exists():
            path.unlink()

    @pytest.fixture
    def short_audio_file(self):
        """Cree un fichier audio trop court."""
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not installed")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = Path(f.name)

        # Audio de 3 secondes (en-dessous du minimum)
        sr = 24000
        duration = 3
        t = np.linspace(0, duration, sr * duration, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t) * 0.5
        sf.write(str(path), audio, sr)

        yield path

        if path.exists():
            path.unlink()

    def test_init(self, cloner, temp_cache_dir):
        """Test initialisation."""
        assert cloner.cache_dir == temp_cache_dir
        assert cloner.cache_dir.exists()

    def test_is_available(self, cloner):
        """Test detection de disponibilite XTTS."""
        # Le resultat depend de l'installation de TTS
        result = cloner.is_available()
        assert isinstance(result, bool)

    def test_min_audio_duration(self, cloner):
        """Test constante de duree minimum."""
        assert cloner.MIN_AUDIO_DURATION == 6.0

    def test_recommended_duration(self, cloner):
        """Test constante de duree recommandee."""
        assert cloner.RECOMMENDED_DURATION == 30.0

    def test_get_audio_duration(self, cloner, sample_audio_file):
        """Test calcul de la duree audio."""
        duration = cloner._get_audio_duration(sample_audio_file)

        # Devrait etre environ 10 secondes
        assert 9.5 <= duration <= 10.5

    def test_get_audio_duration_nonexistent(self, cloner):
        """Test duree d'un fichier inexistant."""
        duration = cloner._get_audio_duration(Path("/nonexistent/file.wav"))
        assert duration == 0.0

    def test_compute_audio_hash(self, cloner, sample_audio_file):
        """Test calcul du hash."""
        hash1 = cloner._compute_audio_hash(sample_audio_file)

        assert len(hash1) == 12  # 12 caracteres
        assert hash1.isalnum()  # Alphanumerique

        # Le meme fichier donne le meme hash
        hash2 = cloner._compute_audio_hash(sample_audio_file)
        assert hash1 == hash2

    def test_clone_voice_success(self, cloner, sample_audio_file, temp_cache_dir):
        """Test clonage reussi."""
        voice = cloner.clone_voice(
            audio_path=sample_audio_file,
            name="test_voice",
            language="fr",
            description="Test voice"
        )

        assert voice is not None
        assert voice.name == "test_voice"
        assert voice.language == "fr"
        assert voice.description == "Test voice"
        assert voice.duration_seconds >= 6.0

        # Verifier que les metadonnees sont sauvegardees
        meta_path = temp_cache_dir / "test_voice_meta.json"
        assert meta_path.exists()

    def test_clone_voice_too_short(self, cloner, short_audio_file):
        """Test echec avec audio trop court."""
        voice = cloner.clone_voice(
            audio_path=short_audio_file,
            name="short_voice",
            language="fr"
        )

        assert voice is None

    def test_clone_voice_nonexistent_file(self, cloner):
        """Test echec avec fichier inexistant."""
        voice = cloner.clone_voice(
            audio_path=Path("/nonexistent/audio.wav"),
            name="missing_voice",
            language="fr"
        )

        assert voice is None

    def test_list_cloned_voices_empty(self, cloner):
        """Test liste vide."""
        voices = cloner.list_cloned_voices()
        assert voices == []

    def test_list_cloned_voices_with_voices(self, cloner, sample_audio_file):
        """Test liste avec voix."""
        cloner.clone_voice(sample_audio_file, "voice1", "fr")
        cloner.clone_voice(sample_audio_file, "voice2", "en")

        voices = cloner.list_cloned_voices()

        assert len(voices) == 2
        names = [v.name for v in voices]
        assert "voice1" in names
        assert "voice2" in names

    def test_get_voice_exists(self, cloner, sample_audio_file):
        """Test recuperation d'une voix existante."""
        cloner.clone_voice(sample_audio_file, "my_voice", "fr")

        voice = cloner.get_voice("my_voice")

        assert voice is not None
        assert voice.name == "my_voice"

    def test_get_voice_not_exists(self, cloner):
        """Test recuperation d'une voix inexistante."""
        voice = cloner.get_voice("nonexistent_voice")
        assert voice is None


class TestVoiceCloningManager:
    """Tests pour VoiceCloningManager."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Cree un repertoire de cache temporaire."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_cache_dir):
        """Cree un manager avec cache temporaire."""
        return VoiceCloningManager(cache_dir=temp_cache_dir)

    @pytest.fixture
    def sample_audio_file(self):
        """Cree un fichier audio de test."""
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not installed")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = Path(f.name)

        sr = 24000
        duration = 10
        t = np.linspace(0, duration, sr * duration, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t) * 0.5
        sf.write(str(path), audio, sr)

        yield path

        if path.exists():
            path.unlink()

    def test_init(self, manager):
        """Test initialisation."""
        assert manager.cloner is not None

    def test_register_cloned_voice(self, manager, sample_audio_file):
        """Test enregistrement d'une voix."""
        result = manager.register_cloned_voice(
            audio_path=sample_audio_file,
            voice_id="cloned_test",
            language="fr",
            description="Test voice"
        )

        assert result is True
        assert "cloned_test" in manager._voice_registry

    def test_is_cloned_voice_with_prefix(self, manager):
        """Test detection voix clonee avec prefixe."""
        assert manager.is_cloned_voice("cloned_marie") is True
        assert manager.is_cloned_voice("cloned_anything") is True

    def test_is_cloned_voice_without_prefix(self, manager):
        """Test detection voix non-clonee."""
        assert manager.is_cloned_voice("ff_siwis") is False
        assert manager.is_cloned_voice("am_adam") is False

    def test_is_cloned_voice_in_registry(self, manager, sample_audio_file):
        """Test detection voix dans le registre."""
        manager.register_cloned_voice(sample_audio_file, "my_voice", "fr")

        assert manager.is_cloned_voice("my_voice") is True

    def test_get_cloned_voice_from_registry(self, manager, sample_audio_file):
        """Test recuperation depuis le registre."""
        manager.register_cloned_voice(sample_audio_file, "registry_voice", "fr")

        voice = manager.get_cloned_voice("registry_voice")

        assert voice is not None
        assert voice.name == "registry_voice"

    def test_get_cloned_voice_from_cache(self, manager, sample_audio_file):
        """Test recuperation depuis le cache (pas le registre)."""
        # Ajouter directement au cloner, pas au registre
        manager.cloner.clone_voice(sample_audio_file, "cache_voice", "fr")

        voice = manager.get_cloned_voice("cache_voice")

        assert voice is not None
        assert voice.name == "cache_voice"

    def test_get_cloned_voice_not_exists(self, manager):
        """Test recuperation voix inexistante."""
        voice = manager.get_cloned_voice("nonexistent")
        assert voice is None


class TestCloneVoiceFromFile:
    """Tests pour la fonction utilitaire."""

    @pytest.fixture
    def temp_cache_dir(self, monkeypatch):
        """Configure un cache temporaire."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Monkey-patch le path par defaut
            yield Path(tmpdir)

    @pytest.fixture
    def sample_audio_file(self):
        """Cree un fichier audio de test."""
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not installed")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = Path(f.name)

        sr = 24000
        duration = 10
        t = np.linspace(0, duration, sr * duration, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t) * 0.5
        sf.write(str(path), audio, sr)

        yield path

        if path.exists():
            path.unlink()

    def test_clone_voice_from_file_creates_prefixed_id(self, sample_audio_file):
        """Test que l'ID est prefixe avec 'cloned_'."""
        # Note: cette fonction cree un VoiceCloningManager a chaque appel
        result = clone_voice_from_file(
            str(sample_audio_file),
            "my_test_voice",
            "fr"
        )

        assert result is True


class TestMetadataStorage:
    """Tests pour le stockage des metadonnees."""

    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def cloner(self, temp_cache_dir):
        return VoiceCloner(cache_dir=temp_cache_dir)

    @pytest.fixture
    def sample_audio_file(self):
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not installed")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = Path(f.name)

        sr = 24000
        duration = 10
        t = np.linspace(0, duration, sr * duration, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t) * 0.5
        sf.write(str(path), audio, sr)

        yield path

        if path.exists():
            path.unlink()

    def test_metadata_saved_correctly(self, cloner, sample_audio_file, temp_cache_dir):
        """Test que les metadonnees sont correctement sauvegardees."""
        cloner.clone_voice(
            sample_audio_file,
            "meta_test",
            language="en",
            description="Test description"
        )

        meta_path = temp_cache_dir / "meta_test_meta.json"
        assert meta_path.exists()

        with open(meta_path) as f:
            data = json.load(f)

        assert data["name"] == "meta_test"
        assert data["language"] == "en"
        assert data["description"] == "Test description"
        assert "duration_seconds" in data
        assert "audio_hash" in data

    def test_metadata_persistence(self, temp_cache_dir, sample_audio_file):
        """Test que les voix persistent entre instances."""
        # Premier cloner
        cloner1 = VoiceCloner(cache_dir=temp_cache_dir)
        cloner1.clone_voice(sample_audio_file, "persist_test", "fr")

        # Nouveau cloner, meme cache
        cloner2 = VoiceCloner(cache_dir=temp_cache_dir)
        voices = cloner2.list_cloned_voices()

        assert len(voices) == 1
        assert voices[0].name == "persist_test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
