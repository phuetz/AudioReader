"""
Configuration pytest pour les tests AudioReader.

Fournit des fixtures communes et la configuration du path.
"""
import pytest
import sys
from pathlib import Path
import tempfile
import numpy as np

# Ajouter le repertoire src au path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))


@pytest.fixture
def temp_dir():
    """Cree un repertoire temporaire pour les tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_audio():
    """Cree un signal audio de test (1 seconde, 24kHz)."""
    sr = 24000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    # Signal sinusoidal 440 Hz
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    return audio


@pytest.fixture
def sample_audio_long():
    """Cree un signal audio de test long (10 secondes)."""
    sr = 24000
    duration = 10.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    return audio


@pytest.fixture
def sample_text_fr():
    """Texte francais de test."""
    return """
    Chapitre 1 : Le commencement

    Il etait une fois, dans un pays lointain, une jeune femme nommee Marie.
    Elle vivait dans une petite maison pres de la foret.

    « Bonjour ! » s'exclama-t-elle en ouvrant la porte.

    Pierre, son voisin, lui repondit : « Belle journee, n'est-ce pas ? »

    Marie sourit. Le soleil brillait et les oiseaux chantaient.
    C'etait le debut d'une grande aventure.
    """


@pytest.fixture
def sample_text_en():
    """Texte anglais de test."""
    return """
    Chapter 1: The Beginning

    Once upon a time, in a faraway land, there lived a young woman named Mary.
    She lived in a small house near the forest.

    "Hello!" she exclaimed, opening the door.

    Peter, her neighbor, replied: "Beautiful day, isn't it?"

    Mary smiled. The sun was shining and the birds were singing.
    It was the beginning of a great adventure.
    """


@pytest.fixture
def sample_script():
    """Script de dialogue de test."""
    return """
    NARRATEUR: Il etait une fois...
    MARIE: Bonjour Pierre !
    PIERRE: Salut Marie, comment vas-tu ?
    MARIE: [excited] Tres bien ! J'ai une grande nouvelle !
    PIERRE: [surprised] Ah oui ? Raconte-moi !
    NARRATEUR: Et ainsi commenca leur conversation.
    """


@pytest.fixture
def sample_audio_file(temp_dir):
    """Cree un fichier audio temporaire."""
    try:
        import soundfile as sf
    except ImportError:
        pytest.skip("soundfile not installed")

    path = temp_dir / "test_audio.wav"
    sr = 24000
    duration = 10.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    sf.write(str(path), audio, sr)

    return path


def pytest_configure(config):
    """Configuration pytest."""
    # Ajouter des markers personnalises
    config.addinivalue_line(
        "markers", "slow: marque les tests lents"
    )
    config.addinivalue_line(
        "markers", "requires_librosa: tests necessitant librosa"
    )
    config.addinivalue_line(
        "markers", "requires_tts: tests necessitant TTS (XTTS)"
    )
    config.addinivalue_line(
        "markers", "requires_soundfile: tests necessitant soundfile"
    )


def pytest_collection_modifyitems(config, items):
    """Modifie la collection de tests."""
    # Verifier les dependances optionnelles
    try:
        import librosa
        has_librosa = True
    except ImportError:
        has_librosa = False

    try:
        import soundfile
        has_soundfile = True
    except ImportError:
        has_soundfile = False

    try:
        from TTS.api import TTS
        has_tts = True
    except ImportError:
        has_tts = False

    skip_librosa = pytest.mark.skip(reason="librosa not installed")
    skip_soundfile = pytest.mark.skip(reason="soundfile not installed")
    skip_tts = pytest.mark.skip(reason="TTS not installed")

    for item in items:
        if "requires_librosa" in item.keywords and not has_librosa:
            item.add_marker(skip_librosa)
        if "requires_soundfile" in item.keywords and not has_soundfile:
            item.add_marker(skip_soundfile)
        if "requires_tts" in item.keywords and not has_tts:
            item.add_marker(skip_tts)
