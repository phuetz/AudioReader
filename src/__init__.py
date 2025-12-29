"""AudioReader - Convertisseur de livres Markdown en audio."""
from .markdown_parser import MarkdownBookParser, Chapter, parse_book
from .tts_engine import (
    create_tts_engine,
    UnifiedTTSEngine,
    EdgeTTSEngine,
    Pyttsx3Engine,
    EngineType,
    TTSEngine,  # Alias pour compatibilité
)

__all__ = [
    # Parser
    'MarkdownBookParser',
    'Chapter',
    'parse_book',
    # TTS
    'create_tts_engine',
    'UnifiedTTSEngine',
    'TTSEngine',
    'EdgeTTSEngine',
    'Pyttsx3Engine',
    'EngineType',
]
