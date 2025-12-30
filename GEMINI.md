# AudioReader Project Context

## Project Overview
**AudioReader** is a sophisticated Python-based tool designed to convert Markdown books into high-quality audiobooks. It leverages the **Kokoro-82M** TTS model (via `onnxruntime`) to provide local, free, and high-fidelity text-to-speech synthesis that rivals commercial services like ElevenLabs.

The project focuses on "broadcast quality" output, implementing features like:
*   **Multi-voice Support:** Automatic character detection and voice assignment (Narrator, Alice, Bob, etc.).
*   **Emotional Intelligence:** Analyzes text for sentiment (joy, fear, anger) and adjusts prosody (speed, pitch) accordingly.
*   **Audio Mastering:** integrated post-processing pipeline (EQ, compression, loudness normalization to -19 LUFS).
*   **Advanced TTS Features (v2.1):** Voice morphing, voice cloning (XTTS-v2), and expressive audio tags (e.g., `[whisper]`, `[laugh]`).

## Key Files & Entry Points

*   **`audio_reader.py`**: The primary Command Line Interface (CLI) for standard usage. Handles book parsing, voice selection, and audio generation.
*   **`src/hq_pipeline_extended.py`**: The "brain" of the v2.1 update. This module implements the advanced pipeline including caching, parallelization, and extended audio features.
*   **`src/tts_kokoro_engine.py`**: The core wrapper for the Kokoro-82M ONNX model.
*   **`generate_audiobook.py`**: A reference script demonstrating how to programmatically use the HQ pipeline for a specific book project ("Les Conquérants...").
*   **`requirements.txt`**: Python dependencies (`kokoro-onnx`, `soundfile`, `ffmpeg-normalize`, etc.).
*   **`README.md`**: Comprehensive documentation on features and usage.
*   **`AMELIORATIONS.md`**: Roadmap and technical notes on implemented/planned audio improvements.

## Architecture

The project follows a modular architecture within the `src/` directory:

*   **TTS Engines:** `tts_engine.py` (base), `tts_kokoro_engine.py`, `tts_unified.py`.
*   **Text Processing:** `text_normalizer.py`, `markdown_parser.py`, `character_detector.py`.
*   **Audio Processing:** `audio_enhancer.py` (mastering), `voice_morphing.py`, `audio_tags.py`.
*   **Pipeline:** `hq_pipeline.py` (v2.0 base), `hq_pipeline_extended.py` (v2.1 advanced).

## Setup & Usage

### Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Requires ffmpeg installed on the system
```

### Basic CLI Usage
```bash
# Convert a book with default settings
python audio_reader.py my_book.md

# Specify language and engine
python audio_reader.py my_book.md --language en --engine kokoro --voice af_heart

# List available voices
python audio_reader.py --list-voices
```

### Advanced/HQ Usage
To utilize the full v2.1 pipeline (voice cloning, advanced emotions), it is recommended to interface programmatically with `src.hq_pipeline_extended` or use a custom driver script like `generate_audiobook.py`.

## Development Conventions

*   **Code Style:** Python 3.10+ typing hints are used extensively.
*   **Audio Standards:** The project aims for ACX/Audible technical standards (-19 LUFS, -3dB peak, 192kbps MP3/M4B).
*   **Testing:** `pytest` is used for testing (config in `pytest.ini`).
*   **Dependencies:** Heavy reliance on `numpy` for audio buffer manipulation and `soundfile`/`wave` for I/O.
