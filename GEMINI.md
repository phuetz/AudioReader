# AudioReader Project Context

## Project Overview
**AudioReader** (v3.0) is a comprehensive Python-based platform for converting books into high-quality audiobooks. It leverages multiple TTS engines including **Kokoro-82M** (via `onnxruntime`), **MMS-TTS** (Meta), and **XTTS-v2** (voice cloning) to provide local, free, and high-fidelity text-to-speech synthesis that rivals commercial services like ElevenLabs.

The project focuses on "broadcast quality" output, implementing features like:
*   **Multi-voice Support:** Automatic character detection and voice assignment.
*   **Emotional Intelligence:** Analyzes text for sentiment and adjusts prosody accordingly.
*   **Audio Mastering:** Integrated post-processing pipeline (EQ, compression, loudness normalization to -19 LUFS).
*   **Voice Cloning (v2.4):** Clone any voice with just 6 seconds of audio using XTTS-v2.
*   **Universal Import (v3.0):** Accept PDF, EPUB, or Markdown as input.
*   **Podcast Server (v3.0):** Stream audiobooks via RSS feed with QR code.
*   **Video Voice Extraction (v3.0):** Extract voice samples from videos for cloning.

## Key Files & Entry Points

*   **`audio_reader.py`**: Unified CLI (standard mode and HQ mode with `--hq`).
*   **`app.py`**: Gradio web interface with all v3.0 features (import, cloning, podcast).
*   **`api_server.py`**: REST API for ChatGPT/external integration.
*   **`mcp_server.py`**: MCP server for Claude Desktop integration.
*   **`src/hq_pipeline_extended.py`**: The advanced pipeline (v2.1-2.4) with all features.
*   **`src/tts_kokoro_engine.py`**: Kokoro-82M ONNX wrapper.
*   **`src/tts_xtts_engine.py`**: XTTS-v2 voice cloning engine.
*   **`requirements.txt`**: Python dependencies.
*   **`README.md`**: Comprehensive documentation (French).
*   **`README_EN.md`**: English documentation.
*   **`CLAUDE.md`**: Claude Code AI context file.

## Architecture

The project follows a modular architecture within the `src/` directory (51 modules):

### TTS Engines (7 modules)
*   `tts_engine.py` - Unified wrapper (auto-selection)
*   `tts_kokoro_engine.py` - Kokoro-82M (ElevenLabs quality)
*   `tts_mms_engine.py` - MMS-TTS Meta (1000+ languages)
*   `tts_xtts_engine.py` - XTTS-v2 (voice cloning)
*   `tts_hybrid_engine.py` - Hybrid MMS+Kokoro + MP3 export
*   `tts_multivoice_xtts.py` - Multi-voice XTTS
*   `tts_unified.py` - TTS abstraction

### Text Processing (5 modules)
*   `markdown_parser.py` - MD, EPUB, multi-file parser
*   `text_normalizer.py` - Numbers, dates, symbols -> words
*   `text_processor.py` - Chunking, pronunciation corrections
*   `french_preprocessor.py` - French-specific normalization
*   `advanced_preprocessor.py` - Complete preprocessing pipeline

### Character & Emotion Analysis (8 modules)
*   `character_detector.py` - Dialogue and speaker detection
*   `dialogue_detector.py` - Dialogue/narration segmentation
*   `dialogue_attribution.py` - WHO is speaking attribution
*   `emotion_analyzer.py` - Sentiment analysis + prosody
*   `emotion_continuity.py` - Smooth emotional transitions
*   `emotion_control.py` - Intensity control + IPA phonemes
*   `narrative_context.py` - Narrative type detection
*   `narration_styles.py` - 7 styles (storytelling, dramatic...)

### Bio-Acoustics & Prosody (6 modules)
*   `bio_acoustics.py` - Realistic synthetic breathing
*   `breath_samples.py` - Hybrid samples/synthesis
*   `intonation_contour.py` - Phrase-level prosodic contours
*   `timing_humanizer.py` - Timing micro-variations
*   `dynamic_voice.py` - Emotion-based voice blending
*   `word_level_control.py` - SSML-like word-by-word control

### Audio Processing (5 modules)
*   `audio_enhancer.py` - EQ, compression, loudness (ffmpeg)
*   `audio_postprocess.py` - Modular post-prod pipeline
*   `audio_processor.py` - In-memory processing (numpy)
*   `audio_crossfade.py` - Crossfade between segments
*   `acx_compliance.py` - ACX/Audible compliance

### v3.0 Platform Modules (3 modules)
*   `input_converter.py` - PDF/EPUB -> Markdown
*   `audio_extractor.py` - Video -> WAV (ffmpeg)
*   `podcast_server.py` - Local RSS server + QR code

## Setup & Usage

### Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# v3.0 dependencies (PDF, EPUB, Podcast)
pip install pymupdf ebooklib beautifulsoup4 qrcode fpdf2

# Download Kokoro model
curl -L -o kokoro-v1.0.onnx "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
curl -L -o voices-v1.0.bin "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
```

### Web Interface (Recommended)
```bash
python app.py
# Open http://localhost:7860
```

### CLI Usage
```bash
# Standard conversion
python audio_reader.py my_book.md

# HQ mode with all features
python audio_reader.py my_book.md --hq --multivoice --master

# Voice cloning (requires XTTS venv)
python audio_reader.py my_book.md --engine xtts --clone my_voice.wav

# List voices
python audio_reader.py --list-voices
```

### XTTS Voice Cloning Setup
```bash
# Create Python 3.10 venv (TTS requires 3.10/3.11)
python3.10 -m venv venv_xtts
source venv_xtts/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install TTS
pip install "transformers>=4.33.0,<4.50.0"
```

## Development Conventions

*   **Code Style:** Python 3.10+ typing hints.
*   **Audio Standards:** ACX/Audible compliance (-19 LUFS, -3dB peak).
*   **Testing:** `pytest` via `python run_tests.py`.
*   **Dependencies:** `numpy` for audio buffers, `soundfile`/`wave` for I/O.

## Version History

*   **v2.0**: HQ Pipeline, multi-voice, emotion analysis
*   **v2.1**: Audio tags, voice morphing, XTTS cloning, cache
*   **v2.2**: MP3 export, book exporter (PDF/EPUB/HTML)
*   **v2.3**: Bio-acoustics, intonation contours, timing humanization
*   **v2.4**: XTTS engine, crossfade, ACX compliance, LLM emotion detection
*   **v3.0**: Complete platform (PDF/EPUB import, video extraction, podcast server, Gradio UI)
