# AudioReader

**Convert Markdown books to high-quality audiobooks**

Powered by [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) - an open-source TTS model that rivals ElevenLabs, 100% free and local.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)

---

## Table of Contents

- [Features](#-features)
- [What's New in v2.3 - ElevenLabs Style](#-whats-new-in-v23---elevenlabs-style)
- [What's New in v2.2 - Multi-Format Export](#-whats-new-in-v22---multi-format-export)
- [What's New in v2.1 - Advanced Features](#-whats-new-in-v21---advanced-features)
- [What's New in v2.0 - HQ Pipeline](#-whats-new-in-v20---hq-pipeline)
- [Installation](#-installation)
- [Usage](#-usage)
- [High Quality Pipeline](#-high-quality-pipeline)
- [Architecture](#-architecture)
- [Audio Quality](#-audio-quality)
- [Distribution](#-distribution)
- [Scientific References](#-scientific-references)
- [Sources and Credits](#-sources-and-credits)

---

## Features

### Speech Synthesis
- **Kokoro-82M**: 82 million parameter model, ElevenLabs-quality
- **Multilingual**: French, English (US/UK), Japanese, Chinese, and more
- **Natural voices**: Realistic intonation and prosody
- **Voice Blending**: Mix voices with weights (e.g., `af_bella:60,am_adam:40`)
- **Auto multi-voice**: Different voices for detected characters
- **Performance**: ~5x real-time on CPU

### Input Formats
- **Markdown**: Single file with headers for chapters
- **Multi-file**: Directory with one .md file per chapter
- **EPUB**: Automatic chapter extraction

### Text Processing
- **Smart chunking**: Splitting at natural boundaries (sentences, paragraphs)
- **Smart pauses**: Configurable pauses between sentences (0.3s) and paragraphs (0.8s)
- **Pronunciation fixes**: Built-in dictionary for acronyms and technical terms
- **Advanced normalization**: Numbers, dates, times, currencies, roman numerals
- **Emotion tags**: Support for `[laugh]`, `[sigh]`, `[cough]`, etc.

### Audio Post-processing
- **Loudness normalization**: EBU R128, -19 LUFS (podcast/audiobook standard)
- **Auto EQ**: Presence (3kHz) + Air (12kHz) + Low-cut (80Hz)
- **De-essing**: Sibilance reduction
- **Compression**: Gentle voice compression
- **Limiter**: True peak at -1.5 dB

### Professional Export
- **ID3 Metadata**: Title, author, narrator, cover art
- **M4B Format**: Audiobook with navigable chapters (Apple Books compatible)
- **MP3 Export**: Separate or combined files
- **Book Export**: PDF, EPUB, HTML, TXT

### Interface
- **Complete CLI**: Command-line scripts with advanced options
- **Web Interface**: Intuitive Gradio application
- **HQ Pipeline**: Dedicated script for maximum quality

---

## What's New in v2.3 - ElevenLabs Style

### Realistic Breathing

New breath generation based on bio-acoustic research:

- **Pink noise** instead of white noise (more natural spectrum)
- **Respiratory formants** (vocal tract resonances)
- **Amplitude jitter** (natural micro-variations)
- **Optional sample support** (hybrid: synthesis + real samples)

### Phrase-Level Intonation Contours

Automatic detection and application of prosodic patterns:

| Type | Pattern | Example |
|------|---------|---------|
| DECLARATIVE | Final descent (-2 to -4 st) | "He left." |
| QUESTION_YN | Final rise (+3 to +5 st) | "Are you coming?" |
| QUESTION_WH | Initial peak then descent | "Where are you going?" |
| EXCLAMATION | Strong peak then rapid descent | "Incredible!" |
| CONTINUATION | Slight rise (+1 st) | "First, ..." |
| SUSPENSE | Slow descent, pause | "And then..." |

### Timing Micro-Variations

Rhythm humanization to avoid the "robotic" effect:

- **Gaussian pause variation** (±5% by default)
- **Adaptive rhythm** based on syntactic structure
- **Micro-pauses before important words** (0.05s)

```python
# v2.3 Configuration
config = ExtendedPipelineConfig(
    enable_advanced_breaths=True,
    enable_intonation_contours=True,
    intonation_strength=0.7,
    enable_timing_humanization=True,
    pause_variation_sigma=0.05,
    enable_emphasis_pauses=True,
)
```

---

## What's New in v2.2 - Multi-Format Export

### MP3 Audio Export

The hybrid engine now supports direct MP3 export:

```python
from src.tts_hybrid_engine import HybridTTSEngine

engine = HybridTTSEngine(mms_language='fra')

# WAV export (default)
engine.synthesize_chapter(text, "chapter.wav")

# MP3 export
engine.synthesize_chapter(text, "chapter.mp3", output_format="mp3")

# MP3 with custom bitrate
engine.synthesize_chapter(text, "chapter.mp3", output_format="mp3", mp3_bitrate="256k")
```

### Book Export (PDF, EPUB, HTML, TXT)

New `book_exporter.py` module to export your books in multiple formats:

```python
from src.book_exporter import BookExporter, export_markdown_book

# Quick export from a Markdown chapters folder
results = export_markdown_book(
    chapters_dir="path/to/chapters",
    output_dir="output/ebook",
    title="My Book",
    author="Author",
    formats=["pdf", "epub", "html", "txt"]
)

# Or advanced usage
exporter = BookExporter(title="My Book", author="Author")
exporter.add_chapter("Chapter 1", "Chapter content...")
exporter.add_chapter_from_markdown("chapter-02.md")

exporter.export_pdf("book.pdf")
exporter.export_epub("book.epub")
exporter.export_html("book.html")
exporter.export_txt("book.txt")
```

| Format | Extension | Description |
|--------|-----------|-------------|
| **PDF** | `.pdf` | Portable document with table of contents |
| **EPUB** | `.epub` | Standard ebook format (Kindle, Kobo, e-readers) |
| **HTML** | `.html` | Web page with automatic dark mode |
| **TXT** | `.txt` | Formatted plain text |

---

## What's New in v2.0 - HQ Pipeline

### Automatic Multi-Voice by Character

The system automatically detects characters and assigns different voices:

```
Text: "Hello!" said Marie. Pierre replied: "How are you?"

Result:
  NARRATOR   -> ff_siwis (default voice)
  Marie      -> af_bella (auto-assigned female voice)
  Pierre     -> am_adam  (auto-assigned male voice)
```

**Automatic detection via:**
- Quotes: `"text"`, `"text"`
- Dialogue dashes: `- text`
- Speech verbs: "said", "replied", "whispered", "shouted"... (+100 verbs EN/FR)

### Advanced Text Normalization

| Type | Before | After |
|------|--------|-------|
| Numbers | `1234` | "one thousand two hundred thirty-four" |
| Dates | `12/25/2024` | "December twenty-fifth, two thousand twenty-four" |
| Times | `2:30 PM` | "two thirty PM" |
| Currencies | `$1,234.56` | "one thousand two hundred... dollars and fifty-six cents" |
| Roman | `Louis XIV` | "Louis the fourteenth" |
| Percentages | `85%` | "eighty-five percent" |

### Narrative Context Detection

The system adapts speed based on content type:

| Context | Speed | Description |
|---------|-------|-------------|
| Action | 1.15x | Fast-paced scenes, tension |
| Description | 0.90x | Descriptive passages |
| Introspection | 0.92x | Inner thoughts |
| Flashback | 0.88x | Memories, dreamy tone |
| Suspense | 0.88x | Tension, long pauses |
| Dialogue | 1.00x | Character speech |

### Emotional Analysis

Automatic emotion detection with adapted prosody:

- **Joy**: Speed +10%, pitch +0.5
- **Sadness**: Speed -10%, pitch -0.3, long pauses
- **Anger**: Speed +15%, volume +20%
- **Fear**: Speed +20%, breathing micro-pauses
- **Suspense**: Speed -15%, long pauses

---

## What's New in v2.1 - Advanced Features

### ElevenLabs v3 Style Audio Tags

Support for expressive tags directly in text:

```
[whispers] I have to tell you something...
[excited] This is incredible! [laugh]
[dramatic] [pause] And then... everything changed.
[sarcastic] Oh, what a surprise...
```

**Supported tags:**

| Category | Tags |
|----------|------|
| Emotions | `[excited]`, `[sad]`, `[angry]`, `[whispers]`, `[fearful]`, `[tender]`, `[dramatic]` |
| Actions | `[sigh]`, `[laugh]`, `[chuckle]`, `[gasp]`, `[cough]`, `[yawn]`, `[sniff]` |
| Pauses | `[pause]`, `[long pause]`, `[beat]`, `[silence]` |
| Styles | `[sarcastic]`, `[cheerful]`, `[serious]`, `[mysterious]`, `[narrator]`, `[announcer]` |

### Voice Morphing

Real-time voice modification:

```python
from src.voice_morphing import VoiceMorpher, VoicePresets

morpher = VoiceMorpher()

# Available presets
preset = VoicePresets.get("more_masculine")  # Deeper voice
preset = VoicePresets.get("younger")         # Younger voice
preset = VoicePresets.get("whisper")         # Whisper
preset = VoicePresets.get("expressive")      # More expressive
```

### Voice Cloning (XTTS-v2)

Clone any voice with just 6 seconds of audio:

```python
from src.voice_cloning import VoiceCloningManager

manager = VoiceCloningManager()

# Clone a voice
manager.register_cloned_voice(
    audio_path="sample.wav",
    voice_id="cloned_marie",
    language="en"
)

# Use the cloned voice
manager.synthesize_with_cloned_voice(
    text="Hello everyone!",
    voice_id="cloned_marie",
    output_path="output.wav"
)
```

### Smart Cache and Parallelization

Performance optimization:

```python
from src.synthesis_cache import SynthesisCache, ParallelSynthesizer

# Cache to avoid regenerating identical segments
cache = SynthesisCache(max_size_mb=1000)

# Parallelization across multiple cores
synth = ParallelSynthesizer(num_workers=4, cache=cache)
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- FFmpeg (for audio processing)
- ~500 MB disk space (model)

### Quick Install

```bash
# Clone the project
git clone https://github.com/your-repo/AudioReader.git
cd AudioReader

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download Kokoro model (~340 MB)
curl -L -o kokoro-v1.0.onnx "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
curl -L -o voices-v1.0.bin "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
```

### Verify Installation

```bash
python audioreader.py --list-voices
```

---

## Usage

### Standard Mode (audioreader.py)

```bash
# Convert a Markdown book
python audioreader.py my_book.md

# With an English voice
python audioreader.py book.md --voice af_heart

# With professional audio mastering
python audioreader.py book.md --mastering

# Export to M4B with chapters
python audioreader.py book.md --format m4b --title "My Book" --author "Author"
```

### High Quality Mode (audioreader_hq.py)

```bash
# Full HQ pipeline (multi-voice + emotions + post-processing)
python audioreader_hq.py book.md -o output/

# Analysis only (no audio generation)
python audioreader_hq.py book.md --analyze-only

# With custom configuration
python audioreader_hq.py book.md --config config_hq.json

# Available options
python audioreader_hq.py book.md \
    --lang en \
    --voice af_heart \
    --speed 1.0 \
    --no-emotions      # Disable emotional analysis
    --no-characters    # Disable multi-voice
    --no-enhance       # Disable post-processing
```

### Custom Multi-Voice Configuration

Create a `config_hq.json` file:

```json
{
  "lang": "en",
  "narrator_voice": "af_heart",
  "voice_mapping": {
    "Marie": "af_bella",
    "Peter": "am_adam",
    "The doctor": "bm_george"
  },
  "auto_assign_voices": true,
  "enable_emotion_analysis": true,
  "enable_narrative_context": true,
  "sentence_pause": 0.3,
  "paragraph_pause": 0.8,
  "target_lufs": -19.0
}
```

### Voice Blending

```bash
# Blend 60% Bella, 40% Adam
python audioreader.py book.md --voice "af_bella:60,am_adam:40"

# 50-50 blend
python audioreader.py book.md --voice "af_bella,am_adam"
```

### Pause Control

```bash
# Longer pauses between paragraphs (1 second)
python audioreader.py book.md --paragraph-pause 1.0

# Shorter pauses between sentences (0.2 seconds)
python audioreader.py book.md --sentence-pause 0.2
```

### Web Interface

```bash
python audioreader.py --gui
# Open http://localhost:7860
```

### Available Voices

| Code | Name | Language | Gender |
|------|------|----------|--------|
| `ff_siwis` | Siwis | French | F |
| `af_heart` | Heart | English US | F |
| `af_bella` | Bella | English US | F |
| `af_nicole` | Nicole | English US | F |
| `af_sarah` | Sarah | English US | F |
| `am_adam` | Adam | English US | M |
| `am_michael` | Michael | English US | M |
| `am_eric` | Eric | English US | M |
| `bf_emma` | Emma | English UK | F |
| `bm_george` | George | English UK | M |
| `jf_alpha` | Alpha | Japanese | F |
| `zf_xiaoxiao` | Xiaoxiao | Chinese | F |

---

## High Quality Pipeline

### Overview

```
                           HQ PIPELINE
    +----------------------------------------------------------+
    |                                                          |
    |   RAW TEXT                                               |
    |       |                                                  |
    |       v                                                  |
    |   [Normalization]  numbers, dates, abbreviations         |
    |       |                                                  |
    |       v                                                  |
    |   [Character Detection]  dialogues, speech verbs         |
    |       |                                                  |
    |       v                                                  |
    |   [Emotional Analysis]  sentiment, intensity             |
    |       |                                                  |
    |       v                                                  |
    |   [Narrative Context]  action, description, thoughts     |
    |       |                                                  |
    |       v                                                  |
    |   [Emotional Continuity]  transition smoothing           |
    |       |                                                  |
    |       v                                                  |
    |   [Voice Assignment]  character -> voice                 |
    |       |                                                  |
    |       v                                                  |
    |   [Multi-Voice Synthesis]  Kokoro TTS                    |
    |       |                                                  |
    |       v                                                  |
    |   [Post-Processing]  EQ, compression, loudness           |
    |       |                                                  |
    |       v                                                  |
    |   HQ AUDIOBOOK                                           |
    |                                                          |
    +----------------------------------------------------------+
```

---

## Architecture

```
AudioReader/
├── audioreader.py          # Main CLI script (standard)
├── audioreader_hq.py       # High quality CLI script
├── app.py                  # Gradio interface
├── postprocess.py          # Legacy audio post-processing
├── example_multivoix.py    # Multi-voice example
├── kokoro-v1.0.onnx        # TTS model (310 MB)
├── voices-v1.0.bin         # Voice data (27 MB)
│
├── src/
│   ├── markdown_parser.py      # Multi-format parser (MD, EPUB)
│   ├── tts_kokoro_engine.py    # Kokoro TTS engine + multi-voice
│   ├── tts_engine.py           # Edge-TTS engine (fallback)
│   ├── text_processor.py       # Chunking, base pronunciation
│   ├── audiobook_builder.py    # Metadata, M4B/MP3 export
│   │
│   │   # --- HQ MODULES v2.0 ---
│   ├── text_normalizer.py      # Numbers, dates, symbols
│   ├── character_detector.py   # Character/dialogue detection
│   ├── emotion_analyzer.py     # Emotional analysis
│   ├── narrative_context.py    # Narrative context
│   ├── emotion_continuity.py   # Emotional continuity
│   ├── advanced_preprocessor.py # Advanced preprocessor
│   ├── hq_pipeline.py          # Unified HQ pipeline
│   ├── audio_enhancer.py       # Broadcast post-processing
│   │
│   │   # --- ADVANCED MODULES v2.1 ---
│   ├── audio_tags.py           # ElevenLabs v3 style tags
│   ├── voice_morphing.py       # Pitch, formant, time stretch
│   ├── voice_cloning.py        # XTTS-v2 cloning
│   ├── synthesis_cache.py      # Cache + parallelization
│   ├── emotion_control.py      # Emotion control + IPA phonemes
│   ├── conversation_generator.py # Multi-speaker dialogues
│   ├── hq_pipeline_extended.py # Extended unified pipeline
│   │
│   │   # --- MODULES v2.2-2.3 ---
│   ├── tts_hybrid_engine.py    # Hybrid MMS+Kokoro engine, MP3
│   ├── tts_mms_engine.py       # MMS-TTS engine (Facebook)
│   ├── book_exporter.py        # PDF, EPUB, HTML, TXT export
│   ├── bio_acoustics.py        # Realistic breathing
│   ├── intonation_contour.py   # Prosodic contours
│   └── timing_humanizer.py     # Rhythm micro-variations
│
├── config_multivoix_example.json  # Example config
├── books/                  # Source books
└── output/                 # Generated audiobooks
```

---

## Audio Quality

### Standard vs HQ Comparison

| Aspect | Standard | HQ Pipeline |
|--------|----------|-------------|
| Voices | Single | Auto multi-character |
| Numbers | "1234" (as-is) | "one thousand two hundred..." |
| Emotions | Fixed | Adaptive |
| Speed | Constant | Context-variable |
| Transitions | Possibly abrupt | Smoothed |
| Loudness | -20 LUFS | -19 LUFS |
| Post-processing | Basic | EQ + Compression + De-essing |

### Target Standards

| Parameter | Value | Standard |
|-----------|-------|----------|
| Loudness | -19 LUFS | EBU R128 / Podcast |
| True Peak | -1.5 dB max | Broadcast |
| Noise floor | < -60 dB | ACX / Audible |
| Sample rate | 44.1 kHz | CD Quality |
| Bitrate | 192 kbps | Distribution |

### Performance Estimates

| Book size | Est. audio | Conversion time* |
|-----------|------------|------------------|
| 50K characters | ~30 min | ~6 min |
| 200K characters | ~2h | ~25 min |
| 500K characters | ~5h | ~1h |
| 1M characters | ~10h | ~2h |

*On modern CPU (Intel i7/AMD Ryzen). HQ pipeline adds ~20% to standard time.

---

## Distribution

### Platforms Accepting AI Voices

| Platform | Status | Link |
|----------|--------|------|
| Google Play Books | Accepts | [play.google.com/books/publish](https://play.google.com/books/publish) |
| Findaway/Spotify | Accepts (ElevenLabs) | [findawayvoices.com](https://findawayvoices.com) |
| Kobo | Accepts | [kobo.com/writinglife](https://www.kobo.com/writinglife) |
| Direct sales | No restrictions | Your website |
| Audible/ACX | **Refuses** | Human voices only |

---

## Scientific References

### Neural Speech Synthesis

1. **A Survey on Neural Speech Synthesis** (2021)
   - Xu Tan, Tao Qin, Frank Soong, Tie-Yan Liu
   - [arXiv:2106.15561](https://arxiv.org/abs/2106.15561)

2. **Deep Learning-based Expressive Speech Synthesis** (2024)
   - EURASIP Journal on Audio, Speech, and Music Processing
   - [DOI: 10.1186/s13636-024-00329-7](https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-024-00329-7)

### Prosody and Emotion

3. **The Sound of Emotional Prosody** (2025)
   - Pauline Larrouy-Maestri, David Poeppel, Marc D. Pell
   - [DOI: 10.1177/17456916231217722](https://journals.sagepub.com/doi/10.1177/17456916231217722)

4. **Text-aware and Context-aware Expressive Audiobook Speech Synthesis** (2024)
   - Interspeech 2024
   - [arXiv:2406.05672](https://arxiv.org/html/2406.05672)

---

## Sources and Credits

### TTS Models

| Project | Description | License |
|---------|-------------|---------|
| **Kokoro-82M** | 82M params TTS model | Apache 2.0 |
| **kokoro-onnx** | ONNX runtime for Kokoro | MIT |
| **Edge-TTS** | Microsoft Edge TTS | LGPL-3.0 |
| **MMS-TTS** | Facebook multilingual TTS | CC-BY-NC |

### Audio Tools

| Project | Description |
|---------|-------------|
| **FFmpeg** | Audio/video processing |
| **ffmpeg-normalize** | Loudness normalization |
| **librosa** | Audio analysis |

---

## License

This project is under the Apache 2.0 license. See [LICENSE](LICENSE) for details.

The Kokoro-82M model is under the Apache 2.0 license.

---

## FAQ

### What's the difference between audioreader.py and audioreader_hq.py?

- `audioreader.py`: Standard mode, single voice, basic processing
- `audioreader_hq.py`: High quality pipeline with automatic multi-voice, emotional analysis, and broadcast post-processing

### How do I configure voices per character?

Create a JSON file with the mapping:

```json
{
  "voice_mapping": {
    "Marie": "af_bella",
    "Peter": "am_adam"
  }
}
```

Then: `python audioreader_hq.py book.md --config config.json`

### Can I publish on Audible?

No, Audible/ACX doesn't accept AI-generated voices. Use Google Play Books, Findaway, or Kobo.

### How does it compare to ElevenLabs quality?

With the HQ pipeline (multi-voice + emotions + post-processing), quality is very close to ElevenLabs, while being 100% free and local.

### How do I export to multiple formats?

Use the book exporter:

```python
from src.book_exporter import export_markdown_book

export_markdown_book(
    chapters_dir="chapters/",
    output_dir="output/",
    title="My Book",
    author="Author",
    formats=["pdf", "epub", "html", "txt"]
)
```

### Is caching automatic?

Yes, with the extended pipeline (`hq_pipeline_extended.py`), caching is enabled by default. Identical segments are only generated once.

### How can I speed up generation?

1. **Enable cache**: Already-generated segments are reused
2. **Parallelization**: Configure `num_workers=4` (or more based on your CPUs)
3. **GPU**: Use XTTS-v2 with GPU for voice cloning

---

*AudioReader v2.3 - Convert your books to professional-quality audiobooks*
