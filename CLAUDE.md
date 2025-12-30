# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AudioReader converts Markdown books into high-quality audiobooks using Kokoro-82M TTS. The project features automatic multi-voice character detection, emotion analysis, and broadcast-quality audio post-processing to achieve results comparable to ElevenLabs.

## Commands

### Running the Application

```bash
# Activate virtual environment first
source venv/bin/activate

# Standard conversion
python audioreader.py livre.md

# High-quality pipeline (multi-voice, emotions, post-processing)
python audioreader_hq.py livre.md -o output/

# Web interface (Gradio)
python audioreader.py --gui

# List available voices
python audioreader.py --list-voices
```

### Testing

```bash
# Run all tests
python run_tests.py

# Verbose mode
python run_tests.py -v

# Specific module tests
python run_tests.py --module audio_tags
python run_tests.py --module voice_morphing
python run_tests.py --module cache
python run_tests.py --module emotion_control
python run_tests.py --module conversation
python run_tests.py --module pipeline

# Run single test file with pytest
pytest tests/test_audio_tags.py -v

# With coverage
python run_tests.py --coverage

# Test bio-acoustics features
python test_bio_improvements.py
```

### Audio Post-processing

```bash
# Apply broadcast mastering to audio files
python postprocess.py output_folder/
```

### Batch Generation

```bash
# Generate all chapters of a book (example script)
python generate_tome1.py
```

## Architecture

### Two Main Pipelines

1. **Standard Pipeline** (`audioreader.py`): Single voice, basic processing
2. **HQ Pipeline** (`audioreader_hq.py`): Multi-voice, emotions, broadcast post-processing

### Core Modules in `src/`

**Text Processing:**
- `markdown_parser.py` - Parses MD, multi-file directories, EPUB
- `text_normalizer.py` - Converts numbers, dates, currencies to words (French)
- `text_processor.py` - Chunking, pronunciation corrections
- `french_preprocessor.py` - French-specific text normalization

**Character & Emotion:**
- `character_detector.py` - Detects dialogue and speakers from text patterns
- `emotion_analyzer.py` - Sentiment analysis with prosody hints
- `narrative_context.py` - Detects context type (action, description, suspense)
- `emotion_continuity.py` - Smooths emotional transitions

**TTS Engines:**
- `tts_kokoro_engine.py` - Primary TTS (Kokoro-82M), supports voice blending
- `tts_engine.py` - Edge-TTS fallback
- `tts_unified.py` - Wrapper that auto-selects best engine per language
- `tts_hybrid_engine.py` - Hybrid approach combining engines (with crossfade)
- `tts_xtts_engine.py` - XTTS-v2 voice cloning engine (requires TTS library)

**Advanced Features (v2.1):**
- `audio_tags.py` - ElevenLabs-style tags (`[whispers]`, `[laugh]`, etc.)
- `voice_morphing.py` - Pitch, formant, time stretch modifications
- `voice_cloning.py` - XTTS-v2 voice cloning
- `synthesis_cache.py` - Intelligent caching + parallel synthesis
- `conversation_generator.py` - Multi-speaker dialogue generation

**Bio-Acoustics (v2.2 - "Style ElevenLabs"):**
- `bio_acoustics.py` - Synthetic biological sounds (breaths, mouth noises, room tone)
- `dynamic_voice.py` - Emotion-based automatic voice blending

**Prosody & Timing (v2.3 - "Style ElevenLabs" amélioré):**
- `breath_samples.py` - Gestionnaire hybride samples/synthèse pour respirations
- `intonation_contour.py` - Contours d'intonation phrase-level (déclaratif, question, exclamation)
- `timing_humanizer.py` - Micro-variations de timing, pauses d'emphase

**Pipelines:**
- `hq_pipeline.py` - Unified HQ processing (v2.0)
- `hq_pipeline_extended.py` - Extended pipeline with all v2.1+ features, integrates bio-acoustics

**Audio:**
- `audio_enhancer.py` - EQ, compression, de-essing, loudness normalization
- `audio_postprocess.py` - Post-processing utilities
- `audiobook_builder.py` - M4B/MP3 export with ID3 metadata
- `audio_crossfade.py` - Crossfade transitions between audio segments (cosine curves)

**Corrections:**
- `corrections_loader.py` - Loads JSON pronunciation glossaries
- `corrections_conquerants.py` - Book-specific corrections
- `corrections_ui.py` - Gradio web UI for managing corrections

**Utilities (v2.4):**
- `preview_generator.py` - Quick 30s preview generation
- `book_exporter.py` - Export to PDF, EPUB, HTML, TXT

### HQ Pipeline Flow

```
Text → Normalizer → Character Detection → Emotion Analysis →
Narrative Context → Emotion Continuity → Voice Assignment →
Dynamic Voice Blending → TTS Synthesis → Bio-Acoustic Pauses →
Audio Enhancement → Output
```

Key v2.2 additions:
- **Dynamic Voice Blending**: Auto-mixes voices based on emotion (e.g., anger adds `am_adam` for deeper tone)
- **Bio-Acoustic Pauses**: Room tone instead of digital silence, synthetic breaths at transitions

### Key Data Classes

- `EnrichedSegment` (advanced_preprocessor.py) - Text segment with speaker, emotion, prosody
- `HQPipelineConfig` (hq_pipeline.py) - Full pipeline configuration
- `DialogueSegment` (character_detector.py) - Detected dialogue with speaker info
- `EmotionAnalysis` (emotion_analyzer.py) - Emotion type, intensity, prosody hints

## Voice System

French default: `ff_siwis`
Voice blend syntax: `"af_bella:60,am_adam:40"` (60% Bella, 40% Adam)

Voices are automatically assigned to detected characters based on gender inference from names.

### Dynamic Voice Blending (v2.2)

`DynamicVoiceManager` automatically adjusts voice mix based on detected emotion:

| Emotion | Blend Target | Effect |
|---------|--------------|--------|
| Anger | `am_adam` | Deeper, authoritative |
| Sadness | `af_bella` | Softer, gentler |
| Fear | `af_sky` | Unstable, trembling |
| Joy | `af_nicole` | Brighter, energetic |
| Suspense | `am_michael` | Calm, measured |

Blend intensity scales with emotion intensity (LOW→EXTREME = 20%→100% of max blend weight).

### Bio-Acoustics (v2.2)

`BioAudioGenerator` replaces digital silence with natural sounds:
- **Room tone**: Very low white noise instead of 0.0 silence
- **Breaths**: `soft`, `sharp`, `deep`, `gasp`, `sigh` types with ADSR envelopes
- **Tag support**: `generate_for_tag()` handles `[gasp]`, `[sigh]`, `[pause]`, etc.
- **True crossfade**: `apply_crossfade()` overlaps signals with cosine curves
- **Variable thresholds**: ±30% randomization for natural rhythm

## Model Files

Required in project root:
- `kokoro-v1.0.onnx` (~310 MB)
- `voices-v1.0.bin` (~27 MB)

## Prosody & Timing (v2.3)

### Advanced Breath Generation

`BioAudioGenerator` v2.3 utilise des techniques avancées:
- **Bruit rose** (1/f spectrum) au lieu de blanc pour un son plus naturel
- **Filtrage formant** (F1~500Hz, F2~1500Hz, F3~2500Hz) simulant le tract vocal
- **Jitter d'amplitude** (±5%) pour des variations micro-temporelles naturelles

Configuration:
```python
generator = BioAudioGenerator(sample_rate=24000, use_advanced_breaths=True)
breath = generator.generate_breath(type="gasp", intensity=0.8)
```

### Hybrid Breath System

`HybridBreathGenerator` permet d'utiliser des samples audio réels:
```python
from src.breath_samples import HybridBreathGenerator
hybrid = HybridBreathGenerator(samples_dir=Path("samples/breaths"))
audio = hybrid.generate_breath("sigh")  # Sample si disponible, sinon synthèse
```

Structure des samples (optionnel):
```
samples/breaths/
├── soft/
├── gasp/
├── sigh/
└── deep/
```

### Intonation Contours

`IntonationProcessor` détecte et applique les contours prosodiques:

| Type | Pattern | Exemple |
|------|---------|---------|
| DECLARATIVE | Descente finale (-2.5 st) | "Il est parti." |
| QUESTION_YN | Montée finale (+4 st) | "Tu viens ?" |
| QUESTION_WH | Pic initial puis descente | "Où vas-tu ?" |
| EXCLAMATION | Pic fort puis descente rapide | "Incroyable !" |
| CONTINUATION | Légère montée (+1 st) | "D'abord, ..." |
| SUSPENSE | Descente lente | "Et puis..." |

Usage:
```python
from src.intonation_contour import IntonationProcessor
processor = IntonationProcessor(sample_rate=24000, strength=0.7)
audio = processor.process(audio, "Tu viens ?")  # Applique montée finale
```

### Timing Humanization

`TimingHumanizer` évite la régularité mécanique:
- **Variation gaussienne** des pauses (±15%)
- **Rythme syntaxique**: clauses subordonnées plus rapides, incises encore plus
- **Micro-pauses d'emphase** avant "jamais", "soudain", "mais", etc.
- **Variation inter-phrase** (±3%) pour éviter la monotonie

```python
from src.timing_humanizer import TimingHumanizer, TextTimingProcessor
humanizer = TimingHumanizer()
pause = humanizer.humanize_pause(0.5)  # Retourne ~0.425-0.575s

processor = TextTimingProcessor(humanizer)
segments = processor.process_text("C'est vraiment incroyable!")
# segments[0].text contient "[pause:0.05] vraiment" etc.
```

## Audio Crossfade (v2.4)

`AudioCrossfader` provides smooth transitions between audio segments:

```python
from src.audio_crossfade import apply_crossfade_to_chapter, AudioCrossfader

# Simple usage for chapter assembly
final_audio = apply_crossfade_to_chapter(audio_segments, sample_rate=24000, crossfade_ms=50)

# Advanced usage
config = CrossfadeConfig(
    crossfade_duration=0.05,  # 50ms
    curve_type='cosine',       # 'linear', 'cosine', 'exponential'
    apply_edge_fades=True
)
crossfader = AudioCrossfader(config)
merged = crossfader.crossfade_segments(segment1, segment2, sample_rate)
```

The hybrid engine now uses crossfade by default:
```python
engine = HybridTTSEngine(use_crossfade=True, crossfade_ms=50)
```

## Quick Preview (v2.4)

Generate 30-second previews to test settings before full conversion:

```python
from src.preview_generator import generate_quick_preview, PreviewGenerator

# Simple usage
success, msg = generate_quick_preview(
    text="Long text...",
    output_path="preview.wav",
    engine_type="hybrid",
    duration=30.0
)

# Advanced: extract representative text
generator = PreviewGenerator()
preview_text = generator.extract_preview_text(full_text, lang='fr')
# Returns ~450 chars including: start, dialogue, emotional passage
```

## Corrections Web UI (v2.4)

Gradio interface for managing pronunciation corrections:

```bash
# Launch the UI
python -m src.corrections_ui --file corrections.json --port 7861

# Or programmatically
from src.corrections_ui import launch_corrections_ui
launch_corrections_ui("corrections.json", share=False, port=7861)
```

Features:
- Add/delete/search corrections
- Preview corrections on text
- Test audio with TTS engine
- Import/export JSON

## XTTS-v2 Engine (v2.4)

High-quality voice cloning with XTTS-v2:

```python
from src.tts_xtts_engine import XTTSEngine, XTTSConfig

config = XTTSConfig(
    default_language="fr",
    use_gpu=True,
    temperature=0.7,
    speed=1.0
)
engine = XTTSEngine(config)

# Register a cloned voice (min 6 seconds of audio)
engine.register_voice("narrator", "samples/narrator_voice.wav")

# Synthesize with the cloned voice
engine.synthesize_chapter(text, "output.wav", voice_id="narrator")
```

Requirements:
```bash
pip install TTS torch torchaudio
```

See `docs/FINE_TUNING_OPTIONS.md` for fine-tuning guide.

## Language

This is a French-focused project. Code comments, docstrings, and output messages are primarily in French. The system handles French text normalization (numbers, dates, Roman numerals) extensively.
