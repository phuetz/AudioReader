# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AudioReader converts Markdown books into high-quality audiobooks using Kokoro-82M TTS. It features automatic multi-voice character detection, emotion analysis, and broadcast-quality audio post-processing. French-focused project — code comments, docstrings, and UI messages are primarily in French.

## Commands

### Python Backend

```bash
source venv/bin/activate

# Standard conversion
python audio_reader.py livre.md

# HQ pipeline (multi-voice, emotions, post-processing)
python audio_reader.py livre.md --hq --master

# With LLM enhancement
python audio_reader.py livre.md --hq --llm-enhance --llm-provider gemini

# Gradio web interface (legacy)
python audio_reader.py --gui

# API server (FastAPI v2)
uvicorn api:create_app --reload --port 8000
# or: python -m api
```

### Frontend (React)

```bash
cd frontend
npm install
npm run dev          # Vite dev server
npm run build        # TypeScript check + Vite build
npm run lint         # ESLint
npm run test         # Vitest (unit tests)
```

### Testing

```bash
source venv/bin/activate

# All Python tests
python run_tests.py
python run_tests.py -v              # verbose
python run_tests.py --coverage      # with coverage

# Single module (available: audio_tags, voice_morphing, cache, emotion_control, conversation, pipeline)
python run_tests.py --module audio_tags

# Single test file
pytest tests/test_audio_tags.py -v

# Frontend tests
cd frontend && npm run test
```

### After Every Code Change

Run `python run_tests.py -v` to verify nothing is broken, then test manually with a short file if needed: `python audio_reader.py test.md --hq`

## Architecture

Three-tier system: **CLI** → **FastAPI API** → **React Frontend**

### Entry Points

| Entry | Purpose |
|-------|---------|
| `audio_reader.py` | CLI tool (standard mode or `--hq` for full pipeline) |
| `api/` | FastAPI v2 app factory (`api/__init__.py` → `create_app()`) |
| `api_server.py` | Legacy standalone API server |
| `app.py` | Gradio web interface (legacy, 5 tabs) |
| `mcp_server.py` | MCP server for Claude Desktop integration |
| `frontend/` | React SPA (Vite + TypeScript + Tailwind + Zustand) |

### Backend: `src/` Modules

**Text Pipeline:** `markdown_parser.py` → `text_normalizer.py` → `text_processor.py` → `french_preprocessor.py`

**Character & Emotion:** `character_detector.py` → `emotion_analyzer.py` → `narrative_context.py` → `emotion_continuity.py` → `dialogue_attribution.py`

**TTS Engines** (all implement similar synthesize interface):
- `tts_kokoro_engine.py` — Primary engine (Kokoro-82M), supports voice blending
- `tts_unified.py` — Wrapper that auto-selects best engine per language
- `tts_hybrid_engine.py` — Combines engines with crossfade
- Others: `tts_qwen3_engine.py` (Qwen3, 10 langs, cloning+instruct), `tts_voxtral_engine.py` (Voxtral/Mistral, cloud+local, 9 langs, cloning), `tts_xtts_engine.py` (voice cloning), `tts_chatterbox_engine.py`, `tts_orpheus_engine.py`, `tts_parler_engine.py`, `tts_engine.py` (Edge-TTS fallback)

**Audio Processing:** `audio_enhancer.py` (EQ/compression/loudness), `bio_acoustics.py` (breaths/room tone), `audio_crossfade.py`, `acx_compliance.py` (Audible standards)

**LLM Integration:** `llm_enhancer.py` — Unified pipeline (Ollama/OpenAI/Anthropic/Gemini) for character validation, emotion detection, auto audio tags, dialogue attribution

**Pipelines:** `hq_pipeline.py` → `hq_pipeline_extended.py` (full v2.1+ features)

### HQ Pipeline Flow

```
Text → Normalizer → Character Detection → Emotion Analysis →
Narrative Context → Emotion Continuity → Voice Assignment →
Dynamic Voice Blending → TTS Synthesis → Bio-Acoustic Pauses →
Audio Enhancement → Output
```

### API v2 (`api/`)

FastAPI app with `create_app()` factory pattern. SQLite persistence via `aiosqlite` (DB at `.audioreader_data/jobs.db`). Routers in `api/routers/`: generation, jobs, voices, files, analysis, podcast, projects, config, streaming, acx, corrections, review, queue, character_profiles, export_platforms, subtitles, summaries, consistency, openai_compat.

### Frontend (`frontend/`)

React 19 SPA with:
- **Routing:** React Router v7, lazy-loaded pages in `src/pages/`
- **State:** Zustand stores in `src/stores/` (useJobStore, useVoiceStore, useProjectStore, useCharacterStore, useAudioStore, usePodcastStore, usePlaylistStore, useSettingsStore)
- **API layer:** Axios client in `src/api/client.ts` (base URL from `VITE_API_URL` env var), endpoints in `src/api/endpoints.ts`, SSE in `src/api/sse.ts`
- **UI:** Tailwind CSS v4, Lucide React icons, reusable components in `src/components/ui/`
- **Pages:** Dashboard, QuickText, BookConversion, Characters, VoiceCloning, VoiceLab, ACXAnalysis, Corrections, Projects, Podcast, Files, Settings, Review, Queue

### Key Data Classes

- `EnrichedSegment` (`advanced_preprocessor.py`) — Text segment with speaker, emotion, prosody
- `HQPipelineConfig` (`hq_pipeline.py`) — Full pipeline configuration
- `DialogueSegment` (`character_detector.py`) — Detected dialogue with speaker info
- `EmotionAnalysis` (`emotion_analyzer.py`) — Emotion type, intensity, prosody hints

## Voice System

- French default voice: `ff_siwis`
- Voice blend syntax: `"af_bella:60,am_adam:40"` (60% Bella, 40% Adam)
- Voices auto-assigned to detected characters based on gender inference from names
- `DynamicVoiceManager` adjusts voice mix based on emotion (anger→deeper, sadness→softer, fear→trembling, joy→brighter)
- Voice presets via `voice_designer.py` (tensor interpolation, random walk, blending)

## Model Files

Required in project root:
- `kokoro-v1.0.onnx` (~310 MB)
- `voices-v1.0.bin` (~27 MB)

## Environment Variables

```bash
GEMINI_API_KEY=...      # For Gemini LLM enhancer
OPENAI_API_KEY=...      # For OpenAI LLM enhancer
ANTHROPIC_API_KEY=...   # For Anthropic LLM enhancer
MISTRAL_API_KEY=...     # For Voxtral TTS (cloud mode)
VOXTRAL_BASE_URL=...    # For Voxtral TTS (local vLLM server URL)
VITE_API_URL=...        # Frontend API base URL (defaults to same origin)
```

## Language

French-focused project. The text normalization pipeline handles French numbers, dates, currencies, Roman numerals extensively. Code comments and UI strings are primarily in French.
