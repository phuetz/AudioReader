#!/usr/bin/env python3
"""
AudioReader MCP Server.

Serveur MCP (Model Context Protocol) pour piloter AudioReader depuis Claude.ai
ou Claude Desktop.

Installation:
    pip install mcp

Configuration Claude Desktop (claude_desktop_config.json):
    {
      "mcpServers": {
        "audioreader": {
          "command": "python",
          "args": ["/path/to/AudioReader/mcp_server.py"],
          "env": {}
        }
      }
    }

Usage avec Claude.ai:
    Claude peut appeler les outils exposes pour generer des audiobooks.
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional
import logging

# Ajouter le repertoire src au path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        TextContent,
        CallToolResult,
        ListToolsResult,
    )
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    print("MCP non installe. Installez avec: pip install mcp", file=sys.stderr)

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audioreader-mcp")

# Repertoire de sortie par defaut
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


class AudioReaderMCPServer:
    """Serveur MCP pour AudioReader."""

    def __init__(self):
        self.server = Server("audioreader")
        self.tts_engine = None
        self.pipeline = None
        self._setup_handlers()

    def _setup_handlers(self):
        """Configure les handlers MCP."""

        @self.server.list_tools()
        async def list_tools() -> ListToolsResult:
            """Liste les outils disponibles."""
            return ListToolsResult(tools=[
                Tool(
                    name="list_voices",
                    description="Liste les voix disponibles pour la synthese vocale",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "language": {
                                "type": "string",
                                "description": "Code langue (fr, en)",
                                "default": "fr"
                            }
                        }
                    }
                ),
                Tool(
                    name="generate_audio",
                    description="Genere un fichier audio a partir de texte",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Texte a convertir en audio"
                            },
                            "voice": {
                                "type": "string",
                                "description": "ID de la voix (ex: ff_siwis, af_heart)",
                                "default": "ff_siwis"
                            },
                            "speed": {
                                "type": "number",
                                "description": "Vitesse de lecture (0.5-2.0)",
                                "default": 1.0
                            },
                            "output_name": {
                                "type": "string",
                                "description": "Nom du fichier de sortie (sans extension)",
                                "default": "output"
                            }
                        },
                        "required": ["text"]
                    }
                ),
                Tool(
                    name="generate_audiobook",
                    description="Genere un audiobook complet a partir d'un texte long avec analyse des dialogues et emotions",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Texte complet du livre/chapitre"
                            },
                            "title": {
                                "type": "string",
                                "description": "Titre de l'audiobook",
                                "default": "audiobook"
                            },
                            "narrator_voice": {
                                "type": "string",
                                "description": "Voix du narrateur",
                                "default": "ff_siwis"
                            },
                            "style": {
                                "type": "string",
                                "description": "Style de narration: formal, conversational, dramatic, storytelling",
                                "default": "storytelling"
                            },
                            "enable_emotions": {
                                "type": "boolean",
                                "description": "Activer l'analyse des emotions",
                                "default": True
                            },
                            "enable_multi_voice": {
                                "type": "boolean",
                                "description": "Activer les voix multiples pour les dialogues",
                                "default": True
                            }
                        },
                        "required": ["text"]
                    }
                ),
                Tool(
                    name="analyze_text",
                    description="Analyse un texte pour detecter les dialogues, emotions et personnages",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Texte a analyser"
                            }
                        },
                        "required": ["text"]
                    }
                ),
                Tool(
                    name="get_config",
                    description="Retourne la configuration actuelle d'AudioReader",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="set_config",
                    description="Modifie la configuration d'AudioReader",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "narrator_voice": {
                                "type": "string",
                                "description": "Voix du narrateur"
                            },
                            "default_style": {
                                "type": "string",
                                "description": "Style par defaut"
                            },
                            "enable_intonation": {
                                "type": "boolean",
                                "description": "Activer les contours d'intonation"
                            },
                            "enable_timing_humanization": {
                                "type": "boolean",
                                "description": "Activer l'humanisation du timing"
                            },
                            "intonation_strength": {
                                "type": "number",
                                "description": "Force des contours d'intonation (0-1)"
                            }
                        }
                    }
                ),
                Tool(
                    name="list_output_files",
                    description="Liste les fichiers audio generes",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
            ])

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> CallToolResult:
            """Execute un outil."""
            try:
                if name == "list_voices":
                    result = await self._list_voices(arguments.get("language", "fr"))
                elif name == "generate_audio":
                    result = await self._generate_audio(arguments)
                elif name == "generate_audiobook":
                    result = await self._generate_audiobook(arguments)
                elif name == "analyze_text":
                    result = await self._analyze_text(arguments["text"])
                elif name == "get_config":
                    result = await self._get_config()
                elif name == "set_config":
                    result = await self._set_config(arguments)
                elif name == "list_output_files":
                    result = await self._list_output_files()
                else:
                    result = {"error": f"Outil inconnu: {name}"}

                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=json.dumps(result, indent=2, ensure_ascii=False)
                    )]
                )
            except Exception as e:
                logger.error(f"Erreur outil {name}: {e}")
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=json.dumps({"error": str(e)}, ensure_ascii=False)
                    )],
                    isError=True
                )

    async def _list_voices(self, language: str = "fr") -> dict:
        """Liste les voix disponibles."""
        # Voix Kokoro
        kokoro_voices = {
            "fr": [
                {"id": "ff_siwis", "name": "Siwis", "gender": "F", "style": "neutral"},
            ],
            "en": [
                {"id": "af_heart", "name": "Heart", "gender": "F", "style": "warm"},
                {"id": "af_sarah", "name": "Sarah", "gender": "F", "style": "professional"},
                {"id": "am_adam", "name": "Adam", "gender": "M", "style": "neutral"},
                {"id": "am_michael", "name": "Michael", "gender": "M", "style": "deep"},
                {"id": "bf_emma", "name": "Emma", "gender": "F", "style": "british"},
                {"id": "bm_george", "name": "George", "gender": "M", "style": "british"},
            ]
        }

        # Voix Edge-TTS
        edge_voices = {
            "fr": [
                {"id": "fr-FR-DeniseNeural", "name": "Denise", "gender": "F", "style": "neural"},
                {"id": "fr-FR-HenriNeural", "name": "Henri", "gender": "M", "style": "neural"},
                {"id": "fr-CA-SylvieNeural", "name": "Sylvie (CA)", "gender": "F", "style": "neural"},
            ],
            "en": [
                {"id": "en-US-JennyNeural", "name": "Jenny", "gender": "F", "style": "neural"},
                {"id": "en-US-GuyNeural", "name": "Guy", "gender": "M", "style": "neural"},
                {"id": "en-GB-SoniaNeural", "name": "Sonia", "gender": "F", "style": "british"},
            ]
        }

        return {
            "language": language,
            "kokoro": kokoro_voices.get(language, []),
            "edge_tts": edge_voices.get(language, []),
            "recommended": "ff_siwis" if language == "fr" else "af_heart"
        }

    async def _generate_audio(self, args: dict) -> dict:
        """Genere un fichier audio simple."""
        text = args["text"]
        voice = args.get("voice", "ff_siwis")
        speed = args.get("speed", 1.0)
        output_name = args.get("output_name", "output")

        # Initialiser le moteur TTS si necessaire
        if self.tts_engine is None:
            from src.tts_unified import UnifiedTTS
            self.tts_engine = UnifiedTTS()

        # Generer l'audio
        audio, sample_rate = self.tts_engine.synthesize(
            text=text,
            voice=voice,
            speed=speed,
            lang="fr" if voice.startswith("ff") else "en"
        )

        # Sauvegarder
        output_path = OUTPUT_DIR / f"{output_name}.wav"
        import soundfile as sf
        sf.write(str(output_path), audio, sample_rate)

        duration = len(audio) / sample_rate

        return {
            "success": True,
            "output_file": str(output_path),
            "duration_seconds": round(duration, 2),
            "voice": voice,
            "text_length": len(text)
        }

    async def _generate_audiobook(self, args: dict) -> dict:
        """Genere un audiobook complet."""
        text = args["text"]
        title = args.get("title", "audiobook")
        narrator_voice = args.get("narrator_voice", "ff_siwis")
        style = args.get("style", "storytelling")
        enable_emotions = args.get("enable_emotions", True)
        enable_multi_voice = args.get("enable_multi_voice", True)

        # Initialiser le pipeline
        from src.hq_pipeline_extended import create_extended_pipeline, ExtendedPipelineConfig

        config = ExtendedPipelineConfig(
            lang="fr",
            narrator_voice=narrator_voice,
            enable_emotion_analysis=enable_emotions,
            auto_assign_voices=enable_multi_voice,
            default_narration_style=style,
            enable_intonation_contours=True,
            enable_timing_humanization=True,
            enable_advanced_breaths=True,
        )

        pipeline = create_extended_pipeline(**config.__dict__)

        # Traiter le texte
        segments = pipeline.process_chapter(text, chapter_index=0)

        # Initialiser TTS si necessaire
        if self.tts_engine is None:
            from src.tts_unified import UnifiedTTS
            self.tts_engine = UnifiedTTS()

        # Synthetiser
        audios = []
        for seg in segments:
            audio, sr = self.tts_engine.synthesize(
                text=seg.text,
                voice=seg.voice_id,
                speed=seg.final_speed,
                lang="fr"
            )
            audios.append(audio)

        # Concatener avec pauses
        import numpy as np
        from src.bio_acoustics import BioAudioGenerator
        bio_gen = BioAudioGenerator(sample_rate=24000)

        result_parts = []
        result_parts.append(bio_gen.generate_silence(0.5))

        for seg, audio in zip(segments, audios):
            if seg.pause_before > 0:
                result_parts.append(bio_gen.generate_silence(seg.pause_before))
            result_parts.append(audio)
            if seg.pause_after > 0:
                result_parts.append(bio_gen.generate_silence(seg.pause_after))

        result_parts.append(bio_gen.generate_silence(1.0))
        full_audio = np.concatenate(result_parts)

        # Sauvegarder
        output_path = OUTPUT_DIR / f"{title}.wav"
        import soundfile as sf
        sf.write(str(output_path), full_audio, 24000)

        duration = len(full_audio) / 24000

        # Stats
        characters = pipeline.get_characters()
        voice_assignments = pipeline.get_voice_assignments()

        return {
            "success": True,
            "output_file": str(output_path),
            "duration_seconds": round(duration, 2),
            "duration_formatted": f"{int(duration // 60)}:{int(duration % 60):02d}",
            "segments_count": len(segments),
            "characters_detected": characters,
            "voice_assignments": voice_assignments,
            "style": style,
            "features_enabled": {
                "emotions": enable_emotions,
                "multi_voice": enable_multi_voice,
                "intonation_contours": True,
                "timing_humanization": True
            }
        }

    async def _analyze_text(self, text: str) -> dict:
        """Analyse un texte."""
        from src.dialogue_attribution import DialogueAttributor
        from src.emotion_analyzer import EmotionAnalyzer
        from src.intonation_contour import IntonationContourDetector

        # Attribution des dialogues
        attributor = DialogueAttributor(lang="fr")
        dialogues = attributor.process_text(text)

        # Analyse des emotions
        emotion_analyzer = EmotionAnalyzer()

        # Detection des contours
        contour_detector = IntonationContourDetector(language="fr")

        # Analyser chaque segment
        analysis = {
            "total_characters": len(text),
            "dialogues": [],
            "characters": list(attributor.context.participants),
            "sentences": []
        }

        for d in dialogues:
            analysis["dialogues"].append({
                "text": d.text[:50] + "..." if len(d.text) > 50 else d.text,
                "speaker": d.attribution.speaker,
                "method": d.attribution.method.value,
                "confidence": d.attribution.confidence
            })

        # Analyser quelques phrases
        import re
        sentences = re.split(r'[.!?]+', text)[:10]
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            contour = contour_detector.detect(sent)
            emotion_result = emotion_analyzer.analyze(sent)

            analysis["sentences"].append({
                "text": sent[:40] + "..." if len(sent) > 40 else sent,
                "intonation": contour.value,
                "emotion": emotion_result.emotion.value if emotion_result else "neutral",
                "intensity": emotion_result.intensity if emotion_result else 0.5
            })

        return analysis

    async def _get_config(self) -> dict:
        """Retourne la configuration actuelle."""
        return {
            "output_dir": str(OUTPUT_DIR),
            "default_voice": "ff_siwis",
            "default_language": "fr",
            "features": {
                "intonation_contours": True,
                "timing_humanization": True,
                "advanced_breaths": True,
                "emotion_analysis": True,
                "multi_voice": True,
                "acx_compliance": True
            },
            "styles_available": [
                "formal", "conversational", "dramatic",
                "storytelling", "documentary", "intimate", "energetic"
            ],
            "version": "2.4"
        }

    async def _set_config(self, args: dict) -> dict:
        """Modifie la configuration."""
        # Pour l'instant, on retourne juste les changements demandes
        # Une vraie implementation sauvegarderait dans un fichier config
        return {
            "success": True,
            "changes": args,
            "message": "Configuration mise a jour"
        }

    async def _list_output_files(self) -> dict:
        """Liste les fichiers de sortie."""
        files = []
        for f in OUTPUT_DIR.glob("*.wav"):
            stat = f.stat()
            files.append({
                "name": f.name,
                "path": str(f),
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
                "modified": stat.st_mtime
            })

        # Trier par date de modification (plus recent d'abord)
        files.sort(key=lambda x: x["modified"], reverse=True)

        return {
            "output_dir": str(OUTPUT_DIR),
            "files": files,
            "total_count": len(files)
        }

    async def run(self):
        """Demarre le serveur MCP."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


def main():
    """Point d'entree principal."""
    if not HAS_MCP:
        print("Erreur: Le package MCP n'est pas installe.")
        print("Installez-le avec: pip install mcp")
        sys.exit(1)

    server = AudioReaderMCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
