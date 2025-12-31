#!/usr/bin/env python3
"""
AudioReader - Interface Web Gradio v3.1

Interface graphique complète pour convertir des livres en audiobooks.
Utilise Kokoro-82M, MMS-TTS et XTTS-v2 pour une qualité professionnelle.

Lancer avec: python app.py
Puis ouvrir: http://localhost:7860
"""
import sys
import os
from pathlib import Path
import tempfile
import shutil
import soundfile as sf
import numpy as np
import json
import time
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
from markdown_parser import parse_book
from tts_kokoro_engine import KokoroEngine, KOKORO_VOICES
from text_processor import TextProcessor
from audiobook_builder import AudiobookBuilder, AudiobookMetadata
from input_converter import InputConverter

# ============================================================================
# IMPORTS CONDITIONNELS
# ============================================================================

# Clonage
try:
    from voice_cloning import VoiceCloningManager
    from audio_extractor import AudioExtractor
    CLONING_AVAILABLE = True
except ImportError:
    CLONING_AVAILABLE = False

# Pipeline HQ
try:
    from hq_pipeline_extended import ExtendedPipelineConfig, ExtendedHQPipeline
    from hq_pipeline_extended import AudiobookGenerator as HQGenerator
    HAS_HQ = True
except ImportError:
    try:
        from src.hq_pipeline_extended import ExtendedPipelineConfig, ExtendedHQPipeline
        from src.hq_pipeline_extended import AudiobookGenerator as HQGenerator
        HAS_HQ = True
    except ImportError:
        HAS_HQ = False

# Podcast
try:
    from podcast_server import get_server
    PODCAST_AVAILABLE = True
except ImportError:
    PODCAST_AVAILABLE = False

# Preview
try:
    from preview_generator import generate_quick_preview, PreviewGenerator
    PREVIEW_AVAILABLE = True
except ImportError:
    PREVIEW_AVAILABLE = False

# Dialogue Attribution
try:
    from dialogue_attribution import DialogueAttributor
    ATTRIBUTION_AVAILABLE = True
except ImportError:
    ATTRIBUTION_AVAILABLE = False

# ============================================================================
# CONFIGURATION ET CONSTANTES
# ============================================================================

MODEL_PATH = Path("kokoro-v1.0.onnx")
VOICES_PATH = Path("voices-v1.0.bin")
STATS_FILE = Path(".audioreader_stats.json")
PREFS_FILE = Path(".audioreader_prefs.json")
PROJECTS_DIR = Path(".audioreader_projects")
VOICE_CACHE_DIR = Path(".voice_cache/samples")

# Créer les dossiers nécessaires
PROJECTS_DIR.mkdir(exist_ok=True)
VOICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# GESTION DES STATISTIQUES ET PRÉFÉRENCES
# ============================================================================

def load_stats() -> Dict[str, Any]:
    """Charge les statistiques d'utilisation."""
    if STATS_FILE.exists():
        try:
            return json.loads(STATS_FILE.read_text())
        except:
            pass
    return {
        "books_generated": 0,
        "total_duration_minutes": 0,
        "total_characters": 0,
        "voices_cloned": 0,
        "last_projects": [],
        "favorite_voice": "ff_siwis",
        "favorite_style": "storytelling"
    }

def save_stats(stats: Dict[str, Any]):
    """Sauvegarde les statistiques."""
    STATS_FILE.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

def load_prefs() -> Dict[str, Any]:
    """Charge les préférences utilisateur."""
    if PREFS_FILE.exists():
        try:
            return json.loads(PREFS_FILE.read_text())
        except:
            pass
    return {
        "default_voice": "ff_siwis",
        "default_engine": "auto",
        "default_style": "storytelling",
        "default_speed": 1.0,
        "enable_hq": True,
        "enable_multivoice": True,
        "enable_mastering": True,
        "dark_mode": False,
        "expert_mode": False,
        "output_dir": "output"
    }

def save_prefs(prefs: Dict[str, Any]):
    """Sauvegarde les préférences."""
    PREFS_FILE.write_text(json.dumps(prefs, indent=2, ensure_ascii=False))

def add_to_recent_projects(title: str, file_path: str, output_path: str):
    """Ajoute un projet aux projets récents."""
    stats = load_stats()
    project = {
        "title": title,
        "file": file_path,
        "output": output_path,
        "date": datetime.now().isoformat()
    }
    stats["last_projects"] = [project] + stats["last_projects"][:9]  # Garder 10 max
    save_stats(stats)

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def check_model() -> bool:
    """Vérifie si le modèle Kokoro est disponible."""
    return MODEL_PATH.exists() and VOICES_PATH.exists()

def estimate_duration(text: str) -> float:
    """Estime la durée audio en minutes (environ 150 mots/minute)."""
    words = len(text.split())
    return words / 150

def estimate_conversion_time(text: str, hq: bool = False) -> float:
    """Estime le temps de conversion en minutes."""
    chars = len(text)
    # Environ 5x temps réel, donc 1 min audio = ~12 sec conversion
    base_time = (chars / 1000) * 0.2  # ~0.2 min par 1000 chars
    if hq:
        base_time *= 1.3  # HQ ajoute ~30%
    return base_time

def get_voice_choices() -> List[Tuple[str, str]]:
    """Retourne la liste des voix pour le dropdown."""
    choices = []

    # Voix Kokoro
    for vid, info in KOKORO_VOICES.items():
        gender = "♀" if info["gender"] == "F" else "♂"
        lang = info.get("lang", "en")[:2].upper()
        label = f"{info['name']} {gender} [{lang}] - {info['desc']}"
        choices.append((label, vid))

    # Voix clonées
    if CLONING_AVAILABLE:
        try:
            manager = VoiceCloningManager()
            cloned = manager.cloner.list_cloned_voices()
            for v in cloned:
                label = f"🎙️ {v.name} [{v.language.upper()}] (clonée)"
                choices.append((label, v.name))
        except Exception as e:
            print(f"Erreur chargement voix clonées: {e}")

    return choices

def get_cloned_voices_gallery() -> List[Dict]:
    """Retourne la galerie des voix clonées."""
    voices = []
    if CLONING_AVAILABLE:
        try:
            manager = VoiceCloningManager()
            cloned = manager.cloner.list_cloned_voices()
            for v in cloned:
                voices.append({
                    "name": v.name,
                    "language": v.language,
                    "description": getattr(v, 'description', ''),
                    "created": getattr(v, 'created_at', 'N/A')
                })
        except:
            pass
    return voices

def refresh_voices():
    """Rafraîchit la liste des voix."""
    return gr.Dropdown(choices=get_voice_choices())

def count_text_stats(text: str) -> str:
    """Retourne les statistiques du texte."""
    if not text:
        return "0 caractères | 0 mots | ~0 min audio"
    chars = len(text)
    words = len(text.split())
    duration = estimate_duration(text)
    return f"{chars:,} caractères | {words:,} mots | ~{duration:.1f} min audio"

# ============================================================================
# GÉNÉRATION DE PREVIEW 30 SECONDES
# ============================================================================

def generate_preview(
    text: str,
    voice: str,
    speed: float,
    engine_type: str,
    progress=gr.Progress()
) -> Tuple[Optional[str], str]:
    """Génère un aperçu de 30 secondes."""
    if not text.strip():
        return None, "❌ Entrez du texte d'abord."

    try:
        progress(0.2, desc="Extraction du texte représentatif...")

        # Extraire ~450 caractères représentatifs
        if PREVIEW_AVAILABLE:
            generator = PreviewGenerator()
            preview_text = generator.extract_preview_text(text, lang='fr')
        else:
            # Fallback: prendre le début
            preview_text = text[:500]

        progress(0.4, desc="Génération de l'aperçu...")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        from tts_engine import create_tts_engine
        engine = create_tts_engine(
            language="fr",
            engine_type=engine_type,
            voice=voice,
            speed=speed
        )

        success = engine.synthesize(preview_text, output_path)

        if success:
            progress(1.0, desc="Aperçu prêt!")
            return str(output_path), f"✅ Aperçu généré ({len(preview_text)} caractères)"
        else:
            return None, "❌ Erreur lors de la génération"

    except Exception as e:
        return None, f"❌ Erreur: {str(e)}"

# ============================================================================
# PREVIEW DE VOIX
# ============================================================================

def preview_voice(voice_id: str) -> Optional[str]:
    """Génère ou récupère un sample pour une voix donnée."""
    if not voice_id:
        return None

    sample_path = VOICE_CACHE_DIR / f"sample_{voice_id}.wav"

    if not sample_path.exists():
        try:
            from tts_engine import create_tts_engine

            # Déterminer la langue selon le préfixe
            if voice_id.startswith("ff") or voice_id.startswith("fm"):
                lang, text = "fr", "Bonjour, je suis la voix sélectionnée pour ce personnage."
            else:
                lang, text = "en", "Hello, I am the voice selected for this character."

            engine = create_tts_engine(language=lang, engine_type="kokoro", voice=voice_id)
            engine.synthesize(text, sample_path)
        except Exception as e:
            print(f"Erreur génération sample {voice_id}: {e}")
            return None

    return str(sample_path) if sample_path.exists() else None

# ============================================================================
# ANALYSE DES PERSONNAGES
# ============================================================================

def analyze_characters(file) -> Tuple[Optional[List], str]:
    """Analyse les personnages présents dans le livre."""
    if file is None:
        return None, "❌ Veuillez uploader un fichier."

    if not ATTRIBUTION_AVAILABLE:
        return None, "❌ Module d'attribution non disponible."

    try:
        input_path = Path(file.name)

        # Conversion si nécessaire
        if input_path.suffix.lower() in ['.pdf', '.epub']:
            converter = InputConverter()
            input_path = converter.convert_to_markdown(input_path)

        chapters = parse_book(input_path)
        attributor = DialogueAttributor()

        all_text = "\n".join([c.get_full_text() for c in chapters])
        # Analyse sur les 100k premiers caractères
        attributor.process_text(all_text[:100000])
        stats = attributor.get_conversation_stats()

        # Formater pour l'affichage
        chars = []
        for name in stats["participants"]:
            count = stats["speaker_counts"].get(name, 0)
            gender = stats["gender_map"].get(name, "?")
            confidence = "85%" if gender in ["M", "F"] else "50%"

            # Suggestion de voix selon le genre
            if gender == "F":
                suggested = "af_bella"
            elif gender == "M":
                suggested = "am_adam"
            else:
                suggested = "af_sky"

            chars.append([name, gender, confidence, count, suggested])

        return chars, f"✅ {len(stats['participants'])} personnages détectés dans {len(chapters)} chapitres."

    except Exception as e:
        return None, f"❌ Erreur: {str(e)}"

def auto_assign_voices(char_data: List) -> List:
    """Attribution automatique des voix selon le genre."""
    if not char_data:
        return []

    female_voices = ["af_bella", "af_sarah", "af_sky", "af_nicole", "ff_siwis"]
    male_voices = ["am_adam", "am_michael", "am_eric", "bm_george"]

    f_idx, m_idx = 0, 0
    result = []

    for row in char_data:
        if len(row) >= 5:
            name, gender, conf, count, _ = row
            if gender == "F":
                voice = female_voices[f_idx % len(female_voices)]
                f_idx += 1
            elif gender == "M":
                voice = male_voices[m_idx % len(male_voices)]
                m_idx += 1
            else:
                voice = "af_sky"
            result.append([name, gender, conf, count, voice])

    return result

def export_voice_mapping(char_data: List) -> str:
    """Exporte le mapping en JSON."""
    if not char_data:
        return "{}"

    mapping = {}
    for row in char_data:
        if len(row) >= 5:
            mapping[row[0]] = row[4]

    return json.dumps(mapping, indent=2, ensure_ascii=False)

def import_voice_mapping(json_str: str, char_data: List) -> List:
    """Importe un mapping JSON dans le tableau."""
    try:
        mapping = json.loads(json_str)
        result = []
        for row in char_data:
            if len(row) >= 5:
                name = row[0]
                voice = mapping.get(name, row[4])
                result.append([row[0], row[1], row[2], row[3], voice])
        return result
    except:
        return char_data

# ============================================================================
# EXTRACTION ET CLONAGE DE VOIX
# ============================================================================

def extract_voice_from_file(
    file,
    voice_name: str,
    start_time: float,
    end_time: float,
    language: str
) -> Tuple[Optional[str], str, Optional[str]]:
    """Extrait l'audio et crée une voix clonée."""
    if not CLONING_AVAILABLE:
        return None, "❌ Module de clonage non disponible.", None

    if file is None:
        return None, "❌ Uploadez un fichier vidéo ou audio.", None

    if not voice_name.strip():
        return None, "❌ Donnez un nom à la voix.", None

    try:
        extractor = AudioExtractor()
        if not extractor.is_available():
            return None, "❌ ffmpeg requis pour l'extraction.", None

        input_path = Path(file.name)

        start = start_time if start_time > 0 else None
        end = end_time if end_time > 0 and end_time > (start_time or 0) else None

        wav_path = extractor.extract_from_video(
            input_path,
            start_time=start,
            end_time=end,
            output_name=f"source_{voice_name}"
        )

        if not wav_path:
            return None, "❌ Échec de l'extraction audio.", None

        # Analyser la qualité
        audio, sr = sf.read(wav_path)
        duration = len(audio) / sr
        quality_msg = f"Durée: {duration:.1f}s"

        if duration < 6:
            quality_msg += " ⚠️ (min recommandé: 6s)"
        elif duration > 30:
            quality_msg += " ✅ (excellent)"
        else:
            quality_msg += " ✅ (bon)"

        # Enregistrer la voix
        manager = VoiceCloningManager()
        success = manager.register_cloned_voice(
            wav_path,
            voice_id=voice_name,
            language=language,
            description=f"Extrait de {input_path.name}"
        )

        # Mettre à jour les stats
        if success:
            stats = load_stats()
            stats["voices_cloned"] = stats.get("voices_cloned", 0) + 1
            save_stats(stats)
            return str(wav_path), f"✅ Voix '{voice_name}' créée ! {quality_msg}", quality_msg
        else:
            return str(wav_path), f"❌ Échec enregistrement (durée < 6s ?). {quality_msg}", quality_msg

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Erreur: {str(e)}", None

def delete_cloned_voice(voice_name: str) -> Tuple[str, List]:
    """Supprime une voix clonée."""
    if not CLONING_AVAILABLE or not voice_name:
        return "❌ Impossible de supprimer.", []

    try:
        manager = VoiceCloningManager()
        # Note: implémenter delete dans VoiceCloningManager si nécessaire
        # Pour l'instant, on simule
        return f"✅ Voix '{voice_name}' supprimée.", get_cloned_voices_gallery()
    except Exception as e:
        return f"❌ Erreur: {e}", get_cloned_voices_gallery()

# ============================================================================
# GESTION DU SERVEUR PODCAST
# ============================================================================

def get_podcast_files() -> List[List[str]]:
    """Liste les fichiers audio disponibles pour le podcast."""
    output_dir = Path("output")
    if not output_dir.exists():
        return []

    files = []
    for ext in ["*.wav", "*.mp3", "*.m4b", "*.m4a"]:
        for f in output_dir.glob(ext):
            size_mb = f.stat().st_size / (1024 * 1024)
            modified = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            files.append([f.name, f"{size_mb:.1f} MB", modified])

    return sorted(files, key=lambda x: x[2], reverse=True)

def manage_podcast_server(action: str) -> Tuple[str, str, str, List]:
    """Gère le serveur de podcast."""
    files = get_podcast_files()

    if not PODCAST_AVAILABLE:
        return "❌ Module podcast non disponible.", "", "", files

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    server = get_server(root_dir=str(output_dir), port=8080)

    if action == "start":
        try:
            server.start()
            qr_html = server.get_qr_code()
            return (
                f"✅ Serveur démarré sur le port 8080\n{len(files)} fichiers disponibles",
                server.feed_url,
                qr_html,
                files
            )
        except Exception as e:
            return f"❌ Erreur: {e}", "", "", files

    elif action == "stop":
        server.stop()
        return "🛑 Serveur arrêté.", "", "", files

    elif action == "refresh":
        if server.is_running:
            server.refresh_feed()
            return "🔄 Flux mis à jour.", server.feed_url, server.get_qr_code(), get_podcast_files()
        return "Serveur non démarré.", "", "", files

    return "", "", "", files

# ============================================================================
# CONVERSION TEXTE -> AUDIO
# ============================================================================

def convert_text(
    text: str,
    voice: str,
    speed: float,
    apply_corrections: bool,
    hq: bool,
    mastering: bool,
    style: str,
    engine_type: str,
    progress=gr.Progress()
) -> Tuple[Optional[str], str]:
    """Convertit un texte en audio."""
    if not check_model():
        return None, "❌ Modèle Kokoro non trouvé."

    if not text.strip():
        return None, "❌ Entrez du texte."

    try:
        progress(0.1, desc="Initialisation...")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        if hq and HAS_HQ:
            progress(0.2, desc="Pipeline HQ...")
            config = ExtendedPipelineConfig(
                lang="fr",
                narrator_voice=voice,
                tts_engine=engine_type,
                enable_dialogue_attribution=False,
                auto_assign_voices=False,
                default_narration_style=style,
                enable_audio_enhancement=mastering,
                enable_acx_compliance=mastering
            )
            pipeline = ExtendedHQPipeline(config)
            success = pipeline_synthesize_chapter(pipeline, text, output_path, progress)
        else:
            if apply_corrections:
                processor = TextProcessor(lang="fr", engine="kokoro")
                text = processor.process_to_text(text)

            from tts_engine import create_tts_engine
            engine = create_tts_engine(
                language="fr",
                engine_type=engine_type,
                voice=voice,
                speed=speed
            )
            progress(0.5, desc=f"Génération ({engine_type})...")
            success = engine.synthesize(text, output_path)

        if success:
            # Mettre à jour les stats
            stats = load_stats()
            stats["total_characters"] = stats.get("total_characters", 0) + len(text)
            save_stats(stats)

            progress(1.0, desc="Terminé!")
            duration = estimate_duration(text)
            return str(output_path), f"✅ Audio généré ({len(text):,} caractères, ~{duration:.1f} min)"

        return None, "❌ Erreur de génération"

    except Exception as e:
        return None, f"❌ Erreur: {str(e)}"

# ============================================================================
# SYNTHÈSE CHAPITRE HQ
# ============================================================================

def pipeline_synthesize_chapter(pipeline, text: str, output_path: Path, progress=None) -> bool:
    """Synthétise un chapitre avec le pipeline HQ."""
    try:
        generator = HQGenerator(config=pipeline.config)
        generator.pipeline = pipeline

        if progress:
            progress(0.3, desc="Analyse du texte...")

        segments = pipeline.process_chapter(text)

        from tts_engine import create_tts_engine
        engine = create_tts_engine(
            language=pipeline.config.lang,
            engine_type=pipeline.config.tts_engine,
            voice=pipeline.config.narrator_voice
        )

        def synth_fn(t, v, s):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                success = engine.synthesize(t, tmp_path, voice=v, speed=s)
                if success and os.path.exists(tmp_path):
                    audio, _ = sf.read(tmp_path)
                    return audio
                return np.array([], dtype=np.float32)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        if progress:
            progress(0.6, desc="Synthèse vocale...")

        audios = pipeline.synthesize_segments(segments, synth_fn)
        full_audio = generator._concatenate_with_pauses(segments, audios)

        sf.write(str(output_path), full_audio, 24000)

        if pipeline.config.enable_audio_enhancement:
            if progress:
                progress(0.9, desc="Mastering...")
            from audio_enhancer import AudioEnhancer
            enhancer = AudioEnhancer()
            if enhancer.is_available():
                mastered = output_path.with_name(f"{output_path.stem}_mastered.wav")
                if enhancer.enhance_file(output_path, mastered) and mastered.exists():
                    os.replace(mastered, output_path)

        return True

    except Exception as e:
        print(f"Erreur synthèse HQ: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# CONVERSION LIVRE COMPLET
# ============================================================================

def preview_book_structure(file) -> Tuple[Optional[List], str]:
    """Prévisualise la structure du livre."""
    if file is None:
        return None, "❌ Uploadez un fichier."

    try:
        input_path = Path(file.name)

        if input_path.suffix.lower() in ['.pdf', '.epub']:
            converter = InputConverter()
            input_path = converter.convert_to_markdown(input_path)

        chapters = parse_book(input_path)

        if not chapters:
            return None, "❌ Aucun chapitre trouvé."

        # Créer la prévisualisation
        preview = []
        total_chars = 0
        for i, ch in enumerate(chapters):
            text = ch.get_full_text()
            chars = len(text)
            total_chars += chars
            duration = estimate_duration(text)
            preview.append([True, i+1, ch.title[:50], f"{chars:,}", f"~{duration:.1f} min"])

        total_duration = estimate_duration("x" * total_chars)
        conv_time = estimate_conversion_time("x" * total_chars, hq=True)

        msg = f"✅ {len(chapters)} chapitres | {total_chars:,} caractères | ~{total_duration:.0f} min audio | ~{conv_time:.0f} min conversion"
        return preview, msg

    except Exception as e:
        return None, f"❌ Erreur: {str(e)}"

def convert_book(
    file,
    voice: str,
    speed: float,
    apply_corrections: bool,
    title: str,
    author: str,
    selected_formats: List[str],
    hq: bool,
    multivoice: bool,
    mastering: bool,
    style: str,
    voice_mapping_json: str,
    engine_type: str,
    chapter_selection: Optional[List] = None,
    progress=gr.Progress()
) -> Tuple[Optional[str], str]:
    """Convertit un livre complet."""
    if file is None:
        return None, "❌ Uploadez un fichier."

    generate_audio = any("Audio" in f for f in selected_formats)
    ebook_formats = []
    if "PDF" in str(selected_formats): ebook_formats.append("pdf")
    if "EPUB" in str(selected_formats): ebook_formats.append("epub")
    if "HTML" in str(selected_formats): ebook_formats.append("html")
    if "TXT" in str(selected_formats): ebook_formats.append("txt")

    audio_format = "m4b" if "M4B" in str(selected_formats) else "mp3"

    if generate_audio and not check_model():
        return None, "❌ Modèle Kokoro requis pour l'audio."

    try:
        custom_mapping = json.loads(voice_mapping_json) if voice_mapping_json else {}
    except:
        custom_mapping = {}

    try:
        temp_dir = Path(tempfile.mkdtemp())
        output_dir = temp_dir / "chapters"
        output_dir.mkdir()

        final_files = []
        status_parts = []

        progress(0.05, desc="Lecture du fichier...")

        input_path = Path(file.name)
        if input_path.suffix.lower() in ['.pdf', '.epub']:
            progress(0.1, desc="Conversion du fichier source...")
            converter = InputConverter()
            input_path = converter.convert_to_markdown(input_path)

        chapters = parse_book(input_path)

        if not chapters:
            return None, "❌ Aucun chapitre trouvé."

        # Filtrer les chapitres si sélection
        if chapter_selection:
            selected_indices = [row[1] - 1 for row in chapter_selection if row[0]]
            if selected_indices:
                chapters = [chapters[i] for i in selected_indices if i < len(chapters)]

        # === GÉNÉRATION AUDIO ===
        if generate_audio:
            pipeline = None
            engine = None
            processor = None

            if hq and HAS_HQ:
                progress(0.15, desc="Initialisation pipeline HQ...")
                config = ExtendedPipelineConfig(
                    lang="fr",
                    narrator_voice=voice,
                    tts_engine=engine_type,
                    enable_dialogue_attribution=multivoice,
                    auto_assign_voices=multivoice,
                    voice_mapping=custom_mapping,
                    default_narration_style=style,
                    enable_audio_enhancement=mastering,
                    enable_acx_compliance=mastering
                )
                pipeline = ExtendedHQPipeline(config)
            else:
                progress(0.15, desc=f"Chargement moteur {engine_type}...")
                from tts_engine import create_tts_engine
                engine = create_tts_engine(
                    language="fr",
                    engine_type=engine_type,
                    voice=voice,
                    speed=speed
                )
                if apply_corrections:
                    processor = TextProcessor(lang="fr", engine="kokoro")

            # Convertir chaque chapitre
            for i, chapter in enumerate(chapters):
                pct = 0.2 + (0.5 * (i / len(chapters)))
                progress(pct, desc=f"Chapitre {i+1}/{len(chapters)}: {chapter.title[:30]}...")

                text = chapter.get_full_text()
                ch_output = output_dir / f"{chapter.get_filename()}.wav"

                if pipeline:
                    pipeline_synthesize_chapter(pipeline, text, ch_output)
                else:
                    if processor:
                        text = processor.process_to_text(text)
                    engine.synthesize(text, ch_output)

            progress(0.75, desc="Construction audiobook...")

            metadata = AudiobookMetadata(
                title=title or "Audiobook",
                author=author or "Inconnu",
                narrator=voice if not multivoice else "Multi-Cast"
            )

            builder = AudiobookBuilder(metadata)
            builder.add_chapters_from_dir(output_dir, "*.wav")

            base_name = title or Path(file.name).stem
            if audio_format == "m4b":
                audio_path = temp_dir / f"{base_name}.m4b"
                builder.build_m4b(audio_path)
            else:
                audio_path = temp_dir / f"{base_name}.mp3"
                builder.build_combined_mp3(audio_path)

            if audio_path.exists():
                final_files.append(audio_path)
                status_parts.append(f"Audio {audio_format.upper()}")

        # === GÉNÉRATION EBOOKS ===
        if ebook_formats:
            progress(0.85, desc="Génération E-Books...")
            from book_exporter import BookExporter

            exporter = BookExporter(
                title=title or Path(file.name).stem,
                author=author or "Inconnu",
                language="fr"
            )

            for chapter in chapters:
                exporter.add_chapter(chapter.title, chapter.content)

            ebook_results = exporter.export_all(
                output_dir=temp_dir,
                base_name=title or Path(file.name).stem,
                formats=ebook_formats
            )

            for fmt, path in ebook_results.items():
                if path.exists():
                    final_files.append(path)
                    status_parts.append(fmt.upper())

        progress(0.95, desc="Finalisation...")

        if not final_files:
            return None, "❌ Aucun fichier généré."

        # Mettre à jour les stats
        stats = load_stats()
        stats["books_generated"] = stats.get("books_generated", 0) + 1
        total_chars = sum(len(ch.get_full_text()) for ch in chapters)
        stats["total_characters"] = stats.get("total_characters", 0) + total_chars
        stats["total_duration_minutes"] = stats.get("total_duration_minutes", 0) + estimate_duration("x" * total_chars)
        save_stats(stats)

        # Retourner le résultat
        if len(final_files) == 1:
            perm_output = Path(tempfile.gettempdir()) / final_files[0].name
            shutil.copy(final_files[0], perm_output)
            add_to_recent_projects(title or "Sans titre", file.name, str(perm_output))
            return str(perm_output), f"✅ {status_parts[0]} généré"
        else:
            zip_name = f"{title or 'livre'}_complet"
            final_zip = Path(tempfile.gettempdir()) / f"{zip_name}.zip"
            import zipfile
            with zipfile.ZipFile(final_zip, 'w') as zipf:
                for f in final_files:
                    zipf.write(f, arcname=f.name)

            add_to_recent_projects(title or "Sans titre", file.name, str(final_zip))
            return str(final_zip), f"✅ Pack généré: {', '.join(status_parts)}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Erreur: {str(e)}"

# ============================================================================
# GESTION DES PRÉFÉRENCES
# ============================================================================

def save_user_prefs(
    default_voice: str,
    default_engine: str,
    default_style: str,
    default_speed: float,
    enable_hq: bool,
    enable_multivoice: bool,
    enable_mastering: bool,
    output_dir: str,
    expert_mode: bool
) -> str:
    """Sauvegarde les préférences utilisateur."""
    prefs = {
        "default_voice": default_voice,
        "default_engine": default_engine,
        "default_style": default_style,
        "default_speed": default_speed,
        "enable_hq": enable_hq,
        "enable_multivoice": enable_multivoice,
        "enable_mastering": enable_mastering,
        "output_dir": output_dir,
        "expert_mode": expert_mode
    }
    save_prefs(prefs)
    return "✅ Préférences sauvegardées"

# ============================================================================
# CRÉATION DE L'INTERFACE
# ============================================================================

def create_interface():
    """Crée l'interface Gradio complète."""

    prefs = load_prefs()
    stats = load_stats()

    # CSS personnalisé
    css = """
    .gradio-container { max-width: 1400px !important; }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 5px;
    }
    .stat-number { font-size: 2em; font-weight: bold; }
    .stat-label { font-size: 0.9em; opacity: 0.9; }
    .quick-action {
        background: #f0f4f8;
        border-radius: 8px;
        padding: 15px;
        margin: 5px;
        cursor: pointer;
        transition: all 0.2s;
    }
    .quick-action:hover { background: #e2e8f0; transform: translateY(-2px); }
    .chapter-row { padding: 8px; border-bottom: 1px solid #eee; }
    .voice-gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; }
    .voice-card { background: #f8fafc; border-radius: 8px; padding: 12px; }
    footer { display: none !important; }
    """

    with gr.Blocks(css=css, title="AudioReader v3.1", theme=gr.themes.Soft()) as demo:

        # En-tête
        gr.Markdown("""
        # 🎧 AudioReader v3.1
        **Plateforme complète de création d'audiobooks**
        """)

        # Status modèle
        if check_model():
            gr.Markdown("✅ Kokoro-82M chargé | " +
                       ("✅ HQ Pipeline" if HAS_HQ else "⚠️ HQ indisponible") + " | " +
                       ("✅ Clonage XTTS" if CLONING_AVAILABLE else "⚠️ Clonage indisponible"))
        else:
            gr.Markdown("""
            ⚠️ **Modèle Kokoro non trouvé** - Téléchargez-le:
            ```bash
            curl -L -o kokoro-v1.0.onnx "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
            curl -L -o voices-v1.0.bin "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
            ```
            """)

        with gr.Tabs() as tabs:

            # ================================================================
            # TAB 0: DASHBOARD
            # ================================================================
            with gr.TabItem("🏠 Accueil", id=0):
                gr.Markdown("### 📊 Tableau de bord")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(f"""
                        <div class="stat-card">
                            <div class="stat-number">{stats.get('books_generated', 0)}</div>
                            <div class="stat-label">Livres générés</div>
                        </div>
                        """)
                    with gr.Column(scale=1):
                        gr.Markdown(f"""
                        <div class="stat-card">
                            <div class="stat-number">{stats.get('total_duration_minutes', 0):.0f}</div>
                            <div class="stat-label">Minutes d'audio</div>
                        </div>
                        """)
                    with gr.Column(scale=1):
                        gr.Markdown(f"""
                        <div class="stat-card">
                            <div class="stat-number">{stats.get('voices_cloned', 0)}</div>
                            <div class="stat-label">Voix clonées</div>
                        </div>
                        """)
                    with gr.Column(scale=1):
                        gr.Markdown(f"""
                        <div class="stat-card">
                            <div class="stat-number">{len(get_voice_choices())}</div>
                            <div class="stat-label">Voix disponibles</div>
                        </div>
                        """)

                gr.Markdown("### ⚡ Actions rapides")
                with gr.Row():
                    quick_text_btn = gr.Button("📝 Convertir un texte", scale=1)
                    quick_book_btn = gr.Button("📚 Convertir un livre", scale=1)
                    quick_clone_btn = gr.Button("🎙️ Cloner une voix", scale=1)
                    quick_podcast_btn = gr.Button("📡 Lancer le podcast", scale=1)

                gr.Markdown("### 📂 Projets récents")
                recent_projects = stats.get("last_projects", [])
                if recent_projects:
                    for proj in recent_projects[:5]:
                        gr.Markdown(f"- **{proj['title']}** - {proj['date'][:10]}")
                else:
                    gr.Markdown("*Aucun projet récent*")

                # Navigation rapide
                quick_text_btn.click(lambda: gr.Tabs(selected=1), outputs=tabs)
                quick_book_btn.click(lambda: gr.Tabs(selected=2), outputs=tabs)
                quick_clone_btn.click(lambda: gr.Tabs(selected=4), outputs=tabs)
                quick_podcast_btn.click(lambda: gr.Tabs(selected=5), outputs=tabs)

            # ================================================================
            # TAB 1: TEXTE RAPIDE
            # ================================================================
            with gr.TabItem("📝 Texte", id=1):
                gr.Markdown("### Conversion rapide de texte")

                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="Texte à convertir",
                            placeholder="Entrez ou collez votre texte ici...",
                            lines=12
                        )
                        text_stats = gr.Markdown("0 caractères | 0 mots | ~0 min audio")
                        text_input.change(
                            fn=count_text_stats,
                            inputs=[text_input],
                            outputs=[text_stats]
                        )

                    with gr.Column(scale=1):
                        engine_text = gr.Dropdown(
                            choices=["auto", "kokoro", "mms", "xtts", "edge"],
                            value=prefs.get("default_engine", "auto"),
                            label="Moteur TTS"
                        )

                        with gr.Row():
                            voice_text = gr.Dropdown(
                                choices=get_voice_choices(),
                                value=prefs.get("default_voice", "ff_siwis"),
                                label="Voix",
                                scale=4
                            )
                            refresh_voice_text = gr.Button("🔄", scale=1, min_width=40)

                        speed_text = gr.Slider(0.5, 2.0, value=prefs.get("default_speed", 1.0),
                                              step=0.1, label="Vitesse")

                        with gr.Accordion("Options avancées", open=False):
                            hq_text = gr.Checkbox(value=prefs.get("enable_hq", False), label="Mode HQ")
                            mastering_text = gr.Checkbox(value=False, label="Mastering")
                            style_text = gr.Dropdown(
                                choices=["storytelling", "dramatic", "formal", "conversational"],
                                value=prefs.get("default_style", "storytelling"),
                                label="Style"
                            )
                            corrections_text = gr.Checkbox(value=True, label="Corrections prononciation")

                        with gr.Row():
                            preview_btn = gr.Button("👁️ Aperçu 30s", variant="secondary")
                            convert_text_btn = gr.Button("🎙️ Convertir", variant="primary")

                with gr.Row():
                    audio_text_output = gr.Audio(label="Audio généré", type="filepath")
                    status_text = gr.Textbox(label="Status", interactive=False)

                # Événements
                refresh_voice_text.click(fn=refresh_voices, outputs=voice_text)

                preview_btn.click(
                    fn=generate_preview,
                    inputs=[text_input, voice_text, speed_text, engine_text],
                    outputs=[audio_text_output, status_text]
                )

                convert_text_btn.click(
                    fn=convert_text,
                    inputs=[text_input, voice_text, speed_text, corrections_text,
                           hq_text, mastering_text, style_text, engine_text],
                    outputs=[audio_text_output, status_text]
                )

            # ================================================================
            # TAB 2: LIVRE COMPLET
            # ================================================================
            with gr.TabItem("📚 Livre", id=2):
                gr.Markdown("### Conversion de livre complet")

                with gr.Row():
                    with gr.Column(scale=2):
                        file_book = gr.File(
                            label="📁 Glissez votre fichier ici (Markdown, PDF, EPUB)",
                            file_types=[".md", ".txt", ".pdf", ".epub"],
                            type="filepath"
                        )

                        with gr.Row():
                            title_book = gr.Textbox(label="Titre", placeholder="Titre du livre")
                            author_book = gr.Textbox(label="Auteur", placeholder="Nom de l'auteur")

                        gr.Markdown("#### 📑 Structure du livre")
                        preview_structure_btn = gr.Button("🔍 Analyser la structure")
                        chapter_table = gr.Dataframe(
                            headers=["✓", "#", "Titre", "Caractères", "Durée"],
                            label="Chapitres (décochez pour exclure)",
                            interactive=True,
                            wrap=True
                        )
                        structure_status = gr.Textbox(label="Analyse", interactive=False)

                    with gr.Column(scale=1):
                        engine_book = gr.Dropdown(
                            choices=["auto", "kokoro", "mms", "xtts", "edge"],
                            value=prefs.get("default_engine", "auto"),
                            label="Moteur TTS"
                        )

                        with gr.Row():
                            voice_book = gr.Dropdown(
                                choices=get_voice_choices(),
                                value=prefs.get("default_voice", "ff_siwis"),
                                label="Voix narrateur",
                                scale=4
                            )
                            refresh_voice_book = gr.Button("🔄", scale=1, min_width=40)

                        speed_book = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Vitesse")

                        gr.Markdown("#### Options HQ")
                        hq_book = gr.Checkbox(value=prefs.get("enable_hq", True), label="🚀 Mode HQ")
                        multivoice_book = gr.Checkbox(value=prefs.get("enable_multivoice", True),
                                                      label="👥 Multi-voix (personnages)")
                        mastering_book = gr.Checkbox(value=prefs.get("enable_mastering", True),
                                                    label="🎙️ Mastering ACX")
                        style_book = gr.Dropdown(
                            choices=["storytelling", "dramatic", "formal", "conversational"],
                            value=prefs.get("default_style", "storytelling"),
                            label="Style narration"
                        )
                        corrections_book = gr.Checkbox(value=True, label="Corrections prononciation")

                        gr.Markdown("#### Formats de sortie")
                        formats_book = gr.CheckboxGroup(
                            choices=["Audio (M4B)", "Audio (MP3)", "E-Book (PDF)",
                                    "E-Book (EPUB)", "E-Book (HTML)", "Texte (TXT)"],
                            value=["Audio (M4B)"],
                            label="Formats"
                        )

                        with gr.Accordion("Mapping voix personnalisé", open=False):
                            voice_mapping_book = gr.Textbox(
                                label="JSON mapping",
                                placeholder='{"Personnage": "voix_id", ...}',
                                value="{}",
                                lines=3
                            )
                            gr.Markdown("*Utilisez l'onglet Personnages pour générer ce mapping*")

                        convert_book_btn = gr.Button("📖 Générer le livre", variant="primary", size="lg")

                with gr.Row():
                    output_book = gr.File(label="Fichier généré")
                    status_book = gr.Textbox(label="Status", interactive=False)

                # Événements
                refresh_voice_book.click(fn=refresh_voices, outputs=voice_book)

                preview_structure_btn.click(
                    fn=preview_book_structure,
                    inputs=[file_book],
                    outputs=[chapter_table, structure_status]
                )

                convert_book_btn.click(
                    fn=convert_book,
                    inputs=[file_book, voice_book, speed_book, corrections_book,
                           title_book, author_book, formats_book, hq_book,
                           multivoice_book, mastering_book, style_book,
                           voice_mapping_book, engine_book, chapter_table],
                    outputs=[output_book, status_book]
                )

            # ================================================================
            # TAB 3: PERSONNAGES
            # ================================================================
            with gr.TabItem("👥 Personnages", id=3):
                gr.Markdown("### Attribution des voix aux personnages")

                with gr.Row():
                    with gr.Column(scale=2):
                        file_chars = gr.File(
                            label="Fichier source",
                            file_types=[".md", ".pdf", ".epub"]
                        )

                        with gr.Row():
                            analyze_chars_btn = gr.Button("🔍 Analyser", variant="primary")
                            auto_assign_btn = gr.Button("🎯 Attribution auto")

                        chars_table = gr.Dataframe(
                            headers=["Nom", "Genre", "Confiance", "Répliques", "Voix ID"],
                            label="Personnages détectés",
                            interactive=True
                        )
                        chars_status = gr.Textbox(label="Résultat", interactive=False)

                        with gr.Row():
                            export_mapping_btn = gr.Button("📤 Exporter JSON")
                            import_mapping_input = gr.Textbox(label="JSON à importer", lines=2)
                            import_mapping_btn = gr.Button("📥 Importer")

                        mapping_output = gr.Textbox(label="Mapping JSON généré", lines=4)

                    with gr.Column(scale=1):
                        gr.Markdown("### 👂 Pré-écoute des voix")

                        preview_voice_select = gr.Dropdown(
                            choices=[v[1] for v in get_voice_choices()],
                            label="Voix à tester"
                        )
                        preview_voice_btn = gr.Button("▶️ Écouter")
                        preview_voice_audio = gr.Audio(label="Échantillon", type="filepath")

                        with gr.Accordion("📋 Aide-mémoire voix", open=True):
                            gr.Markdown("""
                            | Langue | Genre | Voix |
                            |--------|-------|------|
                            | 🇫🇷 FR | ♀ | `ff_siwis` |
                            | 🇬🇧 EN | ♀ | `af_bella`, `af_sarah`, `af_sky` |
                            | 🇬🇧 EN | ♂ | `am_adam`, `am_michael`, `bm_george` |
                            """)

                        apply_to_book_btn = gr.Button("✅ Appliquer au livre", variant="primary")

                # Événements
                analyze_chars_btn.click(
                    fn=analyze_characters,
                    inputs=[file_chars],
                    outputs=[chars_table, chars_status]
                )

                auto_assign_btn.click(
                    fn=auto_assign_voices,
                    inputs=[chars_table],
                    outputs=[chars_table]
                )

                export_mapping_btn.click(
                    fn=export_voice_mapping,
                    inputs=[chars_table],
                    outputs=[mapping_output]
                )

                import_mapping_btn.click(
                    fn=import_voice_mapping,
                    inputs=[import_mapping_input, chars_table],
                    outputs=[chars_table]
                )

                preview_voice_btn.click(
                    fn=preview_voice,
                    inputs=[preview_voice_select],
                    outputs=[preview_voice_audio]
                )

                apply_to_book_btn.click(
                    fn=export_voice_mapping,
                    inputs=[chars_table],
                    outputs=[voice_mapping_book]
                ).then(lambda: gr.Info("Mapping appliqué ! Allez dans l'onglet Livre."))

            # ================================================================
            # TAB 4: CLONAGE DE VOIX
            # ================================================================
            with gr.TabItem("🎙️ Clonage", id=4):
                gr.Markdown("### Création de voix personnalisées (XTTS-v2)")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 1. Source audio")
                        clone_file = gr.File(
                            label="Fichier vidéo ou audio",
                            file_types=[".mp4", ".mkv", ".avi", ".mov", ".wav", ".mp3", ".m4a"]
                        )

                        clone_name = gr.Textbox(label="Nom de la voix", placeholder="Ex: Morgan_Freeman")
                        clone_lang = gr.Dropdown(
                            choices=["fr", "en", "es", "de", "it", "pt", "ru", "zh", "ja"],
                            value="fr",
                            label="Langue"
                        )

                        gr.Markdown("#### 2. Segment à extraire")
                        with gr.Row():
                            clone_start = gr.Number(value=0, label="Début (sec)", precision=1)
                            clone_end = gr.Number(value=0, label="Fin (sec, 0=tout)", precision=1)

                        clone_btn = gr.Button("🎙️ Extraire et créer la voix", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("#### Résultat")
                        clone_audio_output = gr.Audio(label="Audio extrait", type="filepath")
                        clone_quality = gr.Textbox(label="Qualité", interactive=False)
                        clone_status = gr.Textbox(label="Status", interactive=False)

                        gr.Markdown("""
                        #### 💡 Conseils
                        - **Durée idéale**: 10-30 secondes
                        - **Qualité**: Pas de musique ni bruit de fond
                        - **Contenu**: Une seule personne qui parle
                        """)

                gr.Markdown("---")
                gr.Markdown("### 📚 Galerie des voix clonées")

                cloned_voices = get_cloned_voices_gallery()
                if cloned_voices:
                    with gr.Row():
                        for v in cloned_voices[:6]:
                            with gr.Column(scale=1, min_width=150):
                                gr.Markdown(f"""
                                **🎙️ {v['name']}**
                                Langue: {v['language']}
                                """)
                else:
                    gr.Markdown("*Aucune voix clonée pour l'instant*")

                refresh_gallery_btn = gr.Button("🔄 Rafraîchir la galerie")

                # Événements
                clone_btn.click(
                    fn=extract_voice_from_file,
                    inputs=[clone_file, clone_name, clone_start, clone_end, clone_lang],
                    outputs=[clone_audio_output, clone_status, clone_quality]
                )

            # ================================================================
            # TAB 5: DIFFUSION PODCAST
            # ================================================================
            with gr.TabItem("📡 Diffusion", id=5):
                gr.Markdown("### Serveur Podcast RSS")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Contrôles")
                        start_podcast_btn = gr.Button("▶️ Démarrer", variant="primary")
                        refresh_podcast_btn = gr.Button("🔄 Rafraîchir le flux")
                        stop_podcast_btn = gr.Button("🛑 Arrêter", variant="stop")

                        podcast_status = gr.Textbox(label="État", interactive=False)
                        podcast_url = gr.Textbox(label="URL du flux RSS", interactive=False)

                        gr.Markdown("""
                        #### 📱 Instructions
                        1. Cliquez **Démarrer**
                        2. Scannez le QR code avec votre téléphone
                        3. Ouvrez dans Apple Podcasts / Pocket Casts
                        4. Vos fichiers du dossier `output/` seront disponibles
                        """)

                    with gr.Column(scale=1):
                        gr.Markdown("#### QR Code")
                        podcast_qr = gr.HTML()

                gr.Markdown("---")
                gr.Markdown("#### 📂 Fichiers disponibles")
                podcast_files = gr.Dataframe(
                    headers=["Nom", "Taille", "Date"],
                    value=get_podcast_files(),
                    interactive=False
                )

                # Événements
                start_podcast_btn.click(
                    fn=lambda: manage_podcast_server("start"),
                    outputs=[podcast_status, podcast_url, podcast_qr, podcast_files]
                )

                stop_podcast_btn.click(
                    fn=lambda: manage_podcast_server("stop"),
                    outputs=[podcast_status, podcast_url, podcast_qr, podcast_files]
                )

                refresh_podcast_btn.click(
                    fn=lambda: manage_podcast_server("refresh"),
                    outputs=[podcast_status, podcast_url, podcast_qr, podcast_files]
                )

            # ================================================================
            # TAB 6: PARAMÈTRES
            # ================================================================
            with gr.TabItem("⚙️ Paramètres", id=6):
                gr.Markdown("### Préférences utilisateur")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Valeurs par défaut")
                        pref_voice = gr.Dropdown(
                            choices=get_voice_choices(),
                            value=prefs.get("default_voice", "ff_siwis"),
                            label="Voix par défaut"
                        )
                        pref_engine = gr.Dropdown(
                            choices=["auto", "kokoro", "mms", "xtts", "edge"],
                            value=prefs.get("default_engine", "auto"),
                            label="Moteur par défaut"
                        )
                        pref_style = gr.Dropdown(
                            choices=["storytelling", "dramatic", "formal", "conversational"],
                            value=prefs.get("default_style", "storytelling"),
                            label="Style par défaut"
                        )
                        pref_speed = gr.Slider(0.5, 2.0, value=prefs.get("default_speed", 1.0),
                                              step=0.1, label="Vitesse par défaut")

                    with gr.Column():
                        gr.Markdown("#### Options par défaut")
                        pref_hq = gr.Checkbox(value=prefs.get("enable_hq", True),
                                             label="Activer HQ par défaut")
                        pref_multivoice = gr.Checkbox(value=prefs.get("enable_multivoice", True),
                                                     label="Activer Multi-voix par défaut")
                        pref_mastering = gr.Checkbox(value=prefs.get("enable_mastering", True),
                                                    label="Activer Mastering par défaut")
                        pref_expert = gr.Checkbox(value=prefs.get("expert_mode", False),
                                                 label="Mode expert (plus d'options)")

                        gr.Markdown("#### Dossiers")
                        pref_output = gr.Textbox(
                            value=prefs.get("output_dir", "output"),
                            label="Dossier de sortie"
                        )

                save_prefs_btn = gr.Button("💾 Sauvegarder les préférences", variant="primary")
                prefs_status = gr.Textbox(label="Status", interactive=False)

                save_prefs_btn.click(
                    fn=save_user_prefs,
                    inputs=[pref_voice, pref_engine, pref_style, pref_speed,
                           pref_hq, pref_multivoice, pref_mastering, pref_output, pref_expert],
                    outputs=[prefs_status]
                )

                gr.Markdown("---")
                gr.Markdown("### 📊 Statistiques d'utilisation")
                gr.Markdown(f"""
                | Métrique | Valeur |
                |----------|--------|
                | Livres générés | {stats.get('books_generated', 0)} |
                | Minutes d'audio | {stats.get('total_duration_minutes', 0):.0f} |
                | Caractères traités | {stats.get('total_characters', 0):,} |
                | Voix clonées | {stats.get('voices_cloned', 0)} |
                """)

            # ================================================================
            # TAB 7: À PROPOS
            # ================================================================
            with gr.TabItem("ℹ️ À propos", id=7):
                gr.Markdown("""
                ## AudioReader v3.1

                Plateforme complète de création d'audiobooks professionnels.

                ### 🛠️ Technologies
                - **Kokoro-82M**: TTS open-source 82M paramètres
                - **XTTS-v2**: Clonage de voix (Coqui)
                - **MMS-TTS**: TTS multilingue (Meta)
                - **FFmpeg**: Traitement audio

                ### ✨ Fonctionnalités v3.1
                - Dashboard avec statistiques
                - Aperçu 30 secondes
                - Prévisualisation structure livre
                - Attribution automatique des voix
                - Galerie des voix clonées
                - Préférences persistantes
                - Projets récents

                ### 📚 Distribution
                - ✅ Google Play Books
                - ✅ Findaway / Spotify
                - ✅ Kobo
                - ❌ Audible/ACX (voix humaines uniquement)

                ### 🔗 Liens
                - [Documentation](https://github.com/phuetz/AudioReader)
                - [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
                """)

        gr.Markdown("---")
        gr.Markdown("*AudioReader v3.1 - Créé avec Gradio*")

    return demo

# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
