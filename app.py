#!/usr/bin/env python3
"""
AudioReader - Interface Web Gradio

Interface graphique pour convertir des livres Markdown en audiobooks.
Utilise Kokoro-82M pour une qualité proche d'ElevenLabs.

Lancer avec: python app.py
Puis ouvrir: http://localhost:7860
"""
import sys
from pathlib import Path
import tempfile
import shutil
import soundfile as sf
import numpy as np
import json

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
from markdown_parser import parse_book
from tts_kokoro_engine import KokoroEngine, KOKORO_VOICES
from text_processor import TextProcessor
from audiobook_builder import AudiobookBuilder, AudiobookMetadata
from input_converter import InputConverter

# Clonage
try:
    from voice_cloning import VoiceCloningManager
    from audio_extractor import AudioExtractor
    CLONING_AVAILABLE = True
except ImportError:
    CLONING_AVAILABLE = False

# Imports HQ
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


# Configuration
MODEL_PATH = Path("kokoro-v1.0.onnx")
VOICES_PATH = Path("voices-v1.0.bin")


def manage_podcast_server(action: str):
    """Gère le serveur de podcast."""
    if not PODCAST_AVAILABLE:
        return "❌ Module podcast non disponible.", "", ""
        
    # Dossier servi : output à la racine du projet
    root_output = Path("output")
    root_output.mkdir(exist_ok=True)
    
    server = get_server(root_dir=str(root_output), port=8080)
    
    if action == "start":
        try:
            server.start()
            qr_html = server.get_qr_code()
            return (
                f"✅ Serveur démarré !\nURL du flux : {server.feed_url}\n\nCopiez vos fichiers audio dans le dossier 'output' pour qu'ils apparaissent.",
                server.feed_url,
                qr_html
            )
        except Exception as e:
            return f"❌ Erreur démarrage : {e}", "", ""
            
    elif action == "stop":
        server.stop()
        return "🛑 Serveur arrêté.", "", ""
        
    elif action == "refresh":
        if server.is_running:
            server.refresh_feed()
            return "🔄 Flux RSS mis à jour.", server.feed_url, server.get_qr_code()
        else:
            return "Le serveur n'est pas démarré.", "", ""

    return "", "", ""


def check_model():
    """Vérifie si le modèle est disponible."""
    return MODEL_PATH.exists() and VOICES_PATH.exists()


def get_voice_choices():
    """Retourne la liste des voix pour le dropdown (Kokoro + Cloned)."""
    choices = []
    
    # 1. Voix Kokoro
    for vid, info in KOKORO_VOICES.items():
        gender = "♀" if info["gender"] == "F" else "♂"
        label = f"{info['name']} {gender} ({info['desc']})"
        choices.append((label, vid))
        
    # 2. Voix Clonées
    if CLONING_AVAILABLE:
        try:
            manager = VoiceCloningManager()
            cloned = manager.cloner.list_cloned_voices()
            for v in cloned:
                label = f"🎙️ {v.name} ({v.language})"
                choices.append((label, v.name))
        except Exception as e:
            print(f"Erreur chargement voix clonées: {e}")
            
    return choices


def refresh_voices():
    """Rafraîchit la liste des voix."""
    return gr.Dropdown(choices=get_voice_choices())


def extract_voice_from_file(
    file,
    voice_name: str,
    start_time: float,
    end_time: float,
    language: str
):
    """Extrait l'audio et crée une voix clonée."""
    if not CLONING_AVAILABLE:
        return None, "❌ Module de clonage non disponible."
        
    if file is None:
        return None, "❌ Veuillez uploader un fichier (Vidéo/Audio)."
        
    if not voice_name.strip():
        return None, "❌ Veuillez donner un nom à la voix."

    try:
        extractor = AudioExtractor()
        if not extractor.is_available():
            return None, "❌ ffmpeg requis pour l'extraction."

        # Extraction
        input_path = Path(file.name)
        
        # Si start/end sont à 0, on prend tout (None)
        start = start_time if start_time > 0 else None
        end = end_time if end_time > 0 and end_time > start_time else None
        
        wav_path = extractor.extract_from_video(
            input_path, 
            start_time=start, 
            end_time=end,
            output_name=f"source_{voice_name}"
        )
        
        if not wav_path:
            return None, "❌ Échec de l'extraction audio."
            
        # Enregistrement
        manager = VoiceCloningManager()
        success = manager.register_cloned_voice(
            wav_path,
            voice_id=voice_name, # L'ID sera le nom directement pour simplifier
            language=language,
            description=f"Extrait de {input_path.name}"
        )
        
        if success:
            return str(wav_path), f"✅ Voix '{voice_name}' créée avec succès ! Elle est maintenant disponible dans les listes."
        else:
            return str(wav_path), "❌ Erreur lors de l'enregistrement de la voix (durée < 6s ?)."
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Erreur: {str(e)}"


from dialogue_attribution import DialogueAttributor

def preview_voice(voice_id):
    """Génère ou récupère un sample pour une voix donnée."""
    if not voice_id:
        return None
        
    sample_dir = Path(".voice_cache/samples")
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / f"sample_{voice_id}.wav"
    
    if not sample_path.exists():
        # Générer le sample
        try:
            from tts_engine import create_tts_engine
            # On utilise Kokoro ou MMS selon le prefixe
            engine_type = "mms" if voice_id == "fr" else "kokoro"
            
            engine = create_tts_engine(language="fr", engine_type=engine_type, voice=voice_id)
            
            text = "Bonjour, je suis la voix sélectionnée pour ce personnage."
            if voice_id.startswith("a") or voice_id.startswith("b"): # Anglais
                engine = create_tts_engine(language="en", engine_type="kokoro", voice=voice_id)
                text = "Hello, I am the voice selected for this character."
                
            engine.synthesize(text, sample_path)
        except Exception as e:
            print(f"Erreur génération sample {voice_id}: {e}")
            return None
            
    return str(sample_path)


def analyze_characters(file):
    """Analyse les personnages présents dans le livre."""
    if file is None:
        return None, "❌ Veuillez uploader un fichier."

    try:
        # Conversion automatique si nécessaire
        input_path = Path(file.name)
        if input_path.suffix.lower() in ['.pdf', '.epub']:
            converter = InputConverter()
            # On convertit dans le dossier temp du fichier uploadé
            input_path = converter.convert_to_markdown(input_path)
            
        chapters = parse_book(input_path)
        attributor = DialogueAttributor()
        
        all_text = "\n".join([c.get_full_text() for c in chapters])
        # Analyse sur les 100k premiers caractères
        results = attributor.process_text(all_text[:100000])
        stats = attributor.get_conversation_stats()
        
        # Formater pour l'affichage avec suggestion de voix
        chars = []
        for name in stats["participants"]:
            count = stats["speaker_counts"].get(name, 0)
            gender = stats["gender_map"].get(name, "Inconnu")
            
            # Suggestion par défaut
            if gender == "F":
                suggested_voice = "af_bella"
            elif gender == "M":
                suggested_voice = "am_adam"
            else:
                suggested_voice = "af_sky"
                
            chars.append([name, gender, count, suggested_voice])
            
        return chars, f"✅ Analyse terminée : {len(stats['participants'])} personnages trouvés. Modifiez la colonne 'ID Voix' si besoin."
    except Exception as e:
        return None, f"❌ Erreur d'analyse : {str(e)}"


def apply_voice_mapping(char_data):
    """Convertit le tableau de personnages en JSON pour le mapping."""
    if char_data is None or len(char_data) == 0:
        return "{}"
    
    mapping = {}
    for row in char_data:
        if len(row) >= 4:
            name = row[0]
            voice = row[3]
            mapping[name] = voice
            
    return json.dumps(mapping, indent=2, ensure_ascii=False)


def convert_text(
    text: str,
    voice: str,
    speed: float,
    apply_corrections: bool,
    hq: bool = False,
    mastering: bool = False,
    style: str = "storytelling",
    engine_type: str = "xtts",
    progress=gr.Progress()
):
    """Convertit un texte simple en audio."""
    if not check_model():
        return None, "❌ Modèle Kokoro non trouvé. Téléchargez-le d'abord."

    if not text.strip():
        return None, "❌ Veuillez entrer du texte."

    try:
        progress(0.1, desc="Initialisation...")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        if hq and HAS_HQ:
            progress(0.2, desc="Initialisation du pipeline HQ...")
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
            success = pipeline_synthesize_chapter(pipeline, text, output_path)
        else:
            # Traitement du texte
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
            progress(0.5, desc=f"Génération avec {engine_type}...")
            success = engine.synthesize(text, output_path)

        if success:
            progress(1.0, desc="Terminé!")
            return str(output_path), f"✅ Audio généré ({len(text)} caractères)"
        else:
            return None, "❌ Erreur lors de la génération"

    except Exception as e:
        return None, f"❌ Erreur: {str(e)}"


def pipeline_synthesize_chapter(pipeline, text, output_path):
    """Synthétise un chapitre complet avec le pipeline HQ."""
    try:
        import os
        
        # 1. Initialiser le générateur HQ
        generator = HQGenerator(config=pipeline.config)
        generator.pipeline = pipeline
        
        # 2. Processus (Analyse -> Segments)
        segments = pipeline.process_chapter(text)
        
        # 3. Récupérer le moteur
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
            
        # Synthèse
        audios = pipeline.synthesize_segments(segments, synth_fn)
        
        # 4. Concaténation
        full_audio = generator._concatenate_with_pauses(segments, audios)
        
        # 5. Sauvegarde
        sf.write(str(output_path), full_audio, 24000)
        
        # 6. Mastering
        if pipeline.config.enable_audio_enhancement:
            from audio_enhancer import AudioEnhancer
            enhancer = AudioEnhancer()
            if enhancer.is_available():
                mastered_path = output_path.with_name(f"{output_path.stem}_mastered.wav")
                success = enhancer.enhance_file(output_path, mastered_path)
                if success and mastered_path.exists():
                    os.replace(mastered_path, output_path)

        return True
    except Exception as e:
        print(f"Erreur synthèse HQ: {e}")
        return False


def convert_book(
    file,
    voice: str,
    speed: float,
    apply_corrections: bool,
    title: str,
    author: str,
    selected_formats: list[str],
    hq: bool = False,
    multivoice: bool = False,
    mastering: bool = False,
    style: str = "storytelling",
    voice_mapping_json: str = "{}",
    engine_type: str = "xtts",
    progress=gr.Progress()
):
    """Convertit un livre Markdown complet en audiobook et/ou ebooks."""
    if file is None:
        return None, "❌ Veuillez uploader un fichier Markdown."

    # Parse inputs
    generate_audio = any("Audio" in f for f in selected_formats)
    ebook_formats = []
    if "E-Book (PDF)" in selected_formats: ebook_formats.append("pdf")
    if "E-Book (EPUB)" in selected_formats: ebook_formats.append("epub")
    if "E-Book (HTML)" in selected_formats: ebook_formats.append("html")
    if "Texte (TXT)" in selected_formats: ebook_formats.append("txt")
    
    audio_format = "m4b" if "Audio (M4B)" in selected_formats else "mp3"

    if generate_audio and not check_model():
        return None, "❌ Modèle Kokoro non trouvé (requis pour l'audio)."

    try:
        # Parser le mapping si fourni
        try:
            custom_mapping = json.loads(voice_mapping_json) if voice_mapping_json else {}
        except:
            custom_mapping = {}

        # Créer un dossier temporaire
        temp_dir = Path(tempfile.mkdtemp())
        output_dir = temp_dir / "chapters"
        output_dir.mkdir()
        
        final_files = []
        status_messages = []

        progress(0.1, desc="Lecture du fichier...")

        # Conversion automatique si nécessaire
        input_path = Path(file.name)
        if input_path.suffix.lower() in ['.pdf', '.epub']:
            progress(0.15, desc="Conversion du fichier source...")
            converter = InputConverter()
            input_path = converter.convert_to_markdown(input_path)

        # Parser le livre
        chapters = parse_book(input_path)

        if not chapters:
            return None, "❌ Aucun chapitre trouvé dans le fichier."
            
        # --- Génération Audio ---
        if generate_audio:
            # Initialiser le pipeline HQ ou le moteur standard
            pipeline = None
            engine = None
            processor = None

            if hq and HAS_HQ:
                progress(0.2, desc="Initialisation du pipeline HQ...")
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
                progress(0.2, desc=f"Chargement du moteur {engine_type}...")
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
                progress_val = 0.2 + (0.5 * (i / len(chapters)))
                progress(progress_val, desc=f"Audio Chapitre {i+1}/{len(chapters)}")

                text = chapter.get_full_text()
                output_path = output_dir / f"{chapter.get_filename()}.wav"

                if pipeline:
                    success = pipeline_synthesize_chapter(pipeline, text, output_path)
                else:
                    if processor:
                        text = processor.process_to_text(text)
                    success = engine.synthesize(text, output_path)
                
                if not success:
                    print(f"Échec audio au chapitre {i+1}")

            progress(0.7, desc="Construction de l'audiobook...")

            # Construire l'audiobook
            metadata = AudiobookMetadata(
                title=title or "Audiobook",
                author=author or "Inconnu",
                narrator=voice if not multivoice else "Multi-Cast"
            )

            builder = AudiobookBuilder(metadata)
            builder.add_chapters_from_dir(output_dir, "*.wav")

            # Exporter
            base_name = title or Path(file.name).stem
            if audio_format == "m4b":
                audio_path = temp_dir / f"{base_name}.m4b"
                builder.build_m4b(audio_path)
            else:
                audio_path = temp_dir / f"{base_name}.mp3"
                builder.build_combined_mp3(audio_path)
                
            if audio_path.exists():
                final_files.append(audio_path)
                status_messages.append(f"Audio ({audio_format})")

        # --- Génération E-Book ---
        if ebook_formats:
            progress(0.8, desc="Génération des E-Books...")
            from book_exporter import BookExporter
            
            exporter = BookExporter(
                title=title or Path(file.name).stem,
                author=author or "Inconnu",
                language="fr"
            )
            
            # Ajouter les chapitres
            for chapter in chapters:
                exporter.add_chapter(chapter.title, chapter.content)
                
            # Exporter
            ebook_results = exporter.export_all(
                output_dir=temp_dir,
                base_name=title or Path(file.name).stem,
                formats=ebook_formats
            )
            
            for fmt, path in ebook_results.items():
                if path.exists():
                    final_files.append(path)
                    status_messages.append(f"E-Book ({fmt.upper()})")

        progress(1.0, desc="Finalisation...")

        # Retourner le résultat
        if not final_files:
            return None, "❌ Aucun fichier généré."
            
        if len(final_files) == 1:
            # Un seul fichier
            perm_output = Path(tempfile.gettempdir()) / final_files[0].name
            shutil.copy(final_files[0], perm_output)
            return str(perm_output), f"✅ Généré : {status_messages[0]}"
        else:
            # ZIP si plusieurs fichiers
            zip_name = f"{title or 'livre'}_complet"
            zip_path = shutil.make_archive(
                str(Path(tempfile.gettempdir()) / zip_name),
                'zip',
                root_dir=str(temp_dir),
                base_dir="." # Tout le contenu du temp_dir (attention, filtrer si besoin)
            )
            
            # On ne veut zipper que les fichiers finaux, pas le dossier 'chapters'
            # Réécrivons le zip proprement
            import zipfile
            final_zip_path = Path(tempfile.gettempdir()) / f"{zip_name}.zip"
            with zipfile.ZipFile(final_zip_path, 'w') as zipf:
                for f in final_files:
                    zipf.write(f, arcname=f.name)
            
            return str(final_zip_path), f"✅ Pack complet généré : {', '.join(status_messages)}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Erreur: {str(e)}"


def create_interface():
    """Crée l'interface Gradio."""

    # CSS personnalisé
    css = """
    .gradio-container { max-width: 1200px !important; }
    .title { text-align: center; margin-bottom: 20px; }
    """

    with gr.Blocks(css=css, title="AudioReader - Convertisseur Audiobook") as demo:

        gr.Markdown("""
        # 🎧 AudioReader
        ### Convertissez vos livres Markdown en audiobooks haute qualité

        Utilise **Kokoro-82M** - qualité proche d'ElevenLabs, 100% gratuit et local.
        """)

        # Status du modèle
        if check_model():
            gr.Markdown("✅ **Modèle Kokoro chargé**")
        else:
            gr.Markdown("""
            ⚠️ **Modèle non trouvé**

            Téléchargez-le avec:
            ```bash
            curl -L -o kokoro-v1.0.onnx "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
            curl -L -o voices-v1.0.bin "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
            ```
            """)

        with gr.Tabs():
            # Tab 1: Texte simple
            with gr.TabItem("📝 Texte"):
                gr.Markdown("Convertissez un texte en audio.")

                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="Texte à convertir",
                            placeholder="Entrez votre texte ici...",
                            lines=10
                        )

                    with gr.Column(scale=1):
                        engine_select_text = gr.Dropdown(
                            choices=["auto", "kokoro", "mms", "xtts", "edge"],
                            value="xtts",
                            label="Moteur TTS"
                        )

                        with gr.Row():
                            voice_select = gr.Dropdown(
                                choices=get_voice_choices(),
                                value="ff_siwis",
                                label="Voix",
                                scale=3
                            )
                            refresh_btn_text = gr.Button("🔄", scale=0)

                        refresh_btn_text.click(fn=refresh_voices, outputs=voice_select)

                        speed_slider = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Vitesse"
                        )

                        gr.Markdown("### Options HQ")
                        hq_check_text = gr.Checkbox(
                            value=False,
                            label="🚀 Mode HQ"
                        )
                        mastering_check_text = gr.Checkbox(
                            value=False,
                            label="🎙️ Mastering"
                        )
                        style_select_text = gr.Dropdown(
                            choices=["storytelling", "dramatic", "formal", "conversational"],
                            value="storytelling",
                            label="Style"
                        )

                        corrections_check = gr.Checkbox(
                            value=True,
                            label="Appliquer corrections prononciation"
                        )
                        convert_btn = gr.Button("🎙️ Convertir", variant="primary")

                with gr.Row():
                    audio_output = gr.Audio(label="Audio généré", type="filepath")
                    status_text = gr.Textbox(label="Status", interactive=False)

                convert_btn.click(
                    fn=convert_text,
                    inputs=[
                        text_input, voice_select, speed_slider, corrections_check,
                        hq_check_text, mastering_check_text, style_select_text,
                        engine_select_text
                    ],
                    outputs=[audio_output, status_text]
                )

            # Tab 2: Livre complet
            with gr.TabItem("📚 Livre"):
                gr.Markdown("Convertissez un livre Markdown complet en audiobook.")

                with gr.Row():
                    with gr.Column(scale=2):
                        file_input = gr.File(
                            label="Fichier source (Markdown, PDF, EPUB)",
                            file_types=[".md", ".txt", ".pdf", ".epub"]
                        )

                        with gr.Row():
                            title_input = gr.Textbox(
                                label="Titre",
                                placeholder="Titre de l'audiobook"
                            )
                            author_input = gr.Textbox(
                                label="Auteur",
                                placeholder="Nom de l'auteur"
                            )

                    with gr.Column(scale=1):
                        engine_select_book = gr.Dropdown(
                            choices=["auto", "kokoro", "mms", "xtts", "edge"],
                            value="xtts",
                            label="Moteur TTS"
                        )
                        with gr.Row():
                            voice_select_book = gr.Dropdown(
                                choices=get_voice_choices(),
                                value="ff_siwis",
                                label="Voix du narrateur",
                                scale=3
                            )
                            refresh_btn_book = gr.Button("🔄", scale=0)
                            
                        refresh_btn_book.click(fn=refresh_voices, outputs=voice_select_book)
                        speed_slider_book = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Vitesse"
                        )
                        
                        # Options HQ
                        gr.Markdown("### Options Haute Qualité")
                        hq_check = gr.Checkbox(
                            value=True,
                            label="🚀 Activer Mode HQ (v2.4)"
                        )
                        multivoice_check = gr.Checkbox(
                            value=True,
                            label="👥 Multi-voix (détection personnages)"
                        )
                        mastering_check = gr.Checkbox(
                            value=True,
                            label="🎙️ Mastering (ACX/Audible)"
                        )
                        style_select = gr.Dropdown(
                            choices=["storytelling", "dramatic", "formal", "conversational"],
                            value="storytelling",
                            label="Style de narration"
                        )
                        
                        corrections_check_book = gr.Checkbox(
                            value=True,
                            label="Corrections prononciation"
                        )
                        
                        gr.Markdown("### Formats de sortie")
                        formats_select = gr.CheckboxGroup(
                            choices=["Audio (M4B)", "Audio (MP3)", "E-Book (PDF)", "E-Book (EPUB)", "E-Book (HTML)", "Texte (TXT)"],
                            value=["Audio (M4B)"],
                            label="Sélectionnez les formats à générer"
                        )

                        gr.Markdown("### Mapping personnages (optionnel)")
                        voice_mapping_input = gr.Textbox(
                            label="JSON mapping voix",
                            placeholder='{"Personnage": "voix_id", ...}',
                            value="{}",
                            lines=2
                        )

                        convert_book_btn = gr.Button("📖 Créer le livre", variant="primary")

                with gr.Row():
                    book_output = gr.File(label="Audiobook généré")
                    book_status = gr.Textbox(label="Status", interactive=False)

                convert_book_btn.click(
                    fn=convert_book,
                    inputs=[
                        file_input, voice_select_book, speed_slider_book,
                        corrections_check_book, title_input, author_input, formats_select,
                        hq_check, multivoice_check, mastering_check, style_select,
                        voice_mapping_input, engine_select_book
                    ],
                    outputs=[book_output, book_status]
                )

            # Tab 3: Personnages (Nouveau)
            with gr.TabItem("👥 Personnages"):
                gr.Markdown("### Assistant d'attribution des voix")
                gr.Markdown("Analysez votre livre pour identifier les personnages et préparez l'attribution des voix.")
                
                with gr.Row():
                    char_file_input = gr.File(label="Fichier source (Markdown, PDF, EPUB)", file_types=[".md", ".pdf", ".epub"])
                    analyze_btn = gr.Button("🔍 Analyser les personnages", variant="primary")
                
                char_table = gr.Dataframe(
                    headers=["Nom", "Genre détecté", "Nombre de répliques", "ID Voix"],
                    label="Personnages détectés (Modifiez l'ID Voix ici)",
                    interactive=True
                )
                
                char_status = gr.Textbox(label="Résultat de l'analyse", interactive=False)
                
                apply_btn = gr.Button("💾 Appliquer ce mapping à l'audiobook", variant="secondary")
                
                gr.Markdown("### 👂 Pré-écoute des voix")
                with gr.Row():
                    with gr.Column(scale=3):
                        preview_voice_select = gr.Dropdown(
                            choices=[v[1] for v in get_voice_choices()],
                            label="Sélectionnez une voix à tester"
                        )
                    refresh_btn_char = gr.Button("🔄", scale=0)
                    preview_audio_btn = gr.Button("▶️ Écouter", scale=0)
                
                refresh_btn_char.click(
                    fn=lambda: gr.Dropdown(choices=[v[1] for v in get_voice_choices()]), 
                    outputs=preview_voice_select
                )
                
                preview_audio_player = gr.Audio(label="Sample", type="filepath")
                
                preview_audio_btn.click(
                    fn=preview_voice,
                    inputs=[preview_voice_select],
                    outputs=[preview_audio_player]
                )
                
                analyze_btn.click(
                    fn=analyze_characters,
                    inputs=[char_file_input],
                    outputs=[char_table, char_status]
                )
                
                apply_btn.click(
                    fn=apply_voice_mapping,
                    inputs=[char_table],
                    outputs=[voice_mapping_input]
                ).then(
                    lambda: gr.Info("Mapping appliqué ! Retournez dans l'onglet 'Livre' pour lancer la génération.")
                )

                with gr.Accordion("📋 Aide-mémoire des ID Voix", open=False):
                    gr.Markdown("""
                    | Langue | Genre | IDs Recommandés |
                    |--------|-------|-----------------|
                    | **Français** | ♀ | `ff_siwis`, `ff_sophie` |
                    | **Français** | ♂ | `fm_hugo`, `fm_florian` |
                    | **Anglais** | ♀ | `af_bella`, `af_sarah`, `af_sky`, `af_nicole` |
                    | **Anglais** | ♂ | `am_adam`, `am_michael`, `am_eric`, `bm_george` |
                    """)
                
                gr.Markdown("""
                > **Note :** Cette analyse aide à configurer le mode HQ. 
                > En mode HQ, le système tentera d'attribuer automatiquement des voix masculines/féminines 
                > cohérentes aux personnages identifiés.
                """)

            # Tab 4: Clonage de Voix
            with gr.TabItem("🎙️ Clonage de Voix"):
                gr.Markdown("### Créateur de Voix Personnalisée (XTTS v2)")
                gr.Markdown("Créez une nouvelle voix à partir d'un fichier vidéo ou audio. La voix sera ensuite utilisable dans tous les onglets.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        clone_file_input = gr.File(
                            label="Fichier source (Vidéo MP4/MKV ou Audio WAV/MP3)",
                            file_types=[".mp4", ".mkv", ".avi", ".mov", ".wav", ".mp3", ".m4a"]
                        )
                        
                        clone_name_input = gr.Textbox(
                            label="Nom de la voix",
                            placeholder="Ex: Geralt, Narrateur..."
                        )
                        
                        clone_lang_select = gr.Dropdown(
                            choices=["fr", "en", "es", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko"],
                            value="fr",
                            label="Langue de l'échantillon"
                        )
                        
                        with gr.Row():
                            start_time_input = gr.Number(value=0, label="Début (sec)", precision=1)
                            end_time_input = gr.Number(value=0, label="Fin (sec, 0=tout)", precision=1)
                        
                        extract_btn = gr.Button("extract & Créer la Voix", variant="primary")
                        
                    with gr.Column(scale=1):
                        clone_audio_output = gr.Audio(label="Audio extrait", type="filepath")
                        clone_status = gr.Textbox(label="Résultat", interactive=False)
                        
                extract_btn.click(
                    fn=extract_voice_from_file,
                    inputs=[clone_file_input, clone_name_input, start_time_input, end_time_input, clone_lang_select],
                    outputs=[clone_audio_output, clone_status]
                )
                
                gr.Markdown("""
                #### Conseils pour un bon clonage :
                - Utilisez un échantillon de **10 à 30 secondes**.
                - Assurez-vous qu'il n'y a **pas de musique de fond** ni de bruit.
                - Une seule personne doit parler.
                - Une meilleure qualité audio donne un meilleur résultat.
                """)

            # Tab 5: Podcast
            with gr.TabItem("📡 Diffusion"):
                gr.Markdown("### Serveur Podcast Local")
                gr.Markdown("Diffusez vos livres audio sur votre réseau local pour les écouter sur votre téléphone.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        start_server_btn = gr.Button("▶️ Démarrer le serveur", variant="primary")
                        refresh_feed_btn = gr.Button("🔄 Mettre à jour le flux RSS")
                        stop_server_btn = gr.Button("🛑 Arrêter le serveur", variant="stop")
                        
                        gr.Markdown("""
                        #### Comment faire ?
                        1. Cliquez sur **Démarrer**.
                        2. Assurez-vous que votre téléphone est sur le **même Wi-Fi**.
                        3. Scannez le **QR Code** avec votre appli de podcast (ou copiez l'URL).
                        4. Vos fichiers doivent être dans le dossier `output/` du projet.
                        """)
                        
                    with gr.Column(scale=1):
                        podcast_status = gr.Textbox(label="État du serveur", interactive=False)
                        podcast_url = gr.Textbox(label="URL du Flux RSS", interactive=False, show_copy_button=True)
                        podcast_qr = gr.HTML(label="QR Code")
                
                start_server_btn.click(
                    fn=lambda: manage_podcast_server("start"),
                    outputs=[podcast_status, podcast_url, podcast_qr]
                )
                
                stop_server_btn.click(
                    fn=lambda: manage_podcast_server("stop"),
                    outputs=[podcast_status, podcast_url, podcast_qr]
                )
                
                refresh_feed_btn.click(
                    fn=lambda: manage_podcast_server("refresh"),
                    outputs=[podcast_status, podcast_url, podcast_qr]
                )

            # Tab 6: À propos
            with gr.TabItem("ℹ️ À propos"):
                gr.Markdown("""
                ## AudioReader

                Convertisseur de livres Markdown en audiobooks haute qualité.

                ### Technologies utilisées

                - **Kokoro-82M**: Modèle TTS open-source avec 82 millions de paramètres
                - **ONNX Runtime**: Inférence optimisée
                - **FFmpeg**: Traitement audio

                ### Qualité

                Kokoro-82M offre une qualité proche d'ElevenLabs:
                - Voix naturelles et expressives
                - Support multilingue (français, anglais, japonais, chinois...)
                - Performance ~5x temps réel sur CPU

                ### Plateformes de distribution

                Les audiobooks générés peuvent être publiés sur:
                - ✅ Google Play Books
                - ✅ Findaway Voices / Spotify
                - ✅ Kobo
                - ❌ Audible/ACX (n'accepte pas les voix AI)

                ### Crédits

                - Kokoro TTS: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
                - ONNX Runtime: [thewh1teagle/kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)
                """)

        gr.Markdown("""
        ---
        *AudioReader - Généré avec Kokoro TTS | [Sources et documentation](https://github.com/)*
        """)

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
