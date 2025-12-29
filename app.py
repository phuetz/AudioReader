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

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
from markdown_parser import parse_book
from tts_kokoro_engine import KokoroEngine, KOKORO_VOICES
from text_processor import TextProcessor
from audiobook_builder import AudiobookBuilder, AudiobookMetadata


# Configuration
MODEL_PATH = Path("kokoro-v1.0.onnx")
VOICES_PATH = Path("voices-v1.0.bin")


def check_model():
    """Vérifie si le modèle est disponible."""
    return MODEL_PATH.exists() and VOICES_PATH.exists()


def get_voice_choices():
    """Retourne la liste des voix pour le dropdown."""
    choices = []
    for vid, info in KOKORO_VOICES.items():
        gender = "♀" if info["gender"] == "F" else "♂"
        label = f"{info['name']} {gender} ({info['desc']})"
        choices.append((label, vid))
    return choices


def convert_text(
    text: str,
    voice: str,
    speed: float,
    apply_corrections: bool,
    progress=gr.Progress()
):
    """Convertit un texte simple en audio."""
    if not check_model():
        return None, "❌ Modèle Kokoro non trouvé. Téléchargez-le d'abord."

    if not text.strip():
        return None, "❌ Veuillez entrer du texte."

    try:
        progress(0.1, desc="Initialisation...")

        # Traitement du texte
        if apply_corrections:
            processor = TextProcessor(lang="fr", engine="kokoro")
            text = processor.process_to_text(text)

        progress(0.3, desc="Chargement du modèle...")

        # Initialiser le moteur
        engine = KokoroEngine(
            model_path=str(MODEL_PATH),
            voices_path=str(VOICES_PATH),
            voice=voice,
            speed=speed
        )

        progress(0.5, desc="Génération audio...")

        # Générer l'audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = Path(f.name)

        success = engine.synthesize(text, output_path)

        if success:
            progress(1.0, desc="Terminé!")
            return str(output_path), f"✅ Audio généré ({len(text)} caractères)"
        else:
            return None, "❌ Erreur lors de la génération"

    except Exception as e:
        return None, f"❌ Erreur: {str(e)}"


def convert_book(
    file,
    voice: str,
    speed: float,
    apply_corrections: bool,
    title: str,
    author: str,
    output_format: str,
    progress=gr.Progress()
):
    """Convertit un livre Markdown complet en audiobook."""
    if not check_model():
        return None, "❌ Modèle Kokoro non trouvé."

    if file is None:
        return None, "❌ Veuillez uploader un fichier Markdown."

    try:
        # Créer un dossier temporaire
        temp_dir = Path(tempfile.mkdtemp())
        output_dir = temp_dir / "chapters"
        output_dir.mkdir()

        progress(0.1, desc="Lecture du fichier...")

        # Parser le livre
        chapters = parse_book(file.name)

        if not chapters:
            return None, "❌ Aucun chapitre trouvé dans le fichier."

        progress(0.2, desc=f"Chargement du modèle ({len(chapters)} chapitres)...")

        # Initialiser
        engine = KokoroEngine(
            model_path=str(MODEL_PATH),
            voices_path=str(VOICES_PATH),
            voice=voice,
            speed=speed
        )

        processor = TextProcessor(lang="fr", engine="kokoro") if apply_corrections else None

        # Convertir chaque chapitre
        for i, chapter in enumerate(chapters):
            progress_val = 0.2 + (0.6 * (i / len(chapters)))
            progress(progress_val, desc=f"Chapitre {i+1}/{len(chapters)}: {chapter.title}")

            text = chapter.get_full_text()

            if processor:
                text = processor.process_to_text(text)

            output_path = output_dir / f"{chapter.get_filename()}.wav"
            engine.synthesize(text, output_path)

        progress(0.85, desc="Construction de l'audiobook...")

        # Construire l'audiobook
        metadata = AudiobookMetadata(
            title=title or "Audiobook",
            author=author or "Inconnu",
            narrator="Kokoro TTS"
        )

        builder = AudiobookBuilder(metadata)
        builder.add_chapters_from_dir(output_dir, "*.wav")

        # Exporter
        if output_format == "m4b":
            final_output = temp_dir / f"{title or 'audiobook'}.m4b"
            builder.build_m4b(final_output)
        else:
            final_output = temp_dir / f"{title or 'audiobook'}.mp3"
            builder.build_combined_mp3(final_output)

        progress(1.0, desc="Terminé!")

        if final_output.exists():
            # Copier vers un fichier permanent
            perm_output = Path(tempfile.gettempdir()) / final_output.name
            shutil.copy(final_output, perm_output)

            duration = builder.get_total_duration()
            return str(perm_output), f"✅ Audiobook créé: {len(chapters)} chapitres, {duration/60:.1f} minutes"
        else:
            return None, "❌ Erreur lors de la création"

    except Exception as e:
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
                        voice_select = gr.Dropdown(
                            choices=get_voice_choices(),
                            value="ff_siwis",
                            label="Voix"
                        )
                        speed_slider = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Vitesse"
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
                    inputs=[text_input, voice_select, speed_slider, corrections_check],
                    outputs=[audio_output, status_text]
                )

            # Tab 2: Livre complet
            with gr.TabItem("📚 Livre"):
                gr.Markdown("Convertissez un livre Markdown complet en audiobook.")

                with gr.Row():
                    with gr.Column(scale=2):
                        file_input = gr.File(
                            label="Fichier Markdown (.md)",
                            file_types=[".md", ".txt"]
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
                        voice_select_book = gr.Dropdown(
                            choices=get_voice_choices(),
                            value="ff_siwis",
                            label="Voix"
                        )
                        speed_slider_book = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Vitesse"
                        )
                        corrections_check_book = gr.Checkbox(
                            value=True,
                            label="Corrections prononciation"
                        )
                        format_select = gr.Radio(
                            choices=["m4b", "mp3"],
                            value="m4b",
                            label="Format de sortie"
                        )
                        convert_book_btn = gr.Button("📖 Créer l'audiobook", variant="primary")

                with gr.Row():
                    book_output = gr.File(label="Audiobook généré")
                    book_status = gr.Textbox(label="Status", interactive=False)

                convert_book_btn.click(
                    fn=convert_book,
                    inputs=[
                        file_input, voice_select_book, speed_slider_book,
                        corrections_check_book, title_input, author_input, format_select
                    ],
                    outputs=[book_output, book_status]
                )

            # Tab 3: À propos
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
