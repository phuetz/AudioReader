#!/usr/bin/env python3
"""
AudioReader Web Interface - Interface Gradio pour la génération d'audiobooks.

Lance avec: python web_interface.py
Puis ouvre: http://localhost:7860
"""

import gradio as gr
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Imports locaux
try:
    from tts_kokoro_engine import KokoroEngine, KOKORO_VOICES
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    KOKORO_VOICES = {}

from french_preprocessor import FrenchTextPreprocessor


def get_voice_choices():
    """Retourne les choix de voix pour le dropdown."""
    choices = []
    for vid, info in KOKORO_VOICES.items():
        gender = "♀" if info["gender"] == "F" else "♂"
        label = f"{info['name']} {gender} ({info['lang']}) - {info['desc']}"
        choices.append((label, vid))
    return choices


def text_to_speech(
    text: str,
    voice: str,
    speed: float,
    use_preprocessor: bool,
    progress=gr.Progress()
) -> str:
    """Convertit le texte en audio."""
    if not text.strip():
        raise gr.Error("Veuillez entrer du texte à convertir.")

    if not KOKORO_AVAILABLE:
        raise gr.Error("Kokoro n'est pas disponible. Vérifiez l'installation.")

    progress(0.1, desc="Initialisation...")

    # Prétraitement
    if use_preprocessor and voice.startswith("ff_"):
        preprocessor = FrenchTextPreprocessor()
        text = preprocessor.process(text)
        progress(0.2, desc="Texte prétraité...")

    # Initialiser Kokoro
    progress(0.3, desc="Chargement du modèle...")
    engine = KokoroEngine()

    # Déterminer la langue
    voice_info = KOKORO_VOICES.get(voice, {})
    lang = voice_info.get("lang", "en-us")

    # Générer l'audio
    progress(0.5, desc="Génération audio...")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        output_path = f.name

    success = engine.synthesize(
        text=text,
        output_path=output_path,
        voice=voice,
        speed=speed,
        lang=lang
    )

    if not success:
        raise gr.Error("Erreur lors de la génération audio.")

    progress(1.0, desc="Terminé!")
    return output_path


def convert_file(
    file,
    voice: str,
    speed: float,
    use_preprocessor: bool,
    create_m4b: bool,
    author: str,
    progress=gr.Progress()
) -> tuple:
    """Convertit un fichier Markdown en audiobook."""
    if file is None:
        raise gr.Error("Veuillez sélectionner un fichier.")

    from markdown_parser import parse_book
    from audiobook_packager import package_from_directory, AudiobookMetadata

    progress(0.1, desc="Lecture du fichier...")

    # Parser le livre (passer le chemin, pas le contenu)
    chapters = parse_book(file.name, header_level=1)

    if not chapters:
        raise gr.Error("Aucun chapitre trouvé dans le fichier.")

    progress(0.2, desc=f"Trouvé {len(chapters)} chapitre(s)...")

    # Créer un répertoire temporaire
    output_dir = tempfile.mkdtemp(prefix="audiobook_")

    # Initialiser
    engine = KokoroEngine()
    preprocessor = FrenchTextPreprocessor() if use_preprocessor else None

    voice_info = KOKORO_VOICES.get(voice, {})
    lang = voice_info.get("lang", "en-us")

    audio_files = []

    # Générer chaque chapitre
    for i, chapter in enumerate(chapters):
        progress((0.2 + 0.6 * i / len(chapters)), desc=f"Chapitre {i+1}/{len(chapters)}...")

        title = chapter.title
        text = chapter.get_full_text()

        if preprocessor and voice_info.get("lang", "").startswith("fr"):
            text = preprocessor.process(text)

        # Nom de fichier safe
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
        output_path = os.path.join(output_dir, f"{i+1:02d}_{safe_title}.wav")

        engine.synthesize(
            text=text,
            output_path=output_path,
            voice=voice,
            speed=speed,
            lang=lang
        )

        audio_files.append(output_path)

    progress(0.85, desc="Finalisation...")

    # Créer M4B si demandé
    m4b_path = None
    if create_m4b and audio_files:
        book_title = Path(file.name).stem.replace("_", " ").title()
        metadata = AudiobookMetadata(
            title=book_title,
            author=author or "Unknown",
            narrator=voice_info.get("name", "AI Voice")
        )

        m4b_path = os.path.join(output_dir, f"{Path(file.name).stem}.m4b")
        package_from_directory(output_dir, m4b_path, metadata)

    progress(1.0, desc="Terminé!")

    # Retourner le premier fichier audio et le M4B
    first_audio = audio_files[0] if audio_files else None
    return first_audio, m4b_path


# Interface Gradio
def create_interface():
    """Crée l'interface Gradio."""

    voice_choices = get_voice_choices()
    default_voice = "ff_siwis" if "ff_siwis" in KOKORO_VOICES else (
        list(KOKORO_VOICES.keys())[0] if KOKORO_VOICES else None
    )

    with gr.Blocks(
        title="AudioReader - Générateur d'Audiobooks",
        theme=gr.themes.Soft()
    ) as demo:

        gr.Markdown("""
        # 🎧 AudioReader
        ### Générateur d'audiobooks haute qualité avec Kokoro-82M

        Convertissez du texte ou des fichiers Markdown en audiobooks professionnels.
        """)

        with gr.Tabs():
            # Onglet 1: Texte simple
            with gr.TabItem("📝 Texte"):
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="Texte à convertir",
                            placeholder="Entrez votre texte ici...",
                            lines=10
                        )

                    with gr.Column(scale=1):
                        voice_select = gr.Dropdown(
                            choices=voice_choices,
                            value=default_voice,
                            label="Voix"
                        )
                        speed_slider = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Vitesse"
                        )
                        preprocess_check = gr.Checkbox(
                            value=True,
                            label="Préprocesseur français"
                        )
                        generate_btn = gr.Button(
                            "🎙️ Générer",
                            variant="primary"
                        )

                audio_output = gr.Audio(
                    label="Audio généré",
                    type="filepath"
                )

                generate_btn.click(
                    fn=text_to_speech,
                    inputs=[text_input, voice_select, speed_slider, preprocess_check],
                    outputs=audio_output
                )

            # Onglet 2: Fichier Markdown
            with gr.TabItem("📚 Livre"):
                with gr.Row():
                    with gr.Column(scale=2):
                        file_input = gr.File(
                            label="Fichier Markdown",
                            file_types=[".md", ".txt"]
                        )

                    with gr.Column(scale=1):
                        voice_select2 = gr.Dropdown(
                            choices=voice_choices,
                            value=default_voice,
                            label="Voix"
                        )
                        speed_slider2 = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Vitesse"
                        )
                        preprocess_check2 = gr.Checkbox(
                            value=True,
                            label="Préprocesseur français"
                        )
                        m4b_check = gr.Checkbox(
                            value=True,
                            label="Créer M4B avec chapitres"
                        )
                        author_input = gr.Textbox(
                            label="Auteur",
                            placeholder="Nom de l'auteur"
                        )
                        convert_btn = gr.Button(
                            "📖 Convertir",
                            variant="primary"
                        )

                with gr.Row():
                    preview_audio = gr.Audio(
                        label="Aperçu (premier chapitre)",
                        type="filepath"
                    )
                    m4b_output = gr.File(
                        label="Audiobook M4B"
                    )

                convert_btn.click(
                    fn=convert_file,
                    inputs=[
                        file_input, voice_select2, speed_slider2,
                        preprocess_check2, m4b_check, author_input
                    ],
                    outputs=[preview_audio, m4b_output]
                )

            # Onglet 3: À propos
            with gr.TabItem("ℹ️ À propos"):
                gr.Markdown("""
                ## AudioReader

                **AudioReader** est un générateur d'audiobooks open-source utilisant
                le modèle Kokoro-82M.

                ### Fonctionnalités

                - 🎙️ **Voix haute qualité** - 82M paramètres, proche d'ElevenLabs
                - 🇫🇷 **Support français** - Voix Siwis avec préprocesseur
                - 📚 **Chapitrage** - Export M4B avec navigation
                - ⚡ **Rapide** - ~5x temps réel sur CPU

                ### Voix disponibles

                | Langue | Voix | Genre |
                |--------|------|-------|
                | Français | Siwis | ♀ |
                | Anglais | Heart, Bella, Sarah... | ♀ |
                | Anglais | Michael, Fenrir... | ♂ |

                ### Crédits

                - Modèle: [Kokoro-82M](https://github.com/hexgrad/kokoro)
                - Interface: Gradio
                - Développé par: Patrice avec Claude
                """)

        gr.Markdown("""
        ---
        *AudioReader - Généré avec ❤️ et IA*
        """)

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
