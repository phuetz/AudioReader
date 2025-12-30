"""
Interface web Gradio pour gerer les corrections de prononciation.

Permet d'ajouter, modifier et tester les corrections phonetiques
de maniere interactive.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False


class CorrectionManager:
    """
    Gestionnaire de corrections de prononciation.
    """

    def __init__(self, corrections_file: Optional[Path] = None):
        self.corrections_file = corrections_file or Path("corrections.json")
        self.corrections: Dict[str, str] = {}
        self.load()

    def load(self) -> bool:
        """Charge les corrections depuis le fichier."""
        if self.corrections_file.exists():
            try:
                with open(self.corrections_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Support pour le format simple ou le format avec metadata
                    if isinstance(data, dict):
                        if 'corrections' in data:
                            self.corrections = data['corrections']
                        else:
                            self.corrections = data
                    return True
            except Exception as e:
                print(f"Erreur chargement corrections: {e}")
        return False

    def save(self) -> bool:
        """Sauvegarde les corrections dans le fichier."""
        try:
            self.corrections_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.corrections_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'version': '1.0',
                    'description': 'Corrections de prononciation AudioReader',
                    'corrections': self.corrections
                }, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Erreur sauvegarde corrections: {e}")
            return False

    def add(self, original: str, replacement: str) -> bool:
        """Ajoute une correction."""
        if original and replacement:
            self.corrections[original] = replacement
            return self.save()
        return False

    def remove(self, original: str) -> bool:
        """Supprime une correction."""
        if original in self.corrections:
            del self.corrections[original]
            return self.save()
        return False

    def update(self, original: str, new_original: str, replacement: str) -> bool:
        """Met a jour une correction."""
        if original in self.corrections:
            del self.corrections[original]
        if new_original and replacement:
            self.corrections[new_original] = replacement
            return self.save()
        return False

    def apply(self, text: str) -> str:
        """Applique les corrections au texte."""
        result = text
        for original, replacement in self.corrections.items():
            # Utiliser des frontieres de mots pour les patterns alphanumeriques
            if original[0].isalnum() and original[-1].isalnum():
                pattern = re.compile(rf'\b{re.escape(original)}\b', re.IGNORECASE)
            else:
                pattern = re.compile(re.escape(original))
            result = pattern.sub(replacement, result)
        return result

    def get_all(self) -> List[Tuple[str, str]]:
        """Retourne toutes les corrections."""
        return list(self.corrections.items())

    def search(self, query: str) -> List[Tuple[str, str]]:
        """Recherche dans les corrections."""
        query_lower = query.lower()
        return [
            (k, v) for k, v in self.corrections.items()
            if query_lower in k.lower() or query_lower in v.lower()
        ]


def create_corrections_ui(
    corrections_file: Optional[Path] = None,
    tts_engine=None
) -> "gr.Blocks":
    """
    Cree l'interface Gradio pour les corrections.

    Args:
        corrections_file: Fichier de corrections
        tts_engine: Moteur TTS pour les tests (optionnel)

    Returns:
        Application Gradio
    """
    if not HAS_GRADIO:
        raise ImportError("Gradio n'est pas installe. pip install gradio")

    manager = CorrectionManager(corrections_file)

    def refresh_table():
        """Rafraichit le tableau des corrections."""
        corrections = manager.get_all()
        return [[k, v] for k, v in corrections]

    def add_correction(original, replacement):
        """Ajoute une nouvelle correction."""
        if not original or not replacement:
            return "Veuillez remplir les deux champs", refresh_table()

        if manager.add(original.strip(), replacement.strip()):
            return f"Correction ajoutee: {original} -> {replacement}", refresh_table()
        else:
            return "Erreur lors de l'ajout", refresh_table()

    def delete_correction(original):
        """Supprime une correction."""
        if not original:
            return "Selectionnez une correction a supprimer", refresh_table()

        if manager.remove(original.strip()):
            return f"Correction supprimee: {original}", refresh_table()
        else:
            return "Erreur lors de la suppression", refresh_table()

    def search_corrections(query):
        """Recherche dans les corrections."""
        if not query:
            return refresh_table()
        results = manager.search(query)
        return [[k, v] for k, v in results]

    def preview_correction(text):
        """Previsualise l'application des corrections."""
        if not text:
            return ""
        return manager.apply(text)

    def test_audio(text):
        """Genere un audio de test."""
        if not text:
            return None, "Entrez du texte a tester"

        if tts_engine is None:
            return None, "Moteur TTS non configure"

        corrected = manager.apply(text)
        output_path = Path("output/correction_test.wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if hasattr(tts_engine, 'synthesize_chapter'):
                success = tts_engine.synthesize_chapter(corrected, output_path)
            elif hasattr(tts_engine, 'synthesize'):
                success = tts_engine.synthesize(corrected, output_path)
            else:
                return None, "Moteur TTS incompatible"

            if success and output_path.exists():
                return str(output_path), f"Audio genere pour: {corrected[:50]}..."
            else:
                return None, "Echec de la generation audio"
        except Exception as e:
            return None, f"Erreur: {e}"

    def import_json(file):
        """Importe des corrections depuis un fichier JSON."""
        if file is None:
            return "Selectionnez un fichier", refresh_table()

        try:
            with open(file.name, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, dict):
                if 'corrections' in data:
                    new_corrections = data['corrections']
                else:
                    new_corrections = data

                count = 0
                for k, v in new_corrections.items():
                    if manager.add(k, v):
                        count += 1

                return f"{count} corrections importees", refresh_table()
            else:
                return "Format JSON invalide", refresh_table()

        except Exception as e:
            return f"Erreur d'import: {e}", refresh_table()

    def export_json():
        """Exporte les corrections en JSON."""
        export_path = Path("output/corrections_export.json")
        export_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'version': '1.0',
                    'corrections': manager.corrections
                }, f, ensure_ascii=False, indent=2)
            return str(export_path)
        except Exception as e:
            return None

    # Interface Gradio
    with gr.Blocks(title="AudioReader - Corrections de Prononciation") as app:
        gr.Markdown("# Gestionnaire de Corrections de Prononciation")
        gr.Markdown("Ajoutez, modifiez et testez les corrections phonetiques pour ameliorer la synthese vocale.")

        with gr.Tabs():
            # Onglet principal: Gestion des corrections
            with gr.Tab("Corrections"):
                with gr.Row():
                    with gr.Column(scale=2):
                        # Tableau des corrections
                        corrections_table = gr.Dataframe(
                            headers=["Original", "Remplacement"],
                            value=refresh_table(),
                            interactive=False,
                            label="Corrections actuelles"
                        )

                        # Recherche
                        search_input = gr.Textbox(
                            label="Rechercher",
                            placeholder="Tapez pour filtrer..."
                        )
                        search_input.change(
                            search_corrections,
                            inputs=[search_input],
                            outputs=[corrections_table]
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### Ajouter une correction")
                        original_input = gr.Textbox(
                            label="Mot/Expression original",
                            placeholder="ex: API"
                        )
                        replacement_input = gr.Textbox(
                            label="Prononciation",
                            placeholder="ex: A P I"
                        )
                        add_btn = gr.Button("Ajouter", variant="primary")

                        gr.Markdown("### Supprimer")
                        delete_input = gr.Textbox(
                            label="Mot a supprimer",
                            placeholder="Copiez le mot original ici"
                        )
                        delete_btn = gr.Button("Supprimer", variant="stop")

                        status_msg = gr.Textbox(label="Statut", interactive=False)

                # Actions
                add_btn.click(
                    add_correction,
                    inputs=[original_input, replacement_input],
                    outputs=[status_msg, corrections_table]
                )

                delete_btn.click(
                    delete_correction,
                    inputs=[delete_input],
                    outputs=[status_msg, corrections_table]
                )

            # Onglet: Test des corrections
            with gr.Tab("Tester"):
                gr.Markdown("### Previsualiser et tester les corrections")

                test_input = gr.Textbox(
                    label="Texte a tester",
                    placeholder="Entrez du texte pour voir les corrections appliquees...",
                    lines=4
                )

                preview_btn = gr.Button("Previsualiser les corrections")

                preview_output = gr.Textbox(
                    label="Texte corrige",
                    interactive=False,
                    lines=4
                )

                preview_btn.click(
                    preview_correction,
                    inputs=[test_input],
                    outputs=[preview_output]
                )

                gr.Markdown("### Test audio")
                audio_btn = gr.Button("Generer l'audio")
                audio_output = gr.Audio(label="Audio de test")
                audio_status = gr.Textbox(label="Statut", interactive=False)

                audio_btn.click(
                    test_audio,
                    inputs=[test_input],
                    outputs=[audio_output, audio_status]
                )

            # Onglet: Import/Export
            with gr.Tab("Import/Export"):
                gr.Markdown("### Importer des corrections")
                import_file = gr.File(
                    label="Fichier JSON a importer",
                    file_types=[".json"]
                )
                import_btn = gr.Button("Importer")
                import_status = gr.Textbox(label="Statut", interactive=False)

                import_btn.click(
                    import_json,
                    inputs=[import_file],
                    outputs=[import_status, corrections_table]
                )

                gr.Markdown("### Exporter les corrections")
                export_btn = gr.Button("Exporter en JSON")
                export_file = gr.File(label="Fichier exporte")

                export_btn.click(
                    export_json,
                    outputs=[export_file]
                )

            # Onglet: Aide
            with gr.Tab("Aide"):
                gr.Markdown("""
                ## Guide des corrections de prononciation

                ### Pourquoi utiliser des corrections ?

                Les moteurs TTS peuvent mal prononcer certains mots:
                - **Acronymes**: API, URL, JSON
                - **Noms propres**: Noms de marques, personnages
                - **Mots etrangers**: Mots anglais dans un texte francais
                - **Termes techniques**: Jargon specifique

                ### Comment ajouter une correction ?

                1. **Original**: Le mot tel qu'il apparait dans le texte
                2. **Remplacement**: Comment il doit etre prononce

                ### Exemples de corrections

                | Original | Remplacement | Explication |
                |----------|--------------|-------------|
                | API | A P I | Epeler l'acronyme |
                | JSON | jason | Prononciation phonetique |
                | iPhone | aïe faune | Prononciation francisee |
                | km/h | kilometres heure | Expansion de l'unite |
                | 1er | premier | Nombre ordinal |

                ### Conseils

                - Utilisez des espaces pour epeler les lettres: `API` -> `A P I`
                - Pour les mots anglais, ecrivez la prononciation francaise phonetiquement
                - Les corrections sont sensibles a la casse par defaut
                - Testez toujours vos corrections avec l'onglet "Tester"

                ### Format du fichier JSON

                ```json
                {
                    "corrections": {
                        "API": "A P I",
                        "JSON": "jason",
                        "km/h": "kilometres heure"
                    }
                }
                ```
                """)

        # Refresh initial
        app.load(refresh_table, outputs=[corrections_table])

    return app


def launch_corrections_ui(
    corrections_file: Optional[str] = None,
    share: bool = False,
    port: int = 7861
):
    """
    Lance l'interface de corrections.

    Args:
        corrections_file: Chemin du fichier de corrections
        share: Creer un lien public
        port: Port du serveur
    """
    if not HAS_GRADIO:
        print("Gradio n'est pas installe. pip install gradio")
        return

    file_path = Path(corrections_file) if corrections_file else None
    app = create_corrections_ui(file_path)
    app.launch(share=share, server_port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Interface de corrections de prononciation")
    parser.add_argument("-f", "--file", help="Fichier de corrections JSON")
    parser.add_argument("-p", "--port", type=int, default=7861, help="Port du serveur")
    parser.add_argument("--share", action="store_true", help="Creer un lien public")

    args = parser.parse_args()
    launch_corrections_ui(args.file, args.share, args.port)
