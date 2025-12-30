"""
Export de livres en plusieurs formats: PDF, EPUB, HTML, TXT.

Usage:
    from src.book_exporter import BookExporter

    exporter = BookExporter(
        title="Les Conquérants du Pognon",
        author="Nom Auteur",
        language="fr"
    )

    # Ajouter des chapitres
    exporter.add_chapter("Chapitre 1: Le Gamin", chapter1_text)
    exporter.add_chapter("Chapitre 2: L'Ascension", chapter2_text)

    # Exporter
    exporter.export_pdf("livre.pdf")
    exporter.export_epub("livre.epub")
    exporter.export_html("livre.html")
    exporter.export_txt("livre.txt")
"""
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import html


@dataclass
class Chapter:
    """Un chapitre du livre."""
    title: str
    content: str
    index: int


@dataclass
class BookMetadata:
    """Métadonnées du livre."""
    title: str
    author: str = "Auteur Inconnu"
    language: str = "fr"
    publisher: str = ""
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    description: str = ""
    cover_image: Optional[Path] = None
    isbn: str = ""


class BookExporter:
    """
    Exporte un livre en plusieurs formats.

    Formats supportés:
    - PDF: Document portable avec mise en page
    - EPUB: Format ebook standard (Kindle, Kobo, etc.)
    - HTML: Page web unique
    - TXT: Texte brut
    """

    def __init__(
        self,
        title: str,
        author: str = "Auteur Inconnu",
        language: str = "fr",
        **metadata
    ):
        self.metadata = BookMetadata(
            title=title,
            author=author,
            language=language,
            **metadata
        )
        self.chapters: List[Chapter] = []
        self._chapter_index = 0

    def add_chapter(self, title: str, content: str) -> None:
        """Ajoute un chapitre au livre."""
        self.chapters.append(Chapter(
            title=title,
            content=content,
            index=self._chapter_index
        ))
        self._chapter_index += 1

    def add_chapter_from_markdown(self, markdown_path: Path) -> None:
        """Ajoute un chapitre depuis un fichier Markdown."""
        text = Path(markdown_path).read_text(encoding='utf-8')

        # Extraire le titre du premier header
        title_match = re.search(r'^#\s+(.+)$', text, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()
            # Retirer le titre du contenu
            text = text[title_match.end():].strip()
        else:
            title = f"Chapitre {self._chapter_index + 1}"

        # Nettoyer le Markdown
        content = self._clean_markdown(text)
        self.add_chapter(title, content)

    def _clean_markdown(self, text: str) -> str:
        """Nettoie le Markdown pour l'export."""
        # Retirer les headers markdown (on garde le texte)
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)

        # Retirer les blockquotes (épigraphes)
        text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)

        # Retirer les séparateurs
        text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\*\*\*+$', '', text, flags=re.MULTILINE)

        # Nettoyer les lignes vides multiples
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _to_latin1(self, text: str) -> str:
        """Convertit le texte en ASCII étendu compatible avec Helvetica PDF."""
        import unicodedata

        replacements = {
            '—': '-', '–': '-', '―': '-',
            '«': '"', '»': '"',
            '"': '"', '"': '"',
            ''': "'", ''': "'",
            '…': '...',
            'œ': 'oe', 'Œ': 'OE',
            'æ': 'ae', 'Æ': 'AE',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        # Normaliser les caractères accentués en ASCII
        # NFD décompose les caractères accentués, puis on filtre les accents
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')

        # Remplacer tout caractère non-ASCII restant
        text = text.encode('ascii', errors='replace').decode('ascii')

        return text

    def _markdown_to_html(self, text: str) -> str:
        """Convertit le Markdown basique en HTML."""
        # Échapper le HTML existant
        text = html.escape(text)

        # Bold
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)

        # Italic
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)

        # Guillemets français
        text = text.replace('«', '&laquo;').replace('»', '&raquo;')

        # Paragraphes
        paragraphs = text.split('\n\n')
        text = '\n'.join(f'<p>{p.strip()}</p>' for p in paragraphs if p.strip())

        # Tirets de dialogue en début de paragraphe
        text = re.sub(r'<p>\s*[-–—]\s*', '<p class="dialogue">— ', text)

        return text

    # =========================================================================
    # Export PDF
    # =========================================================================

    def export_pdf(
        self,
        output_path: Path,
        font_size: int = 12,
        page_size: str = "A4",
        margins: tuple = (20, 20, 20, 20)  # left, top, right, bottom
    ) -> bool:
        """
        Exporte le livre en PDF.

        Args:
            output_path: Chemin du fichier PDF
            font_size: Taille de police (défaut: 12)
            page_size: Format de page (A4, Letter, A5)
            margins: Marges en mm (gauche, haut, droite, bas)

        Returns:
            True si l'export a réussi
        """
        try:
            from fpdf import FPDF
        except ImportError:
            print("⚠️ fpdf2 non installé. Installation: pip install fpdf2")
            return False

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Créer le PDF avec marges larges pour éviter les débordements
        pdf = FPDF(orientation='P', unit='mm', format=page_size)
        pdf.set_auto_page_break(auto=True, margin=25)
        pdf.set_margins(25, 20, 25)

        # Utiliser Helvetica (compatible Latin-1) car DejaVu cause des problèmes
        main_font = 'Helvetica'

        # Référence locale pour la conversion Latin-1
        to_latin1 = self._to_latin1

        # Page de titre
        pdf.add_page()
        pdf.set_font(main_font, 'B', 24)
        pdf.ln(60)

        # Gérer les titres longs
        title = to_latin1(self.metadata.title)
        if len(title) > 40:
            # Découper le titre sur plusieurs lignes
            import textwrap
            title_lines = textwrap.wrap(title, width=35)
            for line in title_lines:
                pdf.multi_cell(0, 12, line, align='C')
        else:
            pdf.multi_cell(0, 12, title, align='C')

        pdf.ln(20)
        pdf.set_font(main_font, 'I', 16)
        pdf.multi_cell(0, 10, to_latin1(self.metadata.author), align='C')

        if self.metadata.date:
            pdf.ln(40)
            pdf.set_font(main_font, '', 10)
            pdf.multi_cell(0, 8, self.metadata.date, align='C')

        # Table des matières
        if len(self.chapters) > 1:
            pdf.add_page()
            pdf.set_font(main_font, 'B', 18)
            pdf.multi_cell(0, 10, "Table des matieres", align='C')
            pdf.ln(10)

            pdf.set_font(main_font, '', 12)
            for i, chapter in enumerate(self.chapters):
                # Tronquer les titres trop longs pour la TOC
                title = to_latin1(chapter.title)
                if len(title) > 55:
                    title = title[:52] + "..."
                pdf.multi_cell(0, 8, f"{i+1}. {title}")
                pdf.set_x(pdf.l_margin)  # Reset x position to left margin

        # Chapitres
        for chapter in self.chapters:
            pdf.add_page()

            # Titre du chapitre
            pdf.set_font(main_font, 'B', 18)
            chapter_title = to_latin1(chapter.title)
            if len(chapter_title) > 50:
                # Découper les titres longs
                import textwrap
                title_lines = textwrap.wrap(chapter_title, width=45)
                for line in title_lines:
                    pdf.multi_cell(0, 10, line, align='C')
            else:
                pdf.multi_cell(0, 10, chapter_title, align='C')
            pdf.ln(10)

            # Contenu
            content_font_size = min(font_size, 11)  # Max 11pt pour éviter débordements
            pdf.set_font(main_font, '', content_font_size)

            # Traiter le contenu paragraphe par paragraphe
            paragraphs = chapter.content.split('\n\n')
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                # Ignorer les éléments Markdown
                if para.startswith('#') or para.startswith('>') or para == '---':
                    continue

                # Convertir en Latin-1
                para = to_latin1(para)

                # Nettoyer le formatage Markdown
                para = re.sub(r'\*\*(.+?)\*\*', r'\1', para)  # Bold
                para = re.sub(r'\*(.+?)\*', r'\1', para)      # Italic

                # Dialogue (commence par un tiret)
                if para.startswith('-'):
                    para = '- ' + para.lstrip('- ')

                try:
                    pdf.multi_cell(0, 5.5, para)
                    pdf.set_x(pdf.l_margin)
                    pdf.ln(2)
                except Exception:
                    # Si ça échoue encore, découper en lignes plus courtes
                    import textwrap
                    lines = textwrap.wrap(para, width=65)
                    for line in lines:
                        try:
                            pdf.multi_cell(0, 5.5, line)
                            pdf.set_x(pdf.l_margin)
                        except Exception:
                            pass  # Skip problematic lines
                    pdf.ln(2)

        # Sauvegarder
        pdf.output(str(output_path))

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ PDF exporté: {output_path} ({size_mb:.1f} MB)")
        return True

    # =========================================================================
    # Export EPUB
    # =========================================================================

    def export_epub(
        self,
        output_path: Path,
        cover_image: Optional[Path] = None
    ) -> bool:
        """
        Exporte le livre en EPUB.

        Args:
            output_path: Chemin du fichier EPUB
            cover_image: Image de couverture (optionnel)

        Returns:
            True si l'export a réussi
        """
        try:
            from ebooklib import epub
        except ImportError:
            print("⚠️ ebooklib non installé. Installation: pip install ebooklib")
            return False

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Créer le livre EPUB
        book = epub.EpubBook()

        # Métadonnées
        book.set_identifier(f'audioreader-{datetime.now().strftime("%Y%m%d%H%M%S")}')
        book.set_title(self.metadata.title)
        book.set_language(self.metadata.language)
        book.add_author(self.metadata.author)

        if self.metadata.description:
            book.add_metadata('DC', 'description', self.metadata.description)

        # Image de couverture
        cover = cover_image or self.metadata.cover_image
        if cover and Path(cover).exists():
            with open(cover, 'rb') as f:
                book.set_cover('cover.jpg', f.read())

        # Style CSS
        style = '''
        body {
            font-family: Georgia, serif;
            line-height: 1.6;
            margin: 5%;
        }
        h1 {
            text-align: center;
            margin-bottom: 2em;
            page-break-before: always;
        }
        h1:first-of-type {
            page-break-before: avoid;
        }
        p {
            text-indent: 1.5em;
            margin: 0.5em 0;
            text-align: justify;
        }
        p.dialogue {
            text-indent: 0;
        }
        .chapter-title {
            font-size: 1.5em;
            font-weight: bold;
            text-align: center;
            margin: 2em 0;
        }
        '''

        css = epub.EpubItem(
            uid='style',
            file_name='style/main.css',
            media_type='text/css',
            content=style
        )
        book.add_item(css)

        # Créer les chapitres
        epub_chapters = []

        for i, chapter in enumerate(self.chapters):
            # Créer le chapitre EPUB
            c = epub.EpubHtml(
                title=chapter.title,
                file_name=f'chapter_{i+1:02d}.xhtml',
                lang=self.metadata.language
            )

            # Convertir le contenu en HTML
            html_content = self._markdown_to_html(chapter.content)

            c.content = f'''
            <html>
            <head>
                <title>{html.escape(chapter.title)}</title>
                <link rel="stylesheet" type="text/css" href="style/main.css"/>
            </head>
            <body>
                <h1>{html.escape(chapter.title)}</h1>
                {html_content}
            </body>
            </html>
            '''

            c.add_item(css)
            book.add_item(c)
            epub_chapters.append(c)

        # Table des matières
        book.toc = tuple(epub_chapters)

        # Spine (ordre de lecture)
        book.spine = ['nav'] + epub_chapters

        # Ajouter les items de navigation
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())

        # Sauvegarder
        epub.write_epub(str(output_path), book, {})

        size_kb = output_path.stat().st_size / 1024
        print(f"✅ EPUB exporté: {output_path} ({size_kb:.1f} KB)")
        return True

    # =========================================================================
    # Export HTML
    # =========================================================================

    def export_html(
        self,
        output_path: Path,
        single_file: bool = True,
        include_toc: bool = True
    ) -> bool:
        """
        Exporte le livre en HTML.

        Args:
            output_path: Chemin du fichier HTML
            single_file: True pour un seul fichier, False pour un fichier par chapitre
            include_toc: Inclure la table des matières

        Returns:
            True si l'export a réussi
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Style CSS intégré
        css = '''
        :root {
            --bg-color: #fefefe;
            --text-color: #333;
            --accent-color: #8b4513;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --bg-color: #1a1a1a;
                --text-color: #e0e0e0;
                --accent-color: #d4a574;
            }
        }
        body {
            font-family: Georgia, 'Times New Roman', serif;
            line-height: 1.8;
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            background-color: var(--bg-color);
            color: var(--text-color);
        }
        h1 {
            text-align: center;
            color: var(--accent-color);
            border-bottom: 2px solid var(--accent-color);
            padding-bottom: 0.5em;
            margin-top: 2em;
        }
        h1.book-title {
            font-size: 2.5em;
            margin-top: 1em;
        }
        .author {
            text-align: center;
            font-style: italic;
            font-size: 1.2em;
            margin-bottom: 3em;
        }
        .toc {
            background: rgba(128, 128, 128, 0.1);
            padding: 1.5em;
            border-radius: 8px;
            margin: 2em 0;
        }
        .toc h2 {
            margin-top: 0;
            color: var(--accent-color);
        }
        .toc ul {
            list-style: none;
            padding-left: 0;
        }
        .toc li {
            margin: 0.5em 0;
        }
        .toc a {
            color: var(--text-color);
            text-decoration: none;
        }
        .toc a:hover {
            color: var(--accent-color);
        }
        p {
            text-indent: 1.5em;
            margin: 0.8em 0;
            text-align: justify;
        }
        p.dialogue {
            text-indent: 0;
        }
        .chapter {
            margin-bottom: 4em;
            page-break-after: always;
        }
        '''

        if single_file:
            # Un seul fichier HTML
            chapters_html = []
            toc_items = []

            for i, chapter in enumerate(self.chapters):
                chapter_id = f"chapter-{i+1}"
                toc_items.append(f'<li><a href="#{chapter_id}">{html.escape(chapter.title)}</a></li>')

                content_html = self._markdown_to_html(chapter.content)
                chapters_html.append(f'''
                <div class="chapter" id="{chapter_id}">
                    <h1>{html.escape(chapter.title)}</h1>
                    {content_html}
                </div>
                ''')

            toc_html = ''
            if include_toc and len(self.chapters) > 1:
                toc_html = f'''
                <nav class="toc">
                    <h2>Table des matières</h2>
                    <ul>
                        {''.join(toc_items)}
                    </ul>
                </nav>
                '''

            full_html = f'''<!DOCTYPE html>
<html lang="{self.metadata.language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="author" content="{html.escape(self.metadata.author)}">
    <title>{html.escape(self.metadata.title)}</title>
    <style>{css}</style>
</head>
<body>
    <header>
        <h1 class="book-title">{html.escape(self.metadata.title)}</h1>
        <p class="author">{html.escape(self.metadata.author)}</p>
    </header>

    {toc_html}

    <main>
        {''.join(chapters_html)}
    </main>

    <footer style="text-align: center; margin-top: 3em; font-size: 0.9em; opacity: 0.7;">
        <p>Généré par AudioReader - {datetime.now().strftime("%Y-%m-%d")}</p>
    </footer>
</body>
</html>'''

            output_path.write_text(full_html, encoding='utf-8')

        else:
            # Un fichier par chapitre
            output_dir = output_path.parent / output_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)

            for i, chapter in enumerate(self.chapters):
                chapter_file = output_dir / f"chapter_{i+1:02d}.html"
                content_html = self._markdown_to_html(chapter.content)

                chapter_html = f'''<!DOCTYPE html>
<html lang="{self.metadata.language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(chapter.title)} - {html.escape(self.metadata.title)}</title>
    <style>{css}</style>
</head>
<body>
    <h1>{html.escape(chapter.title)}</h1>
    {content_html}
</body>
</html>'''

                chapter_file.write_text(chapter_html, encoding='utf-8')

            # Index
            toc_items = [
                f'<li><a href="chapter_{i+1:02d}.html">{html.escape(c.title)}</a></li>'
                for i, c in enumerate(self.chapters)
            ]

            index_html = f'''<!DOCTYPE html>
<html lang="{self.metadata.language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(self.metadata.title)}</title>
    <style>{css}</style>
</head>
<body>
    <h1 class="book-title">{html.escape(self.metadata.title)}</h1>
    <p class="author">{html.escape(self.metadata.author)}</p>
    <nav class="toc">
        <h2>Chapitres</h2>
        <ul>
            {''.join(toc_items)}
        </ul>
    </nav>
</body>
</html>'''

            (output_dir / "index.html").write_text(index_html, encoding='utf-8')
            output_path = output_dir / "index.html"

        size_kb = output_path.stat().st_size / 1024
        print(f"✅ HTML exporté: {output_path} ({size_kb:.1f} KB)")
        return True

    # =========================================================================
    # Export TXT
    # =========================================================================

    def export_txt(
        self,
        output_path: Path,
        line_width: int = 80
    ) -> bool:
        """
        Exporte le livre en texte brut.

        Args:
            output_path: Chemin du fichier TXT
            line_width: Largeur de ligne (0 pour désactiver le word wrap)

        Returns:
            True si l'export a réussi
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []

        # Titre
        title_line = f"{'=' * line_width}" if line_width else "=" * 60
        lines.append(title_line)
        lines.append(self.metadata.title.center(line_width) if line_width else self.metadata.title)
        lines.append(self.metadata.author.center(line_width) if line_width else self.metadata.author)
        lines.append(title_line)
        lines.append("")
        lines.append("")

        # Table des matières
        if len(self.chapters) > 1:
            lines.append("TABLE DES MATIÈRES")
            lines.append("-" * 20)
            for i, chapter in enumerate(self.chapters):
                lines.append(f"  {i+1}. {chapter.title}")
            lines.append("")
            lines.append("")

        # Chapitres
        for chapter in self.chapters:
            # Séparateur
            sep = "=" * (line_width if line_width else 60)
            lines.append(sep)
            lines.append(chapter.title.upper())
            lines.append(sep)
            lines.append("")

            # Contenu
            content = chapter.content

            # Nettoyer le formatage Markdown
            content = re.sub(r'\*\*(.+?)\*\*', r'\1', content)  # Bold
            content = re.sub(r'\*(.+?)\*', r'\1', content)      # Italic

            # Word wrap si demandé
            if line_width:
                import textwrap
                paragraphs = content.split('\n\n')
                wrapped = []
                for para in paragraphs:
                    para = para.strip()
                    if para:
                        wrapped.append(textwrap.fill(para, width=line_width))
                content = '\n\n'.join(wrapped)

            lines.append(content)
            lines.append("")
            lines.append("")

        # Footer
        lines.append("-" * (line_width if line_width else 60))
        lines.append(f"Généré par AudioReader - {datetime.now().strftime('%Y-%m-%d')}")

        output_path.write_text('\n'.join(lines), encoding='utf-8')

        size_kb = output_path.stat().st_size / 1024
        print(f"✅ TXT exporté: {output_path} ({size_kb:.1f} KB)")
        return True

    # =========================================================================
    # Export All
    # =========================================================================

    def export_all(
        self,
        output_dir: Path,
        base_name: Optional[str] = None,
        formats: Optional[List[str]] = None
    ) -> Dict[str, Path]:
        """
        Exporte le livre dans tous les formats demandés.

        Args:
            output_dir: Dossier de sortie
            base_name: Nom de base des fichiers (défaut: titre slugifié)
            formats: Liste des formats ("pdf", "epub", "html", "txt")
                     Défaut: tous les formats

        Returns:
            Dict mapping format -> chemin du fichier
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not base_name:
            # Slugifier le titre
            base_name = self.metadata.title.lower()
            base_name = re.sub(r'[^a-z0-9]+', '_', base_name)
            base_name = base_name.strip('_')

        if not formats:
            formats = ["pdf", "epub", "html", "txt"]

        results = {}

        for fmt in formats:
            output_path = output_dir / f"{base_name}.{fmt}"

            try:
                if fmt == "pdf":
                    if self.export_pdf(output_path):
                        results["pdf"] = output_path
                elif fmt == "epub":
                    if self.export_epub(output_path):
                        results["epub"] = output_path
                elif fmt == "html":
                    if self.export_html(output_path):
                        results["html"] = output_path
                elif fmt == "txt":
                    if self.export_txt(output_path):
                        results["txt"] = output_path
                else:
                    print(f"⚠️ Format inconnu: {fmt}")
            except Exception as e:
                print(f"⚠️ Erreur export {fmt}: {e}")

        return results


# =============================================================================
# Fonctions utilitaires
# =============================================================================

def export_markdown_book(
    chapters_dir: Path,
    output_dir: Path,
    title: str,
    author: str = "Auteur Inconnu",
    formats: Optional[List[str]] = None,
    chapter_pattern: str = "*.md"
) -> Dict[str, Path]:
    """
    Exporte un livre depuis un dossier de fichiers Markdown.

    Args:
        chapters_dir: Dossier contenant les fichiers .md
        output_dir: Dossier de sortie
        title: Titre du livre
        author: Nom de l'auteur
        formats: Formats à exporter (défaut: tous)
        chapter_pattern: Pattern glob pour les chapitres

    Returns:
        Dict mapping format -> chemin du fichier
    """
    chapters_dir = Path(chapters_dir)

    # Trouver les fichiers chapitres
    chapter_files = sorted(chapters_dir.glob(chapter_pattern))

    if not chapter_files:
        print(f"⚠️ Aucun fichier trouvé dans {chapters_dir}")
        return {}

    print(f"📚 Export de {len(chapter_files)} chapitres...")

    exporter = BookExporter(title=title, author=author)

    for chapter_file in chapter_files:
        print(f"  + {chapter_file.name}")
        exporter.add_chapter_from_markdown(chapter_file)

    return exporter.export_all(output_dir, formats=formats)


if __name__ == "__main__":
    # Test avec du contenu simple
    exporter = BookExporter(
        title="Test Export",
        author="Claude AI"
    )

    exporter.add_chapter(
        "Chapitre 1: Introduction",
        """Ceci est le premier paragraphe du chapitre.

— Bonjour, dit Marie. Comment allez-vous ?

— Très bien, répondit Victor. Et vous ?

Ce dialogue illustre le formatage des dialogues en français."""
    )

    exporter.add_chapter(
        "Chapitre 2: Suite",
        """Le deuxième chapitre continue l'histoire.

Les personnages **importants** se retrouvent dans une situation *délicate*.

Mais tout finira bien."""
    )

    # Export
    from pathlib import Path
    output = Path("output/test_export")
    results = exporter.export_all(output)

    print("\nFichiers générés:")
    for fmt, path in results.items():
        print(f"  {fmt}: {path}")
