"""
Parser Markdown pour extraire les chapitres d'un livre.

Supporte:
- Fichier unique avec headers Markdown
- Répertoire avec plusieurs fichiers (un par chapitre)
- Fichiers EPUB
"""
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import zipfile
import html.parser


@dataclass
class Chapter:
    """Représente un chapitre du livre."""
    number: int
    title: str
    content: str
    source_file: Optional[Path] = None

    def get_full_text(self) -> str:
        """Retourne le texte complet pour la synthèse vocale."""
        return f"{self.title}\n\n{self.content}"

    def get_filename(self) -> str:
        """Génère un nom de fichier sûr pour le chapitre."""
        safe_title = re.sub(r'[^\w\s-]', '', self.title)
        safe_title = re.sub(r'\s+', '_', safe_title)
        return f"{self.number:02d}_{safe_title[:50]}"

    def get_word_count(self) -> int:
        """Retourne le nombre de mots."""
        return len(self.content.split())

    def get_char_count(self) -> int:
        """Retourne le nombre de caractères."""
        return len(self.content)


class MarkdownBookParser:
    """Parse un livre Markdown et extrait les chapitres."""

    def __init__(self, header_level: int = 1):
        """
        Args:
            header_level: Niveau des headers pour les chapitres (1 = #, 2 = ##, etc.)
        """
        self.header_level = header_level
        self.header_pattern = re.compile(
            rf'^{"#" * header_level}\s+(.+)$',
            re.MULTILINE
        )

    def parse_file(self, filepath: str | Path) -> list[Chapter]:
        """Parse un fichier Markdown et retourne la liste des chapitres."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {filepath}")

        content = filepath.read_text(encoding='utf-8')
        return self.parse_content(content)

    def parse_content(self, content: str) -> list[Chapter]:
        """Parse le contenu Markdown et retourne la liste des chapitres."""
        chapters = []

        # Trouver tous les headers
        matches = list(self.header_pattern.finditer(content))

        if not matches:
            # Pas de chapitres trouvés, traiter tout comme un seul chapitre
            cleaned_content = self._clean_text(content)
            if cleaned_content.strip():
                chapters.append(Chapter(
                    number=1,
                    title="Contenu",
                    content=cleaned_content
                ))
            return chapters

        for i, match in enumerate(matches):
            title = match.group(1).strip()
            start = match.end()

            # Fin = début du prochain chapitre ou fin du fichier
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(content)

            chapter_content = content[start:end]
            cleaned_content = self._clean_text(chapter_content)

            chapters.append(Chapter(
                number=i + 1,
                title=title,
                content=cleaned_content
            ))

        return chapters

    def _clean_text(self, text: str) -> str:
        """Nettoie le texte Markdown pour la synthèse vocale."""
        # Supprimer les blocs de code
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]+`', '', text)

        # Supprimer les images
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

        # Convertir les liens en texte simple
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

        # Supprimer le formatage bold/italic
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)

        # Supprimer les headers de niveau inférieur (les convertir en texte)
        text = re.sub(r'^#{2,}\s+', '', text, flags=re.MULTILINE)

        # Supprimer les listes à puces/numérotées (garder le contenu)
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

        # Supprimer les blockquotes
        text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)

        # Supprimer les lignes horizontales
        text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)

        # Normaliser les espaces
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()


class DirectoryBookParser:
    """Parse un livre à partir d'un répertoire avec plusieurs fichiers."""

    def __init__(self, file_pattern: str = "*.md"):
        """
        Args:
            file_pattern: Pattern glob pour les fichiers chapitres
        """
        self.file_pattern = file_pattern

    def parse_directory(self, dirpath: str | Path) -> list[Chapter]:
        """Parse un répertoire et retourne la liste des chapitres."""
        dirpath = Path(dirpath)
        if not dirpath.is_dir():
            raise NotADirectoryError(f"Répertoire non trouvé: {dirpath}")

        # Trouver tous les fichiers correspondant au pattern
        files = sorted(dirpath.glob(self.file_pattern))

        if not files:
            raise FileNotFoundError(f"Aucun fichier {self.file_pattern} trouvé dans {dirpath}")

        chapters = []
        md_parser = MarkdownBookParser(header_level=1)

        for i, filepath in enumerate(files, 1):
            content = filepath.read_text(encoding='utf-8')

            # Extraire le titre du premier header ou du nom de fichier
            title = self._extract_title(content, filepath)

            # Nettoyer le contenu
            cleaned = md_parser._clean_text(content)

            # Retirer le titre du contenu s'il est au début
            cleaned = self._remove_title_from_content(cleaned, title)

            chapters.append(Chapter(
                number=i,
                title=title,
                content=cleaned,
                source_file=filepath
            ))

        return chapters

    def _extract_title(self, content: str, filepath: Path) -> str:
        """Extrait le titre du contenu ou du nom de fichier."""
        # Chercher un header de niveau 1
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()

        # Sinon utiliser le nom de fichier
        name = filepath.stem
        # Nettoyer les préfixes numériques (01_, 001-, etc.)
        name = re.sub(r'^[\d_\-]+', '', name)
        # Remplacer underscores/tirets par espaces
        name = re.sub(r'[_-]+', ' ', name)
        return name.strip().title() or f"Chapitre {filepath.stem}"

    def _remove_title_from_content(self, content: str, title: str) -> str:
        """Retire le titre du début du contenu."""
        lines = content.split('\n')
        if lines and lines[0].strip().lower() == title.lower():
            return '\n'.join(lines[1:]).strip()
        return content


class HTMLTextExtractor(html.parser.HTMLParser):
    """Extracteur de texte depuis HTML."""

    def __init__(self):
        super().__init__()
        self.text = []
        self.current_tag = None
        self.skip_tags = {'script', 'style', 'nav', 'header', 'footer'}

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        if tag in ('p', 'br', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            self.text.append('\n')
        if tag in ('h1', 'h2', 'h3'):
            self.text.append('\n')

    def handle_endtag(self, tag):
        if tag in ('p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            self.text.append('\n')
        self.current_tag = None

    def handle_data(self, data):
        if self.current_tag not in self.skip_tags:
            self.text.append(data)

    def get_text(self) -> str:
        text = ''.join(self.text)
        # Normaliser les espaces
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()


class EPUBParser:
    """Parse un fichier EPUB et extrait les chapitres."""

    def parse_file(self, filepath: str | Path) -> list[Chapter]:
        """Parse un fichier EPUB et retourne la liste des chapitres."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {filepath}")

        chapters = []

        with zipfile.ZipFile(filepath, 'r') as zf:
            # Trouver les fichiers de contenu
            content_files = self._find_content_files(zf)

            for i, (name, content) in enumerate(content_files, 1):
                title = self._extract_title_from_html(content) or f"Chapitre {i}"
                text = self._extract_text_from_html(content)

                if text.strip():
                    chapters.append(Chapter(
                        number=i,
                        title=title,
                        content=text
                    ))

        return chapters

    def _find_content_files(self, zf: zipfile.ZipFile) -> list[tuple[str, str]]:
        """Trouve et retourne les fichiers de contenu dans l'ordre."""
        content_files = []

        for name in zf.namelist():
            if name.endswith(('.xhtml', '.html', '.htm')):
                # Ignorer les fichiers de navigation/toc
                if any(skip in name.lower() for skip in ['toc', 'nav', 'cover', 'title']):
                    continue
                try:
                    content = zf.read(name).decode('utf-8')
                    content_files.append((name, content))
                except:
                    pass

        # Trier par nom de fichier
        content_files.sort(key=lambda x: x[0])
        return content_files

    def _extract_title_from_html(self, html_content: str) -> str:
        """Extrait le titre depuis le HTML."""
        # Chercher un h1 ou h2
        match = re.search(r'<h[12][^>]*>([^<]+)</h[12]>', html_content, re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            # Nettoyer les entités HTML
            title = html.unescape(title) if hasattr(html, 'unescape') else title
            return title
        return ""

    def _extract_text_from_html(self, html_content: str) -> str:
        """Extrait le texte depuis le HTML."""
        extractor = HTMLTextExtractor()
        try:
            extractor.feed(html_content)
            return extractor.get_text()
        except:
            # Fallback: supprimer les tags HTML brutalement
            text = re.sub(r'<[^>]+>', ' ', html_content)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()


def parse_book(filepath: str | Path, header_level: int = 1) -> list[Chapter]:
    """
    Fonction utilitaire pour parser un livre.

    Détecte automatiquement le type d'entrée:
    - Fichier .md: Parse comme Markdown avec headers
    - Fichier .epub: Parse comme EPUB
    - Répertoire: Parse tous les fichiers .md du répertoire

    Args:
        filepath: Chemin vers le fichier/répertoire
        header_level: Niveau des headers pour les chapitres (fichiers .md)

    Returns:
        Liste des chapitres
    """
    filepath = Path(filepath)

    if filepath.is_dir():
        # Répertoire avec plusieurs fichiers
        parser = DirectoryBookParser(file_pattern="*.md")
        return parser.parse_directory(filepath)

    elif filepath.suffix.lower() == '.epub':
        # Fichier EPUB
        parser = EPUBParser()
        return parser.parse_file(filepath)

    else:
        # Fichier Markdown unique
        parser = MarkdownBookParser(header_level=header_level)
        return parser.parse_file(filepath)


def parse_directory(dirpath: str | Path, pattern: str = "*.md") -> list[Chapter]:
    """
    Parse un répertoire contenant les chapitres.

    Args:
        dirpath: Chemin vers le répertoire
        pattern: Pattern glob pour les fichiers (défaut: *.md)

    Returns:
        Liste des chapitres
    """
    parser = DirectoryBookParser(file_pattern=pattern)
    return parser.parse_directory(dirpath)
