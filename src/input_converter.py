"""
Convertisseur d'entrées universel (PDF, EPUB -> Markdown).

Permet d'utiliser n'importe quel ebook comme source pour AudioReader.
"""
import sys
import re
from pathlib import Path
from typing import Optional
import html

# Imports conditionnels pour les formats
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    EPUB_AVAILABLE = True
except ImportError:
    EPUB_AVAILABLE = False


class InputConverter:
    """Convertit divers formats de documents en Markdown structuré."""

    def __init__(self):
        pass

    def convert_to_markdown(self, input_path: Path, output_dir: Optional[Path] = None) -> Path:
        """
        Convertit un fichier (PDF, EPUB) en Markdown.
        
        Args:
            input_path: Chemin du fichier source
            output_dir: Dossier de sortie (temp par défaut)
            
        Returns:
            Chemin du fichier Markdown généré
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {input_path}")

        if output_dir is None:
            output_dir = input_path.parent
        
        output_path = output_dir / f"{input_path.stem}.md"
        
        ext = input_path.suffix.lower()
        
        if ext == ".md" or ext == ".txt":
            # Déjà au bon format, on retourne tel quel ou on copie
            return input_path
        elif ext == ".pdf":
            if not PDF_AVAILABLE:
                raise ImportError("PyMuPDF requis pour les PDF. Installez: pip install pymupdf")
            return self._convert_pdf(input_path, output_path)
        elif ext == ".epub":
            if not EPUB_AVAILABLE:
                raise ImportError("EbookLib et BeautifulSoup requis. Installez: pip install ebooklib beautifulsoup4")
            return self._convert_epub(input_path, output_path)
        else:
            raise ValueError(f"Format non supporté : {ext}")

    def _convert_pdf(self, input_path: Path, output_path: Path) -> Path:
        """Convertit un PDF en Markdown (extraction basique avec heuristique de titres)."""
        doc = fitz.open(input_path)
        md_content = []
        
        # Titre probable (nom du fichier)
        md_content.append(f"# {input_path.stem}\n")
        
        for page_num, page in enumerate(doc):
            text = page.get_text("dict")
            blocks = text["blocks"]
            
            page_content = []
            
            for b in blocks:
                if "lines" in b:
                    for l in b["lines"]:
                        for s in l["spans"]:
                            size = s["size"]
                            text = s["text"].strip()
                            if not text:
                                continue
                                
                            # Heuristique simple pour les titres : grande police
                            # (A ajuster selon les PDF)
                            if size > 14: 
                                page_content.append(f"\n## {text}\n")
                            elif size > 12:
                                page_content.append(f"\n### {text}\n")
                            else:
                                page_content.append(text)
            
            # Reconstruire le texte de la page
            raw_text = " ".join(page_content)
            # Nettoyer un peu
            raw_text = re.sub(r'\s+', ' ', raw_text)
            
            md_content.append(raw_text)
            
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(md_content))
            
        return output_path

    def _convert_epub(self, input_path: Path, output_path: Path) -> Path:
        """Convertit un EPUB en Markdown."""
        book = epub.read_epub(str(input_path))
        md_content = []
        
        # Titre
        title = book.get_metadata('DC', 'title')
        if title:
            md_content.append(f"# {title[0][0]}\n")
        
        # Parcourir les items
        # L'ordre est important, on utilise le spine si possible, sinon l'ordre des items
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            
            # Titres
            for h in soup.find_all(['h1', 'h2', 'h3', 'h4']):
                level = int(h.name[1])
                # Convertir en Markdown header (#, ##, etc)
                # On ajoute +1 car # est le titre du livre
                prefix = '#' * (level + 1)
                h.replace_with(f"\n{prefix} {h.get_text()}\n")
                
            # Paragraphes
            for p in soup.find_all('p'):
                p.replace_with(f"{p.get_text()}\n\n")
                
            text = soup.get_text()
            # Nettoyage des lignes vides excessives
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            md_content.append(text)
            
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n***\n".join(md_content))
            
        return output_path
