"""
Nettoyage intelligent de texte pour AudioReader.

Heuristiques pour corriger les textes mal convertis depuis PDF:
- Suppression des en-tetes/pieds de page recurrents
- Reconstruction de paragraphes coupes
- Correction des cesures
"""
import re
from dataclasses import dataclass
from typing import List, Optional, Dict
from collections import Counter


@dataclass
class CleaningStats:
    """Statistiques de nettoyage."""
    original_chars: int = 0
    cleaned_chars: int = 0
    headers_removed: int = 0
    page_numbers_removed: int = 0
    hyphens_fixed: int = 0
    paragraphs_merged: int = 0


class SmartTextCleaner:
    """
    Nettoyeur intelligent de texte pour PDF mal convertis.

    Detecte et corrige automatiquement les problemes courants
    de conversion PDF -> texte.
    """

    def __init__(self):
        self.stats = CleaningStats()

    def clean(self, raw_text: str) -> str:
        """
        Nettoie le texte brut.

        Args:
            raw_text: Texte brut a nettoyer

        Returns:
            Texte nettoye
        """
        self.stats = CleaningStats(original_chars=len(raw_text))

        text = raw_text

        # 1. Supprimer les numeros de page
        text = self._remove_page_numbers(text)

        # 2. Supprimer les en-tetes/pieds de page recurrents
        text = self._remove_recurring_headers(text)

        # 3. Corriger les cesures
        text = self._fix_hyphens(text)

        # 4. Fusionner les paragraphes coupes
        text = self._merge_broken_paragraphs(text)

        # 5. Normaliser les espaces
        text = self._normalize_whitespace(text)

        self.stats.cleaned_chars = len(text)
        return text

    def _remove_page_numbers(self, text: str) -> str:
        """Supprime les numeros de page isoles."""
        lines = text.split("\n")
        cleaned = []
        count = 0

        for line in lines:
            stripped = line.strip()
            # Ligne contenant uniquement un numero
            if re.match(r'^\d{1,4}$', stripped):
                count += 1
                continue
            # Patterns comme "- 42 -" ou "Page 42"
            if re.match(r'^[-\u2013\u2014]?\s*\d{1,4}\s*[-\u2013\u2014]?$', stripped):
                count += 1
                continue
            if re.match(r'^[Pp]age\s+\d{1,4}$', stripped):
                count += 1
                continue
            cleaned.append(line)

        self.stats.page_numbers_removed = count
        return "\n".join(cleaned)

    def _remove_recurring_headers(self, text: str) -> str:
        """Detecte et supprime les en-tetes/pieds de page recurrents."""
        lines = text.split("\n")
        if len(lines) < 20:
            return text

        # Compter les lignes qui apparaissent souvent (>= 3 fois)
        line_counts = Counter(line.strip() for line in lines if line.strip())
        recurring = {line for line, count in line_counts.items()
                     if count >= 3 and len(line) < 100 and len(line) > 2}

        if not recurring:
            return text

        cleaned = []
        for line in lines:
            if line.strip() in recurring:
                self.stats.headers_removed += 1
                continue
            cleaned.append(line)

        return "\n".join(cleaned)

    def _fix_hyphens(self, text: str) -> str:
        """Corrige les mots coupes par des cesures en fin de ligne."""
        # Pattern: mot- \n suite_du_mot
        original_count = len(re.findall(r'(\w+)-\s*\n\s*(\w+)', text))

        text = re.sub(
            r'(\w+)-\s*\n\s*(\w+)',
            r'\1\2',
            text,
        )

        self.stats.hyphens_fixed = original_count
        return text

    def _merge_broken_paragraphs(self, text: str) -> str:
        """Fusionne les paragraphes coupes artificiellement."""
        lines = text.split("\n")
        merged = []
        count = 0

        i = 0
        while i < len(lines):
            line = lines[i]

            # Si la ligne ne finit pas par un signe de ponctuation
            # et la suivante commence par une minuscule -> fusionner
            if (i + 1 < len(lines)
                and line.strip()
                and not line.strip().endswith(('.', '!', '?', ':', ';', '\u00bb', '"', '\u2019'))
                and not line.strip().startswith('#')
                and lines[i + 1].strip()
                and lines[i + 1].strip()[0].islower()):
                merged.append(line.rstrip() + " " + lines[i + 1].lstrip())
                count += 1
                i += 2
            else:
                merged.append(line)
                i += 1

        self.stats.paragraphs_merged = count
        return "\n".join(merged)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalise les espaces multiples et lignes vides."""
        # Espaces multiples -> simple
        text = re.sub(r'[ \t]+', ' ', text)
        # Plus de 2 lignes vides consecutives -> 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def get_stats(self) -> dict:
        """Retourne les statistiques de nettoyage."""
        return {
            "original_chars": self.stats.original_chars,
            "cleaned_chars": self.stats.cleaned_chars,
            "reduction_percent": round(
                100 * (1 - self.stats.cleaned_chars / max(self.stats.original_chars, 1)), 1
            ),
            "headers_removed": self.stats.headers_removed,
            "page_numbers_removed": self.stats.page_numbers_removed,
            "hyphens_fixed": self.stats.hyphens_fixed,
            "paragraphs_merged": self.stats.paragraphs_merged,
        }
