"""
Resume automatique de chapitre pour AudioReader.

Genere des resumes courts de chapitres pour les metadonnees
des audiobooks (M4B, podcast RSS).
"""
import re
from dataclasses import dataclass
from typing import Optional, List
import json


@dataclass
class ChapterSummary:
    """Resume d'un chapitre."""
    title: str
    summary: str
    word_count: int
    key_characters: List[str]


class ChapterSummarizer:
    """
    Generateur de resumes de chapitres.

    Utilise un LLM si disponible (Ollama), sinon
    extraction extractive (premieres phrases).
    """

    def __init__(self, provider: str = "ollama", model: str = "llama3.2"):
        self.provider = provider
        self.model = model

    def summarize(self, text: str, title: str = "", max_words: int = 50) -> ChapterSummary:
        """
        Resume un chapitre.

        Args:
            text: Texte du chapitre
            title: Titre du chapitre
            max_words: Nombre maximum de mots dans le resume

        Returns:
            ChapterSummary
        """
        word_count = len(text.split())

        # Essayer le LLM
        summary = self._llm_summarize(text, max_words)
        if not summary:
            summary = self._extractive_summarize(text, max_words)

        # Extraire les personnages mentionnes
        characters = self._extract_characters(text)

        return ChapterSummary(
            title=title,
            summary=summary,
            word_count=word_count,
            key_characters=characters,
        )

    def _llm_summarize(self, text: str, max_words: int) -> Optional[str]:
        """Resume via LLM."""
        try:
            import urllib.request

            prompt = f"""Resume ce texte en {max_words} mots maximum. Ecris directement le resume, sans introduction.

Texte :
{text[:3000]}

Resume :"""

            payload = json.dumps({
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3},
            }).encode("utf-8")

            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                summary = data.get("response", "").strip()
                if summary:
                    # Tronquer si trop long
                    words = summary.split()
                    if len(words) > max_words * 1.5:
                        summary = " ".join(words[:max_words]) + "..."
                    return summary

        except Exception:
            pass

        return None

    def _extractive_summarize(self, text: str, max_words: int) -> str:
        """Resume extractif : premieres phrases significatives."""
        # Nettoyer le texte
        clean = re.sub(r'\s+', ' ', text).strip()

        # Decouper en phrases
        sentences = re.split(r'(?<=[.!?])\s+', clean)

        # Filtrer les phrases trop courtes
        sentences = [s for s in sentences if len(s.split()) > 3]

        if not sentences:
            return clean[:max_words * 6] + "..."

        # Prendre les premieres phrases jusqu'a max_words
        summary_words = []
        for sentence in sentences:
            words = sentence.split()
            if len(summary_words) + len(words) <= max_words:
                summary_words.extend(words)
            else:
                break

        if not summary_words and sentences:
            # Au moins la premiere phrase
            summary_words = sentences[0].split()[:max_words]

        return " ".join(summary_words)

    def _extract_characters(self, text: str) -> List[str]:
        """Extrait les noms de personnages probables du texte."""
        # Heuristique : mots capitalises qui apparaissent plusieurs fois
        # et ne sont pas en debut de phrase
        words = re.findall(r'(?<=[a-z\u00e9\u00e8\u00ea] )([A-Z\u00c0-\u00dc][a-z\u00e0-\u00fc]{2,})', text)

        from collections import Counter
        word_counts = Counter(words)

        # Garder ceux qui apparaissent >= 2 fois
        characters = [name for name, count in word_counts.most_common(10) if count >= 2]

        return characters[:5]  # Maximum 5 personnages
