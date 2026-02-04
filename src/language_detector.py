"""
Detection automatique de langue pour AudioReader.

Detecte la langue d'un texte en utilisant langdetect ou des heuristiques.
"""
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple


# Caracteres et mots frequents par langue (heuristique)
LANGUAGE_HINTS: Dict[str, Dict] = {
    "fr": {
        "chars": set("eaisonlurdtcpmqfbghjvwxyz\u00e9\u00e8\u00ea\u00eb\u00e0\u00e2\u00e7\u00f4\u00fb\u00ee\u00ef\u00f9\u00fc\u0153\u00e6"),
        "common_words": {"le", "la", "les", "de", "des", "un", "une", "et", "est", "en", "que", "qui", "dans", "pour", "pas", "sur", "avec", "plus", "tout", "mais", "ce", "il", "elle", "sont", "nous", "vous", "ils"},
        "diacritics": set("\u00e9\u00e8\u00ea\u00eb\u00e0\u00e2\u00e7\u00f4\u00fb\u00ee\u00ef\u00f9\u00fc\u0153\u00e6"),
    },
    "en": {
        "chars": set("eaisonlurdtcpmqfbghjvwxyz"),
        "common_words": {"the", "be", "to", "of", "and", "a", "in", "that", "have", "it", "for", "not", "on", "with", "he", "she", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we"},
        "diacritics": set(),
    },
    "de": {
        "chars": set("eaisonlurdtcpmqfbghjvwxyz\u00e4\u00f6\u00fc\u00df"),
        "common_words": {"der", "die", "das", "und", "ist", "ein", "eine", "ich", "nicht", "auf", "mit", "den", "von", "sie", "sich", "des", "dem", "dass", "es"},
        "diacritics": set("\u00e4\u00f6\u00fc\u00df"),
    },
    "es": {
        "chars": set("eaisonlurdtcpmqfbghjvwxyz\u00e1\u00e9\u00ed\u00f3\u00fa\u00f1\u00fc"),
        "common_words": {"el", "la", "los", "las", "de", "en", "un", "una", "que", "es", "por", "con", "no", "para", "del", "se", "como", "pero", "fue"},
        "diacritics": set("\u00e1\u00e9\u00ed\u00f3\u00fa\u00f1\u00fc"),
    },
    "it": {
        "chars": set("eaisonlurdtcpmqfbghjvwxyz\u00e0\u00e8\u00e9\u00ec\u00f2\u00f9"),
        "common_words": {"il", "lo", "la", "di", "che", "non", "un", "una", "del", "con", "per", "sono", "anche", "questo", "come", "ma", "dalla", "nella"},
        "diacritics": set("\u00e0\u00e8\u00e9\u00ec\u00f2\u00f9"),
    },
    "pt": {
        "chars": set("eaisonlurdtcpmqfbghjvwxyz\u00e1\u00e0\u00e2\u00e3\u00e9\u00ea\u00ed\u00f3\u00f4\u00f5\u00fa\u00fc\u00e7"),
        "common_words": {"de", "que", "os", "um", "uma", "para", "com", "por", "como", "mais", "mas", "foi", "nao", "das", "dos", "na", "no", "ao", "se"},
        "diacritics": set("\u00e1\u00e0\u00e2\u00e3\u00e9\u00ea\u00ed\u00f3\u00f4\u00f5\u00fa\u00fc\u00e7"),
    },
}


@dataclass
class DetectionResult:
    """Resultat de detection de langue."""
    language: str
    confidence: float
    method: str  # "langdetect" ou "heuristic"
    alternatives: List[Tuple[str, float]] = None

    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []


class AutoLanguageDetector:
    """
    Detecteur automatique de langue.

    Utilise langdetect si disponible, sinon des heuristiques
    basees sur les caracteres et mots frequents.
    """

    def __init__(self):
        self._has_langdetect = None

    def _check_langdetect(self) -> bool:
        if self._has_langdetect is None:
            try:
                import langdetect
                self._has_langdetect = True
            except ImportError:
                self._has_langdetect = False
        return self._has_langdetect

    def detect(self, text: str, sample_size: int = 5000) -> DetectionResult:
        """
        Detecte la langue du texte.

        Args:
            text: Texte a analyser
            sample_size: Nombre de caracteres a echantillonner

        Returns:
            DetectionResult avec la langue detectee
        """
        sample = text[:sample_size]

        if self._check_langdetect():
            return self._detect_langdetect(sample)
        return self._detect_heuristic(sample)

    def _detect_langdetect(self, text: str) -> DetectionResult:
        """Detection via la bibliotheque langdetect."""
        import langdetect
        from langdetect import detect_langs

        try:
            results = detect_langs(text)
            if results:
                best = results[0]
                alternatives = [(str(r.lang), round(r.prob, 3)) for r in results[1:4]]
                return DetectionResult(
                    language=str(best.lang),
                    confidence=round(best.prob, 3),
                    method="langdetect",
                    alternatives=alternatives,
                )
        except langdetect.LangDetectException:
            pass

        return self._detect_heuristic(text)

    def _detect_heuristic(self, text: str) -> DetectionResult:
        """Detection par heuristiques (caracteres et mots frequents)."""
        text_lower = text.lower()
        words = set(text_lower.split())
        chars = set(text_lower)

        scores: Dict[str, float] = {}

        for lang, hints in LANGUAGE_HINTS.items():
            score = 0.0

            # Score sur les mots communs
            common = words & hints["common_words"]
            score += len(common) * 2.0

            # Score sur les diacritiques
            diac_found = chars & hints["diacritics"]
            score += len(diac_found) * 3.0

            scores[lang] = score

        if not scores:
            return DetectionResult("fr", 0.5, "heuristic")

        # Trier par score
        sorted_langs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_lang, best_score = sorted_langs[0]

        # Normaliser la confiance
        total = sum(s for _, s in sorted_langs) or 1.0
        confidence = min(best_score / total, 1.0)

        alternatives = [(l, round(s / total, 3)) for l, s in sorted_langs[1:4]]

        return DetectionResult(
            language=best_lang,
            confidence=round(confidence, 3),
            method="heuristic",
            alternatives=alternatives,
        )
