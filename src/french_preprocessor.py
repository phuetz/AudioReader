"""
Préprocesseur de texte français pour TTS.

Gère:
1. Conversion des nombres/dates/heures en toutes lettres (via num2words)
2. Extension des abréviations (M. -> Monsieur, etc.)
3. Gestion des unités (km/h, %, €)
4. Dictionnaire de prononciation personnalisé (acronymes, mots étrangers)
"""
import re
from num2words import num2words
from typing import Dict, List, Tuple

class FrenchTextPreprocessor:
    def __init__(self):
        # Dictionnaire de remplacement direct (minuscule)
        self.replacements = {
            "m.": "monsieur",
            "mme": "madame",
            "mmes": "mesdames",
            "mlle": "mademoiselle",
            "mlles": "mesdemoiselles",
            "dr": "docteur",
            "pr": "professeur",
            "st": "saint",
            "ste": "sainte",
            "etc.": "et cetera",
            "&": "et",
            "+": "plus",
            "=": "égal",
        }
        
        # Acronymes à épeler (ex: S.N.C.F)
        # On ajoute des points pour forcer l'épellation
        self.acronyms = {
            "SNCF": "S.N.C.F",
            "RATP": "R.A.T.P",
            "TGV": "T.G.V",
            "TER": "T.E.R",
            "PDF": "P.D.F",
            "CD": "C.D",
            "DVD": "D.V.D",
            "URL": "U.R.L",
            "HTML": "H.T.M.L",
            "API": "A.P.I",
            "FBI": "F.B.I",
            "CIA": "C.I.A",
            "OVNI": "ovni", # Prononcé comme un mot
            "SIDA": "sida", # Prononcé comme un mot
            "NATO": "nato", # Prononcé comme un mot
            "UNESCO": "unesco",
            "NASA": "naza",
        }
        
        # Mots techniques ou étrangers (phonétisation approximative)
        self.custom_pronunciations = {
            "github": "guite-heube",
            "wifi": "oui-fi",
            "data": "data",
            "design": "dizagne",
            "meeting": "mitingue",
            "marketing": "markétingue",
            "parking": "parkine-gue",
            "weekend": "wikende",
            "week-end": "wikende",
            "newsletter": "niouze-létère",
            "smartphone": "smart-fone",
            "business": "biznesse",
            "challenge": "tchalindje",
            "manager": "manadjeure",
            "management": "manadj-mente",
            "burnout": "beurn-aoute",
            "feedback": "fid-baque",
            "process": "processse",
            "timeline": "taïme-laïne",
            "chat": "tchatte", # Attention au chat (animal) ! Contexte nécessaire idéalement
        }

    def process(self, text: str) -> str:
        """Applique toutes les transformations."""
        text = self._replace_numbers(text)
        text = self._replace_abbreviations(text)
        text = self._handle_acronyms(text)
        text = self._apply_custom_dict(text)
        return text

    def _replace_numbers(self, text: str) -> str:
        """Remplace les nombres par leur équivalent textuel."""
        
        # Fonction callback pour re.sub
        def replace_num(match):
            num_str = match.group(0)
            # Gestion basique des virgules/points
            num_str = num_str.replace(',', '.')
            try:
                val = float(num_str)
                # Si c'est un entier, convertir en int pour éviter "virgule zéro"
                if val.is_integer():
                    return num2words(int(val), lang='fr')
                return num2words(val, lang='fr')
            except ValueError:
                return num_str

        # 1. Années (souvent 4 chiffres, ex: 1995, 2024)
        # On essaie de détecter les années pour éviter "mille neuf cent..." si on préfère "dix-neuf cent" (optionnel)
        # Ici on garde le standard num2words qui est très correct pour le français.

        # 2. Monnaies (ex: 12€, 12,50 $)
        text = re.sub(r'(\d+[.,]?\d*)\s*[€$£]', lambda m: self._convert_currency(m), text)
        
        # 3. Pourcentages
        text = re.sub(r'(\d+[.,]?\d*)\s*%', lambda m: replace_num(m.group(1)) + " pour cent", text)

        # 4. Heures (14h30)
        text = re.sub(r'(\d{1,2})[hH](\d{2})?', lambda m: self._convert_time(m), text)

        # 5. Nombres isolés (reste)
        # On évite de remplacer dans des mots (ex: mp3) -> \b
        text = re.sub(r'\b\d+[.,]?\d*\b', replace_num, text)
        
        return text

    def _convert_currency(self, match) -> str:
        amount = match.group(1).replace(',', '.')
        symbol = match.group(0)[-1]
        currency = "euros"
        if symbol == '$': currency = "dollars"
        if symbol == '£': currency = "livres"
        
        try:
            val = float(amount)
            text = num2words(val, lang='fr')
            # Nettoyage "virgule zéro" pour entiers
            if val.is_integer():
                text = num2words(int(val), lang='fr')
            return f"{text} {currency}"
        except:
            return match.group(0)

    def _convert_time(self, match) -> str:
        hours = int(match.group(1))
        minutes = int(match.group(2)) if match.group(2) else 0
        
        h_text = num2words(hours, lang='fr')
        # Gestion "une heure" vs "un heures" (num2words gère "un")
        if hours == 1: h_text = "une"
        
        res = f"{h_text} heure{'s' if hours > 1 else ''}"
        
        if minutes > 0:
            m_text = num2words(minutes, lang='fr')
            res += f" {m_text}"
            
        return res

    def _replace_abbreviations(self, text: str) -> str:
        """Remplace les abréviations courantes."""
        words = text.split()
        new_words = []
        for word in words:
            lower = word.lower()
            # Nettoyer ponctuation pour check
            clean = re.sub(r'[^\w.]', '', lower)
            
            if clean in self.replacements:
                replacement = self.replacements[clean]
                # Conserver la casse (Majuscule initiale)
                if word[0].isupper():
                    replacement = replacement.capitalize()
                # Remettre la ponctuation (ex: M., -> Monsieur,)
                if word.endswith(',') or word.endswith('?') or word.endswith('!'):
                    replacement += word[-1]
                new_words.append(replacement)
            else:
                new_words.append(word)
        return " ".join(new_words)

    def _handle_acronyms(self, text: str) -> str:
        """Gère les sigles (SNCF -> S.N.C.F)."""
        for acro, replacement in self.acronyms.items():
            # \b pour mot entier uniquement
            text = re.sub(r'\b' + re.escape(acro) + r'\b', replacement, text)
        return text

    def _apply_custom_dict(self, text: str) -> str:
        """Applique le dictionnaire de prononciation."""
        # On trie par longueur décroissante pour remplacer les expressions longues d'abord
        sorted_keys = sorted(self.custom_pronunciations.keys(), key=len, reverse=True)
        
        for key in sorted_keys:
            val = self.custom_pronunciations[key]
            # Case insensitive replace
            pattern = re.compile(r'\b' + re.escape(key) + r'\b', re.IGNORECASE)
            text = pattern.sub(val, text)
            
        return text

if __name__ == "__main__":
    # Test rapide
    prep = FrenchTextPreprocessor()
    tests = [
        "M. Dupont arrive à 14h30.",
        "Cela coûte 12,50€ et j'ai payé 50%.",
        "J'ai vu un OVNI sur GitHub.",
        "Le meeting avec la SNCF est annulé.",
        "En 1998, la France a gagné."
    ]
    
    for t in tests:
        print(f"In : {t}")
        print(f"Out: {prep.process(t)}\n")