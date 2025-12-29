"""
Préprocesseur de texte français pour améliorer la prononciation TTS Kokoro.
Version 9 - Corrections dynamiques via fichiers JSON externes.

Principes:
- SANS tirets (Kokoro les lit littéralement)
- Corrections phonétiques pures
- Protection des acronymes courts en MAJUSCULES
- Chargement dynamique depuis corrections/*.json

Structure corrections/:
- kokoro_fixes.json    : Virgules pour forcer les pauses
- pronunciation.json   : Substitutions phonétiques
- accents.json         : Restauration des accents
- tech_terms.json      : Termes techniques/anglicismes
- user_additions.json  : Corrections utilisateur (prioritaires)
"""

import re
import json
from pathlib import Path
from typing import Dict, Optional


def _find_corrections_dir() -> Optional[Path]:
    """Trouve le dossier corrections/ relatif au projet."""
    # Chercher à partir du fichier actuel
    current = Path(__file__).parent

    # Remonter pour trouver corrections/
    for _ in range(3):
        corrections_dir = current / "corrections"
        if corrections_dir.exists():
            return corrections_dir
        corrections_dir = current.parent / "corrections"
        if corrections_dir.exists():
            return corrections_dir
        current = current.parent

    return None


def _load_json_dict(filepath: Path, key: str) -> Dict[str, str]:
    """Charge un dictionnaire depuis un fichier JSON."""
    if not filepath.exists():
        return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get(key, {})
    except (json.JSONDecodeError, IOError) as e:
        print(f"Attention: Erreur chargement {filepath.name}: {e}")
        return {}


class FrenchTextPreprocessor:
    """
    Préprocesseur pour corriger les mots mal prononcés par Kokoro.

    Charge les corrections depuis corrections/*.json si disponibles,
    sinon utilise les valeurs par défaut hardcodées.
    """

    # Corrections phonétiques par défaut (fallback si JSON non disponible)
    # Basé sur analyse MFCC+DTW (Lisa/ChatGPT - décembre 2024)
    PRONUNCIATION_DICT = {
        # === Mots avec terminaisons problématiques ===

        # -ée/-é finals - maintenant gérés par KOKORO_FIXES avec virgules
        # (les doubles éé ne fonctionnaient pas bien)

        # -ègue: maintenant géré par KOKORO_FIXES avec virgules (co, lègue)

        # Mots avec 'è' ou 'é' intérieur
        "présage": "praizaje",
        "présages": "praizajes",

        # === Confusions phonétiques courantes ===

        # Police - maintenant géré par KOKORO_FIXES avec virgules (po, lice)

        # Coups secs
        "coups secs": "cou sec",
        "coups": "cou",
        "coup": "cou",

        # Verre vs Vert
        "verre d'eau": "vaire deau",
        "verre de": "vaire de",
        "un verre": "un vaire",
        "le verre": "le vaire",
        "son verre": "son vaire",

        # Tour Eiffel - forcer la prononciation
        "Tour Eiffel": "tour Effel",
        "tour Eiffel": "tour Effel",

        # Tout rompre
        "tout rompre": "tourompre",

        # s'annonçaient - le ç problématique
        "s'annonçaient": "sannonssaient",
        "annonçaient": "annonssaient",

        # === Mots validés MFCC+DTW (Lisa) - SANS tirets ===

        "chose": "chauze",
        "choses": "chauzes",
        "quelque chose": "quelque chauze",
        "canapé": "cannapé",
        "canapés": "cannapés",
        "chérie": "chéri",
        "chéri": "chéri",
        "figée": "figé",
        "figé": "figé",
        "fenêtre": "feunaitre",
        "fenêtres": "feunaitres",

        # === Variantes sans accents (PDF/EPUB) ===
        "canape": "cannapé",
        "cherie": "chéri",
        "cheri": "chéri",
        "figee": "figé",
        "fenetre": "feunaitre",
        "presage": "praizaje",
        "collegue": "collègue",
        # Note: vérité, qualité, etc. gérés par KOKORO_FIXES

        # Corrections supplémentaires
        "il se passe": "il se passe",
        "il s'est passé": "il sé passé",
        "Détruis tout": "Détruit tout",
    }

    # Ordinaux (1er, 2e, 5ème, etc.)
    ORDINALS = {
        "1er": "premier",
        "1ere": "première",
        "1ère": "première",
        "2e": "deuxième",
        "2eme": "deuxième",
        "2ème": "deuxième",
        "3e": "troisième",
        "3eme": "troisième",
        "3ème": "troisième",
        "4e": "quatrième",
        "5e": "cinquième",
        "6e": "sixième",
        "7e": "septième",
        "8e": "huitième",
        "9e": "neuvième",
        "10e": "dixième",
        "11e": "onzième",
        "12e": "douzième",
        "13e": "treizième",
        "14e": "quatorzième",
        "15e": "quinzième",
        "16e": "seizième",
        "17e": "dix-septième",
        "18e": "dix-huitième",
        "19e": "dix-neuvième",
        "20e": "vingtième",
        "21e": "vingt-et-unième",
    }

    # Restauration des accents (textes ASCII sans accents)
    ACCENT_RESTORE = {
        # Verbes être/avoir
        "etait": "était",
        "etaient": "étaient",
        "ete": "été",
        "etes": "êtes",

        # Mots courants
        "tempete": "tempête",
        "fenetre": "fenêtre",
        "fenetres": "fenêtres",
        "tete": "tête",
        "fete": "fête",
        "bete": "bête",
        "foret": "forêt",
        "interet": "intérêt",
        "arret": "arrêt",
        "pret": "prêt",
        "enquete": "enquête",

        # è
        "pere": "père",
        "mere": "mère",
        "frere": "frère",
        "lumiere": "lumière",
        "maniere": "manière",
        "matiere": "matière",
        "riviere": "rivière",
        "premiere": "première",
        "derniere": "dernière",
        "derriere": "derrière",
        "carriere": "carrière",

        # é
        "eclair": "éclair",
        "eclairs": "éclairs",
        "ecoute": "écoute",
        "ecoutez": "écoutez",
        "echappa": "échappa",
        "etrange": "étrange",
        "etrangement": "étrangement",
        "etranger": "étranger",
        "ecran": "écran",
        "epaule": "épaule",
        "epouse": "épouse",
        "epreuve": "épreuve",
        "equipe": "équipe",
        "eternel": "éternel",
        "etude": "étude",
        "evenement": "événement",
        "evidence": "évidence",
        "evolution": "évolution",

        # œ/oe
        "coeur": "cœur",
        "soeur": "sœur",
        "oeuvre": "œuvre",
        "oeil": "œil",
        "voeu": "vœu",

        # ç
        "ca": "ça",
        "francais": "français",
        "francaise": "française",
        "garcon": "garçon",
        "lecon": "leçon",
        "facon": "façon",
        "recoit": "reçoit",
        "recu": "reçu",
        "decu": "déçu",
        "commenca": "commença",
        "avanca": "avança",
        "lanca": "lança",
        "s'annoncaient": "s'annonçaient",
        "annoncaient": "annonçaient",

        # Mots spécifiques du texte
        "rentre": "rentré",
        "ruisselaient": "ruisselaient",
        "insensee": "insensée",
        "brumeuse": "brumeuse",
        "resultats": "résultats",
        "resultat": "résultat",
        "decouvrir": "découvrir",
        "decouvert": "découvert",
        "verite": "vérité",
        "realite": "réalité",
        "securite": "sécurité",
        "intensite": "intensité",
        "presage": "présage",
        "collegue": "collègue",
        "redoublait": "redoublait",

        # Lisa additions
        "tres": "très",
        "deja": "déjà",
        "apres": "après",
        "aout": "août",
        "noel": "Noël",
        "naif": "naïf",
        "naive": "naïve",
        "medecin": "médecin",
        "medecine": "médecine",
        "interieur": "intérieur",
        "exterieur": "extérieur",
        "developpeur": "développeur",
        "developpement": "développement",
        "energie": "énergie",
        "economie": "économie",
        "sante": "santé",
    }

    # Abréviations et sigles courants
    ABBREVIATIONS = {
        "M.": "Monsieur",
        "Mme": "Madame",
        "Mlle": "Mademoiselle",
        "Dr": "Docteur",
        "Dr.": "Docteur",
        "Pr": "Professeur",
        "Pr.": "Professeur",
        "St": "Saint",
        "Ste": "Sainte",
        "etc.": "et cetera",
        "etc": "et cetera",
        "n°": "numéro",
        "N°": "numéro",
        "&": "et",
        "RDV": "rendez-vous",
        "rdv": "rendez-vous",
        "TV": "télé",
        "OK": "okay",
        "ok": "okay",
    }

    # Nombres en lettres (pour "Chapitre 7" -> "Chapitre, sept")
    # Virgule pour forcer la pause et éviter confusion avec "c'est"
    NUMBERS_WORDS = {
        "1": "un", "2": "deux", "3": "trois", "4": "quatre", "5": "cinq",
        "6": "six", "7": ", sept", "8": "huit", "9": "neuf", "10": "dix",
        "11": "onze", "12": "douze", "13": "treize", "14": "quatorze", "15": "quinze",
        "16": "seize", "17": "dix, sept", "18": "dix huit", "19": "dix neuf", "20": "vingt",
    }

    # Mots souvent mal lus par Kokoro - Version 8
    # TECHNIQUE: virgules pour forcer les pauses (testé et validé!)
    KOKORO_FIXES = {
        # === Mots tronqués - VIRGULES pour forcer articulation ===
        "invité": "invi, té",
        "invitée": "invi, tée",
        "invités": "invi, tés",
        "collègue": "co, lègue",
        "collègues": "co, lègues",
        "intensité": "inten, sité",
        "d'intensité": "d'inten, sité",  # garder apostrophe!
        "qualité": "qua, lité",
        "vérité": "véri, té",
        "réalité": "réa, lité",
        "sécurité": "sécu, rité",
        "identité": "iden, tité",

        # === Chapitre sept ===
        "Chapitre Sept": "Chapitre, sept",
        "Chapitre sept": "Chapitre, sept",
        "chapitre sept": "chapitre, sept",

        # === Police - virgule aide P/B ===
        "Police": "Po, lice",
        "police": "po, lice",

        # === Découvert ===
        "découvert": "dé, couvert",
        "découvrir": "dé, couvrir",
        "a découvert": "a, découvert",

        # === Autres mots problématiques ===
        "ruisselaient": "ruisse, laient",
        "s'annonçaient": "s'annon, çaient",
        "coups secs": "coups, secs",
        "trois coups secs": "trois coups, secs",

        # === Tour Eiffel ===
        "Tour Eiffel": "Tour, Eiffel",
        "tour Eiffel": "tour, Eiffel",

        # === Détruis tout ===
        "Détruis tout": "Dé, truis tout",

        # === Sons "ill" - virgules ===
        "fille": "fi, lle",
        "filles": "fi, lles",
        "famille": "fami, lle",
        "familles": "fami, lles",
        "travail": "trava, il",
        "travaille": "trava, ille",
        "soleil": "sole, il",
        "oreille": "ore, ille",
        "oreilles": "ore, illes",
        "bouteille": "boute, ille",
        "bouteilles": "boute, illes",
        "feuille": "feu, ille",
        "feuilles": "feu, illes",
        "vieille": "vie, ille",
        "vieilles": "vie, illes",
        "sommeil": "somme, il",
        "conseil": "conse, il",
        "conseils": "conse, ils",
        "appareil": "appare, il",
        "appareils": "appare, ils",
        "merveille": "merve, ille",
        "merveilles": "merve, illes",
    }

    # Termes techniques et anglicismes (Lisa - text_processor.py)
    TECH_TERMS = {
        # Technologie
        "API": "A P I",
        "APIs": "A P I S",
        "URL": "U R L",
        "HTML": "ache té aime aile",
        "CSS": "cé aisse aisse",
        "JSON": "jésonne",
        "SQL": "aisse ku aile",
        "GitHub": "Guite Hub",
        "ChatGPT": "Tchatte Dji Pi Ti",
        "GPT": "Dji Pi Ti",
        "CPU": "cé pé u",
        "GPU": "gé pé u",
        "RAM": "ramme",
        "SSD": "aisse aisse dé",
        "USB": "u aisse bé",
        "WiFi": "ouifi",
        "Bluetooth": "Blou tousse",
        "iPhone": "aïe faune",
        "iPad": "aïe pad",
        "macOS": "mac O S",
        "iOS": "aïe O S",
        "Linux": "Linuxe",
        "Windows": "Ouinedoze",
        "Python": "Païtone",
        "JavaScript": "Java scripte",

        # Entreprises
        "Google": "Gougueule",
        "Microsoft": "Maïcro softe",
        "Amazon": "Amazone",
        "Netflix": "Nète flixe",
        "Spotify": "Spoti faï",
        "Tesla": "Tèsla",
        "WhatsApp": "Ouats app",
        "Instagram": "Insta gramme",
        "Facebook": "Faisse bouk",
        "Twitter": "Touiteur",
        "YouTube": "You tioube",
        "TikTok": "Tik Tok",

        # Expressions anglaises courantes
        "email": "imèle",
        "emails": "imèles",
        "newsletter": "niouze lèteur",
        "deadline": "dèd laïne",
        "feedback": "fid bak",
        "hashtag": "ache tague",
        "startup": "starte eup",
        "podcast": "pod kaste",
        "streaming": "stri mingue",
        "online": "one laïne",
        "offline": "offe laïne",
        "marketing": "markétingue",
        "manager": "mana jeur",
        "business": "biznesse",
        "meeting": "mitingue",
        "design": "dizaïne",
        "designer": "dizaïneur",
        "freelance": "fri lance",
        "branding": "bran dingue",
        "coaching": "ko tchingue",
        "shopping": "cho pingue",
        "timing": "taï mingue",
        "planning": "pla ningue",
        "briefing": "bri fingue",
        "parking": "par kingue",
        "camping": "kam pingue",
        "jogging": "jo guingue",
        "footing": "fou tingue",
        "week-end": "oui kènde",
        "weekend": "oui kènde",

        # Unités
        "km/h": "kilomètres heure",
        "m/s": "mètres par seconde",
        "°C": "degrés Celsius",
        "°F": "degrés Fahrenheit",
    }

    # Apostrophes perdues en extraction PDF/EPUB (Lisa)
    APOSTROPHE_RESTORE = {
        "aujourdhui": "aujourd'hui",
        "daccord": "d'accord",
        "jusqua": "jusqu'à",
        "jusquau": "jusqu'au",
        "jusquici": "jusqu'ici",
        "quelquun": "quelqu'un",
        "quelquune": "quelqu'une",
        "presquile": "presqu'île",
        "lorsquil": "lorsqu'il",
        "lorsquelle": "lorsqu'elle",
        "puisquil": "puisqu'il",
        "puisquelle": "puisqu'elle",
        "quoiquil": "quoiqu'il",
        "quoiquelle": "quoiqu'elle",
    }

    def __init__(self, corrections_dir: Optional[Path] = None):
        """
        Initialise le préprocesseur.

        Args:
            corrections_dir: Chemin vers le dossier corrections/ (auto-détecté si None)
        """
        self._time_pattern = re.compile(r'\b(\d{1,2})h(\d{2})?\b')
        self._ordinal_pattern = re.compile(r'\b(\d{1,2})(er|ere|ère|e|eme|ème)\b', re.IGNORECASE)
        self._chapter_pattern = re.compile(r'\b(Chapitre|CHAPITRE|chapitre)\s+(\d{1,2})\b')

        # Charger les corrections depuis JSON (ou utiliser les valeurs par défaut)
        self._corrections_dir = corrections_dir or _find_corrections_dir()
        self._load_corrections()

        # Trier par longueur décroissante pour éviter les remplacements partiels
        self._sorted_dict_items = sorted(
            self._pronunciation_dict.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        self._sorted_abbrev_items = sorted(
            self.ABBREVIATIONS.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        self._sorted_accent_items = sorted(
            self._accent_restore.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        self._sorted_kokoro_fixes = sorted(
            self._kokoro_fixes.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        self._sorted_tech_terms = sorted(
            self._tech_terms.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        self._sorted_apostrophe = sorted(
            self.APOSTROPHE_RESTORE.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )

    def _load_corrections(self):
        """Charge les corrections depuis les fichiers JSON."""
        # Valeurs par défaut (class-level constants)
        self._pronunciation_dict = dict(self.PRONUNCIATION_DICT)
        self._accent_restore = dict(self.ACCENT_RESTORE)
        self._kokoro_fixes = dict(self.KOKORO_FIXES)
        self._tech_terms = dict(self.TECH_TERMS)

        if not self._corrections_dir:
            return

        # Charger depuis JSON
        json_pronunciation = _load_json_dict(
            self._corrections_dir / "pronunciation.json", "substitutions"
        )
        json_accents = _load_json_dict(
            self._corrections_dir / "accents.json", "restore"
        )
        json_kokoro = _load_json_dict(
            self._corrections_dir / "kokoro_fixes.json", "fixes"
        )
        json_tech = _load_json_dict(
            self._corrections_dir / "tech_terms.json", "terms"
        )

        # Charger les ajouts utilisateur (prioritaires)
        user_kokoro = _load_json_dict(
            self._corrections_dir / "user_additions.json", "kokoro_fixes"
        )
        user_pronunciation = _load_json_dict(
            self._corrections_dir / "user_additions.json", "pronunciation"
        )

        # Fusionner: défauts < JSON < utilisateur
        if json_pronunciation:
            self._pronunciation_dict = json_pronunciation
        self._pronunciation_dict.update(user_pronunciation)

        if json_accents:
            self._accent_restore = json_accents

        if json_kokoro:
            self._kokoro_fixes = json_kokoro
        self._kokoro_fixes.update(user_kokoro)

        if json_tech:
            self._tech_terms = json_tech

    def reload_corrections(self):
        """Recharge les corrections depuis les fichiers JSON."""
        self._load_corrections()
        # Recréer les listes triées
        self._sorted_dict_items = sorted(
            self._pronunciation_dict.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        self._sorted_accent_items = sorted(
            self._accent_restore.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        self._sorted_kokoro_fixes = sorted(
            self._kokoro_fixes.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        self._sorted_tech_terms = sorted(
            self._tech_terms.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )

    def add_correction(self, word: str, replacement: str, category: str = "kokoro_fixes") -> bool:
        """
        Ajoute une correction à user_additions.json.

        Args:
            word: Mot à corriger
            replacement: Remplacement
            category: "kokoro_fixes" ou "pronunciation"

        Returns:
            True si succès
        """
        if not self._corrections_dir:
            print("Erreur: Dossier corrections/ non trouvé")
            return False

        user_file = self._corrections_dir / "user_additions.json"

        try:
            # Charger le fichier existant
            if user_file.exists():
                with open(user_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = {"kokoro_fixes": {}, "pronunciation": {}}

            # Ajouter la correction
            if category not in data:
                data[category] = {}
            data[category][word] = replacement

            # Sauvegarder
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Recharger
            self.reload_corrections()
            print(f"✅ Correction ajoutée: '{word}' → '{replacement}'")
            return True

        except (json.JSONDecodeError, IOError) as e:
            print(f"Erreur sauvegarde: {e}")
            return False

    def get_corrections_info(self) -> dict:
        """Retourne des informations sur les corrections chargées."""
        return {
            "corrections_dir": str(self._corrections_dir) if self._corrections_dir else None,
            "pronunciation_count": len(self._pronunciation_dict),
            "kokoro_fixes_count": len(self._kokoro_fixes),
            "accent_restore_count": len(self._accent_restore),
            "tech_terms_count": len(self._tech_terms),
        }

    def _number_to_french(self, n: int) -> str:
        """Convertit un nombre en français."""
        units = ['', 'un', 'deux', 'trois', 'quatre', 'cinq', 'six', 'sept', 'huit', 'neuf',
                'dix', 'onze', 'douze', 'treize', 'quatorze', 'quinze', 'seize', 'dix-sept',
                'dix-huit', 'dix-neuf']
        tens = ['', '', 'vingt', 'trente', 'quarante', 'cinquante', 'soixante',
               'soixante', 'quatre-vingt', 'quatre-vingt']

        if n < 20:
            return units[n]
        if n < 100:
            t, u = divmod(n, 10)
            if t == 7 or t == 9:
                u += 10
                t -= 1
            if u == 0:
                return tens[t] + ('s' if t == 8 else '')
            elif u == 1 and t not in [8, 9]:
                return tens[t] + ' et un'
            else:
                return tens[t] + '-' + units[u]
        return str(n)

    def _number_to_ordinal(self, n: int, feminine: bool = False) -> str:
        """Convertit un nombre en ordinal français."""
        if n == 1:
            return "première" if feminine else "premier"

        base = self._number_to_french(n)
        # Règles spéciales
        if base.endswith('e'):
            base = base[:-1]
        elif base.endswith('q'):  # cinq
            base = base + 'u'
        elif base.endswith('f'):  # neuf
            base = base[:-1] + 'v'

        return base + "ième"

    def convert_times(self, text: str) -> str:
        """Convertit les heures (19h30 -> dix-neuf heures trente)."""
        def time_to_french(match):
            hours = int(match.group(1))
            minutes = match.group(2)

            hour_words = self._number_to_french(hours)
            result = f"{hour_words} heure" + ("s" if hours > 1 else "")

            if minutes:
                min_val = int(minutes)
                if min_val > 0:
                    result += f" {self._number_to_french(min_val)}"

            return result

        return self._time_pattern.sub(time_to_french, text)

    def convert_ordinals(self, text: str) -> str:
        """Convertit les ordinaux (5e -> cinquième)."""
        # D'abord les ordinaux connus
        for ordinal, replacement in self.ORDINALS.items():
            pattern = re.compile(rf'\b{re.escape(ordinal)}\b', re.IGNORECASE)
            text = pattern.sub(replacement, text)

        # Puis les ordinaux génériques avec regex
        def ordinal_to_french(match):
            num = int(match.group(1))
            suffix = match.group(2).lower()
            feminine = suffix in ['ere', 'ère']
            return self._number_to_ordinal(num, feminine)

        text = self._ordinal_pattern.sub(ordinal_to_french, text)
        return text

    def convert_abbreviations(self, text: str) -> str:
        """Convertit les abréviations courantes."""
        for abbrev, replacement in self._sorted_abbrev_items:
            # Pour les abréviations avec point, escape le point
            pattern = re.compile(rf'\b{re.escape(abbrev)}(?=\s|$|[,;:!?])', re.IGNORECASE)
            text = pattern.sub(replacement, text)
        return text

    def fix_eiffel(self, text: str) -> str:
        """Corrige 'Tour Eiffel' et similaires."""
        # "Tour FL" ou "tour F L" -> "Tour Eiffel"
        text = re.sub(r'\b[Tt]our\s+[Ff]\s*[Ll]\b', 'Tour Eiffel', text)
        text = re.sub(r'\b[Tt]our\s+[Ee]iffel\b', 'Tour Eiffel', text)
        return text

    def restore_accents(self, text: str) -> str:
        """Restaure les accents sur les mots français courants."""
        for word, replacement in self._sorted_accent_items:
            pattern = re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)
            text = pattern.sub(replacement, text)
        return text

    def convert_chapter_numbers(self, text: str) -> str:
        """Convertit 'Chapitre 7' en 'Chapitre Sept'."""
        def chapter_to_words(match):
            prefix = match.group(1)
            num = match.group(2)
            word = self.NUMBERS_WORDS.get(num, num)
            # Capitalize first letter
            return f"{prefix} {word.capitalize()}"

        return self._chapter_pattern.sub(chapter_to_words, text)

    def apply_pronunciation_dict(self, text: str) -> str:
        """Applique les corrections de prononciation."""
        for word, replacement in self._sorted_dict_items:
            pattern = re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)
            text = pattern.sub(replacement, text)
        return text

    def apply_tech_terms(self, text: str) -> str:
        """Applique les corrections pour termes techniques et anglicismes."""
        for word, replacement in self._sorted_tech_terms:
            # Protection: ne pas remplacer si le mot est court et en MAJUSCULES
            # (évite de casser des acronymes comme "CA" pour chiffre d'affaires)
            if len(word) <= 3:
                pattern = re.compile(rf'\b{re.escape(word)}\b')
                text = pattern.sub(
                    lambda m, repl=replacement: m.group(0) if m.group(0).isupper() and word.lower() != word else repl,
                    text
                )
            else:
                pattern = re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)
                text = pattern.sub(replacement, text)
        return text

    def restore_apostrophes(self, text: str) -> str:
        """Restaure les apostrophes perdues en extraction PDF/EPUB."""
        for word, replacement in self._sorted_apostrophe:
            pattern = re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)
            text = pattern.sub(replacement, text)
        return text

    def apply_kokoro_fixes(self, text: str) -> str:
        """Applique les corrections spécifiques Kokoro (dernière étape)."""
        for word, replacement in self._sorted_kokoro_fixes:
            pattern = re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)
            text = pattern.sub(replacement, text)
        return text

    def process(self, text: str) -> str:
        """Traitement complet du texte (v6 - fusion Lisa)."""
        # 1. Restauration des accents (textes ASCII)
        text = self.restore_accents(text)

        # 2. Restauration des apostrophes (textes PDF/EPUB)
        text = self.restore_apostrophes(text)

        # 3. Corrections spécifiques (Eiffel, etc.)
        text = self.fix_eiffel(text)

        # 4. Numéros de chapitres (Chapitre 7 -> Chapitre Sept)
        text = self.convert_chapter_numbers(text)

        # 5. Abréviations
        text = self.convert_abbreviations(text)

        # 6. Ordinaux (5e -> cinquième)
        text = self.convert_ordinals(text)

        # 7. Heures (19h30 -> dix-neuf heures trente)
        text = self.convert_times(text)

        # 8. Termes techniques et anglicismes
        text = self.apply_tech_terms(text)

        # 9. Corrections de prononciation française
        text = self.apply_pronunciation_dict(text)

        # 10. Corrections Kokoro spécifiques (dernière étape)
        text = self.apply_kokoro_fixes(text)

        return text


def preprocess_french_text(text: str) -> str:
    """Fonction utilitaire."""
    return FrenchTextPreprocessor().process(text)


if __name__ == "__main__":
    p = FrenchTextPreprocessor()
    tests = [
        # Ordinaux
        "le 5e arrondissement",
        "la 1ère fois",
        "au 21e siècle",

        # Heures
        "Il était 19h30",

        # Abréviations
        "M. Dupont et Mme Martin",
        "le Dr. House",

        # Mots problématiques
        "trois années",
        "mon collègue",
        "un verre d'eau",
        "la police",
        "des coups secs",
        "un mauvais présage",
        "l'intensité",
        "la vérité",
        "il était invité",

        # Mots validés Lisa
        "sur le canapé",
        "ma chérie",
        "une chose étrange",
        "par la fenêtre",
        "resta figée",

        # Kokoro fixes
        "Les larmes ruisselaient",
        "il a découvert",
        "Que se passe-t-il",
        "des coups secs",

        # Termes techniques (Lisa)
        "J'utilise Python et JavaScript",
        "Le WiFi de Google",
        "Un email de Netflix",
        "Le meeting était online",

        # Apostrophes perdues
        "aujourdhui daccord",
        "jusqua demain",

        # Sons "ill" (Lisa)
        "ma fille et ma famille",
        "le soleil brille",
        "une bouteille de vin",
    ]

    print("=== Test du préprocesseur français (v6 - fusion Lisa) ===\n")
    for t in tests:
        result = p.process(t)
        if result != t:
            print(f"  {t}")
            print(f"  → {result}\n")
        else:
            print(f"  {t} (inchangé)\n")
