"""
Corrections spécifiques pour "Les Conquérants du Pognon".

Ce fichier contient des corrections phonétiques et contextuelles
pour améliorer la qualité de la synthèse vocale.

Usage:
    from corrections_conquerants import apply_corrections
    text = apply_corrections(text)
"""

import re
from typing import Dict, List, Tuple


# =============================================================================
# 1. NOMS PROPRES ET PERSONNAGES
# =============================================================================

NOMS_PERSONNAGES = {
    # Protagoniste principal
    "Victor": "Victor",
    "victor": "Victor",

    # Le caïd - uniformiser sur Kamel
    "Amel": "Kamel",
    "amel": "Kamel",
    "Abel": "Kamel",
    "abel": "Kamel",

    # Le patron texan
    "Crawford": "Crawford",
    "crawford": "Crawford",
    "Kraffor": "Crawford",
    "Traffor": "Crawford",

    # Khan
    "Khan": "Khanne",  # Phonétique pour meilleure prononciation
    "khan": "Khanne",
    "Can": "Khanne",
    "Cannes": "Khanne",
}

# =============================================================================
# 2. LIEUX - FRANCE
# =============================================================================

LIEUX_FRANCE = {
    "Saint Denis": "Saint-Denis",
    "Saint-Denis Ferrier": "Saint-Denis",
    "Champs Élysée": "Champs-Élysées",
    "Champs Elysée": "Champs-Élysées",
    "Champs Elysees": "Champs-Élysées",
}

# =============================================================================
# 3. LIEUX - DUBAÏ ET MOYEN-ORIENT
# =============================================================================

LIEUX_DUBAI = {
    # Dubaï
    "Du bail": "Doubaï",
    "Du bai": "Doubaï",
    "Dubail": "Doubaï",
    "Dubai": "Doubaï",
    "Dubaï": "Doubaï",

    # Quartiers et rues
    "Deral": "Déïra",
    "Deira": "Déïra",
    "Le doff": "Déïra",

    # Sheikh Zayed Road
    "Cheikh Zayel Roal": "Cheikh Zayed Road",
    "Zayiroar": "Cheikh Zayed Road",
    "cheik Zayed": "Cheikh Zayed",
    "Sheikh Zayed": "Cheikh Zayed",

    # Creek (bras de mer)
    "Long de cheik": "le long du Creek",
    "le creek": "le Creek",

    # Golfe Persique
    "Du golf": "du Golfe",
    "du golf": "du Golfe",
    "le golf": "le Golfe",

    # Émirats
    "Zuni": "Émirats Arabes Unis",
    "EAU": "Émirats Arabes Unis",
}

# =============================================================================
# 4. OBJETS ET MARQUES
# =============================================================================

MARQUES_OBJETS = {
    # Cigarettes
    "Cogoises": "Gauloises",
    "cogoises": "Gauloises",
    "Gauloise": "Gauloise",

    # Chaussures
    "Mocassins Gipsy": "mocassins Gucci",
    "mocassins Gipsy": "mocassins Gucci",
    "Loi fer en cuir": "loafers en cuir",
    "loi fer": "loafers",
    "Loafer": "loafer",

    # Voitures
    "Silhouette shadow": "Silver Shadow",
    "silhouette shadow": "Silver Shadow",
    "Silver Shadow": "Silver Shadow",  # Rolls Royce

    # Bateaux
    "Doff": "dhow",
    "doff": "dhow",
    "Dhows": "dhows",
    "dhows": "dhows",  # Boutres traditionnels

    # Autoradios
    "Odoradious": "autoradios",
    "odoradious": "autoradios",

    # Tapis
    "Tapis pécheurs": "tapis persans",
    "tapis pécheurs": "tapis persans",
}

# =============================================================================
# 5. ARGENT ET MONNAIE
# =============================================================================

ARGENT = {
    # Francs
    "20 France": "vingt francs",
    "Tyro franc": "mille francs",
    "L'été de franc": "était de trente francs",
    "500 francs la passé": "cinq cents francs la passe",

    # Chèques de voyage
    "Voyage chèque": "traveller's chèques",
    "voyage chèque": "traveller's chèques",

    # Or
    "10 gramme d'or carra": "dix grammes d'or, vingt-quatre carats",
    "carats": "carats",
}

# =============================================================================
# 6. LOGEMENT ET URBANISME
# =============================================================================

LOGEMENT = {
    "Loyer hum": "loyer HLM",
    "loyer hum": "loyer HLM",
    "Loyer HLM": "loyer HLM",
    "HLM": "HLM",

    "Technique de maître carré": "dix mètres carrés",
    "dix maître carré": "dix mètres carrés",
    "mètre carré": "mètres carrés",

    "Compteurs élèves": "compteurs électriques",
    "compteurs élèves": "compteurs électriques",

    "En fait patte en gris administration": "peinte en gris administration",
    "gris administration": "gris administration",
}

# =============================================================================
# 7. EXPRESSIONS ET MOTS COURANTS
# =============================================================================

EXPRESSIONS = {
    # Famille
    "Catette": "cadette",
    "catette": "cadette",

    # Expressions
    "A veau bon": "À quoi bon",
    "a veau bon": "à quoi bon",

    # Travail/Chômage
    "Jommage télévision": "Chômage, télévision",
    "jommage": "chômage",

    # Poker et jeux
    "Mise minimeur": "mise minimum",
    "mise minimeur": "mise minimum",
    "Tripé sa mise": "triplé sa mise",
    "tripé": "triplé",

    # Argot et rue
    "Baffons": "bas-fonds",
    "baffons": "bas-fonds",
    "Claude de banlieue": "caïd de banlieue",
    "S'importait le trapu": "s'appelait le Trapu",

    # Drogue
    "01 Omane": "héroïnomane",
    "Omane": "héroïnomane",

    # Divers
    "Elias de billets": "liasses de billets",
    "elias": "liasses",
    "Clubs de bops": "clubs de boxe",
    "clubs de bops": "clubs de boxe",
    "Les balles à chaque doigt": "les bagues à chaque doigt",
    "balles": "bagues",  # Attention: contexte important
    "Remonté l'eau grossiste": "remontait au grossiste",
    "Un acide fais-toi": "assieds-toi",
    "Du taque pure": "du tac au tac",
    "Nu for fune": "une fortune",
    "J'avais de bonne cat": "j'avais de bonnes cartes",
    "Ta véc a dal": "t'avais que dalle",
    "Une sevonde": "une seconde",
    "sevonde": "seconde",
    "Fluant le sang frais": "fleurant le sang frais",
    "fluant": "fleurant",
    "Les Clievi": "les clients VIP",
    "Clievi": "clients VIP",
    "La c'est malin": "assez malin",
    "Quand il était sarta": "quand il était certain",
    "sarta": "certain",
    "Il montit": "il mentit",
    "montit": "mentit",
    "Victor vasserra": "Victor serra",
    "vasserra": "serra",
    "Les bureaux des tradir": "les bureaux des traders",
    "tradir": "traders",
    "Tradin": "trading",
    "tradin": "trading",
    "Une solde monsieur": "une seule, monsieur",
    "Pas en prince": "pas un prince",
    "Campin bancal": "lit de camp bancal",
    "campin": "lit de camp",
}

# =============================================================================
# 8. MAQUILLAGE ET APPARENCE
# =============================================================================

APPARENCE = {
    "Yeux soulignés de flou": "yeux soulignés de khôl",
    "soulignés de flou": "soulignés de khôl",
    "de flou": "de khôl",
    "khol": "khôl",
}

# =============================================================================
# 9. PATTERNS REGEX POUR CORRECTIONS CONTEXTUELLES
# =============================================================================

CORRECTIONS_REGEX: List[Tuple[str, str]] = [
    # Nombres + francs
    (r'\b(\d+)\s*France\b', r'\1 francs'),
    (r'\b(\d+)\s*france\b', r'\1 francs'),

    # Heures
    (r'\b(\d+)\s*heure\b', r'\1 heures'),

    # Mètres carrés
    (r'\b(\d+)\s*m2\b', r'\1 mètres carrés'),
    (r'\b(\d+)\s*m²\b', r'\1 mètres carrés'),

    # Grammes d'or
    (r'\b(\d+)\s*grammes?\s*d\'or\b', r'\1 grammes d\'or'),

    # Pourcentages
    (r'\b(\d+)\s*%', r'\1 pour cent'),

    # Prix en euros (si présent)
    (r'\b(\d+)\s*€', r'\1 euros'),
    (r'\b(\d+)\s*euros?\b', r'\1 euros'),
]


# =============================================================================
# FONCTION PRINCIPALE D'APPLICATION DES CORRECTIONS
# =============================================================================

def apply_corrections(text: str) -> str:
    """
    Applique toutes les corrections au texte.

    Args:
        text: Texte original

    Returns:
        Texte corrigé
    """
    if not text:
        return text

    # 1. Corrections simples (dictionnaires)
    all_corrections = {}
    all_corrections.update(NOMS_PERSONNAGES)
    all_corrections.update(LIEUX_FRANCE)
    all_corrections.update(LIEUX_DUBAI)
    all_corrections.update(MARQUES_OBJETS)
    all_corrections.update(ARGENT)
    all_corrections.update(LOGEMENT)
    all_corrections.update(EXPRESSIONS)
    all_corrections.update(APPARENCE)

    # Trier par longueur décroissante pour éviter les remplacements partiels
    sorted_corrections = sorted(all_corrections.items(), key=lambda x: len(x[0]), reverse=True)

    for old, new in sorted_corrections:
        # Utiliser des limites de mots pour éviter les remplacements partiels
        pattern = r'\b' + re.escape(old) + r'\b'
        text = re.sub(pattern, new, text, flags=re.IGNORECASE if old[0].islower() else 0)

    # 2. Corrections regex
    for pattern, replacement in CORRECTIONS_REGEX:
        text = re.sub(pattern, replacement, text)

    return text


def get_phonetic_hints() -> Dict[str, str]:
    """
    Retourne un dictionnaire de hints phonétiques pour le TTS.
    Format: mot -> prononciation phonétique
    """
    return {
        "Crawford": "Craw-ford",
        "Khan": "Khanne",
        "Kamel": "Ka-mel",
        "Sheikh": "Cheikh",
        "Zayed": "Za-yèd",
        "Creek": "Crik",
        "Dubaï": "Dou-baï",
        "Deira": "Dé-i-ra",
        "dhow": "dao",
        "khôl": "kol",
        "HLM": "Hache-èl-ème",
        "VIP": "Vi-aï-pi",
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Tests
    test_phrases = [
        "Victor avait 20 France dans sa poche.",
        "Il fumait des Cogoises sur les Champs Élysée.",
        "Sa Catette habitait dans un Loyer hum.",
        "Crawford l'attendait à Du bail.",
        "Les yeux soulignés de flou, elle portait des Mocassins Gipsy.",
        "Il avait gagné Nu for fune au poker.",
        "Le tradir lui proposa du Tradin.",
        "Long de cheik, les Doff naviguaient.",
    ]

    print("=== Test des corrections ===\n")
    for phrase in test_phrases:
        corrected = apply_corrections(phrase)
        if phrase != corrected:
            print(f"AVANT: {phrase}")
            print(f"APRÈS: {corrected}")
            print()
        else:
            print(f"INCHANGÉ: {phrase}\n")
