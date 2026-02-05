"""
Dictionnaire de prénoms français pour la détection de personnages.

Source: Prénoms courants INSEE + prénoms littéraires fréquents.
"""

# Prénoms masculins français courants (~500+)
FRENCH_NAMES_MALE = {
    # A
    "Aaron", "Abel", "Abraham", "Achille", "Adam", "Adel", "Adrien", "Ahmed",
    "Aïdan", "Alain", "Alan", "Albert", "Albin", "Aldric", "Alex", "Alexandre",
    "Alexis", "Alfred", "Ali", "Alix", "Aloïs", "Alphonse", "Amaury", "Amin",
    "Anatole", "André", "Ange", "Anthony", "Antoine", "Antonin", "Apollinaire",
    "Archibald", "Aristide", "Armand", "Arnaud", "Arthur", "Aston", "Aubry",
    "Auguste", "Augustin", "Aurèle", "Aurélien", "Axel", "Ayoub",
    # B
    "Baptiste", "Barnabé", "Barthélemy", "Basile", "Bastien", "Benjamin",
    "Benoît", "Bernard", "Bertrand", "Billy", "Boris", "Brandon", "Brian",
    "Brice", "Bruno", "Bryan",
    # C
    "Camille", "Carl", "Carlos", "Cédric", "César", "Charles", "Charly",
    "Christian", "Christophe", "Claude", "Clément", "Colin", "Côme", "Conrad",
    "Constant", "Constantin", "Corentin", "Cyril", "Cyrille",
    # D
    "Damien", "Daniel", "David", "Denis", "Didier", "Dimitri", "Dominique",
    "Dorian", "Dylan",
    # E
    "Eddy", "Edgar", "Édouard", "Elie", "Éliot", "Élouan", "Émeric", "Émile",
    "Émilien", "Emmanuel", "Enzo", "Éric", "Ernest", "Erwan", "Éthan", "Étienne",
    "Eugène", "Evan", "Ewan",
    # F
    "Fabien", "Fabrice", "Faustin", "Félix", "Ferdinand", "Fernand", "Firmin",
    "Flavien", "Florent", "Florian", "Francis", "Franck", "François",
    "Frédéric",
    # G
    "Gabin", "Gabriel", "Gaël", "Gaspard", "Gaston", "Gauthier", "Gautier",
    "Geoffrey", "Georges", "Gérald", "Gérard", "Germain", "Gilbert", "Gilles",
    "Gonzague", "Grégoire", "Grégory", "Guillaume", "Gustave", "Guy",
    # H
    "Hadrien", "Hector", "Henri", "Hervé", "Hippolyte", "Hubert", "Hugo",
    # I
    "Ibrahim", "Ignace", "Igor", "Ilan", "Isaac", "Isidore", "Ivan", "Iwan",
    # J
    "Jack", "Jacques", "James", "Janvier", "Jason", "Jean", "Jérémy", "Jérôme",
    "Jessé", "Jimmy", "Joël", "Johan", "Jonathan", "Jordan", "Joseph", "Josh",
    "Joshua", "Jules", "Julien", "Justin",
    # K
    "Kamel", "Karl", "Kévin", "Kilian", "Killian",
    # L
    "Lambert", "Laurent", "Lazare", "Léandre", "Léo", "Léon", "Léonard",
    "Léopold", "Lionel", "Loïc", "Lorenzo", "Loric", "Louis", "Louka", "Loup",
    "Luc", "Lucas", "Lucien", "Ludovic", "Luigi", "Luka",
    # M
    "Maël", "Malo", "Manfred", "Manuel", "Marc", "Marcel", "Marcelin", "Marco",
    "Marius", "Martin", "Mathias", "Mathieu", "Mathis", "Mattéo", "Matthieu",
    "Maurice", "Max", "Maxence", "Maxime", "Maximilien", "Médéric", "Melvin",
    "Michael", "Michel", "Mickaël", "Mohamed", "Morgan", "Moussa",
    # N
    "Nathan", "Nathanaël", "Nicolas", "Noé", "Noël", "Norbert", "Norman",
    # O
    "Octave", "Odilon", "Olivier", "Omar", "Oscar", "Othman", "Oumar", "Owen",
    # P
    "Pablo", "Pascal", "Patrice", "Patrick", "Paul", "Paulin", "Perceval",
    "Philippe", "Pierre", "Pierrick", "Prosper",
    # Q
    "Quentin",
    # R
    "Rachid", "Raoul", "Raphaël", "Raymond", "Régis", "Rémi", "Rémy", "Renaud",
    "René", "Richard", "Robert", "Robin", "Roch", "Rodrigue", "Roger", "Roland",
    "Romain", "Romuald", "Ronan", "Ruben",
    # S
    "Sacha", "Saïd", "Salim", "Samson", "Samuel", "Samy", "Sébastien", "Serge",
    "Simon", "Sofiane", "Stanislas", "Stéphane", "Steve", "Steven", "Sylvain",
    "Sylvestre",
    # T
    "Tanguy", "Téo", "Théo", "Théodore", "Théophile", "Thibaud", "Thibault",
    "Thibaut", "Thierry", "Thomas", "Timothée", "Titouan", "Tom", "Tristan",
    # U
    "Ugo", "Ulysse",
    # V
    "Valentin", "Valéry", "Victor", "Vincent", "Virgile", "Vladimir",
    # W
    "Warren", "Wilfried", "William", "Wilson",
    # X
    "Xavier", "Xénon",
    # Y
    "Yacine", "Yann", "Yannis", "Yanis", "Yoann", "Yohan", "Youssef", "Yvan",
    "Yves", "Yvon",
    # Z
    "Zacharie", "Zéphir", "Zéphirin",
}

# Prénoms féminins français courants (~500+)
FRENCH_NAMES_FEMALE = {
    # A
    "Adèle", "Adeline", "Adélaïde", "Adrienne", "Agathe", "Agnès", "Aïcha",
    "Aïda", "Albane", "Alberta", "Alexandrine", "Alice", "Aline", "Alix",
    "Aloïse", "Alphonsine", "Amandine", "Ambre", "Amélie", "Anaëlle", "Anaïs",
    "Andrée", "Angèle", "Angélique", "Anna", "Annabelle", "Anne", "Annette",
    "Anouk", "Antoinette", "Apolline", "Ariane", "Arianne", "Armelle",
    "Audrey", "Augustine", "Aurélie", "Aurore", "Axelle",
    # B
    "Barbara", "Béatrice", "Bénédicte", "Bérénice", "Bernadette", "Berthe",
    "Blandine", "Brigitte",
    # C
    "Camille", "Candice", "Capucine", "Carine", "Carla", "Caroline", "Cassandra",
    "Catherine", "Cécile", "Céleste", "Célia", "Céline", "Chantal", "Charlène",
    "Charlotte", "Chloé", "Christelle", "Christine", "Claire", "Clara",
    "Clarisse", "Claudette", "Claudine", "Clémence", "Clémentine", "Clotilde",
    "Colette", "Constance", "Coraline", "Corinne", "Cosette",
    # D
    "Danielle", "Daphné", "Delphine", "Denise", "Diane", "Dominique", "Doriane",
    "Dorine", "Dorothée",
    # E
    "Édith", "Edmonde", "Éléonore", "Éliane", "Élisa", "Élisabeth", "Élise",
    "Ella", "Élodie", "Éloïse", "Émeline", "Émérence", "Émilie", "Emma",
    "Emmanuelle", "Ernestine", "Estelle", "Esther", "Eugénie", "Eulalie",
    "Éva", "Ève", "Evelyne",
    # F
    "Fabienne", "Fanny", "Fatima", "Félicie", "Fernande", "Fleur", "Flore",
    "Florence", "Florentine", "Francine", "Françoise",
    # G
    "Gabrielle", "Gaëlle", "Geneviève", "Georgette", "Germaine", "Gertrude",
    "Gisèle", "Gwenaëlle",
    # H
    "Hélène", "Héloïse", "Henriette", "Hortense", "Huguette",
    # I
    "Inès", "Ingrid", "Irène", "Iris", "Isabelle", "Isadora", "Ismérie", "Ivy",
    # J
    "Jacqueline", "Jade", "Janine", "Jeanne", "Jeannette", "Jeannine",
    "Jennifer", "Jessica", "Jocelyne", "Joëlle", "Joséphine", "Josette",
    "Josiane", "Julie", "Juliette", "Justine",
    # K
    "Karen", "Karine", "Katia", "Kelly",
    # L
    "Laëtitia", "Laure", "Laurence", "Lauriane", "Laurie", "Laurine", "Léa",
    "Léonie", "Léontine", "Liliane", "Lilly", "Lilou", "Lily", "Line", "Lisa",
    "Lise", "Lisette", "Lola", "Lorraine", "Lou", "Louise", "Luce", "Lucette",
    "Lucie", "Lucienne", "Lucile", "Lucille", "Ludivine", "Luna", "Lydie",
    # M
    "Madeleine", "Maëlle", "Maëlys", "Magali", "Magalie", "Mahaut", "Maïa",
    "Maïssa", "Manon", "Marceline", "Margaux", "Margot", "Marguerite",
    "Marianne", "Marie", "Marielle", "Marine", "Marion", "Marlène", "Marthe",
    "Martine", "Mathilde", "Maude", "Mauricette", "Maxine", "Mélanie", "Mélina",
    "Mélissa", "Michèle", "Micheline", "Mireille", "Monique", "Morgane",
    "Muriel", "Myriam",
    # N
    "Nadège", "Nadine", "Nadia", "Naïs", "Natacha", "Nathalie", "Nelly",
    "Nicole", "Nina", "Ninon", "Noémie",
    # O
    "Océane", "Odette", "Odile", "Olivia", "Olympe", "Ophélie",
    # P
    "Paola", "Pascale", "Patricia", "Paulette", "Pauline", "Pénélope",
    "Perrine", "Pétronille", "Philippine", "Priscilla", "Prudence",
    # R
    "Rachel", "Raymonde", "Rébecca", "Régine", "Renée", "Rita", "Roberte",
    "Romane", "Rose", "Roseline", "Rosette", "Rosine", "Roxane", "Ruth",
    # S
    "Sabine", "Sabrina", "Salomé", "Sandra", "Sandrine", "Sarah", "Séverine",
    "Sidonie", "Simone", "Sofia", "Solange", "Solène", "Sonia", "Sophie",
    "Stéphanie", "Suzanne", "Suzette", "Sylviane", "Sylvie",
    # T
    "Tatiana", "Tessa", "Thérèse", "Tiphaine",
    # U
    "Ursule",
    # V
    "Valentine", "Valérie", "Vanessa", "Véronique", "Victoire", "Victoria",
    "Victorine", "Vinciane", "Violaine", "Violette", "Virginie", "Viviane",
    "Vivienne",
    # W
    "Wendy",
    # Y
    "Yasmine", "Yolande", "Yvette", "Yvonne",
    # Z
    "Zahra", "Zoé", "Zora",
}

# Combinaison pour recherche rapide (insensible à la casse)
FRENCH_NAMES_MALE_LOWER = {name.lower() for name in FRENCH_NAMES_MALE}
FRENCH_NAMES_FEMALE_LOWER = {name.lower() for name in FRENCH_NAMES_FEMALE}
ALL_FRENCH_NAMES_LOWER = FRENCH_NAMES_MALE_LOWER | FRENCH_NAMES_FEMALE_LOWER


def is_french_name(name: str) -> bool:
    """
    Vérifie si le nom est un prénom français connu.

    Args:
        name: Le nom à vérifier

    Returns:
        True si c'est un prénom français reconnu
    """
    if not name:
        return False
    return name.lower().strip() in ALL_FRENCH_NAMES_LOWER


def get_gender_from_name(name: str) -> str | None:
    """
    Détermine le genre probable d'un prénom français.

    Args:
        name: Le prénom à analyser

    Returns:
        "M" pour masculin, "F" pour féminin, None si inconnu
    """
    if not name:
        return None

    name_lower = name.lower().strip()

    # Vérifier les prénoms masculins
    if name_lower in FRENCH_NAMES_MALE_LOWER:
        # Certains prénoms sont mixtes (Camille, Dominique, etc.)
        if name_lower in FRENCH_NAMES_FEMALE_LOWER:
            return None  # Ambigu
        return "M"

    # Vérifier les prénoms féminins
    if name_lower in FRENCH_NAMES_FEMALE_LOWER:
        return "F"

    return None


def get_name_confidence(name: str) -> float:
    """
    Retourne un score de confiance (0.0-1.0) que le nom est un prénom.

    Args:
        name: Le nom à évaluer

    Returns:
        Score de confiance entre 0.0 et 1.0
    """
    if not name:
        return 0.0

    name_clean = name.strip()
    name_lower = name_clean.lower()

    # Prénom connu dans notre dictionnaire
    if name_lower in ALL_FRENCH_NAMES_LOWER:
        return 0.9

    # Heuristiques pour prénoms non-répertoriés
    confidence = 0.3  # Base pour un mot avec majuscule

    # Commence par une majuscule (attendu pour un prénom)
    if name_clean and name_clean[0].isupper():
        confidence += 0.1

    # Terminaisons typiques de prénoms français
    feminine_endings = ("ine", "ette", "elle", "ie", "ée", "ane", "ène")
    masculine_endings = ("ien", "eau", "ard", "aud", "ert", "ème")

    if name_lower.endswith(feminine_endings) or name_lower.endswith(masculine_endings):
        confidence += 0.15

    # Longueur raisonnable pour un prénom (3-12 caractères)
    if 3 <= len(name_clean) <= 12:
        confidence += 0.1

    # Pas de caractères spéciaux (sauf tiret et apostrophe)
    if all(c.isalpha() or c in "-'" for c in name_clean):
        confidence += 0.05

    return min(confidence, 0.7)  # Max 0.7 pour prénoms non-répertoriés


if __name__ == "__main__":
    # Tests
    test_names = [
        "Marie", "Pierre", "Kamel", "Sophie", "Camille",
        "Jean-Pierre", "Éléonore", "Xénon", "Robert",
        "pas", "coupé", "soudain", "Machintruc"
    ]

    print("=== Test du dictionnaire de prénoms français ===\n")
    for name in test_names:
        is_name = is_french_name(name)
        gender = get_gender_from_name(name)
        conf = get_name_confidence(name)
        print(f"  {name:15} -> prénom: {str(is_name):5}, genre: {str(gender):4}, confiance: {conf:.2f}")
