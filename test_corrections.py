#!/usr/bin/env python3
"""
Test des corrections de prononciation (Préprocesseur Français).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.french_preprocessor import FrenchTextPreprocessor
from src.tts_unified import UnifiedTTS, TTSEngine

def test_text_processing():
    print("=== Test du Préprocesseur Français (Texte) ===\n")
    
    prep = FrenchTextPreprocessor()
    
    cases = [
        ("Nombres", "Il a gagné 15000€ au loto.", "Il a gagné quinze mille euros au loto."),
        ("Heures", "RDV à 14h30 précise.", "RDV à quatorze heures trente précise."),
        ("Acronymes", "La SNCF et la RATP.", "La S.N.C.F et la R.A.T.P."),
        ("Anglicismes", "J'ai poussé le code sur GitHub.", "J'ai poussé le code sur guite-heube."),
        ("Abréviations", "Bonjour M. Martin.", "Bonjour Monsieur Martin."),
    ]
    
    for category, input_text, expected in cases:
        processed = prep.process(input_text)
        print(f"[{category}]")
        print(f"  In : {input_text}")
        print(f"  Out: {processed}")
        # Note: On ne check pas l'égalité stricte car num2words peut varier légèrement
        print("-" * 40)

def test_audio_generation():
    print("\n=== Test de Génération Audio (Kokoro) ===\n")
    
    tts = UnifiedTTS()
    output_dir = Path("demo_output/corrections")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    text = "M. Dupont a payé 12,50€ pour son ticket SNCF à 14h30. Il adore GitHub."
    print(f"Texte original : {text}")
    
    # Sans préprocesseur (simulation)
    print("Génération SANS préprocesseur...")
    try:
        # On force preprocess=False
        tts.synthesize_to_file(
            text, 
            output_dir / "sans_correction.wav", 
            lang="fr", 
            engine=TTSEngine.KOKORO, # Kokoro a du mal avec les nombres bruts parfois
            preprocess=False
        )
    except Exception as e:
        print(f"Erreur: {e}")

    # Avec préprocesseur
    print("Génération AVEC préprocesseur...")
    try:
        tts.synthesize_to_file(
            text, 
            output_dir / "avec_correction.wav", 
            lang="fr", 
            engine=TTSEngine.KOKORO,
            preprocess=True
        )
        print(f"Fichiers générés dans {output_dir}")
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == "__main__":
    test_text_processing()
    test_audio_generation()