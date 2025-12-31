#!/usr/bin/env python3
"""
Test des améliorations Bio-Acoustiques et Respirations.
"""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))

from src.hq_pipeline_extended import AudiobookGenerator, ExtendedPipelineConfig
from src.tts_unified import UnifiedTTS, TTSConfig

def main():
    print("=== Test des Améliorations 'Style ElevenLabs' ===\n")
    
    # 1. Configuration
    config = ExtendedPipelineConfig(
        lang="fr",
        tts_engine="kokoro",
        narrator_voice="ff_siwis",
        enable_audio_tags=True,
        enable_cache=False, # Désactiver pour le test
    )
    
    # Initialiser le moteur TTS
    tts_engine = UnifiedTTS(TTSConfig(use_french_preprocessor=True))
    
    # Créer le générateur d'audiobook
    generator = AudiobookGenerator(config=config, tts_engine=tts_engine)
    
    # 2. Texte de test (conçu pour déclencher les respirations et émotions)
    test_text = """
    C'était une nuit sombre et orageuse. Le vent hurlait dans les arbres, comme une bête blessée.
    
    [dramatic] Pierre s'arrêta brusquement. [pause] Il retint son souffle, aux aguets.
    
    Soudain, une ombre se glissa derrière lui.
    
    « [whispers] Qui est là ? » murmura-t-il, la voix tremblante de peur.
    
    [gasp] Un éclair déchira le ciel, révélant un visage familier.
    
    « [excited] Marie ! C'est toi ? » s'exclama-t-il avec un soulagement immense.
    
    [laugh] Ils éclatèrent de rire, évacuant la tension accumulée. « Tu m'as fait une peur bleue ! »
    """
    
    print("Analyse et génération en cours...")
    output_dir = Path("demo_output/bio_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Générer
    try:
        results = generator.generate_audiobook(
            chapters=[test_text],
            output_dir=output_dir,
            title="BioTest"
        )
        
        print(f"\nSuccès ! Fichier généré : {results['chapters'][0]['path']}")
        print(f"Statistiques du pipeline :")
        for k, v in results['pipeline_stats'].items():
            print(f"  - {k}: {v}")
            
    except Exception as e:
        print(f"\nErreur lors de la génération : {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
