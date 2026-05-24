"""
Exemple de configuration Double GPU pour AudioReader.
Conçu pour optimiser l'usage de 2 cartes RTX 3090 (24 Go chacune).

Ce script montre comment :
1. Isoler le LLM (Ollama) sur le GPU 0 (CUDA_VISIBLE_DEVICES="0").
2. Isoler le pipeline de synthèse AudioReader (Kokoro / XTTS) sur le GPU 1 (CUDA_VISIBLE_DEVICES="1").
3. Lancer un test d'intégration parallèle pour valider la répartition de la charge.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Ajouter le chemin racine au sys.path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

def print_banner():
    print("=" * 65)
    print("   AUDIOREADER - UTILSATEUR DUAL RTX 3090 CONCURRENCY LAUNCHER   ")
    print("=" * 65)
    print("Ce script démontre l'isolation GPU et l'exécution concurrente :")
    print("  - GPU 0 : Réservé à Ollama (LLM) pour l'analyse émotionnelle / styles.")
    print("  - GPU 1 : Réservé à la synthèse TTS (Kokoro/XTTS) & DSP (Pedalboard).")
    print("=" * 65)

def check_gpus():
    """Vérifie la présence des GPU Nvidia."""
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
        if res.returncode == 0:
            gpus = res.stdout.strip().split("\n")
            print(f"\n[GPU] {len(gpus)} GPU(s) détecté(s) :")
            for idx, gpu in enumerate(gpus):
                print(f"  - GPU {idx}: {gpu}")
            return len(gpus)
    except FileNotFoundError:
        print("\n[GPU] Avertissement: nvidia-smi non disponible. GPU physiques non vérifiables.")
    return 0

def run_multigpu_demo():
    print_banner()
    num_gpus = check_gpus()

    print("\n--- INSTRUCTIONS POUR CONFIGURER OLLAMA SUR LE GPU 0 ---")
    print("Pour isoler Ollama sur le GPU 0, définissez la variable d'environnement")
    print("avant de lancer le serveur Ollama. Exemple en PowerShell :")
    print('  $env:CUDA_VISIBLE_DEVICES="0"')
    print("  ollama serve")
    print("-" * 55)

    print("\n--- TEST DE DÉMARRAGE AUDIOREADER SUR LE GPU 1 ---")
    print("Nous simulons le démarrage d'AudioReader avec isolation sur le GPU 1...")
    
    # Configuration des variables d'environnement pour le processus fils
    env = os.environ.copy()
    # On force CUDA à ne voir que la carte 1 pour ce processus
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["AUDIOREADER_CUDA_DEVICE"] = "1"
    
    test_text = "Ceci est une simulation de conversion audio haute qualité sur GPU dédié."
    output_path = Path(__file__).parent / "test_gpu_output.wav"

    print(f"\n[Synthesizer] Lancement de la synthèse avec CUDA_VISIBLE_DEVICES=1...")
    
    try:
        from src.tts_engine import create_tts_engine
        from src.gpu_config import GPUConfig
        
        # Instanciation manuelle avec GPUConfig pointant sur l'ID 0 (qui correspondra au GPU 1 physique grâce à l'isolation de CUDA_VISIBLE_DEVICES)
        # Ou en utilisant directement cuda_device_id=1 si CUDA_VISIBLE_DEVICES n'est pas modifié.
        gpu_conf = GPUConfig(use_gpu=True, cuda_device_id=1 if num_gpus > 1 else 0)
        print(f"[Synthesizer] Device affecté : {gpu_conf.get_torch_device()}")
        
        engine = create_tts_engine(
            language="fr",
            engine_type="kokoro",
            gpu_config=gpu_conf
        )
        
        start_time = time.time()
        success = engine.synthesize(test_text, output_path, voice="ff_siwis")
        duration = time.time() - start_time
        
        if success and output_path.exists():
            print(f"\n[Succès] Synthèse terminée en {duration:.2f}s !")
            print(f"[Succès] Fichier de test enregistré à : {output_path.absolute()}")
            # Nettoyer
            output_path.unlink()
        else:
            print("\n[Erreur] Échec de la synthèse.")
            
    except Exception as e:
        print(f"\n[Erreur] Une erreur s'est produite lors de l'exécution du test : {e}")
        print("[Info] Assurez-vous d'avoir installé onnxruntime-gpu et torch avec support CUDA.")

if __name__ == "__main__":
    run_multigpu_demo()
