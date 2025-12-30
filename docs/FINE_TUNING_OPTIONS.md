# Options de Fine-Tuning pour AudioReader

Ce document explore les options de fine-tuning TTS pour ameliorer la qualite audio des livres audio.

## 1. XTTS-v2 Fine-Tuning

### Avantages
- **Zero-shot voice cloning**: Clone une voix avec seulement 6 secondes d'audio
- **Multilangue**: 17 langues supportees dont le francais
- **Qualite comparable aux solutions commerciales**: 85-95% de similarite

### Configuration requise
- **Audio minimal**: 10-15 minutes pour un fine-tuning complet
- **Temps d'entrainement**: 2-4 heures sur GPU
- **GPU**: Minimum 8GB VRAM, recommande 16GB+

### Installation
```bash
pip install TTS torch torchaudio
```

### Utilisation dans AudioReader
```python
from src.tts_xtts_engine import XTTSEngine, XTTSConfig

# Configuration
config = XTTSConfig(
    default_language="fr",
    use_gpu=True,
    temperature=0.7
)

engine = XTTSEngine(config)

# Enregistrer une voix clonee
engine.register_voice("narrator", "path/to/voice_sample.wav")

# Synthetiser
engine.synthesize_chapter(text, "output.wav", voice_id="narrator")
```

### Note importante (Dec 2024)
Coqui AI a ferme ses portes en decembre 2024. Le projet open-source reste maintenu par la communaute.

Sources:
- [XTTS-v2 sur Hugging Face](https://huggingface.co/coqui/XTTS-v2)
- [Documentation XTTS](https://docs.coqui.ai/en/latest/models/xtts.html)

---

## 2. StyleTTS2 Fine-Tuning

### Avantages
- **Haute qualite**: Approche niveau humain pour le TTS
- **Diffusion de style**: Capture les nuances prosodiques
- **Efficace**: Bon resultat avec seulement 30 minutes d'audio

### Configuration requise
- **Audio minimal**: 30 minutes (decent), 4 heures (excellent)
- **GPU**: Minimum 12GB VRAM pour l'entrainement complet
- **Format**: WAV 22050Hz mono

### Parametres d'entrainement recommandes
```yaml
epochs: 50
joint_epoch: 10  # Debut entrainement adversarial
batch_size: 2
max_length: 220  # tokens
```

### Structure du dataset
```
dataset/
  wavs/
    001.wav
    002.wav
    ...
  metadata.csv  # format: filename|text|speaker
```

Sources:
- [StyleTTS2 Fine-Tune](https://github.com/IIEleven11/StyleTTS2FineTune)
- [StyleTTS2 Original](https://github.com/yl4579/StyleTTS2)
- [Vokan - Modele pre-entraine ameliore](https://huggingface.co/ShoukanLabs/Vokan)

---

## 3. Comparaison pour les Audiobooks

| Critere | XTTS-v2 | StyleTTS2 |
|---------|---------|-----------|
| Qualite voix | ★★★★☆ | ★★★★★ |
| Facilite setup | ★★★☆☆ | ★★☆☆☆ |
| Audio requis | 6s-15min | 30min-4h |
| Temps entrainement | 2-4h | 8-24h |
| VRAM minimum | 8GB | 12GB |
| Langues | 17 | Principalement EN |
| Support FR | Excellent | Limite |

### Recommandation pour AudioReader

**Pour le francais (cas actuel):**
1. **XTTS-v2** est recommande car:
   - Support natif du francais de haute qualite
   - Zero-shot cloning avec 6 secondes
   - Plus simple a integrer

**Pour l'anglais ou qualite maximale:**
1. **StyleTTS2** si:
   - Vous avez 4+ heures d'audio
   - GPU 16GB+ disponible
   - Le francais n'est pas critique

---

## 4. Workflow de Fine-Tuning Recommande

### Etape 1: Preparer les echantillons audio
```bash
# Extraire des segments propres d'un audiobook existant
# Minimum 6 secondes par echantillon
# Format: WAV 22050Hz ou 24000Hz mono
```

### Etape 2: Transcrire les segments
```bash
# Utiliser Whisper pour la transcription
pip install faster-whisper
```

### Etape 3: Fine-tuner le modele
```python
# XTTS-v2 fine-tuning simplifie
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.fine_tune(
    dataset_path="./my_dataset/",
    output_path="./my_model/",
    epochs=10
)
```

### Etape 4: Integrer dans AudioReader
```python
# Utiliser le modele fine-tune
config = XTTSConfig(
    model_name="./my_model/",  # Chemin vers le modele fine-tune
    default_language="fr"
)
engine = XTTSEngine(config)
```

---

## 5. Outils Complementaires

### Pandrator
Outil GUI pour creer des audiobooks avec XTTS, inclut:
- Voice cloning instantane
- Enhancement RVC
- Fine-tuning XTTS integre

### TTS Forge
Pipeline de clonage vocal personnalisable avec XTTS-v2.

Sources:
- [TTS Forge Guide](https://chandanbhagat.com.np/tts-forge-build-your-custom-voice-cloning-pipeline/)
- [Building audiobooks with XTTS-V2](https://medium.com/@jaimonjk/building-audiobooks-using-the-open-source-xtts-v2-model-6bfbbd412fee)

---

## 6. Prochaines Etapes pour AudioReader

1. **Court terme**: Utiliser XTTS-v2 en zero-shot avec des echantillons de voix
2. **Moyen terme**: Creer un dataset de narrateur francais pour fine-tuning
3. **Long terme**: Evaluer StyleTTS2 quand le support francais s'ameliorera

## Configuration AudioReader pour XTTS

Voir `src/tts_xtts_engine.py` pour l'implementation complete.
