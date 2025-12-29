# Pistes d'Amélioration AudioReader

## 1. Post-traitement Audio (Implémenté)

Le script `postprocess.py` applique une chaîne de mastering professionnelle:

```bash
python postprocess.py output_kokoro/
```

**Chaîne de traitement:**
1. **EQ Low-cut** - Coupe les fréquences < 80Hz (rumble, bruit)
2. **Normalisation loudness** - EBU R128, -20 LUFS (standard podcast/audiobook)
3. **Limiteur** - Peak max -3dB (évite distorsion)
4. **Room tone** - 0.75s début, 2s fin (silence naturel)
5. **Export MP3** - 192kbps, 44.1kHz, mono

Sources: [ACX Mastering Guide](https://www.acx.com/mp/blog/mastering-audiobooks-with-alex-the-audio-scientist), [Audacity Audiobook Mastering](https://support.audacityteam.org/audio-editing/audiobook-mastering)

---

## 2. Chunking Intelligent du Texte

### Problème
Les longs paragraphes peuvent créer des pauses non naturelles ou des problèmes de qualité.

### Solution
Découper le texte intelligemment aux frontières naturelles:

```python
import re

def smart_chunk(text: str, max_chars: int = 500) -> list[str]:
    """Découpe le texte aux frontières de phrases."""
    # Priorité: paragraphes > phrases > clauses

    # 1. Séparer par paragraphes
    paragraphs = text.split('\n\n')

    chunks = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            # 2. Séparer par phrases
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current = ""
            for sent in sentences:
                if len(current) + len(sent) <= max_chars:
                    current += " " + sent if current else sent
                else:
                    if current:
                        chunks.append(current)
                    current = sent
            if current:
                chunks.append(current)

    return chunks
```

### Améliorations possibles
- Utiliser spaCy/NLTK pour une meilleure segmentation
- Préserver le contexte entre chunks pour la prosodie
- Crossfade entre les chunks audio

Sources: [Deepgram Text Chunking](https://developers.deepgram.com/docs/text-chunking-for-tts-optimization), [Chatterbox Extended](https://github.com/petermg/Chatterbox-TTS-Extended)

---

## 3. Contrôle de la Prosodie (SSML)

### Tags pour pauses naturelles

```python
def add_prosody_markers(text: str) -> str:
    """Ajoute des marqueurs de prosodie au texte."""

    # Pauses après ponctuation
    text = text.replace('. ', '... ')  # Pause longue
    text = text.replace(', ', ', ... ')  # Pause courte (si supporté)

    # Emphase sur les mots importants (dialogues)
    text = re.sub(r'"([^"]+)"', r'... "\1" ...', text)

    return text
```

### Tags émotionnels (Kokoro/Chatterbox)
```
[laugh] - Rire
[sigh] - Soupir
[gasp] - Halètement
[cough] - Toux
```

### SSML (pour Azure/Google TTS)
```xml
<speak>
  <prosody rate="slow" pitch="+5%">
    Texte avec emphase
  </prosody>
  <break time="500ms"/>
</speak>
```

Sources: [SSML Prosody Guide](https://speechgen.io/en/node/prosody/), [Research: Rule-based Prosody](https://arxiv.org/html/2307.02132)

---

## 4. Amélioration de la Cohérence Cross-Chapitre

### Problème
Le timbre et le style peuvent varier entre les chapitres.

### Solutions

#### A. Prompt audio constant
Utiliser le même segment audio de référence pour tous les chapitres:

```python
# Pour F5-TTS ou Chatterbox avec voice cloning
REFERENCE_AUDIO = "ma_voix_reference.wav"

for chapter in chapters:
    engine.synthesize(
        chapter.text,
        output_path,
        reference_audio=REFERENCE_AUDIO
    )
```

#### B. Style embedding
Extraire et réutiliser l'embedding de style du premier chapitre.

#### C. Post-traitement uniforme
Appliquer les mêmes paramètres de mastering à tous les chapitres.

Sources: [Apple ML Research: Long-form TTS](https://machinelearning.apple.com/research/neural-tts-long-form)

---

## 5. Détection et Correction Automatique

### A. Détection de mots mal prononcés
```python
# Dictionnaire de corrections phonétiques
CORRECTIONS = {
    "ChatGPT": "Tchatte Dji Pi Ti",
    "API": "A P I",
    "GitHub": "Guite Hub",
}

def fix_pronunciation(text: str) -> str:
    for word, phonetic in CORRECTIONS.items():
        text = text.replace(word, phonetic)
    return text
```

### B. Détection de silences anormaux
```bash
# Trouver les silences > 3 secondes
ffmpeg -i audio.wav -af silencedetect=n=-50dB:d=3 -f null -
```

### C. Validation qualité audio
```python
def validate_audio(filepath: Path) -> dict:
    """Vérifie que l'audio respecte les standards."""
    info = get_audio_info(filepath)

    issues = []

    # Vérifier durée
    if info['duration'] < 10:
        issues.append("Trop court (< 10s)")

    # Vérifier sample rate
    if info['sample_rate'] < 44100:
        issues.append(f"Sample rate bas ({info['sample_rate']}Hz)")

    return {"valid": len(issues) == 0, "issues": issues}
```

---

## 6. Modèles TTS Alternatifs à Explorer

| Modèle | Qualité | Vitesse | Avantage |
|--------|---------|---------|----------|
| **Chatterbox** | Excellente | Rapide | Contrôle émotion, beats ElevenLabs |
| **Orpheus 3B** | Humaine | Moyenne | Tags expressifs, multilingue |
| **F5-TTS** | Excellente | Lente | Meilleur voice cloning |
| **Parler-TTS** | Bonne | Rapide | Descriptions naturelles |
| **StyleTTS 2** | Excellente | Moyenne | Transfert de style |

### Installation Chatterbox (si Python 3.10/3.11)
```bash
pip install chatterbox-tts torch torchaudio
```

---

## 7. Interface Utilisateur

### A. Interface CLI améliorée
- Barre de progression avec ETA
- Preview audio avant export complet
- Mode interactif pour corrections

### B. Interface Web (Gradio)
```python
import gradio as gr

def convert_book(file, voice, speed):
    # Conversion...
    return audio_path

demo = gr.Interface(
    fn=convert_book,
    inputs=[
        gr.File(label="Livre Markdown"),
        gr.Dropdown(choices=list(VOICES.keys())),
        gr.Slider(0.5, 2.0, value=1.0)
    ],
    outputs=gr.Audio()
)
```

---

## 8. Métadonnées et Chapitrage

### Ajouter métadonnées MP3
```python
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, TIT2, TPE1, TALB, TRCK

def add_metadata(filepath: Path, title: str, artist: str, album: str, track: int):
    audio = MP3(filepath, ID3=ID3)
    audio.tags.add(TIT2(encoding=3, text=title))
    audio.tags.add(TPE1(encoding=3, text=artist))
    audio.tags.add(TALB(encoding=3, text=album))
    audio.tags.add(TRCK(encoding=3, text=str(track)))
    audio.save()
```

### Créer fichier M4B (audiobook avec chapitres)
```bash
# Concaténer tous les MP3
ffmpeg -f concat -i files.txt -c copy combined.mp3

# Convertir en M4B avec chapitres
ffmpeg -i combined.mp3 -i chapters.txt -map_metadata 1 audiobook.m4b
```

---

## 9. Benchmarks et Tests

### Script de comparaison qualité
```python
def benchmark_engines():
    """Compare différents moteurs TTS."""
    text = "Texte de test standardisé..."

    engines = [
        ("Edge-TTS", EdgeTTSEngine()),
        ("Kokoro", KokoroEngine()),
    ]

    results = []
    for name, engine in engines:
        start = time.time()
        engine.synthesize(text, f"test_{name}.wav")
        elapsed = time.time() - start

        results.append({
            "engine": name,
            "time": elapsed,
            "file_size": Path(f"test_{name}.wav").stat().st_size
        })

    return results
```

---

## Roadmap Suggérée

### Phase 1 (Court terme)
- [x] Post-traitement audio automatique
- [ ] Chunking intelligent du texte
- [ ] Correction prononciation (dictionnaire)

### Phase 2 (Moyen terme)
- [ ] Support Chatterbox (si Python 3.10+)
- [ ] Interface Gradio
- [ ] Métadonnées MP3/M4B

### Phase 3 (Long terme)
- [ ] Voice cloning personnalisé
- [ ] Détection émotion automatique
- [ ] Multi-voix (dialogues)
