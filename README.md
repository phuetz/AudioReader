# AudioReader

**Convertisseur de livres Markdown en audiobooks haute qualite**

Propulse par [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) - un modele TTS open-source qui rivalise avec ElevenLabs, 100% gratuit et local.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)

---

## Table des matieres

- [Fonctionnalites](#-fonctionnalites)
- [Nouveautes v2.4 - Outils et Moteurs](#-nouveautes-v24---outils-et-moteurs)
- [Nouveautes v2.3 - Style ElevenLabs](#-nouveautes-v23---style-elevenlabs)
- [Nouveautes v2.2 - Export Multi-Format](#-nouveautes-v22---export-multi-format)
- [Nouveautes v2.1 - Fonctionnalites Avancees](#-nouveautes-v21---fonctionnalites-avancees)
- [Nouveautes v2.0 - Pipeline HQ](#-nouveautes-v20---pipeline-hq)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Pipeline Haute Qualite](#-pipeline-haute-qualite)
- [Architecture](#-architecture)
- [Qualite Audio](#-qualite-audio)
- [Distribution](#-distribution)
- [References Scientifiques](#-references-scientifiques)
- [Sources et Credits](#-sources-et-credits)

---

## Fonctionnalites

### Synthese Vocale
- **Kokoro-82M**: Modele a 82 millions de parametres, qualite proche d'ElevenLabs
- **Multilingue**: Francais, anglais (US/UK), japonais, chinois, et plus
- **Voix naturelles**: Intonation et prosodie realistes
- **Voice Blending**: Melange de voix avec ponderation (ex: `af_bella:60,am_adam:40`)
- **Multi-voix automatique**: Voix differentes par personnage detecte
- **Performance**: ~5x temps reel sur CPU

### Formats d'entree
- **Markdown**: Fichier unique avec headers pour chapitres
- **Multi-fichiers**: Repertoire avec un fichier .md par chapitre
- **EPUB**: Extraction automatique des chapitres

### Traitement du Texte
- **Chunking intelligent**: Decoupage aux frontieres naturelles (phrases, paragraphes)
- **Pauses intelligentes**: Pauses configurables entre phrases (0.3s) et paragraphes (0.8s)
- **Correction prononciation**: Dictionnaire integre pour acronymes et termes techniques
- **Normalisation avancee**: Nombres, dates, heures, monnaies, chiffres romains
- **Tags emotionnels**: Support des tags `[laugh]`, `[sigh]`, `[cough]`, etc.

### Post-traitement Audio
- **Normalisation loudness**: EBU R128, -19 LUFS (standard podcast/audiobook)
- **EQ automatique**: Presence (3kHz) + Air (12kHz) + Low-cut (80Hz)
- **De-essing**: Reduction des sibilantes
- **Compression**: Compression douce pour voix
- **Limiteur**: True peak a -1.5 dB

### Export Professionnel
- **Metadonnees ID3**: Titre, auteur, narrateur, couverture
- **Format M4B**: Audiobook avec chapitres navigables (compatible Apple Books)
- **Export MP3**: Fichiers separes ou combines

### Interface
- **CLI complete**: Scripts en ligne de commande avec options avancees
- **Interface Web**: Application Gradio intuitive
- **Pipeline HQ**: Script dedie pour qualite maximale

---

## Nouveautes v2.4 - Outils et Moteurs

### Moteur XTTS-v2 (Clonage de Voix)

Nouveau moteur TTS base sur XTTS-v2 de Coqui pour le clonage de voix haute qualite :

```python
from src.tts_xtts_engine import XTTSEngine, XTTSConfig

config = XTTSConfig(
    default_language="fr",
    use_gpu=True,
    temperature=0.7
)
engine = XTTSEngine(config)

# Cloner une voix avec seulement 6 secondes d'audio
engine.register_voice("narrator", "samples/ma_voix.wav")

# Synthetiser avec la voix clonee
engine.synthesize_chapter(text, "output.wav", voice_id="narrator")
```

**Caracteristiques :**
- Clonage avec 6 secondes d'audio minimum
- 17 langues supportees dont le francais
- Qualite comparable aux solutions commerciales

### Crossfade Audio

Transitions fluides entre segments audio pour eliminer les "coutures" audibles :

```python
from src.audio_crossfade import apply_crossfade_to_chapter

# Assembler des segments avec crossfade
final_audio = apply_crossfade_to_chapter(
    audio_segments,
    sample_rate=24000,
    crossfade_ms=50  # 50ms de fondu enchaine
)
```

Le moteur hybride utilise maintenant le crossfade par defaut :

```python
from src.tts_hybrid_engine import create_hybrid_engine

engine = create_hybrid_engine(
    use_crossfade=True,
    crossfade_ms=50
)
```

### Preview Rapide (30 secondes)

Generez un apercu de 30 secondes pour tester les parametres avant la conversion complete :

```python
from src.preview_generator import generate_quick_preview

success, message = generate_quick_preview(
    text=texte_complet,
    output_path="preview.wav",
    engine_type="hybrid",
    duration=30.0
)
```

**Extraction intelligente :**
- Debut du texte (contexte)
- Passage avec dialogue (si present)
- Passage emotionnel (si detecte)

### Interface Web pour Corrections

Interface Gradio pour gerer les corrections de prononciation :

```bash
# Lancer l'interface
python -m src.corrections_ui --file corrections.json --port 7861
```

**Fonctionnalites :**
- Ajout/suppression de corrections
- Recherche dans le glossaire
- Test audio avec moteur TTS
- Import/Export JSON

### Guide de Fine-Tuning

Nouveau document `docs/FINE_TUNING_OPTIONS.md` avec :
- Comparaison XTTS-v2 vs StyleTTS2
- Configuration requise (GPU, audio)
- Workflow de fine-tuning etape par etape
- Recommandations pour le francais

---

## Nouveautes v2.3 - Style ElevenLabs

### Respirations Realistes

Nouvelle generation de respirations basee sur la recherche bio-acoustique :

- **Bruit rose** au lieu de bruit blanc (spectre plus naturel)
- **Formants respiratoires** (resonances du tract vocal)
- **Jitter d'amplitude** (micro-variations naturelles)
- **Support samples optionnels** (hybrid: synthese + samples reels)

### Contours d'Intonation Phrase-Level

Detection et application automatique des patterns prosodiques :

| Type | Pattern | Exemple |
|------|---------|---------|
| DECLARATIVE | Descente finale (-2 a -4 st) | "Il est parti." |
| QUESTION_YN | Montee finale (+3 a +5 st) | "Tu viens ?" |
| QUESTION_WH | Pic initial puis descente | "Ou vas-tu ?" |
| EXCLAMATION | Pic fort puis descente rapide | "Incroyable !" |
| CONTINUATION | Legere montee (+1 st) | "D'abord, ..." |
| SUSPENSE | Descente lente, pause | "Et puis..." |

### Micro-Variations de Timing

Humanisation du rythme pour eviter l'effet "robotique" :

- **Variation gaussienne des pauses** (±5% par defaut)
- **Rythme adaptatif** selon la structure syntaxique
- **Micro-pauses avant mots importants** (0.05s)

```python
# Configuration v2.3
config = ExtendedPipelineConfig(
    enable_advanced_breaths=True,
    enable_intonation_contours=True,
    intonation_strength=0.7,
    enable_timing_humanization=True,
    pause_variation_sigma=0.05,
    enable_emphasis_pauses=True,
)
```

---

## Nouveautes v2.2 - Export Multi-Format

### Export Audio MP3

Le moteur hybride supporte maintenant l'export direct en MP3 :

```python
from src.tts_hybrid_engine import HybridTTSEngine

engine = HybridTTSEngine(mms_language='fra')

# Export WAV (defaut)
engine.synthesize_chapter(text, "chapitre.wav")

# Export MP3
engine.synthesize_chapter(text, "chapitre.mp3", output_format="mp3")

# MP3 avec bitrate personnalise
engine.synthesize_chapter(text, "chapitre.mp3", output_format="mp3", mp3_bitrate="256k")
```

### Export de Livres (PDF, EPUB, HTML, TXT)

Nouveau module `book_exporter.py` pour exporter vos livres en plusieurs formats :

```python
from src.book_exporter import BookExporter, export_markdown_book

# Export rapide depuis un dossier de chapitres Markdown
results = export_markdown_book(
    chapters_dir="path/to/chapters",
    output_dir="output/ebook",
    title="Mon Livre",
    author="Auteur",
    formats=["pdf", "epub", "html", "txt"]
)

# Ou utilisation avancee
exporter = BookExporter(title="Mon Livre", author="Auteur")
exporter.add_chapter("Chapitre 1", "Contenu du chapitre...")
exporter.add_chapter_from_markdown("chapitre-02.md")

exporter.export_pdf("livre.pdf")
exporter.export_epub("livre.epub")
exporter.export_html("livre.html")
exporter.export_txt("livre.txt")
```

| Format | Extension | Description |
|--------|-----------|-------------|
| **PDF** | `.pdf` | Document portable avec table des matieres |
| **EPUB** | `.epub` | Format ebook standard (Kindle, Kobo, liseuses) |
| **HTML** | `.html` | Page web avec dark mode automatique |
| **TXT** | `.txt` | Texte brut formate |

---

## Nouveautes v2.0 - Pipeline HQ

### Multi-Voix Automatique par Personnage

Le systeme detecte automatiquement les personnages et leur attribue des voix differentes :

```
Texte: "Bonjour !" dit Marie. Pierre repondit : "Comment vas-tu ?"

Resultat:
  NARRATEUR  -> ff_siwis (voix par defaut)
  Marie      -> af_bella (voix feminine auto-assignee)
  Pierre     -> am_adam  (voix masculine auto-assignee)
```

**Detection automatique via :**
- Guillemets : `"texte"`, `« texte »`
- Tirets de dialogue : `— texte`
- Verbes de parole : "dit", "repondit", "murmura", "cria"... (+100 verbes FR/EN)

### Normalisation Avancee du Texte

| Type | Avant | Apres |
|------|-------|-------|
| Nombres | `1234` | "mille deux cent trente-quatre" |
| Dates | `25/12/2024` | "vingt-cinq decembre deux mille vingt-quatre" |
| Heures | `14h30` | "quatorze heures trente" |
| Monnaies | `1234,56 EUR` | "mille deux cent... euros et cinquante-six centimes" |
| Romains | `Louis XIV` | "Louis quatorze" |
| Pourcentages | `85%` | "quatre-vingt-cinq pourcent" |
| Telephones | `06 12 34 56 78` | "zero six douze trente-quatre..." |

### Detection du Contexte Narratif

Le systeme adapte la vitesse selon le type de contenu :

| Contexte | Vitesse | Description |
|----------|---------|-------------|
| Action | 1.15x | Scenes rapides, tension |
| Description | 0.90x | Passages descriptifs |
| Introspection | 0.92x | Pensees interieures |
| Flashback | 0.88x | Souvenirs, ton reveur |
| Suspense | 0.88x | Tension, pauses longues |
| Dialogue | 1.00x | Paroles des personnages |

### Analyse Emotionnelle

Detection automatique des emotions avec prosodie adaptee :

- **Joie** : Vitesse +10%, pitch +0.5
- **Tristesse** : Vitesse -10%, pitch -0.3, pauses longues
- **Colere** : Vitesse +15%, volume +20%
- **Peur** : Vitesse +20%, micro-pauses de respiration
- **Suspense** : Vitesse -15%, longues pauses

### Continuite Emotionnelle

- **Lissage des transitions** entre segments (pas de changements brusques)
- **Detection des climax** dramatiques
- **Arc emotionnel** coherent sur le chapitre

---

## Nouveautes v2.1 - Fonctionnalites Avancees

### Audio Tags Style ElevenLabs v3

Support des tags expressifs directement dans le texte :

```
[whispers] Je dois te dire quelque chose...
[excited] C'est incroyable ! [laugh]
[dramatic] [pause] Et puis... tout a change.
[sarcastic] Oh, quelle surprise...
```

**Tags supportes :**

| Categorie | Tags |
|-----------|------|
| Emotions | `[excited]`, `[sad]`, `[angry]`, `[whispers]`, `[fearful]`, `[tender]`, `[dramatic]` |
| Actions | `[sigh]`, `[laugh]`, `[chuckle]`, `[gasp]`, `[cough]`, `[yawn]`, `[sniff]` |
| Pauses | `[pause]`, `[long pause]`, `[beat]`, `[silence]` |
| Styles | `[sarcastic]`, `[cheerful]`, `[serious]`, `[mysterious]`, `[narrator]`, `[announcer]` |

Chaque tag ajuste automatiquement la prosodie (vitesse, pitch, volume, pauses).

### Voice Morphing

Modification de la voix en temps reel :

```python
from src.voice_morphing import VoiceMorpher, VoicePresets

morpher = VoiceMorpher()

# Presets disponibles
preset = VoicePresets.get("more_masculine")  # Voix plus grave
preset = VoicePresets.get("younger")         # Voix plus jeune
preset = VoicePresets.get("whisper")         # Chuchotement
preset = VoicePresets.get("expressive")      # Plus expressif
```

**Parametres de morphing :**

| Parametre | Plage | Description |
|-----------|-------|-------------|
| `pitch_shift` | -12 a +12 | Demi-tons (hauteur de la voix) |
| `formant_shift` | 0.5 a 2.0 | Timbre (masculin/feminin) |
| `time_stretch` | 0.5 a 2.0 | Vitesse sans changer le pitch |
| `breathiness` | 0.0 a 1.0 | Souffle dans la voix |
| `roughness` | 0.0 a 1.0 | Voix rauque |
| `stability` | 0.0 a 1.0 | Expressivite variable |

### Clonage de Voix (XTTS-v2)

Clonez n'importe quelle voix avec seulement 6 secondes d'audio :

```python
from src.voice_cloning import VoiceCloningManager

manager = VoiceCloningManager()

# Cloner une voix
manager.register_cloned_voice(
    audio_path="sample.wav",
    voice_id="cloned_marie",
    language="fr"
)

# Utiliser la voix clonee
manager.synthesize_with_cloned_voice(
    text="Bonjour tout le monde !",
    voice_id="cloned_marie",
    output_path="output.wav"
)
```

**Prerequis :**
```bash
pip install TTS torch torchaudio
```

### Cache Intelligent et Parallelisation

Optimisation des performances :

```python
from src.synthesis_cache import SynthesisCache, ParallelSynthesizer

# Cache pour eviter de regenerer les segments identiques
cache = SynthesisCache(max_size_mb=1000)

# Parallelisation sur plusieurs coeurs
synth = ParallelSynthesizer(num_workers=4, cache=cache)
```

**Statistiques du cache :**
- Taux de hit affiche
- Temps economise calcule
- Nettoyage automatique (LRU)

### Controle d'Emotion et Phonemes IPA

Controle fin de l'expressivite :

```python
from src.emotion_control import EmotionController, EmotionSettings

controller = EmotionController(EmotionSettings(
    intensity=0.8,      # 0.0=neutre, 1.0=tres expressif
    stability=0.7,      # Stabilite de la voix
    style_exaggeration=0.5  # Exageration du style
))

prosody = controller.calculate_prosody(
    base_speed=1.0,
    emotion_type="joy"
)
```

**Phonemes personnalises (IPA) :**

```python
from src.emotion_control import PronunciationManager

manager = PronunciationManager("fr")
manager.add_phoneme("API", "a pe i")
manager.add_phoneme("Python", "pitonne")
manager.add_correction("etc.", "et cetera")

text = manager.process("L'API Python est geniale")
# -> "L'a pe i pitonne est geniale"
```

### Generation de Conversations Multi-Speakers

Creez des dialogues avec plusieurs personnages :

```python
from src.conversation_generator import ConversationGenerator

generator = ConversationGenerator()

script = """
JEAN: Bonjour Marie, comment vas-tu?
MARIE: [cheerful] Tres bien, merci! Et toi?
JEAN: [excited] Super! J'ai une grande nouvelle!
"""

conversation = generator.parse_script(
    script,
    speaker_config={
        "JEAN": {"gender": "male"},
        "MARIE": {"gender": "female"}
    }
)

# Generer l'audio
segments = generator.generate_audio(conversation, synthesize_fn, output_dir)
generator.assemble_audio(segments, "conversation.wav")

# Exporter la timeline (pour montage)
generator.export_timeline(segments, "timeline.json")  # ou .srt, .csv
```

**Formats de script supportes :**
- Format script : `PERSONNAGE: dialogue`
- Format markdown : `**Personnage:** dialogue`
- Format theatre : `PERSONNAGE. - dialogue`

### Post-Processing Broadcast

Pipeline audio professionnel :

```
1. Highpass 80Hz (supprime rumble)
2. De-essing 6kHz (reduit sibilantes)
3. EQ Presence 3kHz (+2dB)
4. EQ Air 12kHz (+1.5dB)
5. Compression 3:1 (dynamique controlee)
6. Loudness -19 LUFS (EBU R128)
7. Limiter -1.5dB (pas de distorsion)
```

---

## Installation

### Prerequis

- Python 3.10 ou superieur
- FFmpeg (pour le traitement audio)
- ~500 MB d'espace disque (modele)

### Installation rapide

```bash
# Cloner le projet
git clone https://github.com/votre-repo/AudioReader.git
cd AudioReader

# Creer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# ou: venv\Scripts\activate  # Windows

# Installer les dependances
pip install -r requirements.txt

# Telecharger le modele Kokoro (~340 MB)
curl -L -o kokoro-v1.0.onnx "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
curl -L -o voices-v1.0.bin "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
```

### Verifier l'installation

```bash
python audioreader.py --list-voices
```

---

## Utilisation

### Mode Standard (audioreader.py)

```bash
# Convertir un livre Markdown
python audioreader.py mon_livre.md

# Avec une voix anglaise
python audioreader.py book.md --voice af_heart

# Avec mastering audio professionnel
python audioreader.py livre.md --mastering

# Export en M4B avec chapitres
python audioreader.py livre.md --format m4b --title "Mon Livre" --author "Auteur"
```

### Mode Haute Qualite (audioreader_hq.py) - NOUVEAU

```bash
# Pipeline HQ complet (multi-voix + emotions + post-processing)
python audioreader_hq.py livre.md -o output/

# Analyse seule (sans generation audio)
python audioreader_hq.py livre.md --analyze-only

# Avec configuration personnalisee
python audioreader_hq.py livre.md --config config_hq.json

# Options disponibles
python audioreader_hq.py livre.md \
    --lang fr \
    --voice ff_siwis \
    --speed 1.0 \
    --no-emotions      # Desactiver l'analyse emotionnelle
    --no-characters    # Desactiver le multi-voix
    --no-enhance       # Desactiver le post-processing
```

### Configuration Multi-Voix Personnalisee

Creer un fichier `config_hq.json` :

```json
{
  "lang": "fr",
  "narrator_voice": "ff_siwis",
  "voice_mapping": {
    "Marie": "af_bella",
    "Pierre": "am_adam",
    "Le docteur": "bm_george"
  },
  "auto_assign_voices": true,
  "enable_emotion_analysis": true,
  "enable_narrative_context": true,
  "sentence_pause": 0.3,
  "paragraph_pause": 0.8,
  "target_lufs": -19.0
}
```

### Voice Blending (melange de voix)

```bash
# Melange 60% Bella, 40% Adam
python audioreader.py livre.md --voice "af_bella:60,am_adam:40"

# Melange 50-50
python audioreader.py livre.md --voice "af_bella,am_adam"
```

### Controle des pauses

```bash
# Pauses plus longues entre paragraphes (1 seconde)
python audioreader.py livre.md --paragraph-pause 1.0

# Pauses plus courtes entre phrases (0.2 secondes)
python audioreader.py livre.md --sentence-pause 0.2
```

### Interface graphique

```bash
python audioreader.py --gui
# Ouvrir http://localhost:7860
```

### Voix disponibles

| Code | Nom | Langue | Genre |
|------|-----|--------|-------|
| `ff_siwis` | Siwis | Francais | F |
| `af_heart` | Heart | Anglais US | F |
| `af_bella` | Bella | Anglais US | F |
| `af_nicole` | Nicole | Anglais US | F |
| `af_sarah` | Sarah | Anglais US | F |
| `am_adam` | Adam | Anglais US | M |
| `am_michael` | Michael | Anglais US | M |
| `am_eric` | Eric | Anglais US | M |
| `bf_emma` | Emma | Anglais UK | F |
| `bm_george` | George | Anglais UK | M |
| `jf_alpha` | Alpha | Japonais | F |
| `zf_xiaoxiao` | Xiaoxiao | Chinois | F |

---

## Pipeline Haute Qualite

### Vue d'ensemble

```
                           PIPELINE HQ
    +----------------------------------------------------------+
    |                                                          |
    |   TEXTE BRUT                                             |
    |       |                                                  |
    |       v                                                  |
    |   [Normalisation]  nombres, dates, abreviations          |
    |       |                                                  |
    |       v                                                  |
    |   [Detection Personnages]  dialogues, verbes de parole   |
    |       |                                                  |
    |       v                                                  |
    |   [Analyse Emotionnelle]  sentiment, intensite           |
    |       |                                                  |
    |       v                                                  |
    |   [Contexte Narratif]  action, description, pensees      |
    |       |                                                  |
    |       v                                                  |
    |   [Continuite Emotionnelle]  lissage transitions         |
    |       |                                                  |
    |       v                                                  |
    |   [Attribution Voix]  personnage -> voix                 |
    |       |                                                  |
    |       v                                                  |
    |   [Synthese Multi-Voix]  Kokoro TTS                      |
    |       |                                                  |
    |       v                                                  |
    |   [Post-Processing]  EQ, compression, loudness           |
    |       |                                                  |
    |       v                                                  |
    |   AUDIOBOOK HQ                                           |
    |                                                          |
    +----------------------------------------------------------+
```

### Modules du Pipeline

**Pipeline HQ v2.0 :**

| Module | Fichier | Description |
|--------|---------|-------------|
| Normalisation | `text_normalizer.py` | Nombres, dates, symboles en mots |
| Personnages | `character_detector.py` | Detection dialogues et locuteurs |
| Emotions | `emotion_analyzer.py` | Analyse sentiment et intensite |
| Contexte | `narrative_context.py` | Type de narration (action, description...) |
| Continuite | `emotion_continuity.py` | Transitions emotionnelles fluides |
| Pipeline | `hq_pipeline.py` | Integration de tous les modules |
| Audio | `audio_enhancer.py` | Post-processing broadcast |

**Modules Avances v2.1 :**

| Module | Fichier | Description |
|--------|---------|-------------|
| Audio Tags | `audio_tags.py` | Tags expressifs style ElevenLabs v3 |
| Morphing | `voice_morphing.py` | Pitch, formant, time stretch |
| Clonage | `voice_cloning.py` | Clonage de voix XTTS-v2 |
| Cache | `synthesis_cache.py` | Cache intelligent + parallelisation |
| Emotion+ | `emotion_control.py` | Controle emotion + phonemes IPA |
| Conversations | `conversation_generator.py` | Dialogues multi-speakers |
| Pipeline+ | `hq_pipeline_extended.py` | Pipeline unifie etendu |

**Modules v2.2-2.3 :**

| Module | Fichier | Description |
|--------|---------|-------------|
| Moteur Hybride | `tts_hybrid_engine.py` | MMS + Kokoro, export MP3 |
| Export Livres | `book_exporter.py` | PDF, EPUB, HTML, TXT |
| Bio-Acoustique | `bio_acoustics.py` | Respirations realistes |
| Intonation | `intonation_contour.py` | Contours prosodiques |
| Timing | `timing_humanizer.py` | Micro-variations de rythme |

**Modules v2.4 :**

| Module | Fichier | Description |
|--------|---------|-------------|
| Moteur XTTS | `tts_xtts_engine.py` | Clonage de voix XTTS-v2 |
| Crossfade | `audio_crossfade.py` | Transitions audio fluides |
| Preview | `preview_generator.py` | Apercu rapide 30 secondes |
| Corrections UI | `corrections_ui.py` | Interface Gradio corrections |

### Exemple de Sortie

```
=== ANALYSE DU TEXTE ===

--- Personnages detectes ---
  NARRATOR             -> ff_siwis       (45 segments)
  Marie                -> af_bella       (12 segments)
  Pierre               -> am_adam        (8 segments)

--- Analyse emotionnelle ---
  Ton dominant:       suspense
  Emotion principale: fear
  Intensite max:      extreme
  Nombre de climax:   2
  Vitesse suggeree:   0.95x

--- Distribution des emotions ---
  neutral     : ████████████         52.3%
  suspense    : ██████               24.1%
  fear        : ████                 15.2%
  joy         : ██                    8.4%
```

---

## Architecture

```
AudioReader/
├── audioreader.py          # Script principal CLI (standard)
├── audioreader_hq.py       # Script CLI haute qualite
├── app.py                  # Interface Gradio
├── postprocess.py          # Post-traitement audio legacy
├── example_multivoix.py    # Exemple multi-voix
├── kokoro-v1.0.onnx        # Modele TTS (310 MB)
├── voices-v1.0.bin         # Donnees voix (27 MB)
│
├── src/
│   ├── markdown_parser.py      # Parser multi-format (MD, EPUB)
│   ├── tts_kokoro_engine.py    # Moteur Kokoro TTS + multi-voix
│   ├── tts_engine.py           # Moteur Edge-TTS (fallback)
│   ├── text_processor.py       # Chunking, prononciation base
│   ├── audiobook_builder.py    # Metadonnees, export M4B/MP3
│   │
│   │   # --- MODULES HQ v2.0 ---
│   ├── text_normalizer.py      # Nombres, dates, symboles
│   ├── character_detector.py   # Detection personnages/dialogues
│   ├── emotion_analyzer.py     # Analyse emotionnelle
│   ├── narrative_context.py    # Contexte narratif
│   ├── emotion_continuity.py   # Continuite emotionnelle
│   ├── advanced_preprocessor.py # Preprocesseur avance
│   ├── hq_pipeline.py          # Pipeline unifie HQ
│   ├── audio_enhancer.py       # Post-processing broadcast
│   │
│   │   # --- MODULES AVANCES v2.1 ---
│   ├── audio_tags.py           # Tags style ElevenLabs v3
│   ├── voice_morphing.py       # Pitch, formant, time stretch
│   ├── voice_cloning.py        # Clonage XTTS-v2
│   ├── synthesis_cache.py      # Cache + parallelisation
│   ├── emotion_control.py      # Controle emotion + phonemes IPA
│   ├── conversation_generator.py # Dialogues multi-speakers
│   ├── hq_pipeline_extended.py # Pipeline unifie etendu
│   │
│   │   # --- MODULES v2.2-2.3 ---
│   ├── tts_hybrid_engine.py    # Moteur hybride MMS+Kokoro, MP3
│   ├── tts_mms_engine.py       # Moteur MMS-TTS (Facebook)
│   ├── book_exporter.py        # Export PDF, EPUB, HTML, TXT
│   ├── bio_acoustics.py        # Respirations realistes
│   ├── intonation_contour.py   # Contours prosodiques
│   ├── timing_humanizer.py     # Micro-variations de rythme
│   │
│   │   # --- MODULES v2.4 ---
│   ├── tts_xtts_engine.py      # Moteur XTTS-v2 (clonage voix)
│   ├── audio_crossfade.py      # Crossfade entre segments
│   ├── preview_generator.py    # Preview rapide 30s
│   └── corrections_ui.py       # Interface Gradio corrections
│
├── docs/
│   └── FINE_TUNING_OPTIONS.md  # Guide de fine-tuning
├── config_multivoix_example.json  # Config exemple
├── books/                  # Livres sources
└── output/                 # Audiobooks generes
```

---

## Qualite Audio

### Comparaison Standard vs HQ

| Aspect | Standard | Pipeline HQ |
|--------|----------|-------------|
| Voix | Une seule | Multi-personnages auto |
| Nombres | "1234" (tel quel) | "mille deux cent..." |
| Emotions | Fixes | Adaptatives |
| Vitesse | Constante | Variable selon contexte |
| Transitions | Abruptes possibles | Lissees |
| Loudness | -20 LUFS | -19 LUFS |
| Post-processing | Basique | EQ + Compression + De-essing |

### Standards vises

| Parametre | Valeur | Standard |
|-----------|--------|----------|
| Loudness | -19 LUFS | EBU R128 / Podcast |
| True Peak | -1.5 dB max | Broadcast |
| Noise floor | < -60 dB | ACX / Audible |
| Sample rate | 44.1 kHz | CD Quality |
| Bitrate | 192 kbps | Distribution |

### Performance et estimation

| Taille livre | Audio estime | Temps conversion* |
|--------------|--------------|-------------------|
| 50K caracteres | ~30 min | ~6 min |
| 200K caracteres | ~2h | ~25 min |
| 500K caracteres | ~5h | ~1h |
| 1M caracteres | ~10h | ~2h |

*Sur CPU moderne (Intel i7/AMD Ryzen). Le pipeline HQ ajoute ~20% au temps standard.

---

## Distribution

### Plateformes acceptant les voix AI

| Plateforme | Status | Lien |
|------------|--------|------|
| Google Play Books | Accepte | [play.google.com/books/publish](https://play.google.com/books/publish) |
| Findaway/Spotify | Accepte (ElevenLabs) | [findawayvoices.com](https://findawayvoices.com) |
| Kobo | Accepte | [kobo.com/writinglife](https://www.kobo.com/writinglife) |
| Vente directe | Aucune restriction | Votre site web |
| Audible/ACX | **Refuse** | Uniquement voix humaines |

---

## References Scientifiques

### Synthese vocale neuronale

1. **A Survey on Neural Speech Synthesis** (2021)
   - Xu Tan, Tao Qin, Frank Soong, Tie-Yan Liu
   - [arXiv:2106.15561](https://arxiv.org/abs/2106.15561)

2. **Deep Learning-based Expressive Speech Synthesis** (2024)
   - EURASIP Journal on Audio, Speech, and Music Processing
   - [DOI: 10.1186/s13636-024-00329-7](https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-024-00329-7)

### Prosodie et emotion

3. **The Sound of Emotional Prosody** (2025)
   - Pauline Larrouy-Maestri, David Poeppel, Marc D. Pell
   - [DOI: 10.1177/17456916231217722](https://journals.sagepub.com/doi/10.1177/17456916231217722)

4. **Text-aware and Context-aware Expressive Audiobook Speech Synthesis** (2024)
   - Interspeech 2024
   - [arXiv:2406.05672](https://arxiv.org/html/2406.05672)

---

## Sources et Credits

### Modeles TTS

| Projet | Description | Licence |
|--------|-------------|---------|
| **Kokoro-82M** | Modele TTS 82M params | Apache 2.0 |
| **kokoro-onnx** | Runtime ONNX pour Kokoro | MIT |
| **Edge-TTS** | Microsoft Edge TTS | LGPL-3.0 |

### Outils audio

| Projet | Description |
|--------|-------------|
| **FFmpeg** | Traitement audio/video |
| **ffmpeg-normalize** | Normalisation loudness |

---

## Licence

Ce projet est sous licence Apache 2.0. Voir [LICENSE](LICENSE) pour plus de details.

Le modele Kokoro-82M est sous licence Apache 2.0.

---

## FAQ

### Quelle est la difference entre audioreader.py et audioreader_hq.py ?

- `audioreader.py` : Mode standard, une seule voix, traitement basique
- `audioreader_hq.py` : Pipeline haute qualite avec multi-voix automatique, analyse emotionnelle, et post-processing broadcast

### Comment configurer les voix par personnage ?

Creez un fichier JSON avec le mapping :

```json
{
  "voice_mapping": {
    "Marie": "af_bella",
    "Pierre": "am_adam"
  }
}
```

Puis : `python audioreader_hq.py livre.md --config config.json`

### Puis-je publier sur Audible ?

Non, Audible/ACX n'accepte pas les voix generees par IA. Utilisez Google Play Books, Findaway, ou Kobo.

### Quelle est la qualite comparee a ElevenLabs ?

Avec le pipeline HQ (multi-voix + emotions + post-processing), la qualite est tres proche d'ElevenLabs, tout en etant 100% gratuit et local.

### Comment utiliser les Audio Tags ?

Ajoutez les tags directement dans votre texte :
```
[whispers] Je te le dis en secret...
[excited] C'est genial !
[laugh] Ha ha ha !
```

### Comment cloner une voix ?

1. Preparez un fichier audio de reference (minimum 6 secondes, idealement 30 secondes)
2. Installez les dependances : `pip install TTS torch torchaudio`
3. Utilisez le `VoiceCloningManager` pour enregistrer et utiliser la voix

### Le cache est-il automatique ?

Oui, avec le pipeline etendu (`hq_pipeline_extended.py`), le cache est actif par defaut. Les segments identiques ne sont generes qu'une seule fois.

### Comment accelerer la generation ?

1. **Activez le cache** : Les segments deja generes sont reutilises
2. **Parallelisation** : Configurez `num_workers=4` (ou plus selon vos CPUs)
3. **GPU** : Utilisez XTTS-v2 avec GPU pour le clonage de voix

---

*AudioReader v2.4 - Convertissez vos livres en audiobooks de qualite professionnelle*
