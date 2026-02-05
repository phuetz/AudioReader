# AudioReader

**Convertisseur de livres Markdown en audiobooks haute qualite**

Propulse par [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) - un modele TTS open-source qui rivalise avec ElevenLabs, 100% gratuit et local.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

---

## Table des matieres

- [Fonctionnalites](#fonctionnalites)
- [Nouveautes v5.0 - Detection Amelioree et Export Professionnel](#nouveautes-v50---detection-amelioree-et-export-professionnel)
- [Nouveautes v4.0 - Interface Moderne et Nouveaux Moteurs](#nouveautes-v40---interface-moderne-et-nouveaux-moteurs)
- [Nouveautes v3.0 - Plateforme Complete](#nouveautes-v30---plateforme-complete)
- [Nouveautes v2.4 - Outils et Moteurs](#nouveautes-v24---outils-et-moteurs)
- [Nouveautes v2.3 - Style ElevenLabs](#nouveautes-v23---style-elevenlabs)
- [Nouveautes v2.2 - Export Multi-Format](#nouveautes-v22---export-multi-format)
- [Nouveautes v2.1 - Fonctionnalites Avancees](#nouveautes-v21---fonctionnalites-avancees)
- [Nouveautes v2.0 - Pipeline HQ](#nouveautes-v20---pipeline-hq)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Pipeline Haute Qualite](#pipeline-haute-qualite)
- [Architecture](#architecture)
- [Qualite Audio](#qualite-audio)
- [Distribution](#distribution)
- [Integration IA (Claude / ChatGPT)](#integration-ia-claude--chatgpt)
- [Docker](#docker)
- [References Scientifiques](#references-scientifiques)
- [Sources et Credits](#sources-et-credits)

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

## Nouveautes v5.0 - Detection Amelioree et Export Professionnel

### 1. Detection de Personnages Amelioree

Le systeme de detection de personnages a ete entierement repense pour eliminer les faux positifs :

**Stop words etendus :** Expansion de 25 a ~200 mots (participes passes, adverbes, noms communs post-verbe, negatifs).

**Scoring de confiance :** Chaque personnage detecte recoit un score de confiance (0.0-1.0) base sur :
- Presence dans le dictionnaire de prenoms francais (+0.3)
- Nombre d'occurrences dans le texte (+0.2)
- Contexte syntaxique (apres verbe de parole, majuscule)

**Dictionnaire de prenoms francais :** ~1000 prenoms INSEE avec detection automatique du genre.

```python
from src.french_names import is_french_name, get_gender_from_name

is_french_name("Marie")      # True
get_gender_from_name("Pierre") # "M"
```

**Validation LLM optionnelle :** Pour les cas ambigus (confiance 0.3-0.7), le systeme peut valider via Ollama/OpenAI.

### 2. Support GPU et Streaming Temps Reel

**Configuration GPU unifiee :**
```python
from src.gpu_config import GPUConfig

config = GPUConfig(
    use_gpu=True,
    device="auto",  # "cuda", "mps", "cpu"
    memory_fraction=0.8
)
print(config.get_device())  # Auto-detect CUDA/MPS/CPU
```

**Moteur Kokoro avec GPU :**
```python
from src.tts_kokoro_engine import KokoroTTSEngine
from src.gpu_config import GPUConfig

engine = KokoroTTSEngine(gpu_config=GPUConfig(use_gpu=True))
print(f"GPU actif: {engine.is_using_gpu}")
```

**Streaming SSE temps reel :**
```bash
# Endpoint streaming
curl -N -X POST http://localhost:8000/api/v2/streaming/synthesize-stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour, ceci est un test de streaming."}'
```

**Interface async pour integration :**
```python
from src.tts_async import AsyncKokoroWrapper

async def stream_audio():
    engine = AsyncKokoroWrapper()
    async for chunk in engine.synthesize_stream("Texte a synthetiser"):
        # chunk.audio contient les samples audio
        play_audio(chunk.audio)
```

### 3. Lecteur Web Integre

**Floating Player global :** Le frontend React dispose maintenant d'un lecteur audio flottant visible sur toutes les pages.

**Playlist avec queue :**
```typescript
// Zustand store avec persistance
const { queue, addTrack, next, toggleShuffle } = usePlaylistStore()

addTrack({ url: '/output/chapitre1.mp3', title: 'Chapitre 1' })
```

**Streaming playback avec Web Audio API :**
```typescript
const { startStreaming, isBuffering, progress } = useStreamingPlayback()

// Demarre le streaming SSE et joue en temps reel
await startStreaming("Texte a synthetiser", { voice: "ff_siwis" })
```

### 4. Effets Sonores Contextuels

**Moteur d'effets proceduraux :**
```python
from src.soundfx_engine import SoundFXEngine, BUILTIN_EFFECTS

engine = SoundFXEngine()

# 12 effets disponibles
audio = engine.generate('whoosh', intensity=0.5, duration=0.3)
audio = engine.generate('impact', intensity=0.7, duration=0.2)
audio = engine.generate('suspense', intensity=0.4, duration=2.0)
```

**Effets integres :**

| Effet | Description | Usage typique |
|-------|-------------|---------------|
| `whoosh` | Balayage frequentiel | Transitions rapides |
| `impact` | Burst de bruit | Moments dramatiques |
| `suspense` | Drone basse frequence | Tension |
| `magic` | Harmoniques + shimmer | Moments fantastiques |
| `page_turn` | Bruit de page | Transitions |
| `chapter_start` | Jingle d'ouverture | Debut de chapitre |

**Integration dans le pipeline HQ :**
```bash
python audio_reader.py livre.md --hq --sound-effects
```

### 5. Voix Francaises Additionnelles

**21 voix Edge TTS francaises :**

| Region | Voix feminines | Voix masculines |
|--------|---------------|-----------------|
| France | Denise, Eloise, Brigitte, Celeste, Coralie, Jacqueline, Josephine, Yvette | Henri, Alain, Claude, Jerome, Maurice, Remy, Yves |
| Canada | Sylvie | Antoine, Jean, Thierry |
| Belgique | Charline | Gerard |
| Suisse | Ariane | Fabrice |

**Voix MMS (Meta) :**
```python
# Voix MMS pour le francais
MMS_VOICES = {
    "fr": [
        {"id": "mms_fr_default", "name": "MMS Francais"},
        {"id": "mms_fr_male", "name": "MMS Homme FR"},
        {"id": "mms_fr_female", "name": "MMS Femme FR"}
    ]
}
```

### 6. Traitement par Lots et Profils

**File d'attente de conversion :**
```python
from src.batch_processor import BatchProcessor, JobPriority

processor = BatchProcessor(max_concurrent=2)

# Ajouter des jobs avec priorites
processor.add_job(Path("livre1.md"), {"hq": True}, JobPriority.HIGH)
processor.add_job(Path("livre2.md"), {"hq": False}, JobPriority.NORMAL)

# Traiter tous les jobs
results = processor.process_all(
    on_job_progress=lambda job, pct: print(f"{job.id}: {pct}%")
)
```

**7 profils de configuration predefinis :**

| Profil | Style | Vitesse | Particularites |
|--------|-------|---------|----------------|
| `podcast` | conversational | 1.05x | -16 LUFS, decontracte |
| `audiobook` | storytelling | 1.0x | Multi-voix, jingles chapitres |
| `dramatic` | dramatic | 0.92x | Effets sonores, intense |
| `fast` | conversational | 1.2x | Sans post-processing |
| `documentary` | documentary | 0.95x | Neutre, informatif |
| `intimate` | intimate | 0.9x | Proche, confidentiel |
| `energetic` | energetic | 1.12x | Dynamique, enthousiaste |

**Utilisation CLI :**
```bash
# Utiliser un profil
python audio_reader.py livre.md --profile audiobook

# Lister les profils
python audio_reader.py --list-profiles

# Batch processing depuis fichier
python audio_reader.py --batch jobs.json
```

### 7. Export Multi-Plateformes

**Export vers Spotify, YouTube, Podcast et ACX :**
```python
from src.platform_exporter import PlatformExporter, ExportMetadata

exporter = PlatformExporter()
metadata = ExportMetadata(
    title="Mon Audiobook",
    author="Auteur",
    narrator="Kokoro TTS"
)

# Export Spotify (MP3 320kbps, -14 LUFS)
result = exporter.export_for_spotify(audio_path, output_path, metadata)

# Export YouTube (video avec waveform)
result = exporter.export_for_youtube(audio_path, output_path, metadata)

# Export Podcast (MP3 128kbps, -16 LUFS, mono)
result = exporter.export_for_podcast(audio_path, output_path, metadata)

# Export ACX/Audible (MP3 192kbps, conformite stricte)
result = exporter.export_for_acx(audio_path, output_path, metadata)
```

**Exigences par plateforme :**

| Plateforme | Format | Loudness | Particularites |
|------------|--------|----------|----------------|
| Spotify | MP3 320kbps | -14 LUFS | Stereo, artwork 3000x3000 |
| YouTube | MP4 | - | Video avec waveform animee |
| Podcast | MP3 128k | -16 LUFS | Mono, tags ID3 |
| ACX/Audible | MP3 192k | -23 to -18 dB RMS | Peak max -3dB, noise floor < -60dB |

### 8. Pipeline LLM Unifie

**Centralisation de toutes les ameliorations basees sur LLM :**

```python
from src.llm_enhancer import create_gemini_enhancer, create_ollama_enhancer

# Avec Gemini 2.5 Flash (rapide, cloud)
enhancer = create_gemini_enhancer(api_key="votre-cle")

# Ou avec Ollama (local, gratuit)
enhancer = create_ollama_enhancer(model="llama3.2")

# Validation de personnage (elimine les faux positifs)
result = enhancer.validate_character_name("coupe", "Il a coupe la parole.")
print(result.is_character)  # False

# Insertion automatique de tags audio
tagged = enhancer.auto_insert_audio_tags("Il murmura doucement...")
# "[whispers] Il murmura doucement..."

# Analyse emotionnelle contextuelle
emotion = enhancer.analyze_emotion_contextual("Soudain, un cri retentit !")
print(emotion.primary_emotion)  # EmotionType.FEAR

# Pipeline complet
result = enhancer.enhance_text_for_tts(
    "Soudain, un cri dechira le silence !",
    insert_tags=True,
    detect_emotions=True,
    suggest_prosody=True
)
```

**Providers supportes :**

| Provider | Type | Modele par defaut |
|----------|------|------------------|
| Ollama | Local, gratuit | llama3.2 |
| OpenAI | Cloud | gpt-4o-mini |
| Anthropic | Cloud | claude-3-haiku |
| **Gemini** | Cloud | **gemini-2.5-flash-preview-05-20** |

**Variables d'environnement :**
```bash
export GEMINI_API_KEY="votre-cle-gemini"
export OPENAI_API_KEY="votre-cle-openai"
export ANTHROPIC_API_KEY="votre-cle-anthropic"
```

### Modules v5.0

| Module | Fichier | Description |
|--------|---------|-------------|
| Prenoms FR | `french_names.py` | ~1000 prenoms INSEE avec genre |
| Config GPU | `gpu_config.py` | Configuration GPU unifiee |
| TTS Async | `tts_async.py` | Interface streaming async |
| Sound FX | `soundfx_engine.py` | Effets sonores proceduraux |
| Batch | `batch_processor.py` | File d'attente avec priorites |
| Profils | `config_profiles.py` | 7 profils predefinis |
| Export | `platform_exporter.py` | Export multi-plateformes |
| Streaming | `api/routers/streaming.py` | Endpoint SSE |
| **LLM Enhancer** | `llm_enhancer.py` | **Pipeline LLM unifie (Gemini/Ollama/OpenAI)** |

---

## Nouveautes v4.0 - Interface Moderne et Nouveaux Moteurs

### Interface React + API v2

AudioReader dispose maintenant d'une **interface web moderne** basee sur React 18, TypeScript et Tailwind CSS v4:

```bash
# Lancer l'interface React + API v2
python audio_reader.py --api-v2
# -> Frontend: http://localhost:5173
# -> API: http://localhost:8000
```

**Fonctionnalites de l'interface:**
- Dashboard avec statistiques et jobs recents
- Conversion de livres avec suivi en temps reel (SSE)
- Galerie de voix avec pre-ecoute
- Gestion des personnages et attribution des voix
- Clonage de voix depuis video
- Serveur podcast integre
- Gestion des fichiers uploades

### Nouveaux Moteurs TTS

Trois nouveaux moteurs TTS gratuits et locaux:

| Moteur | Specialite | Caracteristique |
|--------|------------|-----------------|
| **Chatterbox** | Clonage voix | Controle d'exaggeration emotionnelle (0-1) |
| **Dia 1.6B** | Multi-speakers | Tags natifs `[S1]`/`[S2]`, sons non-verbaux |
| **F5-TTS** | Flow Matching | CPU-friendly, clonage avec ~10s audio |

```bash
# Utiliser Chatterbox avec clonage
python audio_reader.py livre.md --engine chatterbox --clone voix.wav

# Utiliser Dia pour dialogues multi-speakers
python audio_reader.py livre.md --engine dia

# Utiliser F5-TTS
python audio_reader.py livre.md --engine f5 --clone reference.wav
```

**Selecteur intelligent de moteur:**
```bash
# Le moteur optimal est choisi automatiquement
python audio_reader.py livre.md --engine auto
```

### API Compatible OpenAI

Endpoint `/v1/audio/speech` compatible avec le SDK OpenAI:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.audio.speech.create(
    model="kokoro",  # ou "chatterbox", "dia", "f5"
    voice="alloy",   # mappe vers af_bella
    input="Bonjour, ceci est un test."
)
response.stream_to_file("output.mp3")
```

### Mode Analyse (`--dry-run`)

Analysez un livre sans synthese pour estimer temps et ressources:

```bash
python audio_reader.py livre.md --dry-run
```

**Affiche:**
- Nombre de chapitres et mots
- Personnages detectes
- Duree audio estimee
- Temps de traitement estime
- Espace disque requis

### Reprise Apres Interruption (`--resume`)

Reprenez une conversion interrompue exactement ou vous l'avez laissee:

```bash
# Premiere execution (interrompue)
python audio_reader.py livre.md --hq

# Reprise automatique
python audio_reader.py livre.md --hq --resume
```

Le systeme sauvegarde un checkpoint JSON avec le hash MD5 du fichier source.

### Sous-titres Synchronises

Generez des sous-titres SRT ou VTT avec alignement Whisper:

```bash
# Generer sous-titres SRT
python audio_reader.py livre.md --subtitles srt

# Generer sous-titres VTT (Web)
python audio_reader.py livre.md --subtitles vtt
```

### Ambiance Sonore

Ajoutez une musique d'ambiance avec ducking automatique:

```bash
python audio_reader.py livre.md --ambiance rain
python audio_reader.py livre.md --ambiance cafe
python audio_reader.py livre.md --ambiance library
python audio_reader.py livre.md --ambiance forest
python audio_reader.py livre.md --ambiance fireplace
```

### Jingles Inter-Chapitres

Transitions sonores entre chapitres:

```bash
python audio_reader.py livre.md --chapter-jingle chime
python audio_reader.py livre.md --chapter-jingle page_turn
python audio_reader.py livre.md --chapter-jingle orchestral
python audio_reader.py livre.md --chapter-jingle minimal
```

### Simulation Acoustique

Simulez differents environnements d'ecoute:

```python
from src.room_simulator import RoomSimulator

simulator = RoomSimulator()
audio = simulator.process(audio, preset="living_room")
# Presets: studio, living_room, theater, intimate
```

### Detection Automatique de Langue

```bash
# Detection automatique
python audio_reader.py livre.md --language auto
```

Detecte: francais, anglais, allemand, espagnol, italien, portugais.

### Attribution Dialogues par LLM

Utilise Ollama ou OpenAI pour resoudre les attributions ambigues:

```python
from src.llm_dialogue_attributor import LLMDialogueAttributor

attributor = LLMDialogueAttributor(provider="ollama", model="llama3.2")
result = attributor.attribute("Vraiment ?", context="Marie regarda Pierre...")
print(result.speaker)  # "Marie" ou "Pierre"
```

### Nettoyage Intelligent PDF

Nettoie automatiquement les textes mal convertis depuis PDF:

```python
from src.smart_text_cleaner import SmartTextCleaner

cleaner = SmartTextCleaner()
clean_text = cleaner.clean(raw_pdf_text)
# Supprime en-tetes/pieds recurrents, numeros de page, repare cesures
```

### Configuration TOML

Centralisez vos parametres dans `audioreader.toml`:

```toml
[general]
language = "fr"
engine = "kokoro"
voice = "ff_siwis"

[quality]
hq = true
multivoice = true
style = "storytelling"

[audio]
ambiance = "library"
chapter_jingle = "chime"

[postprocessing]
master = true
acx_compliance = true
```

### Docker

```bash
# Build et lancement
docker-compose up -d

# Services:
# - audioreader-api: API v2 sur :8000
# - audioreader-web: Gradio sur :7860
```

---

## Nouveautes v3.0 - Plateforme Complete

### 1. Import Universel (PDF & EPUB)
AudioReader accepte désormais n'importe quel fichier **PDF** ou **EPUB** comme source.
- **Conversion automatique** : Glissez-déposez votre ebook dans l'interface, il est converti en Markdown structuré à la volée.
- **Nettoyage intelligent** : Suppression automatique des en-têtes, pieds de page et numéros de page.

### 2. Clonage de Voix depuis Vidéo 🎙️
Créez votre propre banque de voix à partir de vos films ou vidéos préférés.
- **Extraction Audio** : Uploader un fichier vidéo (MP4, MKV, AVI).
- **Découpage** : Sélectionnez le segment idéal (ex: de 00:10 à 00:25).
- **Enregistrement** : La voix est ajoutée instantanément à votre liste de narrateurs disponibles.

### 3. Casting Acoustique (Pré-écoute) 👂
Plus besoin de choisir "af_bella" à l'aveugle !
- **Bouton Play** : Écoutez un échantillon de chaque voix directement dans l'interface.
- **Génération dynamique** : Si l'échantillon n'existe pas, il est généré à la volée.

### 4. Serveur Podcast RSS 📡
Diffusez vos livres audio sur votre réseau local pour les écouter sur votre téléphone.
- **Flux RSS automatique** : Génère un flux compatible Apple Podcasts / Pocket Casts.
- **QR Code** : Scannez le code affiché dans l'interface pour vous abonner instantanément.
- **Zéro transfert** : Pas besoin de copier les fichiers, tout se fait en streaming Wi-Fi.

### 5. Export Multi-Formats (Ebook + Audio)
Générez en un clic tous les formats pour votre liseuse et votre lecteur audio :
- **Audio** : M4B (avec chapitres) ou MP3.
- **Ebook** : PDF (mise en page soignée), EPUB (compatible Kindle/Kobo), HTML.
- **Archive ZIP** : Téléchargez le tout dans un pack complet.

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

# Installer les dependances de base
pip install -r requirements.txt

# Installer les dependances v3.0 (PDF, EPUB, Podcast)
pip install pymupdf ebooklib beautifulsoup4 qrcode fpdf2

# Telecharger le modele Kokoro (~340 MB)
curl -L -o kokoro-v1.0.onnx "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
curl -L -o voices-v1.0.bin "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
```

### Verifier l'installation

```bash
python audio_reader.py --list-voices
```

---

## Utilisation

### Interface Graphique (Recommande)

La version 3.0 est conçue pour être utilisée via l'interface Web qui regroupe tous les outils (Import, Clonage, Casting, Podcast).

```bash
# Lancer l'interface
python app.py
# ou
python audio_reader.py --gui
```

Ouvrez ensuite **http://localhost:7860** dans votre navigateur.

### Synthese standard (CLI)

La commande par defaut utilise `MMS-TTS` pour le francais ou `Kokoro` pour l'anglais.

```bash
# Convertir un livre avec les reglages par defaut
python audio_reader.py mon_livre.md

# Specifier la langue et la voix
python audio_reader.py mon_livre.md --language en --voice af_heart

# Changer la vitesse
python audio_reader.py mon_livre.md --speed 1.2
```

### Clonage de voix (XTTS)

Pour utiliser le clonage de voix avec le moteur XTTS-v2.

> **Important :** La librairie TTS (Coqui) requiert Python 3.10 ou 3.11 (pas 3.12+).
> Si votre environnement principal utilise Python 3.12+, creez un venv dedie :

```bash
# Creer un environnement dedie Python 3.10
python3.10 -m venv venv_xtts
source venv_xtts/bin/activate

# Installer PyTorch (CPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Installer TTS (inclut XTTS-v2)
pip install TTS

# Downgrade transformers si erreur "BeamSearchScorer"
pip install "transformers>=4.33.0,<4.50.0"
```

**Utilisation :**

```bash
# Activer l'environnement XTTS
source venv_xtts/bin/activate

# Cloner une voix a partir d'un fichier audio (min 6s)
python audio_reader.py mon_livre.md --engine xtts --clone ma_voix.wav

# Multi-clonage (HQ mode)
python audio_reader.py mon_livre.md --hq --multivoice --clone narrateur.wav
```

**Note sur XTTS multi-locuteurs :** XTTS-v2 permet d'utiliser plusieurs voix clonees dans le meme livre. Le pipeline HQ gere automatiquement l'attribution des voix aux differents personnages.

### Synthese Haute Qualite (v2.4)

Utilisez le flag `--hq` pour activer le pipeline etendu avec toutes les fonctionnalites avancees (multi-voix, emotions, styles).

```bash
# Conversion HQ avec mastering broadcast
python audio_reader.py mon_livre.md --hq --master

# HQ avec detection automatique des personnages (multi-voix)
python audio_reader.py mon_livre.md --hq --multivoice

# HQ avec style de narration specifique
python audio_reader.py mon_livre.md --hq --style dramatic

# La totale: HQ, Multi-voix, Style et Mastering
python audio_reader.py mon_livre.md --hq --multivoice --style storytelling --master
```

**Options HQ disponibles :**
- `--hq` : Active le pipeline etendu (Kokoro v2.4).
- `--multivoice` : Analyse le texte pour attribuer des voix differentes aux personnages (Alice, Bob, etc.).
- `--style [storytelling|formal|conversational|dramatic]` : Definit le ton general.
- `--master` : Applique la chaine de mastering professionnelle (ACX/Audible compliance).
- `--no-cache` : Desactive le cache de synthese pour regenerer tous les segments.

---

### Mode Haute Qualite (`audio_reader.py --hq`) - NOUVEAU

```bash
# Pipeline HQ complet (multi-voix + emotions + post-processing)
python audio_reader.py livre.md --hq -o output/

# HQ avec multi-voix et mastering
python audio_reader.py livre.md --hq --multivoice --master

# HQ avec style specifique
python audio_reader.py livre.md --hq --style dramatic
```

### Voice Blending (melange de voix)

```bash
# Melange 60% Bella, 40% Adam
python audio_reader.py livre.md --voice "af_bella:60,am_adam:40"

# Melange 50-50
python audio_reader.py livre.md --voice "af_bella,am_adam"
```

### Controle des pauses

```bash
# Pauses plus longues entre paragraphes (1 seconde)
python audio_reader.py livre.md --paragraph-pause 1.0

# Pauses plus courtes entre phrases (0.2 secondes)
python audio_reader.py livre.md --sentence-pause 0.2
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
├── audio_reader.py         # Script CLI principal (Standard & HQ avec --hq)
├── app.py                  # Interface Web Gradio (v3.0 complete)
├── api_server.py           # API REST FastAPI v1 (pour ChatGPT)
├── mcp_server.py           # Serveur MCP (pour Claude Desktop)
├── postprocess.py          # Post-traitement audio legacy
├── run_tests.py            # Lanceur de tests pytest
├── kokoro-v1.0.onnx        # Modele TTS (310 MB)
├── voices-v1.0.bin         # Donnees voix (27 MB)
├── audioreader.example.toml # Configuration TOML exemple
│
├── api/                    # API v2 (FastAPI)
│   ├── __init__.py         # Factory create_app()
│   ├── dependencies.py     # JobStore, constantes
│   ├── models.py           # Pydantic schemas
│   └── routers/
│       ├── config.py       # /api/v2/health, /api/v2/config
│       ├── generation.py   # /api/v2/generate/*
│       ├── jobs.py         # /api/v2/jobs/* + SSE
│       ├── voices.py       # /api/v2/voices/*
│       ├── files.py        # /api/v2/files/*
│       ├── analysis.py     # /api/v2/analyze
│       ├── podcast.py      # /api/v2/podcast/*
│       ├── projects.py     # /api/v2/projects/*
│       ├── openai_compat.py # /v1/audio/speech (OpenAI API)
│       └── streaming.py    # /api/v2/streaming/* (SSE temps reel v5.0)
│
├── frontend/               # Interface React (v4.0 + v5.0)
│   ├── src/
│   │   ├── pages/          # DashboardPage, BookConversionPage, etc.
│   │   ├── components/     # UI components (Button, Card, etc.)
│   │   │   └── global/     # FloatingPlayer (v5.0)
│   │   ├── stores/         # Zustand stores
│   │   │   └── usePlaylistStore.ts  # Playlist avec queue (v5.0)
│   │   ├── api/            # Client API + SSE
│   │   └── hooks/          # React hooks
│   │       └── useStreamingPlayback.ts  # Web Audio API (v5.0)
│   ├── package.json
│   └── vite.config.ts
│
├── src/
│   │   # --- MOTEURS TTS ---
│   ├── tts_engine.py           # Wrapper unifie (auto-selection)
│   ├── tts_kokoro_engine.py    # Kokoro-82M (qualite ElevenLabs)
│   ├── tts_mms_engine.py       # MMS-TTS Meta (1000+ langues)
│   ├── tts_xtts_engine.py      # XTTS-v2 (clonage voix)
│   ├── tts_hybrid_engine.py    # Hybride MMS+Kokoro + export MP3
│   ├── tts_multivoice_xtts.py  # Multi-voix XTTS
│   ├── tts_unified.py          # Abstraction TTS
│   │
│   │   # --- TRAITEMENT TEXTE ---
│   ├── markdown_parser.py      # Parser MD, EPUB, multi-fichiers
│   ├── text_normalizer.py      # Nombres, dates, symboles -> mots
│   ├── text_processor.py       # Chunking, corrections prononciation
│   ├── french_preprocessor.py  # Normalisation specifique francais
│   ├── advanced_preprocessor.py # Pipeline preprocessing complet
│   │
│   │   # --- ANALYSE PERSONNAGES & EMOTIONS ---
│   ├── character_detector.py   # Detection dialogues et locuteurs
│   ├── dialogue_detector.py    # Segmentation dialogue/narration
│   ├── dialogue_attribution.py # Attribution QUI parle (v2.4)
│   ├── emotion_analyzer.py     # Analyse sentiment + prosodie
│   ├── emotion_continuity.py   # Transitions emotionnelles fluides
│   ├── emotion_control.py      # Controle intensite + phonemes IPA
│   ├── narrative_context.py    # Type narratif (action, description...)
│   ├── narration_styles.py     # 7 styles (storytelling, dramatic...)
│   │
│   │   # --- BIO-ACOUSTIQUE & PROSODIE (v2.3) ---
│   ├── bio_acoustics.py        # Respirations synthetiques realistes
│   ├── breath_samples.py       # Gestionnaire samples/synthese hybride
│   ├── intonation_contour.py   # Contours prosodiques phrase-level
│   ├── timing_humanizer.py     # Micro-variations de timing
│   ├── dynamic_voice.py        # Voice blending selon emotion
│   ├── word_level_control.py   # Controle SSML-like mot-par-mot
│   │
│   │   # --- AUDIO TAGS & MORPHING (v2.1) ---
│   ├── audio_tags.py           # Tags style ElevenLabs v3 (50+)
│   ├── voice_morphing.py       # Pitch, formant, time stretch
│   ├── voice_cloning.py        # Gestionnaire clonage XTTS
│   │
│   │   # --- PIPELINES ---
│   ├── hq_pipeline.py          # Pipeline HQ base (v2.0)
│   ├── hq_pipeline_extended.py # Pipeline HQ etendu (v2.1-2.4)
│   ├── synthesis_cache.py      # Cache intelligent + parallelisation
│   ├── conversation_generator.py # Dialogues multi-speakers
│   │
│   │   # --- POST-TRAITEMENT AUDIO ---
│   ├── audio_enhancer.py       # EQ, compression, loudness (ffmpeg)
│   ├── audio_postprocess.py    # Pipeline post-prod modular
│   ├── audio_processor.py      # Traitement in-memory (numpy)
│   ├── audio_crossfade.py      # Crossfade entre segments
│   ├── acx_compliance.py       # Conformite ACX/Audible (v2.4)
│   │
│   │   # --- EXPORT & EMPAQUETAGE ---
│   ├── audiobook_builder.py    # M4B avec chapitres + ID3
│   ├── audiobook_packager.py   # Alternative empaquetage
│   ├── book_exporter.py        # PDF, EPUB, HTML, TXT
│   │
│   │   # --- MODULES v3.0 ---
│   ├── input_converter.py      # PDF/EPUB -> Markdown
│   ├── audio_extractor.py      # Video -> WAV (ffmpeg)
│   ├── podcast_server.py       # Serveur RSS local + QR code
│   │
│   │   # --- MODULES v4.0 ---
│   ├── tts_chatterbox_engine.py # Chatterbox TTS (clonage + emotion)
│   ├── tts_dia_engine.py       # Dia 1.6B (multi-speakers)
│   ├── tts_f5_engine.py        # F5-TTS (flow matching)
│   ├── engine_selector.py      # Selection intelligente moteur
│   ├── time_estimator.py       # Estimation temps conversion
│   ├── progress_checkpoint.py  # Reprise apres interruption
│   ├── config_loader.py        # Configuration TOML
│   ├── subtitle_generator.py   # Generation SRT/VTT
│   ├── ambiance_engine.py      # Ambiance sonore (rain, cafe...)
│   ├── chapter_jingles.py      # Jingles inter-chapitres
│   ├── room_simulator.py       # Simulation acoustique
│   ├── silence_optimizer.py    # Optimisation silences
│   ├── language_detector.py    # Detection auto langue
│   ├── llm_dialogue_attributor.py # Attribution dialogues LLM
│   ├── smart_text_cleaner.py   # Nettoyage PDF intelligent
│   ├── chapter_summarizer.py   # Resume chapitres (metadata M4B)
│   │
│   │   # --- MODULES v5.0 ---
│   ├── french_names.py         # Dictionnaire prenoms francais (~1000)
│   ├── gpu_config.py           # Configuration GPU unifiee (CUDA/MPS/CPU)
│   ├── tts_async.py            # Interface TTS async/streaming
│   ├── soundfx_engine.py       # Effets sonores proceduraux (12 effets)
│   ├── batch_processor.py      # Traitement par lots avec priorites
│   ├── config_profiles.py      # Profils de configuration (7 predefinis)
│   ├── platform_exporter.py    # Export Spotify/YouTube/Podcast/ACX
│   │
│   │   # --- UTILITAIRES ---
│   ├── preview_generator.py    # Apercu 30 secondes
│   ├── corrections_loader.py   # Charge glossaires JSON
│   ├── corrections_ui.py       # Interface Gradio corrections
│   └── llm_emotion_detector.py # Detection emotion par LLM (Ollama/OpenAI)
│
├── examples/               # Exemples d'utilisation
│   ├── generate_audiobook.py
│   ├── example_multivoix.py
│   └── config_multivoix_example.json
│
├── tests/                  # Tests unitaires (pytest)
│   ├── conftest.py         # Fixtures pytest
│   └── test_*.py           # 21 fichiers de tests
│
├── docs/
│   └── FINE_TUNING_OPTIONS.md
├── voice_samples/          # Samples pour clonage
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
| **Chatterbox** | Resemble AI voice cloning | Apache 2.0 |
| **Dia 1.6B** | Nari Labs multi-speaker | Apache 2.0 |
| **F5-TTS** | Flow matching TTS | MIT |

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

*AudioReader v5.0 - Detection personnages amelioree, streaming temps reel, effets sonores et export multi-plateformes*

---

## FAQ

### Quelle est la difference entre le mode standard et le mode HQ ?

- **Mode standard** : Une seule voix, traitement rapide, ideal pour les essais.
- **Mode HQ (`--hq`)** : Pipeline haute qualite avec multi-voix automatique, analyse emotionnelle, et post-processing broadcast.

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

Puis : `python audio_reader.py livre.md --hq --config config.json`

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

## Integration IA (Claude / ChatGPT)

AudioReader peut etre pilote par des assistants IA via deux interfaces :

| Interface | Protocole | Usage |
|-----------|-----------|-------|
| **MCP Server** | Model Context Protocol | Claude Desktop, Claude.ai |
| **REST API** | HTTP/OpenAPI | ChatGPT Actions, GPTs, clients HTTP |

### Demarrage rapide

```bash
# API REST (ChatGPT, clients HTTP)
pip install fastapi uvicorn python-multipart
python api_server.py
# -> http://localhost:8000

# MCP Server (Claude Desktop)
pip install mcp
# Configurer dans claude_desktop_config.json (voir INTEGRATION.md)
```

Documentation complete : **[INTEGRATION.md](INTEGRATION.md)**

---

## Docker

### Build et Lancement

```bash
# Construire l'image
docker build -t audioreader .

# Lancer avec docker-compose
docker-compose up -d

# Services disponibles:
# - audioreader-api : http://localhost:8000 (API v2)
# - audioreader-web : http://localhost:7860 (Gradio)
```

### Volumes

```yaml
volumes:
  - ./books:/app/books          # Livres source
  - ./output:/app/output        # Audio genere
  - ./voice_samples:/app/voice_samples  # Samples clonage
  - models:/app/models          # Modeles TTS (persistant)
```

### Variables d'Environnement

| Variable | Description | Defaut |
|----------|-------------|--------|
| `AUDIOREADER_ENGINE` | Moteur TTS par defaut | `kokoro` |
| `AUDIOREADER_LANGUAGE` | Langue par defaut | `fr` |
| `AUDIOREADER_VOICE` | Voix par defaut | `ff_siwis` |

---

## CI/CD

### GitHub Actions

Workflows automatises pour tests et releases:

```yaml
# .github/workflows/tests.yml
# - Lint avec ruff
# - Tests pytest sur Python 3.10, 3.11, 3.12
# - Rapport de couverture

# .github/workflows/release.yml
# - Build Docker image
# - Create GitHub Release
# - Trigger sur tags v*
```

### Pre-commit Hooks

```bash
# Installer les hooks
pip install pre-commit
pre-commit install

# Hooks actifs:
# - ruff (lint + format)
# - trailing-whitespace
# - end-of-file-fixer
# - check-yaml, check-json
```

---

## References Scientifiques
