# Pistes d'Amélioration AudioReader

## État du Projet (Mise à jour v2.4)

Le projet a considérablement évolué par rapport à la roadmap initiale. De nombreuses fonctionnalités "futures" sont désormais implémentées dans `src/hq_pipeline_extended.py` et les modules associés.

### Fonctionnalités Implémentées (✅)

- [x] **Post-traitement Audio** (Mastering, EQ, Loudness) - *via `src/audio_enhancer.py`*
- [x] **Chunking Intelligent** (Paragraphes > Phrases > Clauses) - *via `src/text_processor.py`*
- [x] **Correction Prononciation** (Dictionnaire FR/EN) - *via `src/text_processor.py`*
- [x] **Détection Émotion** (Rule-based & LLM) - *via `src/emotion_control.py` & `src/llm_emotion_detector.py`*
- [x] **Multi-voix / Dialogues** (Attribution automatique) - *via `src/dialogue_attribution.py`*
- [x] **Voice Morphing & Cloning** (XTTS-v2) - *via `src/voice_cloning.py`*
- [x] **Bio-acoustique** (Respiration, hésitations) - *via `src/bio_acoustics.py`*
- [x] **Contrôle Intonation** (Courbes prosodiques) - *via `src/intonation_contour.py`*

---

## 1. Intégration CLI (Priorité Haute)

**État actuel:** Le script principal `audio_reader.py` utilise le moteur basique `UnifiedTTSEngine` et n'expose pas les fonctionnalités avancées (multi-voix, styles, bio-acoustique) présentes dans le pipeline HQ.

### Objectifs
- Mettre à jour `audio_reader.py` pour utiliser `HQPipelineExtended` via un flag `--hq` ou par défaut.
- Exposer les options avancées en ligne de commande :
  - `--multivoice` : Active la détection et l'attribution des dialogues.
  - `--style` : Sélectionne un style de narration (ex: "thriller", "children").
  - `--master` : Active le post-traitement audio.

## 2. Interface Utilisateur & Expérience

- [ ] **Barre de progression enrichie** : Utiliser `tqdm` pour une estimation précise du temps restant (ETA).
- [ ] **Mode Interactif** : Permettre à l'utilisateur de valider l'attribution des voix avant la génération (via un fichier de config temporaire ou prompt CLI).
- [ ] **Interface Web (Gradio)** : Créer une interface graphique simple pour uploader un Markdown et écouter le résultat (évoqué dans l'ancienne roadmap).

## 3. Qualité & Tests

- [ ] **Validation LLM** : Améliorer la détection des émotions et l'attribution des dialogues en utilisant des LLM locaux (llama.cpp) ou API pour les cas ambigus.
- [ ] **Tests de non-régression** : Créer un set d'échantillons "Golden Master" pour vérifier que les mises à jour n'altèrent pas la qualité de la voix.

## 4. Packaging & Distribution

- [ ] **Métadonnées Avancées** : Support complet des chapitres M4B avec métadonnées enrichies (cover art, auteurs).
- [ ] **Docker** : Conteneuriser l'application pour faciliter l'installation (dépendances système comme ffmpeg, espeak).

---

## Roadmap Révisée

### Phase 1 : Consolidation (Court terme)
- [ ] Mettre à jour `audio_reader.py` pour supporter le pipeline HQ.
- [ ] Nettoyer les scripts de démo et de test obsolètes à la racine.
- [ ] Améliorer la documentation utilisateur (`README.md`) pour expliquer les fonctionnalités avancées.

### Phase 2 : Interface & Accessibilité (Moyen terme)
- [ ] Interface Web locale (Gradio/Streamlit).
- [ ] Outil de correction interactif pour les dialogues (TUI ou Web).

### Phase 3 : IA & Raffinement (Long terme)
- [ ] Fine-tuning de Kokoro pour le français (actuellement via MMS ou mapping phonétique).
- [ ] Analyse sémantique profonde du livre entier pour une cohérence émotionnelle globale.