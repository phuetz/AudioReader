# Design Doc: Intégration CLI Pipeline HQ

Ce document décrit le plan pour intégrer les fonctionnalités avancées du pipeline HQ (`HQPipelineExtended`) dans l'interface en ligne de commande principale (`audio_reader.py`).

## Problème Actuel
`audio_reader.py` utilise `UnifiedTTSEngine`, qui offre une synthèse basique. Les fonctionnalités avancées (dialogues, émotions, styles, mastering) développées dans `src/hq_pipeline_extended.py` ne sont accessibles que via des scripts Python manuels.

## Solution Proposée
Refondre `audio_reader.py` pour supporter deux modes de fonctionnement :
1.  **Mode Standard** (Legacy/Rapide) : Utilise `UnifiedTTSEngine` (comportement actuel).
2.  **Mode HQ** (Avancé) : Utilise `HQPipelineExtended` et expose ses fonctionnalités.

## Nouveaux Arguments CLI

```bash
# Activation du mode HQ
python audio_reader.py livre.md --hq

# Options spécifiques HQ
python audio_reader.py livre.md --hq \
  --multivoice \
  # Active la détection de dialogues et multi-voix
  --style thriller \
  # Style de narration (defaut: audiobook)
  --master \
  # Active le mastering audio final
  --output-format m4b \
  # Export direct en livre audio chapitré
```

## Changements Techniques

### 1. Import Conditionnel
Importer `src.hq_pipeline_extended` uniquement si le mode `--hq` est activé (pour éviter de charger des dépendances lourdes inutilement si non utilisées).

### 2. Configuration Mapping
Créer une fonction qui traduit les arguments `argparse` en `HQPipelineConfig`:

```python
def config_from_args(args) -> HQPipelineConfig:
    return HQPipelineConfig(
        language=args.language,
        voice_id=args.voice,
        # Options HQ
        enable_multivoice=args.multivoice,
        enable_emotion=True,  # Par défaut en HQ
        mastering_preset="audiobook" if args.master else None,
        # ...
    )
```

### 3. Gestion des Dépendances
Vérifier que les modèles nécessaires (Kokoro, Voice Cloning) sont téléchargés/disponibles avant de lancer le pipeline HQ, et guider l'utilisateur si nécessaire.

### 4. Feedback Utilisateur
Le pipeline HQ étant plus lent (analyse de texte, multiples passes), intégrer une barre de progression détaillée (ex: "Analyse ch.1", "Génération segments", "Mastering").

## Étapes de Migration

1.  Créer une branche/fichier de test `audio_reader_v2.py`.
2.  Implémenter le switch Standard/HQ.
3.  Tester la conversion complète d'un petit chapitre avec `--hq`.
4.  Remplacer `audio_reader.py` une fois validé.
