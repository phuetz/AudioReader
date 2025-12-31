# Fichiers CLI Obsolètes

Ces fichiers ont été remplacés par `audio_reader.py` dans le dossier racine.

## Point d'entrée unique

Utilisez `audio_reader.py` avec les options appropriées :

```bash
# Standard (équivalent à audio_reader_kokoro.py)
python audio_reader.py livre.md

# Haute qualité (équivalent à audioreader_hq.py)
python audio_reader.py livre.md --hq

# Avec mastering ACX
python audio_reader.py livre.md --hq --master

# Multi-voix
python audio_reader.py livre.md --hq --multivoice
```

## Fichiers archivés

| Fichier | Remplacé par |
|---------|--------------|
| `audioreader.py` | `audio_reader.py` |
| `audioreader_hq.py` | `audio_reader.py --hq` |
| `audio_reader_hq.py` | `audio_reader.py --hq` (moteurs expérimentaux) |
| `audio_reader_kokoro.py` | `audio_reader.py` |

## Notes

- `audio_reader_hq.py` contenait des moteurs expérimentaux (Chatterbox, F5-TTS, Orpheus) non intégrés au pipeline principal
- Ces fichiers sont conservés pour référence historique
