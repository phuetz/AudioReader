"""
Configuration TOML centralisee pour AudioReader.

Charge audioreader.toml si present, sinon utilise les valeurs par defaut.
Les arguments CLI ont toujours priorite sur le fichier de configuration.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any

# Python 3.11+ : tomllib dans stdlib
# Python 3.10 : fallback sur tomli
try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]


CONFIG_FILENAME = "audioreader.toml"


@dataclass
class AudioReaderConfig:
    """Configuration centralisee d'AudioReader."""

    # [general]
    language: str = "fr"
    engine: str = "auto"
    voice: str = "ff_siwis"
    speed: float = 1.0
    header_level: int = 1
    output_dir: Optional[str] = None

    # [quality]
    hq: bool = False
    multivoice: bool = False
    style: str = "storytelling"
    master: bool = False
    use_cache: bool = True

    # [voices] mapping personnage -> voice_id
    voice_mapping: Dict[str, str] = field(default_factory=dict)

    # [audio]
    sample_rate: int = 24000
    crossfade_ms: int = 50
    ambiance: Optional[str] = None
    chapter_jingle: Optional[str] = None

    # [postprocessing]
    normalize_loudness: bool = True
    target_lufs: float = -20.0
    acx_compliance: bool = False

    # [subtitles]
    subtitles: Optional[str] = None  # "srt", "vtt", or None
    word_level: bool = False


def load_config(config_path: Optional[Path] = None) -> AudioReaderConfig:
    """
    Charge la configuration depuis un fichier TOML.

    Args:
        config_path: Chemin vers le fichier TOML (defaut: audioreader.toml)

    Returns:
        AudioReaderConfig avec les valeurs du fichier ou les defauts
    """
    if tomllib is None:
        return AudioReaderConfig()

    path = config_path or Path(CONFIG_FILENAME)
    if not path.exists():
        return AudioReaderConfig()

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        return AudioReaderConfig()

    config = AudioReaderConfig()

    # [general]
    general = data.get("general", {})
    if "language" in general:
        config.language = general["language"]
    if "engine" in general:
        config.engine = general["engine"]
    if "voice" in general:
        config.voice = general["voice"]
    if "speed" in general:
        config.speed = float(general["speed"])
    if "header_level" in general:
        config.header_level = int(general["header_level"])
    if "output_dir" in general:
        config.output_dir = general["output_dir"]

    # [quality]
    quality = data.get("quality", {})
    if "hq" in quality:
        config.hq = bool(quality["hq"])
    if "multivoice" in quality:
        config.multivoice = bool(quality["multivoice"])
    if "style" in quality:
        config.style = quality["style"]
    if "master" in quality:
        config.master = bool(quality["master"])
    if "use_cache" in quality:
        config.use_cache = bool(quality["use_cache"])

    # [voices]
    voices = data.get("voices", {})
    mapping = voices.get("mapping", {})
    if mapping:
        config.voice_mapping = dict(mapping)

    # [audio]
    audio = data.get("audio", {})
    if "sample_rate" in audio:
        config.sample_rate = int(audio["sample_rate"])
    if "crossfade_ms" in audio:
        config.crossfade_ms = int(audio["crossfade_ms"])
    if "ambiance" in audio:
        config.ambiance = audio["ambiance"]
    if "chapter_jingle" in audio:
        config.chapter_jingle = audio["chapter_jingle"]

    # [postprocessing]
    pp = data.get("postprocessing", {})
    if "normalize_loudness" in pp:
        config.normalize_loudness = bool(pp["normalize_loudness"])
    if "target_lufs" in pp:
        config.target_lufs = float(pp["target_lufs"])
    if "acx_compliance" in pp:
        config.acx_compliance = bool(pp["acx_compliance"])

    # [subtitles]
    subs = data.get("subtitles", {})
    if "format" in subs:
        config.subtitles = subs["format"]
    if "word_level" in subs:
        config.word_level = bool(subs["word_level"])

    return config


def merge_cli_args(config: AudioReaderConfig, args) -> AudioReaderConfig:
    """
    Fusionne les arguments CLI avec la config TOML.
    Les arguments CLI ont priorite (sauf s'ils sont a leur valeur par defaut).

    Args:
        config: Configuration TOML chargee
        args: argparse.Namespace

    Returns:
        Configuration fusionnee
    """
    # On ne remplace que si l'utilisateur a explicitement passe l'argument
    if hasattr(args, "language") and args.language != "fr":
        config.language = args.language
    if hasattr(args, "engine") and args.engine != "auto":
        config.engine = args.engine
    if hasattr(args, "voice") and args.voice != "ff_siwis":
        config.voice = args.voice
    if hasattr(args, "speed") and args.speed != 1.0:
        config.speed = args.speed
    if hasattr(args, "header_level") and args.header_level != 1:
        config.header_level = args.header_level
    if hasattr(args, "output") and args.output is not None:
        config.output_dir = str(args.output)
    if hasattr(args, "hq") and args.hq:
        config.hq = True
    if hasattr(args, "multivoice") and args.multivoice:
        config.multivoice = True
    if hasattr(args, "style") and args.style != "storytelling":
        config.style = args.style
    if hasattr(args, "master") and args.master:
        config.master = True
    if hasattr(args, "use_cache") and not args.use_cache:
        config.use_cache = False

    return config
