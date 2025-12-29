"""
Post-production audio pour audiobooks.

Fonctionnalités:
- Normalisation audio (loudness)
- Fade in/out
- Suppression du silence
- Ajout de pauses entre chapitres
- Compression dynamique
"""

import subprocess
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import shutil


@dataclass
class PostProcessConfig:
    """Configuration de post-production."""
    # Normalisation
    normalize: bool = True
    target_loudness: float = -20.0  # LUFS (standard audiobook)
    true_peak: float = -3.0  # dB
    loudness_range: float = 11.0  # LU

    # Fades
    fade_in_ms: int = 500  # Fade in au début
    fade_out_ms: int = 1000  # Fade out à la fin

    # Silences - ATTENTION: trim_silence peut être trop agressif
    # Désactivé par défaut après tests (réduisait 2:50 à 1.34s)
    trim_silence: bool = False
    silence_threshold: float = -60.0  # dB (plus bas = moins agressif)
    min_silence_duration: float = 1.0  # secondes (plus long = moins agressif)

    # Pauses
    chapter_pause_ms: int = 2000  # Pause entre chapitres
    intro_silence_ms: int = 750  # Silence au début

    # Compression dynamique (réduit la différence entre sons forts/faibles)
    compress: bool = False
    compression_ratio: float = 3.0


def run_ffmpeg(cmd: List[str], verbose: bool = False) -> bool:
    """Exécute une commande ffmpeg."""
    if not verbose:
        cmd = cmd[:1] + ['-loglevel', 'warning', '-y'] + cmd[1:]
    else:
        cmd = cmd[:1] + ['-y'] + cmd[1:]

    result = subprocess.run(cmd, capture_output=not verbose)
    return result.returncode == 0


def normalize_audio(
    input_file: str,
    output_file: str,
    target_lufs: float = -20.0,
    true_peak: float = -3.0,
    loudness_range: float = 11.0,
    verbose: bool = False
) -> bool:
    """
    Normalise l'audio selon les standards de broadcast.

    Args:
        input_file: Fichier d'entrée
        output_file: Fichier de sortie
        target_lufs: Loudness cible en LUFS (défaut: -20 pour audiobooks)
        true_peak: Peak maximum en dB
        loudness_range: Range de loudness en LU
        verbose: Mode verbeux

    Returns:
        True si succès
    """
    filter_str = f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={loudness_range}"

    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-af', filter_str,
        output_file
    ]

    return run_ffmpeg(cmd, verbose)


def add_fades(
    input_file: str,
    output_file: str,
    fade_in_ms: int = 500,
    fade_out_ms: int = 1000,
    verbose: bool = False
) -> bool:
    """
    Ajoute des fades in/out à l'audio.

    Args:
        input_file: Fichier d'entrée
        output_file: Fichier de sortie
        fade_in_ms: Durée du fade in en ms
        fade_out_ms: Durée du fade out en ms
    """
    # Obtenir la durée du fichier
    probe_cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', input_file
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False

    import json
    data = json.loads(result.stdout)
    duration = float(data['format']['duration'])

    fade_in_s = fade_in_ms / 1000
    fade_out_s = fade_out_ms / 1000
    fade_out_start = duration - fade_out_s

    filter_str = f"afade=t=in:st=0:d={fade_in_s},afade=t=out:st={fade_out_start}:d={fade_out_s}"

    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-af', filter_str,
        output_file
    ]

    return run_ffmpeg(cmd, verbose)


def add_silence(
    input_file: str,
    output_file: str,
    intro_ms: int = 750,
    outro_ms: int = 2000,
    verbose: bool = False
) -> bool:
    """
    Ajoute du silence au début et à la fin.

    Args:
        input_file: Fichier d'entrée
        output_file: Fichier de sortie
        intro_ms: Silence au début en ms
        outro_ms: Silence à la fin en ms
    """
    intro_s = intro_ms / 1000
    outro_s = outro_ms / 1000

    filter_str = f"adelay={intro_ms}|{intro_ms},apad=pad_dur={outro_s}"

    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-af', filter_str,
        output_file
    ]

    return run_ffmpeg(cmd, verbose)


def trim_silence_edges(
    input_file: str,
    output_file: str,
    threshold_db: float = -50.0,
    min_duration: float = 0.5,
    verbose: bool = False
) -> bool:
    """
    Supprime le silence au début et à la fin.

    Args:
        input_file: Fichier d'entrée
        output_file: Fichier de sortie
        threshold_db: Seuil de détection du silence en dB
        min_duration: Durée minimum de silence pour trigger
    """
    filter_str = f"silenceremove=start_periods=1:start_threshold={threshold_db}dB:start_duration={min_duration}:stop_periods=1:stop_threshold={threshold_db}dB:stop_duration={min_duration}"

    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-af', filter_str,
        output_file
    ]

    return run_ffmpeg(cmd, verbose)


def apply_compression(
    input_file: str,
    output_file: str,
    ratio: float = 3.0,
    threshold_db: float = -20.0,
    verbose: bool = False
) -> bool:
    """
    Applique une compression dynamique.

    Réduit la différence entre les sons forts et faibles,
    rendant l'écoute plus confortable (surtout en voiture/transport).

    Args:
        input_file: Fichier d'entrée
        output_file: Fichier de sortie
        ratio: Ratio de compression (ex: 3.0 = 3:1)
        threshold_db: Seuil de compression
    """
    filter_str = f"acompressor=threshold={threshold_db}dB:ratio={ratio}:attack=5:release=50"

    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-af', filter_str,
        output_file
    ]

    return run_ffmpeg(cmd, verbose)


def convert_to_mp3(
    input_file: str,
    output_file: str,
    bitrate: str = "192k",
    sample_rate: int = 44100,
    verbose: bool = False
) -> bool:
    """
    Convertit un fichier audio en MP3.

    Args:
        input_file: Fichier d'entrée (WAV)
        output_file: Fichier de sortie (MP3)
        bitrate: Bitrate audio (défaut: 192k)
        sample_rate: Sample rate (défaut: 44100)
        verbose: Mode verbeux

    Returns:
        True si succès
    """
    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-codec:a', 'libmp3lame',
        '-b:a', bitrate,
        '-ar', str(sample_rate),
        output_file
    ]

    return run_ffmpeg(cmd, verbose)


def get_audio_duration(audio_file: str) -> float:
    """
    Récupère la durée d'un fichier audio en secondes.
    Utilise ffprobe, robuste pour WAV/MP3/M4A.

    Args:
        audio_file: Chemin du fichier audio

    Returns:
        Durée en secondes
    """
    import json
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', audio_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0

    data = json.loads(result.stdout)
    return float(data['format'].get('duration', 0))


def apply_anti_click_fades(
    input_file: str,
    output_file: str,
    fade_ms: int = 10,
    verbose: bool = False
) -> bool:
    """
    Applique des micro-fades pour éviter les clicks aux jonctions.

    Args:
        input_file: Fichier d'entrée
        output_file: Fichier de sortie
        fade_ms: Durée du fade en ms (défaut: 10ms)
        verbose: Mode verbeux

    Returns:
        True si succès
    """
    import json
    # Obtenir la durée du fichier
    probe_cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', input_file
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False

    data = json.loads(result.stdout)
    duration = float(data['format']['duration'])

    fade_s = fade_ms / 1000
    fade_out_start = max(0, duration - fade_s)

    filter_str = f"afade=t=in:st=0:d={fade_s},afade=t=out:st={fade_out_start}:d={fade_s}"

    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-af', filter_str,
        output_file
    ]

    return run_ffmpeg(cmd, verbose)


def postprocess_audio(
    input_file: str,
    output_file: str,
    config: Optional[PostProcessConfig] = None,
    verbose: bool = False
) -> bool:
    """
    Applique la post-production complète à un fichier audio.

    Args:
        input_file: Fichier d'entrée (WAV/MP3)
        output_file: Fichier de sortie
        config: Configuration de post-production
        verbose: Mode verbeux

    Returns:
        True si succès
    """
    if config is None:
        config = PostProcessConfig()

    # Créer un répertoire temporaire pour les étapes intermédiaires
    with tempfile.TemporaryDirectory() as tmpdir:
        current_file = input_file
        step = 0

        def next_temp() -> str:
            nonlocal step
            step += 1
            return os.path.join(tmpdir, f"step_{step}.wav")

        # Étape 1: Trim silence
        if config.trim_silence:
            if verbose:
                print("  → Suppression des silences...")
            next_file = next_temp()
            if trim_silence_edges(current_file, next_file,
                                   config.silence_threshold,
                                   config.min_silence_duration, verbose):
                current_file = next_file

        # Étape 2: Compression dynamique
        if config.compress:
            if verbose:
                print("  → Compression dynamique...")
            next_file = next_temp()
            if apply_compression(current_file, next_file,
                                  config.compression_ratio, verbose=verbose):
                current_file = next_file

        # Étape 3: Normalisation
        if config.normalize:
            if verbose:
                print(f"  → Normalisation à {config.target_loudness} LUFS...")
            next_file = next_temp()
            if normalize_audio(current_file, next_file,
                               config.target_loudness,
                               config.true_peak,
                               config.loudness_range, verbose):
                current_file = next_file

        # Étape 4: Ajout des silences intro/outro
        if config.intro_silence_ms > 0 or config.chapter_pause_ms > 0:
            if verbose:
                print("  → Ajout des silences...")
            next_file = next_temp()
            if add_silence(current_file, next_file,
                           config.intro_silence_ms,
                           config.chapter_pause_ms, verbose):
                current_file = next_file

        # Étape 5: Fades
        if config.fade_in_ms > 0 or config.fade_out_ms > 0:
            if verbose:
                print("  → Ajout des fades...")
            next_file = next_temp()
            if add_fades(current_file, next_file,
                         config.fade_in_ms,
                         config.fade_out_ms, verbose):
                current_file = next_file

        # Copier le résultat final
        shutil.copy2(current_file, output_file)

    return os.path.exists(output_file)


def postprocess_directory(
    input_dir: str,
    output_dir: Optional[str] = None,
    config: Optional[PostProcessConfig] = None,
    pattern: str = "*.wav",
    verbose: bool = False
) -> int:
    """
    Post-traite tous les fichiers audio d'un répertoire.

    Args:
        input_dir: Répertoire d'entrée
        output_dir: Répertoire de sortie (défaut: input_dir/postprocessed)
        config: Configuration
        pattern: Pattern des fichiers à traiter
        verbose: Mode verbeux

    Returns:
        Nombre de fichiers traités avec succès
    """
    from glob import glob

    if output_dir is None:
        output_dir = os.path.join(input_dir, "postprocessed")

    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob(os.path.join(input_dir, pattern)))
    success_count = 0

    print(f"Post-production de {len(files)} fichiers...")

    for i, input_file in enumerate(files, 1):
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename)

        print(f"  [{i}/{len(files)}] {filename}")

        if postprocess_audio(input_file, output_file, config, verbose):
            success_count += 1
        else:
            print(f"    ✗ Erreur")

    print(f"\n✓ {success_count}/{len(files)} fichiers traités")
    return success_count


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Post-production audio pour audiobooks")

    parser.add_argument("input", help="Fichier ou répertoire d'entrée")
    parser.add_argument("-o", "--output", help="Fichier ou répertoire de sortie")
    parser.add_argument("--loudness", type=float, default=-20.0,
                        help="Loudness cible en LUFS (défaut: -20)")
    parser.add_argument("--fade-in", type=int, default=500,
                        help="Fade in en ms (défaut: 500)")
    parser.add_argument("--fade-out", type=int, default=1000,
                        help="Fade out en ms (défaut: 1000)")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Désactiver la normalisation")
    parser.add_argument("--compress", action="store_true",
                        help="Activer la compression dynamique")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Mode verbeux")

    args = parser.parse_args()

    config = PostProcessConfig(
        normalize=not args.no_normalize,
        target_loudness=args.loudness,
        fade_in_ms=args.fade_in,
        fade_out_ms=args.fade_out,
        compress=args.compress
    )

    if os.path.isdir(args.input):
        postprocess_directory(args.input, args.output, config, verbose=args.verbose)
    else:
        output = args.output or args.input.replace('.wav', '_processed.wav')
        print(f"Post-production: {args.input}")
        if postprocess_audio(args.input, output, config, args.verbose):
            print(f"✓ Sortie: {output}")
        else:
            print("✗ Erreur")
