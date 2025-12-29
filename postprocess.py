#!/usr/bin/env python3
"""
Post-traitement audio pour audiobooks professionnels.

Chaîne de traitement:
1. Normalisation loudness (-20 LUFS, standard podcast/audiobook)
2. EQ: roll-off basses fréquences (< 80Hz)
3. Compression douce (optionnel)
4. Ajout room tone (silence naturel)
5. Export MP3 192kbps

Usage:
    python postprocess.py output_kokoro/
    python postprocess.py output_kokoro/ --target-lufs -18
    python postprocess.py input.wav --output mastered.mp3
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional
import shutil


def check_ffmpeg() -> bool:
    """Vérifie que ffmpeg est installé."""
    return shutil.which("ffmpeg") is not None


def add_room_tone(input_path: Path, output_path: Path, head: float = 0.75, tail: float = 2.0) -> bool:
    """
    Ajoute du silence (room tone) au début et à la fin.

    Args:
        input_path: Fichier audio source
        output_path: Fichier de sortie
        head: Silence au début en secondes
        tail: Silence à la fin en secondes
    """
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-af", f"adelay={int(head*1000)}|{int(head*1000)},apad=pad_dur={tail}",
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erreur room tone: {e}")
        return False


def apply_eq_lowcut(input_path: Path, output_path: Path, cutoff: int = 80) -> bool:
    """
    Applique un filtre passe-haut pour couper les basses fréquences.

    Args:
        input_path: Fichier audio source
        output_path: Fichier de sortie
        cutoff: Fréquence de coupure en Hz
    """
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-af", f"highpass=f={cutoff}:poles=2",
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erreur EQ: {e}")
        return False


def normalize_loudness(input_path: Path, output_path: Path, target_lufs: float = -20.0) -> bool:
    """
    Normalise le loudness selon la norme EBU R128.

    Args:
        input_path: Fichier audio source
        output_path: Fichier de sortie
        target_lufs: Niveau cible en LUFS (-20 = standard podcast/audiobook)
    """
    try:
        cmd = [
            "ffmpeg-normalize", str(input_path),
            "-o", str(output_path),
            "-t", str(target_lufs),
            "-tp", "-3",  # True peak max -3dB
            "-lrt", "11",  # Loudness range target
            "--keep-loudness-range-target",
            "-f"  # Force overwrite
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erreur normalisation: {e}")
        return False


def apply_soft_compression(input_path: Path, output_path: Path) -> bool:
    """
    Applique une compression douce pour uniformiser les niveaux.
    """
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-af", "acompressor=threshold=-20dB:ratio=3:attack=5:release=50:makeup=2",
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erreur compression: {e}")
        return False


def apply_limiter(input_path: Path, output_path: Path, limit_db: float = -3.0) -> bool:
    """
    Applique un limiteur pour éviter l'écrêtage.
    """
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-af", f"alimiter=limit={limit_db}dB:attack=5:release=50",
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erreur limiter: {e}")
        return False


def convert_to_mp3(input_path: Path, output_path: Path, bitrate: str = "192k") -> bool:
    """
    Convertit en MP3 pour distribution.
    """
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-b:a", bitrate,
            "-ar", "44100",
            "-ac", "1",  # Mono
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erreur conversion MP3: {e}")
        return False


def get_audio_info(filepath: Path) -> dict:
    """Récupère les infos audio (durée, sample rate, etc.)"""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(filepath)
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        import json
        data = json.loads(result.stdout)

        duration = float(data.get("format", {}).get("duration", 0))
        streams = data.get("streams", [{}])
        audio = next((s for s in streams if s.get("codec_type") == "audio"), {})

        return {
            "duration": duration,
            "sample_rate": int(audio.get("sample_rate", 0)),
            "channels": audio.get("channels", 0),
            "codec": audio.get("codec_name", "unknown")
        }
    except:
        return {}


def process_file(
    input_path: Path,
    output_dir: Path,
    target_lufs: float = -20.0,
    add_room: bool = True,
    apply_eq: bool = True,
    compress: bool = False,
    output_format: str = "mp3"
) -> Optional[Path]:
    """
    Traite un fichier audio avec la chaîne de mastering complète.
    """
    import tempfile

    filename = input_path.stem
    temp_dir = Path(tempfile.mkdtemp())

    current = input_path
    step = 0

    try:
        # Étape 1: EQ (low cut)
        if apply_eq:
            step += 1
            next_file = temp_dir / f"{step}_eq.wav"
            if apply_eq_lowcut(current, next_file):
                current = next_file

        # Étape 2: Normalisation loudness
        step += 1
        next_file = temp_dir / f"{step}_normalized.wav"
        if normalize_loudness(current, next_file, target_lufs):
            current = next_file

        # Étape 3: Compression douce (optionnel)
        if compress:
            step += 1
            next_file = temp_dir / f"{step}_compressed.wav"
            if apply_soft_compression(current, next_file):
                current = next_file

        # Étape 4: Limiter
        step += 1
        next_file = temp_dir / f"{step}_limited.wav"
        if apply_limiter(current, next_file):
            current = next_file

        # Étape 5: Room tone
        if add_room:
            step += 1
            next_file = temp_dir / f"{step}_room.wav"
            if add_room_tone(current, next_file):
                current = next_file

        # Étape finale: Export
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_format == "mp3":
            output_path = output_dir / f"{filename}.mp3"
            convert_to_mp3(current, output_path)
        else:
            output_path = output_dir / f"{filename}.wav"
            shutil.copy(current, output_path)

        return output_path

    finally:
        # Nettoyage
        shutil.rmtree(temp_dir, ignore_errors=True)


def process_directory(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    target_lufs: float = -20.0,
    **kwargs
):
    """Traite tous les fichiers audio d'un dossier."""

    if output_dir is None:
        output_dir = input_dir / "mastered"

    # Trouver les fichiers audio
    audio_files = list(input_dir.glob("*.wav")) + list(input_dir.glob("*.mp3"))
    audio_files = [f for f in audio_files if "mastered" not in str(f)]

    if not audio_files:
        print("Aucun fichier audio trouvé.")
        return

    print(f"\n{'=' * 60}")
    print("   POST-TRAITEMENT AUDIOBOOK")
    print(f"{'=' * 60}")
    print(f"\nFichiers: {len(audio_files)}")
    print(f"Target LUFS: {target_lufs}")
    print(f"Sortie: {output_dir}")

    print(f"\n{'-' * 60}")

    for i, filepath in enumerate(sorted(audio_files), 1):
        print(f"\n  [{i}/{len(audio_files)}] {filepath.name}")

        info = get_audio_info(filepath)
        if info:
            print(f"      Durée: {info['duration']:.1f}s | SR: {info['sample_rate']}Hz")

        result = process_file(filepath, output_dir, target_lufs, **kwargs)

        if result:
            new_info = get_audio_info(result)
            print(f"      → {result.name} ({new_info.get('duration', 0):.1f}s)")
        else:
            print("      → ERREUR")

    # Résumé
    print(f"\n{'=' * 60}")
    print("   TERMINÉ")
    print(f"{'=' * 60}")

    output_files = list(output_dir.glob("*.mp3")) + list(output_dir.glob("*.wav"))
    total_size = sum(f.stat().st_size for f in output_files) / 1024 / 1024

    print(f"\n  Fichiers masterisés: {len(output_files)}")
    print(f"  Taille totale: {total_size:.1f} MB")
    print(f"  Dossier: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Post-traitement audio pour audiobooks professionnels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Chaîne de traitement:
  1. EQ: Filtre passe-haut (coupe < 80Hz)
  2. Normalisation loudness (EBU R128)
  3. Compression douce (optionnel)
  4. Limiteur (-3dB peak)
  5. Room tone (0.75s début, 2s fin)
  6. Export MP3 192kbps

Standards visés:
  - Loudness: -20 LUFS (podcast/audiobook)
  - Peak: -3 dB max
  - Noise floor: < -60 dB
        """
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Fichier ou dossier à traiter"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Dossier de sortie"
    )

    parser.add_argument(
        "-t", "--target-lufs",
        type=float,
        default=-20.0,
        help="Niveau loudness cible en LUFS (défaut: -20)"
    )

    parser.add_argument(
        "--no-room-tone",
        action="store_true",
        help="Ne pas ajouter de silence début/fin"
    )

    parser.add_argument(
        "--no-eq",
        action="store_true",
        help="Ne pas appliquer l'EQ"
    )

    parser.add_argument(
        "--compress",
        action="store_true",
        help="Appliquer une compression douce"
    )

    parser.add_argument(
        "--format",
        choices=["mp3", "wav"],
        default="mp3",
        help="Format de sortie (défaut: mp3)"
    )

    args = parser.parse_args()

    if not check_ffmpeg():
        print("ERREUR: ffmpeg non trouvé. Installez-le d'abord.")
        return 1

    kwargs = {
        "target_lufs": args.target_lufs,
        "add_room": not args.no_room_tone,
        "apply_eq": not args.no_eq,
        "compress": args.compress,
        "output_format": args.format
    }

    if args.input.is_dir():
        process_directory(args.input, args.output, **kwargs)
    elif args.input.is_file():
        output_dir = args.output or args.input.parent / "mastered"
        result = process_file(args.input, output_dir, **kwargs)
        if result:
            print(f"Fichier traité: {result}")
    else:
        print(f"ERREUR: {args.input} n'existe pas")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
