"""
Extracteur audio pour clonage de voix.

Extrait la piste audio d'une vidéo (MP4, MKV, AVI) et la prépare pour le clonage XTTS:
- Conversion en WAV
- Sample rate 24000 Hz (standard XTTS/Kokoro)
- Mono
- Découpage optionnel
"""
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple
import json
import hashlib


class AudioExtractor:
    """Extrait et traite l'audio des vidéos."""

    def __init__(self, output_dir: Path = Path(".voice_cache/extracted")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ffmpeg = shutil.which("ffmpeg")

    def is_available(self) -> bool:
        """Vérifie si ffmpeg est installé."""
        return self.ffmpeg is not None

    def extract_from_video(
        self,
        video_path: Path,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        output_name: Optional[str] = None
    ) -> Optional[Path]:
        """
        Extrait l'audio d'une vidéo.

        Args:
            video_path: Chemin de la vidéo
            start_time: Début en secondes (optionnel)
            end_time: Fin en secondes (optionnel)
            output_name: Nom du fichier de sortie (sans extension)

        Returns:
            Chemin du fichier WAV généré ou None
        """
        if not self.is_available():
            raise RuntimeError("ffmpeg n'est pas installé. Installez-le pour extraire l'audio.")

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Vidéo introuvable: {video_path}")

        # Générer un nom si non fourni
        if not output_name:
            file_hash = hashlib.md5(str(video_path).encode()).hexdigest()[:8]
            output_name = f"{video_path.stem}_{file_hash}"

        output_path = self.output_dir / f"{output_name}.wav"

        # Commande ffmpeg
        # -vn: pas de vidéo
        # -acodec pcm_s16le: WAV PCM 16-bit
        # -ar 24000: 24kHz (optimal pour XTTS)
        # -ac 1: Mono
        cmd = [
            self.ffmpeg,
            "-y",  # Overwrite
            "-i", str(video_path),
        ]

        if start_time is not None:
            cmd.extend(["-ss", str(start_time)])
        
        if end_time is not None:
            cmd.extend(["-to", str(end_time)])

        cmd.extend([
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "24000",
            "-ac", "1",
            str(output_path)
        ])

        try:
            # Exécuter ffmpeg
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"Erreur ffmpeg: {e.stderr.decode()}")
            return None

    def get_duration(self, file_path: Path) -> float:
        """Récupère la durée du fichier avec ffprobe."""
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            return 0.0

        cmd = [
            ffprobe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path)
        ]

        try:
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return float(result.stdout.strip())
        except Exception:
            return 0.0
