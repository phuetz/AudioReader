"""
Export vers différentes plateformes de distribution audio.

Supporte:
- Spotify (MP3 320kbps, -14 LUFS)
- YouTube (vidéo avec waveform animée)
- Podcast (RSS, MP3 variable bitrate)
- ACX/Audible (WAV/M4B, strict compliance)
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import subprocess
import json
import tempfile


@dataclass
class ExportMetadata:
    """Métadonnées pour l'export."""
    title: str
    author: str = "Unknown"
    description: str = ""
    artwork_path: Optional[Path] = None
    language: str = "fr"
    genre: str = "Audiobook"
    year: Optional[int] = None
    narrator: str = ""
    isbn: str = ""
    publisher: str = ""


@dataclass
class ExportResult:
    """Résultat d'un export."""
    success: bool
    output_path: Path
    format: str
    file_size_mb: float
    duration_seconds: float
    message: str = ""


class PlatformExporter:
    """
    Exporte l'audio vers différentes plateformes.

    Chaque plateforme a ses propres exigences de format,
    normalisation et métadonnées.
    """

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        """
        Initialise l'exporteur.

        Args:
            ffmpeg_path: Chemin vers ffmpeg
        """
        self.ffmpeg_path = ffmpeg_path
        self._check_ffmpeg()

    def _check_ffmpeg(self) -> bool:
        """Vérifie que ffmpeg est disponible."""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def _run_ffmpeg(self, args: list, timeout: int = 3600) -> tuple[bool, str]:
        """Exécute une commande ffmpeg."""
        try:
            result = subprocess.run(
                [self.ffmpeg_path] + args,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                return False, result.stderr
            return True, result.stdout
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)

    def _get_audio_info(self, audio_path: Path) -> dict:
        """Récupère les informations d'un fichier audio."""
        args = [
            "-i", str(audio_path),
            "-hide_banner",
            "-show_format",
            "-show_streams",
            "-print_format", "json",
        ]
        try:
            result = subprocess.run(
                ["ffprobe"] + args,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
        except Exception:
            pass
        return {}

    def export_for_spotify(
        self,
        audio_path: Path,
        output_path: Path,
        metadata: ExportMetadata,
    ) -> ExportResult:
        """
        Exporte pour Spotify.

        Exigences Spotify:
        - Format: MP3 320kbps CBR
        - Normalisation: -14 LUFS
        - Sample rate: 44.1kHz
        - Artwork: 3000x3000 pixels minimum
        """
        # Normaliser à -14 LUFS et encoder en MP3 320kbps
        args = [
            "-y",  # Overwrite
            "-i", str(audio_path),
            "-af", "loudnorm=I=-14:TP=-1:LRA=11",
            "-ar", "44100",  # 44.1kHz
            "-ac", "2",  # Stereo
            "-b:a", "320k",  # 320kbps
            "-codec:a", "libmp3lame",
        ]

        # Ajouter l'artwork si disponible
        if metadata.artwork_path and metadata.artwork_path.exists():
            args.extend([
                "-i", str(metadata.artwork_path),
                "-map", "0:a",
                "-map", "1:v",
                "-c:v", "copy",
                "-id3v2_version", "3",
                "-metadata:s:v", "title=Album cover",
                "-metadata:s:v", "comment=Cover (front)",
            ])

        # Métadonnées ID3
        args.extend([
            "-metadata", f"title={metadata.title}",
            "-metadata", f"artist={metadata.author}",
            "-metadata", f"album={metadata.title}",
            "-metadata", f"genre={metadata.genre}",
            "-metadata", f"comment={metadata.description}",
        ])

        if metadata.year:
            args.extend(["-metadata", f"date={metadata.year}"])

        args.append(str(output_path))

        success, message = self._run_ffmpeg(args)

        if success:
            file_size = output_path.stat().st_size / (1024 * 1024)
            info = self._get_audio_info(output_path)
            duration = float(info.get("format", {}).get("duration", 0))

            return ExportResult(
                success=True,
                output_path=output_path,
                format="MP3 320kbps",
                file_size_mb=file_size,
                duration_seconds=duration,
                message="Export Spotify réussi",
            )
        else:
            return ExportResult(
                success=False,
                output_path=output_path,
                format="MP3",
                file_size_mb=0,
                duration_seconds=0,
                message=f"Erreur: {message}",
            )

    def export_for_youtube(
        self,
        audio_path: Path,
        output_path: Path,
        metadata: ExportMetadata,
        background_image: Optional[Path] = None,
        show_waveform: bool = True,
    ) -> ExportResult:
        """
        Exporte pour YouTube.

        Crée une vidéo avec:
        - Image de fond ou artwork
        - Waveform animée optionnelle
        - Audio AAC haute qualité
        """
        # Image de fond
        bg_image = background_image or metadata.artwork_path
        if not bg_image or not bg_image.exists():
            # Créer une image noire par défaut
            bg_image = Path(tempfile.mktemp(suffix=".png"))
            self._run_ffmpeg([
                "-y",
                "-f", "lavfi",
                "-i", "color=black:1920x1080:d=1",
                "-frames:v", "1",
                str(bg_image),
            ])

        if show_waveform:
            # Vidéo avec waveform
            filter_complex = (
                "[0:a]showwaves=s=1920x200:mode=cline:colors=0x00ff00:scale=sqrt[wave];"
                "[1:v]scale=1920:1080[bg];"
                "[bg][wave]overlay=0:H-200[out]"
            )
            args = [
                "-y",
                "-i", str(audio_path),
                "-loop", "1",
                "-i", str(bg_image),
                "-filter_complex", filter_complex,
                "-map", "[out]",
                "-map", "0:a",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                "-pix_fmt", "yuv420p",
            ]
        else:
            # Vidéo statique
            args = [
                "-y",
                "-loop", "1",
                "-i", str(bg_image),
                "-i", str(audio_path),
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                "-pix_fmt", "yuv420p",
            ]

        # Métadonnées
        args.extend([
            "-metadata", f"title={metadata.title}",
            "-metadata", f"artist={metadata.author}",
            "-metadata", f"description={metadata.description}",
        ])

        args.append(str(output_path))

        success, message = self._run_ffmpeg(args, timeout=7200)

        if success:
            file_size = output_path.stat().st_size / (1024 * 1024)
            info = self._get_audio_info(output_path)
            duration = float(info.get("format", {}).get("duration", 0))

            return ExportResult(
                success=True,
                output_path=output_path,
                format="MP4 (YouTube)",
                file_size_mb=file_size,
                duration_seconds=duration,
                message="Export YouTube réussi",
            )
        else:
            return ExportResult(
                success=False,
                output_path=output_path,
                format="MP4",
                file_size_mb=0,
                duration_seconds=0,
                message=f"Erreur: {message}",
            )

    def export_for_podcast(
        self,
        audio_path: Path,
        output_path: Path,
        metadata: ExportMetadata,
        bitrate: str = "128k",
    ) -> ExportResult:
        """
        Exporte pour podcast.

        Standard podcast:
        - Format: MP3 VBR ou CBR
        - Normalisation: -16 LUFS
        - Mono acceptable
        """
        args = [
            "-y",
            "-i", str(audio_path),
            "-af", "loudnorm=I=-16:TP=-1:LRA=11",
            "-ar", "44100",
            "-ac", "1",  # Mono (économise de la bande passante)
            "-b:a", bitrate,
            "-codec:a", "libmp3lame",
            "-metadata", f"title={metadata.title}",
            "-metadata", f"artist={metadata.author}",
            "-metadata", f"album={metadata.title}",
            "-metadata", f"genre=Podcast",
            "-metadata", f"comment={metadata.description}",
        ]

        if metadata.artwork_path and metadata.artwork_path.exists():
            args.extend([
                "-i", str(metadata.artwork_path),
                "-map", "0:a",
                "-map", "1:v",
                "-c:v", "copy",
            ])

        args.append(str(output_path))

        success, message = self._run_ffmpeg(args)

        if success:
            file_size = output_path.stat().st_size / (1024 * 1024)
            info = self._get_audio_info(output_path)
            duration = float(info.get("format", {}).get("duration", 0))

            return ExportResult(
                success=True,
                output_path=output_path,
                format=f"MP3 {bitrate}",
                file_size_mb=file_size,
                duration_seconds=duration,
                message="Export Podcast réussi",
            )
        else:
            return ExportResult(
                success=False,
                output_path=output_path,
                format="MP3",
                file_size_mb=0,
                duration_seconds=0,
                message=f"Erreur: {message}",
            )

    def validate_acx_strict(self, audio_path: Path) -> dict:
        """
        Validation stricte ACX/Audible.

        Retourne un rapport détaillé de conformité.
        """
        try:
            from src.acx_compliance import ACXAnalyzer, ACXStandards

            analyzer = ACXAnalyzer(ACXStandards())
            report = analyzer.analyze_file(str(audio_path))

            return {
                "compliant": report.is_acx_compliant,
                "integrated_lufs": report.analysis.integrated_lufs,
                "peak_db": report.analysis.peak_db,
                "true_peak_db": report.analysis.true_peak_db,
                "noise_floor_db": report.analysis.noise_floor_db,
                "sample_rate": report.analysis.sample_rate,
                "issues": report.issues,
                "recommendations": report.recommendations,
            }
        except ImportError:
            return {
                "compliant": None,
                "error": "Module acx_compliance non disponible",
            }

    def export_for_acx(
        self,
        audio_path: Path,
        output_path: Path,
        metadata: ExportMetadata,
        auto_fix: bool = True,
    ) -> ExportResult:
        """
        Exporte pour ACX/Audible.

        Exigences ACX:
        - Format: MP3 192kbps CBR ou M4B
        - Peak: -3dB max
        - RMS: -23dB to -18dB
        - Noise floor: -60dB ou moins
        - Sample rate: 44.1kHz
        - Mono
        """
        # Vérifier et corriger si nécessaire
        if auto_fix:
            try:
                from src.acx_compliance import make_acx_compliant
                corrected_path = output_path.with_suffix(".corrected.wav")
                success, report = make_acx_compliant(str(audio_path), str(corrected_path))
                if success:
                    audio_path = corrected_path
            except ImportError:
                pass

        # Encoder en MP3 192kbps CBR (standard ACX)
        args = [
            "-y",
            "-i", str(audio_path),
            "-ar", "44100",
            "-ac", "1",  # Mono requis
            "-b:a", "192k",
            "-codec:a", "libmp3lame",
            "-metadata", f"title={metadata.title}",
            "-metadata", f"artist={metadata.narrator or metadata.author}",
            "-metadata", f"album={metadata.title}",
            "-metadata", f"album_artist={metadata.author}",
            "-metadata", f"genre=Audiobook",
            "-metadata", f"publisher={metadata.publisher}",
        ]

        if metadata.isbn:
            args.extend(["-metadata", f"ISRC={metadata.isbn}"])

        if metadata.year:
            args.extend(["-metadata", f"date={metadata.year}"])

        args.append(str(output_path))

        success, message = self._run_ffmpeg(args)

        if success:
            file_size = output_path.stat().st_size / (1024 * 1024)
            info = self._get_audio_info(output_path)
            duration = float(info.get("format", {}).get("duration", 0))

            # Valider le résultat
            validation = self.validate_acx_strict(output_path)

            return ExportResult(
                success=validation.get("compliant", True),
                output_path=output_path,
                format="MP3 192kbps (ACX)",
                file_size_mb=file_size,
                duration_seconds=duration,
                message="ACX compliant" if validation.get("compliant") else f"Issues: {validation.get('issues', [])}",
            )
        else:
            return ExportResult(
                success=False,
                output_path=output_path,
                format="MP3",
                file_size_mb=0,
                duration_seconds=0,
                message=f"Erreur: {message}",
            )


def export_audiobook(
    audio_path: Path,
    output_dir: Path,
    metadata: ExportMetadata,
    platforms: list[str] = None,
) -> dict[str, ExportResult]:
    """
    Exporte un audiobook vers plusieurs plateformes.

    Args:
        audio_path: Chemin vers l'audio source
        output_dir: Dossier de sortie
        metadata: Métadonnées
        platforms: Liste de plateformes ("spotify", "youtube", "podcast", "acx")

    Returns:
        Dict {platform: ExportResult}
    """
    if platforms is None:
        platforms = ["spotify", "podcast", "acx"]

    output_dir.mkdir(parents=True, exist_ok=True)
    exporter = PlatformExporter()
    results = {}

    base_name = audio_path.stem

    if "spotify" in platforms:
        output = output_dir / f"{base_name}_spotify.mp3"
        results["spotify"] = exporter.export_for_spotify(audio_path, output, metadata)

    if "youtube" in platforms:
        output = output_dir / f"{base_name}_youtube.mp4"
        results["youtube"] = exporter.export_for_youtube(audio_path, output, metadata)

    if "podcast" in platforms:
        output = output_dir / f"{base_name}_podcast.mp3"
        results["podcast"] = exporter.export_for_podcast(audio_path, output, metadata)

    if "acx" in platforms:
        output = output_dir / f"{base_name}_acx.mp3"
        results["acx"] = exporter.export_for_acx(audio_path, output, metadata)

    return results


if __name__ == "__main__":
    print("=== Test Platform Exporter ===\n")

    exporter = PlatformExporter()
    print(f"ffmpeg disponible: {exporter._check_ffmpeg()}")

    metadata = ExportMetadata(
        title="Test Audiobook",
        author="Test Author",
        description="Description de test",
    )
    print(f"\nMétadonnées de test:")
    print(f"  Titre: {metadata.title}")
    print(f"  Auteur: {metadata.author}")

    print("\nPlateformes supportées:")
    print("  - Spotify (MP3 320kbps, -14 LUFS)")
    print("  - YouTube (MP4 avec waveform)")
    print("  - Podcast (MP3 VBR, -16 LUFS)")
    print("  - ACX/Audible (MP3 192kbps, strict compliance)")
