"""
Conformite ACX/Audible v2.4.

Standards professionnels pour audiobooks:
- ACX (Audiobook Creation Exchange): normes Amazon/Audible
- EBU R128: standard europeen de loudness

Specifications ACX:
- RMS: -23dB a -18dB
- Peak: -3dB maximum
- Noise floor: -60dB ou moins
- Sample rate: 44.1kHz (MP3) ou 44.1kHz+ (autres)
- Bit depth: 16-bit minimum
- Format: MP3 192kbps+ ou WAV/FLAC

Ce module:
- Analyse la conformite d'un fichier audio
- Corrige automatiquement les problemes
- Genere un rapport de conformite
"""
import subprocess
import json
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum
import numpy as np


class ComplianceLevel(Enum):
    """Niveau de conformite."""
    PASS = "pass"           # Conforme
    WARNING = "warning"     # Proche des limites
    FAIL = "fail"           # Non conforme


@dataclass
class ACXStandards:
    """Standards ACX/Audible."""
    # Loudness (RMS/LUFS)
    rms_min_db: float = -23.0
    rms_max_db: float = -18.0
    lufs_target: float = -20.0
    lufs_tolerance: float = 2.0

    # Peak
    peak_max_db: float = -3.0
    true_peak_max_db: float = -1.0

    # Noise floor
    noise_floor_max_db: float = -60.0

    # Technical
    sample_rate_min: int = 44100
    bit_depth_min: int = 16
    channels: int = 1  # Mono prefere pour audiobooks

    # Room tone
    room_tone_duration_sec: float = 0.5  # Au debut et fin
    room_tone_max_db: float = -60.0

    # Timing
    max_silence_duration_sec: float = 3.0  # Pas plus de 3s de silence
    intro_silence_min_sec: float = 0.5
    intro_silence_max_sec: float = 1.0
    outro_silence_min_sec: float = 1.0
    outro_silence_max_sec: float = 5.0


@dataclass
class AudioAnalysis:
    """Resultat de l'analyse audio."""
    # Loudness
    integrated_lufs: float = 0.0
    loudness_range: float = 0.0
    rms_db: float = 0.0

    # Peak
    peak_db: float = 0.0
    true_peak_db: float = 0.0

    # Noise
    noise_floor_db: float = 0.0

    # Technical
    sample_rate: int = 0
    bit_depth: int = 0
    channels: int = 0
    duration_sec: float = 0.0

    # Room tone
    intro_silence_sec: float = 0.0
    outro_silence_sec: float = 0.0


@dataclass
class ComplianceIssue:
    """Probleme de conformite detecte."""
    parameter: str
    current_value: float
    expected_range: str
    level: ComplianceLevel
    fix_available: bool = True
    fix_description: str = ""


@dataclass
class ComplianceReport:
    """Rapport complet de conformite."""
    file_path: str
    analysis: AudioAnalysis
    issues: List[ComplianceIssue] = field(default_factory=list)
    overall_status: ComplianceLevel = ComplianceLevel.PASS
    is_acx_compliant: bool = True
    recommendations: List[str] = field(default_factory=list)


class ACXAnalyzer:
    """
    Analyse la conformite ACX/Audible d'un fichier audio.
    """

    def __init__(self, standards: Optional[ACXStandards] = None):
        """
        Initialise l'analyseur.

        Args:
            standards: Standards a utiliser (defaut: ACX standard)
        """
        self.standards = standards or ACXStandards()
        self._ffmpeg_available = None

    def is_ffmpeg_available(self) -> bool:
        """Verifie si ffmpeg est disponible."""
        if self._ffmpeg_available is not None:
            return self._ffmpeg_available

        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True
            )
            self._ffmpeg_available = result.returncode == 0
        except FileNotFoundError:
            self._ffmpeg_available = False

        return self._ffmpeg_available

    def analyze_loudness(self, audio_path: str) -> Dict[str, float]:
        """
        Analyse le loudness avec ffmpeg (EBU R128).

        Returns:
            Dict avec integrated, true_peak, lra
        """
        if not self.is_ffmpeg_available():
            return {}

        cmd = [
            "ffmpeg", "-i", audio_path,
            "-af", "loudnorm=print_format=json",
            "-f", "null", "-"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            # Parser la sortie JSON
            output = result.stderr
            json_start = output.rfind('{')
            json_end = output.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = output[json_start:json_end]
                data = json.loads(json_str)
                return {
                    "integrated": float(data.get("input_i", -99)),
                    "true_peak": float(data.get("input_tp", -99)),
                    "lra": float(data.get("input_lra", 0)),
                    "threshold": float(data.get("input_thresh", -99)),
                }
        except Exception as e:
            print(f"Erreur analyse loudness: {e}")

        return {}

    def analyze_audio_properties(self, audio_path: str) -> Dict:
        """
        Analyse les proprietes techniques du fichier.

        Returns:
            Dict avec sample_rate, channels, duration, etc.
        """
        if not self.is_ffmpeg_available():
            return {}

        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            audio_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            data = json.loads(result.stdout)

            # Extraire les infos du premier stream audio
            audio_stream = None
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "audio":
                    audio_stream = stream
                    break

            if audio_stream:
                return {
                    "sample_rate": int(audio_stream.get("sample_rate", 0)),
                    "channels": int(audio_stream.get("channels", 0)),
                    "bit_depth": int(audio_stream.get("bits_per_sample", 16)),
                    "duration": float(data.get("format", {}).get("duration", 0)),
                    "codec": audio_stream.get("codec_name", "unknown"),
                }
        except Exception as e:
            print(f"Erreur analyse proprietes: {e}")

        return {}

    def analyze_peak(self, audio_path: str) -> Dict[str, float]:
        """
        Analyse le peak et true peak.

        Returns:
            Dict avec peak_db, true_peak_db
        """
        if not self.is_ffmpeg_available():
            return {}

        # Utiliser le filtre astats pour les peaks
        cmd = [
            "ffmpeg", "-i", audio_path,
            "-af", "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.Peak_level",
            "-f", "null", "-"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stderr

            # Chercher la valeur de peak
            peak_match = None
            for line in output.split('\n'):
                if 'Peak_level' in line:
                    parts = line.split('=')
                    if len(parts) >= 2:
                        try:
                            peak_match = float(parts[-1].strip())
                        except ValueError:
                            pass

            # Fallback: utiliser volumedetect
            if peak_match is None:
                cmd2 = [
                    "ffmpeg", "-i", audio_path,
                    "-af", "volumedetect",
                    "-f", "null", "-"
                ]
                result2 = subprocess.run(cmd2, capture_output=True, text=True)
                output2 = result2.stderr

                for line in output2.split('\n'):
                    if 'max_volume' in line:
                        parts = line.split(':')
                        if len(parts) >= 2:
                            val = parts[1].strip().replace(' dB', '')
                            try:
                                peak_match = float(val)
                            except ValueError:
                                pass

            return {
                "peak_db": peak_match or 0.0,
                "true_peak_db": peak_match or 0.0  # Approximation
            }
        except Exception as e:
            print(f"Erreur analyse peak: {e}")

        return {}

    def analyze_noise_floor(self, audio_path: str) -> float:
        """
        Estime le noise floor (niveau du bruit de fond).

        Returns:
            Noise floor en dB
        """
        if not self.is_ffmpeg_available():
            return -60.0

        # Utiliser silencedetect pour trouver les passages silencieux
        cmd = [
            "ffmpeg", "-i", audio_path,
            "-af", "silencedetect=n=-50dB:d=0.5",
            "-f", "null", "-"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stderr

            # Si des silences sont detectes a -50dB, le noise floor est probablement < -50dB
            if "silence_start" in output:
                return -55.0  # Estimation
            else:
                return -45.0  # Probablement trop de bruit
        except Exception:
            pass

        return -50.0  # Valeur par defaut

    def analyze(self, audio_path: str) -> AudioAnalysis:
        """
        Analyse complete d'un fichier audio.

        Args:
            audio_path: Chemin du fichier audio

        Returns:
            AudioAnalysis avec tous les parametres
        """
        analysis = AudioAnalysis()

        # Proprietes techniques
        props = self.analyze_audio_properties(audio_path)
        analysis.sample_rate = props.get("sample_rate", 0)
        analysis.channels = props.get("channels", 0)
        analysis.bit_depth = props.get("bit_depth", 16)
        analysis.duration_sec = props.get("duration", 0)

        # Loudness
        loudness = self.analyze_loudness(audio_path)
        analysis.integrated_lufs = loudness.get("integrated", 0.0)
        analysis.loudness_range = loudness.get("lra", 0.0)
        analysis.rms_db = loudness.get("integrated", 0.0)  # Approximation

        # Peak
        peaks = self.analyze_peak(audio_path)
        analysis.peak_db = peaks.get("peak_db", 0.0)
        analysis.true_peak_db = peaks.get("true_peak_db", 0.0)

        # Noise floor
        analysis.noise_floor_db = self.analyze_noise_floor(audio_path)

        return analysis

    def check_compliance(self, analysis: AudioAnalysis) -> ComplianceReport:
        """
        Verifie la conformite aux standards ACX.

        Args:
            analysis: Resultat de l'analyse

        Returns:
            ComplianceReport avec les problemes detectes
        """
        issues = []
        recommendations = []

        # Verifier le loudness/RMS
        if analysis.rms_db < self.standards.rms_min_db:
            issues.append(ComplianceIssue(
                parameter="RMS/Loudness",
                current_value=analysis.rms_db,
                expected_range=f"{self.standards.rms_min_db} to {self.standards.rms_max_db} dB",
                level=ComplianceLevel.FAIL,
                fix_available=True,
                fix_description="Augmenter le volume avec normalisation"
            ))
        elif analysis.rms_db > self.standards.rms_max_db:
            issues.append(ComplianceIssue(
                parameter="RMS/Loudness",
                current_value=analysis.rms_db,
                expected_range=f"{self.standards.rms_min_db} to {self.standards.rms_max_db} dB",
                level=ComplianceLevel.FAIL,
                fix_available=True,
                fix_description="Reduire le volume avec normalisation"
            ))

        # Verifier le peak
        if analysis.peak_db > self.standards.peak_max_db:
            issues.append(ComplianceIssue(
                parameter="Peak",
                current_value=analysis.peak_db,
                expected_range=f"<= {self.standards.peak_max_db} dB",
                level=ComplianceLevel.FAIL,
                fix_available=True,
                fix_description="Appliquer un limiteur"
            ))

        # Verifier le true peak
        if analysis.true_peak_db > self.standards.true_peak_max_db:
            issues.append(ComplianceIssue(
                parameter="True Peak",
                current_value=analysis.true_peak_db,
                expected_range=f"<= {self.standards.true_peak_max_db} dB",
                level=ComplianceLevel.WARNING,
                fix_available=True,
                fix_description="Appliquer un true peak limiteur"
            ))

        # Verifier le noise floor
        if analysis.noise_floor_db > self.standards.noise_floor_max_db:
            issues.append(ComplianceIssue(
                parameter="Noise Floor",
                current_value=analysis.noise_floor_db,
                expected_range=f"<= {self.standards.noise_floor_max_db} dB",
                level=ComplianceLevel.WARNING,
                fix_available=False,
                fix_description="Nettoyer le bruit de fond (difficile a automatiser)"
            ))
            recommendations.append("Envisager une reduction de bruit manuelle")

        # Verifier le sample rate
        if analysis.sample_rate < self.standards.sample_rate_min:
            issues.append(ComplianceIssue(
                parameter="Sample Rate",
                current_value=analysis.sample_rate,
                expected_range=f">= {self.standards.sample_rate_min} Hz",
                level=ComplianceLevel.FAIL,
                fix_available=True,
                fix_description="Re-echantillonner a 44.1kHz"
            ))

        # Verifier les canaux (mono prefere)
        if analysis.channels > self.standards.channels:
            issues.append(ComplianceIssue(
                parameter="Channels",
                current_value=analysis.channels,
                expected_range=f"= {self.standards.channels} (mono)",
                level=ComplianceLevel.WARNING,
                fix_available=True,
                fix_description="Convertir en mono"
            ))

        # Determiner le statut global
        has_fail = any(i.level == ComplianceLevel.FAIL for i in issues)
        has_warning = any(i.level == ComplianceLevel.WARNING for i in issues)

        if has_fail:
            overall_status = ComplianceLevel.FAIL
            is_compliant = False
        elif has_warning:
            overall_status = ComplianceLevel.WARNING
            is_compliant = True
        else:
            overall_status = ComplianceLevel.PASS
            is_compliant = True

        return ComplianceReport(
            file_path="",
            analysis=analysis,
            issues=issues,
            overall_status=overall_status,
            is_acx_compliant=is_compliant,
            recommendations=recommendations
        )


class ACXCorrector:
    """
    Corrige automatiquement les problemes de conformite ACX.
    """

    def __init__(self, standards: Optional[ACXStandards] = None):
        """
        Initialise le correcteur.

        Args:
            standards: Standards cibles
        """
        self.standards = standards or ACXStandards()

    def fix_loudness(
        self,
        input_path: str,
        output_path: str,
        target_lufs: float = -20.0
    ) -> bool:
        """
        Normalise le loudness selon EBU R128.

        Args:
            input_path: Fichier source
            output_path: Fichier de sortie
            target_lufs: Cible LUFS

        Returns:
            True si succes
        """
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
            "-ar", str(self.standards.sample_rate_min),
            "-ac", str(self.standards.channels),
            output_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True)
            return result.returncode == 0
        except Exception as e:
            print(f"Erreur normalisation: {e}")
            return False

    def fix_peak(
        self,
        input_path: str,
        output_path: str,
        peak_limit_db: float = -3.0
    ) -> bool:
        """
        Applique un limiteur de peak.

        Args:
            input_path: Fichier source
            output_path: Fichier de sortie
            peak_limit_db: Limite de peak

        Returns:
            True si succes
        """
        limit_linear = 10 ** (peak_limit_db / 20)

        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", f"alimiter=limit={limit_linear}:attack=5:release=50",
            output_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True)
            return result.returncode == 0
        except Exception as e:
            print(f"Erreur limiteur: {e}")
            return False

    def fix_sample_rate(
        self,
        input_path: str,
        output_path: str,
        target_rate: int = 44100
    ) -> bool:
        """
        Re-echantillonne a la frequence cible.

        Args:
            input_path: Fichier source
            output_path: Fichier de sortie
            target_rate: Frequence cible

        Returns:
            True si succes
        """
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", str(target_rate),
            output_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True)
            return result.returncode == 0
        except Exception as e:
            print(f"Erreur resampling: {e}")
            return False

    def fix_channels(
        self,
        input_path: str,
        output_path: str,
        channels: int = 1
    ) -> bool:
        """
        Convertit en mono/stereo.

        Args:
            input_path: Fichier source
            output_path: Fichier de sortie
            channels: Nombre de canaux cibles

        Returns:
            True si succes
        """
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ac", str(channels),
            output_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True)
            return result.returncode == 0
        except Exception as e:
            print(f"Erreur conversion canaux: {e}")
            return False

    def make_acx_compliant(
        self,
        input_path: str,
        output_path: str,
        verbose: bool = True
    ) -> Tuple[bool, ComplianceReport]:
        """
        Applique toutes les corrections necessaires pour la conformite ACX.

        Args:
            input_path: Fichier source
            output_path: Fichier de sortie
            verbose: Afficher les details

        Returns:
            Tuple (succes, rapport_final)
        """
        analyzer = ACXAnalyzer(self.standards)

        # Analyse initiale
        if verbose:
            print(f"Analyse de {input_path}...")

        analysis = analyzer.analyze(input_path)
        initial_report = analyzer.check_compliance(analysis)

        if initial_report.overall_status == ComplianceLevel.PASS:
            if verbose:
                print("Fichier deja conforme ACX!")
            # Copier simplement le fichier
            import shutil
            shutil.copy2(input_path, output_path)
            return True, initial_report

        # Appliquer les corrections dans un fichier temporaire
        with tempfile.TemporaryDirectory() as tmpdir:
            current_file = input_path
            step = 0

            def next_temp():
                nonlocal step
                step += 1
                return os.path.join(tmpdir, f"step_{step}.wav")

            # Etape 1: Normalisation loudness
            if any(i.parameter in ["RMS/Loudness"] for i in initial_report.issues):
                if verbose:
                    print("  -> Normalisation loudness...")
                next_file = next_temp()
                if self.fix_loudness(current_file, next_file, self.standards.lufs_target):
                    current_file = next_file

            # Etape 2: Limiteur de peak
            if any(i.parameter in ["Peak", "True Peak"] for i in initial_report.issues):
                if verbose:
                    print("  -> Application limiteur...")
                next_file = next_temp()
                if self.fix_peak(current_file, next_file, self.standards.peak_max_db):
                    current_file = next_file

            # Etape 3: Sample rate
            if any(i.parameter == "Sample Rate" for i in initial_report.issues):
                if verbose:
                    print("  -> Re-echantillonnage...")
                next_file = next_temp()
                if self.fix_sample_rate(current_file, next_file, self.standards.sample_rate_min):
                    current_file = next_file

            # Etape 4: Canaux
            if any(i.parameter == "Channels" for i in initial_report.issues):
                if verbose:
                    print("  -> Conversion mono...")
                next_file = next_temp()
                if self.fix_channels(current_file, next_file, self.standards.channels):
                    current_file = next_file

            # Copier le resultat final
            import shutil
            shutil.copy2(current_file, output_path)

        # Verification finale
        if verbose:
            print("Verification finale...")

        final_analysis = analyzer.analyze(output_path)
        final_report = analyzer.check_compliance(final_analysis)
        final_report.file_path = output_path

        if verbose:
            status = "CONFORME" if final_report.is_acx_compliant else "NON CONFORME"
            print(f"Statut final: {status}")

        return final_report.is_acx_compliant, final_report


def analyze_and_report(audio_path: str) -> ComplianceReport:
    """
    Analyse un fichier et genere un rapport de conformite.

    Args:
        audio_path: Chemin du fichier audio

    Returns:
        ComplianceReport detaille
    """
    analyzer = ACXAnalyzer()
    analysis = analyzer.analyze(audio_path)
    report = analyzer.check_compliance(analysis)
    report.file_path = audio_path
    return report


def make_acx_compliant(
    input_path: str,
    output_path: str
) -> Tuple[bool, ComplianceReport]:
    """
    Rend un fichier conforme ACX.

    Args:
        input_path: Fichier source
        output_path: Fichier de sortie

    Returns:
        Tuple (succes, rapport)
    """
    corrector = ACXCorrector()
    return corrector.make_acx_compliant(input_path, output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python acx_compliance.py <audio_file> [--fix output_file]")
        print("\nAnalyse la conformite ACX/Audible d'un fichier audio.")
        print("\nOptions:")
        print("  --fix output_file  Corrige et sauvegarde le fichier conforme")
        sys.exit(1)

    input_file = sys.argv[1]

    if "--fix" in sys.argv:
        idx = sys.argv.index("--fix")
        if idx + 1 < len(sys.argv):
            output_file = sys.argv[idx + 1]
        else:
            output_file = input_file.replace(".wav", "_acx.wav")

        print(f"=== Correction ACX: {input_file} ===\n")
        success, report = make_acx_compliant(input_file, output_file)

        if success:
            print(f"\n Fichier conforme: {output_file}")
        else:
            print(f"\n Problemes non resolus:")
            for issue in report.issues:
                if issue.level == ComplianceLevel.FAIL:
                    print(f"  - {issue.parameter}: {issue.current_value}")
    else:
        print(f"=== Analyse ACX: {input_file} ===\n")
        report = analyze_and_report(input_file)

        print(f"Statut: {report.overall_status.value.upper()}")
        print(f"Conforme ACX: {'Oui' if report.is_acx_compliant else 'Non'}")

        print(f"\nParametres:")
        print(f"  Loudness: {report.analysis.integrated_lufs:.1f} LUFS")
        print(f"  Peak: {report.analysis.peak_db:.1f} dB")
        print(f"  Sample Rate: {report.analysis.sample_rate} Hz")
        print(f"  Channels: {report.analysis.channels}")

        if report.issues:
            print(f"\nProblemes detectes:")
            for issue in report.issues:
                status = "" if issue.level == ComplianceLevel.FAIL else ""
                print(f"  {status} {issue.parameter}: {issue.current_value} "
                      f"(attendu: {issue.expected_range})")

        if report.recommendations:
            print(f"\nRecommandations:")
            for rec in report.recommendations:
                print(f"  - {rec}")
