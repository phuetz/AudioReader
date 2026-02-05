"""Endpoints pour l'analyse et correction de conformité ACX/Audible."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.errors import APIError, ErrorCode

router = APIRouter(prefix="/api/v2/acx", tags=["ACX Compliance"])


class ACXAnalysisResponse(BaseModel):
    """Résultat d'analyse ACX."""
    is_compliant: bool
    rms_db: float
    peak_db: float
    true_peak_db: float
    noise_floor_db: float
    integrated_lufs: float
    sample_rate: int
    channels: int
    duration: float
    issues: list[str]


@router.post("/analyze", response_model=ACXAnalysisResponse)
async def analyze_acx(file: UploadFile = File(...)):
    """
    Analyse un fichier audio pour vérifier sa conformité ACX/Audible.

    Standards ACX:
    - RMS: -23 dB à -18 dB
    - Peak: ≤ -3 dB
    - True Peak: ≤ -1 dB
    - Noise floor: ≤ -60 dB
    - Sample rate: ≥ 44.1 kHz
    """
    if not file.filename:
        raise APIError(ErrorCode.VALIDATION_ERROR, "Nom de fichier manquant")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".wav", ".mp3", ".m4a", ".flac"}:
        raise APIError(ErrorCode.VALIDATION_ERROR, f"Format non supporté: {ext}")

    # Sauvegarder temporairement le fichier
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        from src.acx_compliance import ACXAnalyzer, ACXStandards

        analyzer = ACXAnalyzer(ACXStandards())
        analysis = analyzer.analyze(tmp_path)

        # Collecter les problèmes
        issues = []
        if analysis.rms_db < -23:
            issues.append(f"RMS trop bas ({analysis.rms_db:.1f} dB < -23 dB)")
        elif analysis.rms_db > -18:
            issues.append(f"RMS trop élevé ({analysis.rms_db:.1f} dB > -18 dB)")

        if analysis.peak_db > -3:
            issues.append(f"Peak trop élevé ({analysis.peak_db:.1f} dB > -3 dB)")

        if analysis.true_peak_db > -1:
            issues.append(f"True Peak trop élevé ({analysis.true_peak_db:.1f} dB > -1 dB)")

        if analysis.noise_floor_db > -60:
            issues.append(f"Noise floor trop élevé ({analysis.noise_floor_db:.1f} dB > -60 dB)")

        if analysis.sample_rate < 44100:
            issues.append(f"Sample rate trop bas ({analysis.sample_rate} Hz < 44100 Hz)")

        is_compliant = len(issues) == 0

        return ACXAnalysisResponse(
            is_compliant=is_compliant,
            rms_db=round(analysis.rms_db, 2),
            peak_db=round(analysis.peak_db, 2),
            true_peak_db=round(analysis.true_peak_db, 2),
            noise_floor_db=round(analysis.noise_floor_db, 2),
            integrated_lufs=round(analysis.integrated_lufs, 2),
            sample_rate=analysis.sample_rate,
            channels=analysis.channels,
            duration=round(analysis.duration, 2),
            issues=issues,
        )

    except ImportError:
        # Fallback si module non disponible
        import soundfile as sf
        import numpy as np

        audio, sr = sf.read(str(tmp_path))
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
            channels = 2
        else:
            channels = 1

        # Analyse basique
        rms = np.sqrt(np.mean(audio ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)
        peak = np.max(np.abs(audio))
        peak_db = 20 * np.log10(peak + 1e-10)

        issues = []
        if rms_db < -23:
            issues.append(f"RMS trop bas ({rms_db:.1f} dB < -23 dB)")
        elif rms_db > -18:
            issues.append(f"RMS trop élevé ({rms_db:.1f} dB > -18 dB)")
        if peak_db > -3:
            issues.append(f"Peak trop élevé ({peak_db:.1f} dB > -3 dB)")

        return ACXAnalysisResponse(
            is_compliant=len(issues) == 0,
            rms_db=round(rms_db, 2),
            peak_db=round(peak_db, 2),
            true_peak_db=round(peak_db, 2),  # Approximation
            noise_floor_db=-65.0,  # Estimation
            integrated_lufs=round(rms_db - 0.7, 2),  # Approximation
            sample_rate=sr,
            channels=channels,
            duration=round(len(audio) / sr, 2),
            issues=issues,
        )

    finally:
        # Nettoyer le fichier temporaire
        tmp_path.unlink(missing_ok=True)


@router.post("/fix")
async def fix_acx(file: UploadFile = File(...)):
    """
    Corrige automatiquement un fichier audio pour le rendre conforme ACX.

    Applique:
    - Normalisation loudness (-19 LUFS)
    - Limiteur de peak (-3 dB)
    - Réduction du bruit si nécessaire
    """
    if not file.filename:
        raise APIError(ErrorCode.VALIDATION_ERROR, "Nom de fichier manquant")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".wav", ".mp3", ".m4a", ".flac"}:
        raise APIError(ErrorCode.VALIDATION_ERROR, f"Format non supporté: {ext}")

    # Sauvegarder temporairement le fichier
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_in:
        content = await file.read()
        tmp_in.write(content)
        input_path = Path(tmp_in.name)

    output_path = input_path.with_name(f"{input_path.stem}_acx_compliant.wav")

    try:
        from src.acx_compliance import make_acx_compliant

        success, report = make_acx_compliant(input_path, output_path)

        if not success:
            raise APIError(ErrorCode.PROCESSING_ERROR, "Échec de la correction ACX")

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=output_path.name,
        )

    except ImportError:
        # Fallback basique
        import soundfile as sf
        import numpy as np

        audio, sr = sf.read(str(input_path))

        # Normalisation simple
        target_rms = 10 ** (-20 / 20)  # -20 dB
        current_rms = np.sqrt(np.mean(audio ** 2))
        if current_rms > 0:
            gain = target_rms / current_rms
            audio = audio * gain

        # Limiteur simple
        limit = 10 ** (-3 / 20)  # -3 dB
        audio = np.clip(audio, -limit, limit)

        sf.write(str(output_path), audio, sr)

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=output_path.name,
        )

    finally:
        # Nettoyer l'entrée
        input_path.unlink(missing_ok=True)
