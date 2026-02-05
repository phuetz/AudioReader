"""Endpoints pour la gestion des corrections de prononciation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.dependencies import DATA_DIR
from api.errors import APIError, ErrorCode

router = APIRouter(prefix="/api/v2/corrections", tags=["Corrections"])

# Fichier de corrections par défaut
CORRECTIONS_FILE = DATA_DIR / "corrections.json"


class CorrectionRule(BaseModel):
    """Règle de correction de prononciation."""
    id: str
    pattern: str = Field(..., description="Texte ou regex à rechercher")
    replacement: str = Field(..., description="Texte de remplacement")
    confidence: str = Field(default="high", description="Niveau de confiance: high, medium, low")
    notes: str = Field(default="", description="Notes explicatives")


class CorrectionsList(BaseModel):
    """Liste de corrections."""
    corrections: List[CorrectionRule]
    total: int


class CreateCorrectionRequest(BaseModel):
    """Requête de création de correction."""
    pattern: str = Field(..., min_length=1)
    replacement: str
    confidence: str = Field(default="high")
    notes: str = Field(default="")


class UpdateCorrectionRequest(BaseModel):
    """Requête de mise à jour de correction."""
    pattern: Optional[str] = None
    replacement: Optional[str] = None
    confidence: Optional[str] = None
    notes: Optional[str] = None


class ApplyCorrectionsRequest(BaseModel):
    """Requête d'application des corrections."""
    text: str = Field(..., min_length=1)
    confidence_levels: List[str] = Field(default=["high", "medium"])


class ApplyCorrectionsResponse(BaseModel):
    """Réponse avec texte corrigé."""
    original: str
    corrected: str
    changes_count: int


def _load_corrections() -> List[dict]:
    """Charge les corrections depuis le fichier JSON."""
    if not CORRECTIONS_FILE.exists():
        return []
    try:
        data = json.loads(CORRECTIONS_FILE.read_text(encoding="utf-8"))
        return data.get("corrections", [])
    except Exception:
        return []


def _save_corrections(corrections: List[dict]):
    """Sauvegarde les corrections dans le fichier JSON."""
    CORRECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {"corrections": corrections}
    CORRECTIONS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@router.get("", response_model=CorrectionsList)
async def list_corrections(search: Optional[str] = None):
    """Liste toutes les corrections de prononciation."""
    corrections = _load_corrections()

    if search:
        search_lower = search.lower()
        corrections = [
            c for c in corrections
            if search_lower in c.get("pattern", "").lower()
            or search_lower in c.get("replacement", "").lower()
            or search_lower in c.get("notes", "").lower()
        ]

    return CorrectionsList(
        corrections=[CorrectionRule(**c) for c in corrections],
        total=len(corrections)
    )


@router.get("/{correction_id}", response_model=CorrectionRule)
async def get_correction(correction_id: str):
    """Récupère une correction par son ID."""
    corrections = _load_corrections()
    for c in corrections:
        if c.get("id") == correction_id:
            return CorrectionRule(**c)
    raise APIError(ErrorCode.NOT_FOUND, f"Correction {correction_id} non trouvée", status_code=404)


@router.post("", response_model=CorrectionRule)
async def create_correction(request: CreateCorrectionRequest):
    """Crée une nouvelle correction."""
    corrections = _load_corrections()

    # Vérifier si le pattern existe déjà
    for c in corrections:
        if c.get("pattern") == request.pattern:
            raise APIError(ErrorCode.VALIDATION_ERROR, f"Pattern '{request.pattern}' existe déjà")

    # Générer un ID unique
    import uuid
    correction_id = f"corr_{uuid.uuid4().hex[:8]}"

    new_correction = {
        "id": correction_id,
        "pattern": request.pattern,
        "replacement": request.replacement,
        "confidence": request.confidence,
        "notes": request.notes,
    }

    corrections.append(new_correction)
    _save_corrections(corrections)

    return CorrectionRule(**new_correction)


@router.put("/{correction_id}", response_model=CorrectionRule)
async def update_correction(correction_id: str, request: UpdateCorrectionRequest):
    """Met à jour une correction existante."""
    corrections = _load_corrections()

    for i, c in enumerate(corrections):
        if c.get("id") == correction_id:
            if request.pattern is not None:
                c["pattern"] = request.pattern
            if request.replacement is not None:
                c["replacement"] = request.replacement
            if request.confidence is not None:
                c["confidence"] = request.confidence
            if request.notes is not None:
                c["notes"] = request.notes

            corrections[i] = c
            _save_corrections(corrections)
            return CorrectionRule(**c)

    raise APIError(ErrorCode.NOT_FOUND, f"Correction {correction_id} non trouvée", status_code=404)


@router.delete("/{correction_id}")
async def delete_correction(correction_id: str):
    """Supprime une correction."""
    corrections = _load_corrections()

    for i, c in enumerate(corrections):
        if c.get("id") == correction_id:
            corrections.pop(i)
            _save_corrections(corrections)
            return {"success": True, "message": f"Correction {correction_id} supprimée"}

    raise APIError(ErrorCode.NOT_FOUND, f"Correction {correction_id} non trouvée", status_code=404)


@router.post("/apply", response_model=ApplyCorrectionsResponse)
async def apply_corrections(request: ApplyCorrectionsRequest):
    """Applique les corrections à un texte."""
    corrections = _load_corrections()

    original = request.text
    corrected = request.text
    changes_count = 0

    for c in corrections:
        if c.get("confidence", "high") not in request.confidence_levels:
            continue

        pattern = c.get("pattern", "")
        replacement = c.get("replacement", "")

        if pattern in corrected:
            count = corrected.count(pattern)
            corrected = corrected.replace(pattern, replacement)
            changes_count += count

    return ApplyCorrectionsResponse(
        original=original,
        corrected=corrected,
        changes_count=changes_count
    )


@router.post("/import")
async def import_corrections(corrections: List[CreateCorrectionRequest]):
    """Importe plusieurs corrections d'un coup."""
    existing = _load_corrections()
    existing_patterns = {c.get("pattern") for c in existing}

    import uuid
    added = 0
    skipped = 0

    for req in corrections:
        if req.pattern in existing_patterns:
            skipped += 1
            continue

        correction_id = f"corr_{uuid.uuid4().hex[:8]}"
        existing.append({
            "id": correction_id,
            "pattern": req.pattern,
            "replacement": req.replacement,
            "confidence": req.confidence,
            "notes": req.notes,
        })
        existing_patterns.add(req.pattern)
        added += 1

    _save_corrections(existing)

    return {
        "success": True,
        "added": added,
        "skipped": skipped,
        "total": len(existing)
    }


@router.get("/export/json")
async def export_corrections():
    """Exporte toutes les corrections au format JSON."""
    corrections = _load_corrections()
    return {"corrections": corrections, "total": len(corrections)}
