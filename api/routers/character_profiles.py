"""Profils de personnages persistants — sauvegarde et réutilisation."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.dependencies import DATA_DIR
from api.errors import APIError, ErrorCode

router = APIRouter(prefix="/api/v2", tags=["Character Profiles"])

PROFILES_FILE = DATA_DIR / "character_profiles.json"


class VoiceMorphSettings(BaseModel):
    pitch: float = Field(default=0.0, ge=-1.0, le=1.0)
    formant: float = Field(default=0.0, ge=-1.0, le=1.0)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


class CharacterProfileCreate(BaseModel):
    name: str
    aliases: List[str] = []
    gender: Optional[str] = None
    voice_id: str = "ff_siwis"
    voice_settings: VoiceMorphSettings = VoiceMorphSettings()
    personality_notes: str = ""


class CharacterProfileUpdate(BaseModel):
    name: Optional[str] = None
    aliases: Optional[List[str]] = None
    gender: Optional[str] = None
    voice_id: Optional[str] = None
    voice_settings: Optional[VoiceMorphSettings] = None
    personality_notes: Optional[str] = None


class CharacterProfile(BaseModel):
    id: str
    name: str
    aliases: List[str] = []
    gender: Optional[str] = None
    voice_id: str = "ff_siwis"
    voice_settings: VoiceMorphSettings = VoiceMorphSettings()
    personality_notes: str = ""
    created_at: str = ""
    usage_count: int = 0


def _load_profiles() -> list[dict]:
    if not PROFILES_FILE.exists():
        return []
    try:
        data = json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
        return data.get("profiles", [])
    except Exception:
        return []


def _save_profiles(profiles: list[dict]):
    PROFILES_FILE.write_text(
        json.dumps({"profiles": profiles}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@router.get("/character-profiles")
async def list_profiles(search: Optional[str] = None) -> dict:
    """Liste tous les profils de personnages."""
    profiles = _load_profiles()
    if search:
        q = search.lower()
        profiles = [
            p for p in profiles
            if q in p["name"].lower() or any(q in a.lower() for a in p.get("aliases", []))
        ]
    return {"profiles": [CharacterProfile(**p) for p in profiles], "total": len(profiles)}


@router.post("/character-profiles")
async def create_profile(body: CharacterProfileCreate) -> CharacterProfile:
    """Crée un nouveau profil de personnage."""
    profiles = _load_profiles()
    profile = {
        "id": f"prof_{uuid.uuid4().hex[:6]}",
        "name": body.name,
        "aliases": body.aliases,
        "gender": body.gender,
        "voice_id": body.voice_id,
        "voice_settings": body.voice_settings.model_dump(),
        "personality_notes": body.personality_notes,
        "created_at": datetime.now().isoformat(),
        "usage_count": 0,
    }
    profiles.append(profile)
    _save_profiles(profiles)
    return CharacterProfile(**profile)


@router.get("/character-profiles/{profile_id}")
async def get_profile(profile_id: str) -> CharacterProfile:
    """Récupère un profil par ID."""
    profiles = _load_profiles()
    profile = next((p for p in profiles if p["id"] == profile_id), None)
    if not profile:
        raise APIError(ErrorCode.NOT_FOUND, f"Profil {profile_id} non trouvé", status_code=404)
    return CharacterProfile(**profile)


@router.put("/character-profiles/{profile_id}")
async def update_profile(profile_id: str, body: CharacterProfileUpdate) -> CharacterProfile:
    """Met à jour un profil."""
    profiles = _load_profiles()
    profile = next((p for p in profiles if p["id"] == profile_id), None)
    if not profile:
        raise APIError(ErrorCode.NOT_FOUND, f"Profil {profile_id} non trouvé", status_code=404)

    updates = body.model_dump(exclude_none=True)
    if "voice_settings" in updates:
        updates["voice_settings"] = updates["voice_settings"].model_dump() if hasattr(updates["voice_settings"], "model_dump") else updates["voice_settings"]
    profile.update(updates)
    _save_profiles(profiles)
    return CharacterProfile(**profile)


@router.delete("/character-profiles/{profile_id}")
async def delete_profile(profile_id: str) -> dict:
    """Supprime un profil."""
    profiles = _load_profiles()
    before = len(profiles)
    profiles = [p for p in profiles if p["id"] != profile_id]
    if len(profiles) == before:
        raise APIError(ErrorCode.NOT_FOUND, f"Profil {profile_id} non trouvé", status_code=404)
    _save_profiles(profiles)
    return {"success": True, "message": f"Profil {profile_id} supprimé"}


@router.post("/character-profiles/import-from-analysis")
async def import_from_analysis(body: dict) -> dict:
    """Importe des personnages depuis un résultat d'analyse comme profils."""
    characters = body.get("characters", [])
    if not characters:
        raise APIError(ErrorCode.VALIDATION_ERROR, "Aucun personnage à importer")

    profiles = _load_profiles()
    imported = 0
    for char in characters:
        name = char.get("name", "")
        if not name:
            continue
        # Skip if already exists
        if any(p["name"].lower() == name.lower() for p in profiles):
            continue
        profile = {
            "id": f"prof_{uuid.uuid4().hex[:6]}",
            "name": name,
            "aliases": [],
            "gender": char.get("gender"),
            "voice_id": char.get("suggested_voice", "ff_siwis"),
            "voice_settings": {"pitch": 0.0, "formant": 0.0, "speed": 1.0},
            "personality_notes": "",
            "created_at": datetime.now().isoformat(),
            "usage_count": 0,
        }
        profiles.append(profile)
        imported += 1

    _save_profiles(profiles)
    return {"success": True, "imported": imported, "total": len(profiles)}
