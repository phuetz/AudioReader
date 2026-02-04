"""Endpoints projets : CRUD JSON simple."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter

from api.dependencies import PROJECTS_DIR
from api.errors import APIError, ErrorCode
from api.models import ProjectCreate, ProjectInfo, ProjectUpdate

router = APIRouter(prefix="/api/v2", tags=["Projects"])


def _load_project(project_id: str) -> dict | None:
    path = PROJECTS_DIR / project_id / "project.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save_project(data: dict):
    project_dir = PROJECTS_DIR / data["id"]
    project_dir.mkdir(exist_ok=True)
    path = project_dir / "project.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@router.get("/projects")
async def list_projects():
    """Liste tous les projets."""
    projects = []
    for d in PROJECTS_DIR.iterdir():
        if d.is_dir():
            data = _load_project(d.name)
            if data:
                projects.append(ProjectInfo(**data))
    projects.sort(key=lambda p: p.updated_at, reverse=True)
    return {"projects": projects, "total": len(projects)}


@router.get("/projects/{project_id}", response_model=ProjectInfo)
async def get_project(project_id: str):
    """Détails d'un projet."""
    data = _load_project(project_id)
    if not data:
        raise APIError(ErrorCode.NOT_FOUND, f"Projet {project_id} non trouvé", status_code=404)
    return ProjectInfo(**data)


@router.post("/projects", response_model=ProjectInfo)
async def create_project(request: ProjectCreate):
    """Crée un nouveau projet."""
    project_id = uuid.uuid4().hex[:8]
    now = datetime.now().isoformat()
    data = {
        "id": project_id,
        "name": request.name,
        "description": request.description,
        "settings": {},
        "created_at": now,
        "updated_at": now,
        "files": [],
    }
    _save_project(data)
    return ProjectInfo(**data)


@router.put("/projects/{project_id}", response_model=ProjectInfo)
async def update_project(project_id: str, request: ProjectUpdate):
    """Met à jour un projet."""
    data = _load_project(project_id)
    if not data:
        raise APIError(ErrorCode.NOT_FOUND, f"Projet {project_id} non trouvé", status_code=404)

    if request.name is not None:
        data["name"] = request.name
    if request.description is not None:
        data["description"] = request.description
    if request.settings is not None:
        data["settings"].update(request.settings)
    data["updated_at"] = datetime.now().isoformat()

    _save_project(data)
    return ProjectInfo(**data)


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    """Supprime un projet."""
    project_dir = PROJECTS_DIR / project_id
    if not project_dir.exists():
        raise APIError(ErrorCode.NOT_FOUND, f"Projet {project_id} non trouvé", status_code=404)

    import shutil
    shutil.rmtree(project_dir)
    return {"success": True, "message": f"Projet {project_id} supprimé"}
