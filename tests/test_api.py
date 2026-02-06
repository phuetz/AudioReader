"""Tests API v2 — health, upload, jobs paginés, cycle de vie job."""
import os
import sys
import tempfile

import pytest

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    """Crée un client HTTPX pour tester l'API."""
    import httpx
    from api import create_app

    app = create_app()

    # Init DB before tests
    from api.database import init_db
    await init_db()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.anyio
async def test_health(client):
    """Test endpoint health."""
    r = await client.get("/api/v2/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.anyio
async def test_upload_file(client):
    """Test upload d'un fichier markdown."""
    content = b"# Chapitre 1\n\nHello world"
    files = {"file": ("test.md", content, "text/markdown")}
    r = await client.post("/api/v2/files/upload", files=files)
    assert r.status_code == 200
    data = r.json()
    assert "file_id" in data
    assert data["original_name"] == "test.md"
    assert data["mime_type"] == "text/markdown"


@pytest.mark.anyio
async def test_jobs_paginated(client):
    """Test liste paginée des jobs."""
    r = await client.get("/api/v2/jobs", params={"offset": 0, "limit": 10})
    assert r.status_code == 200
    data = r.json()
    assert "jobs" in data
    assert "total" in data
    assert "offset" in data
    assert "limit" in data
    assert isinstance(data["jobs"], list)


@pytest.mark.anyio
async def test_job_lifecycle(client):
    """Test cycle de vie complet d'un job : création via generate, get, cancel."""
    # Create a job via the generate endpoint
    r = await client.post("/api/v2/generate", json={
        "text": "Bonjour le monde",
        "voice": "ff_siwis",
        "speed": 1.0,
    })
    # May fail if TTS not available, but should return 200 with job_id
    assert r.status_code == 200
    data = r.json()
    job_id = data["job_id"]
    assert job_id

    # Get job
    r = await client.get(f"/api/v2/jobs/{job_id}")
    assert r.status_code == 200
    assert r.json()["job_id"] == job_id

    # Cancel job
    r = await client.post(f"/api/v2/jobs/{job_id}/cancel")
    # May be already done, accept either success or error
    assert r.status_code in (200, 400)

    # Get non-existing job
    r = await client.get("/api/v2/jobs/nonexistent")
    assert r.status_code == 404
