"""
AudioReader API v2 — FastAPI application factory.

Usage:
    uvicorn api:create_app --reload --port 8000
    # ou
    python -m api
"""
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ajouter le répertoire parent au path pour les imports src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.routers import generation, jobs, voices, files, analysis, podcast, projects, config, openai_compat


def create_app() -> FastAPI:
    """Crée et configure l'application FastAPI v2."""
    app = FastAPI(
        title="AudioReader API",
        description="API v2 pour AudioReader — génération d'audiobooks haute qualité",
        version="4.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Routers v2 ---
    app.include_router(config.router)
    app.include_router(generation.router)
    app.include_router(jobs.router)
    app.include_router(voices.router)
    app.include_router(files.router)
    app.include_router(analysis.router)
    app.include_router(podcast.router)
    app.include_router(projects.router)

    # --- OpenAI-compatible API ---
    app.include_router(openai_compat.router)

    # --- Monter l'ancienne API v1 ---
    try:
        from api_server import app as v1_app
        app.mount("/v1", v1_app)
    except Exception:
        pass

    # --- Servir les fichiers audio générés ---
    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    app.mount("/output", StaticFiles(directory=str(output_dir)), name="output")

    return app


app = create_app()


def main():
    """Point d'entrée CLI."""
    import uvicorn
    print("=" * 50)
    print("AudioReader API v2 — http://0.0.0.0:8000")
    print("Docs: http://0.0.0.0:8000/docs")
    print("=" * 50)
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
