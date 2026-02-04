# AudioReader v4.0 — Multi-stage Docker build
# Stage 1: Builder
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    espeak-ng \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app

# Copier le code source
COPY audio_reader.py app.py run_tests.py ./
COPY src/ src/
COPY api/ api/

# Creer les dossiers necessaires
RUN mkdir -p output .audioreader_data/uploads .voice_cache .synthesis_cache

# Telecharger les modeles Kokoro (si pas montes en volume)
# RUN python -c "from huggingface_hub import hf_hub_download; ..."

EXPOSE 7860 8000

# Par defaut: lancer l'API v2
CMD ["python", "-m", "api"]
