# Integration AudioReader avec Claude.ai et ChatGPT

Ce document explique comment piloter AudioReader depuis Claude.ai ou ChatGPT.

## Vue d'ensemble

AudioReader expose deux interfaces pour le pilotage par IA :

| Interface | Protocole | Utilisation |
|-----------|-----------|-------------|
| **MCP Server** | Model Context Protocol | Claude.ai, Claude Desktop |
| **REST API** | HTTP/OpenAPI | ChatGPT Actions, GPTs personnalises, tout client HTTP |

---

## 1. Integration Claude.ai / Claude Desktop (MCP)

### Installation

```bash
# Installer le package MCP
pip install mcp

# Verifier l'installation
python -c "import mcp; print('MCP OK')"
```

### Configuration Claude Desktop

Editez le fichier de configuration Claude Desktop :

**macOS** : `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows** : `%APPDATA%\Claude\claude_desktop_config.json`
**Linux** : `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "audioreader": {
      "command": "python",
      "args": ["/chemin/vers/AudioReader/mcp_server.py"],
      "env": {}
    }
  }
}
```

### Demarrage

Redemarrez Claude Desktop. Le serveur AudioReader sera disponible automatiquement.

### Outils MCP disponibles

| Outil | Description |
|-------|-------------|
| `list_voices` | Liste les voix disponibles |
| `generate_audio` | Genere un audio simple |
| `generate_audiobook` | Genere un audiobook complet |
| `analyze_text` | Analyse dialogues et emotions |
| `get_config` | Configuration actuelle |
| `set_config` | Modifier la configuration |
| `list_output_files` | Liste les fichiers generes |

### Exemples d'utilisation avec Claude

```
Utilisateur : "Genere un audio du texte 'Bonjour, je suis AudioReader'"

Claude : [Appelle l'outil generate_audio]
L'audio a ete genere avec succes !
- Fichier : output/audio_abc123.wav
- Duree : 2.5 secondes
```

```
Utilisateur : "Convertis ce chapitre en audiobook avec des voix differentes pour chaque personnage"

Claude : [Appelle l'outil generate_audiobook]
Audiobook genere :
- 45 segments traites
- 3 personnages detectes : Victor, Marie, le narrateur
- Duree totale : 5:32
```

---

## 2. Integration ChatGPT (REST API)

### Installation

```bash
# Installer les dependances
pip install fastapi uvicorn python-multipart

# Demarrer le serveur
python api_server.py
```

Le serveur demarre sur `http://localhost:8000`.

### Endpoints disponibles

| Methode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Info API |
| GET | `/voices` | Liste des voix |
| POST | `/generate` | Genere audio simple |
| POST | `/audiobook` | Genere audiobook complet |
| POST | `/analyze` | Analyse texte |
| GET | `/config` | Configuration |
| GET | `/files` | Liste fichiers |
| GET | `/files/{name}` | Telecharge fichier |
| GET | `/jobs/{id}` | Statut d'un job |
| GET | `/docs` | Documentation Swagger |
| GET | `/openapi.json` | Spec OpenAPI |

### Creer un GPT personnalise (ChatGPT Plus)

1. Allez sur https://chat.openai.com/gpts
2. Cliquez "Create a GPT"
3. Dans "Configure", allez a "Actions"
4. Cliquez "Create new action"
5. Importez le schema depuis `openapi_chatgpt.json` ou l'URL `/openapi.json`

### Configuration de l'Action

**Schema URL** : `https://votre-serveur.com/openapi.json`

**Authentication** : None (ou API Key si vous ajoutez l'authentification)

**Privacy Policy** : Votre URL de politique de confidentialite

### Instructions pour le GPT

Ajoutez ces instructions dans la configuration du GPT :

```
Tu es un assistant specialise dans la creation d'audiobooks avec AudioReader.

Quand l'utilisateur te demande de creer un audio ou audiobook :
1. Utilise l'action generateAudiobook pour les textes longs
2. Utilise l'action generateAudio pour les textes courts
3. Verifie le statut avec getJobStatus jusqu'a completion
4. Fournis le lien de telechargement quand c'est pret

Pour les textes en francais, utilise la voix ff_siwis.
Pour les textes en anglais, utilise la voix af_heart.

Styles disponibles : storytelling (defaut), dramatic, formal, conversational
```

### Exemple de conversation avec le GPT

```
Utilisateur : Convertis ce texte en audiobook avec un style dramatique :
"La nuit etait sombre. Victor entra dans la piece.
« Qui est la ? » demanda Marie, effrayee."

GPT : Je lance la generation de l'audiobook en style dramatique...
[Appelle generateAudiobook]

La generation est en cours (job: abc123). Je verifie le statut...
[Appelle getJobStatus]

Votre audiobook est pret !
- Duree : 15 secondes
- 2 personnages detectes : Victor, Marie
- Telechargement : [lien]
```

---

## 3. Deploiement en production

### Avec Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["python", "api_server.py"]
```

```bash
docker build -t audioreader-api .
docker run -p 8000:8000 audioreader-api
```

### Avec ngrok (developpement)

```bash
# Installer ngrok
# Demarrer le serveur local
python api_server.py &

# Exposer sur Internet
ngrok http 8000
```

Utilisez l'URL ngrok dans votre configuration ChatGPT Actions.

### Securite (recommande pour production)

Ajoutez une authentification API Key :

```python
# Dans api_server.py
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = "votre-cle-secrete"
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Ajoutez dependencies=[Depends(verify_api_key)] aux routes
```

---

## 4. Exemples de code client

### Python

```python
import requests

# Generer un audio
response = requests.post("http://localhost:8000/generate", json={
    "text": "Bonjour, ceci est un test.",
    "voice": "ff_siwis",
    "speed": 1.0
})
job_id = response.json()["job_id"]

# Attendre la completion
import time
while True:
    status = requests.get(f"http://localhost:8000/jobs/{job_id}").json()
    if status["status"] == "completed":
        print(f"Fichier pret : {status['result']['download_url']}")
        break
    elif status["status"] == "failed":
        print(f"Erreur : {status['error']}")
        break
    time.sleep(1)
```

### JavaScript

```javascript
async function generateAudio(text) {
    // Demarrer la generation
    const response = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, voice: 'ff_siwis' })
    });
    const { job_id } = await response.json();

    // Attendre la completion
    while (true) {
        const status = await fetch(`http://localhost:8000/jobs/${job_id}`);
        const data = await status.json();

        if (data.status === 'completed') {
            return data.result.download_url;
        }
        if (data.status === 'failed') {
            throw new Error(data.error);
        }
        await new Promise(r => setTimeout(r, 1000));
    }
}
```

### cURL

```bash
# Generer un audio
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour!", "voice": "ff_siwis"}'

# Verifier le statut
curl http://localhost:8000/jobs/abc123

# Telecharger le fichier
curl -O http://localhost:8000/files/audio_abc123.wav
```

---

## 5. Troubleshooting

### Le serveur MCP ne demarre pas

```bash
# Verifier que mcp est installe
pip install mcp

# Tester le serveur manuellement
python mcp_server.py
```

### L'API REST ne repond pas

```bash
# Verifier que les dependances sont installees
pip install fastapi uvicorn

# Demarrer en mode debug
uvicorn api_server:app --reload --log-level debug
```

### Erreur de synthese vocale

```bash
# Verifier que Kokoro ou Edge-TTS est disponible
python -c "from src.tts_unified import UnifiedTTS; print('TTS OK')"
```

### Les fichiers audio ne se telecharget pas

Verifiez que le dossier `output/` existe et a les bonnes permissions.

---

## 6. Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Claude.ai     │     │    ChatGPT      │
│  Claude Desktop │     │   GPT Actions   │
└────────┬────────┘     └────────┬────────┘
         │ MCP                   │ HTTP
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  mcp_server.py  │     │  api_server.py  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌─────────────────────┐
         │   AudioReader Core  │
         │  - TTS Unified      │
         │  - HQ Pipeline      │
         │  - Emotion Analyzer │
         │  - Bio Acoustics    │
         └─────────────────────┘
                     │
                     ▼
         ┌─────────────────────┐
         │   output/*.wav      │
         └─────────────────────┘
```
