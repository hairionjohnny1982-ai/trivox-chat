# TriVox Chat

Chat local avec RAG, memoire persistante et recherche de code — propulse par TriVox3.

Remplace open-webui. Utilise TriVox3 pour les embeddings, Qdrant pour le stockage vectoriel, et Ollama pour l'inference LLM.

## Features

- **Chat Ollama** — Tous les modeles installes, streaming, markdown + code highlighting
- **Code Search** — Indexe tes repos GitHub, cherche par description naturelle (FR/EN/Code)
- **Memoire** — Se souvient des conversations passees (Qdrant + TriVox3)
- **RAG** — Contexte automatique depuis la memoire + le code indexe
- **Trilingue** — Francais, Anglais et Code nativement (3 tokenizers)

## Installation

```bash
git clone https://github.com/YOUR_USER/trivox-chat && cd trivox-chat
docker compose up -d
```

Ouvre `http://localhost:3000`

## Configuration

Editer les variables dans `docker-compose.yml` :

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_URL` | URL du serveur Ollama | `http://host.docker.internal:11434` |
| `MODEL_PATH` | Chemin du checkpoint TriVox | `/data/model/trivox_latest.pt` |
| `TOKENIZERS_DIR` | Dossier des tokenizers | `/data/tokenizers` |
| `EMBED_DIM` | Dimension des embeddings (768 pour v2, 384 pour v3) | `768` |

## Architecture

```
Chat UI (port 3000)
    |
FastAPI Backend
    |--- Ollama (LLM inference)
    |--- TriVox Encoder (embeddings FR/EN/Code)
    |--- Qdrant (vector storage)
    |--- GitHub Indexer (clone + chunk + embed)
    |--- Memory Store (conversations + entities)
```

## Stack

- **Backend**: FastAPI + uvicorn
- **Embeddings**: TriVox3 (39M params, 384d, CPU inference ~5ms)
- **Vector DB**: Qdrant
- **LLM**: Ollama (tous modeles supportes)
- **Frontend**: Vanilla JS, marked.js, highlight.js
