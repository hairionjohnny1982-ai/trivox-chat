"""TriVox Chat — Standalone chat + RAG + memory + code search.

Replaces open-webui. Uses TriVox3 for embeddings, Qdrant for vector storage,
Ollama for LLM inference. All models available in Ollama are accessible.
"""
import json
import os
import re
import time
import uuid
import logging
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import (
    OLLAMA_URL, HOST, PORT, DATA_DIR, REPOS_DIR, CONVERSATIONS_DIR,
    MAX_MEMORIES_INJECT, QDRANT_HOST, QDRANT_PORT, EMBED_DIM,
    MODEL_PATH, TOK_FR, TOK_EN, TOK_CODE, MAX_SEQ_LEN
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("trivox-chat")

# Globals
encoder = None
memory_store = None
http_client = None


def init_encoder():
    """Try to load TriVox encoder. Falls back to None if model not found."""
    global encoder
    if os.path.exists(MODEL_PATH) and os.path.exists(TOK_FR):
        try:
            from encoder import Encoder
            encoder = Encoder(MODEL_PATH, TOK_FR, TOK_EN, TOK_CODE, MAX_SEQ_LEN)
            log.info(f"TriVox encoder loaded ({EMBED_DIM}d)")
        except Exception as e:
            log.warning(f"Failed to load TriVox encoder: {e}")
            encoder = None
    else:
        log.warning(f"TriVox model not found at {MODEL_PATH}, running without embeddings")


def init_memory():
    """Try to connect to Qdrant. Falls back to None."""
    global memory_store
    if encoder is None:
        log.warning("No encoder — memory disabled")
        return
    try:
        from memory import MemoryStore
        memory_store = MemoryStore(encoder)
        log.info("Memory store connected (Qdrant)")
    except Exception as e:
        log.warning(f"Qdrant not available: {e} — memory disabled")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=120.0)
    os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
    os.makedirs(REPOS_DIR, exist_ok=True)
    init_encoder()
    init_memory()
    yield
    await http_client.aclose()


app = FastAPI(title="TriVox Chat", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─── Ollama Models ───

@app.get("/api/models")
async def list_models():
    """List all available Ollama models."""
    try:
        resp = await http_client.get(f"{OLLAMA_URL}/api/tags")
        data = resp.json()
        models = []
        for m in data.get("models", []):
            models.append({
                "name": m["name"],
                "size": m.get("size", 0),
                "modified": m.get("modified_at", ""),
                "family": m.get("details", {}).get("family", ""),
            })
        return {"models": models}
    except Exception as e:
        return {"models": [], "error": str(e)}


# ─── Conversations ───

def _conv_path(conv_id: str) -> Path:
    return Path(CONVERSATIONS_DIR) / f"{conv_id}.json"


def _load_conv(conv_id: str) -> dict:
    p = _conv_path(conv_id)
    if p.exists():
        return json.loads(p.read_text())
    return None


def _save_conv(conv: dict):
    p = _conv_path(conv["id"])
    p.write_text(json.dumps(conv, ensure_ascii=False, indent=2))


@app.get("/api/conversations")
async def list_conversations():
    """List all conversations, newest first."""
    convs = []
    for f in Path(CONVERSATIONS_DIR).glob("*.json"):
        try:
            c = json.loads(f.read_text())
            convs.append({
                "id": c["id"],
                "title": c.get("title", "New Chat"),
                "model": c.get("model", ""),
                "created": c.get("created", ""),
                "updated": c.get("updated", ""),
                "message_count": len(c.get("messages", [])),
            })
        except:
            pass
    convs.sort(key=lambda x: x.get("updated", ""), reverse=True)
    return {"conversations": convs}


@app.post("/api/conversations")
async def create_conversation(req: Request):
    body = await req.json()
    conv = {
        "id": str(uuid.uuid4())[:8],
        "title": body.get("title", "New Chat"),
        "model": body.get("model", ""),
        "messages": [],
        "created": datetime.now(timezone.utc).isoformat(),
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    _save_conv(conv)
    return conv


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    conv = _load_conv(conv_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    return conv


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    p = _conv_path(conv_id)
    if p.exists():
        p.unlink()
    return {"ok": True}


@app.put("/api/conversations/{conv_id}/title")
async def update_title(conv_id: str, req: Request):
    body = await req.json()
    conv = _load_conv(conv_id)
    if not conv:
        raise HTTPException(404)
    conv["title"] = body.get("title", conv["title"])
    _save_conv(conv)
    return {"ok": True}


# ─── Chat (streaming) ───

def _build_system_prompt(query: str, conv_id: str = None, project: str = None) -> str:
    """Build system prompt with RAG context from code search + memory."""
    parts = []
    code_results = []
    memories = []

    if memory_store and encoder:
        emb = encoder.encode(query)

        # 1. Search indexed code repos (priority)
        try:
            collections = memory_store.client.get_collections().collections
            repo_collections = [c.name for c in collections if c.name.startswith("repo_")]
            # If project is active, only search that project's collection
            if project:
                target = f"repo_{project.lower().replace('-', '_')}"
                repo_collections = [c for c in repo_collections if c == target]

            for col in repo_collections:
                try:
                    hits = memory_store.client.query_points(
                        collection_name=col, query=emb, limit=8,
                    ).points
                    for h in hits:
                        if h.score >= 0.35:
                            code_results.append({
                                "text": h.payload.get("text", ""),
                                "file": h.payload.get("file", ""),
                                "repo": h.payload.get("repo", col.replace("repo_", "")),
                                "type": h.payload.get("type", "code"),
                                "score": h.score,
                            })
                except Exception:
                    pass
        except Exception as e:
            log.debug(f"Code search error: {e}")

        # 2. Search memories (lower priority)
        try:
            memories = memory_store.recall(query, project=None, top_k=3)
        except Exception:
            memories = []

    # Build the system prompt
    if code_results:
        # Deduplicate
        seen = set()
        unique = []
        for r in sorted(code_results, key=lambda x: x["score"], reverse=True):
            key = r["text"][:150]
            if key not in seen:
                seen.add(key)
                unique.append(r)
        code_results = unique[:6]

        parts.append("Tu es un assistant expert en code. Voici des extraits de code source pertinents trouves dans le projet. Utilise-les pour repondre precisement en citant les fichiers, fonctions et chemins exacts.\n")
        parts.append("## Code source du projet\n")
        for r in code_results:
            repo = r["repo"]
            fpath = r["file"]
            score = r["score"]
            parts.append(f"### {fpath} (repo: {repo}, pertinence: {score*100:.0f}%)")
            parts.append(f"```")
            parts.append(r["text"][:1500])
            parts.append(f"```\n")

    if memories:
        parts.append("## Memoire contextuelle\n")
        for m in memories:
            parts.append(f"- {m['text'][:300]}")
        parts.append("")

    if parts:
        parts.append("---")
        parts.append("Reponds en utilisant les extraits de code ci-dessus. Cite les noms de fichiers et de fonctions. Si tu montres du code, indique toujours dans quel fichier et quel dossier il se trouve.")

    return "\n".join(parts)


async def _generate_title(messages: list, model: str) -> str:
    """Auto-generate conversation title from first exchange."""
    if len(messages) < 2:
        return "New Chat"
    prompt = f"Generate a very short title (max 5 words) for this conversation. Only output the title, nothing else.\n\nUser: {messages[0]['content'][:200]}\nAssistant: {messages[1]['content'][:200]}"
    try:
        resp = await http_client.post(f"{OLLAMA_URL}/api/generate", json={
            "model": model, "prompt": prompt, "stream": False,
            "options": {"num_predict": 20, "temperature": 0.3}
        }, timeout=15.0)
        title = resp.json().get("response", "").strip().strip('"').strip("'")
        return title[:60] if title else "New Chat"
    except:
        return "New Chat"


@app.post("/api/chat")
async def chat(req: Request):
    """Chat endpoint with streaming, RAG, and memory."""
    body = await req.json()
    model = body.get("model", "qwen2.5:7b-instruct-q4_K_M")
    messages = body.get("messages", [])
    conv_id = body.get("conversation_id")
    stream = body.get("stream", True)
    project = body.get("project")

    if not messages:
        raise HTTPException(400, "No messages")

    user_msg = messages[-1].get("content", "")

    # RAG: inject context from code + memory
    rag_context = _build_system_prompt(user_msg, conv_id, project=project)
    ollama_messages = []
    if rag_context:
        ollama_messages.append({"role": "system", "content": rag_context})
    ollama_messages.extend(messages)

    # Save user message to conversation
    conv = None
    if conv_id:
        conv = _load_conv(conv_id)
    if conv is None:
        conv = {
            "id": conv_id or str(uuid.uuid4())[:8],
            "title": "New Chat",
            "model": model,
            "messages": [],
            "created": datetime.now(timezone.utc).isoformat(),
            "updated": datetime.now(timezone.utc).isoformat(),
        }
    conv["messages"].append({"role": "user", "content": user_msg, "timestamp": datetime.now(timezone.utc).isoformat()})
    conv["model"] = model
    conv["updated"] = datetime.now(timezone.utc).isoformat()

    if stream:
        async def stream_response():
            full_response = ""
            try:
                async with http_client.stream("POST", f"{OLLAMA_URL}/api/chat", json={
                    "model": model,
                    "messages": ollama_messages,
                    "stream": True,
                }) as resp:
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                full_response += token
                                yield f"data: {json.dumps({'token': token})}\n\n"
                            if chunk.get("done"):
                                yield f"data: {json.dumps({'done': True})}\n\n"
                        except json.JSONDecodeError:
                            pass
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

            # Save assistant response
            conv["messages"].append({
                "role": "assistant", "content": full_response,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            # Auto-title on first exchange
            if len(conv["messages"]) == 2 and conv["title"] == "New Chat":
                conv["title"] = await _generate_title(conv["messages"], model)
            _save_conv(conv)

            # Store in memory
            if memory_store and full_response:
                try:
                    from chunker import chunk_text
                    chunks = chunk_text(f"User: {user_msg}\nAssistant: {full_response}")
                    for ch in chunks:
                        memory_store.store(ch, project=None)
                except Exception as e:
                    log.debug(f"Memory store error: {e}")

        return StreamingResponse(stream_response(), media_type="text/event-stream")
    else:
        # Non-streaming
        resp = await http_client.post(f"{OLLAMA_URL}/api/chat", json={
            "model": model, "messages": ollama_messages, "stream": False,
        })
        data = resp.json()
        assistant_msg = data.get("message", {}).get("content", "")
        conv["messages"].append({
            "role": "assistant", "content": assistant_msg,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        if len(conv["messages"]) == 2 and conv["title"] == "New Chat":
            conv["title"] = await _generate_title(conv["messages"], model)
        _save_conv(conv)
        return {"message": assistant_msg, "conversation_id": conv["id"]}


# ─── GitHub Repo Indexing ───

@app.post("/api/repos/index")
async def index_repo_endpoint(req: Request):
    """Clone and index a GitHub repo with SSE progress logs."""
    body = await req.json()
    repo_url = body.get("url", "").strip()
    if not repo_url:
        raise HTTPException(400, "Missing repo URL")
    if not encoder:
        raise HTTPException(503, "TriVox encoder not loaded")

    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    repo_dir = os.path.join(REPOS_DIR, repo_name)

    import queue
    log_q: queue.Queue = queue.Queue()

    def do_index():
        import subprocess
        from chunker import chunk_code, chunk_text
        from qdrant_client.models import PointStruct

        # Clone or pull
        if os.path.exists(os.path.join(repo_dir, ".git")):
            log_q.put(f"git pull {repo_name}...")
            subprocess.run(["git", "-C", repo_dir, "pull"], capture_output=True, timeout=120)
            action = "updated"
        else:
            log_q.put(f"git clone {repo_url} (depth=1)...")
            r = subprocess.run(["git", "clone", "--depth", "1", repo_url, repo_dir],
                              capture_output=True, timeout=300)
            if r.returncode != 0:
                log_q.put(f"ERREUR clone: {r.stderr.decode()[:200]}")
                log_q.put("DONE:0")
                return
            action = "cloned"
        log_q.put(f"Repo {action}. Scan des fichiers...")

        # Count files first
        all_files = []
        for root, dirs, files in os.walk(repo_dir):
            dirs[:] = [d for d in dirs if d not in {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".tox", "eggs"}]
            for fname in files:
                if not any(fname.endswith(ext) for ext in [".png", ".jpg", ".gif", ".ico", ".woff", ".ttf", ".pdf", ".zip", ".tar", ".gz", ".bin", ".pt", ".onnx", ".mo", ".po", ".pyc"]):
                    all_files.append(os.path.join(root, fname))
        log_q.put(f"{len(all_files)} fichiers a indexer")

        indexed = 0
        points = []
        for i, fpath in enumerate(all_files):
            rel = os.path.relpath(fpath, repo_dir)
            try:
                text = open(fpath, "r", errors="ignore").read()
                if len(text) > 50000:
                    text = text[:50000]
            except:
                continue

            is_code = any(fpath.endswith(ext) for ext in [".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".cpp", ".h", ".rb", ".php", ".cs", ".swift", ".kt", ".sh", ".sql", ".html", ".css"])
            chunks = chunk_code(text, fname=rel) if is_code else chunk_text(text, prefix=f"[{rel}] ")

            for chunk in chunks:
                emb = encoder.encode(chunk)
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb,
                    payload={"text": chunk[:2000], "file": rel, "repo": repo_name, "type": "code" if is_code else "doc"}
                ))
                indexed += 1

            if (i + 1) % 20 == 0 or (i + 1) == len(all_files):
                log_q.put(f"[{i+1}/{len(all_files)}] {rel} ({indexed} chunks)")

        # Store in Qdrant
        if points and memory_store:
            log_q.put(f"Stockage de {len(points)} chunks dans Qdrant...")
            collection = f"repo_{repo_name.lower().replace('-', '_')}"
            memory_store._ensure_collection(collection)
            batch_size = 100
            for i in range(0, len(points), batch_size):
                memory_store.client.upsert(collection, points[i:i+batch_size])
            log_q.put(f"Collection repo_{repo_name} creee ({len(points)} vecteurs)")

        log_q.put(f"DONE:{indexed}")

    async def stream_logs():
        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(None, do_index)
        while True:
            try:
                msg = log_q.get_nowait()
                yield f"data: {json.dumps({'log': msg})}\n\n"
                if msg.startswith("DONE:"):
                    count = int(msg.split(":")[1])
                    yield f"data: {json.dumps({'done': True, 'repo': repo_name, 'chunks_indexed': count})}\n\n"
                    return
            except queue.Empty:
                await asyncio.sleep(0.3)
                if task.done():
                    exc = task.exception()
                    if exc:
                        yield f"data: {json.dumps({'log': f'ERREUR: {exc}', 'done': True, 'error': str(exc)})}\n\n"
                    return

    return StreamingResponse(stream_logs(), media_type="text/event-stream")


@app.get("/api/repos")
async def list_repos():
    """List indexed repos."""
    repos = []
    if os.path.exists(REPOS_DIR):
        for d in os.listdir(REPOS_DIR):
            dp = os.path.join(REPOS_DIR, d)
            if os.path.isdir(dp) and os.path.exists(os.path.join(dp, ".git")):
                repos.append({"name": d, "path": dp})
    return {"repos": repos}


@app.post("/api/search")
async def search(req: Request):
    """Search code and memories using TriVox3."""
    body = await req.json()
    query = body.get("query", "").strip()
    repo = body.get("repo")
    top_k = body.get("top_k", 10)

    if not query:
        raise HTTPException(400, "Missing query")
    if not encoder:
        raise HTTPException(503, "TriVox encoder not loaded")

    emb = encoder.encode(query)
    results = []

    if memory_store:
        # Search ONLY in repo_* collections (not mem_* which pollute code results)
        try:
            collections = memory_store.client.get_collections().collections
            for col in collections:
                name = col.name
                if not name.startswith("repo_"):
                    continue
                if repo and f"repo_{repo.lower().replace('-','_')}" != name:
                    continue
                try:
                    hits = memory_store.client.query_points(
                        collection_name=name,
                        query=emb,
                        limit=top_k,
                    ).points
                    for h in hits:
                        results.append({
                            "text": h.payload.get("text", ""),
                            "file": h.payload.get("file", ""),
                            "repo": h.payload.get("repo", name),
                            "score": h.score,
                            "type": h.payload.get("type", "unknown"),
                        })
                except:
                    pass
        except Exception as e:
            log.warning(f"Search error: {e}")

    # Deduplicate by text content (keep highest score)
    seen = {}
    for r in results:
        key = r["text"][:200]
        if key not in seen or r["score"] > seen[key]["score"]:
            seen[key] = r
    results = list(seen.values())

    # Sort: code results get a boost over docs
    for r in results:
        r["_sort"] = r["score"] + (0.05 if r.get("type") == "code" else 0)
    results.sort(key=lambda x: x["_sort"], reverse=True)
    for r in results:
        del r["_sort"]

    return {"results": results[:top_k], "query": query}


# ─── Memory Management ───

@app.get("/api/memory/stats")
async def memory_stats():
    """Get memory statistics."""
    if not memory_store:
        return {"status": "disabled", "collections": []}
    try:
        collections = memory_store.client.get_collections().collections
        stats = []
        for col in collections:
            info = memory_store.client.get_collection(col.name)
            stats.append({
                "name": col.name,
                "points": info.points_count,
                "vectors": getattr(info, "vectors_count", info.points_count),
            })
        return {"status": "active", "collections": stats}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ─── System Info ───

@app.get("/api/status")
async def status():
    """System status."""
    return {
        "encoder": "loaded" if encoder else "disabled",
        "memory": "active" if memory_store else "disabled",
        "ollama": OLLAMA_URL,
        "embed_dim": EMBED_DIM,
    }


# ─── Static files (Chat UI) ───
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
