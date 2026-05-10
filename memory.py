"""Memory store using Qdrant — project-isolated, with time-decay."""
import uuid
import time
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from config import QDRANT_HOST, QDRANT_PORT, EMBED_DIM, MIN_SCORE, MAX_MEMORIES_INJECT, MAX_MEMORIES_PER_COLLECTION
from encoder import Encoder

log = logging.getLogger("memory")


class MemoryStore:
    def __init__(self, encoder: Encoder):
        self.encoder = encoder
        self.embed_dim = encoder.embed_dim
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10)
        self._ensure_collection("mem_general")
        log.info("Memory store connected to Qdrant")

    def _ensure_collection(self, name: str):
        try:
            self.client.get_collection(name)
        except Exception:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=self.embed_dim, distance=Distance.COSINE),
            )
            log.info(f"Created collection: {name}")

    def store(self, text: str, project: str = None):
        collection = f"mem_{project.lower().replace(' ', '_')}" if project else "mem_general"
        self._ensure_collection(collection)
        emb = self.encoder.encode(text)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={"text": text[:2000], "timestamp": time.time()},
        )
        self.client.upsert(collection, [point])

    def recall(self, query: str, project: str = None, top_k: int = MAX_MEMORIES_INJECT) -> list[dict]:
        emb = self.encoder.encode(query)
        results = []
        collections_to_search = ["mem_general"]
        if project:
            collections_to_search.append(f"mem_{project.lower().replace(' ', '_')}")

        for col in collections_to_search:
            try:
                hits = self.client.query_points(
                    collection_name=col, query=emb, limit=top_k,
                ).points
                for h in hits:
                    if h.score >= MIN_SCORE:
                        results.append({
                            "text": h.payload.get("text", ""),
                            "score": h.score,
                            "collection": col,
                        })
            except Exception:
                pass

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
