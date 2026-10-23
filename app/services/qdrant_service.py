import os
import uuid
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings

logger = logging.getLogger(__name__)

class LocalQdrantService:
    def __init__(self):
        self.storage_path = settings.QDRANT_STORAGE_PATH
        self.collection_name = settings.QDRANT_COLLECTION
        self.device = settings.get_torch_device()
        self.embed_model_name = settings.EMBEDDING_MODEL
        self._encoder = None
        self._encoder_attempted = False
        self._vector_dim = 384  # Default for all-MiniLM-L6-v2 / bge-small
        self._client = None

    @property
    def client(self) -> QdrantClient:
        """Lazy initialization of Qdrant client."""
        if self._client is None:
            if self.storage_path == ":memory:":
                self._client = QdrantClient(":memory:")
            else:
                os.makedirs(self.storage_path, exist_ok=True)
                self._client = QdrantClient(path=self.storage_path)
            self._ensure_collection()
        return self._client

    @property
    def encoder(self):
        """Lazy initialization of embedding model."""
        if self._encoder is None and not self._encoder_attempted:
            self._encoder_attempted = True
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading local SentenceTransformer model: {self.embed_model_name} on {self.device}")
                self._encoder = SentenceTransformer(self.embed_model_name, device=self.device)
                self._vector_dim = self._encoder.get_sentence_embedding_dimension()
            except Exception as e:
                logger.warning(
                    f"SentenceTransformer not available ({e}). "
                    "Make sure 'sentence-transformers' and 'torch' are installed for full ROCm embeddings."
                )
                self._encoder = None
        return self._encoder

    @property
    def vector_dim(self) -> int:
        if self.encoder is not None:
            return self._encoder.get_sentence_embedding_dimension()
        return self._vector_dim

    def _ensure_collection(self):
        """Creates the Qdrant collection if it does not already exist."""
        try:
            collections = [col.name for col in self.client.get_collections().collections]
            if self.collection_name not in collections:
                logger.info(f"Creating collection '{self.collection_name}' with vector size {self.vector_dim}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_dim,
                        distance=models.Distance.COSINE
                    )
                )
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection: {e}")

    def reset_collection(self):
        """Clears previous run chunks to maintain crisp context per query session."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except Exception:
            pass
        self._ensure_collection()

    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        """Encodes texts using SentenceTransformer or fallback deterministic hash embedding."""
        if self.encoder is not None:
            embeddings = self.encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            return embeddings.tolist()
        
        # Fallback pseudo-embedding if torch/sentence_transformers is temporarily unavailable
        logger.warning("Using fallback vector representation. Install sentence-transformers for true neural embeddings.")
        import hashlib
        vectors = []
        for text in texts:
            vec = [0.0] * self.vector_dim
            tokens = text.lower().split()
            for token in tokens:
                idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.vector_dim
                vec[idx] += 1.0
            norm = sum(x*x for x in vec) ** 0.5 or 1.0
            vectors.append([x / norm for x in vec])
        return vectors

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Takes list of dicts: [{'text': ..., 'url': ..., 'title': ...}]
        Computes local embeddings and upserts into Qdrant.
        """
        if not chunks:
            return 0

        self._ensure_collection()
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self._encode_texts(texts)

        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": chunk["text"],
                        "url": chunk.get("url", ""),
                        "title": chunk.get("title", ""),
                        "chunk_index": i
                    }
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Indexed {len(points)} chunks into Qdrant collection '{self.collection_name}'.")
        return len(points)

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Searches Qdrant using query vector, compatible with both new query_points and legacy search."""
        query_vectors = self._encode_texts([query])
        query_vector = query_vectors[0]

        try:
            # Modern Qdrant client >= 1.10.0
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=limit,
                    with_payload=True
                )
                raw_results = response.points
            else:
                # Legacy Qdrant client fallback
                raw_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_payload=True
                )
        except Exception as e:
            logger.error(f"Qdrant vector search failed: {e}")
            return []

        formatted = []
        for res in raw_results:
            payload = res.payload or {}
            formatted.append({
                "text": payload.get("text", ""),
                "url": payload.get("url", ""),
                "title": payload.get("title", ""),
                "score": float(res.score)
            })
        return formatted

# Singleton instance
qdrant_service = LocalQdrantService()
