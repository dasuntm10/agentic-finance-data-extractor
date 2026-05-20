"""Google embedding-001 wrapper with on-disk content-hash cache.

Cache layout: one JSONL file per company under outputs/<company>/_cache/embeddings.jsonl
where each line is {"sha": <sha256>, "text": <input>, "vector": [...]}.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

from afde.config import google_api_key

log = logging.getLogger(__name__)

EMBED_MODEL = "models/embedding-001"
EMBED_DIM = 768
TASK_TYPE = "SEMANTIC_SIMILARITY"


class EmbedderUnavailable(RuntimeError):
    pass


def _sha(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


class Embedder:
    def __init__(self, cache_dir: Path | str | None = None, api_key: str | None = None) -> None:
        self.api_key = api_key or google_api_key()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cache: dict[str, list[float]] = {}
        self._cache_loaded = False
        self._client = None

    def _ensure_loaded(self) -> None:
        if self._cache_loaded or self.cache_dir is None:
            self._cache_loaded = True
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / "embeddings.jsonl"
        if cache_file.exists():
            for line in cache_file.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                    self._cache[rec["sha"]] = rec["vector"]
                except Exception:
                    continue
        self._cache_loaded = True

    def _persist(self, sha: str, text: str, vector: list[float]) -> None:
        if self.cache_dir is None:
            return
        cache_file = self.cache_dir / "embeddings.jsonl"
        with cache_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"sha": sha, "text": text[:200], "vector": vector}) + "\n")

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise EmbedderUnavailable("GOOGLE_API_KEY not set.")
            try:
                import google.generativeai as genai
            except ImportError as e:
                raise EmbedderUnavailable("google-generativeai not installed; `poetry install`.") from e
            genai.configure(api_key=self.api_key)
            self._client = genai
        return self._client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
    def _embed_one(self, text: str) -> list[float]:
        genai = self._get_client()
        resp = genai.embed_content(model=EMBED_MODEL, content=text, task_type=TASK_TYPE)
        return list(resp["embedding"])

    def embed(self, text: str) -> list[float]:
        self._ensure_loaded()
        sha = _sha(text)
        if sha in self._cache:
            return self._cache[sha]
        vec = self._embed_one(text)
        self._cache[sha] = vec
        self._persist(sha, text, vec)
        return vec

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def cosine(a: list[float], b: list[float]) -> float:
    s = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return s / (na * nb)


# ---------------------------------------------------------------------------
# Offline fallback — deterministic hashing "embedding" so the canonical mapper
# can still find an exact-match synonym without network access.
# ---------------------------------------------------------------------------


class OfflineEmbedder:
    """Cheap fallback: hashes the lowercased input into a deterministic unit vector.

    Not semantically useful — synonyms will not match. Real --offline profile
    swaps this for BGE-small via sentence-transformers.
    """

    def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.strip().lower().encode("utf-8")).digest()
        vec = [b / 255.0 - 0.5 for b in h] * (EMBED_DIM // 32 + 1)
        vec = vec[:EMBED_DIM]
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm else vec

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def get_embedder(cache_dir: Path | str | None = None) -> Embedder | OfflineEmbedder:
    from afde.config import is_offline

    if is_offline() or not google_api_key():
        return OfflineEmbedder()
    return Embedder(cache_dir=cache_dir)
