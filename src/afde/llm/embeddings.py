"""Local sentence-transformers embedder (BGE-small-en-v1.5) with on-disk cache.

Cache layout: one JSONL file per company under ``outputs/<company>/_cache/embeddings.jsonl``
where each line is ``{"sha": <sha256>, "text": <input>, "vector": [...]}``. The
cache makes re-runs deterministic and lets the canonical mapper skip embedding
work entirely on the second pass.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path

log = logging.getLogger(__name__)

EMBED_MODEL = os.environ.get("AFDE_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = 384


class EmbedderUnavailable(RuntimeError):
    pass


def _sha(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


class Embedder:
    """sentence-transformers wrapper with a content-hash JSONL cache."""

    def __init__(self, cache_dir: Path | str | None = None, model_name: str | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model_name = model_name or EMBED_MODEL
        self._cache: dict[str, list[float]] = {}
        self._cache_loaded = False
        self._model = None

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

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            except ImportError as e:
                raise EmbedderUnavailable(
                    "sentence-transformers not installed; `poetry install`."
                ) from e
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed_one(self, text: str) -> list[float]:
        model = self._get_model()
        vec = model.encode([text], normalize_embeddings=True)[0]
        return [float(x) for x in vec]

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
# Deterministic fallback — used only when sentence-transformers is unavailable
# (e.g. the smoke test environment). Hashes the lowercased input into a
# pseudo-random unit vector. Not semantically useful; the canonical mapper's
# rule-based path still works because exact synonyms hit the rule layer first.
# ---------------------------------------------------------------------------


class FallbackEmbedder:
    def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.strip().lower().encode("utf-8")).digest()
        vec = [b / 255.0 - 0.5 for b in h] * (EMBED_DIM // 32 + 1)
        vec = vec[:EMBED_DIM]
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm else vec

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def get_embedder(cache_dir: Path | str | None = None) -> Embedder | FallbackEmbedder:
    try:
        import sentence_transformers  # noqa: F401, PLC0415

        return Embedder(cache_dir=cache_dir)
    except ImportError:
        log.warning("sentence-transformers not installed; using FallbackEmbedder (rule-match only).")
        return FallbackEmbedder()
