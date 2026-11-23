"""Local RAG over the markdown knowledge base.

The embedding model and the in-memory vector index are loaded lazily on the
first search. An empty (or missing) knowledge folder never loads a model.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_CACHE_DIR = "/app/models_cache"
CHUNK_WORDS = 200
MIN_CHUNK_WORDS = 15
DEFAULT_MIN_SCORE = 0.65


def _env_min_score() -> float:
    """Read RAG_MIN_SCORE, falling back to the default when it is invalid."""
    try:
        return float(os.getenv("RAG_MIN_SCORE", DEFAULT_MIN_SCORE))
    except ValueError:
        return DEFAULT_MIN_SCORE


def _normalize(vector: Any) -> np.ndarray:
    """L2-normalize an embedding so dot product is cosine similarity."""
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm else arr


def _title(text: str, fallback: str) -> str:
    """Use the first ``# `` heading as the title, or the filename stem."""
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


def _sections(text: str) -> List[str]:
    """Split markdown at ``#``/``##`` headings, keeping the heading in its section."""
    sections: List[str] = []
    current: List[str] = []
    for line in text.splitlines():
        if re.match(r"^#{1,2}\s+", line) and any(part.strip() for part in current):
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if any(part.strip() for part in current):
        sections.append("\n".join(current).strip())
    return [section for section in sections if section.strip()]


def chunk_markdown(
    text: str,
    source: str,
    max_words: int = CHUNK_WORDS,
    min_words: int = MIN_CHUNK_WORDS,
) -> List[Dict[str, str]]:
    """Split markdown on ``#``/``##`` headings into roughly ``max_words`` words.

    Blank-line-separated blocks are kept whole, so a chunk never begins inside
    a sentence or a list item. Blocks are grouped up to ``max_words``; a block
    that is too large on its own stays intact. Tiny chunks are dropped.
    """
    max_words = max(1, int(max_words))
    min_words = max(1, int(min_words))
    chunks: List[Dict[str, str]] = []

    for section in _sections(text):
        lines = section.splitlines()
        if lines and re.match(r"^#{1,2}\s+", lines[0]):
            heading = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
        else:
            heading = ""
            body = section
        current: List[str] = []
        count = len(heading.split()) if heading else 0

        for block in re.split(r"\n\s*\n", body):
            block = block.strip()
            if not block:
                continue
            block_words = len(block.split())
            if current and count + block_words > max_words:
                chunk_text = _render_chunk(heading, current)
                if len(chunk_text.split()) >= min_words:
                    chunks.append({"source": source, "text": chunk_text})
                current, count = [], len(heading.split()) if heading else 0
            current.append(block)
            count += block_words

        if current:
            chunk_text = _render_chunk(heading, current)
            if len(chunk_text.split()) >= min_words:
                chunks.append({"source": source, "text": chunk_text})

    return chunks


def _render_chunk(heading: str, blocks: List[str]) -> str:
    body = "\n\n".join(blocks)
    return f"{heading}\n\n{body}" if heading else body


class RagEngine:
    """Embed local markdown notes once and answer queries with dot-product search."""

    MODEL_NAME = MODEL_NAME
    CHUNK_WORDS = CHUNK_WORDS
    MIN_CHUNK_WORDS = MIN_CHUNK_WORDS
    DEFAULT_CACHE_DIR = DEFAULT_CACHE_DIR
    DEFAULT_MIN_SCORE = DEFAULT_MIN_SCORE

    def __init__(
        self,
        knowledge_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> None:
        self.knowledge_dir = (
            Path(knowledge_dir)
            if knowledge_dir is not None
            else Path(__file__).resolve().parents[2] / "data" / "knowledge"
        )
        self.cache_dir = cache_dir or os.getenv("FASTEMBED_CACHE", DEFAULT_CACHE_DIR)
        self.min_score = float(min_score) if min_score is not None else _env_min_score()
        self._model: Any = None
        self.chunks: List[Dict[str, str]] = []
        self.embed_texts: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._loaded = False

    def _load_model(self) -> Any:
        """Import and instantiate fastembed only when there is something to embed."""
        from fastembed import TextEmbedding

        return TextEmbedding(model_name=self.MODEL_NAME, cache_dir=self.cache_dir)

    def load_chunks(self) -> List[Dict[str, str]]:
        """Read non-blank ``*.md`` files; keep plain chunks and title-prefixed embed text."""
        self.chunks = []
        self.embed_texts = []
        if not self.knowledge_dir.is_dir():
            return self.chunks
        for path in sorted(self.knowledge_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                continue
            title = _title(text, path.stem)
            for chunk in chunk_markdown(text, path.name, self.CHUNK_WORDS):
                self.chunks.append(chunk)  # returned to the LLM unchanged
                self.embed_texts.append(f"{title}\n\n{chunk['text']}")
        return self.chunks

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        loaded = self.load_chunks()
        if not self.chunks:
            self.chunks = loaded or []
        if not self.embed_texts:
            self.embed_texts = [chunk["text"] for chunk in self.chunks]
        if not self.chunks:
            self._loaded = True
            return
        self._model = self._load_model()
        vectors = [
            _normalize(vector)
            for vector in self._model.embed(self.embed_texts)
        ]
        self._embeddings = np.vstack(vectors) if vectors else None
        self._loaded = True

    def search(
        self, query: str, k: int = 3, min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Return up to ``k`` chunks clearing the score threshold.

        The default threshold is 0.65, or ``RAG_MIN_SCORE`` when configured.
        """
        if k <= 0:
            return []
        threshold = self.min_score if min_score is None else float(min_score)
        self._ensure_loaded()
        if not self.chunks or self._embeddings is None:
            return []

        query_vector = _normalize(next(iter(self._model.query_embed(query))))
        scores = self._embeddings @ query_vector
        ranked = sorted(enumerate(scores), key=lambda pair: float(pair[1]), reverse=True)

        results: List[Dict[str, Any]] = []
        for index, score in ranked:
            value = float(score)
            if value < threshold:
                break
            results.append({
                "source": self.chunks[index]["source"],
                "text": self.chunks[index]["text"],
                "score": value,
            })
            if len(results) >= k:
                break
        return results


rag = RagEngine()
