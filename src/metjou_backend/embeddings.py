"""Local text embeddings (multilingual-e5 on ONNX Runtime via fastembed)."""

from collections.abc import Callable, Iterable, Sequence

import numpy as np
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

from metjou_backend.config import Settings

Vector = list[float]
Backend = Callable[[Sequence[str]], Iterable[np.ndarray]]


def _fastembed_backend(settings: Settings) -> Backend:
    """Loads the model once; fastembed does not ship e5-small, so register it."""
    known = {m["model"] for m in TextEmbedding.list_supported_models()}
    if settings.embedding_model not in known:
        TextEmbedding.add_custom_model(
            model=settings.embedding_model,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=settings.embedding_source),
            dim=settings.embedding_dim,
            model_file=settings.embedding_file,
        )
    model = TextEmbedding(settings.embedding_model, cache_dir=settings.model_cache_dir)
    return lambda texts: model.embed(list(texts))


class Embedder:
    """Turns questions and passages into normalised vectors.

    Why: e5 models are trained with "query: " and "passage: " prefixes and
    rank noticeably worse without them.
    """

    def __init__(self, settings: Settings, backend: Backend | None = None) -> None:
        self._backend = backend or _fastembed_backend(settings)

    def embed_query(self, question: str) -> Vector:
        return self._run([f"query: {question}"])[0]

    def embed_passages(self, passages: Sequence[str]) -> list[Vector]:
        return self._run([f"passage: {p}" for p in passages])

    def _run(self, texts: Sequence[str]) -> list[Vector]:
        return [np.asarray(v, dtype=np.float32).tolist() for v in self._backend(texts)]
