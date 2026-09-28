import os
from collections.abc import Sequence

import numpy as np
import pytest

from metjou_backend.config import Settings
from metjou_backend.embeddings import Embedder


class Recorder:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def __call__(self, texts: Sequence[str]) -> list[np.ndarray]:
        self.seen.extend(texts)
        return [np.ones(3) for _ in texts]


def test_adds_query_prefix() -> None:
    backend = Recorder()
    Embedder(Settings(), backend).embed_query("wat nu?")
    assert backend.seen == ["query: wat nu?"]


def test_adds_passage_prefix() -> None:
    backend = Recorder()
    vectors = Embedder(Settings(), backend).embed_passages(["a", "b"])
    assert backend.seen == ["passage: a", "passage: b"]
    assert vectors == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


@pytest.mark.skipif(not os.getenv("METJOU_TEST_MODEL"), reason="downloads the model (~120 MB)")
def test_real_model_ranks_relevant_passage_higher() -> None:
    embedder = Embedder(Settings())
    q = np.array(embedder.embed_query("Ik denk aan zelfmoord"))
    good, bad = map(
        np.array,
        embedder.embed_passages(
            ["Denk je aan zelfdoding? Bel 113, gratis en anoniem.", "Het museum is open op zondag."]
        ),
    )
    assert len(q) == Settings().embedding_dim
    assert q @ good > q @ bad
