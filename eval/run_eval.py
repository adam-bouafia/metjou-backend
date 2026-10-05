"""Retrieval eval: hit@1, hit@3 and MRR over eval/questions.yaml.

Usage: uv run python eval/run_eval.py [questions.yaml]

Runs hybrid search and vector-only search (an empty keyword query matches
nothing) against the database in METJOU_DATABASE_URL.
"""

import statistics
import sys
from pathlib import Path

import yaml

from metjou_backend import db
from metjou_backend.config import get_settings
from metjou_backend.embeddings import Embedder

K = 10


def main(path: Path) -> None:
    data = yaml.safe_load(path.read_text())
    settings = get_settings()
    conn = db.connect(settings.database_url, settings.embedding_dim)
    embedder = Embedder(settings)
    questions = data["questions"]
    vectors = [embedder.embed_query(item["q"]) for item in questions]

    for mode in ("hybrid", "vector"):
        ranks: list[int | None] = []
        best: list[float] = []
        misses: list[str] = []
        for item, vector in zip(questions, vectors, strict=True):
            text = item["q"] if mode == "hybrid" else ""
            hits = db.search(conn, text, vector, K)
            best.append(hits[0].similarity if hits else 0.0)
            rank = next(
                (i for i, h in enumerate(hits, 1) if any(e in h.url for e in item["expect"])),
                None,
            )
            ranks.append(rank)
            if rank is None or rank > 3:
                top = hits[0].url if hits else "-"
                misses.append(f"  rank {rank}: {item['q']}  (top: {top})")
        n = len(ranks)
        hit1 = sum(r == 1 for r in ranks) / n
        hit3 = sum(r is not None and r <= 3 for r in ranks) / n
        mrr = sum(1 / r for r in ranks if r) / n
        print(f"{mode:7} hit@1 {hit1:.2f}  hit@3 {hit3:.2f}  MRR {mrr:.2f}  (n={n})")
        if mode == "hybrid":
            print("\n".join(misses))
            print(
                f"best similarity, on topic:  min {min(best):.3f}  median "
                f"{statistics.median(best):.3f}"
            )

    off = []
    for q in data.get("off_topic", []):
        hits = db.search(conn, q, embedder.embed_query(q), 1)
        off.append(hits[0].similarity if hits else 0.0)
    if off:
        print(
            f"best similarity, off topic: max {max(off):.3f}  median {statistics.median(off):.3f}"
        )


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "eval/questions.yaml"))
