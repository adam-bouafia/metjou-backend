"""Postgres storage and hybrid search (pgvector plus Dutch full-text)."""

from dataclasses import dataclass
from importlib.resources import files

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from metjou_backend.embeddings import Vector

# Reciprocal rank fusion constant, the usual value from the RRF paper.
_RRF_K = 60
_CANDIDATES = 30

# Semantic and keyword candidates are ranked separately and fused with RRF.
# The keyword query ORs the words, because a question rarely contains every
# word of the passage that answers it.
_SEARCH_SQL = """
with sem as (
    select id, 1 - (embedding <=> %(vec)s) as similarity,
           row_number() over (order by embedding <=> %(vec)s) as rank
    from passages
    order by embedding <=> %(vec)s
    limit %(n)s
),
q as (
    select to_tsquery('dutch', replace(plainto_tsquery('dutch', %(text)s)::text, '&', '|')) as query
),
lex as (
    select p.id, row_number() over (order by ts_rank_cd(p.tsv, q.query) desc) as rank
    from passages p, q
    where p.tsv @@ q.query
    order by ts_rank_cd(p.tsv, q.query) desc
    limit %(n)s
)
select p.document_id, p.text, d.url, d.title, d.organisation,
       coalesce(1.0 / (%(k)s + sem.rank), 0) + coalesce(1.0 / (%(k)s + lex.rank), 0) as score,
       1 - (p.embedding <=> %(vec)s) as similarity
from sem
full outer join lex using (id)
join passages p on p.id = coalesce(sem.id, lex.id)
join documents d on d.id = p.document_id
order by score desc
"""


@dataclass(frozen=True)
class Hit:
    text: str
    url: str
    title: str
    organisation: str
    score: float
    similarity: float


def connect(url: str, dim: int) -> psycopg.Connection:
    conn = psycopg.connect(url, autocommit=True)
    init_schema(conn, dim)
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection, dim: int) -> None:
    sql = files("metjou_backend").joinpath("schema.sql").read_text().format(dim=int(dim))
    conn.execute(sql.encode())


def document_hash(conn: psycopg.Connection, url: str) -> str | None:
    row = conn.execute("select content_hash from documents where url = %s", (url,)).fetchone()
    return row[0] if row else None


def replace_document(
    conn: psycopg.Connection,
    *,
    url: str,
    title: str,
    organisation: str,
    content_hash: str,
    passages: list[str],
    vectors: list[Vector],
) -> None:
    """Stores a page and its passages, replacing an earlier version."""
    with conn.transaction():
        conn.execute("delete from documents where url = %s", (url,))
        row = conn.execute(
            "insert into documents (url, title, organisation, content_hash) "
            "values (%s, %s, %s, %s) returning id",
            (url, title, organisation, content_hash),
        ).fetchone()
        assert row is not None
        with conn.cursor() as cur:
            cur.executemany(
                "insert into passages (document_id, position, text, embedding) "
                "values (%s, %s, %s, %s::vector)",
                [(row[0], i, p, v) for i, (p, v) in enumerate(zip(passages, vectors, strict=True))],
            )


def delete_documents_except(conn: psycopg.Connection, keep: set[str]) -> int:
    """Removes pages that are no longer in the source list."""
    cur = conn.execute("delete from documents where not (url = any(%s))", (list(keep),))
    return cur.rowcount


def search(conn: psycopg.Connection, question: str, vector: Vector, limit: int) -> list[Hit]:
    """Best passages for a question, at most one per page.

    Why one per page: three quotes from three pages give the user more to act
    on than three neighbouring paragraphs of one page.
    """
    rows = conn.execute(
        _SEARCH_SQL,
        {
            "vec": np.asarray(vector, dtype=np.float32),
            "text": question,
            "n": _CANDIDATES,
            "k": _RRF_K,
        },
    ).fetchall()
    hits: list[Hit] = []
    seen: set[int] = set()
    for document_id, text, url, title, organisation, score, similarity in rows:
        if document_id in seen:
            continue
        seen.add(document_id)
        hits.append(Hit(text, url, title, organisation, float(score), float(similarity)))
        if len(hits) == limit:
            break
    return hits
