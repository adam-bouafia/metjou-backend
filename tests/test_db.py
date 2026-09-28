import psycopg

from metjou_backend import db

DIM = 384


def onehot(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


def store(conn: psycopg.Connection, url: str, passages: list[str], axes: list[int]) -> None:
    db.replace_document(
        conn,
        url=url,
        title=url.rsplit("/", 1)[-1],
        organisation="Test",
        content_hash=f"hash-{url}",
        passages=passages,
        vectors=[onehot(a) for a in axes],
    )


def test_replace_document_overwrites_passages(conn: psycopg.Connection) -> None:
    store(conn, "https://a.nl/x", ["oud"], [0])
    store(conn, "https://a.nl/x", ["nieuw een", "nieuw twee"], [1, 2])
    texts = [r[0] for r in conn.execute("select text from passages order by position")]
    assert texts == ["nieuw een", "nieuw twee"]
    assert db.document_hash(conn, "https://a.nl/x") == "hash-https://a.nl/x"


def test_document_hash_is_none_for_unknown_url(conn: psycopg.Connection) -> None:
    assert db.document_hash(conn, "https://a.nl/onbekend") is None


def test_delete_documents_except_keeps_listed(conn: psycopg.Connection) -> None:
    store(conn, "https://a.nl/keep", ["a"], [0])
    store(conn, "https://a.nl/drop", ["b"], [1])
    assert db.delete_documents_except(conn, {"https://a.nl/keep"}) == 1
    assert db.document_hash(conn, "https://a.nl/drop") is None


def test_search_ranks_by_vector(conn: psycopg.Connection) -> None:
    store(conn, "https://a.nl/1", ["eerste pagina"], [0])
    store(conn, "https://a.nl/2", ["tweede pagina"], [1])
    hits = db.search(conn, "xyz", onehot(1), limit=2)
    assert [h.url for h in hits] == ["https://a.nl/2", "https://a.nl/1"]
    assert hits[0].similarity > 0.99


def test_search_uses_dutch_keywords(conn: psycopg.Connection) -> None:
    # Same vector for both; only the stemmed keyword "geslagen" -> "slaan" can decide.
    store(conn, "https://a.nl/1", ["De fietsenstalling is open."], [0])
    store(conn, "https://a.nl/2", ["Word je geslagen door je partner? Bel Veilig Thuis."], [0])
    hits = db.search(conn, "mijn partner heeft mij geslagen", onehot(0), limit=2)
    assert hits[0].url == "https://a.nl/2"


def test_search_returns_one_passage_per_page(conn: psycopg.Connection) -> None:
    store(conn, "https://a.nl/1", ["een", "twee", "drie"], [0, 0, 0])
    store(conn, "https://a.nl/2", ["vier"], [1])
    hits = db.search(conn, "x", onehot(0), limit=3)
    assert [h.url for h in hits] == ["https://a.nl/1", "https://a.nl/2"]
