from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from metjou_backend.api import app, get_search
from metjou_backend.db import Hit


def hit(url: str, similarity: float) -> Hit:
    return Hit("Bel Veilig Thuis: 0800-2000.", url, "Titel", "Veilig Thuis", 0.03, similarity)


@pytest.fixture
def client() -> Iterator[TestClient]:
    # No lifespan: the search dependency is replaced, so no database or model.
    yield TestClient(app)
    app.dependency_overrides.clear()


def use_hits(hits: list[Hit]) -> list[tuple[str, int]]:
    calls: list[tuple[str, int]] = []

    def search(question: str, limit: int) -> list[Hit]:
        calls.append((question, limit))
        return hits

    app.dependency_overrides[get_search] = lambda: search
    return calls


def test_healthz(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}


def test_ask_returns_quotes_with_sources(client: TestClient) -> None:
    calls = use_hits([hit("https://veiligthuis.nl/a", 0.9)])
    body = client.post("/v1/ask", json={"question": "  mijn   partner slaat me "}).json()
    assert calls == [("mijn partner slaat me", 3)]
    assert body["answers"] == [
        {
            "quote": "Bel Veilig Thuis: 0800-2000.",
            "title": "Titel",
            "organisation": "Veilig Thuis",
            "url": "https://veiligthuis.nl/a",
        }
    ]
    assert body["confident"] is True
    assert body["notices"] == [{"kind": "danger", "number": "112"}]


def test_weak_match_is_not_confident(client: TestClient) -> None:
    use_hits([hit("https://a.nl", 0.5)])
    assert client.post("/v1/ask", json={"question": "hallo daar"}).json()["confident"] is False


def test_no_hits_is_not_confident(client: TestClient) -> None:
    use_hits([])
    body = client.post("/v1/ask", json={"question": "hallo daar"}).json()
    assert body == {"notices": [], "answers": [], "confident": False}


@pytest.mark.parametrize("question", ["", "ab", "x" * 501])
def test_rejects_bad_question_length(client: TestClient, question: str) -> None:
    use_hits([])
    assert client.post("/v1/ask", json={"question": question}).status_code == 422
