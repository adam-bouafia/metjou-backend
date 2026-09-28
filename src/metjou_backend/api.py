"""HTTP API for the MetJou app.

Privacy: questions are never stored or logged. Uvicorn's access log only has
the path, and nothing here writes the request body anywhere.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field

from metjou_backend import db
from metjou_backend.config import Settings, get_settings
from metjou_backend.embeddings import Embedder
from metjou_backend.safety import notices_for

Search = Callable[[str, int], list[db.Hit]]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class NoticeOut(BaseModel):
    kind: str
    number: str


class AnswerOut(BaseModel):
    quote: str
    title: str
    organisation: str
    url: str


class AskResponse(BaseModel):
    notices: list[NoticeOut]
    answers: list[AnswerOut]
    # False when even the best passage is a weak match.
    confident: bool


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    # The vector type must exist before connections in the pool register it.
    db.connect(settings.database_url).close()
    with ConnectionPool(
        settings.database_url, min_size=1, max_size=4, configure=register_vector
    ) as pool:
        embedder = Embedder(settings)

        def search(question: str, limit: int) -> list[db.Hit]:
            vector = embedder.embed_query(question)
            with pool.connection() as conn:
                return db.search(conn, question, vector, limit)

        app.state.search = search
        yield


def get_search(request: Request) -> Search:
    search: Search = request.app.state.search
    return search


app = FastAPI(title="MetJou backend", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/ask")
def ask(
    body: AskRequest,
    search: Annotated[Search, Depends(get_search)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AskResponse:
    """Quotes from official sources that answer the question, plus notices.

    The answers are passages as written by the source, never generated text,
    so nothing can be made up.
    """
    question = " ".join(body.question.split())
    hits = search(question, settings.answer_count)
    return AskResponse(
        notices=[NoticeOut(kind=n.kind, number=n.number) for n in notices_for(question)],
        answers=[
            AnswerOut(quote=h.text, title=h.title, organisation=h.organisation, url=h.url)
            for h in hits
        ],
        confident=bool(hits) and hits[0].similarity >= settings.min_similarity,
    )
