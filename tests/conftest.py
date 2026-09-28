import os
from collections.abc import Iterator

import psycopg
import pytest

from metjou_backend import db


@pytest.fixture
def conn() -> Iterator[psycopg.Connection]:
    """Empty database; set METJOU_TEST_DATABASE_URL to run these tests."""
    url = os.getenv("METJOU_TEST_DATABASE_URL")
    if not url:
        pytest.skip("METJOU_TEST_DATABASE_URL not set")
    connection = db.connect(url)
    connection.execute("truncate documents cascade")
    yield connection
    connection.close()
