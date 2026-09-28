import pytest

from metjou_backend.chunking import chunk_text


def words(n: int, word: str = "woord") -> str:
    return " ".join([word] * n)


def test_returns_nothing_for_blank_text() -> None:
    assert chunk_text("\n  \n") == []


def test_keeps_short_text_as_one_passage() -> None:
    assert chunk_text("Bel 112 bij direct gevaar.") == ["Bel 112 bij direct gevaar."]


def test_groups_paragraphs_until_target() -> None:
    text = "\n".join(words(50) for _ in range(5))
    passages = chunk_text(text, target_words=100, max_words=200)
    assert [len(p.split()) for p in passages] == [100, 100, 50]


def test_never_exceeds_max_words() -> None:
    text = "\n".join([words(90), words(90), words(90)])
    for passage in chunk_text(text, target_words=150, max_words=160):
        assert len(passage.split()) <= 160


def test_splits_long_paragraph_on_sentences() -> None:
    sentence = words(30) + "."
    passages = chunk_text(" ".join([sentence] * 10), target_words=50, max_words=70)
    assert all(p.endswith(".") for p in passages)
    assert all(len(p.split()) <= 70 for p in passages)


@pytest.mark.parametrize("tail", [5, 10])
def test_merges_short_tail_into_previous(tail: int) -> None:
    text = "\n".join([words(100), words(tail)])
    assert len(chunk_text(text, target_words=100, max_words=200)) == 1


def test_normalises_whitespace() -> None:
    assert chunk_text("  Bel   113\t nu  ") == ["Bel 113 nu"]
