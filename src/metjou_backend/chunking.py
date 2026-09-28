"""Split page text into passages that can be quoted on their own."""

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _words(text: str) -> int:
    return len(text.split())


def _split_long(paragraph: str, max_words: int) -> list[str]:
    """Splits a paragraph that is too long on sentence boundaries."""
    parts: list[str] = []
    current: list[str] = []
    for sentence in _SENTENCE_END.split(paragraph):
        if current and _words(" ".join([*current, sentence])) > max_words:
            parts.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        parts.append(" ".join(current))
    return parts


def chunk_text(text: str, target_words: int = 120, max_words: int = 200) -> list[str]:
    """Groups paragraphs into passages of about ``target_words`` words.

    Why: a passage is what the assistant quotes, so it must read as a whole
    thought. Paragraphs are never cut unless one alone is over ``max_words``.

    Args:
        text: Page text with one paragraph per line.
        target_words: Size at which a passage is closed.
        max_words: Hard limit; longer paragraphs are split on sentences.

    Returns:
        Passages in page order, without empty ones.
    """
    paragraphs: list[str] = []
    for line in text.splitlines():
        line = " ".join(line.split())
        if not line:
            continue
        if _words(line) > max_words:
            paragraphs.extend(_split_long(line, max_words))
        else:
            paragraphs.append(line)

    passages: list[str] = []
    current: list[str] = []
    for paragraph in paragraphs:
        if current and _words("\n".join([*current, paragraph])) > max_words:
            passages.append("\n".join(current))
            current = []
        current.append(paragraph)
        if _words("\n".join(current)) >= target_words:
            passages.append("\n".join(current))
            current = []
    if current:
        # A short tail reads better as part of the passage before it.
        if passages and _words("\n".join(current)) < target_words // 3:
            merged = passages[-1] + "\n" + "\n".join(current)
            if _words(merged) <= max_words:
                passages[-1] = merged
                return passages
        passages.append("\n".join(current))
    return passages
