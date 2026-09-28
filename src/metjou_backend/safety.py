"""Spots questions that need a phone call first, not a quote.

Why: the assistant must never be the thing someone reads while they should
be calling 112 or 113. The app shows these notices above the answers, in the
user's language, so the API only returns the kind and the number.
"""

import re
from dataclasses import dataclass
from enum import StrEnum


class NoticeKind(StrEnum):
    DANGER = "danger"
    SUICIDE = "suicide"


@dataclass(frozen=True)
class Notice:
    kind: NoticeKind
    number: str


_NOTICES = {
    NoticeKind.DANGER: Notice(NoticeKind.DANGER, "112"),
    NoticeKind.SUICIDE: Notice(NoticeKind.SUICIDE, "113"),
}

# Phrases in the app's five languages (nl, en, fr, es, ar). Kept short and
# specific: a false alarm costs one banner, a miss costs much more, but a
# banner on every question would be ignored.
_PHRASES = {
    NoticeKind.SUICIDE: [
        # nl
        "zelfmoord",
        "zelfdoding",
        "dood willen",
        "wil dood",
        "niet meer leven",
        "einde aan mijn leven",
        "mezelf van kant",
        "mezelf iets aandoen",
        # en
        "suicide",
        "suicidal",
        "kill myself",
        "end my life",
        "want to die",
        "hurt myself",
        # fr
        "suicider",
        "me tuer",
        "envie de mourir",
        "veux mourir",
        "en finir",
        # es
        "suicidio",
        "suicidarme",
        "matarme",
        "quiero morir",
        "quitarme la vida",
        # ar
        "انتحار",
        "أنتحر",
        "انتحر",
        "اقتل نفسي",
        "أقتل نفسي",
        "أريد أن أموت",
    ],
    NoticeKind.DANGER: [
        # nl
        "volgt me",
        "volgt mij",
        "achtervolgt",
        "word gevolgd",
        "in gevaar",
        "bedreigt me",
        "bedreigt mij",
        "slaat me",
        "slaat mij",
        "staat voor de deur",
        "wil me vermoorden",
        "met een mes",
        "een wapen",
        "ik bloed",
        # en
        "following me",
        "in danger",
        "threatening me",
        "hitting me",
        "outside my door",
        "going to kill me",
        "has a knife",
        "has a gun",
        "i am bleeding",
        "i'm bleeding",
        # fr
        "me suit",
        "en danger",
        "me menace",
        "me frappe",
        "devant ma porte",
        "un couteau",
        # es
        "me sigue",
        "en peligro",
        "me amenaza",
        "me pega",
        "un cuchillo",
        "una pistola",
        # ar
        "يلاحقني",
        "يتبعني",
        "في خطر",
        "يهددني",
        "يضربني",
        "سكين",
        "سلاح",
    ],
}


# Arabic attaches the article and some prepositions to the word ("الانتحار").
_ARABIC_PREFIX = r"(?:وال|بال|لل|ال|و|ب|ل)?"


def _pattern(phrases: list[str]) -> re.Pattern[str]:
    # Word boundaries so "mes" in "messen" or "arme" in "armée" never match.
    parts = [
        _ARABIC_PREFIX + re.escape(p) if re.match(r"[\u0600-\u06ff]", p) else re.escape(p)
        for p in phrases
    ]
    return re.compile(r"(?<!\w)(?:" + "|".join(parts) + r")(?!\w)")


_PATTERNS = {kind: _pattern(phrases) for kind, phrases in _PHRASES.items()}


def notices_for(question: str) -> list[Notice]:
    """Notices to show before any answer, most urgent first.

    Args:
        question: The user's question, any of the app's languages.

    Returns:
        Zero, one or two notices; suicide comes first because 113 also
        dispatches help in an emergency, and the reverse is not true.
    """
    text = " ".join(question.lower().replace("’", "'").split())
    return [
        _NOTICES[kind]
        for kind in (NoticeKind.SUICIDE, NoticeKind.DANGER)
        if _PATTERNS[kind].search(text)
    ]
