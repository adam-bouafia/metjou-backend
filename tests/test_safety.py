import pytest

from metjou_backend.safety import NoticeKind, notices_for


def kinds(question: str) -> list[NoticeKind]:
    return [n.kind for n in notices_for(question)]


@pytest.mark.parametrize(
    "question",
    [
        "Ik denk aan zelfmoord",
        "ik wil niet meer leven",
        "I want to die",
        "Je veux mourir",
        "Quiero morir",
        "أفكر في الانتحار",
        "بدي انتحر",
    ],
)
def test_flags_suicide_in_every_language(question: str) -> None:
    assert kinds(question) == [NoticeKind.SUICIDE]


@pytest.mark.parametrize(
    "question",
    [
        "Iemand volgt me al tien minuten",
        "Mijn ex staat voor de deur",
        "Someone is following me",
        "Quelqu'un me suit",
        "Alguien me sigue",
        "شخص يلاحقني",
    ],
)
def test_flags_danger_in_every_language(question: str) -> None:
    assert kinds(question) == [NoticeKind.DANGER]


@pytest.mark.parametrize(
    "question",
    [
        "Hoe doe ik aangifte van stalking?",
        "Wat is dwingende controle?",
        "Waar vind ik messen voor de keuken?",
        "L'armée recrute",
        "رأيت السلاحف في البحر",
    ],
)
def test_no_notice_for_general_questions(question: str) -> None:
    assert kinds(question) == []


def test_suicide_comes_before_danger() -> None:
    assert kinds("Hij bedreigt me en ik wil dood") == [NoticeKind.SUICIDE, NoticeKind.DANGER]


def test_returns_emergency_numbers() -> None:
    assert [n.number for n in notices_for("I want to die, someone is following me")] == [
        "113",
        "112",
    ]


def test_matches_curly_apostrophe() -> None:
    assert kinds("I’m bleeding") == [NoticeKind.DANGER]
