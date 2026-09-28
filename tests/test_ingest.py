import httpx
import pytest

from metjou_backend.ingest import Fetcher, Page, Source, content_hash, extract_page, select_urls

SITEMAP_URLS = [
    "https://s.nl/gebeurtenissen/stalking/a/",
    "https://s.nl/gebeurtenissen/verkeersongeval/b/",
    "https://s.nl/gebeurtenissen/stalking/c/",
    "https://s.nl/gebeurtenissen/stalking/zelfmoordmethodes/",
]


def test_select_urls_filters_sitemap() -> None:
    source = Source("S", include=["/stalking/"], exclude=["methode"])
    assert select_urls(source, SITEMAP_URLS) == [
        "https://s.nl/gebeurtenissen/stalking/a/",
        "https://s.nl/gebeurtenissen/stalking/c/",
    ]


def test_select_urls_puts_explicit_first_and_caps() -> None:
    source = Source("S", urls=["https://s.nl/"], include=["/stalking/"], max_pages=2)
    assert select_urls(source, SITEMAP_URLS) == [
        "https://s.nl/",
        "https://s.nl/gebeurtenissen/stalking/a/",
    ]


def test_select_urls_never_drops_explicit_urls() -> None:
    source = Source("S", urls=["https://s.nl/1", "https://s.nl/2"], max_pages=1)
    assert select_urls(source, []) == ["https://s.nl/1", "https://s.nl/2"]


HTML = """<html><head><title>Stalking | Slachtofferhulp</title></head><body>
<nav><a href="/">Home</a><a href="/contact">Contact</a></nav>
<main><article><h1>Wat is stalking?</h1>
<p>Stalking is het stelselmatig lastigvallen van iemand. Denk aan steeds bellen,
berichten sturen, volgen of opwachten bij huis of werk.</p>
<p>Word je gestalkt? Houd een logboek bij van alles wat er gebeurt en bewaar berichten.
Je kunt aangifte doen bij de politie en Slachtofferhulp Nederland helpt je gratis.</p>
</article></main><footer>Copyright 2026</footer></body></html>"""


def test_extract_page_keeps_article_text() -> None:
    page = extract_page(HTML, "https://s.nl/stalking")
    assert page is not None
    assert page.title == "Wat is stalking?"
    assert "logboek" in page.text
    assert "Copyright" not in page.text


def test_extract_page_returns_none_without_text() -> None:
    assert extract_page("<html><body></body></html>", "https://s.nl/") is None


def test_content_hash_changes_with_text() -> None:
    assert content_hash(Page("t", "a")) != content_hash(Page("t", "b"))


def fetcher(robots: httpx.Response) -> Fetcher:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return robots
        return httpx.Response(200, text="ok")

    return Fetcher(httpx.Client(transport=httpx.MockTransport(handler)), delay=0)


@pytest.mark.parametrize(
    ("robots", "allowed"),
    [
        (httpx.Response(404), True),
        (httpx.Response(200, text="User-agent: *\nDisallow: /"), False),
        (httpx.Response(200, text="User-agent: *\nDisallow: /admin/"), True),
        (httpx.Response(503), False),
    ],
)
def test_fetcher_respects_robots(robots: httpx.Response, allowed: bool) -> None:
    result = fetcher(robots).get("https://s.nl/page")
    assert (result == "ok") is allowed
