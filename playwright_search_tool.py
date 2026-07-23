"""playwright_search_tool.py

Playwright-driven web search tools for Open Deep Research / rdtii, used ALONGSIDE the
existing paid SerpAPI/Serper `GoogleSearchTool` and `DuckDuckGoSearchTool` (not a
replacement for either).

Two search tools are provided:
  - PlaywrightDuckDuckGoSearchTool (RECOMMENDED as primary): scrapes DuckDuckGo's
    server-rendered HTML results page. No JavaScript wait, no CAPTCHA/consent wall in
    normal use, and stable markup -- the most scrape-friendly of the two.
  - PlaywrightGoogleSearchTool (fallback): scrapes Google's JS-rendered results page.
    Broader index (useful for some non-English/regional government sites DuckDuckGo
    misses), but far more bot-detection friction -- occasional CAPTCHA/consent walls
    and markup that shifts to resist scraping.

Why either exists at all: SerpAPI/Serper cost money per query and DuckDuckGo (via the
`ddgs` library) gets rate-limited under heavy/rapid use. These instead launch a real
(headless) Chromium browser via Playwright and scrape results directly -- no paid API,
no third-party search library, no per-query cost or account quota.

They're drop-in-shaped tools: single `query` input, and a similar "numbered list of
title/link/snippet" output shape as smolagents' GoogleSearchTool. Once wired in
alongside the other search tools, the search_agent keeps using its existing tools
(VisitTool, PageUpTool, PageDownTool, FinderTool, FindNextTool, ArchiveSearchTool,
TextInspectorTool) to open, crawl, and read whatever pages these tools find -- only the
initial "search" step differs.

Requirements (not in requirements.txt by default -- add both):
    pip install playwright
    playwright install chromium

Usage:
    from playwright_search_tool import PlaywrightDuckDuckGoSearchTool, PlaywrightGoogleSearchTool
    WEB_TOOLS = [GoogleSearchTool(...), DuckDuckGoSearchTool(), PlaywrightDuckDuckGoSearchTool(), PlaywrightGoogleSearchTool(), ...]

---------------------------------------------------------------------------
Why PlaywrightGoogleSearchTool's extraction logic is NOT based on Google's CSS class
names (div.g, div.MjjYud, etc.):

Google's result-page markup (a) changes its class names frequently and without
notice, and (b) is frequently served *differently* to traffic it suspects is
automated -- a plain, unauthenticated, headless-looking browser can get a
stripped-down or reflowed results page where those old class names simply don't
exist, even though a person opening the same query in a normal browser sees full
results. Relying on class names is why an earlier version of this tool could report
"No Google results found" for a query that clearly has results.

Instead, extraction here is structural: Google's results almost always still put each
organic result's title inside an <h3>, wrapped (directly or via an ancestor) in an
<a href="..."> to the real destination. Walking up from every <h3> to its nearest
linked ancestor is far more resistant to markup/class churn than any fixed selector
list. A handful of anti-bot-detection tweaks (a realistic user agent, a normal
viewport, hiding the `navigator.webdriver` flag, and waiting for actual content
instead of a fixed sleep) also reduce how often Google serves the stripped-down page
in the first place.

DuckDuckGo's HTML endpoint below needs none of this -- it's server-rendered with no
client-side JS to fight, which is exactly why it's the recommended primary tool.
---------------------------------------------------------------------------
"""
from __future__ import annotations

import urllib.parse

from smolagents import Tool

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
)

_CONSENT_BUTTON_LABELS = ("Accept all", "I agree", "Accept", "Reject all", "Alle akzeptieren")

# Hosts that are Google's own chrome (nav bars, "Sign in", "Settings", cached-page
# links, translate links, etc.) rather than an actual search result -- filtered out
# of both the h3-based pass and the broad fallback pass below.
_IGNORED_HOST_FRAGMENTS = (
    "google.com/search", "google.com/preferences", "google.com/advanced_search",
    "google.com/intl", "accounts.google.", "support.google.", "policies.google.",
    "maps.google.", "webcache.googleusercontent.", "translate.google.",
    "/url?q=/search", "consent.google.",
)

_ANTI_DETECTION_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
window.chrome = window.chrome || { runtime: {} };
"""


class PlaywrightDuckDuckGoSearchTool(Tool):
    """Searches DuckDuckGo's plain HTML endpoint (html.duckduckgo.com/html/) with
    Playwright. This is the PREFERRED Playwright search tool for automated use, ahead
    of PlaywrightGoogleSearchTool below:

    - The page is fully server-rendered HTML with no JavaScript required, so there's
      no client-side rendering to wait for and no anti-bot-detection script needed.
    - No CAPTCHA/consent wall in normal use (unlike Google, which frequently serves
      one to automated-looking traffic).
    - Stable, simple markup (a.result__a / .result__snippet) that changes far less
      often than Google's, which is deliberately obfuscated/rotated against scrapers.

    Trade-off: DuckDuckGo's index is thinner than Google's for some non-English or
    regional government sites, which is why PlaywrightGoogleSearchTool is kept as an
    explicit fallback rather than removed.
    """

    name = "playwright_duckduckgo_search"
    description = (
        "Performs a DuckDuckGo web search for your query and returns the top results (title, "
        "link, short snippet) as a numbered list. Uses a real browser (Playwright) against "
        "DuckDuckGo's plain HTML results page -- no JavaScript rendering wait, no CAPTCHA/consent "
        "wall in normal use, and more stable markup than Google's. Use this as your PRIMARY search "
        "tool; fall back to 'playwright_google_search' only if this one comes back empty or a topic "
        "needs Google's broader index (e.g. some non-English regional government sites). Keep "
        "queries short and simple -- the plain topic name plus a country/keyword or two -- rather "
        "than long boolean or quoted-phrase chains."
    )
    inputs = {
        "query": {"type": "string", "description": "The search query to perform. Keep it short and simple."},
    }
    output_type = "string"

    def __init__(self, max_results: int = 8, headless: bool = False, timeout_ms: int = 20000):
        super().__init__()
        self.max_results = max_results
        self.headless = headless
        self.timeout_ms = timeout_ms

    def forward(self, query: str) -> str:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise ImportError(
                "PlaywrightDuckDuckGoSearchTool needs the `playwright` package and its "
                "browser binaries. Install with:\n"
                "    pip install playwright\n"
                "    playwright install chromium"
            ) from e

        query = (query or "").strip()
        if not query:
            return "Error: empty search query."

        url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=self.headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
            try:
                context = browser.new_context(
                    user_agent=_USER_AGENT,
                    locale="en-US",
                    viewport={"width": 1366, "height": 900},
                    extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
                )
                page = context.new_page()
                # Server-rendered HTML -- domcontentloaded is enough, no networkidle wait
                # needed the way the JS-heavy Google results page requires.
                page.goto(url, timeout=self.timeout_ms, wait_until="domcontentloaded")

                extracted = page.evaluate(
                    """
                    () => {
                      const results = [];
                      document.querySelectorAll('.result, .web-result').forEach(block => {
                        const a = block.querySelector('a.result__a');
                        if (!a || !a.href) return;
                        const snippetEl = block.querySelector('.result__snippet');
                        results.push({
                          title: (a.innerText || '').trim(),
                          link: a.href,
                          snippet: snippetEl ? (snippetEl.innerText || '').trim() : '',
                        });
                      });
                      return { results, pageTitle: document.title };
                    }
                    """
                )
                results = extracted.get("results", [])
            finally:
                browser.close()

        if not results:
            return (
                f"No DuckDuckGo results found for '{query}'. Try a shorter/simpler query, or fall "
                "back to the 'playwright_google_search' tool."
            )

        lines = [f"## DuckDuckGo Search Results for '{query}'\n"]
        n = 0
        for r in results:
            link = (r.get("link") or "").strip()
            if not link:
                continue
            n += 1
            title = (r.get("title") or link).strip()
            snippet = (r.get("snippet") or "").strip()
            lines.append(f"{n}. [{title}]({link})\n{snippet}\n")
            if n >= self.max_results:
                break

        return "\n".join(lines)


class PlaywrightGoogleSearchTool(Tool):
    """Searches Google by driving a real headless browser with Playwright, as an
    additional, free alternative alongside the existing paid/rate-limited search
    tools -- not a replacement for them.

    A fresh, isolated browser is launched per call (rather than one shared browser
    kept alive across calls) -- slightly slower per search, but avoids any
    thread-safety concerns from Gradio potentially handling requests on different
    threads, and keeps this tool's lifecycle dead simple (nothing to explicitly shut
    down at app exit).
    """

    name = "web_search"
    description = (
        "Performs a Google web search for your query and returns the top results "
        "(title, link, short snippet) as a numbered list. Uses a real browser "
        "(Playwright) that types the query into google.com directly -- no paid "
        "search API and no third-party search library involved. Keep queries short "
        "and simple -- the plain topic name plus a country/keyword or two -- rather "
        "than long boolean or quoted-phrase chains."
    )
    inputs = {
        "query": {"type": "string", "description": "The search query to perform. Keep it short and simple."},
        "filter_year": {
            "type": "string",
            "description": "(Optional) restrict results to this year, e.g. '2023'.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, max_results: int = 8, headless: bool = False, timeout_ms: int = 20000, debug: bool = False):
        super().__init__()
        self.max_results = max_results
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.debug = debug

    def forward(self, query: str, filter_year: str | None = None) -> str:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise ImportError(
                "PlaywrightGoogleSearchTool needs the `playwright` package and its "
                "browser binaries. Install with:\n"
                "    pip install playwright\n"
                "    playwright install chromium"
            ) from e

        query = (query or "").strip()
        if not query:
            return "Error: empty search query."

        params = {"q": query, "num": str(max(self.max_results, 10)), "hl": "en", "gl": "us", "pws": "0"}
        if filter_year:
            params["tbs"] = f"cdr:1,cd_min:1/1/{filter_year},cd_max:12/31/{filter_year}"
        url = "https://www.google.com/search?" + urllib.parse.urlencode(params)

        page_info = ""
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=self.headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
            try:
                context = browser.new_context(
                    user_agent=_USER_AGENT,
                    locale="en-US",
                    viewport={"width": 1366, "height": 900},
                    extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
                )
                context.add_init_script(_ANTI_DETECTION_INIT_SCRIPT)
                page = context.new_page()
                page.goto(url, timeout=self.timeout_ms, wait_until="domcontentloaded")

                # Best-effort consent-dialog dismissal (Google's EU/UK cookie wall) --
                # harmless no-op if no such dialog appears for this session/region.
                for label in _CONSENT_BUTTON_LABELS:
                    try:
                        page.get_by_role("button", name=label).click(timeout=1200)
                        page.wait_for_load_state("domcontentloaded", timeout=self.timeout_ms)
                        break
                    except Exception:
                        continue

                # Wait for actual result content rather than a fixed sleep -- far more
                # reliable across Google's varying render times. If nothing shows up
                # in time, keep going anyway and let the extraction below report
                # honestly (this also surfaces a captcha/consent page as "no results"
                # rather than hanging).
                try:
                    page.wait_for_selector("h3, #rso, #search", timeout=6000)
                except Exception:
                    pass
                page.wait_for_timeout(400)

                extracted = page.evaluate(
                    """
                    ([ignoredFragments]) => {
                      const isIgnored = (href) => ignoredFragments.some(f => href.includes(f));

                      const results = [];
                      const seenLinks = new Set();

                      // Primary pass: every <h3> (the title Google puts on each organic
                      // result), walking up to its nearest linked ancestor.
                      document.querySelectorAll('h3').forEach(h3 => {
                        const a = h3.closest('a[href]') || h3.parentElement?.querySelector('a[href]');
                        if (!a || !a.href || !a.href.startsWith('http')) return;
                        if (isIgnored(a.href) || seenLinks.has(a.href)) return;

                        // Snippet: look at the result's outer block (a few levels up
                        // from the link) and take its text, minus the title itself.
                        let block = a.closest('div[data-hveid], div.g, div') || a.parentElement;
                        let text = block ? block.innerText : '';
                        const title = h3.innerText || '';
                        if (title && text.startsWith(title)) text = text.slice(title.length);
                        text = text.split('\\n').map(s => s.trim()).filter(Boolean).slice(0, 2).join(' ');

                        seenLinks.add(a.href);
                        results.push({ title: title || a.href, link: a.href, snippet: text });
                      });

                      // Fallback pass: if the h3-based pass found nothing (e.g. an
                      // unusual layout variant), broadly scan external links with
                      // non-trivial link text instead.
                      if (results.length === 0) {
                        document.querySelectorAll('a[href^="http"]').forEach(a => {
                          const href = a.href;
                          if (isIgnored(href) || seenLinks.has(href)) return;
                          const text = (a.innerText || '').trim();
                          if (text.length < 8) return;
                          seenLinks.add(href);
                          results.push({ title: text, link: href, snippet: '' });
                        });
                      }

                      return {
                        results: results.slice(0, 30),
                        pageTitle: document.title,
                        bodyLen: document.body ? document.body.innerText.length : 0,
                      };
                    }
                    """,
                    [_IGNORED_HOST_FRAGMENTS],
                )
                results = extracted.get("results", [])
                if self.debug:
                    page_info = (
                        f" [debug: page title='{extracted.get('pageTitle')}', "
                        f"body chars={extracted.get('bodyLen')}]"
                    )
            finally:
                browser.close()

        if not results:
            year_note = f" (filtered to {filter_year})" if filter_year else ""
            return (
                f"No Google results found for '{query}'{year_note}.{page_info} This can happen if Google "
                "served a consent/CAPTCHA page instead of results -- try again, try a shorter query, or "
                "retry with a different web search tool."
            )

        lines = [f"## Search Results for '{query}'{page_info}\n"]
        n = 0
        for r in results:
            link = (r.get("link") or "").strip()
            if not link:
                continue
            n += 1
            title = (r.get("title") or link).strip()
            snippet = (r.get("snippet") or "").strip()
            lines.append(f"{n}. [{title}]({link})\n{snippet}\n")
            if n >= self.max_results:
                break

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Extensions/keywords that make a link look like the actual downloadable document
# file rather than a landing/details page about it. Deliberately broad (covers the
# common legal-document file types plus generic "get me the file" wording) -- a
# false positive here just means the agent double-checks a link that turns out to
# be another HTML page, which playwright_visit_page reports honestly either way.
# ---------------------------------------------------------------------------
_DOWNLOAD_EXT_RE = r"\.(pdf|docx?|rtf|txt)(\?|#|$)"
_DOWNLOAD_KEYWORD_RE = r"download|full[\s_-]?text|view[\s_-]?(the[\s_-])?(authoris|full)|attachment|/file/"


class PlaywrightVisitPageTool(Tool):
    """Opens a URL with a real headless Chromium browser (Playwright) and returns its
    visible text plus every link on the page, with links that look like an actual
    downloadable document (by file extension or surrounding wording) flagged and
    listed first. This is the "read/browse a page" half of a Playwright-only
    search-and-browse loop -- paired with PlaywrightGoogleSearchTool for the
    "search" half -- so neither step needs the `requests` library, a paid search
    API, or the existing requests-based SimpleTextBrowser.

    If the URL itself resolves directly to a non-HTML file (a PDF, DOCX, etc.),
    that's reported immediately as "this URL IS the document" instead of trying to
    scrape it as a web page -- the agent doesn't need to visit it a second time.
    """

    name = "playwright_visit_page"
    description = (
        "Opens a URL with a real browser (Playwright) and returns the page's visible text plus "
        "every link found on it, with links that look like an actual downloadable document file "
        "(PDF/DOC/DOCX/TXT, or wording like 'Download'/'Full text'/'View authorised version') "
        "flagged with a leading '⬇️ DOWNLOAD?' marker and listed first. Use this to read a search "
        "result page, then call it again on any flagged link found on that page to go one level "
        "deeper (e.g. from a government 'Details' page to the actual PDF it links to). If the URL "
        "you pass in is itself already a direct file (not an HTML page), this tool reports that "
        "immediately instead of trying to scrape it as a web page."
    )
    inputs = {
        "url": {"type": "string", "description": "The exact URL to open (from a search result or a link found on a previously visited page)."},
    }
    output_type = "string"

    def __init__(
        self,
        headless: bool = False,
        timeout_ms: int = 20000,
        max_text_chars: int = 5000,
        max_links: int = 40,
    ):
        super().__init__()
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.max_text_chars = max_text_chars
        self.max_links = max_links

    def forward(self, url: str) -> str:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise ImportError(
                "PlaywrightVisitPageTool needs the `playwright` package and its browser "
                "binaries. Install with:\n"
                "    pip install playwright\n"
                "    playwright install chromium"
            ) from e

        url = (url or "").strip()
        if not url:
            return "Error: empty URL."
        if not (url.startswith("http://") or url.startswith("https://")):
            return f"Error: '{url}' is not an http(s) URL."

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=self.headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
            try:
                context = browser.new_context(
                    user_agent=_USER_AGENT,
                    locale="en-US",
                    viewport={"width": 1366, "height": 900},
                    extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
                    accept_downloads=True,
                )
                context.add_init_script(_ANTI_DETECTION_INIT_SCRIPT)
                page = context.new_page()

                try:
                    response = page.goto(url, timeout=self.timeout_ms, wait_until="domcontentloaded")
                except Exception as e:
                    # A direct file link (e.g. a PDF) sometimes triggers Playwright's own
                    # download handling rather than a normal navigation, which surfaces as
                    # an exception here rather than a response -- treat that as a strong
                    # signal this URL IS a downloadable file, not a real navigation failure.
                    if "Download is starting" in str(e):
                        return (
                            f"'{url}' triggered a file download when opened -- this URL IS the "
                            "document itself (not an HTML landing page). Use it directly as the "
                            "document's URL; no further browsing of this link is needed."
                        )
                    return f"Error: could not load '{url}': {e}"

                status = response.status if response else None
                if status and status >= 400:
                    return f"Error: '{url}' returned HTTP {status}."

                content_type = (response.headers.get("content-type", "") if response else "").lower()
                if content_type and "html" not in content_type and "text/plain" not in content_type:
                    cl = response.headers.get("content-length") if response else None
                    size_note = f", {cl} bytes" if cl and cl.isdigit() else ""
                    return (
                        f"'{url}' is itself a direct downloadable file (Content-Type: "
                        f"{content_type}{size_note}). This URL IS the document -- no further "
                        "browsing of this link is needed."
                    )

                # Best-effort settle for JS-rendered content -- keep going even if the page
                # never truly goes idle (some sites keep long-poll/analytics connections open).
                try:
                    page.wait_for_load_state("networkidle", timeout=min(self.timeout_ms, 8000))
                except Exception:
                    pass

                data = page.evaluate(
                    """
                    ([downloadExtRe, downloadKeywordRe]) => {
                      const extRe = new RegExp(downloadExtRe, 'i');
                      const kwRe = new RegExp(downloadKeywordRe, 'i');
                      const seen = new Set();
                      const links = [];
                      document.querySelectorAll('a[href^="http"]').forEach(a => {
                        const href = a.href;
                        if (!href || seen.has(href)) return;
                        const text = (a.innerText || a.textContent || '').trim().replace(/\\s+/g, ' ');
                        if (!text && !extRe.test(href)) return;
                        seen.add(href);
                        const likelyDownload = extRe.test(href) || kwRe.test(href) || kwRe.test(text);
                        links.push({ href, text: text.slice(0, 120), likelyDownload });
                      });
                      return {
                        title: document.title || '',
                        text: document.body ? document.body.innerText : '',
                        links,
                      };
                    }
                    """,
                    [_DOWNLOAD_EXT_RE, _DOWNLOAD_KEYWORD_RE],
                )
            finally:
                browser.close()

        title = data.get("title", "") or url
        text = (data.get("text", "") or "").strip()
        if len(text) > self.max_text_chars:
            text = text[: self.max_text_chars] + "\n...[truncated]"

        links = data.get("links", [])
        flagged = [ln for ln in links if ln.get("likelyDownload")]
        other = [ln for ln in links if not ln.get("likelyDownload")]
        shown = (flagged + other)[: self.max_links]

        lines = [f"## Page: {title}\nURL: {url}\n", "### Content\n", text or "(no visible text extracted)", ""]
        lines.append(f"### Links found on this page ({len(links)} total, showing {len(shown)})")
        if flagged:
            lines.append(f"({len(flagged)} flagged as possible document downloads -- listed first)")
        for i, ln in enumerate(shown, start=1):
            marker = "⬇️ DOWNLOAD? " if ln.get("likelyDownload") else ""
            label = ln.get("text") or ln.get("href")
            lines.append(f"{i}. {marker}[{label}]({ln.get('href')})")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PlaywrightExtractLegalDocumentLinksTool
#
# Purpose: a dedicated "extract every link, then score/rank it against my
# target topic" step -- run BEFORE opening individual candidate pages one by
# one. Many government/legal-register sites (results listings, regulator
# publication pages, gazette indexes) are JavaScript-rendered, so a plain
# `requests`-based link scrape misses most or all of the real links; this
# reuses the same Playwright browser the rest of this module already drives.
#
# Deliberately a SEPARATE tool from PlaywrightVisitPageTool rather than a mode
# flag on it: PlaywrightVisitPageTool's job is "read this one page's content +
# its links"; this tool's job is "given a page, tell me which of its links are
# actually worth opening for a SPECIFIC target topic" -- it takes a
# topic_keywords argument and returns a ranked, scored list, not raw content.
# Named specifically (not "playwright_extract_links") so it's unambiguous next
# to playwright_visit_page / playwright_duckduckgo_search / playwright_google_search
# in a tool list, and so the agent doesn't confuse it with a generic scraper.
# ---------------------------------------------------------------------------
class PlaywrightExtractLegalDocumentLinksTool(Tool):
    """Opens a URL with a real browser (Playwright -- required for JavaScript-
    rendered listing/index pages a plain HTTP request can't see the links on),
    extracts every link on the page, scores each one for relevance against a
    target legal topic (keyword overlap with the link's visible text/URL, plus
    a bonus for links that look like an actual downloadable document file or
    carry download-style wording), and returns a ranked list -- most relevant
    first -- instead of a flat, unordered dump of every link on the page.

    Use this BEFORE opening individual candidate pages one by one: run it on a
    search-result page or a regulator's publications/listing page, look at the
    top-ranked links, and only then call `playwright_visit_page` on the ones
    that actually look relevant. If it reports zero links at all (a page with
    no outbound links, or a direct file URL), fall back to reading the page's
    own content with `playwright_visit_page` instead.
    """

    name = "playwright_extract_legal_document_links"
    description = (
        "Opens a URL with a real browser (Playwright) and extracts every link on the page, then "
        "scores and ranks each link by how relevant it looks to a target legal topic you specify "
        "(keyword overlap with the link's visible text/URL, plus a bonus for links that look like "
        "an actual downloadable document file -- PDF/DOC/DOCX/TXT -- or carry wording like "
        "'Download'/'Full text'/'View authorised version'). Returns the top-ranked links first, "
        "each with its relevance score, so you can decide which ones are actually worth opening "
        "before visiting them one by one -- use this BEFORE playwright_visit_page on a search-"
        "result page, a regulator's publications/listing page, or any other page likely to link "
        "out to several documents. Needs a real browser because many such pages are JavaScript-"
        "rendered and a plain HTTP request can't see their links at all. If this reports zero "
        "links (e.g. the URL is itself a direct file, or a page with no outbound links), fall "
        "back to reading the page's own content with playwright_visit_page instead."
    )
    inputs = {
        "url": {"type": "string", "description": "The exact URL of the page whose links you want to extract and rank."},
        "topic_keywords": {
            "type": "string",
            "description": (
                "Short, plain keywords describing the legal document/topic you're looking for, e.g. "
                "'cross-border data policy Cambodia' or 'data protection act Philippines'. Used to "
                "score each link's relevance -- keep it short, the same way you'd phrase a search query."
            ),
        },
    }
    output_type = "string"

    def __init__(
        self,
        headless: bool = False,
        timeout_ms: int = 20000,
        max_links_extracted: int = 200,
        max_links_shown: int = 25,
    ):
        super().__init__()
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.max_links_extracted = max_links_extracted
        self.max_links_shown = max_links_shown

    @staticmethod
    def _score_link(text: str, href: str, keywords: list[str]) -> int:
        """Cheap, deterministic relevance score -- no LLM call. Counts keyword
        hits in the link's visible text and URL (text hits weighted higher,
        since a link's own wording is a stronger relevance signal than
        incidental words in its URL path), plus a flat bonus for anything that
        already looks like a direct document download (by extension or
        surrounding wording, reusing this module's own download-detection
        regexes so the signal is consistent with playwright_visit_page)."""
        import re as _re

        text_l = (text or "").lower()
        href_l = (href or "").lower()
        score = 0
        for kw in keywords:
            kw = kw.lower().strip()
            if not kw:
                continue
            if kw in text_l:
                score += 3
            if kw in href_l:
                score += 1
        if _re.search(_DOWNLOAD_EXT_RE, href_l):
            score += 5
        if _re.search(_DOWNLOAD_KEYWORD_RE, href_l) or _re.search(_DOWNLOAD_KEYWORD_RE, text_l):
            score += 3
        return score

    def forward(self, url: str, topic_keywords: str) -> str:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise ImportError(
                "PlaywrightExtractLegalDocumentLinksTool needs the `playwright` package and its "
                "browser binaries. Install with:\n"
                "    pip install playwright\n"
                "    playwright install chromium"
            ) from e

        url = (url or "").strip()
        if not url:
            return "Error: empty URL."
        if not (url.startswith("http://") or url.startswith("https://")):
            return f"Error: '{url}' is not an http(s) URL."

        keywords = [w for w in (topic_keywords or "").replace(",", " ").split() if len(w) > 2]

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=self.headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
            try:
                context = browser.new_context(
                    user_agent=_USER_AGENT,
                    locale="en-US",
                    viewport={"width": 1366, "height": 900},
                    extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
                    accept_downloads=True,
                )
                context.add_init_script(_ANTI_DETECTION_INIT_SCRIPT)
                page = context.new_page()

                try:
                    response = page.goto(url, timeout=self.timeout_ms, wait_until="domcontentloaded")
                except Exception as e:
                    if "Download is starting" in str(e):
                        return (
                            f"'{url}' triggered a file download when opened -- this URL IS a document "
                            "itself, not a listing/index page with links to extract. Use it directly as "
                            "a candidate document; there are no links to rank here."
                        )
                    return f"Error: could not load '{url}': {e}"

                status = response.status if response else None
                if status and status >= 400:
                    return f"Error: '{url}' returned HTTP {status}."

                content_type = (response.headers.get("content-type", "") if response else "").lower()
                if content_type and "html" not in content_type and "text/plain" not in content_type:
                    return (
                        f"'{url}' is itself a direct downloadable file (Content-Type: {content_type}). "
                        "This URL IS a document -- there are no links to extract/rank here."
                    )

                try:
                    page.wait_for_load_state("networkidle", timeout=min(self.timeout_ms, 8000))
                except Exception:
                    pass

                data = page.evaluate(
                    """
                    () => {
                      const seen = new Set();
                      const links = [];
                      document.querySelectorAll('a[href^="http"]').forEach(a => {
                        const href = a.href;
                        if (!href || seen.has(href)) return;
                        const text = (a.innerText || a.textContent || '').trim().replace(/\\s+/g, ' ');
                        seen.add(href);
                        links.push({ href, text: text.slice(0, 160) });
                      });
                      return { title: document.title || '', links };
                    }
                    """
                )
            finally:
                browser.close()

        title = data.get("title", "") or url
        links = data.get("links", [])[: self.max_links_extracted]

        if not links:
            return (
                f"'{url}' (page: {title}) has no outbound http(s) links to extract. Fall back to "
                "reading this page's own content with playwright_visit_page instead."
            )

        scored = []
        for ln in links:
            href, text = ln.get("href", ""), ln.get("text", "")
            scored.append((self._score_link(text, href, keywords), href, text))
        scored.sort(key=lambda t: t[0], reverse=True)

        shown = scored[: self.max_links_shown]
        n_relevant = sum(1 for s, _, _ in scored if s > 0)

        lines = [
            f"## Links extracted from: {title}\nURL: {url}\n",
            f"Keywords used for scoring: {', '.join(keywords) if keywords else '(none given)'}\n",
            f"### Ranked links ({len(links)} total found, {n_relevant} scored relevant, showing top {len(shown)})\n",
            "Open the highest-scoring links first with playwright_visit_page. A score of 0 means no "
            "keyword or download signal matched at all -- usually site navigation/boilerplate, not a "
            "candidate document.\n",
        ]
        for i, (score, href, text) in enumerate(shown, start=1):
            label = text or href
            lines.append(f"{i}. [score={score}] [{label}]({href})")

        if n_relevant == 0:
            lines.append(
                "\nNote: nothing on this page scored above 0 for the given keywords -- consider a "
                "different topic_keywords phrasing, or fall back to reading the page's own content "
                "with playwright_visit_page in case the relevant text (not a link) is on this page "
                "itself."
            )

        return "\n".join(lines)


class PlaywrightReadEmbeddedPdfTool(Tool):
    """Opens a URL using Playwright, scans the page DOM to identify any embedded PDF files
    (via iframe, embed, or object tags, or direct PDF download links), downloads the PDF,
    and extracts its text content using PyMuPDF (fitz) to return to the agent.
    """

    name = "playwright_read_embedded_pdf"
    description = (
        "Opens a webpage URL using Playwright, scans the DOM to identify any embedded PDF files "
        "(in iframes, embed/object tags, or direct PDF links), downloads the PDF, and extracts "
        "its text content using PyMuPDF (fitz). Use this tool if you suspect a page contains an "
        "embedded PDF viewer displaying document text that isn't captured by normal page reading."
    )
    inputs = {
        "url": {"type": "string", "description": "The URL of the webpage that contains the embedded PDF."},
    }
    output_type = "string"

    def __init__(
        self,
        headless: bool = False,
        timeout_ms: int = 20000,
        max_pdf_chars: int = 25000,
    ):
        super().__init__()
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.max_pdf_chars = max_pdf_chars

    def forward(self, url: str) -> str:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise ImportError(
                "PlaywrightReadEmbeddedPdfTool needs the `playwright` package. Install with:\n"
                "    pip install playwright\n"
                "    playwright install chromium"
            ) from e

        import requests
        import tempfile
        import os
        import fitz  # PyMuPDF
        import urllib.parse

        url = (url or "").strip()
        if not url:
            return "Error: empty URL."

        # If the input URL is already a PDF, process it directly
        if url.lower().endswith(".pdf"):
            pdf_urls = [url]
            page_title = "Direct PDF Link"
        else:
            if not (url.startswith("http://") or url.startswith("https://")):
                return f"Error: '{url}' is not an http(s) URL."

            pdf_urls = []
            page_title = url
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=self.headless,
                    args=["--disable-blink-features=AutomationControlled"],
                )
                try:
                    context = browser.new_context(
                        user_agent=_USER_AGENT,
                        locale="en-US",
                        viewport={"width": 1366, "height": 900},
                        extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
                        accept_downloads=True,
                    )
                    context.add_init_script(_ANTI_DETECTION_INIT_SCRIPT)
                    page = context.new_page()

                    try:
                        response = page.goto(url, timeout=self.timeout_ms, wait_until="domcontentloaded")
                        page_title = page.title() or url
                    except Exception as e:
                        if "Download is starting" in str(e):
                            # The page URL itself triggered a direct download
                            pdf_urls = [url]
                        else:
                            return f"Error: could not load '{url}': {e}"

                    if not pdf_urls:
                        # Scan the DOM for embedded PDFs
                        detected = page.evaluate(
                            """
                            () => {
                              const urls = new Set();
                              
                              // Check iframes (common for PDF.js or direct PDF embed)
                              document.querySelectorAll('iframe').forEach(el => {
                                const src = el.src || '';
                                if (!src) return;
                                if (src.toLowerCase().includes('.pdf')) {
                                  urls.add(src);
                                }
                                // Try extracting from common PDF viewer query params like viewer.html?file=URL
                                const fileParam = src.match(/[?&]file=([^&]+)/);
                                if (fileParam) {
                                  try { urls.add(decodeURIComponent(fileParam[1])); } catch(err) {}
                                }
                              });

                              // Check embed tags
                              document.querySelectorAll('embed').forEach(el => {
                                const src = el.src || '';
                                if (src && (src.toLowerCase().includes('.pdf') || (el.type || '').toLowerCase() === 'application/pdf')) {
                                  urls.add(src);
                                }
                              });

                              // Check object tags
                              document.querySelectorAll('object').forEach(el => {
                                const data = el.data || '';
                                if (data && (data.toLowerCase().includes('.pdf') || (el.type || '').toLowerCase() === 'application/pdf')) {
                                  urls.add(data);
                                }
                              });

                              // Check links that look like PDFs
                              document.querySelectorAll('a[href]').forEach(el => {
                                const href = el.href || '';
                                const text = (el.innerText || el.textContent || '').trim().toLowerCase();
                                if (href.toLowerCase().includes('.pdf') || text.includes('view pdf') || text.includes('download pdf')) {
                                  urls.add(href);
                                }
                              });

                              return Array.from(urls);
                            }
                            """
                        )
                        # Resolve absolute URLs
                        for u in detected:
                            resolved = urllib.parse.urljoin(url, u)
                            if resolved not in pdf_urls:
                                pdf_urls.append(resolved)
                finally:
                    browser.close()

        if not pdf_urls:
            return f"No embedded PDFs or direct PDF links found on webpage: '{url}'."

        # Fetch and extract text from the detected PDFs (up to 3 to avoid excessive time/text limit issues)
        lines = [f"## Webpage Analyzed: {page_title}", f"URL: {url}", f"Detected {len(pdf_urls)} embedded PDF(s). Processing..."]
        
        for idx, pdf_url in enumerate(pdf_urls[:3], 1):
            lines.append(f"\n--- PDF #{idx}: {pdf_url} ---")
            temp_path = None
            try:
                headers = {"User-Agent": _USER_AGENT}
                res = requests.get(pdf_url, headers=headers, timeout=20, stream=True)
                if res.status_code >= 400:
                    lines.append(f"Error downloading PDF: HTTP Status {res.status_code}")
                    continue
                
                # Write to temp file
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                    for chunk in res.iter_content(chunk_size=8192):
                        tf.write(chunk)
                    temp_path = tf.name
                
                # Extract text using PyMuPDF (fitz)
                doc = fitz.open(temp_path)
                pdf_text_parts = []
                for p_idx, page in enumerate(doc):
                    page_text = page.get_text() or ""
                    pdf_text_parts.append(page_text)
                    if sum(len(x) for x in pdf_text_parts) > self.max_pdf_chars:
                        pdf_text_parts.append("\n...[truncated due to length limit]...")
                        break
                
                extracted_text = "\n".join(pdf_text_parts).strip()
                if extracted_text:
                    lines.append(extracted_text)
                else:
                    lines.append("(No text could be extracted from this PDF -- it might be scanned image pages)")
            except Exception as pdf_err:
                lines.append(f"Failed to process PDF: {pdf_err}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                        
        return "\n".join(lines)
