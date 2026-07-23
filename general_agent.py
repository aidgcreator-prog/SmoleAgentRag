"""
general_agent.py — Agentic General Chat via smolagents CodeAgent, using
smolagents' own built-in tools (see:
https://huggingface.co/docs/smolagents/tutorials/tools) instead of any
app-specific tool like RAG Chat's RetrieverTool (knowledge_base.py) or
Data Analysis's install_package (data_analysis.py).

Tools used here:
  - DuckDuckGoSearchTool — live web search
  - VisitWebpageTool     — fetch + read a page's full text when a search
                           snippet isn't enough
  - SpeechToTextTool     — transcribes an audio file/URL to text (see
                           https://huggingface.co/docs/smolagents/en/reference/default_tools#smolagents.SpeechToTextTool).
                           Useful if the user's message references a local
                           audio file path or URL they want transcribed.
                           NOTE: this is a separate, simpler code path from
                           the app's own dedicated 🎙️ Speech to Text tab
                           (models.py's transcribe_audio()) — it downloads
                           and lazily loads its own fixed Whisper checkpoint
                           (openai/whisper-large-v3-turbo) the first time
                           the agent actually calls it, is not configurable
                           from the model dropdowns, and doesn't chunk long
                           (>30s) audio or force a language the way
                           models.py's pipeline does. For serious/long
                           transcription work, point the user at the
                           dedicated STT tab instead; this tool is meant
                           for the agent to handle short ad-hoc audio
                           incidentally mentioned mid-conversation.

PythonInterpreterTool is deliberately NOT added: it's only meaningful for
ToolCallingAgent (JSON-style tool calls). CodeAgent already writes and
executes its own Python for every step, so adding PythonInterpreterTool
as a *tool* on top of that would just be a redundant, confusing entry in
the agent's toolbox.

Same caveat as rag_agent.py / data_analysis.py: tool-calling reliability
depends heavily on the underlying model. Qwen3-0.6B (this app's default)
may never call a tool at all, or call it malformed, and just hallucinate
an answer from its own knowledge instead. If the agent seems to ignore
the web tools, switch to a larger model (Qwen3-4B+, Qwen2.5-Coder-3B, or
any capable GGUF model) in the LLM dropdown first.
"""

import inspect
import re
import threading
from typing import Optional

from smolagents import CodeAgent, DuckDuckGoSearchTool, SpeechToTextTool, VisitWebpageTool

import model_registry as mr
import models

_general_agent          = None
_general_agent_model_id = None
_general_agent_lock     = threading.Lock()

# ──────────────────────────────────────────────────────────────────
# Citation tracking — DuckDuckGoSearchTool / VisitWebpageTool are
# smolagents' own built-in tools, not app-specific ones we can add
# call-count/found-count bookkeeping to by editing their source. Instead,
# thin subclasses record every query run / URL actually opened, mirroring
# knowledge_base.RetrieverTool's self-tracked call_count/found_count
# pattern (see rag_agent.py's strict-grounding verification for why: it's
# more reliable than trying to introspect the agent's internal
# step/memory structure, which varies across smolagents versions).
#
# This lets chat.py append a guaranteed-accurate "sources consulted this
# turn" list after every agentic run, independent of whether the model
# actually remembered to write its own in-text citations/References
# section (which — like RAG's strict grounding — depends on the
# underlying model's instruction-following and can't be fully trusted).
# ──────────────────────────────────────────────────────────────────
# smolagents' DuckDuckGoSearchTool formats each hit as a markdown link
# followed by its snippet, e.g. "[Page Title](https://example.com/page)\n
# some snippet text…", under a "## Search Results" header. This pulls the
# real (title, url) pairs straight out of that text so chat.py can cite an
# actual clickable source instead of just repeating the query string (a
# query the model typed is not itself a source — see the "did not provide
# the actual link" issue this fixes).
_MARKDOWN_LINK_RE = re.compile(r'\[([^\]]+)\]\((https?://[^\s\)]+)\)')

# Matches the "### References" (or similar) heading GENERAL_AGENT_INSTRUCTIONS
# asks the model to end its answer with, so the section body can be pulled
# out and parsed for the URLs the model actually claims it used.
_REFERENCES_HEADER_RE = re.compile(r'#{1,6}\s*references?\b', re.IGNORECASE)
_BARE_URL_RE = re.compile(r'https?://\S+')


def extract_model_cited_urls(answer: str) -> list:
    """Pull (title, url) pairs out of the model's own '### References'
    section at the end of its answer.

    Tolerates both formats the model might use:
      - The format GENERAL_AGENT_INSTRUCTIONS explicitly asks for:
        '[1] Page Title — https://example.com/page'
      - Plain markdown links: '[Page Title](https://example.com/page)'

    Returns [] if there's no References section, or it contains no
    parseable URLs — callers must NOT assume the model complied, and
    should cross-check anything returned here against what was actually
    searched/visited this turn (see resolve_actually_used_sources()),
    since a model can still hallucinate a plausible-looking URL here.
    """
    if not answer:
        return []
    header_match = _REFERENCES_HEADER_RE.search(answer)
    if not header_match:
        return []
    section = answer[header_match.end():]

    pairs = []
    seen_urls = set()

    # Markdown-style [Title](url)
    for title, url in _MARKDOWN_LINK_RE.findall(section):
        url = url.strip().rstrip(').,;')
        if url not in seen_urls:
            pairs.append((title.strip(), url))
            seen_urls.add(url)

    # "[n] Title — url" / "Title - url" / bare-URL-on-a-line style
    for line in section.splitlines():
        line = line.strip()
        if not line:
            continue
        url_match = _BARE_URL_RE.search(line)
        if not url_match:
            continue
        url = url_match.group(0).rstrip(').,;')
        if url in seen_urls:
            continue
        title = line[:url_match.start()]
        title = re.sub(r'^\[?\d+\]?[.\-:)]*\s*', '', title).strip(' -—:')
        pairs.append((title or url, url))
        seen_urls.add(url)

    return pairs


def resolve_actually_used_sources(answer: str, urls_visited: list, result_links: list) -> list:
    """Return the (title, url) pairs that should be shown as this turn's
    actually-used references — the INTERSECTION of what the model claims
    it cited (its own '### References' section) and what was REALLY
    searched/visited this run.

    This deliberately does NOT just list every link `web_search` returned
    (`result_links`) — most search hits are never actually read or used
    by the model (e.g. an irrelevant result that happened to rank highly
    for a broad query), and dumping all of them produces a misleading
    "sources" list full of things the model never engaged with. Cross-
    checking against the model's own citations (while still validating
    those citations are real, not hallucinated) narrows it down to what
    was genuinely used.

    Falls back to `urls_visited` alone (pages the agent actually opened
    and read in full via `visit_webpage` — the strongest "used" signal
    available short of the model's own citations) if the model didn't
    write a parseable/matching References section. Only falls back
    further to the full `result_links` dump (every link `web_search`
    returned this turn) as a last resort, when neither a parseable
    citation nor a visited page is available — weaker evidence, but
    still better than reporting zero sources after a turn that genuinely
    searched.
    """
    known_title_by_url = {}
    for u in urls_visited:
        known_title_by_url.setdefault(u, u)
    for title, u in result_links:
        known_title_by_url.setdefault(u, title)

    cited = extract_model_cited_urls(answer)
    used, seen = [], set()
    for title, url in cited:
        real_url = url if url in known_title_by_url else next(
            (k for k in known_title_by_url if k.rstrip('/') == url.rstrip('/')), None
        )
        if real_url and real_url not in seen:
            used.append((title or known_title_by_url[real_url], real_url))
            seen.add(real_url)

    if used:
        return used

    # Model gave nothing usable/real — fall back to pages it actually
    # opened (NOT every search hit) as the next-best "used" evidence.
    if urls_visited:
        return [(u, u) for u in urls_visited if u not in seen]

    # Model never explicitly visited a page either (common — many models
    # answer directly from web_search's own snippets without ever calling
    # visit_webpage) and left no parseable citations. Falling back to
    # every link web_search returned this turn is weaker evidence than an
    # explicit citation or a visited page, but a search DID genuinely run
    # this turn, so showing nothing at all would be misleading — it would
    # look like no research happened when it did. Only used as a last
    # resort, after both stronger signals came up empty.
    return list(result_links)


def strip_trailing_references_section(answer: str) -> str:
    """Cut off the model's own '### References' section (if present) from
    the end of its answer, so chat.py can replace it with the verified
    list from resolve_actually_used_sources() instead of showing both the
    model's (unverified, possibly incomplete/hallucinated) version and a
    second, accurate one back to back.
    """
    if not answer:
        return answer
    header_match = _REFERENCES_HEADER_RE.search(answer)
    if not header_match:
        return answer
    return answer[:header_match.start()].rstrip()


class TrackedDuckDuckGoSearchTool(DuckDuckGoSearchTool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queries_run = []
        # Real (title, url) pairs actually surfaced by a search this run —
        # see _MARKDOWN_LINK_RE above. Deduplicated in insertion order.
        self.result_links = []

    def forward(self, query: str) -> str:
        # Guard against a model looping on the EXACT same query — seen in
        # practice with some larger local GGUF models, which will re-run
        # an identical search 5-6+ times in a row instead of noticing it
        # already has the results (each call still costs a real network
        # round-trip). Case/whitespace-insensitive match; a genuinely
        # different/refined query is still allowed through normally.
        normalized = query.strip().lower()
        if normalized and any(normalized == q.strip().lower() for q in self.queries_run):
            return (
                f"You already searched for '{query}' earlier in this run and "
                "got results — running the exact same query again will only "
                "return the same thing. Stop searching and write your final "
                "answer using what you already found, or search for something "
                "MEANINGFULLY DIFFERENT if you still need more information."
            )
        self.queries_run.append(query)
        result = super().forward(query)
        for title, url in _MARKDOWN_LINK_RE.findall(result or ""):
            pair = (title.strip(), url.strip())
            if pair not in self.result_links:
                self.result_links.append(pair)
        return result


class TrackedVisitWebpageTool(VisitWebpageTool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.urls_visited = []

    def forward(self, url: str) -> str:
        self.urls_visited.append(url)
        return super().forward(url)


# Module-level refs to the CURRENT agent's tracked-tool instances, so
# chat.py can read/reset them without needing to reach into the agent's
# internal tool registry — mirrors rag_agent.py's `_rag_tool` pattern.
_search_tool  = None
_webpage_tool = None

GENERAL_AGENT_INSTRUCTIONS = (
    "You are a helpful, friendly assistant.\n\n"
    "THE ONLY TOOLS THAT EXIST are: `web_search` (live web search), "
    "`visit_webpage` (fetch/read a page's full text), and `transcriber` "
    "(transcribe a local audio file path or direct audio URL). There is "
    "NO other tool — in particular there is no `conversation_history`, "
    "`memory`, `get_previous_messages`, or similar function. NEVER call, "
    "import, or reference a function that isn't one of the three named "
    "above; if you do, it will fail with a 'Forbidden function evaluation' "
    "error and waste a step.\n\n"
    "YOUR OWN CONVERSATION HISTORY IS ALREADY VISIBLE TO YOU: every "
    "earlier user message and your own earlier replies in this "
    "conversation are already included in what you can see — you do not "
    "need a tool to 'retrieve' them, and none exists for that purpose. If "
    "asked what was said earlier/previously/'yesterday' in this chat, "
    "look back at the earlier turns you can already see and answer from "
    "them directly — do NOT guess, invent, or hallucinate content that "
    "isn't actually there. If you genuinely cannot see any earlier turns "
    "(e.g. this is the first message, or conversation memory is off), say "
    "so plainly instead of making something up.\n\n"
    "Use web_search/visit_webpage only when a question needs current "
    "information, specific facts you're unsure of, or anything that could "
    "have changed since your training data. Use `transcriber` only if the "
    "user gives you a local audio file path or a direct URL to an audio "
    "file and asks about its contents. For simple, timeless, purely "
    "conversational questions, or anything answerable from the "
    "conversation you can already see, just answer directly without using "
    "any tool.\n\n"
    "DON'T REPEAT SEARCHES: never run the exact same web_search query "
    "twice in one turn. Once you have enough information to answer, stop "
    "searching and write your final answer — do not keep searching 'just "
    "in case'.\n\n"
    "CITATION REQUIREMENT: whenever your answer includes information or "
    "data that came from `web_search` or `visit_webpage` — i.e. anything "
    "NOT from your own general knowledge — you MUST cite it in-text with a "
    "bracketed number (e.g. [1], [2]) placed right after the specific "
    "claim it supports, and finish your final answer with a "
    "'### References' section listing each numbered source's title and "
    "URL, for example:\n"
    "### References\n"
    "[1] Page Title — https://example.com/page\n\n"
    "Do not add citation numbers or a References section for answers "
    "that come purely from your own knowledge with no tool use."
)


def build_task_with_citation_reminder(user_message: str) -> str:
    """Wrap the user's message with an explicit, PER-TASK reminder of the
    citation requirement — a second layer on top of
    GENERAL_AGENT_INSTRUCTIONS, mirroring rag_agent.build_strict_task()'s
    "belt and braces" reasoning exactly.

    Why this is needed: GENERAL_AGENT_INSTRUCTIONS (including its
    citation requirement) is only ever attached to the agent if this
    installed smolagents version's `CodeAgent.__init__` happens to expose
    an `instructions=` parameter — see the
    `if "instructions" in params: kwargs["instructions"] = ...` guard in
    _build_code_agent() below. On a smolagents version where that
    parameter doesn't exist, that assignment is silently skipped and the
    model never sees the citation requirement AT ALL — it has no way to
    know it's supposed to write a '### References' section, so
    chat.py's resolve_actually_used_sources() always finds nothing to
    parse, even after a turn that genuinely searched the web. Repeating
    the requirement here, in the per-call task text (which always
    reaches the model regardless of smolagents version), closes that
    gap — same reasoning rag_agent.py already applies to its own
    strict-grounding instructions.
    """
    return (
        f"{user_message}\n\n"
        "---\n"
        "Reminder: if you use web_search or visit_webpage to answer this, "
        "cite each such claim in-text with a bracketed number (e.g. [1]) "
        "placed right after the claim, and end your final answer with a "
        "'### References' section listing each numbered source's title "
        "and URL, e.g.:\n"
        "### References\n"
        "[1] Page Title — https://example.com/page\n\n"
        "If you answer purely from your own knowledge with no tool use, "
        "skip the citation numbers and References section entirely."
    )


GENERAL_AGENT_DEFAULT_MAX_STEPS = 8


def _build_code_agent(llm, model_id: str = "") -> CodeAgent:
    global _search_tool, _webpage_tool
    _search_tool  = TrackedDuckDuckGoSearchTool()
    _webpage_tool = TrackedVisitWebpageTool()
    # Larger/slower local GGUF models pay a much higher per-step cost when
    # a parsing/tool-calling loop goes wrong (each failed retry can take
    # 30s-4min+ instead of a few seconds) — scale the step budget down for
    # those so a broken run fails fast instead of grinding through all 8
    # steps. See model_registry.get_max_steps_for_model().
    max_steps = mr.get_max_steps_for_model(model_id, GENERAL_AGENT_DEFAULT_MAX_STEPS)
    kwargs = dict(
        model=llm,
        # SpeechToTextTool() only downloads/loads its Whisper checkpoint
        # lazily on first actual call (smolagents' Tool.setup() pattern —
        # see Tool.__call__ in smolagents/tools.py), so instantiating it
        # here up front costs nothing unless the agent actually uses it.
        tools=[_search_tool, _webpage_tool, SpeechToTextTool()],
        max_steps=max_steps,
        verbosity_level=1,
    )
    # Same markdown-fence compatibility switch used by rag_agent.py and
    # data_analysis.py — many models (esp. "thinking"-tuned ones, or
    # anything running through a raw llama.cpp chat template) write plain
    # ```python fenced blocks instead of the <code></code> tags CodeAgent
    # expects by default, which otherwise fails parsing on every step.
    try:
        params = inspect.signature(CodeAgent.__init__).parameters
        if "code_block_tags" in params:
            kwargs["code_block_tags"] = "markdown"
        if "instructions" in params:
            kwargs["instructions"] = GENERAL_AGENT_INSTRUCTIONS
    except (TypeError, ValueError):
        pass
    return CodeAgent(**kwargs)


def reset_tool_usage() -> None:
    """Call right before agent.run() so this run's tracked queries/URLs
    start from zero — mirrors rag_agent.reset_retriever_stats()."""
    if _search_tool is not None:
        _search_tool.queries_run = []
        _search_tool.result_links = []
    if _webpage_tool is not None:
        _webpage_tool.urls_visited = []


def get_tool_usage() -> tuple:
    """Call right after agent.run() returns. Returns (queries_run,
    urls_visited, result_links) — the ACTUAL search queries run, pages
    explicitly visited via visit_webpage, and (title, url) pairs actually
    surfaced by web_search's own results this turn — tracked directly by
    the tools themselves rather than trusting the model's own in-text
    citations (see the TrackedDuckDuckGoSearchTool / TrackedVisitWebpageTool
    docstring above for why). `result_links` is what should be used to
    build a real, clickable references list — the raw query string alone
    is not a citable source."""
    queries = list(_search_tool.queries_run) if _search_tool is not None else []
    urls    = list(_webpage_tool.urls_visited) if _webpage_tool is not None else []
    links   = list(_search_tool.result_links) if _search_tool is not None else []
    return queries, urls, links


def get_general_agent(model_id: Optional[str] = None):
    """Lazily build (or rebuild, if the model changed) the agentic
    General Chat CodeAgent."""
    global _general_agent, _general_agent_model_id
    target = model_id or models._llm_model_id

    if _general_agent is not None and target == _general_agent_model_id:
        return _general_agent

    with _general_agent_lock:
        if _general_agent is not None and target == _general_agent_model_id:
            return _general_agent

        print(f"[GeneralAgent] Building CodeAgent on '{target}' …")
        llm = models.get_llm(target)
        _general_agent = _build_code_agent(llm, target)
        _general_agent_model_id = target
        # Standard smolagents behaviour: a freshly-built CodeAgent starts
        # with empty memory (agent.memory.steps == []). This only happens
        # on first use in this tab, or after a model switch/reset — see
        # agent_memory.py's module docstring for why this app no longer
        # tries to restore memory from a previous model or app session.
        return _general_agent


def reset_agent():
    """Drop the cached CodeAgent wrapper (does NOT unload the underlying
    LLM itself — that's shared/managed by models.get_llm()). Call this
    whenever the General Chat model changes or the LLM is force-reloaded/
    unloaded elsewhere, so this agent doesn't keep holding a stale model
    reference — mirrors rag_agent.reset_agent() / data_analysis.reset_agent().
    """
    global _general_agent, _general_agent_model_id, _search_tool, _webpage_tool
    _general_agent = None
    _general_agent_model_id = None
    _search_tool = None
    _webpage_tool = None
