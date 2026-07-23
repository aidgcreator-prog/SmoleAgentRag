"""
deep_research_agent.py — Agentic "Deep Research" via a two-agent smolagents
setup, modeled on HuggingFace's own open_deep_research example:
https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research

That example uses a MANAGER agent that plans out a research task and
delegates focused sub-questions to a dedicated browser/search agent
(there: a `ToolCallingAgent` wrapping `GoogleSearchTool` + page-reading
tools, requiring a paid search API key and typically a hosted model like
`o1`). This app has neither a search API key nor a hosted-model
requirement, so the same manager+search-agent shape is rebuilt here using
what's already available:

  - The manager is a `CodeAgent` with NO tools of its own except the
    search sub-agent (passed via `managed_agents=`, which smolagents
    exposes to the manager as a callable — `web_search_agent(task="...")`
    — same as the real example's manager calling its own sub-agent).
  - `planning_interval` makes the manager periodically stop and
    re-evaluate its plan against what it's learned so far — the
    characteristic "deep" part of deep research, as opposed to
    general_agent.py's single-pass agentic chat.
  - The search sub-agent reuses general_agent.py's own
    TrackedDuckDuckGoSearchTool / TrackedVisitWebpageTool (free,
    API-key-free DuckDuckGo search + page reading — already proven out
    by General Chat's agentic mode) instead of GoogleSearchTool, and its
    citation-tracking/verification helpers (resolve_actually_used_sources
    etc.) are reused as-is from general_agent.py rather than duplicated.

Same reliability caveat as every other agentic tab in this app: this
needs a genuinely capable model to work well — small/local models can
struggle even more here than in General Chat's agentic mode, since a
manager also has to correctly invoke a SUB-AGENT (not just a tool) and
periodically re-plan. Expect this to work best on Qwen3-4B and above,
same guidance as the other agentic tabs.
"""

import inspect
import threading
from typing import Optional

from smolagents import CodeAgent

import general_agent
import model_registry as mr
import models

try:
    import playwright_search_tool
    _playwright_available = True
except ImportError:
    _playwright_available = False

if _playwright_available:
    class TrackedPlaywrightDuckDuckGoSearchTool(playwright_search_tool.PlaywrightDuckDuckGoSearchTool):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.queries_run = []
            self.result_links = []

        def forward(self, query: str) -> str:
            self.queries_run.append(query)
            result = super().forward(query)
            import re
            _MARKDOWN_LINK_RE = re.compile(r'\[([^\]]+)\]\((https?://[^\s\)]+)\)')
            for title, url in _MARKDOWN_LINK_RE.findall(result or ""):
                pair = (title.strip(), url.strip())
                if pair not in self.result_links:
                    self.result_links.append(pair)
            return result

    class TrackedPlaywrightGoogleSearchTool(playwright_search_tool.PlaywrightGoogleSearchTool):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.queries_run = []
            self.result_links = []

        def forward(self, query: str, filter_year: str | None = None) -> str:
            self.queries_run.append(query)
            result = super().forward(query, filter_year)
            import re
            _MARKDOWN_LINK_RE = re.compile(r'\[([^\]]+)\]\((https?://[^\s\)]+)\)')
            for title, url in _MARKDOWN_LINK_RE.findall(result or ""):
                pair = (title.strip(), url.strip())
                if pair not in self.result_links:
                    self.result_links.append(pair)
            return result

    class TrackedPlaywrightVisitPageTool(playwright_search_tool.PlaywrightVisitPageTool):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.urls_visited = []

        def forward(self, url: str) -> str:
            self.urls_visited.append(url)
            return super().forward(url)


_manager_agent          = None
_manager_agent_model_id = None
_manager_agent_config   = {}
_manager_agent_lock     = threading.Lock()

# Module-level refs to the CURRENT run's tracked-tool instances (the
# search sub-agent's own tools), so chat.py can read/reset them the same
# way it does for general_agent.py's tools — see
# general_agent.TrackedDuckDuckGoSearchTool / TrackedVisitWebpageTool.
_search_tool  = None
_webpage_tool = None

# The search sub-agent answers ONE focused sub-question per call and has
# no memory of its own across calls, so it needs far fewer steps than the
# manager, which is juggling the whole research task. The manager's
# budget is larger than general_agent.py's single-agent default (12 vs
# 8) since planning + delegation calls both cost steps on top of the
# actual research.
SEARCH_AGENT_DEFAULT_MAX_STEPS = 6
MANAGER_DEFAULT_MAX_STEPS      = 12

# How often (in manager steps) the manager stops to re-plan — smolagents'
# own `planning_interval` mechanism. This is the main thing that makes
# this agent "deep" rather than a single-pass agentic chat: every few
# steps it re-reads what it has found so far and can revise its approach
# instead of blindly executing an initial plan to the end.
DEEP_RESEARCH_PLANNING_INTERVAL = 3

SEARCH_AGENT_DESCRIPTION = (
    "Give this agent ONE focused sub-question (a plain-text string) and "
    "it will search the web and read pages to answer just that "
    "sub-question, returning what it found as plain text. It has NO "
    "memory of any other call you make to it, so every call must be "
    "fully self-contained — include whatever context it needs in the "
    "sub-question itself. Call it once per sub-question; don't ask it "
    "multiple unrelated things in one call."
)

DEEP_RESEARCH_INSTRUCTIONS = (
    "You are a deep-research assistant. You have NO web-search tool of "
    "your own — the ONLY way to get outside information is to delegate a "
    "focused sub-question to the `web_search_agent`, by writing code "
    "like:\n"
    "    result = web_search_agent(task=\"<one focused sub-question>\")\n"
    "    print(result)\n\n"
    "THE ONLY AGENT THAT EXISTS is `web_search_agent`. There is no other "
    "tool or function available — in particular there is no "
    "`web_search`, `visit_webpage`, `conversation_history`, or `memory` "
    "function. Never call, import, or reference anything else; it will "
    "fail with a 'Forbidden function evaluation' error and waste a step.\n\n"
    "YOUR OWN CONVERSATION HISTORY IS ALREADY VISIBLE TO YOU: earlier "
    "turns in this conversation (including your own past reports) are "
    "already included in what you can see. If a follow-up refers back to "
    "your last report, reuse what you already found instead of "
    "re-researching it from scratch — only delegate NEW sub-questions for "
    "genuinely new information.\n\n"
    "Work like this:\n"
    "1. Break the question down into 2-5 focused sub-questions that, "
    "together, cover what's needed to answer it well.\n"
    "2. Call `web_search_agent` once per sub-question. Never invent or "
    "assume a fact yourself — every factual claim in your final answer "
    "must trace back to a `web_search_agent` call from this run (or, for "
    "a follow-up, an earlier turn you can already see).\n"
    "3. Re-check your plan periodically against what you've actually "
    "found: if it changes what you still need to look up, adjust instead "
    "of blindly finishing the original plan.\n"
    "4. Once you have enough to answer well, STOP calling "
    "`web_search_agent` and write your final answer as a well-structured "
    "Markdown report: a short introduction, clearly-headed sections "
    "covering each theme/sub-question, and a brief conclusion.\n"
    "5. CITATION REQUIREMENT: cite sources in-text with a bracketed "
    "number (e.g. [1], [2]) immediately after every claim that came from "
    "a `web_search_agent` call, and finish with a '### References' "
    "section listing each numbered source's title and URL, e.g.:\n"
    "### References\n"
    "[1] Page Title — https://example.com/page"
)


def build_task_with_citation_reminder(user_message: str) -> str:
    """Wrap the user's question with an explicit, PER-TASK reminder of the
    citation requirement — mirrors general_agent.build_task_with_citation_reminder()
    and rag_agent.build_strict_task()'s belt-and-braces pattern exactly.

    DEEP_RESEARCH_INSTRUCTIONS (including its citation requirement) is
    only ever attached to the manager agent if this installed smolagents
    version's CodeAgent.__init__ happens to expose an `instructions=`
    parameter — see the `if "instructions" in params:` guard in
    _build_manager_agent() below. On a version where it doesn't, the
    manager never sees the citation requirement at all, so it never
    writes a '### References' section for
    general_agent.resolve_actually_used_sources() to parse in chat.py's
    chat_deep_research(), even after a run that genuinely delegated
    several web_search_agent calls. Repeating the requirement here — in
    the per-call task text, which always reaches the manager regardless
    of smolagents version — closes that gap.
    """
    return (
        f"{user_message}\n\n"
        "---\n"
        "Reminder: every factual claim in your final report that came "
        "from a web_search_agent call must be cited in-text with a "
        "bracketed number (e.g. [1]) placed right after the claim, and "
        "your final answer must end with a '### References' section "
        "listing each numbered source's title and URL, e.g.:\n"
        "### References\n"
        "[1] Page Title — https://example.com/page"
    )


def _build_search_agent(llm, model_id: str = "", use_playwright: bool = False, headless: bool = True, max_steps: Optional[int] = None, timeout: Optional[int] = None) -> CodeAgent:
    global _search_tool, _webpage_tool
    
    tools = []
    if use_playwright and _playwright_available:
        _search_tool = TrackedPlaywrightDuckDuckGoSearchTool(headless=headless)
        _webpage_tool = TrackedPlaywrightVisitPageTool(headless=headless)
        tools = [
            _search_tool,
            TrackedPlaywrightGoogleSearchTool(headless=headless),
            _webpage_tool,
            playwright_search_tool.PlaywrightExtractLegalDocumentLinksTool(headless=headless),
            playwright_search_tool.PlaywrightReadEmbeddedPdfTool()
        ]
        desc = (
            "Give this agent ONE focused sub-question (a plain-text string) and "
            "it will search the web using Playwright headless browser tools and read pages to answer just that "
            "sub-question, returning what it found as plain text. It has NO "
            "memory of any other call you make to it, so every call must be "
            "fully self-contained — include whatever context it needs in the "
            "sub-question itself. Call it once per sub-question; don't ask it "
            "multiple unrelated things in one call."
        )
    else:
        _search_tool  = general_agent.TrackedDuckDuckGoSearchTool()
        _webpage_tool = general_agent.TrackedVisitWebpageTool()
        tools = [_search_tool, _webpage_tool]
        desc = SEARCH_AGENT_DESCRIPTION

    final_max_steps = max_steps if max_steps is not None else mr.get_max_steps_for_model(model_id, SEARCH_AGENT_DEFAULT_MAX_STEPS)
    kwargs = dict(
        model=llm,
        tools=tools,
        max_steps=final_max_steps,
        verbosity_level=1,
        name="web_search_agent",
        description=desc,
    )
    # Same markdown-fence compatibility switch used by every other agentic
    # tab in this app (general_agent.py / rag_agent.py / data_analysis.py)
    # — see those files' comments for why.
    try:
        params = inspect.signature(CodeAgent.__init__).parameters
        if "code_block_tags" in params:
            kwargs["code_block_tags"] = "markdown"
        if "executor_kwargs" in params and timeout is not None:
            kwargs["executor_kwargs"] = {"timeout_seconds": timeout}
    except (TypeError, ValueError):
        pass
    return CodeAgent(**kwargs)


def _build_manager_agent(llm, search_agent: CodeAgent, model_id: str = "", max_steps: Optional[int] = None, timeout: Optional[int] = None) -> CodeAgent:
    final_max_steps = max_steps if max_steps is not None else mr.get_max_steps_for_model(model_id, MANAGER_DEFAULT_MAX_STEPS)
    kwargs = dict(
        model=llm,
        tools=[],
        managed_agents=[search_agent],
        planning_interval=DEEP_RESEARCH_PLANNING_INTERVAL,
        max_steps=final_max_steps,
        verbosity_level=1,
    )
    try:
        params = inspect.signature(CodeAgent.__init__).parameters
        if "code_block_tags" in params:
            kwargs["code_block_tags"] = "markdown"
        if "instructions" in params:
            kwargs["instructions"] = DEEP_RESEARCH_INSTRUCTIONS
        if "executor_kwargs" in params and timeout is not None:
            kwargs["executor_kwargs"] = {"timeout_seconds": timeout}
    except (TypeError, ValueError):
        pass
    return CodeAgent(**kwargs)


def reset_tool_usage() -> None:
    """Call right before agent.run() so this run's tracked queries/URLs
    start from zero — mirrors general_agent.reset_tool_usage()."""
    if _search_tool is not None:
        _search_tool.queries_run = []
        _search_tool.result_links = []
    if _webpage_tool is not None:
        _webpage_tool.urls_visited = []


def get_tool_usage() -> tuple:
    """Call right after agent.run() returns. Returns (queries_run,
    urls_visited, result_links) from the search sub-agent's own tools —
    see general_agent.get_tool_usage() for the equivalent on that tab.
    Passed to general_agent.resolve_actually_used_sources() so chat.py
    can print a guaranteed-accurate References list for this report."""
    queries = list(_search_tool.queries_run) if _search_tool is not None else []
    urls    = list(_webpage_tool.urls_visited) if _webpage_tool is not None else []
    links   = list(_search_tool.result_links) if _search_tool is not None else []
    return queries, urls, links


def get_deep_research_agent(model_id: Optional[str] = None, use_playwright: bool = False, headless: bool = True, manager_max_steps: Optional[int] = None, search_max_steps: Optional[int] = None, timeout: Optional[int] = None):
    """Lazily build (or rebuild, if the model changed) the manager agent
    and its search sub-agent."""
    global _manager_agent, _manager_agent_model_id, _manager_agent_config
    target = model_id or models._llm_model_id

    current_config = {
        "use_playwright": use_playwright,
        "headless": headless,
        "manager_max_steps": manager_max_steps,
        "search_max_steps": search_max_steps,
        "timeout": timeout
    }

    if _manager_agent is not None and target == _manager_agent_model_id and current_config == _manager_agent_config:
        return _manager_agent

    with _manager_agent_lock:
        if _manager_agent is not None and target == _manager_agent_model_id and current_config == _manager_agent_config:
            return _manager_agent

        print(f"[DeepResearch] Building manager + web_search_agent on '{target}' (Playwright: {use_playwright}, Headless: {headless}) …")
        llm = models.get_llm(target)
        search_agent = _build_search_agent(llm, target, use_playwright, headless, search_max_steps, timeout)
        _manager_agent = _build_manager_agent(llm, search_agent, target, manager_max_steps, timeout)
        _manager_agent_model_id = target
        _manager_agent_config = current_config
        # Standard smolagents behaviour: a freshly-built agent starts with
        # empty memory (see agent_memory.py's module docstring for why
        # this app doesn't try to restore memory from a previous
        # model/session here either).
        return _manager_agent


def reset_agent():
    """Drop the cached manager (and its search sub-agent). Does NOT unload
    the underlying LLM itself — that's shared/managed by models.get_llm().
    Call this whenever the Deep Research tab's model changes or the LLM
    is force-reloaded/unloaded elsewhere — mirrors general_agent.reset_agent().
    """
    global _manager_agent, _manager_agent_model_id, _manager_agent_config, _search_tool, _webpage_tool
    _manager_agent = None
    _manager_agent_model_id = None
    _manager_agent_config = {}
    _search_tool = None
    _webpage_tool = None
