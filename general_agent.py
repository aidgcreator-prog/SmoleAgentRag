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
import threading
from typing import Optional

from smolagents import CodeAgent, DuckDuckGoSearchTool, SpeechToTextTool, VisitWebpageTool

import agent_memory
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
class TrackedDuckDuckGoSearchTool(DuckDuckGoSearchTool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queries_run = []

    def forward(self, query: str) -> str:
        self.queries_run.append(query)
        return super().forward(query)


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

# Tab key used to namespace this agent's persisted memory file on disk —
# see agent_memory.save_agent_memory()/load_agent_memory_into(). Kept as a
# constant so chat.py can call save_agent_memory() after agent.run()
# without needing to know/repeat this tab's internal name.
MEMORY_TAB_KEY = "general"

GENERAL_AGENT_INSTRUCTIONS = (
    "You are a helpful, friendly assistant with access to live web search "
    "(`web_search`), a webpage reader (`visit_webpage`), and an audio "
    "transcriber (`transcriber`). Use web search/webpage reading when a "
    "question needs current information, specific facts you're unsure of, "
    "or anything that could have changed since your training data. Use "
    "`transcriber` if the user gives you a local audio file path or a "
    "direct URL to an audio file and asks about its contents. For simple, "
    "timeless, or purely conversational questions, just answer directly "
    "without using any tool — don't use one unnecessarily.\n\n"
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


def _build_code_agent(llm) -> CodeAgent:
    global _search_tool, _webpage_tool
    _search_tool  = TrackedDuckDuckGoSearchTool()
    _webpage_tool = TrackedVisitWebpageTool()
    kwargs = dict(
        model=llm,
        # SpeechToTextTool() only downloads/loads its Whisper checkpoint
        # lazily on first actual call (smolagents' Tool.setup() pattern —
        # see Tool.__call__ in smolagents/tools.py), so instantiating it
        # here up front costs nothing unless the agent actually uses it.
        tools=[_search_tool, _webpage_tool, SpeechToTextTool()],
        max_steps=8,
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
    if _webpage_tool is not None:
        _webpage_tool.urls_visited = []


def get_tool_usage() -> tuple:
    """Call right after agent.run() returns. Returns (queries_run,
    urls_visited) — the ACTUAL search queries/URLs used this turn,
    tracked directly by the tools themselves rather than trusting the
    model's own in-text citations (see the TrackedDuckDuckGoSearchTool /
    TrackedVisitWebpageTool docstring above for why)."""
    queries = list(_search_tool.queries_run) if _search_tool is not None else []
    urls    = list(_webpage_tool.urls_visited) if _webpage_tool is not None else []
    return queries, urls


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
        _general_agent = _build_code_agent(llm)
        _general_agent_model_id = target
        # A freshly-built CodeAgent starts with empty memory — restore
        # whatever was last persisted to disk FOR THIS EXACT MODEL (see
        # agent_memory.py's module docstring on why memory is never
        # replayed across a model/template switch). No-ops silently if
        # nothing was ever saved for this model yet.
        if agent_memory.load_agent_memory_into(_general_agent, MEMORY_TAB_KEY, target):
            print(f"[GeneralAgent] Restored persisted memory for '{target}'.")
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
