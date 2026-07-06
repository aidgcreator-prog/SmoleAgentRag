"""
rag_agent.py — Agentic RAG via smolagents CodeAgent.

Unlike the previous RAG Chat design (always retrieve context, then inject
it into the prompt before the LLM ever sees the question), this agent is
handed a `retriever` tool (knowledge_base.RetrieverTool, wrapping the same
ChromaDB collection) and decides FOR ITSELF whether/when to call it — and
can call it more than once to refine its search — before writing a final
answer, following the same CodeAgent + Tool pattern as data_analysis.py.

IMPORTANT CAVEAT: tool-calling reliability depends heavily on the
underlying model. `Qwen/Qwen3-0.6B` (the app's default LLM) was originally
kept OUT of the CodeAgent/ToolCallingAgent path for exactly this reason —
it's small enough that it may never call the retriever tool at all, or call
it in a malformed way, and just hallucinate an answer instead. If agentic
RAG Chat seems to ignore the knowledge base, switch to a larger model
(e.g. Qwen3-4B or above) in the LLM dropdown before assuming something is
broken.
"""

import inspect
import threading
from typing import Optional

from smolagents import CodeAgent

import model_registry as mr
import models
from knowledge_base import RetrieverTool

_rag_agent          = None
_rag_agent_model_id = None
_rag_tool           = None
_rag_agent_lock     = threading.Lock()

# ──────────────────────────────────────────────────────────────────
# Strict grounding — layer 1: agent-level system instructions (applied
# once, at agent construction, if this smolagents version supports it).
# ──────────────────────────────────────────────────────────────────
STRICT_SYSTEM_INSTRUCTIONS = (
    "You are a strict retrieval-augmented assistant.\n\n"
    "THE ONLY TOOL THAT EXISTS is `retriever`, which searches the user's "
    "own indexed knowledge base. There is NO other tool — in particular "
    "there is no `conversation_history`, `memory`, `get_previous_messages`, "
    "or similar function. NEVER call, import, or reference a function that "
    "isn't `retriever`; if you do, it will fail with a 'Forbidden function "
    "evaluation' error and waste a step.\n\n"
    "YOUR OWN CONVERSATION HISTORY IS ALREADY VISIBLE TO YOU: every earlier "
    "user message and your own earlier replies in this conversation are "
    "already included in what you can see — you do not need a tool to "
    "'retrieve' them. If asked what was said earlier/previously in this "
    "chat (as opposed to a knowledge-base question), look back at the "
    "earlier turns you can already see and answer from them directly — do "
    "NOT guess or invent content that isn't actually there.\n\n"
    "For actual questions about the user's documents, you must answer ONLY "
    "using information returned by the `retriever` tool. Never answer from "
    "your own general knowledge or training data, even if you believe you "
    "know the answer. Always call `retriever` before answering such a "
    "question. If the retrieved documents don't contain the answer, say "
    "clearly that the knowledge base doesn't contain this information — do "
    "not guess or fill the gap yourself. Don't repeat the exact same "
    "`retriever` query more than once — refine it or stop.\n\n"
    "CITATION REQUIREMENT: every chunk the retriever returns is tagged with "
    "its source in brackets, e.g. '[report.pdf]'. Cite that exact bracketed "
    "tag in-text immediately after every claim you make from it (e.g. "
    "'Revenue grew 12% [report.pdf].'). Finish your final answer with a "
    "'### References' section listing every distinct source you cited, "
    "e.g.:\n"
    "### References\n"
    "- report.pdf\n"
    "- notes.docx"
)

# ──────────────────────────────────────────────────────────────────
# Strict grounding — layer 2: per-question task template (applied every
# call, in chat.py, via build_strict_task()).
# ──────────────────────────────────────────────────────────────────
STRICT_NOT_FOUND_PHRASE = "the knowledge base doesn't contain this information"


def build_strict_task(question: str) -> str:
    """Wrap the user's question with explicit strict-grounding instructions.

    Repeats the same rule as STRICT_SYSTEM_INSTRUCTIONS at the per-task
    level too — belt and braces, since not every smolagents version applies
    custom `instructions` to every step equally, and reinforcement in the
    task itself measurably helps smaller/weaker models comply.
    """
    return (
        "Answer the question below using ONLY information you retrieve via "
        "the `retriever` tool. Do not use your own general knowledge, and do "
        "not guess.\n\n"
        "1. Call `retriever` with a focused search query based on the question.\n"
        "2. If the retrieved documents answer the question, write your final "
        "answer using only that retrieved information. Cite the bracketed "
        "source tag shown on each chunk (e.g. '[report.pdf]') in-text "
        "immediately after every claim drawn from it, and end your answer "
        "with a '### References' section listing every distinct source you "
        "cited.\n"
        "3. If they don't, try ONE more `retriever` call with a rephrased query.\n"
        "4. If you still can't find the answer in the retrieved documents, your "
        f"final answer must clearly state that {STRICT_NOT_FOUND_PHRASE} — do "
        "not fall back on anything you already know.\n\n"
        f"Question: {question}"
    )


# ──────────────────────────────────────────────────────────────────
# Strict grounding — layer 3: post-hoc verification. Doesn't rely on the
# model actually obeying the instructions above, NOR on introspecting
# smolagents' internal step/memory structure (which varies across
# versions and previously caused false "never searched" verdicts even
# when the retriever clearly ran and returned real results — see the
# CodeAgent transcript where `Out: Retrieved documents (sources: ...)`
# was printed but the old introspection still reported called=False).
#
# Instead, the RetrieverTool instance itself tracks how many times it
# was called and how many of those calls found something (see
# knowledge_base.RetrieverTool) — reset before each run, read after.
# ──────────────────────────────────────────────────────────────────
NEVER_SEARCHED_MESSAGE = (
    "⚠️ I didn't search the knowledge base for this, so I'm not answering — "
    "please try rephrasing your question."
)
NOTHING_FOUND_MESSAGE = (
    "⚠️ I searched the indexed knowledge base but found nothing relevant to "
    "this question, so I won't guess. Try rephrasing, or index a document "
    "that covers this topic first."
)


def reset_retriever_stats():
    """Call right before agent.run() so this run's stats start from zero."""
    if _rag_tool is not None:
        _rag_tool.reset_stats()


def get_retriever_stats() -> tuple:
    """Call right after agent.run() returns. Returns (call_count,
    found_count, sources_used) — sources_used is the ACTUAL set of source
    filenames the retriever returned this run, tracked directly by the
    tool itself (see knowledge_base.RetrieverTool) rather than trusting
    the model's own in-text citations — lets chat.py print a
    guaranteed-accurate References list regardless of whether the model
    remembered to write its own."""
    if _rag_tool is None:
        return 0, 0, set()
    return _rag_tool.call_count, _rag_tool.found_count, set(_rag_tool.sources_used)


RAG_AGENT_DEFAULT_MAX_STEPS = 6


def _build_code_agent(llm, tool: RetrieverTool, model_id: str = "") -> CodeAgent:
    # See general_agent._build_code_agent()'s comment / model_registry.
    # get_max_steps_for_model() — larger/slower local GGUF models pay a
    # much higher per-step cost when a parsing loop goes wrong, so their
    # step budget is scaled down to fail fast instead of grinding through
    # the full default.
    max_steps = mr.get_max_steps_for_model(model_id, RAG_AGENT_DEFAULT_MAX_STEPS)
    kwargs = dict(
        model=llm,
        tools=[tool],
        max_steps=max_steps,
        verbosity_level=1,
    )
    # smolagents' CodeAgent normally expects the model to wrap its answer in
    # <code>...</code> tags. In practice a lot of models — especially
    # "thinking"/reasoning-tuned ones, and pretty much anything running
    # through a raw llama.cpp chat template rather than transformers' own
    # template — are far more reliably biased (from code-heavy pretraining)
    # toward plain ```python fenced blocks instead, and will just write
    # prose otherwise, causing every step to fail parsing (see the "regex
    # pattern <code>(.*?)</code> was not found" error). Newer smolagents
    # versions expose `code_block_tags` on CodeAgent to switch to the more
    # broadly-compatible markdown-fence convention — use it when available.
    try:
        params = inspect.signature(CodeAgent.__init__).parameters
        if "code_block_tags" in params:
            kwargs["code_block_tags"] = "markdown"
        if "instructions" in params:
            kwargs["instructions"] = STRICT_SYSTEM_INSTRUCTIONS
    except (TypeError, ValueError):
        pass
    return CodeAgent(**kwargs)


def get_rag_agent(model_id: Optional[str] = None):
    """Lazily build (or rebuild, if the model changed) the agentic-RAG CodeAgent."""
    global _rag_agent, _rag_agent_model_id, _rag_tool
    target = model_id or models._llm_model_id

    if _rag_agent is not None and target == _rag_agent_model_id:
        return _rag_agent

    with _rag_agent_lock:
        if _rag_agent is not None and target == _rag_agent_model_id:
            return _rag_agent

        print(f"[RAGAgent] Building CodeAgent on '{target}' …")
        llm = models.get_llm(target)
        _rag_tool  = RetrieverTool()
        _rag_agent = _build_code_agent(llm, _rag_tool, target)
        _rag_agent_model_id = target
        # Standard smolagents behaviour: a freshly-built CodeAgent starts
        # with empty memory. See agent_memory.py's module docstring for
        # why this app no longer tries to restore memory from a previous
        # model or app session.
        return _rag_agent


def reset_agent():
    """Drop the cached CodeAgent wrapper (does NOT unload the underlying LLM
    itself — that's shared/managed by models.get_llm()). Call this whenever
    the RAG-tab model changes or the LLM is force-reloaded/unloaded
    elsewhere, so this agent doesn't keep holding a stale model reference.
    """
    global _rag_agent, _rag_agent_model_id, _rag_tool
    _rag_agent = None
    _rag_agent_model_id = None
    _rag_tool = None
