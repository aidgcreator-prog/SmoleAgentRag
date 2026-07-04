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

from smolagents import CodeAgent, DuckDuckGoSearchTool, VisitWebpageTool

import models

_general_agent          = None
_general_agent_model_id = None
_general_agent_lock     = threading.Lock()

GENERAL_AGENT_INSTRUCTIONS = (
    "You are a helpful, friendly assistant with access to live web search "
    "(`web_search`) and a webpage reader (`visit_webpage`). Use them when "
    "a question needs current information, specific facts you're unsure "
    "of, or anything that could have changed since your training data. "
    "For simple, timeless, or purely conversational questions, just "
    "answer directly without searching — don't search unnecessarily."
)


def _build_code_agent(llm) -> CodeAgent:
    kwargs = dict(
        model=llm,
        tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
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
        return _general_agent


def reset_agent():
    """Drop the cached CodeAgent wrapper (does NOT unload the underlying
    LLM itself — that's shared/managed by models.get_llm()). Call this
    whenever the General Chat model changes or the LLM is force-reloaded/
    unloaded elsewhere, so this agent doesn't keep holding a stale model
    reference — mirrors rag_agent.reset_agent() / data_analysis.reset_agent().
    """
    global _general_agent, _general_agent_model_id
    _general_agent = None
    _general_agent_model_id = None
