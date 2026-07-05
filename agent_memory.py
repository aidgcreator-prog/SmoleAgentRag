"""
agent_memory.py — Shared multi-turn memory helper for every smolagents
CodeAgent in the app (General Chat agentic mode, RAG Chat agentic mode,
Data Analysis), following the official smolagents guide:
https://huggingface.co/docs/smolagents/tutorials/memory

Background
----------
Previously every agentic tab called `agent.run(task)` with no `reset`
argument, which defaults to `reset=True` — smolagents wipes
`agent.memory.steps` at the start of every single call. That meant each
CodeAgent had *no* memory of anything from earlier in the same
conversation: a follow-up like "and what about last year?" right after an
agent answered a question had no idea what "last year" referred to.

The fix (per "Dynamically change the agent's memory" in the tutorial
above) is to call `agent.run(task, reset=False)` instead, so the agent's
own `TaskStep` / `ActionStep` / `PlanningStep` history accumulates across
turns and gets included when smolagents builds the next prompt.

The trade-off: left unchecked, that memory grows every turn forever,
which eventually bloats the prompt sent to the LLM on every subsequent
message — especially costly for the Data Analysis agent, whose per-turn
task prompt is already long. `cap_agent_memory()` below trims memory back
to only the most recent `max_turns` conversation turns after every run,
exactly the kind of manual memory surgery the tutorial's
"Dynamically change the agent's memory" section describes (there via a
step_callback trimming old screenshots; here via trimming whole old
turns).

Disk persistence
-----------------
smolagents itself has no built-in save/load for `agent.memory` (see
https://github.com/huggingface/smolagents/issues/1216 — a still-open
feature request) — `agent.memory.steps` is just a plain Python list the
docs' own tutorial shows you reading/replacing directly:

    # You could modify the memory as needed here by inputting the memory
    # of another agent.
    # agent.memory.steps = previous_agent.memory.steps

`save_agent_memory()` / `load_agent_memory_into()` below follow that exact
pattern, using each step class's own `.dict()` (the same method
`AgentMemory.get_succinct_steps()` uses internally) to serialize, and the
matching `smolagents.memory` class's constructor (`Cls(**dict)`) to
reconstruct — so it stays correct across smolagents versions without this
app hand-guessing field names.

Memory scope: GLOBAL PER TAB, NOT PER MODEL
--------------------------------------------
Earlier versions of this file namespaced persisted memory by
(tab, model_id), specifically to avoid replaying one model's memory into
a DIFFERENT model/chat-template — that mismatch is what originally caused
the "No user query found in messages" crash on Gemma (see
llama_backend.py's _to_plain_messages() fix).

Memory is now GLOBAL PER TAB instead: one memory file per tab (General /
RAG / Data Analysis), shared across every model used in that tab,
regardless of which model wrote which turn. This means switching models
mid-conversation (e.g. Qwen3 -> Gemma -> a GGUF model) carries the full
conversation forward instead of starting that model over with a blank
slate.

The tradeoff this reintroduces: a step written by one model's tool-call
format, when replayed into a different model/template, can occasionally
produce a slightly confused answer. This is now a soft-degradation risk
rather than a hard crash risk, for two independent reasons:
  1. llama_backend.py's LlamaCppModel._to_plain_messages() already
     guarantees at least one 'user' turn is present (injecting a
     placeholder "(continue)" turn otherwise) before handing messages to
     a GGUF chat template, and its generate() catches template-rejection
     ValueErrors and returns a friendly in-chat message instead of
     raising.
  2. Every chat.py handler (chat_general_agentic, chat_rag's agentic
     path, run_data_analysis) already wraps agent.run() in a try/except
     that surfaces any remaining failure as assistant text with a
     traceback, rather than crashing the whole request.
So a bad cross-model replay degrades to "confused answer" or "visible
error in chat", never a hard crash — an acceptable tradeoff for getting
genuinely global conversation memory.

A one-time migration path is included: if no global memory file exists
yet for a tab, load_agent_memory_into() falls back to the most recently
modified LEGACY per-(tab, model) file (the old naming scheme) so
upgrading from the previous version doesn't silently lose existing
conversation history.
"""

import inspect
import json
from pathlib import Path
from typing import Optional

from smolagents.memory import ActionStep, PlanningStep, TaskStep

# Try importing the components that require manual rehydration from a flattened dict state
try:
    from smolagents.memory import ToolCall
except ImportError:
    ToolCall = None

try:
    from smolagents.memory import Timing
except ImportError:
    Timing = None

try:
    from smolagents.models import ChatMessage
except ImportError:
    ChatMessage = None

AGENT_MEMORY_DIR = Path("./agent_memory_store")

# Maps the saved "type" tag back to the smolagents class used to
# reconstruct a step. Kept explicit (rather than getattr(smolagents.memory,
# name)) so a corrupted/tampered file can never be used to instantiate an
# arbitrary class.
_STEP_CLASSES = {
    "TaskStep": TaskStep,
    "ActionStep": ActionStep,
    "PlanningStep": PlanningStep,
}


def _safe_slug(text: str) -> str:
    """Turn a model id / tab name into a filesystem-safe fragment."""
    keep = "-_."
    return "".join(c if c.isalnum() or c in keep else "_" for c in str(text))[-120:]


def _memory_file_path(tab_key: str) -> Path:
    """Global (not per-model) memory file — one file per tab, shared
    across every model used in that tab. See the module docstring above
    for why this is no longer namespaced by model_id."""
    return AGENT_MEMORY_DIR / f"{_safe_slug(tab_key)}.json"


def _legacy_per_model_files(tab_key: str) -> list:
    """Old naming scheme was '{tab}__{model}.json' (memory namespaced per
    (tab, model_id)). Used only for one-time migration so existing memory
    isn't silently lost when upgrading to global (per-tab) memory. Returns
    matching files newest-modified first."""
    if not AGENT_MEMORY_DIR.exists():
        return []
    prefix = f"{_safe_slug(tab_key)}__"
    return sorted(
        AGENT_MEMORY_DIR.glob(f"{prefix}*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _filter_kwargs_for(cls, data: dict) -> dict:
    """Keep only the keys `cls.__init__` actually accepts.

    `step.dict()` serializes every field the CURRENTLY installed
    smolagents version's dataclass has. If a persisted file was written
    by a different smolagents version (extra/renamed fields — e.g. a
    `ToolCall` that used to carry a `type` field, or a `Timing` that used
    to carry `duration`), blindly doing `cls(**data)` raises a
    TypeError("unexpected keyword argument") and previously caused the
    ENTIRE step to be dropped (see load_agent_memory_into()'s except
    clause). Filtering down to only the accepted kwargs first means a
    version-mismatched extra field is silently ignored instead of
    torpedoing the whole step's reconstruction.
    """
    try:
        params = inspect.signature(cls.__init__).parameters
        allowed = set(params) - {"self"}
        return {k: v for k, v in data.items() if k in allowed}
    except (TypeError, ValueError):
        return data


def _rehydrate_step_data(data: dict) -> dict:
    """step.dict() flattens nested smolagents objects (ToolCall, ChatMessage, Timing)
    into plain dicts when saving. Reconstruct them here before rebuilding the step,
    otherwise smolagents' own to_messages() (which calls e.g. `tool_call.dict()`)
    crashes with "'dict' object has no attribute 'dict'" the first time persisted
    memory is replayed after an app reload.

    Each reconstruction is filtered through _filter_kwargs_for() first so a
    field that doesn't exist on the currently-installed class version
    (e.g. saved by an older/newer smolagents release) is dropped instead
    of failing the whole step.
    """
    data = dict(data)

    if ToolCall is not None and data.get("tool_calls"):
        data["tool_calls"] = [
            ToolCall(**_filter_kwargs_for(ToolCall, tc)) if isinstance(tc, dict) else tc
            for tc in data["tool_calls"]
        ]

    if ChatMessage is not None and isinstance(data.get("model_output_message"), dict):
        data["model_output_message"] = ChatMessage(
            **_filter_kwargs_for(ChatMessage, data["model_output_message"])
        )

    if Timing is not None and isinstance(data.get("timing"), dict):
        data["timing"] = Timing(**_filter_kwargs_for(Timing, data["timing"]))

    return data


def save_agent_memory(agent, tab_key: str) -> None:
    """Persist `agent.memory.steps` to disk as JSON, so conversation
    memory survives an app restart — not just a model swap within the
    same run. Call this after every `agent.run(...)` that used
    `reset=False` (i.e. whenever the "🧠 Conversation Memory" checkbox is
    on), right alongside `cap_agent_memory()`.

    Global per tab: this OVERWRITES the single memory file for `tab_key`
    regardless of which model is currently active — see the module
    docstring for why memory is no longer namespaced per model.

    Each step is serialized as its own `.dict()` (mirrors
    `AgentMemory.get_succinct_steps()`'s internal use of the same method)
    tagged with its class name, so it can be reconstructed exactly by
    `load_agent_memory_into()`. Best-effort: a step that fails to
    serialize (e.g. it's holding something exotic like a raw PIL image in
    `observations_images`) is skipped rather than aborting the whole save,
    since losing one step's detail is far better than losing all memory
    or crashing the chat turn over a save-side error.
    """
    try:
        AGENT_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        serialized = []
        for step in agent.memory.steps:
            step_type = type(step).__name__
            if step_type not in _STEP_CLASSES:
                continue  # unknown/future step type — skip rather than guess
            try:
                data = step.dict()
            except Exception:
                continue
            try:
                json.dumps(data)  # cheap validity probe before committing to the list
            except TypeError:
                # Something in this step isn't JSON-serializable (e.g. a
                # PIL.Image in observations_images, or a raw exception
                # object). Drop just the offending fields we know about
                # rather than the whole step.
                data.pop("observations_images", None)
                data.pop("task_images", None)
                try:
                    json.dumps(data)
                except TypeError:
                    continue
            serialized.append({"type": step_type, "data": data})

        path = _memory_file_path(tab_key)
        path.write_text(json.dumps(serialized, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[AgentMemory] Could not save memory for '{tab_key}': {e}")


def load_agent_memory_into(agent, tab_key: str) -> bool:
    """Load previously-saved steps from disk into `agent.memory.steps`,
    replacing whatever (empty, freshly-built-agent) memory it currently
    has — the same direct-assignment pattern smolagents' own tutorial uses
    for copying memory between agents:
        agent.memory.steps = previous_agent.memory.steps

    Global per tab: loads the ONE memory file for `tab_key`, shared across
    every model — see the module docstring for the tradeoffs this implies
    (a step written by one model, replayed into a different model's
    template, can occasionally look a little "off", but is guaranteed not
    to hard-crash the turn — see llama_backend.py's guards and every
    chat.py handler's try/except).

    If no global file exists yet (e.g. right after upgrading from the old
    per-(tab, model) scheme), falls back to the most recently modified
    LEGACY file for this tab so existing history isn't silently lost.

    Returns True if memory was found and loaded, False otherwise (no
    saved file yet, or it failed to parse — either way the agent is left
    with whatever memory it already had, so this is safe to call
    unconditionally right after building a fresh agent).
    """
    path = _memory_file_path(tab_key)
    if not path.exists():
        legacy_candidates = _legacy_per_model_files(tab_key)
        if not legacy_candidates:
            return False
        path = legacy_candidates[0]
        print(f"[AgentMemory] No global memory file for '{tab_key}' yet — "
              f"migrating from legacy file '{path.name}'.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        steps = []
        for entry in raw:
            cls = _STEP_CLASSES.get(entry.get("type"))
            if cls is None:
                continue
            try:
                # Rehydrate step data to turn flattened sub-dicts back into formal smolagents objects
                rehydrated_data = _rehydrate_step_data(entry["data"])
                steps.append(cls(**rehydrated_data))
            except Exception as e:
                # A single malformed/incompatible step (e.g. saved by an
                # older smolagents version with different fields) — skip
                # just that step rather than discarding the whole history.
                print(f"[AgentMemory] Skipped one incompatible step while "
                      f"loading '{tab_key}': {e}")
                continue
        if not steps:
            return False
        agent.memory.steps = steps
        return True
    except Exception as e:
        print(f"[AgentMemory] Could not load memory for '{tab_key}': {e}")
        return False


def clear_saved_memory(tab_key: str, model_id: Optional[str] = None) -> None:
    """Delete persisted memory file(s) from disk — call this alongside
    the existing 'Clear' button handlers in ui.py (which already reset
    the in-RAM agent memory via reset_agent()) so a cleared conversation
    stays cleared across a restart too.

    Global per tab: deletes the single global memory file for `tab_key`,
    plus any leftover LEGACY per-(tab, model) files (old naming scheme),
    so a stale legacy file can't get picked up by the migration fallback
    in load_agent_memory_into() after a Clear. `model_id` is accepted
    only for backward-compatible call signatures and is otherwise
    ignored, since memory is no longer namespaced per model.
    """
    try:
        _memory_file_path(tab_key).unlink(missing_ok=True)
        for legacy_path in _legacy_per_model_files(tab_key):
            try:
                legacy_path.unlink()
            except OSError:
                pass
    except Exception as e:
        print(f"[AgentMemory] Could not clear saved memory for '{tab_key}': {e}")


def cap_agent_memory(agent, max_turns: int = 6) -> None:
    """Keep only the most recent `max_turns` conversation turns in
    `agent.memory.steps`.

    Each turn starts with a `TaskStep` (the user's message, or in Data
    Analysis's case the full task prompt) and is followed by whatever
    `ActionStep`/`PlanningStep`s the agent took to answer it. This drops
    whole old turns rather than truncating mid-turn, so memory always
    stays structurally valid for smolagents to replay.

    The agent's system-prompt step lives separately in
    `agent.memory.system_prompt` and is never touched by this.
    """
    steps = agent.memory.steps
    task_indices = [i for i, s in enumerate(steps) if isinstance(s, TaskStep)]
    if len(task_indices) > max_turns:
        cutoff = task_indices[-max_turns]
        agent.memory.steps = steps[cutoff:]


def reset_if_context_changed(reset_fn, state: dict, new_key) -> bool:
    """Helper for tabs where a NEW context (e.g. a newly uploaded dataset)
    should invalidate old agent memory even though the underlying model
    hasn't changed — otherwise the agent would keep "remembering" a
    previous file's columns/stats while analyzing a different one.

    `state` is a small mutable dict (e.g. module-level `{"key": None}`)
    used to remember the last context key seen. Returns True if a reset
    was triggered.
    """
    if state.get("key") != new_key:
        reset_fn()
        state["key"] = new_key
        return True
    return False
