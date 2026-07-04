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

Memory is saved per (tab, model_id): replaying one model's memory into a
DIFFERENT model/chat-template is exactly what caused the "No user query
found in messages" crash on Gemma (see llama_backend.py's
_to_plain_messages() fix) — different templates format/expect turns
differently, so persisted memory is only loaded back for the same model
it was created with.
"""

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


def _memory_file_path(tab_key: str, model_id: str) -> Path:
    return AGENT_MEMORY_DIR / f"{_safe_slug(tab_key)}__{_safe_slug(model_id)}.json"


def _rehydrate_step_data(data: dict) -> dict:
    """step.dict() flattens nested smolagents objects (ToolCall, ChatMessage, Timing)
    into plain dicts when saving. Reconstruct them here before rebuilding the step,
    otherwise smolagents' own to_messages() (which calls e.g. `tool_call.dict()`)
    crashes with "'dict' object has no attribute 'dict'" the first time persisted
    memory is replayed after an app reload.
    """
    data = dict(data)
    
    if ToolCall is not None and data.get("tool_calls"):
        data["tool_calls"] = [
            ToolCall(**tc) if isinstance(tc, dict) else tc
            for tc in data["tool_calls"]
        ]
        
    if ChatMessage is not None and isinstance(data.get("model_output_message"), dict):
        data["model_output_message"] = ChatMessage(**data["model_output_message"])
        
    if Timing is not None and isinstance(data.get("timing"), dict):
        data["timing"] = Timing(**data["timing"])
        
    return data


def save_agent_memory(agent, tab_key: str, model_id: str) -> None:
    """Persist `agent.memory.steps` to disk as JSON, so conversation
    memory survives an app restart — not just a model swap within the
    same run. Call this after every `agent.run(...)` that used
    `reset=False` (i.e. whenever the "🧠 Conversation Memory" checkbox is
    on), right alongside `cap_agent_memory()`.

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

        path = _memory_file_path(tab_key, model_id)
        path.write_text(json.dumps(serialized, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[AgentMemory] Could not save memory for '{tab_key}'/'{model_id}': {e}")


def load_agent_memory_into(agent, tab_key: str, model_id: str) -> bool:
    """Load previously-saved steps from disk into `agent.memory.steps`,
    replacing whatever (empty, freshly-built-agent) memory it currently
    has — the same direct-assignment pattern smolagents' own tutorial uses
    for copying memory between agents:
        agent.memory.steps = previous_agent.memory.steps

    Returns True if memory was found and loaded, False otherwise (no
    saved file yet, or it failed to parse — either way the agent is left
    with whatever memory it already had, so this is safe to call
    unconditionally right after building a fresh agent).
    """
    path = _memory_file_path(tab_key, model_id)
    if not path.exists():
        return False
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
                      f"loading '{tab_key}'/'{model_id}': {e}")
                continue
        if not steps:
            return False
        agent.memory.steps = steps
        return True
    except Exception as e:
        print(f"[AgentMemory] Could not load memory for '{tab_key}'/'{model_id}': {e}")
        return False


def clear_saved_memory(tab_key: str, model_id: Optional[str] = None) -> None:
    """Delete persisted memory file(s) from disk — call this alongside
    the existing 'Clear' button handlers in ui.py (which already reset
    the in-RAM agent memory via reset_agent()) so a cleared conversation
    stays cleared across a restart too. If `model_id` is None, deletes
    every saved file for that tab (every model it was ever saved under).
    """
    try:
        if model_id is not None:
            _memory_file_path(tab_key, model_id).unlink(missing_ok=True)
            return
        prefix = f"{_safe_slug(tab_key)}__"
        if not AGENT_MEMORY_DIR.exists():
            return
        for f in AGENT_MEMORY_DIR.glob(f"{prefix}*.json"):
            try:
                f.unlink()
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

