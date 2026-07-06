"""
agent_memory.py — Shared multi-turn memory helpers for every smolagents
CodeAgent in the app (General Chat agentic mode, RAG Chat agentic mode,
Data Analysis), following the official smolagents memory guide exactly:
https://huggingface.co/docs/smolagents/tutorials/memory

Standard smolagents memory model
---------------------------------
smolagents keeps an agent's own history in `agent.memory.steps` — a plain
Python list of TaskStep / ActionStep / PlanningStep objects. Two rules
from the official tutorial cover everything this app needs:

  1. `agent.run(task, reset=False)` keeps `agent.memory.steps` across
     calls, instead of the `reset=True` default that wipes it at the
     start of every single call — this is what lets a follow-up like
     "and what about last year?" refer back to what was already asked.
     Every chat handler (chat.py / data_analysis.py) already does this
     correctly: `agent.run(task, reset=not use_memory)`.
  2. Memory can be edited directly, as a plain list. The tutorial's own
     "Dynamically change the agent's memory" section does exactly this
     (`agent.memory.steps = previous_agent.memory.steps`, or trimming via
     a step_callback) — cap_agent_memory() below does the same thing
     (trim old turns by reassigning `agent.memory.steps`), and
     reset_memory() below calls smolagents' own `agent.memory.reset()`
     for the same effect when the whole conversation should be dropped.

This module is intentionally just those two small helpers now. It used
to also serialize `agent.memory.steps` to JSON on disk so memory would
survive an app restart, not just a model swap within the same run — that
was NEVER part of smolagents' own memory model (smolagents doesn't ship
save/load for `agent.memory` at all — see the still-open
https://github.com/huggingface/smolagents/issues/1216), and hand-rolling
it was the source of several real bugs:

  - Reconstructing ToolCall / ChatMessage / Timing objects from a
    flattened dict is version-sensitive; a smolagents upgrade can
    silently rename/drop a field, which needed several defensive
    "filter down to whatever kwargs this version's constructor accepts"
    layers just to avoid crashing on load — a lot of fragility for a
    feature smolagents itself doesn't officially support.
  - Memory was persisted GLOBALLY per tab (shared across every model used
    in that tab), so replaying a step written by one model's tool-call
    format into a DIFFERENT model's chat template could produce a
    genuinely confused answer — a real quality/correctness bug, not just
    a theoretical risk.
  - Worse, it actively defeated Data Analysis's "a new dataset was
    uploaded, so old memory about the previous one shouldn't carry over"
    logic: that code correctly detected the dataset had changed and reset
    the agent — but building a replacement agent right afterward then
    reloaded the *exact same stale, pre-reset memory straight back off
    disk*, silently undoing the reset it was supposed to perform.

This file now sticks to the plain, official, RAM-only workflow instead:
an agent's memory lives exactly as long as its Python object does. It's
gone the moment the model is switched, "Clear" is clicked, or the app
restarts — matching smolagents' own tutorial exactly, and matching the
"🧠 Conversation Memory (Experimental)" checkbox's own label: this trades
"memory survives a restart / model switch" for "memory behaves correctly
and predictably", which is the right trade given the bugs above.
"""

from smolagents.memory import TaskStep


def cap_agent_memory(agent, max_turns: int = 6) -> None:
    """Keep only the most recent `max_turns` conversation turns in
    `agent.memory.steps` — the same "edit the list directly" pattern
    smolagents' own tutorial uses for trimming memory, just dropping
    whole old turns instead of individual fields/screenshots.

    Each turn starts with a `TaskStep` (the user's message, or in Data
    Analysis's case the full task prompt) and is followed by whatever
    `ActionStep`/`PlanningStep`s the agent took to answer it. This drops
    whole old turns rather than truncating mid-turn, so memory always
    stays structurally valid for smolagents to replay.

    The agent's system prompt lives separately in
    `agent.memory.system_prompt` and is never touched by this.
    """
    steps = agent.memory.steps
    task_indices = [i for i, s in enumerate(steps) if isinstance(s, TaskStep)]
    if len(task_indices) > max_turns:
        cutoff = task_indices[-max_turns]
        agent.memory.steps = steps[cutoff:]


def reset_memory(agent) -> None:
    """Wipe an agent's own conversation memory in place, WITHOUT rebuilding
    the agent object itself.

    Calls smolagents' own `agent.memory.reset()` — see smolagents/memory.py:
    `AgentMemory.reset()` sets `self.steps = []` and explicitly leaves
    `self.system_prompt` (a separate `SystemPromptStep`, not part of
    `steps`) untouched. Using the library's own method here rather than
    reassigning `agent.memory.steps = []` by hand keeps this in step with
    whatever AgentMemory.reset() does internally in a future smolagents
    version, instead of this file quietly re-implementing (and
    potentially drifting from) that logic.

    Used whenever something other than an explicit model switch should
    invalidate memory — e.g. Data Analysis's "a new dataset was uploaded"
    check (see reset_if_context_changed() below). Clearing memory in
    place like this — rather than tearing down and rebuilding the whole
    CodeAgent, as an earlier version of this file did — is both cheaper
    (no tool re-initialization) and avoids the stale-memory-reload bug
    described in this module's docstring.
    """
    agent.memory.reset()


def reset_if_context_changed(agent, state: dict, new_key) -> bool:
    """Helper for tabs where a NEW context (e.g. a newly uploaded dataset)
    should invalidate old agent memory even though the underlying
    model/agent object hasn't changed — otherwise the agent would keep
    "remembering" a previous file's columns/stats while analyzing a
    different one.

    `state` is a small mutable dict (e.g. module-level `{"key": None}`)
    used to remember the last context key seen. Returns True if a reset
    was triggered.
    """
    if state.get("key") != new_key:
        reset_memory(agent)
        state["key"] = new_key
        return True
    return False
