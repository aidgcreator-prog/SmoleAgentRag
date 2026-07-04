"""
chat.py — Chat-turn handlers for the General Chat, RAG Chat, and Vision
Chat tabs, plus the <think>-tag reasoning/answer formatter shared by the
text tabs.
"""

import re
import time

from PIL import Image

import knowledge_base as kb
import model_registry as mr
import models
import general_agent
import rag_agent
import agent_memory
from hardware import DEVICE

# NOTE: agent_memory.save_agent_memory() persists to disk under the tab
# key constants general_agent.MEMORY_TAB_KEY / rag_agent.MEMORY_TAB_KEY —
# see agent_memory.py's module docstring for why memory is namespaced per
# (tab, model_id) rather than shared/replayed across model switches.

# How many prior user+assistant exchanges to feed back into the *direct*
# (non-agentic) chat paths as short-term conversation memory. Agentic
# paths don't use this — they get real memory from smolagents' own
# agent.memory (see general_agent.py / rag_agent.py / agent_memory.py).
DIRECT_CHAT_MEMORY_TURNS = 6

# How many turns of memory an agentic CodeAgent is allowed to keep before
# older turns get dropped (see agent_memory.cap_agent_memory()).
AGENTIC_MEMORY_TURNS = 6


def format_llm_response(text: str) -> str:
    m = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)

    if not m:
        return text

    thinking = m.group(1).strip()
    answer = m.group(2).strip()

    return f"""
<details style="
margin-bottom:12px;
border:1px solid #555;
border-radius:8px;
background:#2d2d2d;
padding:10px;
">
<summary style="
cursor:pointer;
font-weight:bold;
color:#ffcc66;
">
🧠 Reasoning (click to expand)
</summary>

<div style="
margin-top:10px;
color:#cfcfcf;
font-family:monospace;
white-space:pre-wrap;
line-height:1.5;
">
{thinking}
</div>

</details>

<div style="
border-left:5px solid #4CAF50;
padding:12px;
background:#1f1f1f;
border-radius:8px;
font-size:16px;
line-height:1.6;
">

<b>💬 Answer</b>

{answer}

</div>
"""


def _strip_response_html(text: str) -> str:
    """Recover the plain-text answer from a previously HTML-formatted
    assistant reply (format_llm_response() wraps it in a collapsible
    <think> accordion plus an <hr><sub>...timing/model footer</sub>) so it
    can be fed back to the LLM as conversation memory without leaking
    markup, stale timing numbers, or its own past "reasoning" commentary.
    """
    if not text:
        return text
    text = re.sub(r"<details.*?</details>", "", text, flags=re.DOTALL)
    text = re.sub(r"<hr>.*$", "", text, flags=re.DOTALL)
    m = re.search(r"<b>\U0001F4AC Answer</b>\s*(.*?)</div>", text, flags=re.DOTALL)
    if m:
        text = m.group(1)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def _recent_memory_messages(history: list, max_turns: int = DIRECT_CHAT_MEMORY_TURNS) -> list:
    """Build a clean list of prior {"role", "content"} turns from the
    Gradio chatbot history, to hand to models._call_llm() as short-term
    conversation memory for the *direct* (non-agentic) chat paths.

    Keeps only the last `max_turns` user+assistant exchanges and strips
    HTML formatting from assistant replies (see _strip_response_html())
    so old markup/timing footers don't pollute the prompt.
    """
    if not history:
        return []
    cleaned = []
    for turn in history:
        role, content = turn.get("role"), turn.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "assistant":
            content = _strip_response_html(content)
            if not content:
                continue
        elif role != "user":
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned[-(max_turns * 2):]


def chat_general_direct(user_message: str, history: list, model_label: str, use_memory: bool = True):
    if not user_message.strip():
        return history, ""
    history = history or []
    history.append({"role": "user", "content": user_message})
    model_id = mr.MODEL_OPTIONS.get(model_label, mr.DEFAULT_LLM_MODEL)
    try:
        system = "You are a helpful, friendly assistant."
        # Memory: feed back the recent conversation (minus the message we
        # just appended above, which is passed separately as `user`) so
        # follow-ups like "and what about X?" have something to refer to.
        # Skipped entirely when the "🧠 Conversation Memory" checkbox is off.
        memory_messages = _recent_memory_messages(history[:-1]) if use_memory else []
        ans, elapsed = models._call_llm(model_id, system, user_message, history=memory_messages)
        formatted = format_llm_response(ans)

        response = (
            formatted +
            f"\n\n<hr><sub>⏱ {elapsed:.1f}s | model: <code>{model_id}</code> ({DEVICE.upper()})</sub>"
        )

    except Exception as e:
        import traceback
        response = f"❌ {e}\n\n{traceback.format_exc()}"
    history.append({"role": "assistant", "content": response})
    return history, ""


def chat_general_agentic(user_message: str, history: list, model_label: str, use_memory: bool = True):
    """Agentic General Chat: a smolagents CodeAgent (see general_agent.py)
    with the library's own built-in web-search/webpage tools
    (DuckDuckGoSearchTool, VisitWebpageTool), deciding for itself whether
    a question needs a web lookup before answering. Unlike RAG Chat's
    agent, this has no knowledge-base grounding requirement — it's meant
    for open-ended questions, not strictly-cited document Q&A.
    """
    if not user_message.strip():
        return history, ""
    history = history or []
    history.append({"role": "user", "content": user_message})
    model_id = mr.MODEL_OPTIONS.get(model_label, mr.DEFAULT_LLM_MODEL)
    try:
        agent = general_agent.get_general_agent(model_id)
        general_agent.reset_tool_usage()
        t0     = time.time()
        # reset=False keeps this CodeAgent's own memory (agent.memory.steps)
        # across turns instead of wiping it every message — see
        # https://huggingface.co/docs/smolagents/tutorials/memory. Capped
        # right after so a long conversation doesn't grow the prompt
        # unboundedly turn after turn. When the "🧠 Conversation Memory"
        # checkbox is off, reset=True instead, so every message starts
        # from a clean slate (no memory to cap either).
        result = agent.run(user_message, reset=not use_memory)
        if use_memory:
            agent_memory.cap_agent_memory(agent, max_turns=AGENTIC_MEMORY_TURNS)
            # Persist to disk too — see agent_memory.py's module docstring.
            agent_memory.save_agent_memory(agent, general_agent.MEMORY_TAB_KEY, model_id)
        elapsed = time.time() - t0

        turn_count = sum(1 for s in agent.memory.steps if s.__class__.__name__ == "TaskStep")
        ans = str(result)
        # Guaranteed-accurate references list, appended regardless of
        # whether the model remembered its own in-text citations/
        # "### References" section (general_agent.GENERAL_AGENT_INSTRUCTIONS
        # asks it to, but — like RAG's strict grounding — that depends on
        # the model's instruction-following and can't be fully trusted).
        # Built from what the tracked tools ACTUALLY searched/visited this
        # turn — see general_agent.TrackedDuckDuckGoSearchTool /
        # TrackedVisitWebpageTool.
        queries, urls = general_agent.get_tool_usage()
        if urls or queries:
            ref_lines = []
            if urls:
                ref_lines.append("**🔗 Pages consulted this turn:**")
                ref_lines.extend(f"- {u}" for u in urls)
            if queries:
                ref_lines.append("**🔍 Searches run this turn:**")
                ref_lines.extend(f"- \"{q}\"" for q in queries)
            ans += "\n\n---\n" + "\n".join(ref_lines)
        formatted = format_llm_response(ans)
        response = (
            formatted +
            f"\n\n<hr><sub>⏱ {elapsed:.1f}s | model: <code>{model_id}</code> "
            f"({DEVICE.upper()}) | agentic · web search + code · memory: "
            f"{'on (' + str(turn_count) + ' turn(s))' if use_memory else 'off'}</sub>"
        )
    except Exception as e:
        import traceback
        response = f"❌ {e}\n\n{traceback.format_exc()}"
    history.append({"role": "assistant", "content": response})
    return history, ""


def chat_general(user_message: str, history: list, model_label: str, use_agentic: bool = False, use_memory: bool = True):
    """General Chat entry point. Dispatches to:

    - use_agentic=False (default, unchanged behaviour): one direct LLM
      call, no tools — see chat_general_direct().
    - use_agentic=True: a CodeAgent with web-search/webpage tools decides
      for itself whether to look things up — see chat_general_agentic().
      Same reliability caveat as agentic RAG Chat: works best with
      capable models (roughly Qwen3-4B and above); small models may never
      call a tool at all.

    `use_memory` controls the "🧠 Conversation Memory" checkbox: whether
    prior turns in this conversation are remembered (direct path) or kept
    in the CodeAgent's own memory (agentic path). Memory is never
    persisted to disk either way — see agent_memory.py.
    """
    if use_agentic:
        return chat_general_agentic(user_message, history, model_label, use_memory)
    return chat_general_direct(user_message, history, model_label, use_memory)


def chat_rag_direct(user_message: str, history: list, model_label: str, use_memory: bool = True):
    """Non-agentic RAG: always retrieve context from ChromaDB first (same
    retrieval call the agentic path's `retriever` tool wraps), then hand
    that context straight to the LLM in one direct call — no CodeAgent,
    no tool-calling, no code-generation steps to parse.

    This exists because tool-calling reliability depends heavily on the
    underlying model (see rag_agent.py's caveat about Qwen3-0.6B): small
    or older models often never call the retriever tool, call it
    malformed, or burn every retry step failing to parse, and end up
    hallucinating or timing out instead of answering. Retrieval here is
    unconditional and handled entirely by the app, so even a very small
    model just has to read the provided context and answer — nothing to
    invoke, nothing to parse.
    """
    if not user_message.strip():
        return history, ""
    history = history or []
    history.append({"role": "user", "content": user_message})
    model_id = mr.MODEL_OPTIONS.get(model_label, mr.DEFAULT_LLM_MODEL)
    try:
        context, sources = kb.retrieve_context(user_message)
        if not context:
            ans = rag_agent.NOTHING_FOUND_MESSAGE
            formatted = format_llm_response(ans)
            response = (
                formatted +
                f"\n\n<hr><sub>model: <code>{model_id}</code> ({DEVICE.upper()}) "
                f"| direct RAG (no relevant context found)</sub>"
            )
        else:
            system = (
                "You are a strict retrieval-augmented assistant. Answer the "
                "user's question using ONLY the context below, which was "
                "retrieved from the user's own indexed knowledge base. Never "
                "use your own general knowledge or training data, even if you "
                "believe you know the answer. If the context doesn't contain "
                "the answer, say clearly that the knowledge base doesn't "
                "contain this information — do not guess or fill the gap "
                "yourself.\n\n"
                "CITATION REQUIREMENT: each context chunk below is tagged "
                "with its source in brackets, e.g. '[report.pdf]'. Cite that "
                "exact bracketed tag in-text immediately after every claim "
                "you make from it, and finish your answer with a "
                "'### References' section listing every distinct source you "
                "cited.\n\n"
                f"Context:\n{context}"
            )
            # Memory: same short-term conversation recall as General Chat's
            # direct path, so a follow-up question can refer back to what
            # was just discussed, on top of the always-fresh retrieval above.
            # Skipped entirely when the "🧠 Conversation Memory" checkbox is off.
            memory_messages = _recent_memory_messages(history[:-1]) if use_memory else []
            ans, elapsed = models._call_llm(model_id, system, user_message, history=memory_messages)
            # Guaranteed-accurate references list, appended regardless of
            # whether the model remembered its own in-text citations/
            # "### References" section — built from the sources ChromaDB
            # ACTUALLY returned for this query (not trusting the model to
            # report them correctly itself).
            if sources:
                refs = "\n".join(f"- {s}" for s in sorted(sources))
                ans += f"\n\n---\n**📚 Sources retrieved this turn:**\n{refs}"
            formatted = format_llm_response(ans)
            response = (
                formatted +
                f"\n\n<hr><sub>⏱ {elapsed:.1f}s | model: <code>{model_id}</code> "
                f"({DEVICE.upper()}) | direct RAG</sub>"
            )
    except Exception as e:
        import traceback
        response = f"❌ {e}\n\n{traceback.format_exc()}"
    history.append({"role": "assistant", "content": response})
    return history, ""


def chat_rag(user_message: str, history: list, model_label: str, use_agentic: bool = True, use_memory: bool = True):
    """RAG Chat entry point. Dispatches to one of two retrieval strategies:

    - use_agentic=True (default): a smolagents CodeAgent (see rag_agent.py)
      decides for itself whether/when to call the `retriever` tool against
      ChromaDB — and can call it more than once to refine its search —
      before writing a final answer. Strictly grounded: the agent is
      instructed (system + per-task level) to answer ONLY from retrieved
      content, and the answer is also checked after the fact against the
      RetrieverTool's own self-tracked call/found counts
      (rag_agent.get_retriever_stats()). If it never searched, or searched
      and found nothing relevant, the answer is replaced with a clear
      "not in the knowledge base" message instead of trusting whatever the
      model wrote. Best with capable models (roughly Qwen3-4B and above) —
      small models often fumble the tool-calling/code-parsing steps.

    - use_agentic=False: see chat_rag_direct() — always retrieves context
      first, then asks the LLM directly in one call. No tool-calling
      required, so small/older/weaker models (e.g. Qwen3-0.6B) can use RAG
      reliably too, at the cost of never refining or skipping the search.

    `use_memory` controls the "🧠 Conversation Memory" checkbox — see
    chat_general()'s docstring for what it does on each path.
    """
    if not use_agentic:
        return chat_rag_direct(user_message, history, model_label, use_memory)

    if not user_message.strip():
        return history, ""
    history = history or []
    history.append({"role": "user", "content": user_message})
    model_id = mr.MODEL_OPTIONS.get(model_label, mr.DEFAULT_LLM_MODEL)
    try:
        agent = rag_agent.get_rag_agent(model_id)
        rag_agent.reset_retriever_stats()
        task = rag_agent.build_strict_task(user_message)

        t0     = time.time()
        # reset=False keeps this CodeAgent's memory across turns — see
        # https://huggingface.co/docs/smolagents/tutorials/memory — capped
        # right after via agent_memory.cap_agent_memory() so the prompt
        # doesn't grow without bound over a long conversation. reset=True
        # (memory checkbox off) makes every message stateless instead.
        result = agent.run(task, reset=not use_memory)
        if use_memory:
            agent_memory.cap_agent_memory(agent, max_turns=AGENTIC_MEMORY_TURNS)
            # Persist to disk too — see agent_memory.py's module docstring.
            agent_memory.save_agent_memory(agent, rag_agent.MEMORY_TAB_KEY, model_id)
        elapsed = time.time() - t0

        call_count, found_count, sources_used = rag_agent.get_retriever_stats()
        if call_count == 0:
            ans = rag_agent.NEVER_SEARCHED_MESSAGE
        elif found_count == 0:
            ans = rag_agent.NOTHING_FOUND_MESSAGE
        else:
            ans = str(result)
            # Guaranteed-accurate references list, appended regardless of
            # whether the model remembered its own in-text citations/
            # "### References" section (rag_agent.STRICT_SYSTEM_INSTRUCTIONS
            # asks it to, but that depends on the model's instruction-
            # following, same caveat as the strict-grounding check above).
            # Built from sources the retriever ACTUALLY returned this turn
            # — see knowledge_base.RetrieverTool.sources_used.
            if sources_used:
                refs = "\n".join(f"- {s}" for s in sorted(sources_used))
                ans += f"\n\n---\n**📚 Sources retrieved this turn:**\n{refs}"

        formatted = format_llm_response(ans)
        response = (
            formatted +
            f"\n\n<hr><sub>⏱ {elapsed:.1f}s | model: <code>{model_id}</code> "
            f"({DEVICE.upper()}) | agentic RAG · strict grounding "
            f"(searched {call_count}x) · memory: {'on' if use_memory else 'off'}</sub>"
        )
    except Exception as e:
        import traceback
        response = f"❌ {e}\n\n{traceback.format_exc()}"
    history.append({"role": "assistant", "content": response})
    return history, ""


def chat_vision(user_message: str, uploaded_image, history: list,
                vlm_label: str, use_visual_rag: bool, use_memory: bool = True):
    if not user_message.strip() and uploaded_image is None:
        return history, None
    history = history or []
    history.append({"role": "user", "content": user_message or "(image)"})
    vlm_id = mr.VLM_OPTIONS.get(vlm_label, mr.DEFAULT_VLM_MODEL)
    try:
        # Pre-load VLM outside vlm_answer so errors surface cleanly
        models.get_vlm(vlm_id)
        pil_images = []
        if uploaded_image is not None:
            if isinstance(uploaded_image, Image.Image):
                pil_images.append(uploaded_image)
            elif isinstance(uploaded_image, str):
                pil_images.append(Image.open(uploaded_image).convert("RGB"))
        if use_visual_rag and user_message:
            pil_images.extend(kb.visual_retrieve(user_message, top_k=2))
        context, _ = kb.retrieve_context(user_message) if user_message else ("", [])
        # Memory: Vision Chat has no smolagents CodeAgent (and images
        # aren't replayed turn-to-turn to keep prompts small), but folding
        # the last couple of Q&A exchanges' *text* in as extra context lets
        # simple follow-ups ("what color is it?" after "what's in the
        # photo?") still make sense without re-uploading the image.
        recent = _recent_memory_messages(history[:-1], max_turns=3) if use_memory else []
        if recent:
            convo = "\n".join(f"{t['role']}: {t['content']}" for t in recent)
            context = f"Recent conversation:\n{convo}\n\n{context}" if context else f"Recent conversation:\n{convo}"
        t0  = time.time()
        ans = models.vlm_answer(user_message, pil_images, context=context, model_id=vlm_id)
        elapsed  = time.time() - t0
        response = (f"{ans}\n\n*⏱ {elapsed:.1f}s | VLM: `{vlm_id}` ({DEVICE.upper()})"
                    f" | images: {len(pil_images)}*")
    except Exception as e:
        import traceback
        response = f"❌ {e}\n\n{traceback.format_exc()}"
    history.append({"role": "assistant", "content": response})
    return history, None
