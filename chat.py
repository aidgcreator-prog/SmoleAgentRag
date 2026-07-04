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
from hardware import DEVICE


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


def chat_general_direct(user_message: str, history: list, model_label: str):
    if not user_message.strip():
        return history, ""
    history = history or []
    history.append({"role": "user", "content": user_message})
    model_id = mr.MODEL_OPTIONS.get(model_label, mr.DEFAULT_LLM_MODEL)
    try:
        system = "You are a helpful, friendly assistant."
        ans, elapsed = models._call_llm(model_id, system, user_message)
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


def chat_general_agentic(user_message: str, history: list, model_label: str):
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
        t0     = time.time()
        result = agent.run(user_message)
        elapsed = time.time() - t0

        formatted = format_llm_response(str(result))
        response = (
            formatted +
            f"\n\n<hr><sub>⏱ {elapsed:.1f}s | model: <code>{model_id}</code> "
            f"({DEVICE.upper()}) | agentic · web search + code</sub>"
        )
    except Exception as e:
        import traceback
        response = f"❌ {e}\n\n{traceback.format_exc()}"
    history.append({"role": "assistant", "content": response})
    return history, ""


def chat_general(user_message: str, history: list, model_label: str, use_agentic: bool = False):
    """General Chat entry point. Dispatches to:

    - use_agentic=False (default, unchanged behaviour): one direct LLM
      call, no tools — see chat_general_direct().
    - use_agentic=True: a CodeAgent with web-search/webpage tools decides
      for itself whether to look things up — see chat_general_agentic().
      Same reliability caveat as agentic RAG Chat: works best with
      capable models (roughly Qwen3-4B and above); small models may never
      call a tool at all.
    """
    if use_agentic:
        return chat_general_agentic(user_message, history, model_label)
    return chat_general_direct(user_message, history, model_label)


def chat_rag_direct(user_message: str, history: list, model_label: str):
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
                f"Context:\n{context}"
            )
            ans, elapsed = models._call_llm(model_id, system, user_message)
            formatted = format_llm_response(ans)
            response = (
                formatted +
                f"\n\n<hr><sub>⏱ {elapsed:.1f}s | model: <code>{model_id}</code> "
                f"({DEVICE.upper()}) | direct RAG · sources: {', '.join(sources)}</sub>"
            )
    except Exception as e:
        import traceback
        response = f"❌ {e}\n\n{traceback.format_exc()}"
    history.append({"role": "assistant", "content": response})
    return history, ""


def chat_rag(user_message: str, history: list, model_label: str, use_agentic: bool = True):
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
    """
    if not use_agentic:
        return chat_rag_direct(user_message, history, model_label)

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
        result = agent.run(task)
        elapsed = time.time() - t0

        call_count, found_count = rag_agent.get_retriever_stats()
        if call_count == 0:
            ans = rag_agent.NEVER_SEARCHED_MESSAGE
        elif found_count == 0:
            ans = rag_agent.NOTHING_FOUND_MESSAGE
        else:
            ans = str(result)

        formatted = format_llm_response(ans)
        response = (
            formatted +
            f"\n\n<hr><sub>⏱ {elapsed:.1f}s | model: <code>{model_id}</code> "
            f"({DEVICE.upper()}) | agentic RAG · strict grounding "
            f"(searched {call_count}x)</sub>"
        )
    except Exception as e:
        import traceback
        response = f"❌ {e}\n\n{traceback.format_exc()}"
    history.append({"role": "assistant", "content": response})
    return history, ""


def chat_vision(user_message: str, uploaded_image, history: list,
                vlm_label: str, use_visual_rag: bool):
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
