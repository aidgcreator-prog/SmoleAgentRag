"""
model_registry.py — Central registry of model choices (LLM / VLM / STT /
visual retriever) shown in the UI dropdowns, plus shared path/size
constants, and the "rescan GGUF folder" action that rebuilds MODEL_OPTIONS
at runtime.
"""

from typing import Optional

import gradio as gr

import llama_backend
from i18n import LANGUAGES

# ──────────────────────────────────────────────────────────────────
# Shared constants
# ──────────────────────────────────────────────────────────────────
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
CHROMA_PERSIST_DIR  = "./chroma_db"
VISUAL_INDEX_DIR    = "./visual_index"
CHUNK_SIZE          = 1024
CHUNK_OVERLAP       = 128
TOP_K               = 4
MAX_NEW_TOKENS      = 512

DATA_ANALYSIS_DIR   = "./data_analysis"
DATA_UPLOAD_DIR      = f"{DATA_ANALYSIS_DIR}/uploads"
DATA_OUTPUT_DIR      = f"{DATA_ANALYSIS_DIR}/outputs"
DATA_AGENT_MAX_STEPS = 12

QWEN3_IDS    = {"Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B",
                "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B"}
QWEN36_IDS   = {"Qwen/Qwen3.6-27B", "Qwen/Qwen3.6-35B-A3B"}
ALL_QWEN_IDS = QWEN3_IDS | QWEN36_IDS
QWEN_VL_IDS  = {"Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"}
SMOL_VLM_IDS = {"HuggingFaceTB/SmolVLM-256M-Instruct", "HuggingFaceTB/SmolVLM-500M-Instruct",
                "HuggingFaceTB/SmolVLM2-2.2B-Instruct"}
ORNITH_IDS = {
    "deepreinforce-ai/Ornith-1.0-9B",
}

# ──────────────────────────────────────────────────────────────────
# LLM (HuggingFace + GGUF) options
# ──────────────────────────────────────────────────────────────────
BASE_MODEL_OPTIONS = {
    "🟢 Qwen3-0.6B   (~1.2 GB RAM | fastest)": "Qwen/Qwen3-0.6B",
    "🟡 Qwen3-1.7B   (~3 GB RAM)":             "Qwen/Qwen3-1.7B",
    "🟠 Qwen2.5-Coder-3B (~6 GB RAM | small coding/agent model)": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "🟡 Qwen3-4B     (~7 GB RAM)":             "Qwen/Qwen3-4B",
    "🔵 Gemma-4-E2B  (~4 GB RAM)":             "google/gemma-4-E2B-it",
}

# MODEL_OPTIONS starts as a copy of the base HuggingFace models. Any local
# .gguf models found under llama_backend.LLAMA_CPP_MODEL_DIR are merged in
# on top of it so they appear in the same dropdowns. The folder is
# user-configurable — via the LLAMA_CPP_MODEL_DIR environment variable at
# startup, or live from the "📁 GGUF Model Folder" box in the UI (see
# rescan_gguf_models() below). Kept as a single dict object that is mutated
# in place (never reassigned) so every module that imported it sees updates.
MODEL_OPTIONS = dict(BASE_MODEL_OPTIONS)
MODEL_OPTIONS.update(llama_backend.discover_gguf_models())

DEFAULT_LLM_LABEL = "🟢 Qwen3-0.6B   (~1.2 GB RAM | fastest)"
DEFAULT_LLM_MODEL = MODEL_OPTIONS[DEFAULT_LLM_LABEL]


def rescan_gguf_models(folder_path: Optional[str], lang_key: str = "kh"):
    """Set (or change) the GGUF model folder at runtime and rebuild
    MODEL_OPTIONS, without needing to edit code or restart the app.

    Returns (status_message, dropdown_update) repeated for every model
    dropdown in the UI so they all refresh with the newly discovered
    .gguf models immediately.
    """
    l = LANGUAGES.get(lang_key, LANGUAGES["kh"])
    folder_path = (folder_path or "").strip()

    # Persists to user_config.json and updates llama_backend.LLAMA_CPP_MODEL_DIR
    llama_backend.set_model_dir(folder_path)

    MODEL_OPTIONS.clear()
    MODEL_OPTIONS.update(BASE_MODEL_OPTIONS)

    if not folder_path:
        msg = l["gguf_scan_disabled"]
    elif not llama_backend.LLAMA_CPP_AVAILABLE:
        msg = l["gguf_scan_no_backend"]
    else:
        found = llama_backend.discover_gguf_models(folder_path)
        MODEL_OPTIONS.update(found)
        if found:
            msg = l["gguf_scan_found"].format(n=len(found), dir=folder_path)
        else:
            msg = l["gguf_scan_empty"].format(dir=folder_path)

    choices = list(MODEL_OPTIONS.keys())
    default_value = DEFAULT_LLM_LABEL if DEFAULT_LLM_LABEL in choices else (choices[0] if choices else None)
    dd_update = gr.update(choices=choices, value=default_value)
    return msg, dd_update, dd_update, dd_update


# ──────────────────────────────────────────────────────────────────
# Vision LLM (VLM) options
# ──────────────────────────────────────────────────────────────────
VLM_OPTIONS = {
    "🔵 SmolVLM-256M  (~0.5 GB RAM | tiny)":  "HuggingFaceTB/SmolVLM-256M-Instruct",
    "🔵 SmolVLM-500M  (~1 GB RAM | recommended)": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "🟢 Qwen2.5-VL-3B (~6 GB RAM)":           "Qwen/Qwen2.5-VL-3B-Instruct",
}
DEFAULT_VLM_LABEL = "🔵 SmolVLM-500M  (~1 GB RAM | recommended)"
DEFAULT_VLM_MODEL = VLM_OPTIONS[DEFAULT_VLM_LABEL]

# ──────────────────────────────────────────────────────────────────
# Visual retriever (ColPali-style, for visual PDF RAG) options
# ──────────────────────────────────────────────────────────────────
VISUAL_RETRIEVER_OPTIONS = {
    "vidore/colsmolvlm-v0.1  (~2 GB | recommended)": "vidore/colsmolvlm-v0.1",
    "vidore/colqwen2-v1.0    (~8 GB | higher accuracy)": "vidore/colqwen2-v1.0",
}
DEFAULT_VISUAL_RETRIEVER = "vidore/colsmolvlm-v0.1"

# ──────────────────────────────────────────────────────────────────
# Speech-to-Text (Whisper) options
# ──────────────────────────────────────────────────────────────────
STT_OPTIONS = {
    "🟢 Whisper-tiny    (~1 GB RAM | fastest)":   "openai/whisper-tiny",
    "🟡 Whisper-base    (~1 GB RAM)":              "openai/whisper-base",
    "🟡 Whisper-small   (~2 GB RAM | recommended)": "openai/whisper-small",
    "🔵 Whisper-large-v3 (~10 GB RAM | best accuracy, multilingual incl. Khmer)": "openai/whisper-large-v3",
    "🇰🇭 Whisper-small — ខ្មែរ (~1 GB RAM | Khmer-tuned)": "seanghay/whisper-small-khmer-v2",
    "🇰🇭 Whisper-large-v3-turbo — ខ្មែរ (~6 GB RAM | best for Khmer)": "metythorn/whisper-large-v3-turbo-mixed-20eps-clean-text-197k",
}
DEFAULT_STT_LABEL = "🟡 Whisper-small   (~2 GB RAM | recommended)"
DEFAULT_STT_MODEL = STT_OPTIONS[DEFAULT_STT_LABEL]
