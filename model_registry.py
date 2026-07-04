"""
model_registry.py — Central registry of model choices (LLM / VLM / STT /
visual retriever) shown in the UI dropdowns, plus shared path/size
constants, and the "rescan GGUF folder" action that rebuilds MODEL_OPTIONS
at runtime.
"""

from typing import Optional

import gradio as gr

import llama_backend
import user_config
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
DATA_AGENT_MAX_STEPS = 20

# ──────────────────────────────────────────────────────────────────
# Context window (n_ctx) — how many tokens of prompt+history+generation
# the LLM can hold at once. Mainly meaningful for the GGUF/llama.cpp
# backend (llama_backend.LlamaCppModel), where it's a fixed size set at
# load time — HF/transformers models size their own context from the
# checkpoint's trained max_position_embeddings and aren't affected by
# this setting.
#
# Bumping this fixes errors like:
#   "Requested tokens (16461) exceed context window of 16384"
# which happens once an agentic conversation's accumulated prompt +
# memory + tool-call history grows past whatever n_ctx the model was
# loaded with (16384 tokens, previously hardcoded in models.py). A
# larger window uses more VRAM/RAM (roughly linearly with n_ctx), so
# this is exposed as a user choice rather than just maxed out by default.
#
# Persisted the same way as llama_backend's GGUF folder (see
# user_config.py) so the choice survives an app restart.
# ──────────────────────────────────────────────────────────────────
CONTEXT_WINDOW_OPTIONS = {
    "4K   (4,096 tokens — lowest memory)":            4096,
    "8K   (8,192 tokens)":                            8192,
    "16K  (16,384 tokens — default)":                 16384,
    "32K  (32,768 tokens)":                           32768,
    "64K  (65,536 tokens — needs more VRAM/RAM)":     65536,
    "128K (131,072 tokens — needs a lot of VRAM/RAM)": 131072,
}
DEFAULT_CONTEXT_WINDOW_LABEL = "16K  (16,384 tokens — default)"
DEFAULT_CONTEXT_WINDOW = CONTEXT_WINDOW_OPTIONS[DEFAULT_CONTEXT_WINDOW_LABEL]


def get_saved_context_window() -> int:
    """Read the persisted context window (n_ctx), falling back to the
    default if nothing was ever saved (fresh install) or the saved value
    is corrupt/not one of our known sizes anymore."""
    try:
        n = int(user_config.USER_CONFIG.get("context_window", DEFAULT_CONTEXT_WINDOW))
        return n if n > 0 else DEFAULT_CONTEXT_WINDOW
    except (TypeError, ValueError):
        return DEFAULT_CONTEXT_WINDOW


def get_saved_context_window_label() -> str:
    """Reverse-lookup the dropdown label matching the persisted n_ctx, for
    initializing the UI dropdown's value to whatever was saved last time."""
    saved = get_saved_context_window()
    for label, value in CONTEXT_WINDOW_OPTIONS.items():
        if value == saved:
            return label
    return DEFAULT_CONTEXT_WINDOW_LABEL


def set_context_window(n_ctx: int) -> None:
    """Persist the chosen context window so it survives an app restart —
    mirrors llama_backend.set_model_dir()'s persistence pattern."""
    user_config.save_user_config({"context_window": int(n_ctx)})

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
    MODEL_OPTIONS (text LLMs) and VLM_OPTIONS (vision-model pairs),
    without needing to edit code or restart the app.

    Returns (status_message, text_dropdown_update x3, vlm_dropdown_update)
    so every model dropdown in the UI (General/RAG/Data-Analysis LLM
    pickers, plus the Vision LLM picker) refreshes with the newly
    discovered .gguf models immediately.
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
        vlm_found_count = len(llama_backend.discover_gguf_vlm_models(folder_path))
        if found or vlm_found_count:
            msg = l["gguf_scan_found"].format(n=len(found), dir=folder_path)
            if vlm_found_count:
                msg += f" (+ {vlm_found_count} vision model pair(s) found for the 🎨 Vision LLM dropdown)"
        else:
            msg = l["gguf_scan_empty"].format(dir=folder_path)

    # Vision GGUF pairs (main model + mmproj) are scanned/rebuilt
    # independently of the plain text-model branch above —
    # _rebuild_gguf_vlm_options() already no-ops safely on an
    # empty/invalid folder or a missing llama-cpp-python install.
    _rebuild_gguf_vlm_options(folder_path)

    choices = list(MODEL_OPTIONS.keys())
    default_value = DEFAULT_LLM_LABEL if DEFAULT_LLM_LABEL in choices else (choices[0] if choices else None)
    dd_update = gr.update(choices=choices, value=default_value)

    vlm_choices = list(VLM_OPTIONS.keys())
    vlm_default = DEFAULT_VLM_LABEL if DEFAULT_VLM_LABEL in vlm_choices else (vlm_choices[0] if vlm_choices else None)
    vlm_dd_update = gr.update(choices=vlm_choices, value=vlm_default)

    return msg, dd_update, dd_update, dd_update, vlm_dd_update


# ──────────────────────────────────────────────────────────────────
# Vision LLM (VLM) options
# ──────────────────────────────────────────────────────────────────
BASE_VLM_OPTIONS = {
    "🔵 SmolVLM-256M  (~0.5 GB RAM | tiny)":  "HuggingFaceTB/SmolVLM-256M-Instruct",
    "🔵 SmolVLM-500M  (~1 GB RAM | recommended)": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "🟢 Qwen2.5-VL-3B (~6 GB RAM)":           "Qwen/Qwen2.5-VL-3B-Instruct",
}

# VLM_OPTIONS starts as a copy of the base HuggingFace VLMs. Any local
# GGUF vision-model pairs (main .gguf + matching mmproj .gguf) found
# under llama_backend.LLAMA_CPP_MODEL_DIR are merged in on top of it —
# same pattern as MODEL_OPTIONS for text LLMs — so they appear in the
# same "🎨 Vision LLM" dropdown. Kept as a single dict object that is
# mutated in place (never reassigned) so every module that imported it
# sees updates.
VLM_OPTIONS = dict(BASE_VLM_OPTIONS)
DEFAULT_VLM_LABEL = "🔵 SmolVLM-500M  (~1 GB RAM | recommended)"
DEFAULT_VLM_MODEL = VLM_OPTIONS[DEFAULT_VLM_LABEL]

# Maps a GGUF vision model's main .gguf path -> its paired mmproj (vision
# projector) .gguf path (see llama_backend.discover_gguf_vlm_models()).
# models.get_vlm() looks this up by model_id, since llama.cpp needs BOTH
# files' paths to actually load a GGUF vision model — unlike text-only
# GGUF models, which need just the one file.
GGUF_VLM_MMPROJ_MAP = {}


def _rebuild_gguf_vlm_options(folder: Optional[str] = None):
    """(Re)populate VLM_OPTIONS with discovered GGUF vision-model pairs
    from `folder` (or the current LLAMA_CPP_MODEL_DIR) and refresh
    GGUF_VLM_MMPROJ_MAP to match. Mutates both dicts in place (never
    reassigns), same pattern as MODEL_OPTIONS's text-LLM rescan."""
    VLM_OPTIONS.clear()
    VLM_OPTIONS.update(BASE_VLM_OPTIONS)
    GGUF_VLM_MMPROJ_MAP.clear()
    for label, (model_path, mmproj_path) in llama_backend.discover_gguf_vlm_models(folder).items():
        VLM_OPTIONS[label] = model_path
        GGUF_VLM_MMPROJ_MAP[model_path] = mmproj_path


_rebuild_gguf_vlm_options()

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
