"""
model_registry.py — Central registry of model choices (LLM / VLM / STT /
visual retriever) shown in the UI dropdowns, plus shared path/size
constants, and the "rescan GGUF folder" action that rebuilds MODEL_OPTIONS
at runtime.
"""

import re
from typing import Optional

import gradio as gr

import llama_backend
import user_config
from hardware import HardwareManager
from i18n import LANGUAGES

# ──────────────────────────────────────────────────────────────────
# Shared constants
# ──────────────────────────────────────────────────────────────────
# Embedding model — dynamic default based on detected hardware tier,
# mirroring the "Embedding Model" column of README.md's "Model combos by
# hardware tier" table exactly:
#   CPU-only / 8GB / 16GB VRAM -> BGE-M3     (small, works everywhere)
#   24GB VRAM                  -> Qwen3-Embedding-4B
#   48GB+ VRAM                 -> Jina Embeddings v4
# Unknown hardware (detection failed) falls back to BGE-M3 — the safest
# default, since guessing a larger embedding model on unknown hardware
# risks an OOM on the very first index/query call.
#
# This only changes the DEFAULT — like context_window (see
# get_saved_context_window() below), a value the user explicitly saved
# to user_config.json always wins over the hardware-detected default, so
# switching hardware tiers never silently overrides a deliberate choice.
# ──────────────────────────────────────────────────────────────────
EMBED_OPTIONS = {
    "BGE-M3 (~2 GB RAM | multilingual, recommended default)": "BAAI/bge-m3",
    "Qwen3-Embedding-4B (~8 GB RAM | 24GB+ VRAM tier)":        "Qwen/Qwen3-Embedding-4B",
    "Jina Embeddings v4 (~qwen3-based | 48GB+ VRAM tier)":     "jinaai/jina-embeddings-v4",
}

_EMBED_MODEL_BY_TIER = {
    HardwareManager.TIER_48GB_VRAM: "jinaai/jina-embeddings-v4",
    HardwareManager.TIER_24GB_VRAM: "Qwen/Qwen3-Embedding-4B",
    HardwareManager.TIER_16GB_VRAM: "BAAI/bge-m3",
    HardwareManager.TIER_8GB_VRAM:  "BAAI/bge-m3",
    HardwareManager.TIER_CPU_ONLY:  "BAAI/bge-m3",
    HardwareManager.TIER_UNKNOWN:   "BAAI/bge-m3",
}


def get_recommended_embed_model() -> str:
    """The embedding model README.md's hardware-tier table recommends for
    THIS machine, based on live-detected VRAM/RAM (see
    hardware.HardwareManager.detect_hardware_tier()). Falls back to
    BGE-M3 if the tier can't be determined."""
    tier = HardwareManager.detect_hardware_tier()
    return _EMBED_MODEL_BY_TIER.get(tier, "BAAI/bge-m3")


def get_default_embed_model() -> str:
    """The embedding model actually used unless the user has a saved
    override in user_config.json — a persisted choice always wins (same
    pattern as get_saved_context_window()), so re-detecting hardware on
    every restart never silently reverts an explicit user pick."""
    saved = user_config.USER_CONFIG.get("embed_model")
    if saved:
        return saved
    return get_recommended_embed_model()


def set_embed_model(model_id: str) -> None:
    """Persist an explicit embedding-model choice — mirrors
    llama_backend.set_model_dir() / set_context_window()'s persistence
    pattern. NOTE: changing this only takes effect on the next embedding
    model load (models.get_embed_model() caches the loaded model); if one
    is already loaded, the caller is responsible for resetting
    models._embed_model, since an embedding model can't be hot-swapped
    mid-session without also fully re-indexing every stored vector in
    ChromaDB (embeddings from different models aren't comparable)."""
    user_config.save_user_config({"embed_model": model_id})


DEFAULT_EMBED_MODEL = get_default_embed_model()
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


# ──────────────────────────────────────────────────────────────────
# Agentic CodeAgent step budget — scaled down for larger/slower local
# models. A broken step-parsing loop (e.g. a model that writes plain
# prose instead of a ```python fenced block) costs roughly the same
# wall-clock time PER STEP regardless of model size, but a 14B+ GGUF
# model can take 30-250+ seconds per step where a small HF model takes
# a few seconds — so the same default max_steps that's a harmless safety
# net on a small model turns into a 4-20+ minute stall on a large one
# before the agent finally gives up. See general_agent.py / rag_agent.py
# for where this is applied.
# ──────────────────────────────────────────────────────────────────
_PARAM_SIZE_RE = re.compile(r'(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])')


def estimate_model_param_billions(model_id: str) -> Optional[float]:
    """Best-effort parse of a model's parameter count (in billions) from
    its id/filename, e.g.:
      'Qwen3.6-14B-A3B-FableVibes-Q8_0.gguf'      -> 14.0
      'qwen3-coder-30b-a3b-compacted-19b-256k...' -> 30.0 (first match wins)
      'Qwen/Qwen3-0.6B'                           -> 0.6
    Returns None if no confident '<number>B' pattern is found (e.g. an
    unusually-named checkpoint) — callers should treat that as "unknown
    size", not "small", since guessing wrong in the small direction would
    silently remove the safety margin this exists to add.
    """
    if not model_id:
        return None
    m = _PARAM_SIZE_RE.search(str(model_id))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def get_max_steps_for_model(model_id: str, default_max_steps: int) -> int:
    """Scale an agentic CodeAgent's max_steps down for models estimated to
    be 10B+ parameters, so a broken parsing/tool-calling loop on a large,
    slow local GGUF model fails fast instead of grinding through the full
    default step budget (each failed step can take a minute or more on
    these models — see the module docstring above). Leaves
    `default_max_steps` untouched if the model's size can't be confidently
    parsed from its id/filename, since an unrecognized name is more often
    a normal HuggingFace repo id (usually small/fast) than a huge unnamed
    checkpoint, and it's safer to keep the normal budget than cut off a
    model that turns out to be small.
    """
    size_b = estimate_model_param_billions(model_id)
    if size_b is None:
        return default_max_steps
    if size_b >= 20:
        return min(default_max_steps, 4)
    if size_b >= 10:
        return min(default_max_steps, 5)
    return default_max_steps

QWEN3_IDS    = {"Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B",
                "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B"}
# Qwen3.6 (April 2026) — hybrid linear-/full-attention MoE ("qwen3_5_moe")
# and dense architectures, superseding Qwen3.5. Only ships in 27B (dense)
# and 35B-A3B (MoE, ~3B active params/token) sizes — there is no small
# (<20B) Qwen3.6 checkpoint, so BASE_MODEL_OPTIONS below still uses plain
# Qwen3 for the 0.6B/1.7B/4B/8B/14B tiers, and only swaps in Qwen3.6 at
# the top end (replacing the older Qwen3-32B).
QWEN36_IDS   = {"Qwen/Qwen3.6-27B", "Qwen/Qwen3.6-35B-A3B"}
ALL_QWEN_IDS = QWEN3_IDS | QWEN36_IDS
QWEN_VL_IDS  = {"Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"}
SMOL_VLM_IDS = {"HuggingFaceTB/SmolVLM-256M-Instruct", "HuggingFaceTB/SmolVLM-500M-Instruct",
                "HuggingFaceTB/SmolVLM2-2.2B-Instruct"}
ORNITH_IDS = {
    "deepreinforce-ai/Ornith-1.0-9B",
}
# Gemma 4 (all sizes) needs transformers >= 5.10.1 — see
# models._MIN_TRANSFORMERS_VERSION["gemma4"] for the guard that checks
# this before load. Includes google/gemma-4-E2B-it, the app's own
# pre-existing default LLM entry — it was previously unguarded (silently
# unloadable on a stock transformers>=4.51.0 install per requirements.txt)
# until this set + guard were added.
GEMMA4_IDS = {
    "google/gemma-4-E2B-it",
    "google/gemma-4-12B-it",
    "google/gemma-4-26B-A4B-it",
    "google/gemma-4-31B-it",
}

# ──────────────────────────────────────────────────────────────────
# LLM (HuggingFace + GGUF) options
# ──────────────────────────────────────────────────────────────────
BASE_MODEL_OPTIONS = {
    "🟢 Qwen3-0.6B   (~1.2 GB RAM | fastest)": "Qwen/Qwen3-0.6B",
    "🟡 Qwen3-1.7B   (~3 GB RAM)":             "Qwen/Qwen3-1.7B",
    "🟠 Qwen2.5-Coder-3B (~6 GB RAM | small coding/agent model)": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "🟡 Qwen3-4B     (~7 GB RAM)":             "Qwen/Qwen3-4B",
    "🔵 Gemma-4-E2B  (~4 GB RAM | needs transformers>=5.10.1)": "google/gemma-4-E2B-it",
    # Registered from README.md's "Model combos by hardware tier" table —
    # these are the HF/transformers-loadable equivalents of that table's
    # GGUF recommendations (same checkpoints, full BF16 precision instead
    # of a quantized .gguf file — so actual RAM/VRAM use is higher than
    # the README's GGUF-quantized figures for the same model name).
    # Qwen3 dense sizes — same "qwen3" architecture as the existing
    # Qwen3-0.6B/1.7B/4B above, already covered by this app's pinned
    # transformers>=4.51.0 (see requirements.txt) — no new version guard
    # needed, unlike Qwen3.6/Gemma 4 below. No Qwen3.6 checkpoint exists
    # at these sizes yet, so these stay on plain Qwen3.
    "🟠 Qwen3-8B     (~16 GB RAM | 8GB-VRAM tier)":  "Qwen/Qwen3-8B",
    "🔴 Qwen3-14B    (~30 GB RAM | 16GB-VRAM tier)": "Qwen/Qwen3-14B",
    # Qwen3.6 (April 2026, newest Qwen generation) — replaces the older
    # Qwen3-32B at the top of the dense lineup, and adds the 35B-A3B MoE
    # variant (only ~3B active params/token, so it stays fast even on
    # CPU-only rigs — see README's hardware-tier table). Needs
    # transformers>=5.2.0 — see models._MIN_TRANSFORMERS_VERSION["qwen36"].
    "🔴 Qwen3.6-27B   (~55 GB RAM | 24GB+/48GB+-VRAM tier | needs transformers>=5.2.0)": "Qwen/Qwen3.6-27B",
    "🔴 Qwen3.6-35B-A3B (~70 GB RAM, MoE ~3B active | fast even on CPU | needs transformers>=5.2.0)": "Qwen/Qwen3.6-35B-A3B",
    # Gemma 4 — needs transformers>=5.10.1 (see models._MIN_TRANSFORMERS_
    # VERSION["gemma4"] and model_registry.GEMMA4_IDS above); loading with
    # an older transformers raises a clear upgrade error instead of a
    # cryptic AutoModel crash. Natively multimodal/encoder-free — loads
    # here via the plain text-LLM path (smolagents' TransformersModel),
    # which works for inference/generation, though the vision/audio
    # towers ride along unused; use the 🎨 Vision LLM dropdown instead if
    # you specifically want Gemma 4's image understanding.
    "🟣 Gemma-4-12B-it    (~25 GB RAM | 16GB-VRAM tier | needs transformers>=5.10.1)": "google/gemma-4-12B-it",
    "🟣 Gemma-4-26B-A4B-it (~52 GB RAM, MoE ~4B active | 24GB-VRAM tier | needs transformers>=5.10.1)": "google/gemma-4-26B-A4B-it",
    "🟣 Gemma-4-31B-it    (~63 GB RAM | 48GB+-VRAM tier | needs transformers>=5.10.1)": "google/gemma-4-31B-it",
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
# NOTE: unlike DEFAULT_EMBED_MODEL / DEFAULT_VLM_LABEL above, the default
# LLM deliberately does NOT auto-upgrade to a hardware-tier-recommended
# larger model. The embedding model is an invisible backend component and
# the VLM is only loaded on-demand for image questions, so picking a
# bigger one by default is low-surprise. The main chat LLM is different:
# silently defaulting a capable machine to a 25-70GB model would mean a
# much longer first load with no explicit action from the user, and (for
# the newly-registered Gemma 4 / Qwen3.6 sizes) a version-guard error on
# any install that hasn't upgraded transformers yet — bad first impression
# either way. The larger recommended models above are fully selectable in
# the dropdown; users on capable hardware can opt in deliberately.
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
    "🟠 Qwen2.5-VL-7B (~15 GB RAM | hardware-tier recommended, 16GB+ VRAM)": "Qwen/Qwen2.5-VL-7B-Instruct",
}

# VLM_OPTIONS starts as a copy of the base HuggingFace VLMs. Any local
# GGUF vision-model pairs (main .gguf + matching mmproj .gguf) found
# under llama_backend.LLAMA_CPP_MODEL_DIR are merged in on top of it —
# same pattern as MODEL_OPTIONS for text LLMs — so they appear in the
# same "🎨 Vision LLM" dropdown. Kept as a single dict object that is
# mutated in place (never reassigned) so every module that imported it
# sees updates.
VLM_OPTIONS = dict(BASE_VLM_OPTIONS)

# Default VLM selection — dynamic per detected hardware tier, mirroring
# README.md's "Vision Model (HF/transformers)" column: Qwen2.5-VL-7B-
# Instruct is the recommended pick from the 16GB-VRAM tier upward, since
# every GGUF vision option in that table needs a llama_backend.py
# chat-handler update this app doesn't have yet (see the README's
# "Compatibility note"). Below that tier, the existing small SmolVLM-500M
# stays the default — a 7B VLM would be a poor default on modest/CPU-only
# hardware. A saved user override (see set_default_vlm()) always wins,
# same persisted-choice pattern as DEFAULT_EMBED_MODEL above.
_VLM_LABEL_BY_TIER = {
    HardwareManager.TIER_48GB_VRAM: "🟠 Qwen2.5-VL-7B (~15 GB RAM | hardware-tier recommended, 16GB+ VRAM)",
    HardwareManager.TIER_24GB_VRAM: "🟠 Qwen2.5-VL-7B (~15 GB RAM | hardware-tier recommended, 16GB+ VRAM)",
    HardwareManager.TIER_16GB_VRAM: "🟠 Qwen2.5-VL-7B (~15 GB RAM | hardware-tier recommended, 16GB+ VRAM)",
    HardwareManager.TIER_8GB_VRAM:  "🔵 SmolVLM-500M  (~1 GB RAM | recommended)",
    HardwareManager.TIER_CPU_ONLY:  "🔵 SmolVLM-500M  (~1 GB RAM | recommended)",
    HardwareManager.TIER_UNKNOWN:   "🔵 SmolVLM-500M  (~1 GB RAM | recommended)",
}


def get_recommended_vlm_label() -> str:
    """The Vision LLM label README.md's hardware-tier table recommends
    for THIS machine. Falls back to the small SmolVLM-500M default if the
    tier can't be determined."""
    tier = HardwareManager.detect_hardware_tier()
    return _VLM_LABEL_BY_TIER.get(tier, "🔵 SmolVLM-500M  (~1 GB RAM | recommended)")


def get_default_vlm_label() -> str:
    """The VLM label actually selected by default in the UI, unless the
    user has a saved override in user_config.json — mirrors
    get_default_embed_model()'s persisted-choice-wins pattern."""
    saved = user_config.USER_CONFIG.get("default_vlm_label")
    if saved and saved in VLM_OPTIONS:
        return saved
    return get_recommended_vlm_label()


def set_default_vlm(label: str) -> None:
    """Persist an explicit default-VLM choice — same pattern as
    set_embed_model()."""
    user_config.save_user_config({"default_vlm_label": label})


DEFAULT_VLM_LABEL = get_default_vlm_label()
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
