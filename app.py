"""
RAG Agent - smolagents + ChromaDB + Dynamic Hardware Support + Gradio UI
=======================================================================
Tabs:
  💬 General Chat  — direct LLM conversation, no retrieval
  📚 RAG Chat      — always retrieves from ChromaDB before answering
  🖼️ Vision Chat   — image upload + optional visual RAG
  🎙️ Speech to Text — transcribe audio to text using Whisper
  📂 Knowledge Base — index / manage documents
"""

import os
import warnings
import time
import threading
import base64
import re
import subprocess
import sys
import platform
from io import BytesIO
from pathlib import Path
from typing import Optional, Callable

import gradio as gr
import torch
from PIL import Image

# ──────────────────────────────────────────────────────────────────
# Localization (i18n)
# ──────────────────────────────────────────────────────────────────
LANGUAGES = {
    "kh": {
        "title": "🔍 ភ្នាក់ងារ RAG",
        "subtitle": "smolagents · ChromaDB · Gemma4 / Qwen3.6 · {device} · បង្កើតដោយ LocalAiLab",
        "tab_general": "💬 ការសន្ទនាទូទៅ",
        "tab_general_desc": "ការសន្ទនាផ្ទាល់ជាមួយ LLM — មិនមានការទាញយកទិន្នន័យឡើយ។",
        "tab_rag": "📚 ការសន្ទនា RAG",
        "tab_rag_desc": "រាល់សំណួរនឹងទាញយកទិន្នន័យពីមូលដ្ឋានចំណេះដឹងជាមុនសិន រួចទើប LLM ឆ្លើយដោយប្រើតែបរិបទនោះ។",
        "tab_vision": "🖼️ ការសន្ទនាចក្ខុវិស័យ",
        "tab_vision_desc": "បង្ហោះរូបភាព និងសួរអំពីវា។",
        "tab_kb": "📂 មូលដ្ឋានចំណេះដឹង",
        "tab_stt": "🎙️ និយាយទៅជាអក្សរ",
        "tab_stt_desc": "បង្ហោះ ឬថតសំឡេង ដើម្បីបំលែងវាទៅជាអក្សរ។",
        "tab_about": "ℹ️ អំពីកម្មវិធី",
        "placeholder_gen": "និយាយអ្វីមួយ...",
        "placeholder_rag": "សួរអំពីឯកសាររបស់អ្នក...",
        "placeholder_vis": "សួរអំពីរូបភាព...",
        "btn_send": "ផ្ញើ ▶",
        "btn_load": "🔄 ទាញយក",
        "btn_clear": "🗑️ សម្អាត",
        "btn_index": "📥 បញ្ចូលឯកសារ",
        "btn_load_ds": "⬇️ ទាញយក & បញ្ចូល",
        "btn_refresh": "🔄 បញ្ជូនឡើងវិញ",
        "btn_delete": "🗑️ លុបដែលបានជ្រើសរើស",
        "btn_clear_all": "💥 លុបទាំងអស់",
        "label_llm": "🤖 LLM",
        "label_vlm": "🎨 Vision LLM",
        "label_stt": "🎙️ ម៉ូដែលនិយាយទៅជាអក្សរ",
        "label_stt_lang": "🌐 ភាសានៃសំឡេង",
        "btn_transcribe": "📝 បំលែងជាអក្សរ",
        "stt_audio_label": "ថត ឬបង្ហោះសំឡេង",
        "label_vis_rag": "🔍 ក៏ទាញយកបរិបទអត្ថបទផងដែរ",
        "label_vis_ret": "🖼️ អ្នកទាញយកចក្ខុវិស័យ (PDFs)",
        "label_ds_name": "Dataset",
        "label_ds_text": "Text col",
        "label_ds_src": "Source col",
        "label_res": "លទ្ធផល",
        "doc_table_headers": ["ប្រភព", "ប្រភេទ", "ទំព័រ", "ចំនួន Chunk"],
        "label_kb_docs": "📋 ឯកសារដែលបានបញ្ចូល",
        "header_docs": "--- \n### 📋 ឯកសារដែលបានបញ្ចូល",
        "accordion_add": "📤 បន្ថែមឯកសារ",
        "tab_upload": "បង្ហោះឯកសារ",
        "tab_hf": "HuggingFace Dataset",
        "file_label": "ទម្លាក់ PDF / TXT / MD",
        "about_tabs_title": "## ផ្ទាំង",
        "about_tabs_desc": "| ផ្ទាំង | ការពិពណ៌នា |\n|---|---|\n| 💬 ការសន្ទនាទូទៅ | ការសន្ទនាផ្ទាល់ជាមួយ LLM — មិនមានការទាញយក |\n| 📚 ការសន្ទនា RAG | ទាញយកពីមូលដ្ឋានចំណេះដឹងជាមុន រួចឆ្លើយ |\n| 🖼️ ការសន្ទនាចក្ខុវិស័យ | យល់ដឹងរូបភាព ជាមួយបរិបទអត្ថបទ |\n| 🎙️ និយាយទៅជាអក្សរ | បំលែងសំឡេងជាអក្សរ ដោយប្រើ Whisper |\n| 📂 មូលដ្ឋានចំណេះដឹង | បង្ហោះ និងគ្រប់គ្រងឯកសារ |",
        "about_arch_title": "## ស្ថាបត្យកម្ម",
        "about_arch_desc": "| សមាសធាតុ | លម្អិត |\n|---|---|",
        "about_speed_title": "## ល្បឿនរំពឹងទុក ({device})",
        "about_speed_desc": "| កិច្ចការ | ពេលវេលា |\n|---|---|\n| បញ្ចូលឯកសារ | ១០–៦០ វិ |\n| ឆ្លើយបែប General / RAG | ១–៥ នាទី |\n| ឆ្លើយបែប Vision | ២–៨ នាទី |",
        "lang_label": "ភាសា",
        "lang_options": ["Khmer", "English"],
        "status_refreshed": "🔄 បញ្ជូនឡើងវិញនូវស្ថានភាព",
        "env_fixed": "✅ បរិយាកាសត្រូវបានកែប្រែ! សូម RESTART កម្មវិធីដើម្បីអនុវត្តការផ្លាស់ប្តូរ។",
        "env_starting": "🚀 កំពុងចាប់ផ្តើមការកែប្រែបរិយាកាស...",
        "env_detected": "🔍 រកឃើញ GPU: ",
        "env_using_index": "🔗 កំពុងប្រើ PyTorch index: ",
        "env_installing": "🛠️ កំពុងដំឡើង...",
        "env_success": "✨ ការដំឡើងជោគជ័យ!",
        "env_failed": "❌ ការដំឡើងបានបរាជ័យ។ សូមពិនិត្យមើលកំណត់ត្រាខាងលើ។",
        "err_vlm": "❌ Error ក្នុងពេលបង្កើត VLM: ",
        "err_gen": "❌ Error: ",
        "err_rag": "❌ Error: ",
        "err_vis": "❌ Error: ",
        "err_upload": "⚠️ មិនអាចដោះស្រាយផ្លូវបាន: ",
        "err_unsupported": "⚠️ មិនគាំទ្រ: ",
        "err_no_files": "មិនមានឯកសារត្រូវបានបង្ហោះទេ។",
        "err_nothing_indexed": "គ្មានអ្វីត្រូវបានបញ្ចូលទេ។",
        "err_no_rows": "⚠️ មិនមានជួរណាត្រូវបានជ្រើសរើស។",
        "err_nothing_deleted": "⚠️ គ្មានអ្វីត្រូវបានលុបឡើយ។",
        "err_deleted": "🗑️ បានលុប: ",
        "err_clear_msg": "🗑️ ឯកសារទាំងអស់ត្រូវបានសម្អាត។",
        "err_empty_kb": "មូលដ្ឋានចំណេះដឹងទទេស្អាត។",
        "err_empty_visual": "empty",
        "err_visual_ready": "✅ រួចរាល់",
        "err_source_unknown": "unknown",
        "err_type_unknown": "unknown",
        "err_page_empty": "—",
        "err_status_bar": "📚 Text chunks: {n}  |  Visual index: {vis}  |  Device: {dev}",
        "err_reasoning": "🧠 ការគិត (ចុចដើម្បីពង្រីក)",
        "err_answer": "💬 ចម្លើយ",
        "err_src_none": " | គ្មានឯកសារបញ្ចូលឡើយ",
        "err_src_list": " | ប្រភព: ",
        "err_elapsed": "⏱ {elapsed:.1f}s | model: <code>{model}</code> ({dev})"
    },
    "en": {
        "title": "🔍 RAG Agent",
        "subtitle": "smolagents · ChromaDB · Gemma4 / Qwen3.6 · {device} · Developed by LocalAiLab",
        "tab_general": "💬 General Chat",
        "tab_general_desc": "Direct conversation with the LLM — no retrieval.",
        "tab_rag": "📚 RAG Chat",
        "tab_rag_desc": "Every question retrieves from the knowledge base first, then the LLM answers using only that context.",
        "tab_vision": "🖼️ Vision Chat",
        "tab_vision_desc": "Upload an image and ask questions about it.",
        "tab_kb": "📂 Knowledge Base",
        "tab_stt": "🎙️ Speech to Text",
        "tab_stt_desc": "Upload or record audio to transcribe it into text.",
        "tab_about": "ℹ️ About",
        "placeholder_gen": "Say anything ...",
        "placeholder_rag": "Ask about your documents ...",
        "placeholder_vis": "Ask about the image ...",
        "btn_send": "Send ▶",
        "btn_load": "🔄 Load",
        "btn_clear": "🗑️ Clear",
        "btn_index": "📥 Index files",
        "btn_load_ds": "⬇️ Load & index",
        "btn_refresh": "🔄 Refresh",
        "btn_delete": "🗑️ Delete Selected",
        "btn_clear_all": "💥 Clear ALL",
        "label_llm": "🤖 LLM",
        "label_vlm": "🎨 Vision LLM",
        "label_stt": "🎙️ Speech-to-Text Model",
        "label_stt_lang": "🌐 Audio Language",
        "btn_transcribe": "📝 Transcribe",
        "stt_audio_label": "Record or upload audio",
        "label_vis_rag": "🔍 Also retrieve text context",
        "label_vis_ret": "🖼️ Visual retriever (PDFs)",
        "label_ds_name": "Dataset",
        "label_ds_text": "Text col",
        "label_ds_src": "Source col",
        "label_res": "Result",
        "doc_table_headers": ["Source", "Type", "Pages", "Chunks"],
        "label_kb_docs": "📋 Indexed Documents",
        "header_docs": "--- \n### 📋 Indexed Documents",
        "accordion_add": "📤 Add Documents",
        "tab_upload": "Upload Files",
        "tab_hf": "HuggingFace Dataset",
        "file_label": "Drop PDF / TXT / MD",
        "about_tabs_title": "## Tabs",
        "about_tabs_desc": "| Tab | Description |\n|---|---|\n| 💬 General Chat | Direct LLM conversation — no retrieval |\n| 📚 RAG Chat | Retrieves from knowledge base first, then answers |\n| 🖼️ Vision Chat | Image understanding with optional text context |\n| 🎙️ Speech to Text | Transcribe audio into text using Whisper |\n| 📂 Knowledge Base | Upload & manage indexed documents |",
        "about_arch_title": "## Architecture",
        "about_arch_desc": "| Component | Detail |\n|---|---|",
        "about_speed_title": "## Expected speed ({device})",
        "about_speed_desc": "| Task | Time |\n|---|---|\n| Index a document | 10–60 s |\n| General / RAG answer | 1–5 min |\n| Vision answer | 2–8 min |",
        "lang_label": "Language",
        "lang_options": ["English", "Khmer"],
        "status_refreshed": "🔄 Status Refreshed",
        "env_fixed": "✅ Environment fixed! Please RESTART the app to apply changes.",
        "env_starting": "🚀 Starting environment fix...",
        "env_detected": "🔍 Detected GPU: ",
        "env_using_index": "🔗 Using PyTorch index: ",
        "env_installing": "🛠️ Installing...",
        "env_success": "✨ Installation successful!",
        "env_failed": "❌ Installation failed. Check the logs above.",
        "err_vlm": "❌ Error during VLM generation: ",
        "err_gen": "❌ Error: ",
        "err_rag": "❌ Error: ",
        "err_vis": "❌ Error: ",
        "err_upload": "⚠️ Cannot resolve path: ",
        "err_unsupported": "⚠️ Unsupported: ",
        "err_no_files": "No files uploaded.",
        "err_nothing_indexed": "Nothing indexed.",
        "err_no_rows": "⚠️ No rows selected.",
        "err_nothing_deleted": "⚠️ Nothing deleted.",
        "err_deleted": "🗑️ Deleted: ",
        "err_clear_msg": "🗑️ All documents cleared.",
        "err_empty_kb": "The knowledge base is empty.",
        "err_empty_visual": "empty",
        "err_visual_ready": "✅ ready",
        "err_source_unknown": "unknown",
        "err_type_unknown": "unknown",
        "err_page_empty": "—",
        "err_status_bar": "📚 Text chunks: {n}  |  Visual index: {vis}  |  Device: {dev}",
        "err_reasoning": "🧠 Reasoning (click to expand)",
        "err_answer": "💬 Answer",
        "err_src_none": " | no docs indexed",
        "err_src_list": " | sources: ",
        "err_elapsed": "⏱ {elapsed:.1f}s | model: <code>{model}</code> ({dev})"
    }
}

class HardwareManager:
    @staticmethod
    def detect_nvidia_cuda_version() -> Optional[str]:
        try:
            import subprocess
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "CUDA Version:" in line:
                        return line.split("CUDA Version:")[1].strip()
            return None
        except Exception:
            return None

    @staticmethod
    def get_required_torch_index(cuda_version: str) -> str:
        if not cuda_version:
            return "https://download.pytorch.org/whl/cpu"

        try:
            v_parts = cuda_version.split('.')
            major = v_parts[0]
            minor = v_parts[1]

            if major == "11":
                return "https://download.pytorch.org/whl/cu118"
            elif major == "12":
                if minor in ("1", "2", "3"):
                    return "https://download.pytorch.org/whl/cu121"
                elif minor in ("4", "5", "6"):
                    return "https://download.pytorch.org/whl/cu124"
                elif minor in ("7", "8", "9"):
                    return "https://download.pytorch.org/whl/cu128"
                else:
                    return "https://download.pytorch.org/whl/cu128"
            return "https://download.pytorch.org/whl/cpu"
        except Exception:
            return "https://download.pytorch.org/whl/cpu"

    @staticmethod
    def get_system_status() -> dict:
        status = {
            "gpu_brand": "none",
            "cuda_version": None,
            "torch_cuda_available": False,
            "current_device": "cpu",
            "recommended_index": "https://download.pytorch.org/whl/cpu"
        }

        cuda_ver = HardwareManager.detect_nvidia_cuda_version()
        if cuda_ver:
            status["gpu_brand"] = "nvidia"
            status["cuda_version"] = cuda_ver
            status["recommended_index"] = HardwareManager.get_required_torch_index(cuda_ver)

        try:
            import torch
            status["torch_cuda_available"] = torch.cuda.is_available()
            if status["torch_cuda_available"]:
                status["current_device"] = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                status["current_device"] = "mps"
            else:
                status["current_device"] = "cpu"
        except Exception:
            pass

        if status["gpu_brand"] == "none":
            try:
                import subprocess
                result = subprocess.run(["wmic", "path", "win32_VideoController", "get", "name"],
                                        capture_output=True, text=True, check=False)
                if "Radeon" in result.stdout or "AMD" in result.stdout:
                    status["gpu_brand"] = "amd"
                    status["recommended_index"] = "https://download.pytorch.org/whl/rocm6.2"
            except Exception:
                pass

        return status

    @staticmethod
    def fix_environment_gen():
        status = HardwareManager.get_system_status()
        if status["torch_cuda_available"]:
            yield "✅ Environment is already correctly configured for GPU!", "cuda", ""
            return

        index = status["recommended_index"]
        yield "🚀 Starting environment fix...", "cuda", "🚀 Starting environment fix..."
        yield f"🔍 Detected GPU: {status['gpu_brand'].upper()}", "cuda", f"🔍 Detected GPU: {status['gpu_brand'].upper()}"
        yield f"🔗 Using PyTorch index: {index}", "cuda", f"🔗 Using PyTorch index: {index}"

        try:
            cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", index]
            yield f"Running: {' '.join(cmd)}", "cuda", f"Running: {' '.join(cmd)}"

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

            for line in process.stdout:
                yield "🛠️ Installing...", "cuda", line.strip()

            process.wait()

            if process.returncode == 0:
                yield "✨ Installation successful!", "cuda", "✨ Installation successful!"
                yield "✅ Environment fixed! Please RESTART the app to apply changes.", "cuda", "✅ Environment fixed! Please RESTART the app to apply changes."
            else:
                yield f"❌ Installation failed with exit code {process.returncode}", "cpu", f"❌ Installation failed with exit code {process.returncode}"
                yield "❌ Installation failed. Check the logs above.", "cpu", "❌ Installation failed. Check the logs above."

        except Exception as e:
            yield f"❌ Error: {str(e)}", "cpu", f"❌ Error: {str(e)}"


if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

TORCH_DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# ──────────────────────────────────────────────────────────────────
# Developer branding (LocalAiLab logo, shown in the header)
# ──────────────────────────────────────────────────────────────────
_LOGO_PATH = Path(__file__).parent / "assets" / "logo.jpg"

def _load_logo_b64() -> str:
    try:
        data = _LOGO_PATH.read_bytes()
        return base64.b64encode(data).decode("utf-8")
    except Exception:
        return ""

DEVELOPER_LOGO_B64 = _load_logo_b64()
DEVELOPER_NAME = "LocalAiLab"

MODEL_OPTIONS = {
    "🟢 Qwen3-0.6B   (~1.2 GB RAM | fastest)": "Qwen/Qwen3-0.6B",
    "🟡 Qwen3-1.7B   (~3 GB RAM)":             "Qwen/Qwen3-1.7B",
    "🟡 Qwen3-4B     (~7 GB RAM)":             "Qwen/Qwen3-4B",
    "🔵 Gemma-4-E2B  (~4 GB RAM)":             "google/gemma-4-E2B-it",
}
DEFAULT_LLM_LABEL = "🟢 Qwen3-0.6B   (~1.2 GB RAM | fastest)"
DEFAULT_LLM_MODEL = MODEL_OPTIONS[DEFAULT_LLM_LABEL]

VLM_OPTIONS = {
    "🔵 SmolVLM-256M  (~0.5 GB RAM | tiny)":  "HuggingFaceTB/SmolVLM-256M-Instruct",
    "🔵 SmolVLM-500M  (~1 GB RAM | recommended)": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "🟢 Qwen2.5-VL-3B (~6 GB RAM)":           "Qwen/Qwen2.5-VL-3B-Instruct",
}
DEFAULT_VLM_LABEL = "🔵 SmolVLM-500M  (~1 GB RAM | recommended)"
DEFAULT_VLM_MODEL = VLM_OPTIONS[DEFAULT_VLM_LABEL]

VISUAL_RETRIEVER_OPTIONS = {
    "vidore/colsmolvlm-v0.1  (~2 GB | recommended)": "vidore/colsmolvlm-v0.1",
    "vidore/colqwen2-v1.0    (~8 GB | higher accuracy)": "vidore/colqwen2-v1.0",
}
DEFAULT_VISUAL_RETRIEVER = "vidore/colsmolvlm-v0.1"

STT_OPTIONS = {
    "🟢 Whisper-tiny    (~1 GB RAM | fastest)":   "openai/whisper-tiny",
    "🟡 Whisper-base    (~1 GB RAM)":              "openai/whisper-base",
    "🟡 Whisper-small   (~2 GB RAM | recommended)": "openai/whisper-small",
    "🔵 Whisper-large-v3 (~10 GB RAM | best accuracy, multilingual incl. Khmer)": "openai/whisper-large-v3",
}
DEFAULT_STT_LABEL = "🟡 Whisper-small   (~2 GB RAM | recommended)"
DEFAULT_STT_MODEL = STT_OPTIONS[DEFAULT_STT_LABEL]

DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
CHROMA_PERSIST_DIR  = "./chroma_db"
VISUAL_INDEX_DIR    = "./visual_index"
CHUNK_SIZE          = 1024
CHUNK_OVERLAP       = 128
TOP_K               = 4
MAX_NEW_TOKENS      = 512

QWEN3_IDS   = {"Qwen/Qwen3-0.6B","Qwen/Qwen3-1.7B","Qwen/Qwen3-4B",
               "Qwen/Qwen3-8B","Qwen/Qwen3-14B","Qwen/Qwen3-32B"}
QWEN36_IDS  = {"Qwen/Qwen3.6-27B","Qwen/Qwen3.6-35B-A3B"}
ALL_QWEN_IDS= QWEN3_IDS | QWEN36_IDS
QWEN_VL_IDS = {"Qwen/Qwen2.5-VL-3B-Instruct","Qwen/Qwen2.5-VL-7B-Instruct"}
SMOL_VLM_IDS= {"HuggingFaceTB/SmolVLM-256M-Instruct","HuggingFaceTB/SmolVLM-500M-Instruct",
               "HuggingFaceTB/SmolVLM2-2.2B-Instruct"}

_embed_model         = None
_chroma_col          = None
_llm                 = None
_llm_model_id        = DEFAULT_LLM_MODEL
_vlm_model           = None
_vlm_processor       = None
_vlm_model_id        = None
_visual_retriever    = None
_visual_retriever_id = None
_llm_lock            = threading.Lock()
_vlm_lock            = threading.Lock()
_stt_pipeline        = None
_stt_model_id        = None
_stt_lock            = threading.Lock()


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        print(f"[RAG] Loading embedding model '{DEFAULT_EMBED_MODEL}' on {DEVICE.upper()} …")
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(DEFAULT_EMBED_MODEL, device=DEVICE)
    return _embed_model


def encode_texts(texts: list, normalize: bool = True):
    vecs = get_embed_model().encode(texts, normalize_embeddings=normalize,
                                    show_progress_bar=False)
    return vecs.tolist() if hasattr(vecs, "tolist") else vecs


def get_chroma_collection(name: str = "rag_docs"):
    global _chroma_col
    if _chroma_col is None:
        import chromadb
        client      = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _chroma_col = client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"})
        print(f"[RAG] ChromaDB ready — {_chroma_col.count()} chunks indexed.")
    return _chroma_col


def get_llm(model_id: Optional[str] = None):
    global _llm, _llm_model_id
    target = model_id or _llm_model_id

    if _llm is not None and target == _llm_model_id:
        return _llm

    with _llm_lock:
        if _llm is not None and target == _llm_model_id:
            return _llm

        from smolagents import TransformersModel
        print(f"[RAG] Loading LLM '{target}' on {DEVICE.upper()} …")
        _llm = TransformersModel(
            model_id=target,
            device_map=DEVICE,
            torch_dtype=TORCH_DTYPE,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.6,
            top_p=0.95,
            trust_remote_code=True,
        )
        _llm_model_id = target
        return _llm


def _call_llm(model_id: str, system: str, user: str) -> tuple[str, float]:
    from smolagents.models import ChatMessage
    llm = get_llm(model_id)
    messages = [
        ChatMessage(role="system", content=[{"type": "text", "text": system}]),
        ChatMessage(role="user",   content=[{"type": "text", "text": user}]),
    ]
    t0  = time.time()
    out = llm(messages)
    if hasattr(out, "content"):
        c = out.content
        if isinstance(c, list):
            ans = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
        else:
            ans = str(c)
    else:
        ans = str(out)
    return ans.strip(), time.time() - t0


def get_vlm(model_id: Optional[str] = None):
    global _vlm_model, _vlm_processor, _vlm_model_id

    target = model_id or DEFAULT_VLM_MODEL

    if _vlm_model is not None and target == _vlm_model_id:
        return _vlm_model, _vlm_processor

    print(f"[VLM] Loading '{target}' on {DEVICE.upper()} …")

    from transformers import AutoProcessor

    if target in QWEN_VL_IDS:
        from transformers import Qwen2_5_VLForConditionalGeneration

        _vlm_processor = AutoProcessor.from_pretrained(
            target,
            trust_remote_code=True,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )

        _vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            target,
            torch_dtype=TORCH_DTYPE,
            device_map=DEVICE,
            trust_remote_code=True,
        )

        _vlm_model._arch = "qwen_vl"

    elif target in SMOL_VLM_IDS:
        try:
            from transformers import AutoModelForImageTextToText
            ModelClass = AutoModelForImageTextToText
            print("[VLM] Using AutoModelForImageTextToText")
        except ImportError:
            try:
                from transformers import AutoModelForVision2Seq
                ModelClass = AutoModelForVision2Seq
                print("[VLM] Using AutoModelForVision2Seq")
            except ImportError:
                from transformers import AutoModel
                ModelClass = AutoModel
                print("[VLM] Falling back to AutoModel")

        _vlm_processor = AutoProcessor.from_pretrained(
            target,
            trust_remote_code=True,
        )

        _vlm_model = ModelClass.from_pretrained(
            target,
            torch_dtype=TORCH_DTYPE,
            device_map=DEVICE,
            trust_remote_code=True,
        )

        _vlm_model._arch = "smolvlm"

    else:
        raise ValueError(f"Unknown VLM: {target}")

    _vlm_model_id = target

    return _vlm_model, _vlm_processor


def vlm_answer(question: str, images: list, context: str = "", model_id: Optional[str] = None) -> str:
    try:
        model, processor = get_vlm(model_id)
        arch = getattr(model, "_arch", "smolvlm")
        system_prompt = "You are a helpful assistant. Answer based on images and context."
        user_text = question + (f"\n\nContext:\n{context}" if context else "")
        if arch == "qwen_vl":
            from qwen_vl_utils import process_vision_info
            content = [{"type": "image", "image": img} for img in images]
            content.append({"type": "text", "text": user_text})
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user",   "content": content}]
            text_in = processor.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True)
            img_in, _ = process_vision_info(messages)
            inputs = processor(text=[text_in], images=img_in,
                               padding=True, return_tensors="pt").to(DEVICE)
        else:
            content = [{"type": "image"} for _ in images]
            content.append({"type": "text", "text": user_text})
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user",   "content": content}]
            text_in = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs  = processor(text=text_in, images=images or None, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        trimmed = out[0][inputs["input_ids"].shape[-1]:]
        return processor.decode(trimmed, skip_special_tokens=True)
    except Exception as e:
        import traceback
        return f"❌ Error during VLM generation: {str(e)}\n\n{traceback.format_exc()}"


def get_stt_pipeline(model_id: Optional[str] = None):
    global _stt_pipeline, _stt_model_id
    target = model_id or DEFAULT_STT_MODEL

    if _stt_pipeline is not None and target == _stt_model_id:
        return _stt_pipeline

    with _stt_lock:
        if _stt_pipeline is not None and target == _stt_model_id:
            return _stt_pipeline

        print(f"[STT] Loading '{target}' on {DEVICE.upper()} …")
        from transformers import pipeline as hf_pipeline

        # pipeline() wants an integer device index for cuda, "mps", or -1 for cpu
        if DEVICE == "cuda":
            device_arg = 0
        elif DEVICE == "mps":
            device_arg = "mps"
        else:
            device_arg = -1

        _stt_pipeline = hf_pipeline(
            "automatic-speech-recognition",
            model=target,
            torch_dtype=TORCH_DTYPE,
            device=device_arg,
        )
        _stt_model_id = target
        return _stt_pipeline


def transcribe_audio(audio_path: Optional[str], language: Optional[str] = None,
                     model_id: Optional[str] = None) -> str:
    if not audio_path:
        return ""
    try:
        asr = get_stt_pipeline(model_id)
        gen_kwargs = {}
        if language and language != "auto":
            gen_kwargs["language"] = language
            gen_kwargs["task"] = "transcribe"
        result = asr(audio_path, generate_kwargs=gen_kwargs or None)
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        return text.strip()
    except Exception as e:
        import traceback
        return f"❌ {e}\n\n{traceback.format_exc()}"


def get_visual_retriever(model_id: Optional[str] = None):
    global _visual_retriever, _visual_retriever_id
    target = model_id or DEFAULT_VISUAL_RETRIEVER
    if _visual_retriever is not None and target == _visual_retriever_id:
        return _visual_retriever
    try:
        from byaldi import RAGMultiModalModel
        index_path = Path(VISUAL_INDEX_DIR) / "main"
        if index_path.exists():
            _visual_retriever = RAGMultiModalModel.from_index(str(index_path), verbose=0)
        else:
            _visual_retriever = RAGMultiModalModel.from_pretrained(target, verbose=0)
        _visual_retriever_id = target
    except ImportError:
        _visual_retriever = None
    return _visual_retriever


def index_pdf_visual(filepath: str, retriever_id: str) -> str:
    try:
        from byaldi import RAGMultiModalModel
        global _visual_retriever, _visual_retriever_id
        index_path = Path(VISUAL_INDEX_DIR) / "main"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        if _visual_retriever is None or _visual_retriever_id != retriever_id:
            _visual_retriever = RAGMultiModalModel.from_pretrained(retriever_id, verbose=0)
            _visual_retriever_id = retriever_id
        if index_path.exists():
            _visual_retriever.add_to_index(input_item=filepath,
                                           store_collection_with_index=True,
                                           doc_id=int(time.time()))
        else:
            _visual_retriever.index(input_path=filepath, index_name="main",
                                    index_root=VISUAL_INDEX_DIR,
                                    store_collection_with_index=True, overwrite=False)
        return f"✅ Visual index updated: '{Path(filepath).name}'"
    except ImportError:
        return "⚠️ byaldi not installed — skipping visual index."
    except Exception as e:
        return f"❌ {e}"


def visual_retrieve(query: str, top_k: int = 3) -> list:
    retriever = get_visual_retriever()
    if retriever is None:
        return []
    try:
        results = retriever.search(query, k=top_k)
        images  = []
        for r in results:
            if hasattr(r, "base64") and r.base64:
                images.append(Image.open(BytesIO(base64.b64decode(r.base64))).convert("RGB"))
        return images
    except Exception as e:
        print(f"[VIS] {e}")
        return []


def _chunk_text(text: str):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start: start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _safe_meta(meta: dict, chunk_index: int) -> dict:
    safe = {k: v if isinstance(v, (str, int, float, bool)) else str(v)
            for k, v in meta.items()}
    safe["chunk_index"] = chunk_index
    return safe


def index_texts(texts: list, metadatas: list) -> int:
    col = get_chroma_collection()
    all_chunks, all_metas, all_ids = [], [], []
    for text, meta in zip(texts, metadatas):
        for j, chunk in enumerate(_chunk_text(text)):
            uid = f"{meta.get('source','doc')}__{len(all_chunks)}"
            all_chunks.append(chunk)
            all_metas.append(_safe_meta(meta, j))
            all_ids.append(uid)
    if not all_chunks:
        return 0
    for i in range(0, len(all_chunks), 64):
        vecs = encode_texts(all_chunks[i:i+64])
        col.upsert(embeddings=vecs, documents=all_chunks[i:i+64],
                   metadatas=all_metas[i:i+64], ids=all_ids[i:i+64])
    return len(all_chunks)


def index_pdf_file(filepath: str, visual_retriever_id: str = DEFAULT_VISUAL_RETRIEVER) -> str:
    msgs = []
    try:
        import fitz
        doc = fitz.open(filepath)
        texts, metas = [], []
        for n, page in enumerate(doc):
            t = page.get_text()
            if t.strip():
                texts.append(t)
                metas.append({"source": Path(filepath).name, "page": n+1, "type": "pdf"})
        doc.close()
        msgs.append(f"✅ Text: {index_texts(texts, metas)} chunks")
    except Exception as e:
        msgs.append(f"⚠️ Text failed: {e}")
    msgs.append(f"🖼️ {index_pdf_visual(filepath, visual_retriever_id)}")
    return "\n".join(msgs)


def index_txt_file(filepath: str) -> str:
    try:
        text = Path(filepath).read_text(encoding="utf-8", errors="replace")
        k    = index_texts([text], [{"source": Path(filepath).name, "type": "txt"}])
        return f"✅ {k} chunks from '{Path(filepath).name}'"
    except Exception as e:
        return f"❌ {e}"


def _get_file_path(f) -> Optional[str]:
    if f is None:          return None
    if isinstance(f, str): return f
    if isinstance(f, dict):return f.get("path") or f.get("name") or f.get("tmp_path")
    if hasattr(f, "path"): return f.path
    if hasattr(f, "name"): return f.name
    return None


def index_uploaded_files(files, visual_retriever_label: str) -> str:
    if not files:
        return "No files uploaded."
    retriever_id = VISUAL_RETRIEVER_OPTIONS.get(visual_retriever_label, DEFAULT_VISUAL_RETRIEVER)
    if not isinstance(files, list):
        files = [files]
    flat = []
    for item in files:
        flat.extend(item) if isinstance(item, list) else flat.append(item)
    msgs = []
    for f in flat:
        path = _get_file_path(f)
        if not path:
            msgs.append(f"⚠️ Cannot resolve path: {f!r}")
            continue
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            msgs.append(index_pdf_file(path, retriever_id))
        elif ext in (".txt", ".md"):
            msgs.append(index_txt_file(path))
        else:
            msgs.append(f"⚠️ Unsupported: {ext}")
    return "\n".join(msgs) or "Nothing indexed."


def index_hf_dataset(dataset_name: str, text_col: str, source_col: str = ""):
    try:
        import datasets as ds
        dataset = ds.load_dataset(dataset_name, split="train")
        texts, metas = [], []
        for row in dataset:
            t = row.get(text_col, "")
            if not t:
                continue
            src = row.get(source_col, dataset_name) if source_col else dataset_name
            texts.append(str(t))
            metas.append({"source": str(src), "type": "hf_dataset"})
        k = index_texts(texts, metas)
        return f"✅ {k} chunks from '{dataset_name}'", get_doc_table()
    except Exception as e:
        return f"❌ {e}", get_doc_table()


def get_doc_table() -> list:
    col = get_chroma_collection()
    if col.count() == 0:
        return []
    from collections import defaultdict
    result = col.get(include=["metadatas"])
    agg = defaultdict(lambda: {"type": "", "pages": set(), "chunks": 0})
    for m in result["metadatas"]:
        src = m.get("source", "unknown")
        agg[src]["type"]   = m.get("type", "unknown")
        agg[src]["chunks"] += 1
        if m.get("page"):
            agg[src]["pages"].add(m["page"])
    return [[src, info["type"],
             str(len(info["pages"])) if info["pages"] else "—",
             info["chunks"]]
            for src, info in sorted(agg.items())]


def delete_selected_sources(selected_rows: list, doc_table_data: list) -> tuple:
    if not selected_rows:
        return get_doc_table(), "⚠️ No rows selected."
    col, deleted = get_chroma_collection(), []
    for row_idx in selected_rows:
        if row_idx >= len(doc_table_data):
            continue
        src    = doc_table_data[row_idx][0]
        result = col.get(where={"source": src}, include=["metadatas"])
        if result["ids"]:
            col.delete(ids=result["ids"])
            deleted.append(f"'{src}' ({len(result['ids'])} chunks)")
    msg = ("🗑️ Deleted: " + ", ".join(deleted)) if deleted else "⚠️ Nothing deleted."
    return get_doc_table(), msg


def clear_index() -> tuple:
    global _chroma_col
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        client.delete_collection("rag_docs")
    except Exception:
        pass
    _chroma_col = None
    get_chroma_collection()
    return [], "🗑️ All documents cleared."


def get_index_stats(lang_key: str = "en") -> str:
    l = LANGUAGES[lang_key]
    n       = get_chroma_collection().count()
    vis_str = l["err_visual_ready"] if (Path(VISUAL_INDEX_DIR) / "main").exists() else l["err_empty_visual"]
    return l["err_status_bar"].format(n=n, vis=vis_str, dev=DEVICE.upper())


def retrieve_context(query: str) -> tuple[str, list[str]]:
    col = get_chroma_collection()
    if col.count() == 0:
        return "", []
    results = col.query(query_embeddings=encode_texts([query]),
                        n_results=min(TOP_K, col.count()))
    docs  = results["documents"][0]
    metas = results["metadatas"][0]
    if not docs:
        return "", []
    context = "\n\n".join(f"[{m.get('source','?')}]\n{d}"
                          for d, m in zip(docs, metas))
    sources = list({m.get("source", "?") for m in metas})
    return context, sources


import re

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

def chat_general(user_message: str, history: list, model_label: str):
    if not user_message.strip():
        return history, ""
    history = history or []
    history.append({"role": "user", "content": user_message})
    model_id = MODEL_OPTIONS.get(model_label, DEFAULT_LLM_MODEL)
    try:
        system = "You are a helpful, friendly assistant."
        ans, elapsed = _call_llm(model_id, system, user_message)
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


def chat_rag(user_message: str, history: list, model_label: str):
    if not user_message.strip():
        return history, ""
    history = history or []
    history.append({"role": "user", "content": user_message})
    model_id = MODEL_OPTIONS.get(model_label, DEFAULT_LLM_MODEL)
    try:
        context, sources = retrieve_context(user_message)
        system = (
            "You are a helpful assistant. "
            "Answer the user's question using ONLY the provided context. "
            "If the context does not contain the answer, say so clearly."
        )
        if context:
            user_prompt = f"Context:\n{context}\n\nQuestion: {user_message}"
        else:
            user_prompt = (f"Question: {user_message}\n\n"
                           "(The knowledge base is empty — please index some documents first.)")
        ans, elapsed = _call_llm(model_id, system, user_prompt)
        src_str  = f" | sources: {', '.join(sources)}" if sources else " | no docs indexed"
        formatted = format_llm_response(ans)
        response = (
            formatted +
            f"\n\n<hr><sub>⏱ {elapsed:.1f}s | model: <code>{model_id}</code> ({DEVICE.upper()}){src_str}</sub>"
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
    vlm_id = VLM_OPTIONS.get(vlm_label, DEFAULT_VLM_MODEL)
    try:
        # Pre-load VLM outside vlm_answer so errors surface cleanly
        get_vlm(vlm_id)
        pil_images = []
        if uploaded_image is not None:
            if isinstance(uploaded_image, Image.Image):
                pil_images.append(uploaded_image)
            elif isinstance(uploaded_image, str):
                pil_images.append(Image.open(uploaded_image).convert("RGB"))
        if use_visual_rag and user_message:
            pil_images.extend(visual_retrieve(user_message, top_k=2))
        context, _ = retrieve_context(user_message) if user_message else ("", [])
        t0  = time.time()
        ans = vlm_answer(user_message, pil_images, context=context, model_id=vlm_id)
        elapsed  = time.time() - t0
        response = (f"{ans}\n\n*⏱ {elapsed:.1f}s | VLM: `{vlm_id}` ({DEVICE.upper()})"
                    f" | images: {len(pil_images)}*")
    except Exception as e:
        import traceback
        response = f"❌ {e}\n\n{traceback.format_exc()}"
    history.append({"role": "assistant", "content": response})
    return history, None


CSS = """
.status-bar   { font-size:0.82rem; color:#888; padding:4px 8px; }
.header-wrap  { display:flex; align-items:baseline; gap:12px; margin-bottom:6px; }
.header-title { font-size:1.6rem; font-weight:700; }
.header-sub   { font-size:0.9rem; color:#aaa; }
.dev-logo     { width:56px; height:56px; border-radius:50%; object-fit:cover;
                box-shadow:0 0 8px rgba(120,80,255,0.6); flex-shrink:0; margin-right:4px; }
"""

def refresh_system_ui():
    status = HardwareManager.get_system_status()
    return (
        "🔄 Status Refreshed",
        status["current_device"].upper(),
        status["cuda_version"] or "N/A",
        "✅ Yes" if status["torch_cuda_available"] else "❌ No",
        "" # Clear logs
    )

def fix_system_ui_generator():
    status = HardwareManager.get_system_status()
    yield (
        "🚀 Starting fix...",
        status["current_device"].upper(),
        status["cuda_version"] or "N/A",
        "✅ Yes" if status["torch_cuda_available"] else "❌ No",
        "Starting..."
    )

    for msg, dev, log in HardwareManager.fix_environment_gen():
        yield (
            msg,
            dev.upper(),
            status["cuda_version"] or "N/A",
            "✅ Yes" if status["torch_cuda_available"] else "❌ No",
            log
        )


def build_ui():
    # Start with Khmer as default
    L = LANGUAGES["kh"]

    with gr.Blocks(title="🔍 RAG Agent") as demo:
        lang_state = gr.State("kh")

        # ── Header ────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=8):
                with gr.Row():
                    if DEVELOPER_LOGO_B64:
                        with gr.Column(scale=0, min_width=64):
                            gr.HTML(
                                f'<img src="data:image/jpeg;base64,{DEVELOPER_LOGO_B64}" '
                                f'class="dev-logo" alt="{DEVELOPER_NAME} logo" />'
                            )
                    with gr.Column():
                        header_title = gr.HTML(
                            f'<div class="header-wrap"><span class="header-title">{L["title"]}</span></div>'
                        )
                        header_sub = gr.HTML(
                            f'<div class="header-wrap"><span class="header-sub">'
                            f'{L["subtitle"].format(device=DEVICE.upper())}</span></div>'
                        )
            with gr.Column(scale=2, variant="panel"):
                lang_dropdown = gr.Dropdown(
                    choices=["Khmer", "English"], value="Khmer",
                    label="🌐 Language", scale=1
                )

        # ── Tabs ──────────────────────────────────────────────────
        with gr.Tabs():

            # ── Tab 1: General Chat ───────────────────────────────
            with gr.Tab(L["tab_general"]) as tab_gen:
                gen_desc    = gr.Markdown(L["tab_general_desc"])
                bot_gen     = gr.Chatbot(height=440)
                with gr.Row():
                    msg_gen  = gr.Textbox(placeholder=L["placeholder_gen"], show_label=False, scale=8)
                    send_gen = gr.Button(L["btn_send"], variant="primary", scale=1)
                with gr.Row():
                    model_dd_gen   = gr.Dropdown(choices=list(MODEL_OPTIONS.keys()), value=DEFAULT_LLM_LABEL, label=L["label_llm"], scale=6)
                    reload_gen     = gr.Button(L["btn_load"], size="sm", scale=2)
                    reload_gen_out = gr.Textbox(show_label=False, interactive=False, scale=4)
                clear_gen = gr.Button(L["btn_clear"], size="sm")

            # ── Tab 2: RAG Chat ───────────────────────────────────
            with gr.Tab(L["tab_rag"]) as tab_rag:
                rag_desc    = gr.Markdown(L["tab_rag_desc"])
                bot_rag     = gr.Chatbot(height=440)
                with gr.Row():
                    msg_rag  = gr.Textbox(placeholder=L["placeholder_rag"], show_label=False, scale=8)
                    send_rag = gr.Button(L["btn_send"], variant="primary", scale=1)
                with gr.Row():
                    model_dd_rag   = gr.Dropdown(choices=list(MODEL_OPTIONS.keys()), value=DEFAULT_LLM_LABEL, label=L["label_llm"], scale=6)
                    reload_rag     = gr.Button(L["btn_load"], size="sm", scale=2)
                    reload_rag_out = gr.Textbox(show_label=False, interactive=False, scale=4)
                clear_rag = gr.Button(L["btn_clear"], size="sm")

            # ── Tab 3: Vision Chat ────────────────────────────────
            with gr.Tab(L["tab_vision"]) as tab_vis:
                vis_desc    = gr.Markdown(L["tab_vision_desc"])
                bot_vis     = gr.Chatbot(height=400)
                with gr.Row():
                    msg_vis    = gr.Textbox(placeholder=L["placeholder_vis"], show_label=False, scale=6)
                    img_upload = gr.Image(type="pil", sources=["upload", "clipboard"], scale=2)
                    send_vis   = gr.Button(L["btn_send"], variant="primary", scale=1)
                with gr.Row():
                    vlm_dd       = gr.Dropdown(choices=list(VLM_OPTIONS.keys()), value=DEFAULT_VLM_LABEL, label=L["label_vlm"], scale=5)
                    vis_rag_chk  = gr.Checkbox(label=L["label_vis_rag"], value=True, scale=2)
                    load_vlm_btn = gr.Button(L["btn_load"], size="sm", scale=2)
                    load_vlm_out = gr.Textbox(show_label=False, interactive=False, scale=3)
                clear_vis = gr.Button(L["btn_clear"], size="sm")

            # ── Tab 4: Speech to Text ─────────────────────────────
            with gr.Tab(L["tab_stt"]) as tab_stt:
                stt_desc  = gr.Markdown(L["tab_stt_desc"])
                stt_audio = gr.Audio(label=L["stt_audio_label"], sources=["microphone", "upload"], type="filepath")
                with gr.Row():
                    stt_dd      = gr.Dropdown(choices=list(STT_OPTIONS.keys()), value=DEFAULT_STT_LABEL, label=L["label_stt"], scale=5)
                    stt_lang_dd = gr.Dropdown(
                        choices=[("Auto-detect", "auto"), ("English", "english"), ("Khmer", "khmer"),
                                 ("French", "french"), ("Chinese", "chinese"), ("Japanese", "japanese")],
                        value="auto", label=L["label_stt_lang"], scale=3,
                    )
                    load_stt_btn = gr.Button(L["btn_load"], size="sm", scale=2)
                    load_stt_out = gr.Textbox(show_label=False, interactive=False, scale=3)
                transcribe_btn = gr.Button(L["btn_transcribe"], variant="primary")
                stt_output     = gr.Textbox(label=L["label_res"], lines=8, interactive=True)

            # ── Tab 5: Knowledge Base ─────────────────────────────
            with gr.Tab(L["tab_kb"]) as tab_kb:
                with gr.Accordion(L["accordion_add"], open=True) as acc_add:
                    with gr.Tabs():
                        with gr.Tab(L["tab_upload"]) as tab_upload:
                            file_up    = gr.File(label=L["file_label"], file_types=[".pdf",".txt",".md"], file_count="multiple")
                            vis_ret_dd = gr.Dropdown(choices=list(VISUAL_RETRIEVER_OPTIONS.keys()), value=list(VISUAL_RETRIEVER_OPTIONS.keys())[0], label=L["label_vis_ret"])
                            up_btn     = gr.Button(L["btn_index"], variant="primary")
                            up_msg     = gr.Textbox(label=L["label_res"], interactive=False, lines=4)
                        with gr.Tab(L["tab_hf"]) as tab_hf:
                            with gr.Row():
                                ds_name    = gr.Textbox(label=L["label_ds_name"], value="m-ric/huggingface_doc", scale=3)
                                ds_textcol = gr.Textbox(label=L["label_ds_text"], value="text", scale=1)
                                ds_srccol  = gr.Textbox(label=L["label_ds_src"], value="source", scale=1)
                            ds_btn = gr.Button(L["btn_load_ds"], variant="secondary")
                            ds_msg = gr.Textbox(label=L["label_res"], interactive=False, lines=2)

                kb_header = gr.Markdown(L["header_docs"])
                doc_table = gr.Dataframe(
                    headers=L["doc_table_headers"],
                    datatype=["str","str","str","number"],
                    value=get_doc_table,
                    interactive=True, wrap=True,
                )
                with gr.Row():
                    refresh_btn    = gr.Button(L["btn_refresh"], size="sm", scale=2)
                    delete_sel_btn = gr.Button(L["btn_delete"], variant="stop", size="sm", scale=2)
                    clear_all_btn  = gr.Button(L["btn_clear_all"], variant="stop", size="sm", scale=2)
                action_msg         = gr.Textbox(label="", interactive=False, lines=1)
                selected_rows_state = gr.State([])

            # ── Tab 6: About ──────────────────────────────────────
            with gr.Tab(L["tab_about"]) as tab_about:
                about_tabs_title_md  = gr.Markdown(L["about_tabs_title"])
                about_tabs_desc_md   = gr.Markdown(L["about_tabs_desc"])
                about_arch_title_md  = gr.Markdown(L["about_arch_title"])
                about_arch_desc_md   = gr.Markdown(L["about_arch_desc"])
                about_speed_title_md = gr.Markdown(L["about_speed_title"].format(device=DEVICE.upper()))
                about_speed_desc_md  = gr.Markdown(L["about_speed_desc"])

        status_bar = gr.Textbox(
            value=get_index_stats("kh"), interactive=False,
            show_label=False, elem_classes=["status-bar"]
        )

        # ── Event handlers ────────────────────────────────────────

        # General Chat
        def reload_gen_fn(label):
            global _llm; _llm = None
            mid = MODEL_OPTIONS.get(label, DEFAULT_LLM_MODEL)
            try:
                get_llm(mid); return f"✅ '{mid}' loaded"
            except Exception as e:
                return f"❌ {e}"

        msg_gen.submit(chat_general,  [msg_gen, bot_gen, model_dd_gen], [bot_gen, msg_gen])
        send_gen.click(chat_general,  [msg_gen, bot_gen, model_dd_gen], [bot_gen, msg_gen])
        clear_gen.click(lambda: ([], ""), outputs=[bot_gen, msg_gen])
        reload_gen.click(reload_gen_fn, [model_dd_gen], [reload_gen_out])

        # RAG Chat
        def reload_rag_fn(label):
            global _llm; _llm = None
            mid = MODEL_OPTIONS.get(label, DEFAULT_LLM_MODEL)
            try:
                get_llm(mid); return f"✅ '{mid}' loaded"
            except Exception as e:
                return f"❌ {e}"

        msg_rag.submit(chat_rag,  [msg_rag, bot_rag, model_dd_rag], [bot_rag, msg_rag])
        send_rag.click(chat_rag,  [msg_rag, bot_rag, model_dd_rag], [bot_rag, msg_rag])
        clear_rag.click(lambda: ([], ""), outputs=[bot_rag, msg_rag])
        reload_rag.click(reload_rag_fn, [model_dd_rag], [reload_rag_out])

        # Vision Chat
        def load_vlm_fn(label):
            global _vlm_model, _vlm_processor
            _vlm_model = _vlm_processor = None
            mid = VLM_OPTIONS.get(label, DEFAULT_VLM_MODEL)
            try:
                get_vlm(mid); return f"✅ '{mid}' loaded"
            except Exception as e:
                return f"❌ {e}"

        send_vis.click(chat_vision,  [msg_vis, img_upload, bot_vis, vlm_dd, vis_rag_chk], [bot_vis, img_upload])
        msg_vis.submit(chat_vision,  [msg_vis, img_upload, bot_vis, vlm_dd, vis_rag_chk], [bot_vis, img_upload])
        clear_vis.click(lambda: ([], None), outputs=[bot_vis, img_upload])
        load_vlm_btn.click(load_vlm_fn, [vlm_dd], [load_vlm_out])

        # Speech to Text
        def load_stt_fn(label):
            global _stt_pipeline, _stt_model_id
            _stt_pipeline = None; _stt_model_id = None
            mid = STT_OPTIONS.get(label, DEFAULT_STT_MODEL)
            try:
                get_stt_pipeline(mid); return f"✅ '{mid}' loaded"
            except Exception as e:
                return f"❌ {e}"

        def do_transcribe(audio_path, stt_label, lang_choice):
            mid = STT_OPTIONS.get(stt_label, DEFAULT_STT_MODEL)
            return transcribe_audio(audio_path, language=lang_choice, model_id=mid)

        transcribe_btn.click(do_transcribe, [stt_audio, stt_dd, stt_lang_dd], [stt_output])
        load_stt_btn.click(load_stt_fn, [stt_dd], [load_stt_out])

        # Knowledge Base
        def on_select(evt: gr.SelectData, current):
            row = evt.index[0]
            if row in current: current.remove(row)
            else:              current.append(row)
            return current

        def do_upload(files, vis_ret_label):
            import traceback
            try:
                return index_uploaded_files(files, vis_ret_label), get_doc_table()
            except Exception as e:
                return f"❌ {traceback.format_exc()}", get_doc_table()

        def do_delete(selected, table_data):
            rows = table_data if isinstance(table_data, list) else table_data.values.tolist()
            new_table, msg = delete_selected_sources(selected, rows)
            return new_table, msg, []

        def do_clear():
            table, msg = clear_index()
            return table, msg, []

        doc_table.select(on_select,        [selected_rows_state], [selected_rows_state])
        up_btn.click(do_upload,            [file_up, vis_ret_dd], [up_msg, doc_table])
        ds_btn.click(index_hf_dataset,     [ds_name, ds_textcol, ds_srccol], [ds_msg, doc_table])
        refresh_btn.click(get_doc_table,   outputs=[doc_table])
        delete_sel_btn.click(do_delete,    [selected_rows_state, doc_table], [doc_table, action_msg, selected_rows_state])
        clear_all_btn.click(do_clear,      outputs=[doc_table, action_msg, selected_rows_state])

        # ── Language switcher ─────────────────────────────────────
        def switch_lang(lang_name):
            lk = "kh" if lang_name == "Khmer" else "en"
            l  = LANGUAGES[lk]
            return (
                lk,
                # header
                f'<div class="header-wrap"><span class="header-title">{l["title"]}</span></div>',
                f'<div class="header-wrap"><span class="header-sub">{l["subtitle"].format(device=DEVICE.upper())}</span></div>',
                # General Chat
                gr.update(value=l["tab_general_desc"]),
                gr.update(placeholder=l["placeholder_gen"]),
                gr.update(value=l["btn_send"]),
                gr.update(label=l["label_llm"]),
                gr.update(value=l["btn_load"]),
                gr.update(value=l["btn_clear"]),
                # RAG Chat
                gr.update(value=l["tab_rag_desc"]),
                gr.update(placeholder=l["placeholder_rag"]),
                gr.update(value=l["btn_send"]),
                gr.update(label=l["label_llm"]),
                gr.update(value=l["btn_load"]),
                gr.update(value=l["btn_clear"]),
                # Vision Chat
                gr.update(value=l["tab_vision_desc"]),
                gr.update(placeholder=l["placeholder_vis"]),
                gr.update(value=l["btn_send"]),
                gr.update(label=l["label_vlm"]),
                gr.update(label=l["label_vis_rag"]),
                gr.update(value=l["btn_load"]),
                gr.update(value=l["btn_clear"]),
                # STT
                gr.update(value=l["tab_stt_desc"]),
                gr.update(label=l["stt_audio_label"]),
                gr.update(label=l["label_stt"]),
                gr.update(label=l["label_stt_lang"]),
                gr.update(value=l["btn_load"]),
                gr.update(value=l["btn_transcribe"]),
                gr.update(label=l["label_res"]),
                # Knowledge Base
                gr.update(label=l["file_label"]),
                gr.update(label=l["label_vis_ret"]),
                gr.update(value=l["btn_index"]),
                gr.update(label=l["label_res"]),
                gr.update(label=l["label_ds_name"]),
                gr.update(label=l["label_ds_text"]),
                gr.update(label=l["label_ds_src"]),
                gr.update(value=l["btn_load_ds"]),
                gr.update(label=l["label_res"]),
                gr.update(value=l["header_docs"]),
                gr.update(value=l["btn_refresh"]),
                gr.update(value=l["btn_delete"]),
                gr.update(value=l["btn_clear_all"]),
                # About
                gr.update(value=l["about_tabs_title"]),
                gr.update(value=l["about_tabs_desc"]),
                gr.update(value=l["about_arch_title"]),
                gr.update(value=l["about_arch_desc"]),
                gr.update(value=l["about_speed_title"].format(device=DEVICE.upper())),
                gr.update(value=l["about_speed_desc"]),
                # Status bar
                get_index_stats(lk),
            )

        _lang_outputs = [
            lang_state, header_title, header_sub,
            # General Chat
            gen_desc, msg_gen, send_gen, model_dd_gen, reload_gen, clear_gen,
            # RAG Chat
            rag_desc, msg_rag, send_rag, model_dd_rag, reload_rag, clear_rag,
            # Vision Chat
            vis_desc, msg_vis, send_vis, vlm_dd, vis_rag_chk, load_vlm_btn, clear_vis,
            # STT
            stt_desc, stt_audio, stt_dd, stt_lang_dd, load_stt_btn, transcribe_btn, stt_output,
            # Knowledge Base
            file_up, vis_ret_dd, up_btn, up_msg,
            ds_name, ds_textcol, ds_srccol, ds_btn, ds_msg,
            kb_header, refresh_btn, delete_sel_btn, clear_all_btn,
            # About
            about_tabs_title_md, about_tabs_desc_md,
            about_arch_title_md, about_arch_desc_md,
            about_speed_title_md, about_speed_desc_md,
            # Status bar
            status_bar,
        ]

        lang_dropdown.change(switch_lang, [lang_dropdown], _lang_outputs)

        # Initialise to Khmer on load
        demo.load(lambda: switch_lang("Khmer"), outputs=_lang_outputs)

        return demo
if __name__ == "__main__":
    print("[RAG] Pre-loading embeddings and ChromaDB …")
    get_embed_model()
    get_chroma_collection()
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="violet"),
        css=CSS,
    )
