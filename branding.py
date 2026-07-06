"""
branding.py — LocalAiLab branding (logo, app name/version) and the
bilingual ℹ️ About tab content.
"""

import base64
from pathlib import Path

_LOGO_PATH = Path(__file__).parent / "image" / "logo.jpg"


def _load_logo_b64() -> str:
    try:
        data = _LOGO_PATH.read_bytes()
        return base64.b64encode(data).decode("utf-8")
    except Exception:
        return ""


DEVELOPER_LOGO_B64 = _load_logo_b64()
DEVELOPER_NAME = "LocalAiLab"
APP_NAME_EN = "Multipurpose AI Assistant"
APP_NAME_KH = "ជំនួយការ AI ពហុមុខងារ"
APP_VERSION = "0.0.1-beta"


# ──────────────────────────────────────────────────────────────────
# About tab — always shown bilingually (Khmer first, English below),
# independent of the language dropdown. Kept as static strings so it
# stays accurate to the current tab order / feature set at a glance.
# ──────────────────────────────────────────────────────────────────
def about_content_kh(device: str, version: str) -> str:
    return f"""
### 🔖 {APP_NAME_KH} — កំណែ {version}

ជំនួយការ AI ពហុមុខងារ ដែលដំណើរការនៅលើកុំព្យូទ័ររបស់អ្នកផ្ទាល់ ពហុភាសា (ខ្មែរ/អង់គ្លេស) និងពហុម៉ូដាល — សន្ទនាទូទៅ វិភាគឯកសារតាមរយៈ RAG យល់ដឹងរូបភាព
បំលែងសំឡេងទៅជាអក្សរ វិភាគទិន្នន័យ CSV/Excel ដោយ AI Agent និងគ្រប់គ្រងមូលដ្ឋានចំណេះដឹង — ដំណើរការទាំងស្រុងនៅលើកុំព្យូទ័ររបស់អ្នក
ដោយប្រើម៉ូដែល HuggingFace (transformers) ឬម៉ូដែលមូលដ្ឋាន GGUF តាមរយៈ llama.cpp។ បង្កើតដោយ LocalAiLab។

---

## ផ្ទាំង

| ផ្ទាំង | ការពិពណ៌នា |
|---|---|
| 💬 ការសន្ទនាទូទៅ | ការសន្ទនាផ្ទាល់ជាមួយ LLM — មិនមានការទាញយក |
| 📚 ការសន្ទនា RAG | ទាញយកពីមូលដ្ឋានចំណេះដឹងជាមុន រួចឆ្លើយ |
| 🔬 ស្រាវជ្រាវស៊ីជម្រៅ | Agent គ្រប់គ្រង + agent ស្វែងរកតាមអ៊ីនធឺណិត — បំបែកសំណួរ រៀបចំផែនការឡើងវិញ ស្វែងរកជាបន្តបន្ទាប់ រួចសរសេររបាយការណ៍ Markdown ដែលមានប្រភពយោង |
| 🖼️ ការសន្ទនាចក្ខុវិស័យ | យល់ដឹងរូបភាព ជាមួយបរិបទអត្ថបទ |
| 🎙️ និយាយទៅជាអក្សរ | បំលែងសំឡេងជាអក្សរ ដោយប្រើ Whisper |
| 📊 វិភាគទិន្នន័យ | Agent វិភាគ CSV/Excel បង្កើតក្រាហ្វិក និងរបាយការណ៍ |
| 📂 មូលដ្ឋានចំណេះដឹង | បង្ហោះ (PDF / TXT / MD / DOCX) និងគ្រប់គ្រងឯកសារ |

## ស្ថាបត្យកម្ម

| សមាសធាតុ | លម្អិត |
|---|---|
| LLM | ម៉ូដែល HuggingFace (Qwen3, Gemma-4) **ឬ** ម៉ូដែល GGUF មូលដ្ឋានតាមរយៈ llama.cpp |
| Vision LLM | SmolVLM / Qwen2.5-VL |
| Speech-to-Text | Whisper (រួមទាំងម៉ូដែលដែលបានកែសម្រួលសម្រាប់ភាសាខ្មែរ) |
| Embedding | BAAI/bge-m3 |
| Vector store | ChromaDB (`./chroma_db/`) |
| Visual Index | ColSmolVLM / ColQwen2 (`./visual_index/`) |
| ប្រភេទឯកសារដែលបញ្ចូលបាន | PDF, TXT, MD, DOCX |
| ប្រភេទភ្នាក់ងារ | `smolagents.CodeAgent` |
| UI | Gradio |

## ល្បឿនរំពឹងទុក ({device})

| កិច្ចការ | ពេលវេលា |
|---|---|
| បញ្ចូលឯកសារ | ១០–៦០ វិ |
| ឆ្លើយបែប General / RAG | ១–៥ នាទី |
| ឆ្លើយបែប Vision | ២–៨ នាទី |

> 💡 ម៉ូដែល GGUF (llama.cpp) ត្រូវបានស្កេនដោយស្វ័យប្រវត្តិពីថតដែលអ្នកកំណត់ (ប្រអប់ "📁 ថតម៉ូដែល GGUF" ខាងលើ ឬអថេរបរិស្ថាន `LLAMA_CPP_MODEL_DIR`) ហើយបង្ហាញនៅក្នុងបញ្ជីទម្លាក់ម៉ូដែលដូចគ្នានឹងម៉ូដែល HuggingFace។
"""


def about_content_en(device: str, version: str) -> str:
    return f"""
### 🔖 {APP_NAME_EN} — Version {version}

A local, bilingual (Khmer/English), multi-modal AI assistant — general chat, document RAG, vision chat, speech-to-text,
AI-driven CSV/Excel data analysis, and knowledge base management — running entirely on your own machine, using either
HuggingFace (transformers) models or local GGUF models via llama.cpp. Built by LocalAiLab.

---

## Tabs

| Tab | Description |
|---|---|
| 💬 General Chat | Direct LLM conversation — no retrieval |
| 📚 RAG Chat | Retrieves from knowledge base first, then answers |
| 🔬 Deep Research | Manager agent + web-search sub-agent — breaks the question down, re-plans as it goes, and writes a structured Markdown report with sources |
| 🖼️ Vision Chat | Image understanding with optional text context |
| 🎙️ Speech to Text | Transcribe audio into text using Whisper |
| 📊 Data Analysis | Agent analyzes CSV/Excel, builds charts and a report |
| 📂 Knowledge Base | Upload (PDF / TXT / MD / DOCX) & manage indexed documents |

## Architecture

| Component | Detail |
|---|---|
| LLM | HuggingFace models (Qwen3, Gemma-4) **or** local GGUF models via llama.cpp |
| Vision LLM | SmolVLM / Qwen2.5-VL |
| Speech-to-Text | Whisper (including Khmer-tuned variants) |
| Embedding | BAAI/bge-m3 |
| Vector store | ChromaDB (`./chroma_db/`) |
| Visual Index | ColSmolVLM / ColQwen2 (`./visual_index/`) |
| Supported document types | PDF, TXT, MD, DOCX |
| Agent type | `smolagents.CodeAgent` |
| UI | Gradio |

## Expected speed ({device})

| Task | Time |
|---|---|
| Index a document | 10–60 s |
| General / RAG answer | 1–5 min |
| Vision answer | 2–8 min |

> 💡 GGUF (llama.cpp) models are auto-discovered from whatever folder you configure (the "📁 GGUF Model Folder" box above, or the `LLAMA_CPP_MODEL_DIR` environment variable) and appear in the same model dropdown as the HuggingFace models.
"""
