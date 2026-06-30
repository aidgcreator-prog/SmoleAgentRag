# 🔍 RAG Agent — smolagents · ChromaDB · Multi-Modal · GPU/CPU

🇬🇧 **[English](#-english)** | 🇰🇭 **[ខ្មែរ](#-ខ្មែរ)**

---

## 🇬🇧 English

A versatile local RAG (Retrieval-Augmented Generation) agent built with [smolagents](https://github.com/huggingface/smolagents), featuring a Gradio UI, persistent ChromaDB storage, multi-modal capabilities (Vision/VLM), and Speech-to-Text transcription. It automatically detects and uses your GPU (CUDA, AMD, or Mac MPS) if available, falling back to CPU otherwise.

The UI is fully bilingual — switch between **Khmer** and **English** at any time using the language dropdown in the top-right corner.

---

### 🏗️ Architecture

| Component | Default |
|---|---|
| LLM | `Qwen/Qwen3-0.6B` |
| Vision LLM | `HuggingFaceTB/SmolVLM-500M-Instruct` |
| Speech-to-Text | `openai/whisper-small` |
| Embedding | `BAAI/bge-m3` |
| Visual Retriever | `vidore/colsmolvlm-v0.1` |
| Vector store | ChromaDB (`./chroma_db/`) |
| Visual Index | `vidore/colsmolvlm-v0.1` (`./visual_index/`) |
| Agent type | `smolagents.CodeAgent` |
| UI | Gradio |

> **Note**: The system includes a `HardwareManager` that automatically detects your hardware (NVIDIA, AMD, or Apple Silicon) and can help fix your environment via the UI.

---

### 🚀 Quick start

#### 1. Install dependencies

```bash
# Create a venv (optional but recommended)
python -m venv .venv && source .venv/bin/activate

# Install core dependencies (handles complex visual, audio, and text extraction libs)
pip install -r requirements.txt

# Install PyTorch (automatically detects CUDA/MPS/AMD if installed)
# Note: If you have an NVIDIA GPU, you may need to install the specific
# CUDA wheel as described in SETUP.bat or via the UI's "Fix Environment" feature.
pip install torch torchvision
```

On Windows, you can instead just double-click **SETUP.bat** to install Python, create the virtual environment, detect your GPU, and install everything automatically.

#### 2. Index your documents

**Via CLI:**

```bash
# Index a PDF
python index_docs.py --pdf ./docs/my_paper.pdf

# Index a text or markdown file
python index_docs.py --txt ./docs/notes.md

# Index a folder
python index_docs.py --dir ./docs

# Index a HuggingFace dataset
python index_docs.py --hf-dataset m-ric/huggingface_doc --text-col text --source-col source

# Show current index statistics
python index_docs.py --stats

# Clear the entire index
python index_docs.py --clear
```

**Via UI:**

Launch the app and use the **📂 Knowledge Base** tab.

#### 3. Launch the app

```bash
python app.py
```

Or on Windows, double-click **RUN.bat**.

Open [http://localhost:7861](http://localhost:7861) in your browser.

> The default port is **7861** (not Gradio's usual 7860) to avoid conflicts if you're already running another Gradio app on your machine. You can change this by editing `server_port` in `app.py`.

---

### 🖥️ User Interface

The app is organized into six main tabs:

1.  **💬 General Chat**: Direct conversation with the LLM. No document retrieval.
2.  **📚 RAG Chat**: Every question retrieves relevant text chunks from ChromaDB before answering.
3.  **🖼️ Vision Chat**: Upload images and ask questions. Supports **Hybrid Context Retrieval** (retrieving text context alongside visual interaction) and **Visual RAG** (retrieving images based on your query).
4.  **🎙️ Speech to Text**: Record from your microphone or upload an audio file and transcribe it to text using Whisper. Supports auto-detect or forced-language transcription (English, Khmer, French, Chinese, Japanese).
5.  **📂 Knowledge Base**: Manage your indexed documents, view the document table, and clear the index.
6.  **ℹ️ About**: View system status, hardware detection, and model performance expectations.

#### Key Features
*   **Bilingual UI**: Full Khmer/English interface — switch instantly with the language dropdown.
*   **Environment Self-Fixing**: Use the "Fix Environment" button in the UI to automatically install the correct PyTorch version for your GPU.
*   **Reasoning Visibility**: Model `<think>` tags are automatically rendered into a clean, collapsible UI element.
*   **Visual Retriever Selection**: Choose between different visual retrievers (e.g., `colsmolvlm` or `colqwen2`) directly in the UI.
*   **Speech-to-Text**: Transcribe audio in multiple languages (including Khmer) using Whisper, with selectable model sizes.

---

### ⚙️ Model Settings

You can change models at runtime via the UI's model selection dropdowns.

#### Text LLMs
| VRAM/RAM | Recommended model |
|---|---|
| ~1.2 GB | `Qwen/Qwen3-0.6B` (fastest) |
| ~3 GB | `Qwen/Qwen3-1.7B` |
| ~7 GB | `Qwen/Qwen3-4B` |
| ~4 GB | `google/gemma-4-E2B-it` |

#### Vision LLMs (VLM)
| RAM | Recommended model |
|---|---|
| ~0.5 GB | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| ~1 GB | `HuggingFaceTB/SmolVLM-500M-Instruct` |
| ~6 GB | `Qwen/Qwen2.5-VL-3B-Instruct` |

#### Speech-to-Text (Whisper)
| RAM | Recommended model |
|---|---|
| ~1 GB | `openai/whisper-tiny` (fastest) |
| ~1 GB | `openai/whisper-base` |
| ~2 GB | `openai/whisper-small` (recommended) |
| ~10 GB | `openai/whisper-large-v3` (best accuracy, multilingual incl. Khmer) |

---

### 📂 File structure

```
.
├── app.py           # Main app: agent, tools, indexing helpers, STT, Gradio UI
├── index_docs.py    # CLI indexing script
├── requirements.txt # Python dependencies
├── SETUP.bat        # Windows one-click installer
├── RUN.bat          # Windows one-click launcher
├── README.md        # This file
├── chroma_db/       # Auto-created; ChromaDB persistent storage
└── visual_index/    # Auto-created; Visual index storage
```

---
---

## 🇰🇭 ខ្មែរ

ភ្នាក់ងារ RAG (Retrieval-Augmented Generation) ដែលដំណើរការនៅលើម៉ាស៊ីនរបស់អ្នកផ្ទាល់ បានបង្កើតឡើងដោយប្រើ [smolagents](https://github.com/huggingface/smolagents) មានចំណុចប្រទាក់ Gradio ការផ្ទុកទិន្នន័យជាប់លាប់ដោយប្រើ ChromaDB សមត្ថភាពពហុម៉ូដាល (រូបភាព/VLM) និងការបំលែងសំឡេងទៅជាអក្សរ។ ប្រព័ន្ធនេះនឹងរកឃើញ និងប្រើប្រាស់ GPU របស់អ្នកដោយស្វ័យប្រវត្តិ (CUDA, AMD, ឬ Mac MPS) ប្រសិនបើមាន បើពុំនោះទេនឹងប្រើ CPU ជំនួសវិញ។

ចំណុចប្រទាក់អាចប្តូរភាសាបាន — អ្នកអាចប្តូររវាង **ភាសាខ្មែរ** និង **អង់គ្លេស** នៅពេលណាក៏បាន ដោយប្រើបញ្ជីទម្លាក់ភាសានៅជ្រុងខាងលើខាងស្តាំ។

---

### 🏗️ ស្ថាបត្យកម្ម

| សមាសធាតុ | លំនាំដើម |
|---|---|
| LLM | `Qwen/Qwen3-0.6B` |
| Vision LLM | `HuggingFaceTB/SmolVLM-500M-Instruct` |
| Speech-to-Text | `openai/whisper-small` |
| Embedding | `BAAI/bge-m3` |
| Visual Retriever | `vidore/colsmolvlm-v0.1` |
| Vector store | ChromaDB (`./chroma_db/`) |
| Visual Index | `vidore/colsmolvlm-v0.1` (`./visual_index/`) |
| ប្រភេទភ្នាក់ងារ | `smolagents.CodeAgent` |
| UI | Gradio |

> **ចំណាំ**៖ ប្រព័ន្ធនេះមាន `HardwareManager` ដែលរកឃើញផ្នែករឹងរបស់អ្នកដោយស្វ័យប្រវត្តិ (NVIDIA, AMD, ឬ Apple Silicon) និងអាចជួយជួសជុលបរិយាកាសរបស់អ្នកតាមរយៈចំណុចប្រទាក់។

---

### 🚀 ការចាប់ផ្តើមរហ័ស

#### ១. ដំឡើង Dependencies

```bash
# បង្កើត venv (មិនចាំបាច់ ប៉ុន្តែគួរធ្វើ)
python -m venv .venv && source .venv/bin/activate

# ដំឡើង dependencies ស្នូល (គ្រប់គ្រងបណ្ណាល័យស្មុគស្មាញសម្រាប់រូបភាព សំឡេង និងការដកស្រង់អត្ថបទ)
pip install -r requirements.txt

# ដំឡើង PyTorch (រកឃើញ CUDA/MPS/AMD ដោយស្វ័យប្រវត្តិប្រសិនបើមាន)
# ចំណាំ៖ ប្រសិនបើអ្នកមាន GPU NVIDIA អ្នកប្រហែលជាត្រូវដំឡើង wheel CUDA ជាក់លាក់
# ដូចបានរៀបរាប់ក្នុង SETUP.bat ឬតាមរយៈមុខងារ "Fix Environment" នៅក្នុងចំណុចប្រទាក់
pip install torch torchvision
```

នៅលើ Windows អ្នកគ្រាន់តែចុចពីរដងលើ **SETUP.bat** ដើម្បីដំឡើង Python បង្កើត virtual environment រកឃើញ GPU របស់អ្នក និងដំឡើងអ្វីៗគ្រប់យ៉ាងដោយស្វ័យប្រវត្តិ។

#### ២. បញ្ចូលឯកសាររបស់អ្នក

**តាមរយៈ CLI:**

```bash
# បញ្ចូល PDF
python index_docs.py --pdf ./docs/my_paper.pdf

# បញ្ចូលឯកសារអត្ថបទ ឬ markdown
python index_docs.py --txt ./docs/notes.md

# បញ្ចូលថតមួយ
python index_docs.py --dir ./docs

# បញ្ចូល HuggingFace dataset
python index_docs.py --hf-dataset m-ric/huggingface_doc --text-col text --source-col source

# បង្ហាញស្ថិតិការបញ្ចូលបច្ចុប្បន្ន
python index_docs.py --stats

# សម្អាតការបញ្ចូលទាំងអស់
python index_docs.py --clear
```

**តាមរយៈចំណុចប្រទាក់ (UI):**

ចាប់ផ្តើមកម្មវិធី ហើយប្រើផ្ទាំង **📂 មូលដ្ឋានចំណេះដឹង**។

#### ៣. ដំណើរការកម្មវិធី

```bash
python app.py
```

ឬនៅលើ Windows ចុចពីរដងលើ **RUN.bat**។

បើកកម្មវិធីរុករកតាមអាសយដ្ឋាន [http://localhost:7861](http://localhost:7861)។

> ច្រកលំនាំដើមគឺ **7861** (មិនមែន 7860 ដែលជាលំនាំដើមរបស់ Gradio) ដើម្បីជៀសវាងការប៉ះទង្គិចគ្នាប្រសិនបើអ្នកកំពុងដំណើរការកម្មវិធី Gradio ផ្សេងទៀតស្រាប់។ អ្នកអាចប្តូរវាបានដោយកែ `server_port` នៅក្នុង `app.py`។

---

### 🖥️ ចំណុចប្រទាក់អ្នកប្រើប្រាស់

កម្មវិធីនេះមានផ្ទាំងសំខាន់ៗចំនួនប្រាំមួយ៖

1.  **💬 ការសន្ទនាទូទៅ**៖ ការសន្ទនាផ្ទាល់ជាមួយ LLM។ មិនមានការទាញយកឯកសារឡើយ។
2.  **📚 ការសន្ទនា RAG**៖ រាល់សំណួរនឹងទាញយកអត្ថបទពាក់ព័ន្ធពី ChromaDB ជាមុនសិន មុននឹងឆ្លើយ។
3.  **🖼️ ការសន្ទនាចក្ខុវិស័យ**៖ បង្ហោះរូបភាព ហើយសួរសំណួរ។ គាំទ្រ **ការទាញយកបរិបទចម្រុះ** (ទាញយកអត្ថបទរួមជាមួយការប្រាស្រ័យទាក់ទងជារូបភាព) និង **Visual RAG** (ទាញយករូបភាពតាមសំណួររបស់អ្នក)។
4.  **🎙️ និយាយទៅជាអក្សរ**៖ ថតសំឡេងពីមីក្រូហ្វូន ឬបង្ហោះឯកសារសំឡេង ហើយបំលែងវាទៅជាអក្សរដោយប្រើ Whisper។ គាំទ្រការរកឃើញភាសាដោយស្វ័យប្រវត្តិ ឬជ្រើសរើសភាសាជាក់លាក់ (អង់គ្លេស ខ្មែរ បារាំង ចិន ជប៉ុន)។
5.  **📂 មូលដ្ឋានចំណេះដឹង**៖ គ្រប់គ្រងឯកសារដែលបានបញ្ចូល មើលតារាងឯកសារ និងសម្អាតការបញ្ចូល។
6.  **ℹ️ អំពីកម្មវិធី**៖ មើលស្ថានភាពប្រព័ន្ធ ការរកឃើញផ្នែករឹង និងការរំពឹងទុកលើដំណើរការម៉ូដែល។

#### លក្ខណៈពិសេសសំខាន់ៗ
*   **ចំណុចប្រទាក់ពីរភាសា**៖ ចំណុចប្រទាក់ពេញលេញជាភាសាខ្មែរ/អង់គ្លេស — ប្តូរបានភ្លាមៗដោយប្រើបញ្ជីទម្លាក់ភាសា។
*   **ការជួសជុលបរិយាកាសដោយខ្លួនឯង**៖ ប្រើប៊ូតុង "Fix Environment" ក្នុងចំណុចប្រទាក់ ដើម្បីដំឡើង PyTorch ត្រឹមត្រូវសម្រាប់ GPU របស់អ្នកដោយស្វ័យប្រវត្តិ។
*   **ភាពមើលឃើញនៃការគិត**៖ ស្លាក `<think>` របស់ម៉ូដែលនឹងបង្ហាញជាផ្នែកដែលអាចពង្រីក/បង្រួមបាន។
*   **ការជ្រើសរើស Visual Retriever**៖ ជ្រើសរើសរវាង visual retriever ផ្សេងៗគ្នា (ឧ. `colsmolvlm` ឬ `colqwen2`) ដោយផ្ទាល់ក្នុងចំណុចប្រទាក់។
*   **និយាយទៅជាអក្សរ**៖ បំលែងសំឡេងជាច្រើនភាសា (រួមទាំងភាសាខ្មែរ) ដោយប្រើ Whisper ជាមួយការជ្រើសរើសទំហំម៉ូដែល។

---

### ⚙️ ការកំណត់ម៉ូដែល

អ្នកអាចប្តូរម៉ូដែលនៅពេលដំណើរការ តាមរយៈបញ្ជីទម្លាក់ជ្រើសរើសម៉ូដែលក្នុងចំណុចប្រទាក់។

#### Text LLMs
| VRAM/RAM | ម៉ូដែលដែលណែនាំ |
|---|---|
| ~1.2 GB | `Qwen/Qwen3-0.6B` (លឿនបំផុត) |
| ~3 GB | `Qwen/Qwen3-1.7B` |
| ~7 GB | `Qwen/Qwen3-4B` |
| ~4 GB | `google/gemma-4-E2B-it` |

#### Vision LLMs (VLM)
| RAM | ម៉ូដែលដែលណែនាំ |
|---|---|
| ~0.5 GB | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| ~1 GB | `HuggingFaceTB/SmolVLM-500M-Instruct` |
| ~6 GB | `Qwen/Qwen2.5-VL-3B-Instruct` |

#### Speech-to-Text (Whisper)
| RAM | ម៉ូដែលដែលណែនាំ |
|---|---|
| ~1 GB | `openai/whisper-tiny` (លឿនបំផុត) |
| ~1 GB | `openai/whisper-base` |
| ~2 GB | `openai/whisper-small` (ណែនាំ) |
| ~10 GB | `openai/whisper-large-v3` (ត្រឹមត្រូវបំផុត គាំទ្រច្រើនភាសារួមទាំងខ្មែរ) |

---

### 📂 រចនាសម្ព័ន្ធឯកសារ

```
.
├── app.py           # កម្មវិធីសំខាន់៖ agent, tools, ឧបករណ៍បញ្ចូលឯកសារ, STT, ចំណុចប្រទាក់ Gradio
├── index_docs.py    # ស្គ្រីប CLI សម្រាប់បញ្ចូលឯកសារ
├── requirements.txt # Dependencies របស់ Python
├── SETUP.bat        # កម្មវិធីដំឡើងលើ Windows ដោយចុចតែម្តង
├── RUN.bat          # កម្មវិធីដំណើរការលើ Windows ដោយចុចតែម្តង
├── README.md        # ឯកសារនេះ
├── chroma_db/       # បង្កើតដោយស្វ័យប្រវត្តិ; ការផ្ទុកទិន្នន័យជាប់លាប់របស់ ChromaDB
└── visual_index/    # បង្កើតដោយស្វ័យប្រវត្តិ; ការផ្ទុក Visual Index
```
