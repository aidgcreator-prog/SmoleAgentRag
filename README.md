# 🤖 Multipurpose AI Assistant — by LocalAiLab

🇰🇭 **[ខ្មែរ](#-ខ្មែរ)** | 🇬🇧 **[English](#-english)**

---

## 🇰🇭 ខ្មែរ

ជំនួយការ AI ពហុមុខងារ ដោយ **LocalAiLab** ដែលដំណើរការនៅលើម៉ាស៊ីនរបស់អ្នកផ្ទាល់ បានបង្កើតឡើងដោយប្រើ [smolagents](https://github.com/huggingface/smolagents) មានចំណុចប្រទាក់ Gradio ការសន្ទនាទូទៅ វិភាគឯកសារតាមរយៈ RAG (Retrieval-Augmented Generation) ជាមួយការផ្ទុកទិន្នន័យជាប់លាប់ដោយប្រើ ChromaDB សមត្ថភាពពហុម៉ូដាល (រូបភាព/VLM) ការបំលែងសំឡេងទៅជាអក្សរ និងការវិភាគទិន្នន័យ CSV/Excel ដោយ AI Agent។ ប្រព័ន្ធនេះនឹងរកឃើញ និងប្រើប្រាស់ GPU របស់អ្នកដោយស្វ័យប្រវត្តិ (CUDA, AMD, ឬ Mac MPS) ប្រសិនបើមាន បើពុំនោះទេនឹងប្រើ CPU ជំនួសវិញ។ ក្រៅពីម៉ូដែល HuggingFace/transformers ជាលំនាំដើម កម្មវិធីនេះក៏អាចប្រើម៉ូដែលមូលដ្ឋាន **GGUF តាមរយៈ llama.cpp** ផងដែរ។

ចំណុចប្រទាក់អាចប្តូរភាសាបាន — អ្នកអាចប្តូររវាង **ភាសាខ្មែរ** និង **អង់គ្លេស** នៅពេលណាក៏បាន ដោយប្រើបញ្ជីទម្លាក់ភាសានៅជ្រុងខាងលើខាងស្តាំ។ ផ្ទាំង **ℹ️ អំពីកម្មវិធី** បង្ហាញព័ត៌មានទាំងពីរភាសាជានិច្ច (ខ្មែរខាងលើ អង់គ្លេសខាងក្រោម) ដោយមិនអាស្រ័យលើបញ្ជីទម្លាក់ភាសានោះទេ។

---

### 🏗️ ស្ថាបត្យកម្ម

| សមាសធាតុ | លំនាំដើម |
|---|---|
| LLM | `Qwen/Qwen3-0.6B` (HuggingFace) **ឬ** ម៉ូដែល `.gguf` មូលដ្ឋានតាមរយៈ llama.cpp |
| Vision LLM | `HuggingFaceTB/SmolVLM-500M-Instruct` |
| Speech-to-Text | `openai/whisper-small` |
| Embedding | `BAAI/bge-m3` |
| Visual Retriever | `vidore/colsmolvlm-v0.1` |
| Vector store | ChromaDB (`./chroma_db/`) |
| Visual Index | `vidore/colsmolvlm-v0.1` (`./visual_index/`) |
| ប្រភេទឯកសារបញ្ចូល | PDF, TXT, MD, DOCX |
| ប្រភេទភ្នាក់ងារ | `smolagents.CodeAgent` |
| UI | Gradio |

> **ចំណាំ**៖ ប្រព័ន្ធនេះមាន `HardwareManager` ដែលរកឃើញផ្នែករឹងរបស់អ្នកដោយស្វ័យប្រវត្តិ (NVIDIA, AMD, ឬ Apple Silicon) និងអាចជួយជួសជុលបរិយាកាសរបស់អ្នកតាមរយៈចំណុចប្រទាក់។

---

### 🚀 ការចាប់ផ្តើមរហ័ស

#### ១. ដំឡើង Dependencies

```bash
# បង្កើត venv (មិនចាំបាច់ ប៉ុន្តែគួរធ្វើ)
python -m venv .venv && source .venv/bin/activate

# ដំឡើង dependencies ស្នូល (គ្រប់គ្រងបណ្ណាល័យស្មុគស្មាញសម្រាប់រូបភាព សំឡេង និងការដកស្រង់អត្ថបទ រួមទាំង python-docx)
pip install -r requirements.txt

# ដំឡើង PyTorch (រកឃើញ CUDA/MPS/AMD ដោយស្វ័យប្រវត្តិប្រសិនបើមាន)
# ចំណាំ៖ ប្រសិនបើអ្នកមាន GPU NVIDIA អ្នកប្រហែលជាត្រូវដំឡើង wheel CUDA ជាក់លាក់
# ដូចបានរៀបរាប់ក្នុង SETUP.bat ឬតាមរយៈមុខងារ "Fix Environment" នៅក្នុងចំណុចប្រទាក់
pip install torch torchvision
```

**ជម្រើស — ការគាំទ្រម៉ូដែល llama.cpp (GGUF)**

`requirements.txt` រួមបញ្ចូល `llama-cpp-python` ជាមូលដ្ឋាន ប៉ុន្តែការដំឡើងធម្មតាតាមរយៈ `pip install -r requirements.txt` ផ្តល់ជូនតែ **wheel សម្រាប់ CPU ប៉ុណ្ណោះ**។ ដើម្បីទទួលបានការបង្កើនល្បឿនតាមរយៈ GPU និងកម្មវិធីដែលត្រូវនឹង CPU របស់អ្នកពិតប្រាកដ សូមប្រើ **SETUP.bat** ជំនួសវិញ — វានឹង៖

1. រកឃើញ GPU របស់អ្នក (NVIDIA/CUDA, AMD/ROCm ឬគ្មាន GPU)
2. សាកល្បង prebuilt wheel CUDA ជាច្រើនកំណែ ចាប់ពីត្រូវនឹង driver របស់អ្នកបំផុត
3. **ធ្វើតេស្តផ្ទុកម៉ូដែលពិតប្រាកដ** ដើម្បីប្រាកដថា wheel នោះដំណើរការលើ CPU របស់អ្នក (មិនគ្រាន់តែពិនិត្យថាការដំឡើងជោគជ័យ) — នេះការពារបញ្ហា "Illegal Instruction" ដែលកើតឡើងនៅពេល wheel សន្មតថា CPU គាំទ្រ AVX512 ប៉ុន្តែតាមពិតមិនគាំទ្រ
4. ប្រសិនបើគ្មាន wheel ណាដំណើរការ នឹងសាកល្បងសាងសង់ពី source ជាមួយ `CMAKE_ARGS=-DGGML_CUDA=on` (ត្រូវការ Visual Studio Build Tools + CUDA Toolkit)
5. ចុងក្រោយបំផុត ត្រលប់ទៅ CPU-only wheel ប្រសិនបើគ្មានផ្លូវ GPU ណាមួយដំណើរការ — ម៉ូដែល GGUF នៅតែប្រើប្រាស់បាន គ្រាន់តែយឺតជាង

ចំពោះការដំឡើងដោយដៃវិញ (Windows, NVIDIA):

```powershell
pip uninstall llama-cpp-python -y
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --force-reinstall --no-cache-dir
```

ប្តូរ `cu124` ទៅជាកំណែ CUDA ដែលត្រូវនឹង driver របស់អ្នក (`nvidia-smi` បង្ហាញវានៅជ្រុងខាងស្តាំខាងលើ)។ ប្រសិនបើ wheel នោះគាំង ("Illegal Instruction") សូមសាងសង់ពី source ជំនួសវិញ៖

```powershell
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
$env:FORCE_CMAKE = "1"
pip install llama-cpp-python --no-cache-dir --force-reinstall
```

**Flash Attention**: កម្មវិធីនេះស្នើសុំ `flash_attn=True` ដោយស្វ័យប្រវត្តិនៅពេលផ្ទុកម៉ូដែល GGUF ណាមួយ លុះត្រាតែប្រព័ន្ធរបស់អ្នកមិនមាន GPU ។ ម៉ូដែលមួយចំនួន (ជាពិសេសម៉ូដែលដែលប្រើ hybrid local/global attention ដូចជា Gemma-3/4) មិនគាំទ្រ Flash Attention នៅឡើយទេនៅក្នុង llama.cpp — ក្នុងករណីនេះកម្មវិធីនឹងបន្តដំណើរការធម្មតាដោយមិនប្រើ Flash Attention ដោយស្វ័យប្រវត្តិ ដោយមិនគាំង។

ដាក់ឯកសារ `.gguf` របស់អ្នកទៅក្នុងថតណាមួយដែលអ្នកចង់បាន — **គ្មានផ្លូវត្រូវបានកំណត់ស្រាប់ក្នុងកូដទេ**។ កំណត់ថតនោះតាមវិធីណាមួយក្នុងចំណោមពីរ៖ (១) ដាក់អថេរបរិស្ថាន `LLAMA_CPP_MODEL_DIR` មុនចាប់ផ្តើមកម្មវិធី ឬ (២) វាយផ្លូវថតចូលទៅក្នុងប្រអប់ "📁 ថតម៉ូដែល GGUF" នៅផ្នែកខាងលើចំណុចប្រទាក់ រួចចុច "🔍 ស្កេន" — មិនចាំបាច់ចាប់ផ្តើមកម្មវិធីឡើងវិញឡើយ។ កម្មវិធីនឹងស្កេនរកឯកសារ `.gguf` ក្នុងថតនោះ ហើយបង្ហាញឈ្មោះម៉ូដែលក្នុងបញ្ជីទម្លាក់ម៉ូដែលដូចគ្នានឹងម៉ូដែល HuggingFace ព្រមទាំងសម្គាល់ថាតើម៉ូដែលនោះនឹងដំណើរការលើ GPU ឬ CPU ។ មិនចាំបាច់ដំណើរការ `llama-server` ដាច់ដោយឡែកទេ — ម៉ូដែលត្រូវបានផ្ទុកផ្ទាល់នៅក្នុងដំណើរការរបស់ `app.py`។

នៅលើ Windows អ្នកគ្រាន់តែចុចពីរដងលើ **SETUP.bat** ដើម្បីដំឡើង Python បង្កើត virtual environment រកឃើញ GPU របស់អ្នក និងដំឡើងអ្វីៗគ្រប់យ៉ាងដោយស្វ័យប្រវត្តិ។

#### ២. បញ្ចូលឯកសាររបស់អ្នក

**តាមរយៈ CLI:**

```bash
# បញ្ចូល PDF
python index_docs.py --pdf ./docs/my_paper.pdf

# បញ្ចូលឯកសារអត្ថបទ ឬ markdown
python index_docs.py --txt ./docs/notes.md

# បញ្ចូលឯកសារ Word (.docx)
python index_docs.py --docx ./docs/report.docx

# បញ្ចូលថតមួយ (រកឃើញ .pdf / .txt / .md / .docx ដោយស្វ័យប្រវត្តិ)
python index_docs.py --dir ./docs

# បញ្ចូល HuggingFace dataset
python index_docs.py --hf-dataset m-ric/huggingface_doc --text-col text --source-col source

# បង្ហាញស្ថិតិការបញ្ចូលបច្ចុប្បន្ន
python index_docs.py --stats

# សម្អាតការបញ្ចូលទាំងអស់
python index_docs.py --clear
```

**តាមរយៈចំណុចប្រទាក់ (UI):**

ចាប់ផ្តើមកម្មវិធី ហើយប្រើផ្ទាំង **📂 មូលដ្ឋានចំណេះដឹង** (ដាក់នៅចុងបញ្ជីផ្ទាំង មុនផ្ទាំង ℹ️ អំពីកម្មវិធី)។ អាចទម្លាក់ឯកសារ PDF, TXT, MD ឬ DOCX។

#### ៣. ដំណើរការកម្មវិធី

```bash
python app.py
```

ឬនៅលើ Windows ចុចពីរដងលើ **RUN.bat**។

បើកកម្មវិធីរុករកតាមអាសយដ្ឋាន [http://localhost:7861](http://localhost:7861)។

> ច្រកលំនាំដើមគឺ **7861** (មិនមែន 7860 ដែលជាលំនាំដើមរបស់ Gradio) ដើម្បីជៀសវាងការប៉ះទង្គិចគ្នាប្រសិនបើអ្នកកំពុងដំណើរការកម្មវិធី Gradio ផ្សេងទៀតស្រាប់។ អ្នកអាចប្តូរវាបានដោយកែ `server_port` នៅក្នុង `app.py`។

---

### 🖥️ ចំណុចប្រទាក់អ្នកប្រើប្រាស់

កម្មវិធីនេះមានផ្ទាំងសំខាន់ៗចំនួនប្រាំពីរ តាមលំដាប់ដូចខាងក្រោម៖

1.  **💬 ការសន្ទនាទូទៅ**៖ ការសន្ទនាផ្ទាល់ជាមួយ LLM។ មិនមានការទាញយកឯកសារឡើយ។
2.  **📚 ការសន្ទនា RAG**៖ រាល់សំណួរនឹងទាញយកអត្ថបទពាក់ព័ន្ធពី ChromaDB ជាមុនសិន មុននឹងឆ្លើយ។
3.  **🖼️ ការសន្ទនាចក្ខុវិស័យ**៖ បង្ហោះរូបភាព ហើយសួរសំណួរ។ គាំទ្រ **ការទាញយកបរិបទចម្រុះ** (ទាញយកអត្ថបទរួមជាមួយការប្រាស្រ័យទាក់ទងជារូបភាព) និង **Visual RAG** (ទាញយករូបភាពតាមសំណួររបស់អ្នក)។
4.  **🎙️ និយាយទៅជាអក្សរ**៖ ថតសំឡេងពីមីក្រូហ្វូន ឬបង្ហោះឯកសារសំឡេង ហើយបំលែងវាទៅជាអក្សរដោយប្រើ Whisper។ គាំទ្រការរកឃើញភាសាដោយស្វ័យប្រវត្តិ ឬជ្រើសរើសភាសាជាក់លាក់ (អង់គ្លេស ខ្មែរ បារាំង ចិន ជប៉ុន)។
5.  **📊 វិភាគទិន្នន័យ**៖ បង្ហោះ CSV ឬ Excel ហើយឱ្យ AI Agent វិភាគទិន្នន័យ បង្កើតក្រាហ្វិក និងសរសេររបាយការណ៍ Markdown។
6.  **📂 មូលដ្ឋានចំណេះដឹង**៖ គ្រប់គ្រងឯកសារដែលបានបញ្ចូល (PDF/TXT/MD/DOCX) មើលតារាងឯកសារ និងសម្អាតការបញ្ចូល។
7.  **ℹ️ អំពីកម្មវិធី**៖ មើលស្ថានភាពប្រព័ន្ធ ស្ថាបត្យកម្ម និងការរំពឹងទុកលើដំណើរការម៉ូដែល — បង្ហាញជានិច្ចជាភាសាខ្មែរនិងអង់គ្លេសទាំងពីរ។

#### លក្ខណៈពិសេសសំខាន់ៗ
*   **ចំណុចប្រទាក់ពីរភាសា**៖ ចំណុចប្រទាក់ពេញលេញជាភាសាខ្មែរ/អង់គ្លេស — ប្តូរបានភ្លាមៗដោយប្រើបញ្ជីទម្លាក់ភាសា។
*   **ម៉ូដែលពីរប្រភេទ**៖ ប្រើម៉ូដែល HuggingFace/transformers ឬម៉ូដែលមូលដ្ឋាន GGUF តាមរយៈ llama.cpp ក្នុងបញ្ជីទម្លាក់ដូចគ្នា។
*   **ការគាំទ្រឯកសារ DOCX**៖ ការបញ្ចូលឯកសារគាំទ្រឯកសារ Word (.docx) បន្ថែមលើ PDF/TXT/MD រួមទាំងអត្ថបទក្នុងតារាង។
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
| ប្រែប្រួល | ម៉ូដែល `.gguf` ណាមួយនៅក្នុងថតដែលអ្នកកំណត់ (សូមមើលផ្នែក "ការគាំទ្រម៉ូដែល llama.cpp (GGUF)" ខាងលើ) (តាមរយៈ llama.cpp) |

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
├── app.py           # កម្មវិធីសំខាន់៖ agent, tools, ឧបករណ៍បញ្ចូលឯកសារ (PDF/TXT/MD/DOCX), STT, ចំណុចប្រទាក់ Gradio
├── index_docs.py    # ស្គ្រីប CLI សម្រាប់បញ្ចូលឯកសារ
├── requirements.txt # Dependencies របស់ Python (រួមទាំង python-docx, llama-cpp-python)
├── SETUP.bat        # កម្មវិធីដំឡើងលើ Windows ដោយចុចតែម្តង
├── RUN.bat          # កម្មវិធីដំណើរការលើ Windows ដោយចុចតែម្តង
├── README.md        # ឯកសារនេះ
├── chroma_db/       # បង្កើតដោយស្វ័យប្រវត្តិ; ការផ្ទុកទិន្នន័យជាប់លាប់របស់ ChromaDB
└── visual_index/    # បង្កើតដោយស្វ័យប្រវត្តិ; ការផ្ទុក Visual Index
```

---
---

## 🇬🇧 English

A local, multipurpose AI assistant built by **LocalAiLab** with [smolagents](https://github.com/huggingface/smolagents), featuring a Gradio UI, general chat, document RAG (Retrieval-Augmented Generation) with persistent ChromaDB storage, multi-modal capabilities (Vision/VLM), Speech-to-Text transcription, and AI-driven CSV/Excel data analysis. It automatically detects and uses your GPU (CUDA, AMD, or Mac MPS) if available, falling back to CPU otherwise. Besides the default HuggingFace/transformers models, the app can also run local **GGUF models via llama.cpp**.

The UI is fully bilingual — switch between **Khmer** and **English** at any time using the language dropdown in the top-right corner. The **ℹ️ About** tab always shows both languages (Khmer above, English below) regardless of that dropdown.

---

### 🏗️ Architecture

| Component | Default |
|---|---|
| LLM | `Qwen/Qwen3-0.6B` (HuggingFace) **or** a local `.gguf` model via llama.cpp |
| Vision LLM | `HuggingFaceTB/SmolVLM-500M-Instruct` |
| Speech-to-Text | `openai/whisper-small` |
| Embedding | `BAAI/bge-m3` |
| Visual Retriever | `vidore/colsmolvlm-v0.1` |
| Vector store | ChromaDB (`./chroma_db/`) |
| Visual Index | `vidore/colsmolvlm-v0.1` (`./visual_index/`) |
| Supported document types | PDF, TXT, MD, DOCX |
| Agent type | `smolagents.CodeAgent` |
| UI | Gradio |

> **Note**: The system includes a `HardwareManager` that automatically detects your hardware (NVIDIA, AMD, or Apple Silicon) and can help fix your environment via the UI.

---

### 🚀 Quick start

#### 1. Install dependencies

```bash
# Create a venv (optional but recommended)
python -m venv .venv && source .venv/bin/activate

# Install core dependencies (handles complex visual, audio, and text extraction libs, incl. python-docx)
pip install -r requirements.txt

# Install PyTorch (automatically detects CUDA/MPS/AMD if installed)
# Note: If you have an NVIDIA GPU, you may need to install the specific
# CUDA wheel as described in SETUP.bat or via the UI's "Fix Environment" feature.
pip install torch torchvision
```

**Optional — llama.cpp (GGUF) model support**

`requirements.txt` already includes `llama-cpp-python`, but `pip install -r requirements.txt` alone only gives you a **CPU-only** build. For real GPU acceleration and a build that actually matches your CPU, use **SETUP.bat** instead — it will:

1. Detect your GPU (NVIDIA/CUDA, AMD/ROCm, or none)
2. Probe several prebuilt CUDA wheel tiers, newest-compatible-with-your-driver first
3. **Actually load a model as a smoke test**, not just check that `pip install` succeeded — this catches the "Illegal Instruction" crash that happens when a prebuilt wheel assumes CPU features (like AVX512) your CPU doesn't have
4. Fall back to compiling from source with `CMAKE_ARGS=-DGGML_CUDA=on` if no prebuilt wheel works (requires Visual Studio Build Tools + CUDA Toolkit)
5. Fall back to a CPU-only build as a last resort — GGUF models still work, just slower

To do this by hand on Windows/NVIDIA:

```powershell
pip uninstall llama-cpp-python -y
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --force-reinstall --no-cache-dir
```

Swap `cu124` for whatever CUDA tier matches your driver (`nvidia-smi` shows it in the top-right). If that wheel crashes with an "Illegal Instruction" error, build from source instead:

```powershell
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
$env:FORCE_CMAKE = "1"
pip install llama-cpp-python --no-cache-dir --force-reinstall
```

**Flash Attention**: the app requests `flash_attn=True` automatically for any GGUF model whenever a GPU is available. Some architectures (notably hybrid local/global attention models like Gemma-3/4) don't yet support Flash Attention in llama.cpp — in that case the app transparently falls back to running without it instead of crashing.

Drop your `.gguf` files into any folder you like — **no path is hardcoded in the code**. Point the app at that folder either of two ways: (1) set the `LLAMA_CPP_MODEL_DIR` environment variable before launching, or (2) type the folder path into the "📁 GGUF Model Folder" box at the top of the UI and click "🔍 Scan" — no restart needed. The app then auto-discovers the `.gguf` files in that folder, lists them in the same model dropdown as the HuggingFace models, and tags each one as GPU- or CPU-mode based on what's actually available. No separate `llama-server` process is required; models load directly inside `app.py`.

On Windows, you can instead just double-click **SETUP.bat** to install Python, create the virtual environment, detect your GPU, and install everything automatically.

#### 2. Index your documents

**Via CLI:**

```bash
# Index a PDF
python index_docs.py --pdf ./docs/my_paper.pdf

# Index a text or markdown file
python index_docs.py --txt ./docs/notes.md

# Index a Word document
python index_docs.py --docx ./docs/report.docx

# Index a folder (auto-detects .pdf / .txt / .md / .docx)
python index_docs.py --dir ./docs

# Index a HuggingFace dataset
python index_docs.py --hf-dataset m-ric/huggingface_doc --text-col text --source-col source

# Show current index statistics
python index_docs.py --stats

# Clear the entire index
python index_docs.py --clear
```

**Via UI:**

Launch the app and use the **📂 Knowledge Base** tab (last tab before ℹ️ About). PDF, TXT, MD, and DOCX uploads are all supported.

#### 3. Launch the app

```bash
python app.py
```

Or on Windows, double-click **RUN.bat**.

Open [http://localhost:7861](http://localhost:7861) in your browser.

> The default port is **7861** (not Gradio's usual 7860) to avoid conflicts if you're already running another Gradio app on your machine. You can change this by editing `server_port` in `app.py`.

---

### 🖥️ User Interface

The app is organized into seven main tabs, in this order:

1.  **💬 General Chat**: Direct conversation with the LLM. No document retrieval.
2.  **📚 RAG Chat**: Every question retrieves relevant text chunks from ChromaDB before answering.
3.  **🖼️ Vision Chat**: Upload images and ask questions. Supports **Hybrid Context Retrieval** (retrieving text context alongside visual interaction) and **Visual RAG** (retrieving images based on your query).
4.  **🎙️ Speech to Text**: Record from your microphone or upload an audio file and transcribe it to text using Whisper. Supports auto-detect or forced-language transcription (English, Khmer, French, Chinese, Japanese).
5.  **📊 Data Analysis**: Upload a CSV or Excel file and have the AI agent explore it, build charts, and write a Markdown report.
6.  **📂 Knowledge Base**: Manage your indexed documents (PDF/TXT/MD/DOCX), view the document table, and clear the index.
7.  **ℹ️ About**: View system status, architecture, and model performance expectations — always shown bilingually (Khmer + English).

#### Key Features
*   **Bilingual UI**: Full Khmer/English interface — switch instantly with the language dropdown.
*   **Two model backends**: Run HuggingFace/transformers models or local GGUF models via llama.cpp side by side, in the same dropdown.
*   **DOCX support**: Document indexing supports Word (.docx) files in addition to PDF/TXT/MD, including text inside tables.
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
| varies | Any `.gguf` model placed in your configured folder (see "llama.cpp (GGUF) model support" above) (via llama.cpp) |

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
├── app.py           # Main app: agent, tools, indexing helpers (PDF/TXT/MD/DOCX), STT, Gradio UI
├── index_docs.py    # CLI indexing script
├── requirements.txt # Python dependencies (incl. python-docx, llama-cpp-python)
├── SETUP.bat        # Windows one-click installer
├── RUN.bat          # Windows one-click launcher
├── README.md        # This file
├── chroma_db/       # Auto-created; ChromaDB persistent storage
└── visual_index/    # Auto-created; Visual index storage
```
