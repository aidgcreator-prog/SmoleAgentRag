# 🤖 Multipurpose AI Assistant — by LocalAiLab

🇰🇭 **[ខ្មែរ](#-ខ្មែរ)** | 🇬🇧 **[English](#-english)**

---

## 🇰🇭 ខ្មែរ

ជំនួយការ AI ពហុមុខងារ ដោយ **LocalAiLab** ដែលដំណើរការនៅលើម៉ាស៊ីនរបស់អ្នកផ្ទាល់ បានបង្កើតឡើងដោយប្រើ [smolagents](https://github.com/huggingface/smolagents)។ មានចំណុចប្រទាក់ Gradio ការសន្ទនាទូទៅ វិភាគឯកសារតាមរយៈ RAG (Retrieval-Augmented Generation) ជាមួយការផ្ទុកទិន្នន័យជាប់លាប់ដោយ ChromaDB សមត្ថភាពពហុម៉ូដាល (រូបភាព/VLM) ការបំលែងសំឡេងទៅជាអក្សរ និងការវិភាគទិន្នន័យ CSV/Excel ដោយ AI Agent។ ប្រព័ន្ធនេះរកឃើញ និងប្រើ GPU របស់អ្នកដោយស្វ័យប្រវត្តិ (CUDA, AMD, ឬ Mac MPS) ប្រសិនបើមាន ឬប្រើ CPU ជំនួសវិញ។ ក្រៅពីម៉ូដែល HuggingFace/transformers ជាលំនាំដើម កម្មវិធីនេះក៏អាចប្រើម៉ូដែលមូលដ្ឋាន **GGUF តាមរយៈ llama.cpp** ផងដែរ។

ចំណុចប្រទាក់អាចប្តូរភាសាបានភ្លាមៗ (**ខ្មែរ** ⇄ **អង់គ្លេស**) នៅជ្រុងខាងលើស្តាំ។ ផ្ទាំង **ℹ️ អំពីកម្មវិធី** បង្ហាញព័ត៌មានទាំងពីរភាសាជានិច្ច (ខ្មែរខាងលើ អង់គ្លេសខាងក្រោម) ដោយមិនអាស្រ័យលើបញ្ជីទម្លាក់ភាសានោះទេ។

### 🆕 អ្វីដែលថ្មីក្នុងកំណែនេះ

- **🔬 ផ្ទាំង Deep Research ថ្មី**៖ agent គ្រប់គ្រង + agent ស្វែងរកតាមអ៊ីនធឺណិតដាច់ដោយឡែក បំបែកសំណួរជាសំណួររង រៀបចំផែនការឡើងវិញឥតឈប់ឈរ (`planning_interval`) រួចសរសេររបាយការណ៍ Markdown ដែលមានប្រភពយោងផ្ទៀងផ្ទាត់
- **ម៉ូដែលថ្មី**៖ គាំទ្រគ្រួសារ **Gemma 4** (E2B/12B/26B-A4B/31B, អាជ្ញាប័ណ្ណ Apache 2.0) និង **Qwen3.6** (27B dense / 35B-A3B MoE) — ត្រូវការ `transformers` កំណែថ្មីជាងមុន (សូមមើលតារាង Text LLMs ខាងក្រោម)
- **🧠 ចងចាំការសន្ទនា (Conversation Memory — កំពុងសាកល្បង)**៖ ប្រអប់ថ្មីនៅគ្រប់ផ្ទាំង agentic អនុញ្ញាតឱ្យសំណួរបន្តអាចយោងលើអ្វីដែលបាននិយាយពីមុន (RAM-only មិនរក្សាទុកទៅថាសទេ)
- **🧠 វិន្ដូបរិបទ (Context Window) ដែលកំណត់បានតាមចិត្ត**៖ ជ្រើសរើសពី 4K ដល់ 128K សម្រាប់ម៉ូដែល GGUF ដោយផ្ទាល់ក្នុង UI
- **ការរកឃើញ GPU មិនត្រូវគ្នា**៖ បង្ហាញការព្រមានច្បាស់លាស់ក្នុង UI ប្រសិនបើ GPU ចាស់ (ឧ. Pascal/sm_6x) មិនត្រូវបានគាំទ្រដោយ PyTorch build បច្ចុប្បន្ន ជំនួសឱ្យការគាំងស្ងាត់ៗ
- **Data Analysis Agent** ឥឡូវអាចដំឡើង Python package ខ្វះខាតដោយខ្លួនឯង (`install_package` tool) ហើយធ្វើ EDA ពេញលេញជាមួយក្រាហ្វិកច្រើន

### 📑 មាតិកា
- [ស្ថាបត្យកម្ម](#️-ស្ថាបត្យកម្ម)
- [ការចាប់ផ្តើមរហ័ស](#-ការចាប់ផ្តើមរហ័ស)
- [ចំណុចប្រទាក់អ្នកប្រើប្រាស់](#️-ចំណុចប្រទាក់អ្នកប្រើប្រាស់)
- [ការកំណត់ម៉ូដែល](#️-ការកំណត់ម៉ូដែល)
- [រចនាសម្ព័ន្ធឯកសារ](#-រចនាសម្ព័ន្ធឯកសារ)

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

> **ចំណាំ**៖ `HardwareManager` រកឃើញផ្នែករឹងរបស់អ្នកដោយស្វ័យប្រវត្តិ (NVIDIA, AMD, ឬ Apple Silicon) ហើយអាចជួយជួសជុលបរិយាកាសរបស់អ្នកតាមរយៈប៊ូតុង "Fix Environment" ក្នុងចំណុចប្រទាក់។

---

### 🚀 ការចាប់ផ្តើមរហ័ស

#### ១. ដំឡើង Dependencies

```bash
# បង្កើត venv (មិនចាំបាច់ ប៉ុន្តែគួរធ្វើ)
python -m venv .venv && source .venv/bin/activate

# ដំឡើង dependencies ស្នូល (រូបភាព សំឡេង និងការដកស្រង់អត្ថបទ រួមទាំង python-docx)
pip install -r requirements.txt

# ដំឡើង PyTorch (រកឃើញ CUDA/MPS/AMD ដោយស្វ័យប្រវត្តិប្រសិនបើមាន)
pip install torch torchvision
```

នៅលើ Windows គ្រាន់តែចុចពីរដងលើ **SETUP.bat** ដើម្បីធ្វើអ្វីៗខាងលើដោយស្វ័យប្រវត្តិ (រកឃើញ GPU ដំឡើង PyTorch ត្រឹមត្រូវ ជាដើម)។

<details>
<summary><strong>ជម្រើស — ការគាំទ្រម៉ូដែល llama.cpp (GGUF)</strong></summary>

`requirements.txt` រួមបញ្ចូល `llama-cpp-python` ជាមូលដ្ឋាន ប៉ុន្តែការដំឡើងធម្មតាតាមរយៈ `pip install -r requirements.txt` ផ្តល់ជូនតែ **wheel សម្រាប់ CPU ប៉ុណ្ណោះ**។ ដើម្បីទទួលបានការបង្កើនល្បឿនតាមរយៈ GPU ប្រើ **SETUP.bat** ជំនួសវិញ — វានឹង៖

1. រកឃើញ GPU របស់អ្នក (NVIDIA/CUDA, AMD/ROCm ឬគ្មាន GPU)
2. សាកល្បង prebuilt wheel CUDA ចាប់ពីត្រូវនឹង driver របស់អ្នកបំផុត
3. **ធ្វើតេស្តផ្ទុកម៉ូដែលពិតប្រាកដ** ដើម្បីការពារបញ្ហា "Illegal Instruction" (wheel សន្មតថា CPU គាំទ្រ AVX512 ប៉ុន្តែតាមពិតមិនគាំទ្រ)
4. ប្រសិនបើគ្មាន wheel ណាដំណើរការ សាកល្បងសាងសង់ពី source ជាមួយ `CMAKE_ARGS=-DGGML_CUDA=on` (ត្រូវការ Visual Studio Build Tools + CUDA Toolkit)
5. ចុងក្រោយបំផុត ត្រលប់ទៅ CPU-only wheel — ម៉ូដែល GGUF នៅតែប្រើប្រាស់បាន គ្រាន់តែយឺតជាង

ចំពោះការដំឡើងដោយដៃវិញ (Windows, NVIDIA):

```powershell
pip uninstall llama-cpp-python -y
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --force-reinstall --no-cache-dir
```

ប្តូរ `cu124` ទៅជាកំណែ CUDA ដែលត្រូវនឹង driver របស់អ្នក (`nvidia-smi` បង្ហាញវានៅជ្រុងខាងស្តាំខាងលើ)។ ប្រសិនបើ wheel គាំង ("Illegal Instruction") សូមសាងសង់ពី source ជំនួសវិញ៖

```powershell
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
$env:FORCE_CMAKE = "1"
pip install llama-cpp-python --no-cache-dir --force-reinstall
```

**Flash Attention**៖ កម្មវិធីស្នើសុំ `flash_attn=True` ដោយស្វ័យប្រវត្តិនៅពេលមាន GPU។ ម៉ូដែលមួយចំនួន (ឧ. hybrid local/global attention ដូចជា Gemma-3/4) មិនគាំទ្រ Flash Attention នៅឡើយក្នុង llama.cpp — កម្មវិធីនឹងបន្តដំណើរការដោយមិនប្រើវាដោយស្វ័យប្រវត្តិ ដោយមិនគាំង។

**ថតម៉ូដែល GGUF**៖ ដាក់ឯកសារ `.gguf` ទៅក្នុងថតណាមួយ — គ្មានផ្លូវត្រូវបានកំណត់ស្រាប់ក្នុងកូដទេ។ កំណត់ថតតាមវិធីណាមួយ៖ (១) អថេរបរិស្ថាន `LLAMA_CPP_MODEL_DIR` មុនចាប់ផ្តើមកម្មវិធី ឬ (២) វាយផ្លូវថតចូលទៅក្នុងប្រអប់ "📁 ថតម៉ូដែល GGUF" នៅផ្នែកខាងលើចំណុចប្រទាក់ រួចចុច "🔍 ស្កេន" (មិនចាំបាច់ចាប់ផ្តើមឡើងវិញ)។ មិនចាំបាច់ដំណើរការ `llama-server` ដាច់ដោយឡែកទេ — ម៉ូដែលត្រូវបានផ្ទុកផ្ទាល់នៅក្នុងដំណើរការរបស់ `app.py`។

</details>

#### ២. បញ្ចូលឯកសាររបស់អ្នក

**តាមរយៈ CLI:**

```bash
python index_docs.py --pdf ./docs/my_paper.pdf        # PDF
python index_docs.py --txt ./docs/notes.md            # TXT/MD
python index_docs.py --docx ./docs/report.docx        # DOCX
python index_docs.py --dir ./docs                     # ថតទាំងមូល (auto-detect)
python index_docs.py --hf-dataset m-ric/huggingface_doc --text-col text --source-col source
python index_docs.py --stats                          # បង្ហាញស្ថិតិ
python index_docs.py --clear                          # សម្អាតទាំងអស់
```

**តាមរយៈចំណុចប្រទាក់ (UI):** ចាប់ផ្តើមកម្មវិធី ហើយប្រើផ្ទាំង **📂 មូលដ្ឋានចំណេះដឹង**។ អាចទម្លាក់ឯកសារ PDF, TXT, MD ឬ DOCX។

#### ៣. ដំណើរការកម្មវិធី

```bash
python app.py
```

ឬនៅលើ Windows ចុចពីរដងលើ **RUN.bat**។ បើកកម្មវិធីរុករកតាមអាសយដ្ឋាន [http://localhost:7861](http://localhost:7861) (ច្រកលំនាំដើម **7861** ជៀសវាងការប៉ះទង្គិចនឹង Gradio លំនាំដើម 7860 — ប្តូរបានតាមរយៈ `server_port` នៅក្នុង `app.py`)។

---

### 🖥️ ចំណុចប្រទាក់អ្នកប្រើប្រាស់

កម្មវិធីនេះមានផ្ទាំងសំខាន់ៗចំនួនប្រាំបី តាមលំដាប់ដូចខាងក្រោម៖

| # | ផ្ទាំង | ការពិពណ៌នា |
|---|---|---|
| 1 | 💬 **ការសន្ទនាទូទៅ** | ការសន្ទនាផ្ទាល់ជាមួយ LLM — មិនមានការទាញយកឯកសារ។ របៀប Agent (ស្រេចចិត្ត) អាចស្វែងរកតាមអ៊ីនធឺណិត |
| 2 | 🖼️ **ការសន្ទនាចក្ខុវិស័យ** | បង្ហោះរូបភាព ហើយសួរសំណួរ — គាំទ្រការទាញយកបរិបទចម្រុះ និង Visual RAG |
| 3 | 🎙️ **និយាយទៅជាអក្សរ** | ថត/បង្ហោះសំឡេង ហើយបំលែងទៅជាអក្សរដោយ Whisper (រកឃើញភាសាស្វ័យប្រវត្តិ ឬកំណត់ភាសាផ្ទាល់) |
| 4 | 📊 **វិភាគទិន្នន័យ** | បង្ហោះ CSV/Excel ឱ្យ AI Agent វិភាគ បង្កើតក្រាហ្វិក និងសរសេររបាយការណ៍ Markdown |
| 5 | 📂 **មូលដ្ឋានចំណេះដឹង** | គ្រប់គ្រងឯកសារដែលបានបញ្ចូល (PDF/TXT/MD/DOCX) មើលតារាង និងសម្អាតការបញ្ចូល |
| 6 | 📚 **ការសន្ទនា RAG** | ទាញយកពីមូលដ្ឋានចំណេះដឹងជាមុន (ដោយផ្ទាល់ ឬដោយ Agent) មុននឹងឆ្លើយ |
| 7 | 🔬 **ស្រាវជ្រាវស៊ីជម្រៅ** | Agent គ្រប់គ្រង + agent ស្វែងរកតាមអ៊ីនធឺណិត បំបែកសំណួរជាសំណួររង រៀបចំផែនការឡើងវិញ រួចសរសេររបាយការណ៍ Markdown ដែលមានប្រភពយោង (ត្រូវការម៉ូដែលធំ)
| 8 | ℹ️ **អំពីកម្មវិធី** | ស្ថានភាពប្រព័ន្ធ ស្ថាបត្យកម្ម និងល្បឿនរំពឹងទុក — បង្ហាញជានិច្ចជាភាសាខ្មែរនិងអង់គ្លេសទាំងពីរ |

#### លក្ខណៈពិសេសសំខាន់ៗ
- **ចំណុចប្រទាក់ពីរភាសា**៖ ខ្មែរ/អង់គ្លេស ប្តូរបានភ្លាមៗតាមបញ្ជីទម្លាក់ភាសា
- **ម៉ូដែលពីរប្រភេទ**៖ HuggingFace/transformers ឬ GGUF (llama.cpp) ក្នុងបញ្ជីទម្លាក់ដូចគ្នា
- **ចងចាំការសន្ទនា (កំពុងសាកល្បង)**៖ ប្រអប់ "🧠 ចងចាំការសន្ទនា" នៅគ្រប់ផ្ទាំង agentic — បើកដើម្បីឱ្យសំណួរបន្តអាចយោងលើអ្វីដែលបាននិយាយពីមុន។ ការចងចាំមាននៅតែក្នុងវគ្គដំណើរការបច្ចុប្បន្នប៉ុណ្ណោះ (មិនរក្សាទុកទៅថាសទេ) ហើយនឹងត្រូវកំណត់ចេញនៅពេលប្តូរម៉ូដែល ចុច "សម្អាត" ឬចាប់ផ្តើមកម្មវិធីឡើងវិញ
- **ការគាំទ្រឯកសារ DOCX**៖ ការបញ្ចូលឯកសារគាំទ្រ Word (.docx) រួមទាំងអត្ថបទក្នុងតារាង
- **ការជួសជុលបរិយាកាសដោយខ្លួនឯង**៖ ប៊ូតុង "Fix Environment" ដំឡើង PyTorch ត្រឹមត្រូវសម្រាប់ GPU របស់អ្នកដោយស្វ័យប្រវត្តិ
- **ភាពមើលឃើញនៃការគិត**៖ ស្លាក `<think>` របស់ម៉ូដែលបង្ហាញជាផ្នែកដែលអាចពង្រីក/បង្រួមបាន
- **ការជ្រើសរើស Visual Retriever**៖ ជ្រើសរើសរវាង `colsmolvlm` ឬ `colqwen2` ដោយផ្ទាល់ក្នុងចំណុចប្រទាក់

---

### ⚙️ ការកំណត់ម៉ូដែល

អ្នកអាចប្តូរម៉ូដែលនៅពេលដំណើរការ តាមរយៈបញ្ជីទម្លាក់ក្នុងចំណុចប្រទាក់។

<details open>
<summary><strong>Text LLMs</strong></summary>

| VRAM/RAM | ម៉ូដែលដែលណែនាំ |
|---|---|
| ~1.2 GB | `Qwen/Qwen3-0.6B` (លឿនបំផុត — លំនាំដើម) |
| ~3 GB | `Qwen/Qwen3-1.7B` |
| ~6 GB | `Qwen/Qwen2.5-Coder-3B-Instruct` (small coding/agent model) |
| ~7 GB | `Qwen/Qwen3-4B` |
| ~4 GB | `google/gemma-4-E2B-it` |
| ប្រែប្រួល | ម៉ូដែល `.gguf` ណាមួយនៅក្នុងថតដែលអ្នកកំណត់ (តាមរយៈ llama.cpp) |

</details>

<details open>
<summary><strong>Vision LLMs (VLM)</strong></summary>

| RAM | ម៉ូដែលដែលណែនាំ |
|---|---|
| ~0.5 GB | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| ~1 GB | `HuggingFaceTB/SmolVLM-500M-Instruct` (ណែនាំ — លំនាំដើម) |
| ~6 GB | `Qwen/Qwen2.5-VL-3B-Instruct` |
| ប្រែប្រួល | គូម៉ូដែល GGUF (main + mmproj) នៅក្នុងថត llama.cpp របស់អ្នក |

</details>

<details open>
<summary><strong>Speech-to-Text (Whisper)</strong></summary>

| RAM | ម៉ូដែលដែលណែនាំ |
|---|---|
| ~1 GB | `openai/whisper-tiny` (លឿនបំផុត) |
| ~1 GB | `openai/whisper-base` |
| ~2 GB | `openai/whisper-small` (ណែនាំ — លំនាំដើម) |
| ~10 GB | `openai/whisper-large-v3` (ត្រឹមត្រូវបំផុត — ច្រើនភាសារួមទាំងខ្មែរ) |
| ~1 GB | `seanghay/whisper-small-khmer-v2` 🇰🇭 (កែសម្រួលសម្រាប់ខ្មែរ) |
| ~6 GB | `metythorn/whisper-large-v3-turbo-mixed-20eps-clean-text-197k` 🇰🇭 (ល្អបំផុតសម្រាប់ខ្មែរ) |

</details>

<details open>
<summary><strong>ជម្រើសម៉ូដែលតាមកម្រិតផ្នែករឹង (ជម្រើសកម្រិតខ្ពស់ជាង)</strong></summary>

តារាងខាងក្រោមផ្តល់ជូននូវសំណុំម៉ូដែល GGUF ដែលមានសមត្ថភាពខ្ពស់ជាងសម្រាប់អ្នកប្រើប្រាស់ដែលមានផ្នែករឹងខ្លាំង — ត្រូវទាញយក ហើយកំណត់ចេញពី "📁 ថតម៉ូដែល GGUF" ដូចម៉ូដែល `.gguf` ផ្សេងទៀត។

| ផ្នែករឹង | ម៉ូដែល Agent (សន្ទនា/Data Analysis) | ម៉ូដែល RAG | ម៉ូដែលចក្ខុវិស័យ (HF/transformers) | ម៉ូដែល Embedding |
|---|---|---|---|---|
| **CPU តែប៉ុណ្ណោះ (64–128 GB RAM)** | **Qwen3.6-35B-A3B (GGUF, MoE — ត្រូវការតែ ~3B active params/token ដូច្នេះលឿននៅលើ CPU)**<br>https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF | **Gemma-4-12B-it (GGUF)**<br>https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF | **Qwen2.5-VL-7B-Instruct (GGUF)**<br>https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF | **BGE-M3**<br>https://huggingface.co/BAAI/bge-m3 |
| **8 GB VRAM** | **Qwen3-8B (GGUF)**<br>https://huggingface.co/Qwen/Qwen3-8B-GGUF | **Gemma-4-12B-it (GGUF) — ⚠️ តឹងចង្អៀត សូមមើលចំណាំខាងក្រោម** (ដូចខាងលើ) | **Qwen2.5-VL-7B-Instruct (GGUF)** (ដូចខាងលើ) | **BGE-M3** (ដូចខាងលើ) |
| **16 GB VRAM** | **Qwen3-14B (GGUF)**<br>https://huggingface.co/MaziyarPanahi/Qwen3-14B-GGUF | **Gemma-4-12B-it (GGUF)** (ដូចខាងលើ) | **Qwen2.5-VL-7B-Instruct (GGUF)** (ដូចខាងលើ) | **BGE-M3** (ដូចខាងលើ) |
| **24 GB VRAM** | **Qwen3-32B (GGUF)**<br>https://huggingface.co/MaziyarPanahi/Qwen3-32B-GGUF | **Gemma-4-26B-A4B-it (GGUF)**<br>https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF | **Gemma-4-12B-it (GGUF)**<br>https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF | **Qwen3-Embedding-4B**<br>https://huggingface.co/Qwen/Qwen3-Embedding-4B |
| **48 GB+ VRAM** | **Qwen3-32B (GGUF)** (ដូចខាងលើ) | **Gemma-4-31B-it (GGUF)**<br>https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF | **Gemma-4-31B-it (GGUF)** (ដូចខាងលើ) | **Jina Embeddings v4**<br>https://huggingface.co/jinaai/jina-embeddings-v4 |

> ⚠️ **ចំណាំពិសេសអំពី 8 GB VRAM**៖ `Gemma-4-12B-it` នៅ Q4_K_M ត្រូវការទំហំ weight ត្រឹមតែ ~7.4 GB ដោយខ្លួនឯង — នេះស្ទើរតែមិនទុកទំហំសម្រាប់ KV cache ទេនៅលើកាត 8GB ជាពិសេសបើ Context Window (សូមមើលបញ្ជីទម្លាក់ "🧠 វិន្ដូបរិបទ" ក្នុង UI) ត្រូវបានកំណត់ធំជាង 8K។ ប្រសិនបើអ្នកជួប OOM លើកម្រិតនេះ សូមបន្ថយ Context Window ជាមុនសិន ឬប្តូរទៅ `Gemma-4-E4B-it` ដែលស្រាលជាង។
>
> **CPU តែប៉ុណ្ណោះ**៖ ចាប់តាំងពី Qwen3.6-35B-A3B ជាម៉ូដែល MoE (ត្រូវការតែ ~3B active parameters ក្នុងមួយ token ទោះបីជាទំហំសរុប 35B) វាដំណើរការបានលឿនគួរសមនៅលើ CPU ដែលមាន RAM 64–128GB — ខុសពីម៉ូដែល dense ដូចជា Qwen3-8B ដែលត្រូវការគណនាគ្រប់ parameter ទាំងអស់ជានិច្ច។

> ⚠️ **សំខាន់**៖ `Gemma-4-12B-it` / `Gemma-4-31B-it` ខ្លួនវាផ្ទាល់ជាម៉ូដែលចក្ខុវិស័យ (image-in, encoder-free) ដែរ ប៉ុន្តែកម្មវិធីនេះមិនទាន់ស្គាល់ chat-handler របស់វានៅឡើយទេ (`llama_backend.py` បច្ចុប្បន្នស្គាល់តែ LLaVA/MiniCPM-V/Moondream/nanoLLaVA) — ដូច្នេះការប្រើវាជា "🎨 Vision LLM" (GGUF) ក្នុងកម្មវិធីនេះ ត្រូវការកែកូដបន្ថែមសិន។ `Qwen2.5-VL-7B-Instruct` (HuggingFace/transformers) នៅតែជាជម្រើសសុវត្ថិភាពបំផុតដែលដំណើរការភ្លាមៗ។ ចំណែក `Qwen3-VL` (ជំនាន់ថ្មីជាង Qwen2.5-VL) ត្រូវបានគាំទ្រដោយ llama.cpp ចាប់តាំងពីចុងខែតុលា ២០២៥ តាមរយៈឧបករណ៍ `llama-mtmd-cli`/`llama-server` ថ្មី ប៉ុន្តែក៏ត្រូវការការកែសម្រួល `llama_backend.py` ដូចគ្នា មុននឹងអាចប្រើក្នុងកម្មវិធីនេះបាន។

</details>

---

### 📂 រចនាសម្ព័ន្ធឯកសារ

```
.
├── app.py            # ចំណុចចូលកម្មវិធី (launch) — ឡូជិកពិតត្រូវបានបំបែកទៅជាម៉ូឌុលខាងក្រោម
├── i18n.py           # ខ្សែអក្សរចំណុចប្រទាក់ (ខ្មែរ/អង់គ្លេស)
├── hardware.py       # ការរកឃើញ GPU/Device និងការជួសជុលបរិយាកាសដោយខ្លួនឯង
├── user_config.py    # ការកំណត់ដែលរក្សាទុក (ឧ. ថតម៉ូដែល GGUF, context window)
├── llama_backend.py  # backend ជម្រើស llama.cpp (ម៉ូដែល GGUF, text + vision)
├── branding.py       # ស្លាកសញ្ញា ឈ្មោះកម្មវិធី/កំណែ និងមាតិកាផ្ទាំង ℹ️ អំពីកម្មវិធី
├── model_registry.py # បញ្ជីជម្រើសម៉ូដែល (LLM/VLM/STT) និងការស្កេន GGUF ឡើងវិញ
├── models.py         # ការផ្ទុក/ដោះស្រាយ/ដំណើរការ LLM, VLM, STT
├── knowledge_base.py # ការបញ្ចូលឯកសារ (PDF/TXT/MD/DOCX), ChromaDB, Visual Index, ការទាញយក
├── agent_memory.py   # ការចងចាំចម្រុះវេនសម្រាប់ CodeAgent (RAM តែប៉ុណ្ណោះ — មិនរក្សាទុកទៅថាសទេ)
├── general_agent.py  # CodeAgent សម្រាប់ការសន្ទនាទូទៅបែប Agent (web search)
├── rag_agent.py      # CodeAgent សម្រាប់ RAG បែប Agent (retriever tool)
├── deep_research_agent.py # Agent គ្រប់គ្រង + agent ស្វែងរកតាមអ៊ីនធឺណិត សម្រាប់ស្រាវជ្រាវស៊ីជម្រៅ
├── data_analysis.py  # CodeAgent សម្រាប់វិភាគ CSV/Excel
├── chat.py           # Handler សន្ទនាសម្រាប់ផ្ទាំង General/RAG/Vision
├── ui.py             # ការសង់ចំណុចប្រទាក់ Gradio និងការភ្ជាប់ event ទាំងអស់
├── index_docs.py     # ស្គ្រីប CLI សម្រាប់បញ្ចូលឯកសារ
├── requirements.txt  # Dependencies របស់ Python (រួមទាំង python-docx, llama-cpp-python)
├── SETUP.bat/.ps1    # កម្មវិធីដំឡើងលើ Windows ដោយចុចតែម្តង
├── RUN.bat/.ps1      # កម្មវិធីដំណើរការលើ Windows ដោយចុចតែម្តង
├── README.md         # ឯកសារនេះ
├── chroma_db/        # បង្កើតដោយស្វ័យប្រវត្តិ; ការផ្ទុកទិន្នន័យជាប់លាប់របស់ ChromaDB
└── visual_index/     # បង្កើតដោយស្វ័យប្រវត្តិ; ការផ្ទុក Visual Index
```

---
---

## 🇬🇧 English

A local, multipurpose AI assistant built by **LocalAiLab** with [smolagents](https://github.com/huggingface/smolagents), featuring a Gradio UI, general chat, document RAG (Retrieval-Augmented Generation) with persistent ChromaDB storage, multi-modal capabilities (Vision/VLM), Speech-to-Text transcription, and AI-driven CSV/Excel data analysis. It automatically detects and uses your GPU (CUDA, AMD, or Mac MPS) if available, falling back to CPU otherwise. Besides the default HuggingFace/transformers models, the app can also run local **GGUF models via llama.cpp**.

The UI is fully bilingual — switch between **Khmer** and **English** instantly using the language dropdown in the top-right corner. The **ℹ️ About** tab always shows both languages (Khmer above, English below), regardless of that dropdown.

### 🆕 What's New in This Release

- **🔬 New Deep Research tab**: a manager agent + a dedicated web-search sub-agent break a question into sub-questions, periodically re-plan (`planning_interval`), and write a verified, sourced Markdown report
- **New models**: added the **Gemma 4** family (E2B/12B/26B-A4B/31B, now Apache 2.0-licensed) and the **Qwen3.6** family (27B dense / 35B-A3B MoE) — both need a newer `transformers` release (see the Text LLMs table below)
- **🧠 Conversation Memory (Experimental)**: a new toggle on every agentic tab lets follow-up questions refer back to earlier turns (RAM-only, never persisted to disk)
- **🧠 Configurable Context Window**: pick anywhere from 4K to 128K tokens for GGUF models directly in the UI
- **GPU-incompatibility detection**: the UI now shows a clear warning if a detected GPU (e.g. an older Pascal/sm_6x card) isn't supported by the installed PyTorch build, instead of silently falling back to CPU with no explanation
- **Data Analysis agent** can now install missing Python packages itself (`install_package` tool) and runs a fuller EDA with multiple charts

### 📑 Table of Contents
- [Architecture](#️-architecture)
- [Quick Start](#-quick-start)
- [User Interface](#️-user-interface)
- [Model Settings](#️-model-settings)
- [File Structure](#-file-structure)

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

> **Note**: `HardwareManager` automatically detects your hardware (NVIDIA, AMD, or Apple Silicon) and can help fix your environment via the "Fix Environment" button in the UI.

---

### 🚀 Quick Start

#### 1. Install dependencies

```bash
# Create a venv (optional but recommended)
python -m venv .venv && source .venv/bin/activate

# Install core dependencies (vision, audio, and text extraction, incl. python-docx)
pip install -r requirements.txt

# Install PyTorch (auto-detects CUDA/MPS/AMD if available)
pip install torch torchvision
```

On Windows, just double-click **SETUP.bat** to do all of the above automatically (detects your GPU, installs the matching PyTorch build, etc).

<details>
<summary><strong>Optional — llama.cpp (GGUF) model support</strong></summary>

`requirements.txt` already includes `llama-cpp-python`, but `pip install -r requirements.txt` alone only gives you a **CPU-only** build. For real GPU acceleration, use **SETUP.bat** instead — it will:

1. Detect your GPU (NVIDIA/CUDA, AMD/ROCm, or none)
2. Probe several prebuilt CUDA wheel tiers, newest-compatible-with-your-driver first
3. **Actually load a model as a smoke test** — catches the "Illegal Instruction" crash that happens when a prebuilt wheel assumes CPU features (like AVX512) your CPU doesn't have
4. Fall back to compiling from source with `CMAKE_ARGS=-DGGML_CUDA=on` if no prebuilt wheel works (requires Visual Studio Build Tools + CUDA Toolkit)
5. Fall back to a CPU-only build as a last resort — GGUF models still work, just slower

To do this by hand on Windows/NVIDIA:

```powershell
pip uninstall llama-cpp-python -y
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --force-reinstall --no-cache-dir
```

Swap `cu124` for whatever CUDA tier matches your driver (`nvidia-smi` shows it top-right). If that wheel crashes with "Illegal Instruction", build from source instead:

```powershell
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
$env:FORCE_CMAKE = "1"
pip install llama-cpp-python --no-cache-dir --force-reinstall
```

**Flash Attention**: the app requests `flash_attn=True` automatically whenever a GPU is available. Some architectures (notably hybrid local/global attention models like Gemma-3/4) don't yet support Flash Attention in llama.cpp — the app transparently falls back to running without it instead of crashing.

**GGUF model folder**: drop your `.gguf` files into any folder — no path is hardcoded. Point the app at it either by (1) setting the `LLAMA_CPP_MODEL_DIR` environment variable before launching, or (2) typing the folder path into the "📁 GGUF Model Folder" box in the UI and clicking "🔍 Scan" (no restart needed). No separate `llama-server` process is required — models load directly inside `app.py`.

</details>

#### 2. Index your documents

**Via CLI:**

```bash
python index_docs.py --pdf ./docs/my_paper.pdf        # Index a PDF
python index_docs.py --txt ./docs/notes.md            # Index a text/markdown file
python index_docs.py --docx ./docs/report.docx        # Index a Word document
python index_docs.py --dir ./docs                     # Index a folder (auto-detects types)
python index_docs.py --hf-dataset m-ric/huggingface_doc --text-col text --source-col source
python index_docs.py --stats                          # Show index statistics
python index_docs.py --clear                          # Clear the entire index
```

**Via UI:** Launch the app and use the **📂 Knowledge Base** tab. PDF, TXT, MD, and DOCX uploads are all supported.

#### 3. Launch the app

```bash
python app.py
```

Or on Windows, double-click **RUN.bat**. Open [http://localhost:7861](http://localhost:7861) in your browser (default port **7861**, to avoid conflicting with Gradio's usual 7860 — change via `server_port` in `app.py`).

---

### 🖥️ User Interface

The app is organized into eight main tabs, in this order:

| # | Tab | Description |
|---|---|---|
| 1 | 💬 **General Chat** | Direct LLM conversation — no retrieval. Optional Agentic Mode adds live web search |
| 2 | 🖼️ **Vision Chat** | Upload an image and ask questions — supports hybrid text context and Visual RAG |
| 3 | 🎙️ **Speech to Text** | Record or upload audio, transcribed via Whisper (auto-detect or forced language) |
| 4 | 📊 **Data Analysis** | Upload CSV/Excel; the AI agent explores it, builds charts, and writes a report |
| 5 | 📂 **Knowledge Base** | Manage indexed documents (PDF/TXT/MD/DOCX), view the table, clear the index |
| 6 | 📚 **RAG Chat** | Retrieves from the knowledge base first (directly or agentically), then answers |
| 7 | 🔬 **Deep Research** | A manager agent + web-search sub-agent break the question into sub-questions, re-plan as they go, and write a structured Markdown report with sources (needs a capable model) |
| 8 | ℹ️ **About** | System status, architecture, and performance expectations — always bilingual |

#### Key Features
- **Bilingual UI**: full Khmer/English interface — switch instantly with the language dropdown
- **Two model backends**: HuggingFace/transformers or local GGUF models via llama.cpp, in the same dropdown
- **Conversation Memory (Experimental)**: a "🧠 Conversation Memory" toggle on every agentic tab lets follow-up questions refer back to earlier turns. Memory only lives for the current running session (not saved to disk), and resets on a model switch, "Clear", or an app restart
- **DOCX support**: document indexing supports Word (.docx) files, including text inside tables
- **Environment Self-Fixing**: the "Fix Environment" button installs the correct PyTorch build for your GPU automatically
- **Reasoning Visibility**: model `<think>` tags render as a clean, collapsible UI element
- **Visual Retriever Selection**: choose between `colsmolvlm` or `colqwen2` directly in the UI

---

### ⚙️ Model Settings

You can change models at runtime via the UI's model selection dropdowns.

<details open>
<summary><strong>Text LLMs</strong></summary>

| VRAM/RAM | Recommended model |
|---|---|
| ~1.2 GB | `Qwen/Qwen3-0.6B` (fastest — default) |
| ~3 GB | `Qwen/Qwen3-1.7B` |
| ~6 GB | `Qwen/Qwen2.5-Coder-3B-Instruct` (small coding/agent model) |
| ~7 GB | `Qwen/Qwen3-4B` |
| ~4 GB | `google/gemma-4-E2B-it` |
| varies | Any `.gguf` model in your configured folder (via llama.cpp) |

</details>

<details open>
<summary><strong>Vision LLMs (VLM)</strong></summary>

| RAM | Recommended model |
|---|---|
| ~0.5 GB | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| ~1 GB | `HuggingFaceTB/SmolVLM-500M-Instruct` (recommended — default) |
| ~6 GB | `Qwen/Qwen2.5-VL-3B-Instruct` |
| varies | GGUF vision-model pairs (main + mmproj) in your llama.cpp folder |

</details>

<details open>
<summary><strong>Speech-to-Text (Whisper)</strong></summary>

| RAM | Recommended model |
|---|---|
| ~1 GB | `openai/whisper-tiny` (fastest) |
| ~1 GB | `openai/whisper-base` |
| ~2 GB | `openai/whisper-small` (recommended — default) |
| ~10 GB | `openai/whisper-large-v3` (best accuracy, multilingual incl. Khmer) |
| ~1 GB | `seanghay/whisper-small-khmer-v2` 🇰🇭 (Khmer-tuned) |
| ~6 GB | `metythorn/whisper-large-v3-turbo-mixed-20eps-clean-text-197k` 🇰🇭 (best for Khmer) |

</details>

<details open>
<summary><strong>Model combos by hardware tier (higher-end options)</strong></summary>

For people with beefier hardware who want noticeably stronger local models than the defaults above, here's a tiered combo — all still `.gguf` files you drop into your configured GGUF folder like any other model.

| Hardware | Agent Model (Chat/Data Analysis) | RAG Model | Vision Model (HF/transformers) | Embedding Model |
|---|---|---|---|---|
| **CPU Only (64–128 GB RAM)** | **Qwen3.6-35B-A3B (GGUF, MoE — only ~3B active params/token, so it stays fast on CPU)**<br>https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF | **Gemma-4-12B-it (GGUF)**<br>https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF | **Qwen2.5-VL-7B-Instruct (GGUF)**<br>https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF | **BGE-M3**<br>https://huggingface.co/BAAI/bge-m3 |
| **8 GB VRAM** | **Qwen3-8B (GGUF)**<br>https://huggingface.co/Qwen/Qwen3-8B-GGUF | **Gemma-4-12B-it (GGUF) — ⚠️ tight fit, see note below** (same as above) | **Qwen2.5-VL-7B-Instruct (GGUF)** (same as above) | **BGE-M3** (same as above) |
| **16 GB VRAM** | **Qwen3-14B (GGUF)**<br>https://huggingface.co/MaziyarPanahi/Qwen3-14B-GGUF | **Gemma-4-12B-it (GGUF)** (same as above) | **Qwen2.5-VL-7B-Instruct (GGUF)** (same as above) | **BGE-M3** (same as above) |
| **24 GB VRAM** | **Qwen3-32B (GGUF)**<br>https://huggingface.co/MaziyarPanahi/Qwen3-32B-GGUF | **Gemma-4-26B-A4B-it (GGUF)**<br>https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF | **Gemma-4-12B-it (GGUF)**<br>https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF | **Qwen3-Embedding-4B**<br>https://huggingface.co/Qwen/Qwen3-Embedding-4B |
| **48 GB+ VRAM** | **Qwen3-32B (GGUF)** (same as above) | **Gemma-4-31B-it (GGUF)**<br>https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF | **Gemma-4-31B-it (GGUF)** (same as above) | **Jina Embeddings v4**<br>https://huggingface.co/jinaai/jina-embeddings-v4 |

> ⚠️ **8 GB VRAM note**: `Gemma-4-12B-it` at Q4_K_M is already ~7.4 GB of weights alone — that leaves very little headroom for KV cache on an 8 GB card, especially if the "🧠 Context Window" dropdown in the UI is set above 8K. If you hit an out-of-memory error at this tier, lower the context window first, or switch to the lighter `Gemma-4-E4B-it` instead.
>
> **CPU Only**: Qwen3.6-35B-A3B is a MoE model — only ~3B of its 35B total parameters are active per token — so it stays reasonably fast on a 64–128 GB RAM CPU-only rig, unlike a dense model like Qwen3-8B, which always computes every parameter.

> ⚠️ **Compatibility note**: `Gemma-4-12B-it` / `Gemma-4-31B-it` are themselves natively multimodal (encoder-free, take image input directly) — but this app's GGUF vision loader (`llama_backend.py`) only recognizes the LLaVA/MiniCPM-V/Moondream/nanoLLaVA chat-handler families right now, not Gemma 4's format. So using them as this app's "🎨 Vision LLM" (GGUF path) needs a small code update first — until then, `Qwen2.5-VL-7B-Instruct` via the HuggingFace/transformers backend is the option that works out of the box. Similarly, `Qwen3-VL` (a newer, likely stronger vision-language family than Qwen2.5-VL) gained llama.cpp support in late October 2025 via the newer `llama-mtmd-cli`/`llama-server` multimodal path, but also isn't wired up in this app's handler-detection list yet.

</details>

---

### 📂 File Structure

```
.
├── app.py             # Entry point (launch) — actual logic lives in the modules below
├── i18n.py            # Khmer/English UI strings
├── hardware.py        # GPU/device detection, environment self-fix
├── user_config.py     # Persisted settings (e.g. GGUF model folder, context window)
├── llama_backend.py   # Optional llama.cpp (GGUF) model backend (text + vision)
├── branding.py        # Logo, app name/version, ℹ️ About tab content
├── model_registry.py  # Model dropdown options (LLM/VLM/STT) + GGUF rescan
├── models.py          # LLM/VLM/STT loading, caching, unloading, inference
├── knowledge_base.py  # Document indexing (PDF/TXT/MD/DOCX), ChromaDB, visual index, retrieval
├── agent_memory.py    # Multi-turn memory helper for CodeAgents (in-RAM only — no disk persistence, see file docstring)
├── general_agent.py   # Agentic General Chat CodeAgent (web search tools)
├── rag_agent.py        # Agentic RAG CodeAgent (retriever tool)
├── deep_research_agent.py # Manager + web-search sub-agent for Deep Research
├── data_analysis.py   # CodeAgent for CSV/Excel exploration
├── chat.py            # Chat-turn handlers for General/RAG/Vision tabs
├── ui.py              # Gradio Blocks UI + all event wiring
├── index_docs.py      # CLI indexing script
├── requirements.txt   # Python dependencies (incl. python-docx, llama-cpp-python)
├── SETUP.bat/.ps1     # Windows one-click installer
├── RUN.bat/.ps1       # Windows one-click launcher
├── README.md          # This file
├── chroma_db/         # Auto-created; ChromaDB persistent storage
└── visual_index/      # Auto-created; visual index storage
```
