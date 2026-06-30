# 🔍 RAG Agent — smolagents · ChromaDB · Multi-Modal · GPU/CPU

A versatile local RAG (Retrieval-Augmented Generation) agent built with [smolagents](https://github.com/huggingface/smolagents), featuring a Gradio UI, persistent ChromaDB storage, and multi-modal capabilities (Vision/VLM). It automatically detects and uses your GPU (CUDA, AMD, or Mac MPS) if available, falling back to CPU otherwise.

---

## 🏗️ Architecture

| Component | Default |
|---|---|
| LLM | `Qwen/Qwen3-0.6B` |
| Vision LLM | `HuggingFaceTB/SmolVLM-500M-Instruct` |
| Embedding | `BAAI/bge-m3` |
| Visual Retriever | `vidore/colsmolvlm-v0.1` |
| Vector store | ChromaDB (`./chroma_db/`) |
| Visual Index | `vidore/colsmolvlm-v0.1` (`./visual_index/`) |
| Agent type | `smolagents.CodeAgent` |
| UI | Gradio |

> **Note**: The system includes a `HardwareManager` that automatically detects your hardware (NVIDIA, AMD, or Apple Silicon) and can help fix your environment via the UI.

---

## 🚀 Quick start

### 1. Install dependencies

```bash
# Create a venv (optional but recommended)
python -m venv .venv && source .venv/bin/activate

# Install core dependencies (handles complex visual and text extraction libs)
pip install -r requirements.txt

# Install PyTorch (automatically detects CUDA/MPS/AMD if installed)
# Note: If you have an NVIDIA GPU, you may need to install the specific 
# CUDA wheel as described in the SETUP.bat or via the UI's "Fix Environment" feature.
pip install torch torchvision
```

### 2. Index your documents

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

### 3. Launch the app

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

---

## 🖥️ User Interface

The app is organized into five main tabs:

1.  **💬 General Chat**: Direct conversation with the LLM. No document retrieval.
2.  **📚 RAG Chat**: Every question retrieves relevant text chunks from ChromaDB before answering.
3.  **🖼️ Vision Chat**: Upload images and ask questions. Supports **Hybrid Context Retrieval** (retrieving text context alongside visual interaction) and **Visual RAG** (retrieving images based on your query).
4.  **📂 Knowledge Base**: Manage your indexed documents, view the document table, and clear the index.
5.  **ℹ️ About**: View system status, hardware detection, and model performance expectations.

### Key Features
*   **Environment Self-Fixing**: Use the "Fix Environment" button in the UI to automatically install the correct PyTorch version for your GPU.
*   **Reasoning Visibility**: Model `<think>` tags are automatically rendered into a clean, collapsible UI element.
*   **Visual Retriever Selection**: Choose between different visual retrievers (e.g., `colsmolvlm` or `colqwen2`) directly in the UI.

---

## ⚙️ Model Settings

You can change models at runtime via the UI's model selection dropdowns.

### Text LLMs
| VRAM/RAM | Recommended model |
|---|---|
| ~1.2 GB | `Qwen/Qwen3-0.6B` (fastest) |
| ~3 GB | `Qwen/Qwen3-1.7B` |
| ~7 GB | `Qwen/Qwen3-4B` |
| ~4 GB | `google/gemma-4-E2B-it` |

### Vision LLMs (VLM)
| RAM | Recommended model |
|---|---|
| ~0.5 GB | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| ~1 GB | `HuggingFaceTB/SmolVLM-500M-Instruct` |
| ~6 GB | `Qwen/Qwen2.5-VL-3B-Instruct` |

---

## 📂 File structure

```
.
├── app.py           # Main app: agent, tools, indexing helpers, Gradio UI
├── index_docs.py    # CLI indexing script
├── requirements.txt # Python dependencies
├── README.md        # This file
├── chroma_db/       # Auto-created; ChromaDB persistent storage
└── visual_index/    # Auto-created; Visual index storage
```
