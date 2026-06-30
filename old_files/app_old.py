"""
RAG Agent - smolagents + ChromaDB + Local GPU + Gradio UI
=========================================================
Vision capabilities added:
  1. VLM Chat     : Qwen2.5-VL-3B (image upload in chat → multimodal answer)
  2. Visual PDF   : ColSmolVLM via colpali-engine (render PDF pages as images,
                    retrieve visually — no text extraction needed)
  3. Gemma-4-E2B  : already multimodal, image passed natively

Models:
  - Text LLM     : google/gemma-4-E2B-it  or  Qwen3-x  (text RAG)
  - Vision LLM   : Qwen/Qwen2.5-VL-3B-Instruct  (image-aware chat)
  - Text Embed   : BAAI/bge-m3            (100+ langs incl. Khmer)
  - Visual Embed : vidore/colsmolvlm-v0.1 (PDF-as-image retrieval)
  - Vector Store : ChromaDB (text) + colpali-engine index (visual)
  - Agent        : smolagents CodeAgent
  - UI           : Gradio
"""

import time
import threading
import tempfile
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional

import gradio as gr
import torch
from PIL import Image

# ──────────────────────────────────────────────────────────────────
# Model catalogue — text LLMs
# ──────────────────────────────────────────────────────────────────
MODEL_OPTIONS = {
    "🔵 Gemma-4-E2B-it       (smallest Gemma 4  | ~4 GB  | MoE 2B | VISION ✅)":  "google/gemma-4-E2B-it",
    "🟠 Qwen3.6-35B-A3B      (smallest Qwen3.6  | ~8 GB  | MoE 3B)":             "Qwen/Qwen3.6-35B-A3B",
    "🟢 Qwen3-0.6B           (Qwen3 dense 0.6B  | ~1.2 GB | fastest)":           "Qwen/Qwen3-0.6B",
    "🟡 Qwen3-1.7B           (Qwen3 dense 1.7B  | ~3 GB)":                       "Qwen/Qwen3-1.7B",
    "🟡 Qwen3-4B             (Qwen3 dense 4B    | ~7 GB)":                       "Qwen/Qwen3-4B",
    "🔴 Qwen3.6-27B          (Qwen3.6 dense 27B | ~50 GB | multi-GPU)":          "Qwen/Qwen3.6-27B",
    "🔴 Gemma-4-12B-it       (Gemma 4 12B       | ~24 GB | VISION ✅)":           "google/gemma-4-12B-it",
}
DEFAULT_LLM_LABEL = "🔵 Gemma-4-E2B-it       (smallest Gemma 4  | ~4 GB  | MoE 2B | VISION ✅)"
DEFAULT_LLM_MODEL = MODEL_OPTIONS[DEFAULT_LLM_LABEL]

# Vision LLM options (separate dropdown)
VLM_OPTIONS = {
    "🟢 Qwen2.5-VL-3B  (~6 GB VRAM | multilingual | recommended)": "Qwen/Qwen2.5-VL-3B-Instruct",
    "🟡 Qwen2.5-VL-7B  (~14 GB VRAM | higher quality)":            "Qwen/Qwen2.5-VL-7B-Instruct",
    "🔵 SmolVLM-256M   (~1 GB VRAM | tiny & fast)":                "HuggingFaceTB/SmolVLM-256M-Instruct",
    "🔵 SmolVLM-500M   (~2 GB VRAM | better quality)":             "HuggingFaceTB/SmolVLM-500M-Instruct",
}
DEFAULT_VLM_LABEL = "🟢 Qwen2.5-VL-3B  (~6 GB VRAM | multilingual | recommended)"
DEFAULT_VLM_MODEL = VLM_OPTIONS[DEFAULT_VLM_LABEL]

# Visual retriever options
VISUAL_RETRIEVER_OPTIONS = {
    "vidore/colsmolvlm-v0.1  (~2 GB | lightest | recommended)": "vidore/colsmolvlm-v0.1",
    "vidore/colqwen2-v1.0    (~8 GB | higher accuracy)":        "vidore/colqwen2-v1.0",
}
DEFAULT_VISUAL_RETRIEVER = "vidore/colsmolvlm-v0.1"

# Config
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
CHROMA_PERSIST_DIR  = "./chroma_db"
VISUAL_INDEX_DIR    = "./visual_index"
CHUNK_SIZE          = 1024
CHUNK_OVERLAP       = 128
TOP_K               = 4
MAX_NEW_TOKENS      = 1024
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"

GEMMA4_IDS   = {"google/gemma-4-E2B-it", "google/gemma-4-E4B-it",
                "google/gemma-4-12B-it",  "google/gemma-4-31B-it"}
QWEN3_IDS    = {"Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B",
                "Qwen/Qwen3-8B",   "Qwen/Qwen3-14B",  "Qwen/Qwen3-32B"}
QWEN36_IDS   = {"Qwen/Qwen3.6-27B", "Qwen/Qwen3.6-35B-A3B"}
ALL_QWEN_IDS = QWEN3_IDS | QWEN36_IDS
QWEN_VL_IDS  = {"Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct",
                "Qwen/Qwen2.5-VL-72B-Instruct"}
SMOL_VLM_IDS = {"HuggingFaceTB/SmolVLM-256M-Instruct", "HuggingFaceTB/SmolVLM-500M-Instruct",
                "HuggingFaceTB/SmolVLM2-2.2B-Instruct"}

# ──────────────────────────────────────────────────────────────────
# Singletons
# ──────────────────────────────────────────────────────────────────
_embed_model         = None
_chroma_col          = None
_agent               = None
_vlm_model           = None
_vlm_processor       = None
_vlm_model_id        = None
_visual_retriever    = None
_visual_retriever_id = None
_agent_lock          = threading.Lock()
_vlm_lock            = threading.Lock()
_llm_model_id        = DEFAULT_LLM_MODEL


# ──────────────────────────────────────────────────────────────────
# Text Embedding (bge-m3)
# ──────────────────────────────────────────────────────────────────
def get_embed_model():
    global _embed_model
    if _embed_model is None:
        print(f"[RAG] Loading embedding model '{DEFAULT_EMBED_MODEL}' on {DEVICE} …")
        try:
            from FlagEmbedding import BGEM3FlagModel
            _embed_model = BGEM3FlagModel(DEFAULT_EMBED_MODEL,
                                          use_fp16=(DEVICE == "cuda"), device=DEVICE)
            _embed_model._backend = "flag"
        except ImportError:
            from sentence_transformers import SentenceTransformer
            _embed_model = SentenceTransformer(DEFAULT_EMBED_MODEL, device=DEVICE)
            _embed_model._backend = "sbert"
    return _embed_model


def encode_texts(texts: list, normalize: bool = True):
    model   = get_embed_model()
    backend = getattr(model, "_backend", "sbert")
    if backend == "flag":
        import numpy as np
        result = model.encode(texts, batch_size=12, max_length=CHUNK_SIZE,
                              return_dense=True, return_sparse=False, return_colbert_vecs=False)
        vecs = result["dense_vecs"]
        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs  = vecs / np.maximum(norms, 1e-10)
        return vecs
    else:
        return model.encode(texts, normalize_embeddings=normalize, show_progress_bar=False)


# ──────────────────────────────────────────────────────────────────
# ChromaDB
# ──────────────────────────────────────────────────────────────────
def get_chroma_collection(name: str = "rag_docs"):
    global _chroma_col
    if _chroma_col is None:
        import chromadb
        client    = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _chroma_col = client.get_or_create_collection(name=name,
                          metadata={"hnsw:space": "cosine"})
        print(f"[RAG] ChromaDB ready — {_chroma_col.count()} chunks indexed.")
    return _chroma_col


# ──────────────────────────────────────────────────────────────────
# VLM (Vision Language Model) — Qwen2.5-VL or SmolVLM
# ──────────────────────────────────────────────────────────────────
def get_vlm(model_id: Optional[str] = None):
    global _vlm_model, _vlm_processor, _vlm_model_id
    target = model_id or DEFAULT_VLM_MODEL
    if _vlm_model is not None and target == _vlm_model_id:
        return _vlm_model, _vlm_processor

    print(f"[VLM] Loading vision model '{target}' on {DEVICE} …")

    if target in QWEN_VL_IDS:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        _vlm_processor = AutoProcessor.from_pretrained(target,
                             min_pixels=256*28*28, max_pixels=1280*28*28)
        _vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            target,
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            device_map="auto",
            attn_implementation="sdpa",
        )
        _vlm_model._arch = "qwen_vl"

    elif target in SMOL_VLM_IDS:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        _vlm_processor = AutoProcessor.from_pretrained(target)
        _vlm_model = AutoModelForVision2Seq.from_pretrained(
            target,
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            device_map="auto",
        )
        _vlm_model._arch = "smolvlm"

    else:
        raise ValueError(f"Unknown VLM model: {target}")

    _vlm_model_id = target
    print(f"[VLM] Vision model ready ({target}).")
    return _vlm_model, _vlm_processor


def vlm_answer(question: str, images: list, context: str = "") -> str:
    """
    Run a VLM inference with optional retrieved text context + images.
    images: list of PIL.Image
    """
    model, processor = get_vlm()
    arch = getattr(model, "_arch", "smolvlm")

    system_prompt = (
        "You are a helpful multimodal assistant. "
        "Answer based on the provided images and context."
    )
    context_block = f"\n\nRelevant context from knowledge base:\n{context}" if context else ""
    user_text     = question + context_block

    if arch == "qwen_vl":
        from qwen_vl_utils import process_vision_info
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": content},
        ]
        text_input = processor.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text_input], images=image_inputs,
                           padding=True, return_tensors="pt").to(model.device)
    else:
        # SmolVLM
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": content},
        ]
        text_input = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text_input, images=images if images else None,
                           return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                   temperature=0.3, do_sample=True)
    trimmed = generated[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(trimmed, skip_special_tokens=True)


# ──────────────────────────────────────────────────────────────────
# Visual Retriever (ColSmolVLM / ColQwen2) via colpali-engine
# ──────────────────────────────────────────────────────────────────
def get_visual_retriever(model_id: Optional[str] = None):
    global _visual_retriever, _visual_retriever_id
    target = model_id or DEFAULT_VISUAL_RETRIEVER
    if _visual_retriever is not None and target == _visual_retriever_id:
        return _visual_retriever
    try:
        from byaldi import RAGMultiModalModel
        index_path = Path(VISUAL_INDEX_DIR) / "main"
        if index_path.exists():
            _visual_retriever = RAGMultiModalModel.from_index(
                str(index_path), verbose=0)
        else:
            _visual_retriever = RAGMultiModalModel.from_pretrained(target, verbose=0)
        _visual_retriever_id = target
        print(f"[VIS] Visual retriever ready ({target}).")
    except ImportError:
        print("[VIS] byaldi not installed — visual PDF indexing disabled.")
        _visual_retriever = None
    return _visual_retriever


def index_pdf_visual(filepath: str, retriever_id: str) -> str:
    """Index a PDF visually (page-as-image) using ColSmolVLM."""
    try:
        from byaldi import RAGMultiModalModel
        global _visual_retriever, _visual_retriever_id
        index_path = Path(VISUAL_INDEX_DIR) / "main"
        index_path.parent.mkdir(parents=True, exist_ok=True)

        # Load or re-init retriever
        if _visual_retriever is None or _visual_retriever_id != retriever_id:
            _visual_retriever = RAGMultiModalModel.from_pretrained(retriever_id, verbose=0)
            _visual_retriever_id = retriever_id

        if index_path.exists():
            # Add to existing index
            _visual_retriever.add_to_index(
                input_item=filepath,
                store_collection_with_index=True,
                doc_id=int(time.time()),
            )
        else:
            _visual_retriever.index(
                input_path=filepath,
                index_name="main",
                index_root=VISUAL_INDEX_DIR,
                store_collection_with_index=True,
                overwrite=False,
            )
        return f"✅ Visual index updated: '{Path(filepath).name}'"
    except ImportError:
        return "❌ byaldi not installed. Run: pip install byaldi"
    except Exception as e:
        return f"❌ Visual indexing error: {e}"


def visual_retrieve(query: str, top_k: int = 3) -> list:
    """Return list of PIL images from visual index matching query."""
    retriever = get_visual_retriever()
    if retriever is None:
        return []
    try:
        results = retriever.search(query, k=top_k)
        images  = []
        for r in results:
            if hasattr(r, "base64") and r.base64:
                img_bytes = base64.b64decode(r.base64)
                images.append(Image.open(BytesIO(img_bytes)).convert("RGB"))
        return images
    except Exception as e:
        print(f"[VIS] Visual retrieval error: {e}")
        return []


# ──────────────────────────────────────────────────────────────────
# Gemma-4 text-only wrapper
# ──────────────────────────────────────────────────────────────────
def _build_gemma4_model(model_id: str):
    from smolagents.models import Model, ChatMessage
    from transformers import AutoProcessor, AutoModelForCausalLM
    print(f"[RAG] Loading Gemma 4 '{model_id}' on {DEVICE} …")
    processor = AutoProcessor.from_pretrained(model_id)
    hf_model  = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto", trust_remote_code=True)
    class Gemma4Model(Model):
        def generate(self, messages, stop_sequences=None, **kwargs):
            text   = processor.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
            inputs = processor(text=[text], return_tensors="pt").to(hf_model.device)
            with torch.no_grad():
                out = hf_model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                        temperature=0.3, do_sample=True,
                                        pad_token_id=processor.tokenizer.eos_token_id)
            new_tokens = out[0][inputs["input_ids"].shape[-1]:]
            decoded    = processor.decode(new_tokens, skip_special_tokens=True)
            if stop_sequences:
                for s in stop_sequences:
                    if s in decoded:
                        decoded = decoded[:decoded.index(s)]
            return ChatMessage(role="assistant", content=decoded)
    return Gemma4Model()


# ──────────────────────────────────────────────────────────────────
# Text RAG Agent
# ──────────────────────────────────────────────────────────────────
def get_agent(model_id: Optional[str] = None):
    global _agent, _llm_model_id
    target = model_id or _llm_model_id
    if _agent is not None and target == _llm_model_id:
        return _agent
    from smolagents import CodeAgent, TransformersModel
    if target in GEMMA4_IDS:
        llm = _build_gemma4_model(target)
    else:
        is_qwen = target in ALL_QWEN_IDS
        llm = TransformersModel(
            model_id=target, device_map="auto",
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            max_new_tokens=MAX_NEW_TOKENS, temperature=0.6, top_p=0.95,
            presence_penalty=1.5 if is_qwen else 0.0, trust_remote_code=True,
        )
    _agent       = CodeAgent(tools=[build_retriever_tool()], model=llm,
                             max_steps=5, verbosity_level=1)
    _llm_model_id = target
    return _agent


# ──────────────────────────────────────────────────────────────────
# Retriever Tool
# ──────────────────────────────────────────────────────────────────
def build_retriever_tool():
    from smolagents import Tool
    class ChromaRetrieverTool(Tool):
        name        = "document_retriever"
        description = ("Searches the knowledge base and returns relevant document chunks. "
                       "Use whenever you need factual information from indexed documents.")
        inputs = {
            "query": {"type": "string", "description": "Affirmative search statement."},
            "top_k": {"type": "integer", "description": f"Results to return (default {TOP_K}).",
                      "nullable": True},
        }
        output_type = "string"
        def forward(self, query: str, top_k: int = TOP_K) -> str:
            col = get_chroma_collection()
            if col.count() == 0:
                return "⚠️ Knowledge base is empty. Index some documents first."
            q_vec   = encode_texts([query]).tolist()
            results = col.query(query_embeddings=q_vec, n_results=min(top_k, col.count()))
            docs, metas, dists = results["documents"][0], results["metadatas"][0], results["distances"][0]
            if not docs:
                return "No relevant documents found."
            parts = []
            for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
                src = meta.get("source", "unknown")
                pg  = f" | page {meta['page']}" if meta.get("page") else ""
                parts.append(f"=== Doc {i+1} (source: {src}{pg}, relevance: {1-dist:.3f}) ===\n{doc}")
            return "\n\n".join(parts)
    return ChromaRetrieverTool()


# ──────────────────────────────────────────────────────────────────
# Indexing helpers
# ──────────────────────────────────────────────────────────────────
def _chunk_text(text: str):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start: start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def index_texts(texts: list, metadatas: list) -> int:
    col = get_chroma_collection()
    all_chunks, all_metas, all_ids = [], [], []
    for text, meta in zip(texts, metadatas):
        for j, chunk in enumerate(_chunk_text(text)):
            uid = f"{meta.get('source','doc')}__{len(all_chunks)}"
            all_chunks.append(chunk)
            all_metas.append({**meta, "chunk_index": j})
            all_ids.append(uid)
    if not all_chunks:
        return 0
    for i in range(0, len(all_chunks), 64):
        vecs = encode_texts(all_chunks[i:i+64]).tolist()
        col.upsert(embeddings=vecs, documents=all_chunks[i:i+64],
                   metadatas=all_metas[i:i+64], ids=all_ids[i:i+64])
    return len(all_chunks)


def index_pdf_file(filepath: str, visual_retriever_id: str = DEFAULT_VISUAL_RETRIEVER):
    """Index PDF both as text (bge-m3) and visually (ColSmolVLM)."""
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
        k = index_texts(texts, metas)
        msgs.append(f"✅ Text: {k} chunks from '{Path(filepath).name}'")
    except Exception as e:
        msgs.append(f"⚠️ Text extraction failed: {e}")

    # Visual indexing
    vis_msg = index_pdf_visual(filepath, visual_retriever_id)
    msgs.append(f"🖼️ Visual: {vis_msg}")
    return "\n".join(msgs)


def index_txt_file(filepath: str):
    try:
        text = Path(filepath).read_text(encoding="utf-8", errors="replace")
        k    = index_texts([text], [{"source": Path(filepath).name, "type": "txt"}])
        return f"✅ {k} chunks from '{Path(filepath).name}'"
    except Exception as e:
        return f"❌ {e}"


def index_uploaded_files(files, visual_retriever_label: str) -> str:
    if not files:
        return "No files uploaded."
    retriever_id = VISUAL_RETRIEVER_OPTIONS.get(visual_retriever_label, DEFAULT_VISUAL_RETRIEVER)
    msgs = []
    for f in files:
        path = f.name if hasattr(f, "name") else str(f)
        ext  = Path(path).suffix.lower()
        if ext == ".pdf":
            msg = index_pdf_file(path, retriever_id)
        elif ext in (".txt", ".md"):
            msg = index_txt_file(path)
        else:
            msg = f"⚠️ Unsupported: {ext}"
        msgs.append(msg)
    return "\n".join(msgs)


def index_hf_dataset(dataset_name: str, text_col: str, source_col: str = "") -> str:
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
        return f"✅ {k} chunks from '{dataset_name}'"
    except Exception as e:
        return f"❌ {e}"


# ──────────────────────────────────────────────────────────────────
# Document table helpers
# ──────────────────────────────────────────────────────────────────
def get_doc_table() -> list:
    col = get_chroma_collection()
    if col.count() == 0:
        return []
    result = col.get(include=["metadatas"])
    from collections import defaultdict
    agg = defaultdict(lambda: {"type": "", "pages": set(), "chunks": 0})
    for m in result["metadatas"]:
        src = m.get("source", "unknown")
        agg[src]["type"]   = m.get("type", "unknown")
        agg[src]["chunks"] += 1
        if m.get("page"):
            agg[src]["pages"].add(m["page"])
    rows = []
    for src, info in sorted(agg.items()):
        pages_str = str(len(info["pages"])) if info["pages"] else "—"
        rows.append([src, info["type"], pages_str, info["chunks"]])
    return rows


def delete_selected_sources(selected_rows: list, doc_table_data: list) -> tuple:
    if not selected_rows:
        return get_doc_table(), "⚠️ No rows selected."
    col = get_chroma_collection()
    deleted = []
    for row_idx in selected_rows:
        if row_idx >= len(doc_table_data):
            continue
        source_name = doc_table_data[row_idx][0]
        result = col.get(where={"source": source_name}, include=["metadatas"])
        if result["ids"]:
            col.delete(ids=result["ids"])
            deleted.append(f"'{source_name}' ({len(result['ids'])} chunks)")
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
    return [], "🗑️ All text documents cleared."


def get_index_stats() -> str:
    col = get_chroma_collection()
    n   = col.count()
    vis_idx = Path(VISUAL_INDEX_DIR) / "main"
    vis_str = "✅ ready" if vis_idx.exists() else "empty"
    gpu_info = ""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_info = f"  |  GPU: {torch.cuda.get_device_name(0)} {alloc:.1f}/{total:.1f} GB"
    return f"📚 Text chunks: {n}  |  Visual index: {vis_str}  |  Device: {DEVICE}{gpu_info}"


# ──────────────────────────────────────────────────────────────────
# Chat handlers
# ──────────────────────────────────────────────────────────────────
def chat_text(user_message: str, history: list, model_label: str):
    """Standard text RAG chat."""
    if not user_message.strip():
        return history, ""
    history = history or []
    history.append({"role": "user", "content": user_message})
    model_id = MODEL_OPTIONS.get(model_label, DEFAULT_LLM_MODEL)
    try:
        with _agent_lock:
            agent = get_agent(model_id)
        t0    = time.time()
        query = user_message + " /no_think" if model_id in ALL_QWEN_IDS else user_message
        ans   = agent.run(query)
        elapsed  = time.time() - t0
        response = f"{ans}\n\n*⏱ {elapsed:.1f}s  |  model: `{model_id}`*"
    except Exception as e:
        response = f"❌ Agent error: {e}"
    history.append({"role": "assistant", "content": response})
    return history, ""


def chat_vision(user_message: str, uploaded_images, history: list,
                vlm_label: str, use_visual_rag: bool):
    """Vision-aware chat: image upload + optional visual RAG from indexed PDFs."""
    if not user_message.strip() and not uploaded_images:
        return history, ""

    history = history or []

    # Build display message
    display_msg = user_message or "(image)"
    history.append({"role": "user", "content": display_msg})

    vlm_id = VLM_OPTIONS.get(vlm_label, DEFAULT_VLM_MODEL)

    try:
        with _vlm_lock:
            get_vlm(vlm_id)

        pil_images = []

        # 1. Images uploaded directly in chat
        if uploaded_images:
            if isinstance(uploaded_images, list):
                for img in uploaded_images:
                    if isinstance(img, str):
                        pil_images.append(Image.open(img).convert("RGB"))
                    elif isinstance(img, Image.Image):
                        pil_images.append(img)
            elif isinstance(uploaded_images, str):
                pil_images.append(Image.open(uploaded_images).convert("RGB"))

        # 2. Visual RAG — retrieve matching PDF pages as images
        if use_visual_rag and user_message:
            retrieved_imgs = visual_retrieve(user_message, top_k=2)
            pil_images.extend(retrieved_imgs)

        # 3. Text context from ChromaDB
        text_context = ""
        if user_message:
            col = get_chroma_collection()
            if col.count() > 0:
                q_vec   = encode_texts([user_message]).tolist()
                results = col.query(query_embeddings=q_vec, n_results=min(TOP_K, col.count()))
                docs    = results["documents"][0]
                text_context = "\n\n".join(docs[:2]) if docs else ""

        t0  = time.time()
        ans = vlm_answer(user_message, pil_images, context=text_context)
        elapsed = time.time() - t0

        img_count = len(pil_images)
        response  = (f"{ans}\n\n"
                     f"*⏱ {elapsed:.1f}s  |  VLM: `{vlm_id}`"
                     f"  |  images: {img_count}  |  visual RAG: {'on' if use_visual_rag else 'off'}*")

    except Exception as e:
        response = f"❌ Vision error: {e}"

    history.append({"role": "assistant", "content": response})
    return history, None


# ──────────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────────
CSS = """
.status-bar   { font-size:0.82rem; color:#888; padding:4px 8px; }
.header-wrap  { display:flex; align-items:baseline; gap:12px; margin-bottom:6px; }
.header-title { font-size:1.6rem; font-weight:700; }
.header-sub   { font-size:0.9rem; color:#aaa; }
.vision-badge { background:#6366f1; color:white; border-radius:4px;
                padding:2px 6px; font-size:0.75rem; }
"""

def build_ui():
    with gr.Blocks(title="🔍 RAG Agent + Vision", theme=gr.themes.Soft(primary_hue="violet"),
                   css=CSS) as demo:

        gr.HTML("""
        <div class="header-wrap">
          <span class="header-title">🔍 RAG Agent</span>
          <span class="header-sub">smolagents · ChromaDB · bge-m3 · ColSmolVLM · Qwen2.5-VL · Local GPU</span>
        </div>""")

        with gr.Tabs():

            # ══════════════════════════════════════════════════════
            # TAB 1 — Text Chat (standard RAG)
            # ══════════════════════════════════════════════════════
            with gr.Tab("💬 Text Chat"):
                chatbot_text = gr.Chatbot(label="Text RAG", height=440,
                                          type="messages", show_copy_button=True)
                with gr.Row():
                    msg_text = gr.Textbox(placeholder="Ask about your documents …",
                                          show_label=False, scale=8)
                    send_text = gr.Button("Send ▶", variant="primary", scale=1)
                with gr.Row():
                    model_dd   = gr.Dropdown(choices=list(MODEL_OPTIONS.keys()),
                                             value=DEFAULT_LLM_LABEL,
                                             label="🤖 Text LLM", scale=6)
                    reload_btn = gr.Button("🔄 Reload", size="sm", scale=2)
                    reload_out = gr.Textbox(show_label=False, interactive=False, scale=4)
                clear_text_btn = gr.Button("🗑️ Clear", size="sm")

                def reload_model(label):
                    global _agent
                    _agent = None
                    mid = MODEL_OPTIONS.get(label, DEFAULT_LLM_MODEL)
                    try:
                        get_agent(mid)
                        mem = (f" | GPU {torch.cuda.memory_allocated()/1e9:.1f} GB"
                               if torch.cuda.is_available() else "")
                        return f"✅ '{mid}' loaded{mem}"
                    except Exception as e:
                        return f"❌ {e}"

                msg_text.submit(chat_text, [msg_text, chatbot_text, model_dd],
                                [chatbot_text, msg_text])
                send_text.click(chat_text, [msg_text, chatbot_text, model_dd],
                                [chatbot_text, msg_text])
                clear_text_btn.click(lambda: ([], ""), outputs=[chatbot_text, msg_text])
                reload_btn.click(reload_model, [model_dd], [reload_out])

            # ══════════════════════════════════════════════════════
            # TAB 2 — Vision Chat
            # ══════════════════════════════════════════════════════
            with gr.Tab("🖼️ Vision Chat"):
                gr.Markdown(
                    "Upload images **and/or** ask questions — the VLM sees your images, "
                    "retrieved PDF pages (visual RAG), and text context all at once."
                )
                chatbot_vis = gr.Chatbot(label="Vision RAG", height=400,
                                         type="messages", show_copy_button=True)
                with gr.Row():
                    msg_vis = gr.Textbox(placeholder="Ask about the image or your documents …",
                                         show_label=False, scale=6)
                    img_upload = gr.Image(label="Upload image", type="pil",
                                          sources=["upload", "clipboard"], scale=2)
                    send_vis = gr.Button("Send ▶", variant="primary", scale=1)

                with gr.Row():
                    vlm_dd = gr.Dropdown(choices=list(VLM_OPTIONS.keys()),
                                         value=DEFAULT_VLM_LABEL,
                                         label="🎨 Vision LLM", scale=5)
                    vis_rag_toggle = gr.Checkbox(label="🔍 Visual RAG (retrieve PDF pages)",
                                                  value=True, scale=2)
                    load_vlm_btn = gr.Button("🔄 Load VLM", size="sm", scale=2)
                    load_vlm_out = gr.Textbox(show_label=False, interactive=False, scale=3)

                clear_vis_btn = gr.Button("🗑️ Clear", size="sm")

                def load_vlm(label):
                    global _vlm_model, _vlm_processor
                    _vlm_model = _vlm_processor = None
                    mid = VLM_OPTIONS.get(label, DEFAULT_VLM_MODEL)
                    try:
                        get_vlm(mid)
                        mem = (f" | GPU {torch.cuda.memory_allocated()/1e9:.1f} GB"
                               if torch.cuda.is_available() else "")
                        return f"✅ '{mid}' loaded{mem}"
                    except Exception as e:
                        return f"❌ {e}"

                send_vis.click(chat_vision,
                               [msg_vis, img_upload, chatbot_vis, vlm_dd, vis_rag_toggle],
                               [chatbot_vis, img_upload])
                msg_vis.submit(chat_vision,
                               [msg_vis, img_upload, chatbot_vis, vlm_dd, vis_rag_toggle],
                               [chatbot_vis, img_upload])
                clear_vis_btn.click(lambda: ([], None), outputs=[chatbot_vis, img_upload])
                load_vlm_btn.click(load_vlm, [vlm_dd], [load_vlm_out])

            # ══════════════════════════════════════════════════════
            # TAB 3 — Knowledge Base
            # ══════════════════════════════════════════════════════
            with gr.Tab("📂 Knowledge Base"):

                with gr.Accordion("📤 Add Documents", open=True):
                    with gr.Tabs():
                        with gr.Tab("Upload Files"):
                            file_up = gr.File(label="Drop PDF / TXT / MD",
                                              file_types=[".pdf", ".txt", ".md"],
                                              file_count="multiple")
                            vis_ret_dd = gr.Dropdown(
                                choices=list(VISUAL_RETRIEVER_OPTIONS.keys()),
                                value=list(VISUAL_RETRIEVER_OPTIONS.keys())[0],
                                label="🖼️ Visual retriever (for PDF pages)",
                            )
                            up_btn = gr.Button("📥 Index files", variant="primary")
                            up_msg = gr.Textbox(label="Result", interactive=False, lines=4)

                        with gr.Tab("HuggingFace Dataset"):
                            with gr.Row():
                                ds_name    = gr.Textbox(label="Dataset", value="m-ric/huggingface_doc", scale=3)
                                ds_textcol = gr.Textbox(label="Text col", value="text", scale=1)
                                ds_srccol  = gr.Textbox(label="Source col", value="source", scale=1)
                            ds_btn = gr.Button("⬇️ Load & index", variant="secondary")
                            ds_msg = gr.Textbox(label="Result", interactive=False, lines=2)

                gr.Markdown("---\n### 📋 Indexed Documents")
                gr.Markdown("Click rows to select, then **Delete Selected** to remove.")

                doc_table = gr.Dataframe(
                    headers=["Source", "Type", "Pages", "Chunks"],
                    datatype=["str", "str", "str", "number"],
                    value=get_doc_table,
                    interactive=True, wrap=True,
                    col_count=(4, "fixed"),
                )
                with gr.Row():
                    refresh_btn    = gr.Button("🔄 Refresh", size="sm", scale=2)
                    delete_sel_btn = gr.Button("🗑️ Delete Selected", variant="stop", size="sm", scale=2)
                    clear_all_btn  = gr.Button("💥 Clear ALL", variant="stop", size="sm", scale=2)
                action_msg = gr.Textbox(label="", interactive=False, lines=1)

                selected_rows_state = gr.State([])

                def on_select(evt: gr.SelectData, current):
                    row = evt.index[0]
                    if row in current:
                        current.remove(row)
                    else:
                        current.append(row)
                    return current

                def do_upload(files, vis_ret_label):
                    msg   = index_uploaded_files(files, vis_ret_label)
                    table = get_doc_table()
                    return msg, table

                def do_delete(selected, table_data):
                    rows = table_data if isinstance(table_data, list) else table_data.values.tolist()
                    new_table, msg = delete_selected_sources(selected, rows)
                    return new_table, msg, []

                def do_clear():
                    table, msg = clear_index()
                    return table, msg, []

                doc_table.select(on_select, [selected_rows_state], [selected_rows_state])
                up_btn.click(do_upload, [file_up, vis_ret_dd], [up_msg, doc_table])
                ds_btn.click(index_hf_dataset, [ds_name, ds_textcol, ds_srccol], [ds_msg, doc_table])
                refresh_btn.click(get_doc_table, outputs=[doc_table])
                delete_sel_btn.click(do_delete, [selected_rows_state, doc_table],
                                     [doc_table, action_msg, selected_rows_state])
                clear_all_btn.click(do_clear, outputs=[doc_table, action_msg, selected_rows_state])

            # ══════════════════════════════════════════════════════
            # TAB 4 — About
            # ══════════════════════════════════════════════════════
            with gr.Tab("ℹ️ About"):
                vram_info = ""
                if torch.cuda.is_available():
                    name  = torch.cuda.get_device_name(0)
                    total = torch.cuda.get_device_properties(0).total_memory / 1e9
                    vram_info = f"\n**Detected GPU:** `{name}` — {total:.0f} GB VRAM\n"
                gr.Markdown(f"""
## Architecture
| Component | Model | Notes |
|---|---|---|
| Text LLM | `google/gemma-4-E2B-it` | Default — also a VLM natively |
| Vision LLM | `Qwen/Qwen2.5-VL-3B-Instruct` | Chat image upload + visual RAG answers |
| Text Embedding | `BAAI/bge-m3` | 100+ languages incl. Khmer |
| Visual Retriever | `vidore/colsmolvlm-v0.1` | PDF-as-image retrieval (ColBERT-style) |
| Vector Store | ChromaDB | Text chunks, cosine similarity |
| Agent | `smolagents.CodeAgent` | Tool-calling agent |
{vram_info}
## Vision Capabilities
| Feature | How it works |
|---|---|
| **Image upload in chat** | Upload any image → VLM sees it + answers |
| **Visual RAG** | PDF pages indexed as images → retrieved visually by ColSmolVLM |
| **Text + image fusion** | VLM receives: user image + retrieved PDF pages + text chunks |
| **Khmer images** | Qwen2.5-VL supports multilingual OCR including Khmer script |

## VLM VRAM Guide
| Model | VRAM | Notes |
|---|---|---|
| `SmolVLM-256M` | ~1 GB | Tiny, fast, basic quality |
| `SmolVLM-500M` | ~2 GB | Better quality, still tiny |
| `Qwen2.5-VL-3B` | ~6 GB | **Recommended** — multilingual, strong |
| `Qwen2.5-VL-7B` | ~14 GB | High quality |

## Visual Retrievers
| Model | VRAM | Notes |
|---|---|---|
| `colsmolvlm-v0.1` | ~2 GB | Lightest, good for consumer GPU |
| `colqwen2-v1.0` | ~8 GB | Higher accuracy |

## Quick start
```
SETUP.bat   ← run once (installs everything including byaldi + qwen-vl-utils)
RUN.bat     ← launch app → http://localhost:7860
```
                """)

        gr.Textbox(value=get_index_stats, interactive=False,
                   show_label=False, elem_classes=["status-bar"])

    return demo


if __name__ == "__main__":
    print("[RAG] Pre-loading embeddings and ChromaDB …")
    get_embed_model()
    get_chroma_collection()
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)
