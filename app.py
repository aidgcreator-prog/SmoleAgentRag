"""
Multipurpose AI Assistant — by LocalAiLab
smolagents + ChromaDB + Dynamic Hardware Support + Gradio UI
=======================================================================
Tabs:
  💬 General Chat  — direct LLM conversation, no retrieval
  📚 RAG Chat      — always retrieves from ChromaDB before answering
  🖼️ Vision Chat   — image upload + optional visual RAG
  🎙️ Speech to Text — transcribe audio to text using Whisper
  📊 Data Analysis — AI agent explores CSV/Excel, builds charts + report
  📂 Knowledge Base — index / manage documents
  ℹ️ About         — bilingual system info

This file is now just the entry point. Implementation lives in:
  i18n.py            — Khmer/English UI strings
  hardware.py        — GPU detection, environment self-fix
  user_config.py     — persisted settings (e.g. GGUF folder)
  llama_backend.py   — optional llama.cpp (GGUF) model backend
  branding.py        — logo, app name/version, About tab content
  model_registry.py  — model dropdown options + GGUF rescan
  models.py          — LLM/VLM/STT loading, caching, inference
  knowledge_base.py  — document indexing, ChromaDB, visual index, retrieval
  data_analysis.py   — CodeAgent for CSV/Excel exploration
  chat.py            — chat-turn handlers for the text/vision tabs
  ui.py              — Gradio Blocks UI + all event wiring
"""

import warnings

import gradio as gr

import models
from models import get_chroma_collection
from ui import build_ui, CSS

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

if __name__ == "__main__":
    print("[RAG] Pre-loading embeddings and ChromaDB …")
    models.get_embed_model()
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
