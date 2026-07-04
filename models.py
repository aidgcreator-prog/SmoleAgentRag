"""
models.py — Lazy-loaded model backends: embedding model, ChromaDB handle,
text LLM (HuggingFace or GGUF via llama_backend), Vision LLM, and
Speech-to-Text pipeline. Each model type keeps its own module-level cache
and lock, following the lazy-load/lock pattern used throughout the app.
"""

import gc
import threading
import time
from typing import Optional

import torch

import llama_backend
import model_registry as mr
from hardware import DEVICE, TORCH_DTYPE
from i18n import LANGUAGES

_embed_model         = None
_chroma_col          = None
_llm                 = None
_llm_model_id        = mr.DEFAULT_LLM_MODEL
_vlm_model           = None
_vlm_processor       = None
_vlm_model_id        = None
_llm_lock            = threading.Lock()
_vlm_lock            = threading.Lock()
_stt_pipeline        = None
_stt_model_id        = None
_stt_lock            = threading.Lock()


def _release_model(obj):
    """Explicitly free memory held by a model before dropping the last
    Python reference to it.

    Just setting a global cache variable to None does NOT free GPU/CPU
    memory right away:
      - transformers/torch models: the CUDA caching allocator keeps the
        freed blocks around until torch.cuda.empty_cache() runs (and that
        can only reclaim memory from tensors that have already been
        garbage-collected).
      - llama.cpp (llama-cpp-python) models: the native context / KV-cache
        lives in a C++ object. Its Python wrapper's __del__ will eventually
        close it, but only once the GC actually runs the finalizer — which
        is not guaranteed to happen before the next model tries to load,
        causing an OOM when switching models back-to-back.

    Calling this before overwriting/loading a new model avoids the
    "previous model didn't offload, so the next one fails to load" issue.
    """
    if obj is None:
        return
    try:
        # llama.cpp backend: LlamaCppModel wraps the native llama_cpp.Llama
        # instance as `.llm` — close it explicitly to free the KV-cache /
        # context right now instead of waiting on GC.
        inner = getattr(obj, "llm", None)
        if inner is not None and hasattr(inner, "close"):
            try:
                inner.close()
            except Exception:
                pass

        # transformers / smolagents TransformersModel: move the underlying
        # nn.Module off the GPU before dropping it so CUDA's allocator can
        # actually reclaim the memory once we empty_cache() below.
        model_attr = getattr(obj, "model", None)
        if model_attr is not None and hasattr(model_attr, "to"):
            try:
                model_attr.to("cpu")
            except Exception:
                pass
    except Exception:
        pass

    del obj
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        print(f"[RAG] Loading embedding model '{mr.DEFAULT_EMBED_MODEL}' on {DEVICE.upper()} …")
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(mr.DEFAULT_EMBED_MODEL, device=DEVICE)
    return _embed_model


def encode_texts(texts: list, normalize: bool = True):
    vecs = get_embed_model().encode(texts, normalize_embeddings=normalize,
                                    show_progress_bar=False)
    return vecs.tolist() if hasattr(vecs, "tolist") else vecs


def get_chroma_collection(name: str = "rag_docs"):
    global _chroma_col
    if _chroma_col is None:
        import chromadb
        client      = chromadb.PersistentClient(path=mr.CHROMA_PERSIST_DIR)
        _chroma_col = client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"})
        print(f"[RAG] ChromaDB ready — {_chroma_col.count()} chunks indexed.")
    return _chroma_col


def reset_chroma_collection():
    """Used by clear_index() in knowledge_base.py to drop and recreate the
    collection object after deleting it from disk."""
    global _chroma_col
    _chroma_col = None


def get_llm(model_id: Optional[str] = None):
    global _llm, _llm_model_id
    target = model_id or _llm_model_id

    if _llm is not None and target == _llm_model_id:
        return _llm

    with _llm_lock:
        if _llm is not None and target == _llm_model_id:
            return _llm

        if _llm is not None:
            print(f"[RAG] Unloading previous LLM '{_llm_model_id}' before loading '{target}' …")
            _release_model(_llm)
            _llm = None

        if str(target).lower().endswith(".gguf"):
            if not llama_backend.LLAMA_CPP_AVAILABLE:
                raise RuntimeError(
                    "llama-cpp-python is not installed. Run SETUP.bat to install "
                    "a hardware-matched build, or install it manually with the "
                    "appropriate CUDA/Metal/ROCm build flags for GPU support."
                )
            _llm = llama_backend.LlamaCppModel(
                model_path=target,
                n_ctx=16384,
                # -1 offloads every layer to GPU; on a CPU-only llama-cpp-python
                # build this is harmless (llama.cpp silently ignores it and runs
                # on CPU), but we set 0 explicitly once we know for sure so the
                # load logs are accurate instead of claiming a GPU offload that
                # won't happen.
                n_gpu_layers=-1 if llama_backend.LLAMA_CPP_GPU_AVAILABLE else 0,
                # Flash attention only helps (and is only reliably supported)
                # when the model is actually running on GPU; LlamaCppModel
                # itself also falls back gracefully if a specific model's
                # architecture rejects FA even when the GPU build supports it.
                flash_attn=llama_backend.LLAMA_CPP_GPU_AVAILABLE,
                temperature=0.6,
                top_p=0.95,
                max_new_tokens=mr.MAX_NEW_TOKENS,
            )
        else:
            from smolagents import TransformersModel
            print(f"[RAG] Loading LLM '{target}' on {DEVICE.upper()} …")
            _llm = TransformersModel(
                model_id=target,
                device_map=DEVICE,
                torch_dtype=TORCH_DTYPE,
                max_new_tokens=mr.MAX_NEW_TOKENS,
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

    target = model_id or mr.DEFAULT_VLM_MODEL

    if _vlm_model is not None and target == _vlm_model_id:
        return _vlm_model, _vlm_processor

    if _vlm_model is not None:
        print(f"[VLM] Unloading previous VLM '{_vlm_model_id}' before loading '{target}' …")
        _release_model(_vlm_model)
        _vlm_model = None
        _vlm_processor = None

    print(f"[VLM] Loading '{target}' on {DEVICE.upper()} …")

    from transformers import AutoProcessor

    if target in mr.QWEN_VL_IDS:
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

    elif target in mr.SMOL_VLM_IDS:
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
            out = model.generate(**inputs, max_new_tokens=mr.MAX_NEW_TOKENS)
        trimmed = out[0][inputs["input_ids"].shape[-1]:]
        return processor.decode(trimmed, skip_special_tokens=True)
    except Exception as e:
        import traceback
        return f"❌ Error during VLM generation: {str(e)}\n\n{traceback.format_exc()}"


def get_stt_pipeline(model_id: Optional[str] = None):
    global _stt_pipeline, _stt_model_id
    target = model_id or mr.DEFAULT_STT_MODEL

    if _stt_pipeline is not None and target == _stt_model_id:
        return _stt_pipeline

    with _stt_lock:
        if _stt_pipeline is not None and target == _stt_model_id:
            return _stt_pipeline

        if _stt_pipeline is not None:
            print(f"[STT] Unloading previous STT model '{_stt_model_id}' before loading '{target}' …")
            _release_model(_stt_pipeline)
            _stt_pipeline = None

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
            # Whisper's underlying generate() refuses more than 3000 mel
            # input features (30s of audio) unless either return_timestamps
            # is set or the pipeline chunks the audio itself first. Setting
            # chunk_length_s here makes the pipeline split any longer
            # recording into <=30s windows (with a little overlap via
            # stride_length_s so words at chunk boundaries aren't cut off),
            # run each window through generate() normally, and stitch the
            # text back together — so recordings/uploads of any length just
            # work, with no extra changes needed in transcribe_audio().
            chunk_length_s=30,
            stride_length_s=5,
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

        # Decode the audio file ourselves via soundfile/librosa (both already
        # in requirements.txt) instead of handing transformers a raw filepath,
        # which requires ffmpeg to be separately installed and on PATH.
        import numpy as np
        target_sr = getattr(getattr(asr, "feature_extractor", None), "sampling_rate", 16000)

        try:
            import soundfile as sf
            audio_array, src_sr = sf.read(audio_path, dtype="float32", always_2d=False)
        except Exception:
            import librosa
            audio_array, src_sr = librosa.load(audio_path, sr=None, mono=False)

        audio_array = np.asarray(audio_array, dtype="float32")
        if audio_array.ndim == 2:
            # Collapse multi-channel audio to mono (channel axis is whichever is smaller)
            channel_axis = 0 if audio_array.shape[0] < audio_array.shape[1] else 1
            audio_array = audio_array.mean(axis=channel_axis).astype("float32")

        if src_sr != target_sr:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=src_sr, target_sr=target_sr)

        result = asr({"array": audio_array, "sampling_rate": target_sr}, generate_kwargs=gen_kwargs)
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        return text.strip()
    except Exception as e:
        import traceback
        return f"❌ {e}\n\n{traceback.format_exc()}"


# ──────────────────────────────────────────────────────────────────
# Explicit "force unload / offload from VRAM" actions — wired to an
# "Unload from VRAM" button on each relevant tab. These free memory
# immediately via _release_model() without loading any replacement,
# unlike get_llm()/get_vlm()/get_stt_pipeline() which only release the
# *previous* model right before loading a new one.
# ──────────────────────────────────────────────────────────────────
def unload_llm_fn(lang_key: str = "kh") -> str:
    global _llm, _llm_model_id
    l = LANGUAGES.get(lang_key, LANGUAGES["kh"])
    if _llm is None:
        return l["btn_unload_none"]
    mid = _llm_model_id
    _release_model(_llm)
    _llm = None
    return l["msg_unloaded"].format(model=mid)


def unload_vlm_fn(lang_key: str = "kh") -> str:
    global _vlm_model, _vlm_processor, _vlm_model_id
    l = LANGUAGES.get(lang_key, LANGUAGES["kh"])
    if _vlm_model is None:
        return l["btn_unload_none"]
    mid = _vlm_model_id
    _release_model(_vlm_model)
    _vlm_model = None
    _vlm_processor = None
    _vlm_model_id = None
    return l["msg_unloaded"].format(model=mid)


def unload_stt_fn(lang_key: str = "kh") -> str:
    global _stt_pipeline, _stt_model_id
    l = LANGUAGES.get(lang_key, LANGUAGES["kh"])
    if _stt_pipeline is None:
        return l["btn_unload_none"]
    mid = _stt_model_id
    _release_model(_stt_pipeline)
    _stt_pipeline = None
    _stt_model_id = None
    return l["msg_unloaded"].format(model=mid)


def force_reload_llm(target_model_id: str):
    """Used by the tab-level 'Load' buttons: fully release whatever LLM is
    currently loaded (regardless of id) and load `target_model_id` fresh."""
    global _llm
    _release_model(_llm)
    _llm = None
    return get_llm(target_model_id)


def force_reload_vlm(target_model_id: str):
    global _vlm_model, _vlm_processor
    _release_model(_vlm_model)
    _vlm_model = _vlm_processor = None
    return get_vlm(target_model_id)


def force_reload_stt(target_model_id: str):
    global _stt_pipeline, _stt_model_id
    _release_model(_stt_pipeline)
    _stt_pipeline = None
    _stt_model_id = None
    return get_stt_pipeline(target_model_id)
