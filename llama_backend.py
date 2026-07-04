"""
llama_backend.py — Optional llama.cpp (GGUF) backend, alongside the
HuggingFace/transformers backend used elsewhere in the app.

If llama-cpp-python isn't installed, GGUF models simply won't show up in
the model dropdowns — everything else keeps working.
"""

import contextlib
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

import user_config

try:
    from llama_cpp import Llama as LlamaCppBackend
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LlamaCppBackend = None
    LLAMA_CPP_AVAILABLE = False


def _detect_llama_cpp_gpu_support() -> bool:
    """Best-effort check of whether the installed llama-cpp-python build was
    actually compiled with GPU offload support (CUDA / Metal / ROCm).

    This is independent of torch's CUDA detection — llama-cpp-python is a
    separate compiled extension and, unless installed via SETUP.ps1 (which
    picks a hardware-matched wheel) or built from source with the right
    CMAKE_ARGS, pip installs a CPU-only build by default even on a machine
    with a working NVIDIA GPU.

    Falls back to True (optimistic) if the capability can't be introspected,
    since requesting GPU layers on a CPU-only build simply no-ops rather
    than crashing — the model still loads, just entirely on CPU.
    """
    if not LLAMA_CPP_AVAILABLE:
        return False
    try:
        import llama_cpp as _llama_cpp_module
        supports_fn = getattr(_llama_cpp_module, "llama_supports_gpu_offload", None)
        if supports_fn is not None:
            return bool(supports_fn())
    except Exception:
        pass
    return True


LLAMA_CPP_GPU_AVAILABLE = _detect_llama_cpp_gpu_support()
if LLAMA_CPP_AVAILABLE:
    print(f"[llama.cpp] GPU offload support: "
          f"{'available' if LLAMA_CPP_GPU_AVAILABLE else 'NOT available (CPU-only build — GGUF models will run on CPU)'}")

# ──────────────────────────────────────────────────────────────────
# GGUF model folder — NOT hardcoded. Resolution order:
#   1. LLAMA_CPP_MODEL_DIR environment variable, if set (highest priority —
#      lets you override per-launch without touching saved settings)
#   2. The path remembered from a previous run (user_config.json), so the
#      user is only asked for it once instead of on every startup
#   3. Empty (feature off) if neither is set
# It can also be set or changed at runtime from the "📁 GGUF Model Folder"
# box in the UI (see set_model_dir() below) — no code edits or restart
# required either way, and doing so updates user_config.json immediately.
# ──────────────────────────────────────────────────────────────────
LLAMA_CPP_MODEL_DIR = (
    os.environ.get("LLAMA_CPP_MODEL_DIR", "").strip()
    or str(user_config.USER_CONFIG.get("gguf_model_dir", "")).strip()
)


def set_model_dir(folder_path: str) -> None:
    """Update the module-level GGUF folder and persist it, so the change is
    visible to every other module that reads llama_backend.LLAMA_CPP_MODEL_DIR
    and survives an app restart.
    """
    global LLAMA_CPP_MODEL_DIR
    folder_path = (folder_path or "").strip()
    LLAMA_CPP_MODEL_DIR = folder_path
    user_config.save_user_config({"gguf_model_dir": folder_path})


def discover_gguf_models(folder: Optional[str] = None) -> dict:
    """Scan a folder for .gguf files and return {label: path} entries
    that can be merged straight into MODEL_OPTIONS.

    If `folder` is not given, the current global LLAMA_CPP_MODEL_DIR is
    used. An empty/unset folder simply yields no GGUF models — the app
    works fine without one.
    """
    if not LLAMA_CPP_AVAILABLE:
        return {}
    folder = folder if folder is not None else LLAMA_CPP_MODEL_DIR
    if not folder:
        return {}
    p = Path(folder)
    if not p.exists():
        return {}
    mode_tag = "llama.cpp | GPU" if LLAMA_CPP_GPU_AVAILABLE else "llama.cpp | CPU only — slow"
    found = {}
    for f in sorted(p.glob("*.gguf")):
        # mmproj (multimodal projector / CLIP vision encoder) files are
        # only usable paired with a main vision model — see
        # discover_gguf_vlm_models() below — they're not a standalone
        # text LLM and would just fail to load if picked here.
        if "mmproj" in f.stem.lower():
            continue
        size_gb = f.stat().st_size / (1024 ** 3)
        label = f"🦙 {f.stem}  (~{size_gb:.1f} GB | {mode_tag})"
        found[label] = str(f)
    return found


def discover_gguf_vlm_models(folder: Optional[str] = None) -> dict:
    """Scan a folder for GGUF vision-language model PAIRS.

    Unlike a text-only GGUF model (one .gguf file is enough),
    llama.cpp's multimodal support needs TWO files: the main model .gguf
    plus its matching "mmproj" (multimodal projector / CLIP vision
    encoder) .gguf. A main .gguf with no mmproj file anywhere in the
    folder simply can't run as a vision model, so it's skipped here (it
    can still show up as a normal TEXT model via discover_gguf_models()).

    Returns {label: (model_path, mmproj_path)}.

    Pairing heuristic (best effort — GGUF metadata doesn't record this):
      - Exactly one mmproj file in the folder -> pair it with EVERY main
        model file (the common case: a release ships one model file +
        one mmproj file together).
      - Multiple mmproj files -> pair each main model with whichever
        mmproj shares the longest matching filename prefix (vision GGUF
        releases usually name the mmproj after the model, e.g.
        'Qwen2-VL-7B-Instruct-Q4_K_M.gguf' +
        'Qwen2-VL-7B-Instruct-mmproj-f16.gguf').
    """
    if not LLAMA_CPP_AVAILABLE:
        return {}
    folder = folder if folder is not None else LLAMA_CPP_MODEL_DIR
    if not folder:
        return {}
    p = Path(folder)
    if not p.exists():
        return {}

    all_gguf     = sorted(p.glob("*.gguf"))
    mmproj_files = [f for f in all_gguf if "mmproj" in f.stem.lower()]
    main_files   = [f for f in all_gguf if "mmproj" not in f.stem.lower()]
    if not mmproj_files or not main_files:
        return {}

    def _shared_prefix_len(a: str, b: str) -> int:
        n = 0
        for ca, cb in zip(a.lower(), b.lower()):
            if ca != cb:
                break
            n += 1
        return n

    mode_tag = "llama.cpp | GPU" if LLAMA_CPP_GPU_AVAILABLE else "llama.cpp | CPU only — slow"
    found = {}
    for main_f in main_files:
        best_mmproj = (mmproj_files[0] if len(mmproj_files) == 1
                       else max(mmproj_files, key=lambda m: _shared_prefix_len(main_f.stem, m.stem)))
        size_gb = main_f.stat().st_size / (1024 ** 3)
        mm_gb   = best_mmproj.stat().st_size / (1024 ** 3)
        label = (f"🦙👁️ {main_f.stem}  (~{size_gb:.1f} GB + mmproj "
                 f"~{mm_gb:.1f} GB | {mode_tag})")
        found[label] = (str(main_f), str(best_mmproj))
    return found


@contextlib.contextmanager
def _capture_native_output():
    """Capture stdout/stderr at the OS file-descriptor level.

    llama.cpp is a C/C++ library — its logging (even when llama-cpp-python's
    `verbose=True` is set) writes directly to the process's real fd 1/2,
    bypassing Python's `sys.stdout`/`sys.stderr` entirely. A normal
    `contextlib.redirect_stdout` can't see it. Redirecting the actual file
    descriptors is the only way to capture the real ggml/llama.cpp error
    line (bad magic number, unknown architecture, missing split-file part,
    truncated download, etc.) instead of just the generic Python wrapper
    exception.
    """
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    tmp = tempfile.TemporaryFile(mode="w+b")
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(tmp.fileno(), stdout_fd)
        os.dup2(tmp.fileno(), stderr_fd)
        yield tmp
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def _raise_gguf_load_error(model_path: str, n_ctx: int, n_gpu_layers: int,
                            original_exc: Exception):
    """Reload the model once more with verbose=True, capturing llama.cpp's
    native log output, and raise a RuntimeError that includes the *real*
    reason the load failed — instead of just the generic wrapper message
    that verbose=False normally leaves us with.
    """
    diag_text = ""
    try:
        with _capture_native_output() as tmp:
            try:
                LlamaCppBackend(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    flash_attn=False,
                    verbose=True,
                )
            except Exception:
                pass  # we only care about the captured log, not this exception
        tmp.seek(0)
        diag_text = tmp.read().decode("utf-8", errors="replace").strip()
        tmp.close()
    except Exception:
        diag_text = ""

    diag_tail = "\n".join(diag_text.splitlines()[-15:]) if diag_text else \
        "(no additional native log captured — this environment may not " \
        "support fd-level output redirection)"

    raise RuntimeError(
        f"Failed to load GGUF model '{model_path}'.\n\n"
        f"Real llama.cpp log from a diagnostic reload:\n"
        f"{'-' * 60}\n{diag_tail}\n{'-' * 60}\n\n"
        "Common causes:\n"
        "  1. The file is corrupted or an incomplete download — check its "
        "size against the source repo.\n"
        "  2. It's one part of a split multi-file GGUF (e.g. "
        "'-00001-of-00002.gguf') and the other part(s) are missing from "
        "the folder — keep all parts together and point at part 1.\n"
        "  3. The installed llama-cpp-python build's llama.cpp version is "
        "too old for this model's architecture/metadata (common for newer "
        "or uncommon MoE/hybrid-attention checkpoints) — try `pip install "
        "llama-cpp-python --upgrade --force-reinstall` (or rerun "
        "SETUP.bat), or re-download the file.\n\n"
        f"Original error: {original_exc}"
    ) from original_exc


class LlamaCppModel:
    """Minimal smolagents-compatible Model wrapper around llama-cpp-python.

    Satisfies the smolagents Model contract: callable/`generate()` taking a
    list of chat messages and returning an object with a `.content` attribute.
    Lets .gguf models sit in the same MODEL_OPTIONS dropdown, and be passed
    to smolagents' CodeAgent, as the HuggingFace TransformersModel entries.
    """

    def __init__(self, model_path: str, n_ctx: int = 16384,
                 n_gpu_layers: int = -1, temperature: float = 0.6,
                 top_p: float = 0.95, max_new_tokens: int = 512,
                 flash_attn: bool = True):
        self.model_id = model_path
        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        print(f"[llama.cpp] Loading '{model_path}' (n_gpu_layers={n_gpu_layers}, flash_attn={flash_attn}) …")
        try:
            self.llm = LlamaCppBackend(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,   # -1 = offload all layers if the build supports GPU
                flash_attn=flash_attn,
                verbose=False,
            )
        except TypeError:
            # Installed llama-cpp-python is too old to accept flash_attn= at all
            print("[llama.cpp] This llama-cpp-python build doesn't support the "
                  "flash_attn parameter — loading without it.")
            self.llm = LlamaCppBackend(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
        except Exception as e:
            if flash_attn:
                # Some architectures (e.g. hybrid local/global attention models
                # with mismatched per-layer head dims) reject flash attention.
                # Retry without it — but if THIS also fails, the problem isn't
                # flash attention at all (e.g. an unsupported/newer GGUF
                # architecture, a corrupted download, or a missing split-file
                # part), so run one more diagnostic reload with verbose
                # logging captured, and surface the real llama.cpp error
                # instead of a confusing raw double traceback.
                print(f"[llama.cpp] flash_attn=True failed to load ({e}); "
                      f"retrying without flash attention …")
                try:
                    self.llm = LlamaCppBackend(
                        model_path=model_path,
                        n_ctx=n_ctx,
                        n_gpu_layers=n_gpu_layers,
                        flash_attn=False,
                        verbose=False,
                    )
                except Exception as e2:
                    _raise_gguf_load_error(model_path, n_ctx, n_gpu_layers, e2)
            else:
                # Loading without flash attention failed on the first try —
                # still worth a diagnostic reload to surface the real reason
                # (corrupted file, missing split part, unsupported arch, …)
                # rather than the bare wrapper exception.
                _raise_gguf_load_error(model_path, n_ctx, n_gpu_layers, e)

    # Some GGUF checkpoints — particularly MoE "thinking"/harmony-style
    # fine-tunes — embed a chat template that llama.cpp/llama-cpp-python
    # doesn't always render identically to how the model was actually
    # trained. When that happens, fragments of the model's internal
    # channel-routing tokens (e.g. "<|channel|>analysis", "<channel|>",
    # "<|message|>") leak into the plain-text output instead of being
    # consumed as structural tokens — which then confuses anything
    # downstream trying to parse the text (e.g. smolagents' code-block
    # regex, or format_llm_response()'s <think> tag matching).
    _LEAKED_CONTROL_TOKEN_RE = re.compile(
        r"<\|?/?(?:channel|message|start|end|return|call)\|?>",
        re.IGNORECASE,
    )

    @classmethod
    def _sanitize_content(cls, text: str) -> str:
        """Best-effort cleanup of leaked chat-template control tokens.

        This does NOT fix the underlying template mismatch — it only
        strips known leaked fragments so downstream parsers see clean
        text. If a model consistently leaks these, its GGUF conversion
        likely ships a chat template that doesn't match how the model
        was fine-tuned, and switching to a different quantization/
        conversion of the same model (or using it via the HuggingFace/
        transformers backend instead) is the more reliable fix.
        """
        if not text:
            return text
        return cls._LEAKED_CONTROL_TOKEN_RE.sub("", text).strip()

    @staticmethod
    def _flatten_content(content) -> str:
        if isinstance(content, list):
            return " ".join(
                c.get("text", "") if isinstance(c, dict) else str(c) for c in content
            )
        return str(content)

    def _to_plain_messages(self, messages: list) -> list:
        plain = []
        for m in messages:
            role = getattr(m, "role", None) or m.get("role")
            content = getattr(m, "content", None)
            if content is None:
                content = m.get("content")
            # `role` is smolagents' MessageRole — a `str`+`Enum` hybrid.
            # Since Python 3.11, str() on a str-based Enum member no longer
            # returns the raw value; it returns "MessageRole.USER" instead
            # of "user" (a well-known 3.11 str-Enum __str__ change). Blindly
            # calling str(role) therefore corrupted EVERY role sent to
            # llama.cpp's create_chat_completion (not just this fallback
            # path) — the chat template got "MessageRole.USER" instead of
            # "user" on every turn, and the "any role == 'user'" check below
            # then always failed too, firing the "(continue)" fallback on
            # every single call instead of only genuine no-user-turn cases.
            # `.value` gives the correct raw string for an Enum member,
            # and is a no-op passthrough for a plain str role.
            role_value = getattr(role, "value", role)
            plain.append({"role": str(role_value), "content": self._flatten_content(content)})

        # Some GGUF chat templates (notably Gemma's) call Jinja's `strip()`
        # (or similar) directly on the LAST message's expected-to-be-user
        # content, or otherwise assume at least one role="user" turn is
        # present, and raise a jinja2/ValueError ("No user query found in
        # messages") if that assumption is violated. This can happen when
        # replayed agent memory (agent.run(task, reset=False)) is fed into
        # a *different* model/template than the one that produced it (e.g.
        # right after a model switch), or when a memory-capping edge case
        # leaves only system/assistant turns. Guarantee at least one user
        # turn exists so the template never sees an empty case.
        if not any(p["role"] == "user" for p in plain):
            # With the role.value fix above, this should now only fire for
            # genuinely broken/incompatible replayed memory (e.g. a
            # corrupted persisted-memory file) — not on every call. Log it
            # loudly so that if it DOES fire, it's immediately visible in
            # the console instead of silently confusing whatever model is
            # loaded (which previously just saw a bare, meaningless
            # "(continue)" and tried to answer it literally).
            print("[llama.cpp] WARNING: no 'user' role found in the message "
                  "history sent to this GGUF model — injecting a placeholder "
                  "user turn so the chat template doesn't reject the request. "
                  "This usually means replayed agent memory is malformed or "
                  "incompatible (e.g. a corrupted entry in "
                  "./agent_memory_store/). Consider clicking 'Clear' on the "
                  "affected tab to drop its saved memory.")
            plain.append({"role": "user", "content": "(continue)"})

        return plain

    def generate(self, messages: list, stop_sequences: Optional[list] = None, **kwargs):
        from smolagents.models import ChatMessage
        plain_messages = self._to_plain_messages(messages)
        try:
            out = self.llm.create_chat_completion(
                messages=plain_messages,
                stop=stop_sequences or [],
                max_tokens=kwargs.get("max_tokens", self.max_new_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
            )
        except ValueError as e:
            # Chat-template rendering failures — e.g. a model's Jinja
            # template rejecting a message list it doesn't like the shape
            # of (missing user turn, unexpected role ordering, etc.) — are
            # a template/model-compatibility issue, not a crash-worthy app
            # bug. Surface it as assistant text (mirrors how the rest of
            # the app's chat handlers already catch-and-display exceptions)
            # instead of letting it propagate as an unhandled traceback.
            return ChatMessage(
                role="assistant",
                content=(
                    f"⚠️ This model's chat template couldn't process the "
                    f"conversation (possibly after a memory replay across a "
                    f"model switch): {e}\n\n"
                    f"Try clearing/resetting the conversation, or rephrasing "
                    f"your message."
                ),
            )
        text = out["choices"][0]["message"]["content"]
        text = self._sanitize_content(text)
        return ChatMessage(role="assistant", content=text)

    # smolagents Model instances are called directly in some code paths
    def __call__(self, messages: list, stop_sequences: Optional[list] = None, **kwargs):
        return self.generate(messages, stop_sequences=stop_sequences, **kwargs)


class LlamaCppVLMModel:
    """Loader/wrapper for GGUF vision-language models via llama.cpp's
    multimodal support.

    Used directly by models.vlm_answer() — Vision Chat has no smolagents
    CodeAgent involved, so unlike LlamaCppModel above, this doesn't need
    to satisfy smolagents' Model contract.

    llama.cpp needs TWO files to run a vision model: the main .gguf plus
    a "mmproj" (multimodal projector / CLIP vision encoder) .gguf, loaded
    through one of llama-cpp-python's `chat_handler` classes. Which
    handler class matches a given model isn't recorded in the GGUF
    metadata discover_gguf_vlm_models() can see, so this makes a
    best-effort guess from the filename and falls back to the most
    broadly-compatible handler (Llava15ChatHandler — works for the
    majority of llava-clip-style vision GGUF releases) if nothing more
    specific matches. A wrong handler guess doesn't necessarily crash,
    but can produce poor answers, since it feeds the model the wrong
    image-token/prompt format — if that happens, check whether a newer
    llama-cpp-python ships a more specific handler for that model family.
    """

    # Filename substring -> chat_handler class name in
    # llama_cpp.llama_chat_format. Checked in order; first match wins.
    _HANDLER_HINTS = (
        ("minicpm-v-2.6", "MiniCPMv26ChatHandler"),
        ("minicpm-v",     "MiniCPMv26ChatHandler"),
        ("moondream2",    "Moondream2ChatHandler"),
        ("moondream",     "MoondreamChatHandler"),
        ("nanollava",     "NanoLlavaChatHandler"),
        ("nanovlm",       "NanoLlavaChatHandler"),
        ("llava-1.6",     "Llava16ChatHandler"),
        ("llava-v1.6",    "Llava16ChatHandler"),
        ("llava1.6",      "Llava16ChatHandler"),
        ("llava-1.5",     "Llava15ChatHandler"),
        ("llava-v1.5",    "Llava15ChatHandler"),
    )
    _DEFAULT_HANDLER = "Llava15ChatHandler"

    def __init__(self, model_path: str, mmproj_path: str, n_ctx: int = 4096,
                 n_gpu_layers: int = -1):
        self.model_path  = model_path
        self.mmproj_path = mmproj_path

        from llama_cpp import llama_chat_format

        handler_name = self._DEFAULT_HANDLER
        stem_lower = Path(model_path).stem.lower()
        for hint, name in self._HANDLER_HINTS:
            if hint in stem_lower and hasattr(llama_chat_format, name):
                handler_name = name
                break
        if not hasattr(llama_chat_format, handler_name):
            raise RuntimeError(
                f"Installed llama-cpp-python has no '{handler_name}' chat "
                f"handler — upgrade it to use GGUF vision models: "
                f"pip install --upgrade llama-cpp-python"
            )
        self.handler_name = handler_name
        HandlerClass = getattr(llama_chat_format, handler_name)
        print(f"[llama.cpp VLM] Loading '{model_path}' + mmproj "
              f"'{mmproj_path}' using {handler_name} …")
        chat_handler = HandlerClass(clip_model_path=mmproj_path, verbose=False)

        try:
            self.llm = LlamaCppBackend(
                model_path=model_path,
                chat_handler=chat_handler,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                logits_all=True,  # required for multimodal on some llama-cpp-python versions
                verbose=False,
            )
        except TypeError:
            # Newer llama-cpp-python releases dropped logits_all entirely.
            self.llm = LlamaCppBackend(
                model_path=model_path,
                chat_handler=chat_handler,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )

    @staticmethod
    def _image_to_data_url(img) -> str:
        import base64
        from io import BytesIO
        buf = BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def answer(self, question: str, images: list, context: str = "",
               max_tokens: int = 512, temperature: float = 0.6) -> str:
        user_text = question + (f"\n\nContext:\n{context}" if context else "")
        content = [{"type": "image_url", "image_url": {"url": self._image_to_data_url(img)}}
                   for img in images]
        content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer based on images and context."},
            {"role": "user", "content": content},
        ]
        out = self.llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=temperature,
        )
        return out["choices"][0]["message"]["content"].strip()
