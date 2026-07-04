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
        size_gb = f.stat().st_size / (1024 ** 3)
        label = f"🦙 {f.stem}  (~{size_gb:.1f} GB | {mode_tag})"
        found[label] = str(f)
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
            plain.append({"role": str(role), "content": self._flatten_content(content)})
        return plain

    def generate(self, messages: list, stop_sequences: Optional[list] = None, **kwargs):
        from smolagents.models import ChatMessage
        plain_messages = self._to_plain_messages(messages)
        out = self.llm.create_chat_completion(
            messages=plain_messages,
            stop=stop_sequences or [],
            max_tokens=kwargs.get("max_tokens", self.max_new_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            top_p=kwargs.get("top_p", self.top_p),
        )
        text = out["choices"][0]["message"]["content"]
        text = self._sanitize_content(text)
        return ChatMessage(role="assistant", content=text)

    # smolagents Model instances are called directly in some code paths
    def __call__(self, messages: list, stop_sequences: Optional[list] = None, **kwargs):
        return self.generate(messages, stop_sequences=stop_sequences, **kwargs)
