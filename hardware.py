"""
hardware.py — GPU/device detection and the "Fix Environment" self-repair
helper (installs a hardware-matched PyTorch build at runtime).
"""

import subprocess
import sys
from typing import Optional

import torch
import warnings


class HardwareManager:
    @staticmethod
    def detect_vram_gb() -> Optional[float]:
        """Best-effort total VRAM (in GB) of the primary CUDA GPU, if any.

        Returns None if there's no CUDA GPU, or detection fails for any
        reason — callers should treat None as "unknown", not "zero", since
        a failed *detection* is not the same as genuinely having no GPU.
        """
        try:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return props.total_memory / (1024 ** 3)
        except Exception:
            pass
        return None

    @staticmethod
    def detect_system_ram_gb() -> Optional[float]:
        """Best-effort total system RAM (in GB).

        Prefers `psutil` (not a hard requirement of this app — see
        requirements.txt) since it's cross-platform; falls back to
        reading /proc/meminfo directly on Linux if psutil isn't
        installed. Returns None if neither approach works (e.g. Windows
        without psutil) — callers should treat this the same as an
        unknown GPU: fall back to a conservative default tier rather than
        guessing a number that could be wildly wrong.
        """
        try:
            import psutil
            return psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            pass
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / (1024 ** 2)
        except Exception:
            pass
        return None

    # ──────────────────────────────────────────────────────────────
    # Hardware tiers — mirrors the "Model combos by hardware tier" table
    # in README.md exactly, so the HF-side dynamic model registry (see
    # model_registry.get_recommended_hf_models()) recommends the same
    # models the README documents for each tier. Checked from the
    # top (most capable) down, so a machine that qualifies for a higher
    # tier is never mistakenly placed in a lower one.
    # ──────────────────────────────────────────────────────────────
    TIER_48GB_VRAM = "48gb_vram"
    TIER_24GB_VRAM = "24gb_vram"
    TIER_16GB_VRAM = "16gb_vram"
    TIER_8GB_VRAM  = "8gb_vram"
    TIER_CPU_ONLY  = "cpu_only"
    TIER_UNKNOWN   = "unknown"

    @staticmethod
    def detect_hardware_tier() -> str:
        """Classify the current machine into one of the README's hardware
        tiers, based on detected VRAM (if a CUDA GPU is present) or system
        RAM (CPU-only case). Returns TIER_UNKNOWN if neither VRAM nor RAM
        could be determined — callers should fall back to the app's
        existing small defaults (BASE_MODEL_OPTIONS) in that case rather
        than guessing a tier that might recommend a model too large for
        the actual machine.
        """
        vram_gb = HardwareManager.detect_vram_gb()
        if vram_gb is not None:
            if vram_gb >= 44:   # 48 GB tier — a little headroom below the nominal 48
                return HardwareManager.TIER_48GB_VRAM
            if vram_gb >= 20:   # 24 GB tier
                return HardwareManager.TIER_24GB_VRAM
            if vram_gb >= 14:   # 16 GB tier
                return HardwareManager.TIER_16GB_VRAM
            if vram_gb >= 6:    # 8 GB tier — some headroom below nominal 8
                return HardwareManager.TIER_8GB_VRAM
            # A CUDA GPU exists but has less VRAM than any tier above
            # assumes — safer to fall back to CPU-only-style (smaller)
            # recommendations than to recommend an 8GB-tier model that
            # won't fit.
            return HardwareManager.TIER_CPU_ONLY

        ram_gb = HardwareManager.detect_system_ram_gb()
        if ram_gb is not None and ram_gb >= 56:   # 64GB tier — some headroom below nominal 64
            return HardwareManager.TIER_CPU_ONLY

        return HardwareManager.TIER_UNKNOWN

    @staticmethod
    def detect_nvidia_cuda_version() -> Optional[str]:
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "CUDA Version:" in line:
                        return line.split("CUDA Version:")[1].strip()
            return None
        except Exception:
            return None

    @staticmethod
    def get_required_torch_index(cuda_version: str) -> str:
        if not cuda_version:
            return "https://download.pytorch.org/whl/cpu"

        try:
            v_parts = cuda_version.split('.')
            major = v_parts[0]
            minor = v_parts[1]

            if major == "11":
                return "https://download.pytorch.org/whl/cu118"
            elif major == "12":
                if minor in ("1", "2", "3"):
                    return "https://download.pytorch.org/whl/cu121"
                elif minor in ("4", "5", "6"):
                    return "https://download.pytorch.org/whl/cu124"
                elif minor in ("7", "8", "9"):
                    return "https://download.pytorch.org/whl/cu128"
                else:
                    return "https://download.pytorch.org/whl/cu128"
            return "https://download.pytorch.org/whl/cpu"
        except Exception:
            return "https://download.pytorch.org/whl/cpu"

    @staticmethod
    def get_system_status() -> dict:
        status = {
            "gpu_brand": "none",
            "cuda_version": None,
            "torch_cuda_available": False,
            "current_device": "cpu",
            "recommended_index": "https://download.pytorch.org/whl/cpu"
        }

        cuda_ver = HardwareManager.detect_nvidia_cuda_version()
        if cuda_ver:
            status["gpu_brand"] = "nvidia"
            status["cuda_version"] = cuda_ver
            status["recommended_index"] = HardwareManager.get_required_torch_index(cuda_ver)

        try:
            status["torch_cuda_available"] = torch.cuda.is_available()
            if status["torch_cuda_available"]:
                status["current_device"] = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                status["current_device"] = "mps"
            else:
                status["current_device"] = "cpu"
        except Exception:
            pass

        if status["gpu_brand"] == "none":
            try:
                result = subprocess.run(["wmic", "path", "win32_VideoController", "get", "name"],
                                        capture_output=True, text=True, check=False)
                if "Radeon" in result.stdout or "AMD" in result.stdout:
                    status["gpu_brand"] = "amd"
                    status["recommended_index"] = "https://download.pytorch.org/whl/rocm6.2"
            except Exception:
                pass

        return status

    @staticmethod
    def fix_environment_gen():
        status = HardwareManager.get_system_status()
        if status["torch_cuda_available"]:
            yield "✅ Environment is already correctly configured for GPU!", "cuda", ""
            return

        index = status["recommended_index"]
        yield "🚀 Starting environment fix...", "cuda", "🚀 Starting environment fix..."
        yield f"🔍 Detected GPU: {status['gpu_brand'].upper()}", "cuda", f"🔍 Detected GPU: {status['gpu_brand'].upper()}"
        yield f"🔗 Using PyTorch index: {index}", "cuda", f"🔗 Using PyTorch index: {index}"

        try:
            cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", index]
            yield f"Running: {' '.join(cmd)}", "cuda", f"Running: {' '.join(cmd)}"

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

            for line in process.stdout:
                yield "🛠️ Installing...", "cuda", line.strip()

            process.wait()

            if process.returncode == 0:
                yield "✨ Installation successful!", "cuda", "✨ Installation successful!"
                yield "✅ Environment fixed! Please RESTART the app to apply changes.", "cuda", "✅ Environment fixed! Please RESTART the app to apply changes."
            else:
                yield f"❌ Installation failed with exit code {process.returncode}", "cpu", f"❌ Installation failed with exit code {process.returncode}"
                yield "❌ Installation failed. Check the logs above.", "cpu", "❌ Installation failed. Check the logs above."

        except Exception as e:
            yield f"❌ Error: {str(e)}", "cpu", f"❌ Error: {str(e)}"


# ──────────────────────────────────────────────────────────────────
# Real GPU/PyTorch kernel-compatibility check
# ──────────────────────────────────────────────────────────────────
# `torch.cuda.is_available()` only checks that the CUDA *driver* and
# runtime can be initialized — it does NOT mean this specific PyTorch
# build actually ships compiled kernels for this specific GPU's compute
# capability. PyTorch wheels only include kernels ("sm_XX" binaries) for
# a fixed list of architectures; older cards (e.g. Pascal/sm_6x,
# Maxwell/sm_5x) have been dropped from recent stable releases even
# though `torch.cuda.is_available()` still happily reports True and the
# model even *loads* onto the device — it only blows up the moment a
# real CUDA kernel is launched (e.g. during model loading's
# tie_weights()), with "CUDA error: no kernel image is available for
# execution on the device".
#
# This is exactly the gap that let a machine with e.g. an NVIDIA GeForce
# MX230 (compute capability 6.1) silently get a "cuda-capable" verdict
# from is_available(), install a GPU build via SETUP.ps1, and then crash
# on first real use. SETUP.ps1 now does a real-kernel smoke test right
# after installing torch (mirroring the same pattern already used there
# for llama-cpp-python) and falls back to a CPU wheel automatically if it
# fails — but a user could still end up here via a manual `pip install
# torch` re-install, a different Python environment, a driver/GPU swap,
# etc. So this same check is repeated here at app-import time as a
# runtime safety net: DEVICE only resolves to "cuda" if a real kernel
# actually runs successfully, not just because is_available() said so.
# ──────────────────────────────────────────────────────────────────
def _detect_gpu_kernel_incompatibility() -> Optional[dict]:
    """Best-effort: if a CUDA GPU is visible but this torch build has no
    compiled kernels for its compute capability, return a small dict with
    enough detail for the UI to build a clear, actionable warning
    (gpu_name, compute capability, and the archs this torch build DOES
    support). Returns None if the GPU is fully usable (or there's no GPU
    at all — that's a normal CPU-only setup, not a warning-worthy state).
    """
    if not torch.cuda.is_available():
        return None
    try:
        gpu_name = torch.cuda.get_device_name(0)
        cc_major, cc_minor = torch.cuda.get_device_capability(0)
        cc_str = f"{cc_major}.{cc_minor}"
        try:
            arch_list = torch.cuda.get_arch_list()
        except Exception:
            arch_list = []
        archs_str = ", ".join(a.replace("sm_", "") for a in arch_list) if arch_list else "unknown"

        # The real test: actually launch a kernel, not just check
        # is_available(). Wrapped so a torch UserWarning about the CC
        # mismatch (torch itself often prints one on this exact call)
        # doesn't spam the console a second time here — SETUP.ps1 /
        # the terminal already shows torch's own version of this warning;
        # this function only needs the pass/fail result.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                x = torch.randn(8, device="cuda")
                _ = x @ x
                torch.cuda.synchronize()
                kernel_ok = True
            except Exception:
                kernel_ok = False

        if kernel_ok:
            return None

        return {
            "gpu_name": gpu_name,
            "compute_capability": cc_str,
            "supported_archs": archs_str,
        }
    except Exception:
        # Detection itself failing shouldn't block startup or produce a
        # false warning — treat as "couldn't determine", not "broken".
        return None


# ── Resolved once at import time ────────────────────────────────────
_GPU_INCOMPATIBILITY_INFO = _detect_gpu_kernel_incompatibility()

if torch.cuda.is_available() and _GPU_INCOMPATIBILITY_INFO is None:
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
    if _GPU_INCOMPATIBILITY_INFO is not None:
        print(
            f"[Hardware] WARNING: GPU '{_GPU_INCOMPATIBILITY_INFO['gpu_name']}' "
            f"(compute capability {_GPU_INCOMPATIBILITY_INFO['compute_capability']}) "
            f"was detected, but this PyTorch build has no compiled kernels for it "
            f"(supports: {_GPU_INCOMPATIBILITY_INFO['supported_archs']}). "
            f"Falling back to CPU — see the warning banner in the UI for how to fix this."
        )


def get_gpu_incompatibility_info() -> Optional[dict]:
    """Read-only accessor for ui.py: returns the dict built at import time
    (see _detect_gpu_kernel_incompatibility() above) describing a
    detected-but-unusable GPU, or None if the GPU is fine / there's no
    GPU. Used to render a one-time warning banner in the UI so a
    non-technical user isn't left guessing why everything is "running on
    CPU" despite clearly having an NVIDIA card — the banner explains it
    and points at the fix.
    """
    return _GPU_INCOMPATIBILITY_INFO


TORCH_DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32


def refresh_system_ui():
    status = HardwareManager.get_system_status()
    return (
        "🔄 Status Refreshed",
        # Report the DEVICE this app is ACTUALLY using (resolved once at
        # import time above, already accounting for the real-kernel
        # compatibility check) rather than status["current_device"], which
        # only reflects torch.cuda.is_available() and would misleadingly
        # say "CUDA" even on a machine that silently fell back to CPU due
        # to a GPU/PyTorch-build compute-capability mismatch.
        DEVICE.upper(),
        status["cuda_version"] or "N/A",
        "✅ Yes" if status["torch_cuda_available"] else "❌ No",
        ""  # Clear logs
    )


def fix_system_ui_generator():
    status = HardwareManager.get_system_status()
    yield (
        "🚀 Starting fix...",
        status["current_device"].upper(),
        status["cuda_version"] or "N/A",
        "✅ Yes" if status["torch_cuda_available"] else "❌ No",
        "Starting..."
    )

    for msg, dev, log in HardwareManager.fix_environment_gen():
        yield (
            msg,
            dev.upper(),
            status["cuda_version"] or "N/A",
            "✅ Yes" if status["torch_cuda_available"] else "❌ No",
            log
        )
