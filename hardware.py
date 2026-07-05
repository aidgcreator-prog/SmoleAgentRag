"""
hardware.py — GPU/device detection and the "Fix Environment" self-repair
helper (installs a hardware-matched PyTorch build at runtime).
"""

import subprocess
import sys
from typing import Optional

import torch


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


# ── Resolved once at import time ────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

TORCH_DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32


def refresh_system_ui():
    status = HardwareManager.get_system_status()
    return (
        "🔄 Status Refreshed",
        status["current_device"].upper(),
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
