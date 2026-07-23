@echo off
setlocal
set "SCRIPT_DIR=%~dp0"

REM This launcher is intentionally hardware-agnostic — it never assumes
REM any specific GPU/CPU. All hardware detection (NVIDIA/AMD/CPU-only,
REM driver CUDA version, per-GPU compute capability) and the real-kernel
REM verification of the installed PyTorch build happen dynamically inside
REM SETUP.ps1 for WHATEVER machine this is run on, since this app is
REM distributed to end users with unknown/varied hardware — see SETUP.ps1
REM Step 4 (GPU detection), Step 5b (GPU wheel verification + automatic
REM CPU fallback), and Step 8 (Playwright + Chromium install/verification
REM for Deep Research's optional browser tools) for the actual logic.
REM
REM Any arguments passed to this .bat are forwarded straight through to
REM SETUP.ps1 — e.g. double-click normally to be asked interactively
REM whether to install/build llama-cpp-python (the slowest setup step),
REM or run from a command prompt as:
REM   SETUP.bat -SkipLlamaCpp
REM to skip it without being asked (GGUF models remain usable afterward
REM via the "llama-server (external process)" backend — see the app's
REM "🖥️ llama-server.exe Path" / "⚙️ LLM Backend" UI controls), or:
REM   SETUP.bat -InstallLlamaCppForced -NonInteractive
REM for a fully unattended install that still builds it.

if not exist "%SCRIPT_DIR%SETUP.ps1" (
    echo [ERROR] SETUP.ps1 not found next to this file.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%SETUP.ps1" %*
set EXITCODE=%errorlevel%

exit /b %EXITCODE%
