@echo off
setlocal EnableDelayedExpansion
title RAG Agent — Running
color 0B

echo.
echo  ============================================================
echo   RAG Agent — smolagents + ChromaDB + Gemma4 / Qwen3.6
echo  ============================================================
echo.

:: ── Check we are in the right folder ─────────────────────────────────────────
if not exist "%~dp0app.py" (
    echo  [ERROR] app.py not found. Run this bat from the rag_agent folder.
    pause
    exit /b 1
)
cd /d "%~dp0"

:: ── Check setup was run ───────────────────────────────────────────────────────
if not exist ".venv\Scripts\python.exe" (
    echo  [ERROR] Virtual environment not found.
    echo.
    echo  Please run SETUP.bat first!
    echo.
    pause
    exit /b 1
)

:: ── Activate venv ─────────────────────────────────────────────────────────────
call ".venv\Scripts\activate.bat"

:: ── Show GPU info ─────────────────────────────────────────────────────────────
echo  Hardware status:
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>nul
if %errorlevel% neq 0 (
    wmic path win32_VideoController get name 2>nul | findstr /i "Radeon\|AMD" >nul
    if !errorlevel! equ 0 (
        echo  (AMD GPU detected — ROCm support may be required)
    ) else (
        echo  (No NVIDIA/AMD GPU detected — running on CPU)
    )
) else (
    echo  (NVIDIA GPU detected)
)

:: ── Launch app ────────────────────────────────────────────────────────────────
echo.
echo  Starting RAG Agent...
echo  Open your browser at:  http://localhost:7860
echo.
echo  Press Ctrl+C to stop the app.
echo  ============================================================
echo.

python app.py

:: If app exits with error
if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] App exited with an error (code %errorlevel%).
    echo  Check the output above for details.
    echo.
    echo  Common fixes:
    echo    - Not enough VRAM: switch to a smaller model in the UI dropdown.
    echo    - Missing packages:  re-run SETUP.bat
    echo    - Port 7860 in use:  close other Gradio apps and retry.
    pause
)

