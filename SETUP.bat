@echo off
setlocal EnableDelayedExpansion
title RAG Agent — Setup
color 0A

echo.
echo  ============================================================
echo   RAG Agent Setup — smolagents + ChromaDB + Gemma4 / Qwen3.6
echo  ============================================================
echo.

:: ── STEP 0: Check we are in the right folder ─────────────────────────────────
if not exist "%~dp0app.py" (
    echo  [ERROR] app.py not found. Please run this bat from the rag_agent folder.
    pause
    exit /b 1
)
cd /d "%~dp0"

:: ── STEP 1: Check / Install Python (requires 3.9+) ──────────────────────────
echo  [1/7] Checking Python installation...
set NEED_PYTHON=0
set PY_FOUND=0

:: Check if python command exists at all
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [WARN] Python not found in PATH.
    set NEED_PYTHON=1
    goto :do_python_install
)

:: Python exists — capture the version string (e.g. "Python 3.11.9" or "Python 2.7.18")
set PY_FOUND=1
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v

:: Extract major and minor version numbers
for /f "tokens=1,2 delims=." %%a in ("!PY_VER!") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)

echo  Found Python !PY_VER!

:: ── Reject Python 2 ────────────────────────────────────────────────────────
if "!PY_MAJOR!"=="2" (
    echo  [WARN] Python 2 is not supported ^(found !PY_VER!^).
    echo         Python 3.9 or higher is required.
    set NEED_PYTHON=1
    goto :do_python_install
)

:: ── Reject Python 3.0 – 3.8 ───────────────────────────────────────────────
if "!PY_MAJOR!"=="3" (
    if !PY_MINOR! LSS 9 (
        echo  [WARN] Python !PY_VER! is too old. Minimum required: Python 3.9
        echo         Will install Python 3.11 alongside your current version.
        set NEED_PYTHON=1
        goto :do_python_install
    )
)

:: ── Python 3.9+ confirmed ──────────────────────────────────────────────────
echo  [OK] Python !PY_VER! meets the minimum requirement ^(3.9+^).
goto :python_ready

:do_python_install
echo.
echo  Downloading Python 3.11.9 installer...
curl -L --progress-bar -o "%TEMP%\python_installer.exe" ^
    "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] Download failed. Check your internet connection and try again.
    echo          Or install Python 3.11 manually from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo  Installing Python 3.11.9 silently (this may take 1-2 minutes)...
:: InstallAllUsers=1  → available system-wide
:: PrependPath=1      → adds python to PATH for all users
:: Include_pip=1      → includes pip
:: Include_launcher=1 → installs py launcher (py.exe)
"%TEMP%\python_installer.exe" /quiet ^
    InstallAllUsers=1 ^
    PrependPath=1 ^
    Include_pip=1 ^
    Include_launcher=1 ^
    Include_test=0

if %errorlevel% neq 0 (
    echo  [ERROR] Python installation failed ^(exit code %errorlevel%^).
    echo          Please install manually from: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Refresh PATH from registry so python is visible in this session
call :RefreshPath

:: Verify python is now callable
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] Python was installed but is still not found in PATH.
    echo          This can happen when the PATH change needs a new shell session.
    echo.
    echo          Please CLOSE this window, open a NEW command prompt, and
    echo          run SETUP.bat again.
    pause
    exit /b 1
)

:: Confirm installed version
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo  [OK] Python !PY_VER! installed successfully.

:python_ready
:: Final sanity — make sure pip is available too
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [WARN] pip not found — attempting to bootstrap it...
    python -m ensurepip --upgrade
    if %errorlevel% neq 0 (
        echo  [ERROR] Could not install pip. Please run:  python -m ensurepip --upgrade
        pause
        exit /b 1
    )
)
echo  [OK] pip is available.

:: ── STEP 2: Create virtual environment ───────────────────────────────────────
echo.
echo  [2/7] Creating virtual environment (.venv)...
if exist ".venv\Scripts\python.exe" (
    echo  [OK] .venv already exists, skipping creation.
) else (
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo  [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  [OK] Virtual environment created.
)

:: Activate venv for remainder of script
call ".venv\Scripts\activate.bat"
echo  [OK] Virtual environment activated.

:: ── STEP 3: Upgrade pip ───────────────────────────────────────────────────────
echo.
echo  [3/7] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo  [OK] pip up to date.

:: ── STEP 4: Detect GPU — NVIDIA / AMD ROCm / CPU ─────────────────────────────
echo.
echo  [4/7] Detecting GPU...
set GPU_BRAND=none
set CUDA_VERSION=cpu
set TORCH_INDEX=https://download.pytorch.org/whl/cpu

:: ── Check NVIDIA first ────────────────────────────────────────────────────────
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    set GPU_BRAND=nvidia
    :: Parse CUDA version reported by nvidia-smi  (e.g. "CUDA Version: 12.4")
    for /f "tokens=9" %%c in ('nvidia-smi ^| findstr /i "CUDA Version"') do set RAW_CUDA=%%c
    echo  [OK] NVIDIA GPU detected. Driver CUDA: !RAW_CUDA!

    :: Map driver CUDA version to the closest PyTorch wheel
    if "!RAW_CUDA:~0,2!"=="11" (
        set CUDA_VERSION=cu118
        set TORCH_INDEX=https://download.pytorch.org/whl/cu118
    ) else if "!RAW_CUDA:~0,4!"=="12.1" (
        set CUDA_VERSION=cu121
        set TORCH_INDEX=https://download.pytorch.org/whl/cu121
    ) else if "!RAW_CUDA:~0,4!"=="12.2" (
        set CUDA_VERSION=cu121
        set TORCH_INDEX=https://download.pytorch.org/whl/cu121
    ) else if "!RAW_CUDA:~0,4!"=="12.3" (
        set CUDA_VERSION=cu121
        set TORCH_INDEX=https://download.pytorch.org/whl/cu121
    ) else if "!RAW_CUDA:~0,4!"=="12.4" (
        set CUDA_VERSION=cu124
        set TORCH_INDEX=https://download.pytorch.org/whl/cu124
    ) else if "!RAW_CUDA:~0,4!"=="12.5" (
        set CUDA_VERSION=cu124
        set TORCH_INDEX=https://download.pytorch.org/whl/cu124
    ) else if "!RAW_CUDA:~0,4!"=="12.6" (
        set CUDA_VERSION=cu124
        set TORCH_INDEX=https://download.pytorch.org/whl/cu124
    ) else if "!RAW_CUDA:~0,4!"=="12.7" (
        set CUDA_VERSION=cu128
        set TORCH_INDEX=https://download.pytorch.org/whl/cu128
    ) else if "!RAW_CUDA:~0,4!"=="12.8" (
        set CUDA_VERSION=cu128
        set TORCH_INDEX=https://download.pytorch.org/whl/cu128
    ) else if "!RAW_CUDA:~0,4!"=="12.9" (
        set CUDA_VERSION=cu128
        set TORCH_INDEX=https://download.pytorch.org/whl/cu128
    ) else (
        :: Unknown / future CUDA — default to latest known wheel (cu128)
        echo  [WARN] Unknown CUDA version '!RAW_CUDA!' — defaulting to cu128 wheel.
        set CUDA_VERSION=cu128
        set TORCH_INDEX=https://download.pytorch.org/whl/cu128
    )
    echo  [OK] Will install PyTorch for CUDA !CUDA_VERSION!
    goto :gpu_done
)

:: ── Check AMD ROCm ────────────────────────────────────────────────────────────
:: Method 1: rocm-smi (ROCm toolkit installed)
rocm-smi >nul 2>&1
if %errorlevel% equ 0 (
    set GPU_BRAND=amd_rocm
    goto :amd_detected
)

:: Method 2: rocminfo (alternate ROCm tool)
rocminfo >nul 2>&1
if %errorlevel% equ 0 (
    set GPU_BRAND=amd_rocm
    goto :amd_detected
)

:: Method 3: Check Windows Device Manager via WMIC for AMD/Radeon GPU
wmic path win32_VideoController get name 2>nul | findstr /i "Radeon\|AMD" >nul 2>&1
if %errorlevel% equ 0 (
    set GPU_BRAND=amd_no_rocm
    goto :amd_no_rocm
)

:: ── No GPU found → CPU ────────────────────────────────────────────────────────
echo  [WARN] No GPU detected (no nvidia-smi, rocm-smi, or AMD GPU in device list).
echo         Installing PyTorch for !CUDA_VERSION!.
set GPU_BRAND=cpu
set CUDA_VERSION=cpu
set TORCH_INDEX=https://download.pytorch.org/whl/cpu
goto :gpu_done

:amd_detected
:: AMD GPU with ROCm toolkit — use ROCm PyTorch wheel
echo  [OK] AMD GPU with ROCm detected.
:: Detect ROCm version from rocm-smi or rocminfo
set ROCM_VERSION=6.2
for /f "tokens=*" %%r in ('rocm-smi --showdriverversion 2^>nul ^| findstr /i "ROCm\|version"') do (
    set ROCM_RAW=%%r
)
:: PyTorch currently ships ROCm 6.2 wheels for Windows (rocm6.2)
:: Map to available wheel — update this list as PyTorch ships new ROCm builds
set CUDA_VERSION=rocm6.2
set TORCH_INDEX=https://download.pytorch.org/whl/rocm6.2
echo  [OK] Will install PyTorch for ROCm (wheel: !CUDA_VERSION!)
echo  [NOTE] If the install fails, check https://pytorch.org for the latest ROCm wheel.
goto :gpu_done

:amd_no_rocm
:: AMD GPU found in device manager but ROCm tools are NOT installed
echo.
echo  ┌─────────────────────────────────────────────────────────────┐
echo  │  AMD GPU detected but ROCm toolkit is NOT installed.        │
echo  │                                                             │
echo  │  For GPU acceleration you need AMD ROCm for Windows.        │
echo  │  Download: https://rocm.docs.amd.com/en/latest/             │
echo  │                                                             │
echo  │  Supported AMD GPUs (ROCm on Windows):                      │
echo  │    RX 6000 series, RX 7000 series, Instinct MI series       │
echo  │                                                             │
echo  │  Falling back to !CUDA_VERSION! PyTorch for now.            │
echo  │  Re-run SETUP.bat after installing ROCm to get GPU support. │
echo  └─────────────────────────────────────────────────────────────┘
echo.
set GPU_BRAND=amd_no_rocm
set CUDA_VERSION=cpu
set TORCH_INDEX=https://download.pytorch.org/whl/cpu

:gpu_done

:: ── STEP 5: Install PyTorch ───────────────────────────────────────────────────
echo.
if "!CUDA_VERSION!"=="cpu" (
    echo  [5/7] Installing PyTorch ^(CPU-only^)...
) else (
    echo  [5/7] Installing PyTorch ^(!CUDA_VERSION!^)...
)
echo        This may take several minutes ^(torch is ~2-3 GB^)...
python -m pip install torch torchvision torchaudio --index-url !TORCH_INDEX! --quiet
if %errorlevel% neq 0 (
    echo  [ERROR] PyTorch installation failed.
    if "!GPU_BRAND!"=="amd_rocm" (
        echo  [HINT] ROCm wheels may not be available for your ROCm version.
        echo         Try: https://pytorch.org/get-started/locally/ to find the right wheel.
    )
    pause
    exit /b 1
)
echo  [OK] PyTorch installed ^(!CUDA_VERSION!^).

:: ── STEP 6: Install project requirements ─────────────────────────────────────
echo.
echo  [6/7] Installing project dependencies (requirements.txt)...
echo        This may take several minutes...

:: Install without the torch line (already installed above with correct CUDA wheel)
echo  Installing core packages...
python -m pip install ^
    "smolagents[transformers]>=1.8.0" ^
    "chromadb>=0.5.0" ^
    "sentence-transformers>=3.0.0" ^
    "FlagEmbedding>=1.2.0" ^
    "transformers>=4.51.0" ^
    "accelerate>=0.30.0" ^
    "bitsandbytes>=0.43.0" ^
    "PyMuPDF>=1.24.0" ^
    "datasets>=2.20.0" ^
    "gradio>=4.40.0" ^
    "Pillow>=10.0.0" ^
    --quiet

echo  Installing vision packages...
python -m pip install ^
    "qwen-vl-utils>=0.0.8" ^
    "byaldi>=0.0.6" ^
    "colpali-engine>=0.3.5" ^
    "pdf2image>=1.17.0" ^
    --quiet

:: pdf2image requires Poppler on Windows
echo.
echo  ┌──────────────────────────────────────────────────────────────┐
echo  │  IMPORTANT: pdf2image requires Poppler for PDF→image         │
echo  │  conversion (needed for visual PDF indexing).                │
echo  │                                                              │
echo  │  Install Poppler for Windows:                                │
echo  │  https://github.com/oschwartz10612/poppler-windows/releases  │
echo  │                                                              │
echo  │  Then add poppler/Library/bin to your PATH.                  │
echo  │  Without it, visual PDF indexing will be disabled.           │
echo  │  Text-based indexing and chat will still work fine.          │
echo  └──────────────────────────────────────────────────────────────┘
echo.

if %errorlevel% neq 0 (
    echo  [WARN] Some packages may have failed. Retrying individually...
    for %%p in (
        "smolagents[transformers]"
        chromadb
        sentence-transformers
        "transformers>=4.51.0"
        accelerate
        bitsandbytes
        PyMuPDF
        datasets
        gradio
    ) do (
        echo  Installing %%p...
        python -m pip install %%p --quiet
    )
)
echo  [OK] Dependencies installed.

:: ── STEP 7: Quick smoke test ──────────────────────────────────────────────────
echo.
echo  [7/7] Running smoke test...
python -c "import torch, chromadb, gradio, smolagents; cuda=torch.cuda.is_available(); dev=torch.cuda.get_device_name(0) if cuda else 'CPU only'; print('  torch:', torch.__version__, '| GPU available:', cuda, '|', dev); print('  All imports OK')"
if %errorlevel% neq 0 (
    echo  [WARN] Smoke test had issues — check output above.
) else (
    echo  [OK] Smoke test passed.
)

:: ── Done ──────────────────────────────────────────────────────────────────────
echo.
echo  ============================================================
echo   Setup complete!
echo  ============================================================
echo.
echo   Next steps:
echo     1. Launch the app:        double-click  RUN.bat
echo     2. Open the "Knowledge Base" tab to upload and index your documents.
echo.
echo   The app will open at:  http://localhost:7860
echo.
pause
exit /b 0

:: ── Helper: refresh PATH from registry after Python silent install ────────────
:RefreshPath
for /f "tokens=2*" %%a in (
    'reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v PATH 2^>nul'
) do set "SYS_PATH=%%b"
for /f "tokens=2*" %%a in (
    'reg query "HKCU\Environment" /v PATH 2^>nul'
) do set "USR_PATH=%%b"
set "PATH=!SYS_PATH!;!USR_PATH!"
goto :eof
