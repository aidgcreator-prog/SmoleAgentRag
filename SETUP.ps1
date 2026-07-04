$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
try { $Host.UI.RawUI.WindowTitle = "ជំនួយការ AI ពហុមុខងារ - ការដំឡើង" } catch {}

function Refresh-Path {
    $sys = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $usr = [System.Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$sys;$usr"
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " ការដំឡើងជំនួយការ AI ពហុមុខងារ ដោយ LocalAiLab - smolagents + ChromaDB + llama.cpp" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# ── STEP 0: Check we are in the right folder ─────────────────────
if (-not (Test-Path (Join-Path $root "app.py"))) {
    Write-Host "[កំហុស] រកមិនឃើញ app.py ។ សូមដំណើរការស្គ្រីបនេះពីក្នុងថតឫសនៃកម្មវិធី (ថតដែលមាន app.py) ។" -ForegroundColor Red
    Read-Host "ចុច Enter ដើម្បីបិទ"
    exit 1
}

# ── STEP 1: Check / Install Python (requires 3.9+) ───────────────
Write-Host "[1/8] កំពុងពិនិត្យមើលការដំឡើង Python..."
$needPython = $false
$pyVer = $null

$pyCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pyCmd) {
    Write-Host "[ព្រមាន] រកមិនឃើញ Python នៅក្នុង PATH ។" -ForegroundColor Yellow
    $needPython = $true
} else {
    $verOut = (& python --version) 2>&1
    $pyVer = ($verOut -replace "Python\s+", "").Trim()
    Write-Host " បានរកឃើញ Python $pyVer"
    $parts = $pyVer.Split(".")
    $major = [int]$parts[0]
    $minor = [int]$parts[1]

    if ($major -eq 2) {
        Write-Host "[ព្រមាន] មិនគាំទ្រ Python 2 ទេ (បានរកឃើញ $pyVer) ។" -ForegroundColor Yellow
        Write-Host "        Python 3.9 ឬខ្ពស់ជាងនេះ ត្រូវបានទាមទារ។"
        $needPython = $true
    } elseif ($major -eq 3 -and $minor -lt 9) {
        Write-Host "[ព្រមាន] Python $pyVer ចាស់ពេក។ ត្រូវការយ៉ាងតិច Python 3.9" -ForegroundColor Yellow
        Write-Host "        នឹងដំឡើង Python 3.11 ជាមួយកំណែបច្ចុប្បន្នរបស់អ្នក។"
        $needPython = $true
    } else {
        Write-Host "[OK] Python $pyVer បំពេញតម្រូវការអប្បបរមា (3.9+) ។" -ForegroundColor Green
    }
}

if ($needPython) {
    Write-Host ""
    Write-Host "កំពុងទាញយកកម្មវិធីដំឡើង Python 3.11.9..."
    $installerPath = Join-Path $env:TEMP "python_installer.exe"
    try {
        Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" -OutFile $installerPath -UseBasicParsing
    } catch {
        Write-Host ""
        Write-Host "[កំហុស] ការទាញយកបានបរាជ័យ។ សូមពិនិត្យការតភ្ជាប់អ៊ីនធឺណិតរបស់អ្នក ហើយសាកល្បងម្តងទៀត។" -ForegroundColor Red
        Write-Host "        ឬដំឡើង Python 3.11 ដោយផ្ទាល់ពី: https://www.python.org/downloads/"
        Read-Host "ចុច Enter ដើម្បីបិទ"
        exit 1
    }

    Write-Host " កំពុងដំឡើង Python 3.11.9 ដោយស្ងាត់ (អាចចំណាយពេល ១-២ នាទី)..."
    $installArgs = @("/quiet", "InstallAllUsers=1", "PrependPath=1", "Include_pip=1", "Include_launcher=1", "Include_test=0")
    $proc = Start-Process -FilePath $installerPath -ArgumentList $installArgs -Wait -PassThru
    if ($proc.ExitCode -ne 0) {
        Write-Host "[កំហុស] ការដំឡើង Python បានបរាជ័យ (exit code $($proc.ExitCode)) ។" -ForegroundColor Red
        Write-Host "        សូមដំឡើងដោយផ្ទាល់ពី: https://www.python.org/downloads/"
        Read-Host "ចុច Enter ដើម្បីបិទ"
        exit 1
    }

    Refresh-Path
    $pyCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pyCmd) {
        Write-Host ""
        Write-Host "[កំហុស] Python ត្រូវបានដំឡើងរួចហើយ ប៉ុន្តែនៅតែរកមិនឃើញនៅក្នុង PATH ។" -ForegroundColor Red
        Write-Host "        ករណីនេះកើតឡើងនៅពេលការផ្លាស់ប្តូរ PATH ត្រូវការសម័យបញ្ជាថ្មី។"
        Write-Host ""
        Write-Host "        សូម បិទ បង្អួចនេះ បើក PowerShell ថ្មី ហើយ"
        Write-Host "        ដំណើរការ SETUP.bat ម្តងទៀត។"
        Read-Host "ចុច Enter ដើម្បីបិទ"
        exit 1
    }
    $pyVer = ((& python --version) 2>&1) -replace "Python\s+", ""
    Write-Host "[OK] Python $pyVer បានដំឡើងដោយជោគជ័យ។" -ForegroundColor Green
}

# Final sanity — pip
& python -m pip --version *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ព្រមាន] រកមិនឃើញ pip - កំពុងព្យាយាមដំឡើងវា..." -ForegroundColor Yellow
    & python -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[កំហុស] មិនអាចដំឡើង pip បានទេ។ សូមដំណើរការ:  python -m ensurepip --upgrade" -ForegroundColor Red
        Read-Host "ចុច Enter ដើម្បីបិទ"
        exit 1
    }
}
Write-Host "[OK] pip អាចប្រើប្រាស់បាន។" -ForegroundColor Green

# ── STEP 2: Create virtual environment ────────────────────────────
Write-Host ""
Write-Host "[2/8] កំពុងបង្កើត virtual environment (.venv)..."
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    Write-Host "[OK] .venv មានរួចហើយ កំពុងរំលងការបង្កើត។" -ForegroundColor Green
} else {
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[កំហុស] បរាជ័យក្នុងការបង្កើត virtual environment ។" -ForegroundColor Red
        Read-Host "ចុច Enter ដើម្បីបិទ"
        exit 1
    }
    Write-Host "[OK] Virtual environment ត្រូវបានបង្កើត។" -ForegroundColor Green
}

$activateScript = Join-Path $root ".venv\Scripts\Activate.ps1"
try {
    & $activateScript
    Write-Host "[OK] Virtual environment ត្រូវបានធ្វើឱ្យសកម្ម។" -ForegroundColor Green
} catch {
    Write-Host "[ព្រមាន] មិនអាចធ្វើឱ្យ venv សកម្មបានទេ - នឹងប្រើ .venv\Scripts\python.exe ដោយផ្ទាល់។" -ForegroundColor Yellow
}

# ── STEP 3: Upgrade pip ────────────────────────────────────────────
Write-Host ""
Write-Host "[3/8] កំពុងធ្វើបច្ចុប្បន្នភាព pip..."
& python -m pip install --upgrade pip --quiet
Write-Host "[OK] pip ទាន់សម័យហើយ។" -ForegroundColor Green

# ── STEP 4: Detect GPU — NVIDIA / AMD ROCm / CPU ──────────────────
Write-Host ""
Write-Host "[4/8] កំពុងរកឃើញ GPU..."
$gpuBrand = "none"
$cudaVersion = "cpu"
$torchIndex = "https://download.pytorch.org/whl/cpu"

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$gpuDone = $false

if ($nvidiaSmi) {
    $smiOut = & nvidia-smi 2>$null
    if ($LASTEXITCODE -eq 0) {
        $gpuBrand = "nvidia"
        $cudaLine = $smiOut | Select-String "CUDA Version"
        $rawCuda = $null
        if ($cudaLine) {
            if ($cudaLine.Line -match "CUDA Version:\s*([0-9]+\.[0-9]+)") {
                $rawCuda = $matches[1]
            }
        }
        Write-Host "[OK] រកឃើញ GPU NVIDIA ។ កំណែ CUDA Driver: $rawCuda" -ForegroundColor Green

        if ($rawCuda) {
            $cmajor = $rawCuda.Split(".")[0]
            $cfull = $rawCuda
            if ($cmajor -eq "11") {
                $cudaVersion = "cu118"
            } elseif ($cfull -match "^12\.(1|2|3)") {
                $cudaVersion = "cu121"
            } elseif ($cfull -match "^12\.(4|5|6)") {
                $cudaVersion = "cu124"
            } elseif ($cfull -match "^12\.(7|8|9)") {
                $cudaVersion = "cu128"
            } else {
                Write-Host "[ព្រមាន] កំណែ CUDA មិនស្គាល់ '$rawCuda' - កំពុងប្រើលំនាំដើម cu128 ។" -ForegroundColor Yellow
                $cudaVersion = "cu128"
            }
        } else {
            $cudaVersion = "cu128"
        }
        $torchIndex = "https://download.pytorch.org/whl/$cudaVersion"
        Write-Host "[OK] នឹងដំឡើង PyTorch សម្រាប់ CUDA $cudaVersion" -ForegroundColor Green
        $gpuDone = $true
    }
}

if (-not $gpuDone) {
    $rocmSmi = Get-Command rocm-smi -ErrorAction SilentlyContinue
    $rocminfo = Get-Command rocminfo -ErrorAction SilentlyContinue
    if ($rocmSmi -or $rocminfo) {
        $gpuBrand = "amd_rocm"
        Write-Host "[OK] រកឃើញ GPU AMD ជាមួយ ROCm ។" -ForegroundColor Green
        $cudaVersion = "rocm6.2"
        $torchIndex = "https://download.pytorch.org/whl/rocm6.2"
        Write-Host "[OK] នឹងដំឡើង PyTorch សម្រាប់ ROCm (wheel: $cudaVersion)" -ForegroundColor Green
        Write-Host "[ចំណាំ] ប្រសិនបើការដំឡើងបរាជ័យ សូមពិនិត្យ https://pytorch.org សម្រាប់ wheel ROCm ចុងក្រោយ។"
        $gpuDone = $true
    } else {
        $isAmd = $false
        try {
            $vc = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
            if ($vc -and ($vc.Name -join ";") -match "Radeon|AMD") { $isAmd = $true }
        } catch {}
        if ($isAmd) {
            $gpuBrand = "amd_no_rocm"
            Write-Host ""
            Write-Host "+-----------------------------------------------------------------+" -ForegroundColor Yellow
            Write-Host "|  រកឃើញ GPU AMD ប៉ុន្តែមិនទាន់ដំឡើង ROCm toolkit ទេ។              |" -ForegroundColor Yellow
            Write-Host "|                                                                   |" -ForegroundColor Yellow
            Write-Host "|  ដើម្បីប្រើ GPU អ្នកត្រូវការ AMD ROCm សម្រាប់ Windows ។             |" -ForegroundColor Yellow
            Write-Host "|  ទាញយក: https://rocm.docs.amd.com/en/latest/                    |" -ForegroundColor Yellow
            Write-Host "|                                                                   |" -ForegroundColor Yellow
            Write-Host "|  GPU AMD ដែលគាំទ្រ (ROCm លើ Windows):                             |" -ForegroundColor Yellow
            Write-Host "|    RX 6000 series, RX 7000 series, Instinct MI series            |" -ForegroundColor Yellow
            Write-Host "|                                                                   |" -ForegroundColor Yellow
            Write-Host "|  កំពុងប្រើ CPU PyTorch ជាបណ្តោះអាសន្ន។                            |" -ForegroundColor Yellow
            Write-Host "|  ដំណើរការ SETUP.bat ម្តងទៀត បន្ទាប់ពីដំឡើង ROCm ។                 |" -ForegroundColor Yellow
            Write-Host "+-----------------------------------------------------------------+" -ForegroundColor Yellow
            Write-Host ""
            $cudaVersion = "cpu"
            $torchIndex = "https://download.pytorch.org/whl/cpu"
        } else {
            Write-Host "[ព្រមាន] រកមិនឃើញ GPU ទេ (គ្មាន nvidia-smi, rocm-smi ឬ GPU AMD ក្នុងបញ្ជីឧបករណ៍) ។" -ForegroundColor Yellow
            Write-Host "        កំពុងដំឡើង PyTorch សម្រាប់ CPU ។"
            $gpuBrand = "cpu"
            $cudaVersion = "cpu"
            $torchIndex = "https://download.pytorch.org/whl/cpu"
        }
    }
}

# ── STEP 5: Install PyTorch ─────────────────────────────────────────
Write-Host ""
if ($cudaVersion -eq "cpu") {
    Write-Host "[5/8] កំពុងដំឡើង PyTorch (CPU-only)..."
} else {
    Write-Host "[5/8] កំពុងដំឡើង PyTorch ($cudaVersion)..."
}
Write-Host "      អាចចំណាយពេលច្រើននាទី (torch មានទំហំប្រហែល ២-៣ GB)..."
& python -m pip install torch torchvision torchaudio --index-url $torchIndex --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "[កំហុស] ការដំឡើង PyTorch បានបរាជ័យ។" -ForegroundColor Red
    if ($gpuBrand -eq "amd_rocm") {
        Write-Host "[គន្លឹះ] wheel ROCm ប្រហែលជាមិនមានសម្រាប់កំណែ ROCm របស់អ្នកទេ។" -ForegroundColor Yellow
        Write-Host "        សាកល្បង: https://pytorch.org/get-started/locally/ ដើម្បីរក wheel ត្រឹមត្រូវ។"
    }
    Read-Host "ចុច Enter ដើម្បីបិទ"
    exit 1
}
Write-Host "[OK] PyTorch ត្រូវបានដំឡើង ($cudaVersion) ។" -ForegroundColor Green

# ── STEP 6: Install llama-cpp-python (GGUF backend), hardware-aware ──
Write-Host ""
Write-Host "[6/8] កំពុងដំឡើង llama-cpp-python (សម្រាប់ម៉ូដែល GGUF)..."

# Must match the GGUF model folder used by app.py — used to find a real
# .gguf file for the thorough smoke test below, if the user has one.
# Not hardcoded: set the LLAMA_CPP_MODEL_DIR environment variable before
# running this script to point it at your own folder (e.g.
#   $env:LLAMA_CPP_MODEL_DIR = "D:\models\gguf"
# ). If it's not set, this smoke test is simply skipped — GGUF models can
# still be pointed at any folder later from the app's own UI.
$LLAMA_CPP_MODEL_DIR = $env:LLAMA_CPP_MODEL_DIR


# Candidate prebuilt-wheel CUDA tiers (abetlen.github.io index), tried
# newest-compatible-first based on the driver's reported CUDA version.
# NVIDIA drivers are backward compatible, so a slightly-older wheel tier
# than the driver reports is expected to work fine.
function Get-LlamaCppCudaTiers([string]$driverCudaVersion) {
    if (-not $driverCudaVersion) { return @() }
    $parts = $driverCudaVersion.Split(".")
    $major = [int]$parts[0]
    $minor = if ($parts.Length -gt 1) { [int]$parts[1] } else { 0 }
    if ($major -ge 13) { return @("cu132", "cu130", "cu125", "cu124", "cu121") }
    if ($major -eq 12) {
        if ($minor -ge 5) { return @("cu125", "cu124", "cu123", "cu122", "cu121") }
        if ($minor -eq 4) { return @("cu124", "cu123", "cu122", "cu121") }
        if ($minor -eq 3) { return @("cu123", "cu122", "cu121") }
        if ($minor -eq 2) { return @("cu122", "cu121") }
        return @("cu121")
    }
    if ($major -eq 11) { return @("cu118") }
    return @()
}

# Smoke-test, tier 1 (fast): does the installed build actually import AND
# initialize the backend without crashing? Catches the "wheel assumes
# AVX512 but this CPU doesn't have it" illegal-instruction crash — a plain
# successful `pip install` does NOT guarantee the wheel will even run.
function Test-LlamaCppBackendInit {
    & python -c "import llama_cpp; llama_cpp.llama_backend_init(); print('OK')" *> $null
    return ($LASTEXITCODE -eq 0)
}

# Smoke-test, tier 2 (thorough): actually load one of the user's real .gguf
# files. This is the test that matters most — a wheel can pass the generic
# backend-init check above while still shipping an OLDER llama.cpp version
# that doesn't understand a newer/uncommon model architecture's GGUF
# metadata (this happened with a Gemma MoE checkpoint). If no .gguf files
# are present yet (fresh install, user hasn't dropped models in), this test
# is skipped and only the generic check applies.
function Test-LlamaCppRealModelLoad([int]$TimeoutSec = 240) {
    if ([string]::IsNullOrWhiteSpace($LLAMA_CPP_MODEL_DIR)) { return $true }
    if (-not (Test-Path $LLAMA_CPP_MODEL_DIR)) { return $true }
    $sample = Get-ChildItem -Path $LLAMA_CPP_MODEL_DIR -Filter "*.gguf" -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $sample) { return $true }

    Write-Host "   កំពុងផ្ទៀងផ្ទាត់ដោយផ្ទុកម៉ូដែលពិតប្រាកដ: $($sample.Name) (អាចចំណាយពេលរហូតដល់ $TimeoutSec វិនាទី)..."
    $code = "import llama_cpp; m = llama_cpp.Llama(model_path=r'$($sample.FullName)', n_gpu_layers=-1, n_ctx=512, verbose=False); print('OK')"

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName               = "python"
    $psi.Arguments              = "-c `"$code`""
    $psi.UseShellExecute        = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    try {
        $proc = [System.Diagnostics.Process]::Start($psi)
    } catch {
        return $false
    }
    $finished = $proc.WaitForExit($TimeoutSec * 1000)
    if (-not $finished) {
        try { $proc.Kill() } catch {}
        Write-Host "   [ព្រមាន] ការផ្ទុកគំរូបានលើសពេលកំណត់ ($TimeoutSec វិនាទី) ។" -ForegroundColor Yellow
        return $false
    }
    return ($proc.ExitCode -eq 0)
}

function Test-LlamaCppWheel {
    if (-not (Test-LlamaCppBackendInit)) { return $false }
    return (Test-LlamaCppRealModelLoad)
}

# Pin every tier to the SAME (latest available) llama-cpp-python release so
# we don't silently end up on an older version for one CUDA tier just
# because it happened to install successfully. If we can't determine a
# latest version (offline, pip index unsupported, etc.) we fall back to
# each tier's own latest, same as before.
$llamaCppLatestVersion = $null
try {
    $verOut = & python -m pip index versions llama-cpp-python 2>$null
    $verLine = $verOut | Select-String "Available versions:"
    if ($verLine -and ($verLine.Line -match "Available versions:\s*([0-9][0-9A-Za-z\.\-]*)")) {
        $llamaCppLatestVersion = $matches[1]
        Write-Host " កំណែ llama-cpp-python ថ្មីបំផុតដែលរកឃើញ: $llamaCppLatestVersion"
    }
} catch {}

$llamaCppInstalled = $false
$llamaCppMode       = "none"

if ($gpuBrand -eq "nvidia") {
    $tiers = Get-LlamaCppCudaTiers $rawCuda
    foreach ($tier in $tiers) {
        $installedThisTier = $false
        if ($llamaCppLatestVersion) {
            Write-Host " សាកល្បង prebuilt wheel: $tier (កំណែ $llamaCppLatestVersion)..."
            & python -m pip install "llama-cpp-python==$llamaCppLatestVersion" --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/$tier" --force-reinstall --no-cache-dir --quiet 2>$null
            if ($LASTEXITCODE -eq 0) { $installedThisTier = $true }
        }
        if (-not $installedThisTier) {
            Write-Host " សាកល្បង prebuilt wheel: $tier (កំណែថ្មីបំផុតដែលមាន)..."
            & python -m pip install llama-cpp-python --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/$tier" --force-reinstall --no-cache-dir --quiet 2>$null
            if ($LASTEXITCODE -eq 0) { $installedThisTier = $true }
        }

        if ($installedThisTier -and (Test-LlamaCppWheel)) {
            Write-Host "[OK] llama-cpp-python ($tier, CUDA) ដំឡើងជោគជ័យ និងឆ្លងកាត់ការសាកល្បង (រួមទាំងផ្ទុកគំរូម៉ូដែលពិតប្រាកដ)។" -ForegroundColor Green
            $llamaCppInstalled = $true
            $llamaCppMode       = "cuda-$tier"
            break
        } else {
            Write-Host "[ព្រមាន] $tier wheel មិនដំណើរការត្រឹមត្រូវលើម៉ាស៊ីននេះទេ (ការដំឡើងបរាជ័យ ការសាកល្បងបរាជ័យ ឬស្ថាបត្យកម្មម៉ូដែលមិនត្រូវគ្នា) - សាកល្បង tier បន្ទាប់..." -ForegroundColor Yellow
        }
    }

    if (-not $llamaCppInstalled) {
        # No prebuilt wheel worked — try compiling from source, which
        # auto-detects this exact CPU's instruction set instead of guessing.
        $vsWhere  = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
        $hasVS    = Test-Path $vsWhere
        $hasNvcc  = $null -ne (Get-Command nvcc -ErrorAction SilentlyContinue)
        if ($hasVS -and $hasNvcc) {
            Write-Host " prebuilt wheels មិនដំណើរការទេ - កំពុងសាងសង់ពី source ជាមួយ CUDA (អាចចំណាយពេលច្រើននាទី)..." -ForegroundColor Yellow
            $env:CMAKE_ARGS  = "-DGGML_CUDA=on"
            $env:FORCE_CMAKE = "1"
            & python -m pip install llama-cpp-python --no-cache-dir --force-reinstall --quiet
            $buildExit = $LASTEXITCODE
            Remove-Item Env:\CMAKE_ARGS -ErrorAction SilentlyContinue
            Remove-Item Env:\FORCE_CMAKE -ErrorAction SilentlyContinue
            if ($buildExit -eq 0 -and (Test-LlamaCppWheel)) {
                Write-Host "[OK] llama-cpp-python ត្រូវបានសាងសង់ដោយជោគជ័យជាមួយ CUDA (compiled for this CPU) ។" -ForegroundColor Green
                $llamaCppInstalled = $true
                $llamaCppMode       = "cuda-source"
            } else {
                Write-Host "[ព្រមាន] ការសាងសង់ពី source បានបរាជ័យ។" -ForegroundColor Yellow
            }
        } else {
            Write-Host "[ព្រមាន] រកមិនឃើញ Visual Studio Build Tools ឬ CUDA Toolkit (nvcc) - មិនអាចសាងសង់ពី source បានទេ។" -ForegroundColor Yellow
            Write-Host "         ដើម្បីបើកមុខងារ GPU: ដំឡើង Visual Studio Build Tools 2022 (Desktop development with C++)"
            Write-Host "         និង CUDA Toolkit ដែលត្រូវនឹងកំណែ driver របស់អ្នក រួចដំណើរការ SETUP.bat ម្តងទៀត។"
        }
    }
} elseif ($gpuBrand -eq "amd_rocm") {
    Write-Host " ចំណាំ: llama-cpp-python មិនមាន prebuilt wheel សម្រាប់ ROCm ទេ ហើយការគាំទ្រ HIP លើ Windows នៅមានកម្រិត។" -ForegroundColor Yellow
    Write-Host "        កំពុងសាកល្បងសាងសង់ពី source ជាមួយ HIP (អាចនឹងបរាជ័យ អាស្រ័យលើកំណែ ROCm របស់អ្នក)..."
    $env:CMAKE_ARGS = "-DGGML_HIP=on"
    & python -m pip install llama-cpp-python --no-cache-dir --force-reinstall --quiet
    $buildExit = $LASTEXITCODE
    Remove-Item Env:\CMAKE_ARGS -ErrorAction SilentlyContinue
    if ($buildExit -eq 0 -and (Test-LlamaCppWheel)) {
        Write-Host "[OK] llama-cpp-python ត្រូវបានសាងសង់ដោយជោគជ័យជាមួយ HIP/ROCm ។" -ForegroundColor Green
        $llamaCppInstalled = $true
        $llamaCppMode       = "rocm-source"
    } else {
        Write-Host "[ព្រមាន] ការសាងសង់ HIP បានបរាជ័យ - នឹងប្រើ CPU ជំនួសវិញ។" -ForegroundColor Yellow
    }
}

if (-not $llamaCppInstalled) {
    # No supported GPU, or every GPU path above failed — CPU-only build.
    # This still fully supports GGUF models via the general chat / RAG /
    # data-analysis tabs; it's just slower than GPU offload. We only run
    # the fast generic check here (not the real-model load test) since a
    # large model loading purely on CPU can legitimately take several
    # minutes without being broken, and there's no further fallback to try.
    Write-Host " កំពុងដំឡើង llama-cpp-python (CPU-only)..."
    & python -m pip install llama-cpp-python --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/cpu" --force-reinstall --no-cache-dir --quiet
    if ($LASTEXITCODE -eq 0 -and (Test-LlamaCppBackendInit)) {
        Write-Host "[OK] llama-cpp-python (CPU-only) ត្រូវបានដំឡើង។ ម៉ូដែល GGUF នឹងដំណើរការនៅលើ CPU (យឺតជាង GPU)។" -ForegroundColor Yellow
        $llamaCppInstalled = $true
        $llamaCppMode       = "cpu"
    } else {
        Write-Host "[ព្រមាន] ការដំឡើង llama-cpp-python បានបរាជ័យទាំងស្រុង។ ម៉ូដែល GGUF (.gguf) នឹងមិនអាចប្រើបានទេ" -ForegroundColor Red
        Write-Host "         (ម៉ូដែល HuggingFace/transformers នៅតែដំណើរការធម្មតា)។"
    }
}

# ── STEP 7: Install project requirements ────────────────────────────
Write-Host ""
Write-Host "[7/8] កំពុងដំឡើង dependencies របស់គម្រោង (requirements.txt)..."
Write-Host "      អាចចំណាយពេលច្រើននាទី..."

Write-Host " កំពុងដំឡើង packages ស្នូល..."
& python -m pip install `
    "smolagents[transformers]>=1.8.0" `
    "chromadb>=0.5.0" `
    "sentence-transformers>=3.0.0" `
    "FlagEmbedding>=1.2.0" `
    "transformers>=4.51.0" `
    "accelerate>=0.30.0" `
    "bitsandbytes>=0.43.0" `
    "PyMuPDF>=1.24.0" `
    "datasets>=2.20.0" `
    "gradio>=4.40.0" `
    "Pillow>=10.0.0" `
    --quiet
$coreExit = $LASTEXITCODE

Write-Host " កំពុងដំឡើង packages ចក្ខុវិស័យ (vision)..."
& python -m pip install `
    "qwen-vl-utils>=0.0.8" `
    "byaldi>=0.0.6" `
    "colpali-engine>=0.3.5" `
    "pdf2image>=1.17.0" `
    --quiet

Write-Host ""
Write-Host "+------------------------------------------------------------------+" -ForegroundColor Cyan
Write-Host "|  សំខាន់: pdf2image ត្រូវការ Poppler សម្រាប់បម្លែង PDF->រូបភាព     |" -ForegroundColor Cyan
Write-Host "|  (ត្រូវការសម្រាប់ការបញ្ចូល PDF បែបចក្ខុវិស័យ) ។                  |" -ForegroundColor Cyan
Write-Host "|                                                                    |" -ForegroundColor Cyan
Write-Host "|  ដំឡើង Poppler សម្រាប់ Windows:                                    |" -ForegroundColor Cyan
Write-Host "|  https://github.com/oschwartz10612/poppler-windows/releases      |" -ForegroundColor Cyan
Write-Host "|                                                                    |" -ForegroundColor Cyan
Write-Host "|  បន្ទាប់មកបន្ថែម poppler/Library/bin ទៅក្នុង PATH របស់អ្នក។       |" -ForegroundColor Cyan
Write-Host "|  បើគ្មានវាទេ ការបញ្ចូល PDF បែបចក្ខុវិស័យនឹងត្រូវបានបិទ។           |" -ForegroundColor Cyan
Write-Host "|  ការបញ្ចូលអត្ថបទ និងការជជែកនៅតែដំណើរការធម្មតា។                   |" -ForegroundColor Cyan
Write-Host "+------------------------------------------------------------------+" -ForegroundColor Cyan
Write-Host ""

if ($coreExit -ne 0) {
    Write-Host "[ព្រមាន] Package មួយចំនួនអាចនឹងបានបរាជ័យ។ កំពុងព្យាយាមម្តងទៀតម្តងមួយៗ..." -ForegroundColor Yellow
    $pkgs = @("smolagents[transformers]", "chromadb", "sentence-transformers", "transformers>=4.51.0", "accelerate", "bitsandbytes", "PyMuPDF", "datasets", "gradio")
    foreach ($p in $pkgs) {
        Write-Host " កំពុងដំឡើង $p..."
        & python -m pip install $p --quiet
    }
}
Write-Host "[OK] Dependencies ត្រូវបានដំឡើង។" -ForegroundColor Green

# ── STEP 8: Quick smoke test ─────────────────────────────────────────
Write-Host ""
Write-Host "[8/8] កំពុងសាកល្បងប្រព័ន្ធ (smoke test)..."
& python -c "import torch, chromadb, gradio, smolagents; cuda=torch.cuda.is_available(); dev=torch.cuda.get_device_name(0) if cuda else 'CPU only'; print('  torch:', torch.__version__, '| GPU available:', cuda, '|', dev); print('  All imports OK')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ព្រមាន] ការសាកល្បងមានបញ្ហា - សូមពិនិត្យលទ្ធផលខាងលើ។" -ForegroundColor Yellow
} else {
    Write-Host "[OK] ការសាកល្បងបានជោគជ័យ។" -ForegroundColor Green
}

if ($llamaCppInstalled) {
    Write-Host "[OK] llama-cpp-python (GGUF): ដំណើរការក្នុងម៉ូដ '$llamaCppMode' ។" -ForegroundColor Green
} else {
    Write-Host "[ព្រមាន] llama-cpp-python (GGUF): មិនអាចដំឡើងបានទេ - ម៉ូដែល .gguf នឹងមិនអាចប្រើបានទេ។" -ForegroundColor Yellow
}

# ── Done ───────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " ការដំឡើងបានបញ្ចប់!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host " ជំហានបន្ទាប់:"
Write-Host "   ១. ចាប់ផ្តើមកម្មវិធី:        ចុចពីរដងលើ  RUN.bat"
Write-Host "   ២. បើកផ្ទាំង `"Knowledge Base`" ដើម្បីបង្ហោះ និងបញ្ចូលឯកសាររបស់អ្នក។"
Write-Host ""
Write-Host " កម្មវិធីនឹងបើកនៅ:  http://localhost:7861"
Write-Host ""
Read-Host "ចុច Enter ដើម្បីបិទ"
