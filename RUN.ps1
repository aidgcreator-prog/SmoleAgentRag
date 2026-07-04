$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
try { $Host.UI.RawUI.WindowTitle = "ជំនួយការ AI ពហុមុខងារ - កំពុងដំណើរការ" } catch {}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " ជំនួយការ AI ពហុមុខងារ ដោយ LocalAiLab - smolagents + ChromaDB + llama.cpp" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# ── Check we are in the right folder ─────────────────────────────
if (-not (Test-Path (Join-Path $root "app.py"))) {
    Write-Host "[កំហុស] រកមិនឃើញ app.py ។ សូមដំណើរការស្គ្រីបនេះពីក្នុងថតឫសនៃកម្មវិធី (ថតដែលមាន app.py) ។" -ForegroundColor Red
    Read-Host "ចុច Enter ដើម្បីបិទ"
    exit 1
}

# ── Check setup was run ──────────────────────────────────────────
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "[កំហុស] រកមិនឃើញ Virtual environment ។" -ForegroundColor Red
    Write-Host ""
    Write-Host "សូមដំណើរការ SETUP.bat (ឬ SETUP.ps1) ជាមុនសិន!" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "ចុច Enter ដើម្បីបិទ"
    exit 1
}

# ── Activate venv ─────────────────────────────────────────────────
$activateScript = Join-Path $root ".venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    try {
        & $activateScript
    } catch {
        Write-Host "[ព្រមាន] មិនអាចធ្វើឱ្យ venv សកម្មបានទេ (Execution Policy?) ។" -ForegroundColor Yellow
        Write-Host "         កំពុងបន្តដោយប្រើ python ក្នុង .venv ដោយផ្ទាល់។"
    }
} else {
    Write-Host "[ព្រមាន] រកមិនឃើញ .venv\Scripts\Activate.ps1 - venv ហាក់ដូចជាមិនពេញលេញ។" -ForegroundColor Yellow
    Write-Host "         កម្មវិធីនឹងដំណើរការដោយប្រើ Python ប្រព័ន្ធរបស់អ្នក ជំនួសឱ្យ venv ។"
    Write-Host "         ដំណោះស្រាយ: លុបថត .venv ហើយដំណើរការ SETUP.bat ម្តងទៀត"
    Write-Host ""
}

# ── Show GPU info ─────────────────────────────────────────────────
Write-Host "ស្ថានភាពផ្នែករឹង:"
$nvidiaOk = $false
try {
    $gpuInfo = & nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $gpuInfo) {
        Write-Host " $gpuInfo"
        $nvidiaOk = $true
    }
} catch {}

if ($nvidiaOk) {
    Write-Host " (រកឃើញ GPU NVIDIA)"
} else {
    $isAmd = $false
    try {
        $vc = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
        if ($vc -and ($vc.Name -join ";") -match "Radeon|AMD") { $isAmd = $true }
    } catch {}
    if ($isAmd) {
        Write-Host " (រកឃើញ GPU AMD - ប្រហែលជាត្រូវការ ROCm)"
    } else {
        Write-Host " (មិនរកឃើញ GPU NVIDIA/AMD ទេ - កំពុងដំណើរការលើ CPU)"
    }
}

# ── Launch app ────────────────────────────────────────────────────
Write-Host ""
Write-Host "កំពុងចាប់ផ្តើមជំនួយការ AI ពហុមុខងារ..."
Write-Host "បើកកម្មវិធីរុករករបស់អ្នកនៅ:  http://localhost:7861"
Write-Host ""
Write-Host "ចុច Ctrl+C ដើម្បីបញ្ឈប់កម្មវិធី។"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

python app.py
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    Write-Host ""
    Write-Host "[កំហុស] កម្មវិធីបានបិទដោយមានកំហុស (លេខកូដ $exitCode) ។" -ForegroundColor Red
    Write-Host "សូមពិនិត្យមើលលទ្ធផលខាងលើសម្រាប់ព័ត៌មានលម្អិត។"
    Write-Host ""
    Write-Host "ដំណោះស្រាយទូទៅ:"
    Write-Host "  - VRAM មិនគ្រប់គ្រាន់: ប្តូរទៅម៉ូដែលតូចជាងនៅក្នុងបញ្ជីទម្លាក់លើ UI ។"
    Write-Host "  - Packages បាត់: ដំណើរការ SETUP.bat ម្តងទៀត"
    Write-Host "  - Port 7861 កំពុងប្រើ: បិទកម្មវិធី Gradio ផ្សេងទៀត ហើយសាកល្បងម្តងទៀត។"
    Read-Host "ចុច Enter ដើម្បីបិទ"
}
