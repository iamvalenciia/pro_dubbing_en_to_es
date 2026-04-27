param(
    [string]$PythonVersion = "3.10",
    [switch]$RecreateVenv,
    [switch]$RunApp
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"

if ($RecreateVenv -and (Test-Path $venvPath)) {
    Write-Host "[setup] RecreateVenv enabled. Removing existing .venv..." -ForegroundColor Yellow

    # Kill any process whose executable lives inside .venv (covers python, pythonw, pylance, etc.)
    $lockingProcs = Get-Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -and $_.Path.StartsWith($venvPath, [System.StringComparison]::OrdinalIgnoreCase) }

    foreach ($proc in $lockingProcs) {
        Write-Host "[setup] Stopping locked process: PID=$($proc.Id) NAME=$($proc.ProcessName)" -ForegroundColor Yellow
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }

    # Give OS 1 second to release handles after killing processes
    Start-Sleep -Milliseconds 1000

    # Primary removal; fall back to cmd rd which bypasses some PowerShell handle locks
    try {
        Remove-Item -Recurse -Force $venvPath -ErrorAction Stop
    }
    catch {
        Write-Host "[setup] Remove-Item failed ($($_.Exception.Message)). Trying cmd rd..." -ForegroundColor Yellow
        cmd /c "rd /s /q `"$venvPath`""
        if (Test-Path $venvPath) {
            throw "Could not remove $venvPath. Close VS Code / any terminal using the venv Python, then retry."
        }
    }
    Write-Host "[setup] .venv removed." -ForegroundColor Green
}

if (-not (Test-Path $venvPath)) {
    Write-Host "[setup] Creating venv with Python $PythonVersion..." -ForegroundColor Cyan
    py -$PythonVersion -m venv .venv
}

if (-not (Test-Path $venvPython)) {
    throw "Could not find venv python at $venvPython"
}

Write-Host "[setup] Activating venv..." -ForegroundColor Cyan
. $activateScript

Write-Host "[setup] Upgrading pip/setuptools/wheel..." -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel

Write-Host "[setup] Installing requirements.txt..." -ForegroundColor Cyan
python -m pip install -r requirements.txt

Write-Host "[setup] Ensuring torch stack is installed (needed by pyvideotrans GPU checks)..." -ForegroundColor Cyan
$torchInstalled = (python -c "import importlib.util; print('1' if importlib.util.find_spec('torch') else '0')").Trim()
if ($torchInstalled -ne "1") {
    Write-Host "[setup] torch not found. Installing torch/torchvision/torchaudio (CUDA 12.8 wheels)..." -ForegroundColor Yellow
    python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[setup] CUDA wheel install failed. Falling back to default PyPI wheels..." -ForegroundColor Yellow
        python -m pip install torch torchvision torchaudio
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install torch stack."
        }
    }
}

Write-Host "[check] Torch version:" -ForegroundColor Cyan
python -c "import torch; print(torch.__version__)"

Write-Host "[check] Running import smoke test..." -ForegroundColor Cyan
python -c "import modelscope, addict, requests, pydub; print('SMOKE_OK_LOCAL')"

Write-Host "[check] Active python:" -ForegroundColor Cyan
where.exe python

if ($RunApp) {
    Write-Host "[run] Starting app with venv interpreter..." -ForegroundColor Green
    python main_ui.py
}
else {
    Write-Host "[done] Environment ready. Start app with: .\.venv\Scripts\python.exe main_ui.py" -ForegroundColor Green
}
