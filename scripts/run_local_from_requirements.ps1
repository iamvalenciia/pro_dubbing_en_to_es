param(
    [string]$PythonVersion = "3.11",
    [switch]$RunApp
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

function Resolve-GlobalPythonExe {
    param([string]$Version)

    if (-not [string]::IsNullOrWhiteSpace($Version)) {
        try {
            $candidate = (& py -$Version -c "import sys; print(sys.executable)" 2>$null).Trim()
            if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($candidate) -and (Test-Path $candidate)) {
                return $candidate
            }
        }
        catch {
        }
    }

    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -and (Test-Path $cmd.Source)) {
        return $cmd.Source
    }

    throw "No se encontró Python global. Instala Python y asegúrate de que esté en PATH."
}

$pythonExe = Resolve-GlobalPythonExe -Version $PythonVersion
Write-Host "[setup] Usando Python global: $pythonExe" -ForegroundColor Cyan

Write-Host "[setup] Upgrading pip/setuptools/wheel..." -ForegroundColor Cyan
& $pythonExe -m pip install --upgrade pip setuptools wheel

Write-Host "[setup] Installing requirements.txt..." -ForegroundColor Cyan
& $pythonExe -m pip install -r requirements.txt

Write-Host "[setup] Ensuring torch stack is installed (needed by pyvideotrans GPU checks)..." -ForegroundColor Cyan
$torchInstalled = (& $pythonExe -c "import importlib.util; print('1' if importlib.util.find_spec('torch') else '0')").Trim()
if ($torchInstalled -ne "1") {
    Write-Host "[setup] torch not found. Installing torch/torchvision/torchaudio (CUDA 12.8 wheels)..." -ForegroundColor Yellow
    & $pythonExe -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[setup] CUDA wheel install failed. Falling back to default PyPI wheels..." -ForegroundColor Yellow
        & $pythonExe -m pip install torch torchvision torchaudio
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install torch stack."
        }
    }
}

Write-Host "[check] Torch version:" -ForegroundColor Cyan
& $pythonExe -c "import torch; print(torch.__version__)"

Write-Host "[check] Running import smoke test..." -ForegroundColor Cyan
& $pythonExe -c "import modelscope, addict, requests, pydub; print('SMOKE_OK_LOCAL')"

Write-Host "[check] Active python:" -ForegroundColor Cyan
Write-Host $pythonExe

if ($RunApp) {
    Write-Host "[run] Starting app with global interpreter..." -ForegroundColor Green
    & $pythonExe main_ui.py
}
else {
    Write-Host "[done] Environment ready. Start app with: $pythonExe main_ui.py" -ForegroundColor Green
}
