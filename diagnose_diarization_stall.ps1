# Diagnóstico para "Extracting embeddings" cuellos de botella
# Uso: .\diagnose_diarization_stall.ps1

Write-Host "`n=== DIARIZATION STALL DIAGNOSTIC ===" -ForegroundColor Cyan

# 0. Validaciones hard-gate por síntoma (árbol QDP)
Write-Host "`n[0] Hard-gate checks (ORT + imports críticos):" -ForegroundColor Green

$runtimePy = $env:PYVIDEOTRANS_PYTHON
if (-not $runtimePy -or -not (Test-Path $runtimePy)) {
    $py310 = Join-Path $env:LOCALAPPDATA "Programs\Python\Python310\python.exe"
    if (Test-Path $py310) {
        $runtimePy = $py310
    } else {
        $runtimePy = "python"
    }
}
Write-Host "  Runtime Python: $runtimePy"

$ortCudaOk = $false
$ortProviders = @()
$criticalDeps = @{
    hdbscan = $false
    umap = $false
    datasets = $false
    simplejson = $false
    sortedcontainers = $false
    addict = $false
}

try {
    $ortJson = & $runtimePy -c "import json, onnxruntime as ort; p=ort.get_available_providers(); print(json.dumps({'providers':p,'cuda_ok':('CUDAExecutionProvider' in p)}))" 2>$null
    $ortData = $ortJson | ConvertFrom-Json
    $ortProviders = @($ortData.providers)
    $ortCudaOk = [bool]$ortData.cuda_ok
} catch {
    Write-Host "  [FAIL] ORT check failed: $($_.Exception.Message)" -ForegroundColor Red
}

try {
    $depsJson = & $runtimePy -c "import json, importlib.util as u; mods=['hdbscan','umap','datasets','simplejson','sortedcontainers','addict']; print(json.dumps({m: bool(u.find_spec(m)) for m in mods}))" 2>$null
    $depsData = $depsJson | ConvertFrom-Json
    foreach ($k in @($criticalDeps.Keys)) {
        $criticalDeps[$k] = [bool]$depsData.$k
    }
} catch {
    Write-Host "  [FAIL] Dependency import check failed: $($_.Exception.Message)" -ForegroundColor Red
}

$depsMissing = @($criticalDeps.GetEnumerator() | Where-Object { -not $_.Value } | ForEach-Object { $_.Key })
$depsOk = ($depsMissing.Count -eq 0)

Write-Host "  ORT providers: $($ortProviders -join ', ')"
if ($ortCudaOk) {
    Write-Host "  [PASS] CUDAExecutionProvider presente" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] CUDAExecutionProvider ausente" -ForegroundColor Red
}

if ($depsOk) {
    Write-Host "  [PASS] Imports críticos OK: hdbscan, umap, datasets, simplejson, sortedcontainers, addict" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] Imports críticos faltantes: $($depsMissing -join ', ')" -ForegroundColor Red
}

$symptomPhase1Pass = $ortCudaOk -and $depsOk
if ($symptomPhase1Pass) {
    Write-Host "  [PASS][SÍNTOMA: Fase 1 atascada en diarización] Hard-gates superados" -ForegroundColor Green
} else {
    Write-Host "  [FAIL][SÍNTOMA: Fase 1 atascada en diarización] Revisar ORT CUDA/dependencias" -ForegroundColor Red
}

$symptomOfflinePass = $criticalDeps["simplejson"]
if ($symptomOfflinePass) {
    Write-Host "  [PASS][SÍNTOMA: Startup offline simplejson] simplejson importable" -ForegroundColor Green
} else {
    Write-Host "  [FAIL][SÍNTOMA: Startup offline simplejson] simplejson no disponible" -ForegroundColor Red
}

# 1. Buscar procesos Python con pyvideotrans
Write-Host "`n[1] Procesos Python activos:" -ForegroundColor Green
$py_procs = Get-Process python* -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -match "python" }
if ($py_procs) {
    foreach ($proc in $py_procs) {
        $mem_mb = [math]::Round($proc.WorkingSet64 / 1MB, 2)
        Write-Host "  PID: $($proc.Id), Memory: ${mem_mb}MB, Name: $($proc.ProcessName)"
        
        # Si está usando >4GB, probablemente está en OOM
        if ($mem_mb -gt 4000) {
            Write-Host "    ⚠️  HIGH MEMORY USAGE - Possible OOM situation" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "  No Python processes found"
}

# 2. GPU status
Write-Host "`n[2] GPU Status:" -ForegroundColor Green
try {
    $nvidia_output = & nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader,nounits
    if ($nvidia_output) {
        Write-Host "  $nvidia_output"
        Write-Host "  (Format: used_mem/total_mem (MB), GPU_util%, Memory_util%)"
    }
} catch {
    Write-Host "  nvidia-smi not available"
}

# 3. Recent log entries
Write-Host "`n[3] Recent phase1 log entries (last 20):" -ForegroundColor Green
$log_file = "$PSScriptRoot\temp_workspace\qdp_data\logs\pipeline.log"
if (Test-Path $log_file) {
    Get-Content $log_file -Tail 20 | ForEach-Object {
        if ($_ -match "DIAR|Extracting|embedding") {
            Write-Host "  $_" -ForegroundColor Yellow
        } else {
            Write-Host "  $_"
        }
    }
} else {
    Write-Host "  Log file not found at $log_file"
}

# 4. What to do
Write-Host "`n[4] DIAGNOSTIC OPTIONS:" -ForegroundColor Magenta
Write-Host "  A) Press Ctrl+C to stop the current run (if stuck)"
Write-Host "  B) Wait 5 more minutes - embedding extraction can be slow on large files"
Write-Host "  C) Check GPU memory with: nvidia-smi -l 1  (every 1 second)"
Write-Host "  D) Kill process: Stop-Process -Id <PID>"

Write-Host "`n[5] OPTIMIZATION PLAN:" -ForegroundColor Cyan
Write-Host "  • If GPU memory is full (>90%) → batch size is too large"
Write-Host "  • If GPU util is <20% → CPU-bound (HDBSCAN clustering issue)"
Write-Host "  • If GPU util is >80% → Normal, embedding extraction is working"
Write-Host "  • After 60 min total → abort and use batch_size=8 (PYVIDEOTRANS_DIARIZATION_BATCH_SIZE=8)"

Write-Host "`n"
