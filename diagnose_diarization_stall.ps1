# Diagnóstico para "Extracting embeddings" cuellos de botella
# Uso: .\diagnose_diarization_stall.ps1

Write-Host "`n=== DIARIZATION STALL DIAGNOSTIC ===" -ForegroundColor Cyan

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
