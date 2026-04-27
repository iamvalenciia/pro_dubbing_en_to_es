# Diagnóstico de audio en el video final generado

$latestVideo = "C:\Users\juanf\OneDrive\Escritorio\qwen-en-to-es\output\full_video_en_test30s_dubbed.mp4"

if (-not (Test-Path $latestVideo)) {
    Write-Host "❌ Video final no encontrado: $latestVideo" -ForegroundColor Red
    exit 1
}

Write-Host "📹 Video encontrado: $latestVideo" -ForegroundColor Green
Write-Host ""

# 1. Probe con ffprobe para ver streams
Write-Host "[PASO 1] Inspeccionar streams de audio con ffprobe..." -ForegroundColor Cyan
ffprobe -v error -select_streams a -show_entries stream=codec_type,codec_name,sample_rate,channels,duration -of default=noprint_wrappers=1:nokey=1:type=flat $latestVideo

Write-Host ""
Write-Host "[PASO 2] Detalles completos del video..." -ForegroundColor Cyan
ffprobe -v error -show_format -show_streams $latestVideo | Select-String -Pattern "codec|duration|channels" -CaseSensitive

Write-Host ""
Write-Host "[PASO 3] Revisar carpeta de caché para archivos intermedios..." -ForegroundColor Cyan
$tempDirs = Get-ChildItem "C:\Users\juanf\OneDrive\Escritorio\qwen-en-to-es\pyvideotrans\tmp" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($tempDirs) {
    $latestTmp = $tempDirs[0].FullName
    Write-Host "Carpeta de caché más reciente: $latestTmp" -ForegroundColor Yellow
    Write-Host ""
    
    # Buscar archivos de audio
    $audioFiles = @(
        "vocal.wav"
        "instrument.wav"
        "target.wav"
        "es.m4a"
        "lastend.wav"
        "bgm_file_extend_volume.wav"
    )
    
    foreach ($fname in $audioFiles) {
        $fpath = Join-Path $latestTmp $fname
        if (Test-Path $fpath) {
            $size = (Get-Item $fpath).Length
            $duration = (&{ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $fpath} 2>$null)
            Write-Host "✓ $fname : $($size) bytes, duration: $duration s" -ForegroundColor Green
        }
    }
    
    # Buscar en subdirs también
    Write-Host ""
    Write-Host "Buscando archivos .wav en subcarpetas..." -ForegroundColor Yellow
    $wavs = Get-ChildItem $latestTmp -Filter "*.wav" -Recurse -ErrorAction SilentlyContinue
    foreach ($wav in $wavs) {
        $relPath = $wav.FullName -replace [regex]::Escape($latestTmp), ""
        Write-Host "  - $relPath" -ForegroundColor Cyan
    }
}

Write-Host ""
Write-Host "[PASO 4] Extraer audio del video final para inspeccionar..." -ForegroundColor Cyan
$extractedAudio = "C:\Users\juanf\OneDrive\Escritorio\qwen-en-to-es\diagnostic_extracted_audio.wav"
ffmpeg -y -hide_banner -loglevel warning -i $latestVideo -vn -acodec pcm_s16le -ar 16000 $extractedAudio
if (Test-Path $extractedAudio) {
    $audioSize = (Get-Item $extractedAudio).Length
    $audioDuration = (&{ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $extractedAudio} 2>$null)
    Write-Host "✓ Audio extraído: $audioSize bytes, duration: $audioDuration s" -ForegroundColor Green
    
    # Calcular amplitud RMS aproximada con ffmpeg
    Write-Host ""
    Write-Host "[PASO 5] Verificar si el audio tiene contenido (no está silencioso)..." -ForegroundColor Cyan
    $volumeStats = ffmpeg -hide_banner -loglevel warning -i $extractedAudio -af "volumedetect" -f null nul 2>&1 | Select-String "mean_volume|max_volume"
    if ($volumeStats) {
        Write-Host $volumeStats -ForegroundColor Yellow
    } else {
        Write-Host "⚠️  No se pudieron obtener estadísticas de volumen (podría ser silencio)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "✅ Diagnóstico completado." -ForegroundColor Green
