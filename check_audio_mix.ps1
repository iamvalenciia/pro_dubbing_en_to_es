param()

$latestVideo = "C:\Users\juanf\OneDrive\Escritorio\qwen-en-to-es\output\full_video_en_test30s_dubbed.mp4"

if (-not (Test-Path $latestVideo)) {
    Write-Host "[ERROR] Video final no encontrado: $latestVideo" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Video encontrado: $latestVideo" -ForegroundColor Green
Write-Host ""

Write-Host "[PASO 1] Inspeccionar streams de audio con ffprobe..." -ForegroundColor Cyan
ffprobe -v error -select_streams a -show_entries stream=codec_type,codec_name,sample_rate,channels,duration -of default=noprint_wrappers=1:nokey=1:type=flat "$latestVideo"

Write-Host ""
Write-Host "[PASO 2] Extraer audio del video final..." -ForegroundColor Cyan
$extractedAudio = "C:\Users\juanf\OneDrive\Escritorio\qwen-en-to-es\diagnostic_extracted_audio.wav"
ffmpeg -y -hide_banner -loglevel warning -i "$latestVideo" -vn -acodec pcm_s16le -ar 16000 "$extractedAudio"

if (Test-Path $extractedAudio) {
    Write-Host "[OK] Audio extraido exitosamente" -ForegroundColor Green
    $audioSize = (Get-Item $extractedAudio).Length
    Write-Host "    Tamaño: $audioSize bytes" -ForegroundColor Yellow
    
    Write-Host ""
    Write-Host "[PASO 3] Detectar volumen de audio..." -ForegroundColor Cyan
    ffmpeg -hide_banner -loglevel warning -i "$extractedAudio" -af "volumedetect" -f null nul 2>&1 | Select-String "mean_volume|max_volume"
} else {
    Write-Host "[ERROR] No se pudo extraer audio" -ForegroundColor Red
}

Write-Host ""
Write-Host "[PASO 4] Revisar ultima carpeta de caché..." -ForegroundColor Cyan
$tmpRoot = "C:\Users\juanf\OneDrive\Escritorio\qwen-en-to-es\pyvideotrans\tmp"
$latestFolder = Get-ChildItem $tmpRoot -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if ($latestFolder) {
    Write-Host "Carpeta mas reciente: $($latestFolder.FullName)" -ForegroundColor Yellow
    $wavFiles = Get-ChildItem $latestFolder.FullName -Filter "*.wav" -Recurse -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 5
    if ($wavFiles) {
        Write-Host "Archivos .wav encontrados (ultimos 5):" -ForegroundColor Cyan
        foreach ($wav in $wavFiles) {
            $size = $wav.Length
            Write-Host "  - $($wav.Name): $size bytes" -ForegroundColor Gray
        }
    }
}

Write-Host ""
Write-Host "[OK] Diagnostico finalizado." -ForegroundColor Green
