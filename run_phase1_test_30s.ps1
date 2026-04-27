param(
    [Parameter(Mandatory = $true)]
    [string]$VideoPath,

    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [string]$CliPath = ".\pyvideotrans\cli.py",
    [int]$DurationSec = 30,
    [string]$SourceLanguage = "en",
    [string]$TargetLanguage = "es",
    [int]$RecognType = 0,
    [string]$ModelName = "large-v3-turbo",
    [int]$TranslateType = 5,
    [switch]$NoCuda,
    [switch]$NoDiariz,
    [int]$NumSpeakers = 0,
    [switch]$NoClearCache,
    [switch]$RunPhase2,
    [string]$GoogleApiKey,
    [string]$GoogleModel = "gemini-3.1-flash-lite-preview",
    [double]$BackAudioVolume = 0.28,
    [string]$VoiceRefId,
    [string]$SpeakerVoiceMapFile
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
    param(
        [int]$N,
        [string]$Text
    )
    Write-Host ("[TEST PASO {0}] {1}" -f $N, $Text) -ForegroundColor Cyan
}

function Import-DotEnvFile {
    param([string]$EnvFilePath)

    if (-not (Test-Path $EnvFilePath)) {
        return
    }

    foreach ($rawLine in Get-Content $EnvFilePath) {
        $line = [string]$rawLine
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        $trimmed = $line.Trim()
        if ($trimmed.StartsWith("#")) {
            continue
        }
        $eqIndex = $trimmed.IndexOf("=")
        if ($eqIndex -lt 1) {
            continue
        }
        $key = $trimmed.Substring(0, $eqIndex).Trim()
        $value = $trimmed.Substring($eqIndex + 1).Trim()
        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        if (-not [string]::IsNullOrWhiteSpace($key) -and -not [string]::IsNullOrWhiteSpace($value)) {
            [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
}

function Resolve-ApiKeyArgument {
    param([string]$RawValue)

    if ([string]::IsNullOrWhiteSpace($RawValue)) {
        return $null
    }

    $namedValue = [System.Environment]::GetEnvironmentVariable($RawValue, "Process")
    if (-not [string]::IsNullOrWhiteSpace($namedValue)) {
        return $namedValue
    }

    return $RawValue
}

function Ensure-QwenTtsRuntime {
    param([string]$PyExe)

    $hasQwenTts = (& $PyExe -c "import importlib.util; print('1' if importlib.util.find_spec('qwen_tts') else '0')").Trim()
    if ($hasQwenTts -eq "1") {
        Write-Host "[preflight] qwen_tts ya está instalado." -ForegroundColor Green
        return
    }

    Write-Host "[preflight] qwen_tts no está instalado. Instalando runtime Qwen TTS..." -ForegroundColor Yellow

    & $PyExe -m pip install transformers==4.57.6 accelerate==1.12.0 einops sox
    if ($LASTEXITCODE -ne 0) {
        throw "Falló instalación de dependencias base para Qwen TTS."
    }

    & $PyExe -m pip install qwen-asr==0.0.6
    if ($LASTEXITCODE -ne 0) {
        throw "Falló instalación de qwen-asr==0.0.6."
    }

    & $PyExe -m pip install --no-deps qwen-tts==0.1.1
    if ($LASTEXITCODE -ne 0) {
        throw "Falló instalación de qwen-tts==0.1.1."
    }

    $verify = (& $PyExe -c "import qwen_tts; print('QWEN_TTS_OK')").Trim()
    if ($verify -notmatch "QWEN_TTS_OK") {
        throw "qwen_tts sigue sin poder importarse después de la instalación."
    }
    Write-Host "[preflight] Runtime Qwen TTS listo." -ForegroundColor Green
}

function Resolve-AnalyzeTargetDir {
    param([string]$VideoBaseName)

    $vtOutput = Join-Path $projectRoot "pyvideotrans\output"
    if (-not (Test-Path $vtOutput)) {
        return $null
    }

    $safe = ($VideoBaseName -replace '[\s\. #*?!:"]', '-')
    $candidates = @(Get-ChildItem -Path $vtOutput -Directory -ErrorAction SilentlyContinue |
        Where-Object {
            (Test-Path (Join-Path $_.FullName "speaker_profile.json")) -and (
                $_.Name -like "*$safe*" -or $_.Name -like "*$VideoBaseName*"
            )
        } |
        Sort-Object LastWriteTime -Descending)

    if ($candidates.Count -gt 0) {
        return $candidates[0].FullName
    }
    return $null
}

function Resolve-LatestOutputDir {
    param([string]$VideoBaseName)

    $vtOutput = Join-Path $projectRoot "pyvideotrans\output"
    if (-not (Test-Path $vtOutput)) {
        return $null
    }

    $safe = ($VideoBaseName -replace '[\s\. #*?!:"]', '-')
    $candidates = @(Get-ChildItem -Path $vtOutput -Directory -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -like "*$safe*" -or $_.Name -like "*$VideoBaseName*"
        } |
        Sort-Object LastWriteTime -Descending)

    if ($candidates.Count -gt 0) {
        return $candidates[0].FullName
    }
    return $null
}

function Publish-Phase2Artifacts {
    param(
        [string]$OutputDir,
        [string]$BaseSafeName,
        [int]$TestDurationSec
    )

    if (-not $OutputDir -or -not (Test-Path $OutputDir)) {
        throw "No se encontró la carpeta de salida de Fase 2."
    }

    $publishDir = Join-Path $projectRoot "output"
    New-Item -ItemType Directory -Force -Path $publishDir | Out-Null

    $finalMp4 = @(Get-ChildItem -Path $OutputDir -Filter *.mp4 -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1)
    $dubAudio = Join-Path $OutputDir "es.m4a"
    $dubSrt = Join-Path $OutputDir "es.srt"

    $published = @{}
    if ($finalMp4.Count -gt 0) {
        $destMp4 = Join-Path $publishDir ("{0}_test{1}s_dubbed.mp4" -f $BaseSafeName, $TestDurationSec)
        Copy-Item -Force $finalMp4[0].FullName $destMp4
        $published["mp4"] = $destMp4
    }
    if (Test-Path $dubAudio) {
        $destAudio = Join-Path $publishDir ("{0}_test{1}s_dubbed_audio.m4a" -f $BaseSafeName, $TestDurationSec)
        Copy-Item -Force $dubAudio $destAudio
        $published["audio"] = $destAudio
    }
    if (Test-Path $dubSrt) {
        $destSrt = Join-Path $publishDir ("{0}_test{1}s_subtitles_es.srt" -f $BaseSafeName, $TestDurationSec)
        Copy-Item -Force $dubSrt $destSrt
        $published["srt"] = $destSrt
    }

    return $published
}

function Get-SpeakersFromProfile {
    param([string]$TargetDir)

    $profilePath = Join-Path $TargetDir "speaker_profile.json"
    $identityPath = Join-Path $TargetDir "speaker_identity.json"
    $speakers = @()

    if (-not (Test-Path $profilePath)) {
        return @(@{ speaker_id = "spk0"; ai_label = "narrador" })
    }

    $profile = Get-Content $profilePath -Raw | ConvertFrom-Json
    $identityMap = @{}
    if (Test-Path $identityPath) {
        try {
            $identity = Get-Content $identityPath -Raw | ConvertFrom-Json
            foreach ($item in ($identity.speakers | Where-Object { $_ })) {
                if ($item.speaker_id) {
                    $identityMap[[string]$item.speaker_id] = $item
                }
            }
        }
        catch {
        }
    }

    foreach ($spk in ($profile.speakers | Where-Object { $_ })) {
        $sid = [string]$spk.speaker_id
        if ([string]::IsNullOrWhiteSpace($sid)) {
            continue
        }
        $aiLabel = $sid
        if ($identityMap.ContainsKey($sid)) {
            $role = [string]$identityMap[$sid].role_label
            $gender = [string]$identityMap[$sid].gender_label
            $parts = @()
            if ($role -and $role -ne "-") { $parts += $role }
            if ($gender -and $gender -ne "-") { $parts += $gender }
            if ($parts.Count -gt 0) {
                $aiLabel = ($parts -join " ")
            }
        }
        $speakers += @{ speaker_id = $sid; ai_label = $aiLabel }
    }

    if ($speakers.Count -eq 0) {
        $speakers += @{ speaker_id = "spk0"; ai_label = "narrador" }
    }
    return $speakers
}

function Build-SpeakerVoiceMap {
    param(
        [array]$Speakers,
        [string]$CatalogPath,
        [string]$SingleVoiceRefId,
        [string]$MapFile
    )

    if ($MapFile) {
        if (-not (Test-Path $MapFile)) {
            throw "No existe SpeakerVoiceMapFile: $MapFile"
        }
        return (Get-Content $MapFile -Raw | ConvertFrom-Json -AsHashtable)
    }

    if (-not (Test-Path $CatalogPath)) {
        throw "No existe catálogo de voces: $CatalogPath"
    }

    $catalogJson = Get-Content $CatalogPath -Raw | ConvertFrom-Json
    $voices = @($catalogJson.voices)
    if ($voices.Count -eq 0) {
        throw "El catálogo de voces está vacío: $CatalogPath"
    }

    Write-Step 6 "Voces disponibles para referencia"
    for ($i = 0; $i -lt $voices.Count; $i++) {
        $v = $voices[$i]
        Write-Host ("  [{0}] {1} | {2}s" -f $i, $v.ref_id, [int]$v.duration_sec)
    }

    $selectedRef = $SingleVoiceRefId
    if ([string]::IsNullOrWhiteSpace($selectedRef)) {
        $choice = Read-Host "Selecciona índice de voz para usar en TODOS los speakers (Enter=0)"
        if ([string]::IsNullOrWhiteSpace($choice)) {
            $choice = "0"
        }
        if ($choice -notmatch '^\d+$') {
            throw "Índice inválido: $choice"
        }
        $idx = [int]$choice
        if ($idx -lt 0 -or $idx -ge $voices.Count) {
            throw "Índice fuera de rango: $idx"
        }
        $selectedRef = [string]$voices[$idx].ref_id
    }

    $selectedVoice = $voices | Where-Object { $_.ref_id -eq $selectedRef } | Select-Object -First 1
    if (-not $selectedVoice) {
        throw "ref_id no encontrado en catálogo: $selectedRef"
    }

    $map = @{}
    foreach ($spk in $Speakers) {
        $sid = [string]$spk.speaker_id
        $map[$sid] = @{
            ref = [string]$selectedVoice.normalized_path
            ref_id = [string]$selectedVoice.ref_id
        }
    }

    Write-Host ("[voice-map] Asignada voz '{0}' a {1} speaker(s)." -f $selectedVoice.ref_id, $Speakers.Count) -ForegroundColor Yellow
    return $map
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot
Import-DotEnvFile -EnvFilePath (Join-Path $projectRoot ".env")

if (-not (Test-Path $PythonExe)) {
    throw "No se encontró Python del venv en: $PythonExe"
}
if (-not (Test-Path $CliPath)) {
    throw "No se encontró pyvideotrans/cli.py en: $CliPath"
}
if (-not (Test-Path $VideoPath)) {
    throw "No se encontró el video de entrada: $VideoPath"
}
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    throw "ffmpeg no está en PATH. Instálalo o agrega ffmpeg.exe al PATH."
}

Write-Step 1 "Preparando entorno"
$resolvedGoogleApiKey = Resolve-ApiKeyArgument -RawValue $GoogleApiKey
if ($resolvedGoogleApiKey) {
    $env:API_GOOGLE_STUDIO = $resolvedGoogleApiKey
}
if (-not $env:GEMINI_MODEL -and $GoogleModel) {
    $env:GEMINI_MODEL = $GoogleModel
}
if ($env:API_GOOGLE_STUDIO) {
    Write-Host ("[env] API_GOOGLE_STUDIO=SET | GEMINI_MODEL={0}" -f $GoogleModel) -ForegroundColor Green
}
else {
    Write-Host "[env] API_GOOGLE_STUDIO no está configurada. Gemini puede omitirse." -ForegroundColor Yellow
}

$tempDir = Join-Path $projectRoot "temp_workspace\qdp_data\temp_processing"
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

$inputFull = (Resolve-Path $VideoPath).Path
$baseName = [System.IO.Path]::GetFileNameWithoutExtension($inputFull)
$ext = [System.IO.Path]::GetExtension($inputFull)
if ([string]::IsNullOrWhiteSpace($ext)) {
    $ext = ".mp4"
}
$safeName = ($baseName -replace '[\s\. #*?!:"]', '-')
$testVideo = Join-Path $tempDir ("test{0}s_{1}{2}" -f $DurationSec, $safeName, $ext)

Write-Step 2 ("Recortando video de prueba a {0}s" -f $DurationSec)
& ffmpeg -y -hide_banner -loglevel warning -i $inputFull -t $DurationSec -c copy $testVideo
if (-not (Test-Path $testVideo)) {
    throw "No se pudo crear el clip de prueba: $testVideo"
}

$phase1Args = @(
    $CliPath,
    "--task", "analyze",
    "--name", $testVideo,
    "--source_language_code", $SourceLanguage,
    "--target_language_code", $TargetLanguage,
    "--recogn_type", $RecognType,
    "--model_name", $ModelName,
    "--translate_type", $TranslateType
)
if (-not $NoCuda) {
    $phase1Args += "--cuda"
}
if (-not $NoDiariz) {
    $phase1Args += @("--enable_diariz", "--nums_diariz", $NumSpeakers)
}
if ($NoClearCache) {
    $phase1Args += "--no-clear-cache"
}

Write-Step 3 "Ejecutando Fase 1 (analyze)"
Write-Host ("[phase1] Command: {0} {1}" -f $PythonExe, ($phase1Args -join ' ')) -ForegroundColor DarkGray
& $PythonExe @phase1Args
if ($LASTEXITCODE -ne 0) {
    throw "Fase 1 falló con código $LASTEXITCODE"
}

Write-Step 4 "Fase 1 completada"
$targetDir = Resolve-AnalyzeTargetDir -VideoBaseName ([System.IO.Path]::GetFileNameWithoutExtension($testVideo))
if ($targetDir) {
    Write-Host ("[phase1] target_dir detectado: {0}" -f $targetDir) -ForegroundColor Green
}
else {
    Write-Host "[phase1] No se detectó target_dir automáticamente. Se seguirá con speaker map por defecto." -ForegroundColor Yellow
}

if (-not $RunPhase2) {
    Write-Host "[phase1-test] OK. Fase 1 finalizada. Usa -RunPhase2 para continuar con doblaje." -ForegroundColor Green
    Write-Host "[phase1-test] Clip test usado: $testVideo" -ForegroundColor Green
    exit 0
}

Write-Step 5 "Preparando selección de voz para Fase 2"
$speakers = @(@{ speaker_id = "spk0"; ai_label = "narrador" })
if ($targetDir) {
    $speakers = Get-SpeakersFromProfile -TargetDir $targetDir
}
Write-Host "[speakers] Detectados:" -ForegroundColor Cyan
foreach ($spk in $speakers) {
    Write-Host ("  - {0} ({1})" -f $spk.speaker_id, $spk.ai_label)
}

$voiceCatalogPath = Join-Path $projectRoot "input\voice_refs\voice_refs_catalog.json"
$speakerMapPath = Join-Path $projectRoot "input\voice_refs\speaker_voice_map.json"
$speakerMap = Build-SpeakerVoiceMap -Speakers $speakers -CatalogPath $voiceCatalogPath -SingleVoiceRefId $VoiceRefId -MapFile $SpeakerVoiceMapFile

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $speakerMapPath) | Out-Null
$speakerMapJson = $speakerMap | ConvertTo-Json -Depth 8
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($speakerMapPath, $speakerMapJson, $utf8NoBom)
Write-Host ("[voice-map] Escrito en: {0}" -f $speakerMapPath) -ForegroundColor Green

Write-Step 7 "Preflight de runtime Qwen TTS"
Ensure-QwenTtsRuntime -PyExe $PythonExe

# Estándar por defecto para fondo original sutil en cualquier ejecución del script.
$backAudioVol = [Math]::Round($BackAudioVolume, 2)
if ($backAudioVol -lt 0.0) { $backAudioVol = 0.0 }
if ($backAudioVol -gt 1.0) { $backAudioVol = 1.0 }
$env:QDP_BACKAUDIO_VOLUME = [string]$backAudioVol
$env:QDP_BACKAUDIO_SOURCE = "original"
Write-Host ("[env] QDP_BACKAUDIO_VOLUME={0}" -f $env:QDP_BACKAUDIO_VOLUME) -ForegroundColor Green
Write-Host ("[env] QDP_BACKAUDIO_SOURCE={0}" -f $env:QDP_BACKAUDIO_SOURCE) -ForegroundColor Green

$phase2Args = @(
    $CliPath,
    "--task", "vtv",
    "--name", $testVideo,
    "--source_language_code", $SourceLanguage,
    "--target_language_code", $TargetLanguage,
    "--recogn_type", $RecognType,
    "--model_name", $ModelName,
    "--translate_type", $TranslateType,
    "--tts_type", "1",
    "--voice_role", "clone",
    "--subtitle_type", "0",
    "--video_autorate"
)

if (-not $NoCuda) {
    $phase2Args += "--cuda"
}
if (-not $NoDiariz) {
    $phase2Args += @("--enable_diariz", "--nums_diariz", $NumSpeakers)
}
if ($NoClearCache) {
    $phase2Args += "--no-clear-cache"
}
else {
    $phase2Args += "--no-clear-cache"
}

Write-Step 8 "Ejecutando Fase 2 (vtv + voice clone)"
Write-Host ("[phase2] Command: {0} {1}" -f $PythonExe, ($phase2Args -join ' ')) -ForegroundColor DarkGray
& $PythonExe @phase2Args
if ($LASTEXITCODE -ne 0) {
    throw "Fase 2 falló con código $LASTEXITCODE"
}

Write-Step 9 "Pipeline TDD completado (Fase 1 + Fase 2)"
$phase2OutputDir = Resolve-LatestOutputDir -VideoBaseName ([System.IO.Path]::GetFileNameWithoutExtension($testVideo))
$publishedArtifacts = Publish-Phase2Artifacts -OutputDir $phase2OutputDir -BaseSafeName $safeName -TestDurationSec $DurationSec

Write-Host ("[done] Clip test: {0}" -f $testVideo) -ForegroundColor Green
if ($publishedArtifacts.ContainsKey("mp4")) {
    Write-Host ("[done] Video doblado final: {0}" -f $publishedArtifacts["mp4"]) -ForegroundColor Green
}
if ($publishedArtifacts.ContainsKey("audio")) {
    Write-Host ("[done] Audio doblado final: {0}" -f $publishedArtifacts["audio"]) -ForegroundColor Green
}
if ($publishedArtifacts.ContainsKey("srt")) {
    Write-Host ("[done] Subtítulos ES: {0}" -f $publishedArtifacts["srt"]) -ForegroundColor Green
}
