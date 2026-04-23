param(
    [string]$ImageTag = "iamvalenciia/dubbing-app:preflight"
)

$ErrorActionPreference = "Stop"

Write-Host "[1/3] Building image: $ImageTag"
docker build -t $ImageTag .
if ($LASTEXITCODE -ne 0) {
    throw "docker build failed"
}

Write-Host "[2/3] Running dependency checks (pip check + import smoke)"
$knownPipMismatch = "qwen-tts 0.1.1 has requirement transformers==4.57.3, but you have transformers 4.57.6."
$pipCheckOutput = docker run --rm $ImageTag python -m pip check 2>&1 | Out-String
$pipCheckExit = $LASTEXITCODE
Write-Host $pipCheckOutput
if ($pipCheckExit -ne 0) {
    $requirementLines = @(
        ($pipCheckOutput -split "`r?`n") |
            Where-Object { $_ -match " has requirement " } |
            ForEach-Object { $_.Trim() }
    )

    $unexpectedLines = @(
        $requirementLines |
            Where-Object { $_ -ne $knownPipMismatch }
    )

    if ($unexpectedLines.Count -eq 0 -and $pipCheckOutput.Contains($knownPipMismatch)) {
        Write-Warning "pip check reported only the known qwen-tts/transformers pin mismatch; continuing preflight."
    }
    else {
        throw "pip check failed with unexpected dependency issues"
    }
}

docker run --rm $ImageTag python -c "import tenacity, ten_vad, gradio, fastapi, uvicorn, librosa, soundfile, faster_whisper, qwen_tts, qwen_asr, zhconv; print('imports_ok')"
if ($LASTEXITCODE -ne 0) {
    throw "import smoke failed"
}

Write-Host "[3/3] Running PyVideoTrans CLI startup smoke"
docker run --rm $ImageTag python /app/pyvideotrans/cli.py --help
if ($LASTEXITCODE -ne 0) {
    throw "cli smoke failed"
}

Write-Host "Preflight OK for image: $ImageTag"
