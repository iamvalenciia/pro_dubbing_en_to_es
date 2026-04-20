param(
    [switch]$NoPush,
    [string]$Tag = "latest",
    [switch]$DeepClean
)

$ErrorActionPreference = "Continue"
$IMAGE = "iamvalenciia/dubbing-app"
$PROJECT = "C:\Users\juanf\OneDrive\Escritorio\qwen-en-to-es"
$VHDX = "C:\Users\juanf\AppData\Local\Docker\wsl\disk\docker_data.vhdx"

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
if ($DeepClean) {
    Write-Host "     DOCKER DEEP CLEAN & BUILD UTILITY   " -ForegroundColor Red
} else {
    Write-Host "     DOCKER FAST BUILD INCREMENTAL       " -ForegroundColor Cyan
}
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

if ($DeepClean) {
    Write-Host "### INICIANDO LIMPIEZA NUCLEAR ###" -ForegroundColor Red
    
    # Step 1: Prune
    Write-Host "[1/5] Limpiando cache de Docker..." -ForegroundColor Yellow
    docker system prune -a --volumes -f 2>$null
    Write-Host "      [OK] Caché eliminada." -ForegroundColor Green

    # Step 2: KILL Docker Desktop completamente
    Write-Host "[2/5] Forzando cierre de procesos Docker..." -ForegroundColor Yellow
    $dockerProcesses = @("Docker Desktop", "com.docker.backend", "com.docker.proxy", "vpnkit", "DockerCli")
    foreach ($proc in $dockerProcesses) {
        Get-Process -Name "$proc*" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 5
    Write-Host "      [OK] Procesos terminados." -ForegroundColor Green

    # Step 3: Shutdown WSL
    Write-Host "[3/5] Apagando máquina virtual WSL..." -ForegroundColor Yellow
    wsl --shutdown
    Start-Sleep -Seconds 3
    Write-Host "      [OK] WSL apagado." -ForegroundColor Green

    # Step 4: Compact VHDX
    Write-Host "[4/5] Compactando disco VHDX..." -ForegroundColor Yellow
    if (-not (Test-Path $VHDX)) {
        Write-Host "      [ERROR] El archivo VHDX no existe en la ruta." -ForegroundColor Red
        exit 1
    }

    $tempFile = [System.IO.Path]::GetTempFileName()
    $diskpartScript = "select vdisk file=`"$VHDX`"`nattach vdisk readonly`ncompact vdisk`ndetach vdisk`nexit"
    $diskpartScript | Set-Content $tempFile -Encoding ASCII

    $diskpartOut = diskpart /s $tempFile 2>&1
    $diskpartCode = $LASTEXITCODE
    Remove-Item $tempFile -Force

    if ($diskpartCode -ne 0) {
        Write-Host "      [ERROR] Diskpart falló." -ForegroundColor Red
        exit 1
    }
    Write-Host "      [OK] VHDX compactado exitosamente." -ForegroundColor Green

    # Step 5: Restart Docker Desktop
    Write-Host "[5/5] Reiniciando Docker Desktop..." -ForegroundColor Yellow
    if (Test-Path "C:\Program Files\Docker\Docker\Docker Desktop.exe") {
        Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    } else {
        exit 1
    }

    Start-Sleep -Seconds 10
    $maxWait = 300
    $waited = 0
    $dockerReady = $false

    Write-Host "      Realizando Smart Polling del Daemon..." -ForegroundColor DarkYellow
    while ($waited -lt $maxWait) {
        $result = docker version --format '{{.Server.Version}}' 2>$null
        if ($LASTEXITCODE -eq 0 -and $result) {
            $dockerReady = $true
            break
        }
        Start-Sleep -Seconds 5
        $waited += 5
    }

    if (-not $dockerReady) {
        Write-Host "      [ERROR] Docker no arrancó." -ForegroundColor Red
        exit 1
    }
    Write-Host "      [OK] Docker Engine en línea." -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "-> MODO DE CONSTRUCCIÓN SÚPER RÁPIDA ACTIVO (Usando Caché)" -ForegroundColor Green
    Write-Host "-> (Cuando necesites ahorrar espacio en disco C:, usa el flag '-DeepClean')" -ForegroundColor DarkGray
    Write-Host ""
}

$FULL_TAG = $IMAGE + ":" + $Tag

# FINAL PASES: Build y Push siempre se ejecutan
Write-Host "[BUILD] Compilando imagen de Docker ($FULL_TAG)..." -ForegroundColor Yellow
Push-Location $PROJECT
docker build --progress=plain -t $FULL_TAG .
$buildExit = $LASTEXITCODE
Pop-Location

if ($buildExit -ne 0) {
    Write-Host "      [ERROR] Falló el Build de la imagen (Código $buildExit). Proceso detenido." -ForegroundColor Red
    exit 1
}
Write-Host "      [OK] Build completado con éxito." -ForegroundColor Green

if (-not $NoPush) {
    Write-Host "[PUSH] Sincronizando imagen con el Docker Hub..." -ForegroundColor Yellow
    docker push $FULL_TAG
    if ($LASTEXITCODE -ne 0) {
        Write-Host "      [ERROR] Falló el Push de la imagen." -ForegroundColor Red
        exit 1
    }
    Write-Host "      [OK] Imagen subida exitosamente." -ForegroundColor Green
} else {
    Write-Host "[PUSH] Push omitido (-NoPush especificado)." -ForegroundColor DarkYellow
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host " COMPLETADO: $FULL_TAG " -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
