@echo off
setlocal

set "ROOT_DIR=%~dp0"
set "PYVIDEOTRANS_PYTHON=%ROOT_DIR%.venv-py310\Scripts\python.exe"

if not exist "%PYVIDEOTRANS_PYTHON%" (
  echo [ERROR] No se encontro Python del engine en:
  echo         %PYVIDEOTRANS_PYTHON%
  echo Ejecuta primero la preparacion del entorno .venv-py310.
  exit /b 1
)

REM Configure HuggingFace Hub to use standard cache (required for ctranslate2)
set "HF_HOME=%USERPROFILE%\.cache\huggingface"
set "HF_HUB_CACHE=%USERPROFILE%\.cache\huggingface\hub"
set "TORCH_HOME=%USERPROFILE%\.cache\torch"

echo [INFO] PYVIDEOTRANS_PYTHON=%PYVIDEOTRANS_PYTHON%
echo [INFO] HF_HOME=%HF_HOME%
python "%ROOT_DIR%main_ui.py"
