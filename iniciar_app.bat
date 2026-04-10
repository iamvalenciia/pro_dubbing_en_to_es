@echo off
title EN-ES Dubbing App
cd /d "%~dp0"
echo ============================================
echo   EN-ES Voice Dubbing  ^|  localhost:7860
echo ============================================
echo.
echo Iniciando servidor... (30-60 seg para cargar modelos)
echo NO cierres esta ventana mientras uses la app.
echo Para parar: Ctrl+C en esta ventana.
echo.
set PYTHONUTF8=1
python app.py
pause
