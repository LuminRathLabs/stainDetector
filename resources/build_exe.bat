@echo off
setlocal
cd /d "%~dp0"

echo ========================================================
echo   Generando Ejecutable: Detector de Manchas (Actual)
echo ========================================================

:: 1. Activar entorno virtual si existe
if exist "..\venv_py311\Scripts\activate.bat" (
    echo Activando entorno virtual...
    call "..\venv_py311\Scripts\activate.bat"
) else (
    echo [ADVERTENCIA] No se encontro el entorno virtual en ..\venv_py311
    echo Intentando con el Python del sistema...
)

:: 2. Asegurar PyInstaller
echo Verificando PyInstaller...
python -m pip install pyinstaller --quiet

:: 3. Ejecutar PyInstaller
echo Compilando... Esto puede tardar varios minutos debido a PyTorch...
python -m PyInstaller --clean detect_manchas_actual.spec

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================
    echo   ¡EXITO! El ejecutable esta en: dist\detect_manchas\
    echo ========================================================
) else (
    echo.
    echo [ERROR] Hubo un problema al compilar.
)

pause
