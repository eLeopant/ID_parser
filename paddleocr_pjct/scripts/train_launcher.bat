@echo off
REM ============================================================
REM PaddleOCR Detection Model Fine-tuning Launcher
REM Model: PP-OCRv4 mobile detector (student)
REM ============================================================

REM Switch to project root directory (parent of scripts/)
set "PROJECT_DIR=%~dp0.."
cd /d "%PROJECT_DIR%" || exit /b 1

REM Add cuDNN DLL path (installed by conda)
if defined CONDA_PREFIX (
    set "PATH=%CONDA_PREFIX%\Library\bin;%PATH%"
)

REM Config file path
set "CONFIG_PATH=.\PaddleOCR\configs\det\PP-OCRv4\PP-OCRv4_mobile_det.yml"

echo ============================================================
echo   PaddleOCR Training - PP-OCRv4 Mobile Detector
echo   Config: %CONFIG_PATH%
echo   Train data: .\data\train\
echo   Eval data:  .\data\val\
echo   Epoch: 50, Batch Size: 8
echo ============================================================

python .\PaddleOCR\tools\train.py -c "%CONFIG_PATH%"

echo.
echo Training finished. Model saved to: .\output\PP-OCRv4_mobile_det\
pause
