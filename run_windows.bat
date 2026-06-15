@echo off
echo.
echo ============================================
echo   Radio Trainer - Full Setup
echo ============================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Install Python 3.10+ from python.org
    echo Make sure to check "Add Python to PATH"
    pause
    exit /b 1
)

echo Installing ALL packages (wait 3-5 minutes)...
echo.

pip install flask torch --index-url https://download.pytorch.org/whl/cpu
pip install faster-whisper pydub pyaudioop-lts soundfile noisereduce matplotlib numpy

echo.
echo ============================================
echo   Done! Starting server...
echo   Open browser: http://localhost:5000
echo   Press Ctrl+C to stop
echo ============================================
echo.
python app.py
pause
