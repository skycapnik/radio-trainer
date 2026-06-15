#!/bin/bash
echo ""
echo "============================================"
echo "  Тренажёр радиообмена — локальный запуск"
echo "============================================"
echo ""

# Проверяем Python
if ! command -v python3 &> /dev/null; then
    echo "[ОШИБКА] Python3 не найден. Установите: sudo apt install python3 python3-pip"
    exit 1
fi

# Проверяем ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "[ОШИБКА] ffmpeg не найден."
    echo "  Ubuntu/Debian:  sudo apt install ffmpeg"
    echo "  macOS:          brew install ffmpeg"
    exit 1
fi

echo "[1/2] Устанавливаю/обновляю зависимости..."
pip3 install -q flask faster-whisper torch pydub soundfile noisereduce matplotlib numpy

echo "[2/2] Запускаю сервер..."
echo ""
echo "  Откройте браузер: http://localhost:5000"
echo "  Для остановки нажмите Ctrl+C"
echo ""
python3 app.py
