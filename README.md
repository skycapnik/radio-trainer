# ✈️ Тренажёр радиообмена (локальная версия без Gradio)

Полный аналог Jupyter-ноутбука v15, переписанный на Flask.  
Работает **без интернета**, без Gradio, без Google Colab.

---

## 📦 Что нужно установить один раз

### 1. Python 3.10+
- Windows: https://www.python.org/downloads/
- Linux: `sudo apt install python3 python3-pip`

### 2. ffmpeg
- **Windows**: https://ffmpeg.org/download.html → скачать, распаковать, добавить папку `bin\` в PATH
- **Linux**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`

### 3. Пакеты Python — устанавливаются автоматически при первом запуске.

---

## 🚀 Запуск

### Windows
Дважды кликнуть `run_windows.bat`  
(или в командной строке: `python app.py`)

### Linux / macOS
```bash
chmod +x run_linux.sh
./run_linux.sh
```

После запуска откройте браузер: **http://localhost:5000**

---

## ⚙️ Настройки (в начале app.py)

| Параметр | По умолчанию | Описание |
|---|---|---|
| `MODEL_SIZE` | `large-v3` | Модель Whisper. Для слабых ПК: `small` или `base` |
| `N_SCENARIOS` | `2` | Количество сценариев в тесте |
| `USE_NOISE_REDUCTION` | `True` | Шумоподавление |
| `MIC_DELAY_SEC` | `3` | Задержка до начала записи |

### Модели Whisper по скорости/точности
| Модель | Размер | VRAM/RAM | Скорость |
|---|---|---|---|
| `large-v3` | 1.5 ГБ | 4+ ГБ | медленно, точнее всего |
| `medium` | 769 МБ | 2+ ГБ | средне |
| `small` | 244 МБ | 1+ ГБ | быстро |
| `base` | 74 МБ | 512 МБ | очень быстро, менее точно |

---

## 🌐 Доступ с других устройств в сети

Сервер уже слушает `0.0.0.0:5000`, так что с телефона или другого ПК в той же сети:  
`http://<IP-вашего-компьютера>:5000`

Узнать IP: `ipconfig` (Windows) или `ip a` (Linux).

---

## 📁 Структура
```
radio_trainer/
├── app.py              ← главный файл (логика + сервер)
├── templates/
│   └── index.html      ← веб-интерфейс
├── static/
│   ├── charts/         ← временные графики (создаются автоматически)
│   └── audio_tmp/      ← временные аудиофайлы
├── requirements.txt
├── run_windows.bat
└── run_linux.sh
```
