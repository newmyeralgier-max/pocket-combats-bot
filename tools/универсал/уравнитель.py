import json
import os
from datetime import datetime

# === Настройки путей ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUTS_DIR = os.path.join(BASE_DIR, "характеристики2")
JSON_PATH = os.path.join(CUTS_DIR, "not_full.json")


# === Логгер ===
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# === Проверка наличия JSON ===
if not os.path.exists(JSON_PATH):
    raise FileNotFoundError(f"❌ Файл JSON не найден: {JSON_PATH}")

log(f"Загружаем JSON: {JSON_PATH}")
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

log(f"✅ Загружено {len(data)} записей")

# === Проверка папки с вырезками ===
if not os.path.isdir(CUTS_DIR):
    raise FileNotFoundError(f"❌ Папка с вырезками не найдена: {CUTS_DIR}")

cuts_files = [f for f in os.listdir(CUTS_DIR) if os.path.isfile(os.path.join(CUTS_DIR, f))]
log(f"📂 Найдено {len(cuts_files)} файлов в {CUTS_DIR}")

# === Обработка каждого файла ===
processed = 0
skipped = 0

for filename in cuts_files:
    file_path = os.path.join(CUTS_DIR, filename)

    # Пример фильтрации по JSON
    if filename not in data:
        log(f"⚠ Пропущен (нет в JSON): {filename}")
        skipped += 1
        continue

    # Здесь твоя логика обработки вырезки
    log(f"🔄 Обрабатываем: {filename}")
    # ... твой код обработки ...
    processed += 1

# === Итог ===
log(f"Готово. Обработано: {processed}, пропущено: {skipped}")
