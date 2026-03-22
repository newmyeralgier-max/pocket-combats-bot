import json
import os
import unicodedata
from pathlib import Path

# === НАСТРОЙКИ ===
# Корневая папка с ассетами
TPL_DIR = Path(r"C:\bot\tpl")
# Куда писать not_*.json
CFG_DIR = Path(r"C:\bot\tools\cfg")

# Папки для каждого списка (можно менять по желанию)
FOLDERS = {
    "not_icon": ["иконки_предметов", "иконки_рун", "монстры"],
    "not_name": [
        "имя_предметов" "монстры" "описание_рун",
    ],
    "not_full": ["описание_предметов"],
}

# Режим теста (True = dry-run, только лог)
DRY_RUN = False

# Сколько примеров отсутствующих показывать в логе
SHOW_EXAMPLES = 10


# === ФУНКЦИИ ===
def normalize_filename(name: str) -> str:
    """Нормализует имя файла: убирает приписки, регистр, Unicode."""
    name = unicodedata.normalize("NFC", name)
    name = name.lower()
    # убираем расширение
    name = os.path.splitext(name)[0]
    # убираем приписки
    suffixes = ["_icon", "_name", "_full", "_item", "— копия", "- копия", " копия", "(1)", "(2)"]
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if name.endswith(suf.lower()):
                name = name[: -len(suf)]
                changed = True
    return name.strip(" _-")


def collect_normalized_files(folder_list):
    """Собирает нормализованные имена файлов из списка папок."""
    found = set()
    for folder in folder_list:
        folder_path = TPL_DIR / folder
        if not folder_path.exists():
            print(f"[!] Папка не найдена: {folder_path}")
            continue
        for root, _, files in os.walk(folder_path):
            for f in files:
                found.add(normalize_filename(f))
    return found


def load_not_list(key):
    """Загружает текущий not_*.json, если есть."""
    not_file = CFG_DIR / f"{key}.json"
    if not_file.exists():
        with open(not_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return set(data)
            except Exception as e:
                print(f"[!] Ошибка чтения {not_file}: {e}")
                return set()
    return set()


def save_not_list(key, data):
    """Сохраняет not_*.json."""
    not_file = CFG_DIR / f"{key}.json"
    with open(not_file, "w", encoding="utf-8") as f:
        json.dump(sorted(data), f, ensure_ascii=False, indent=2)
    print(f"[OK] Файл {not_file} обновлён ({len(data)} элементов)")


def process_not_list(key):
    """Обрабатывает один список not_*"""
    target_folders = FOLDERS[key]
    existing = collect_normalized_files(target_folders)
    missing = load_not_list(key)

    still_missing = sorted(m for m in missing if normalize_filename(m) not in existing)

    print(f"\n=== {key} ===")
    print(f"Папки: {target_folders}")
    print(
        f"Было в списке: {len(missing)} | Найдено: {len(missing) - len(still_missing)} | Осталось: {len(still_missing)}"
    )

    if still_missing:
        print("Примеры отсутствующих:")
        for ex in still_missing[:SHOW_EXAMPLES]:
            print("  -", ex)

    if DRY_RUN:
        print("[DRY-RUN] Список не будет перезаписан")
    else:
        save_not_list(key, still_missing)


# === ЗАПУСК ===
if __name__ == "__main__":
    for key in FOLDERS:
        process_not_list(key)
