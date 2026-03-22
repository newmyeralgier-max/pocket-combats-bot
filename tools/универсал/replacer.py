import json
import os
import re
import shutil

# === НАСТРОЙКИ ===
BASE_DIR = "C:/bot"
MAPPING_FILE = os.path.join(BASE_DIR, "tools", "cfg", "mapping.json")
PROJECT_DIRS = [
    os.path.join(BASE_DIR, "script"),
    os.path.join(BASE_DIR, "tools"),
]
BACKUP_DIR = os.path.join(BASE_DIR, "backup_replace")
DRY_RUN = True  # True = только показать, что будет заменено
ENCODING = "utf-8"

# Исключения по файлам/папкам (регулярки)
EXCLUDE_PATTERNS = [
    r"\.png$",
    r"\.jpg$",
    r"\.jpeg$",
    r"\.gif$",
    r"\.bmp$",
    r"\\venv\\",
    r"/venv/",
    r"\\__pycache__\\",
    r"/__pycache__/",
    r"\\.git\\",
    r"/.git/",
]

# === ЗАГРУЗКА МАППИНГА ===
with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    mapping = json.load(f)

# Сортируем по длине ключа, чтобы длинные пути заменялись первыми
mapping_items = sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True)


# === ФУНКЦИИ ===
def should_exclude(path: str) -> bool:
    for pat in EXCLUDE_PATTERNS:
        if re.search(pat, path, flags=re.IGNORECASE):
            return True
    return False


def backup_file(src_path: str):
    rel_path = os.path.relpath(src_path, BASE_DIR)
    dst_path = os.path.join(BACKUP_DIR, rel_path)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)


def process_file(file_path: str):
    try:
        with open(file_path, "r", encoding=ENCODING) as f:
            content = f.read()
    except Exception:
        return 0, 0

    replaced_count = 0
    for old, new in mapping_items:
        if old in content:
            content = content.replace(old, new)
            replaced_count += 1

    if replaced_count > 0:
        if DRY_RUN:
            print(f"[DRY_RUN] {file_path} — замен: {replaced_count}")
        else:
            backup_file(file_path)
            with open(file_path, "w", encoding=ENCODING) as f:
                f.write(content)
            print(f"[OK] {file_path} — замен: {replaced_count}")
    return replaced_count, 1 if replaced_count > 0 else 0


# === ЗАПУСК ===
def main():
    total_files = 0
    total_replacements = 0
    for root_dir in PROJECT_DIRS:
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                path = os.path.join(root, file)
                if should_exclude(path):
                    continue
                rep_count, file_changed = process_file(path)
                total_replacements += rep_count
                total_files += file_changed
    print(f"\n[ИТОГ] Файлов изменено: {total_files}, замен всего: {total_replacements}")


if __name__ == "__main__":
    main()
