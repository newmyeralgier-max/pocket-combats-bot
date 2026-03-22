import os
import json

# === НАСТРОЙКИ ===
FOLDER_PATH = r"C:\Obsidian\New-Life\Учеба"  # Папка, которую проверяем
OUTPUT_DIR = r"C:\bot\tools\obsidian"  # Куда сохранить JSON
MIN_KB = 0        # Минимальный размер файла в КБ
MAX_KB = 2 # Максимальный размер файла в КБ
INCLUDE_SUBFOLDERS = True  # True — искать и в подпапках, False — только в указанной папке

# === ЛОГИКА ===
def find_small_files(folder_path, min_kb, max_kb, include_subfolders=False):
    result = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            size_kb = os.path.getsize(file_path) / 1024
            if min_kb <= size_kb <= max_kb:
                name_without_ext = os.path.splitext(file)[0]
                result.append(name_without_ext)
        if not include_subfolders:
            break
    return sorted(result)

def save_json(data, folder_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    folder_name = os.path.basename(os.path.normpath(folder_path))
    output_path = os.path.join(output_dir, f"not_{folder_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] Найдено файлов: {len(data)}")
    print(f"[SAVE] Результат сохранён в: {output_path}")

# === ЗАПУСК ===
files_list = find_small_files(FOLDER_PATH, MIN_KB, MAX_KB, INCLUDE_SUBFOLDERS)
save_json(files_list, FOLDER_PATH, OUTPUT_DIR)
