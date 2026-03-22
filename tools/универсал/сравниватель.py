import json
import os

# ===== НАСТРОЙКИ =====
FOLDER_1 = r"c:\bot\tpl\иконки_предметов"  # Первая папка (эталон)
FOLDER_2 = r"c:\bot\tpl\alpha"  # Вторая папка (с чем сравниваем)
OUTPUT_DIR = r"c:\bot\tools\cfg"  # Куда сохранить JSON


# ===== ЛОГИКА =====
def compare_folders(folder1, folder2, output_dir):
    if not os.path.isdir(folder1):
        print(f"[ERR] Папка не найдена: {folder1}")
        return
    if not os.path.isdir(folder2):
        print(f"[ERR] Папка не найдена: {folder2}")
        return
    os.makedirs(output_dir, exist_ok=True)

    # Списки файлов (только имена, без путей)
    files1 = {f for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))}
    files2 = {f for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))}

    # Что есть в первой, но нет во второй
    missing = sorted(list(files1 - files2))

    # Имя JSON: not_<имя первой папки>.json
    folder1_name = os.path.basename(os.path.normpath(folder2))
    output_path = os.path.join(output_dir, f"not_{folder1_name}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(missing, f, ensure_ascii=False, indent=2)

    print(f"[OK] Найдено отсутствующих файлов: {len(missing)}")
    print(f"[SAVE] Результат сохранён в: {output_path}")


# ===== ЗАПУСК =====
compare_folders(FOLDER_1, FOLDER_2, OUTPUT_DIR)
