import os
import shutil

# ===== НАСТРОЙКИ =====
SOURCE_FOLDER = r"c:\bot\tpl\источник"  # Папка, откуда берём файлы
TARGET_FOLDER = r"c:\bot\tpl\назначение"  # Папка, куда добавляем файлы
NAME_PART = "rej_"  # Подстрока в имени файла для копирования
FILE_FORMAT = "png"  # "all" — любой формат, или "png"/"jpg"/"txt" и т.д.
OVERWRITE = False  # True — перезаписывать, False — пропускать, если уже есть


# ===== ЛОГИКА =====
def add_files(src_folder, dst_folder, name_part, file_format, overwrite):
    if not os.path.isdir(src_folder):
        print(f"[ERR] Папка-источник не найдена: {src_folder}")
        return
    os.makedirs(dst_folder, exist_ok=True)

    added = 0
    for fname in os.listdir(src_folder):
        full_src = os.path.join(src_folder, fname)

        if not os.path.isfile(full_src):
            continue

        # Проверка подстроки в имени
        if name_part.lower() not in fname.lower():
            continue

        # Проверка формата
        if file_format.lower() != "all":
            if not fname.lower().endswith("." + file_format.lower()):
                continue

        full_dst = os.path.join(dst_folder, fname)

        if not overwrite and os.path.exists(full_dst):
            print(f"[SKIP] Уже существует: {fname}")
            continue

        try:
            shutil.copy2(full_src, full_dst)
            print(f"[ADD] {fname}")
            added += 1
        except Exception as e:
            print(f"[ERR] Не удалось скопировать {fname}: {e}")

    print(f"[DONE] Добавлено файлов: {added}")


# ===== ЗАПУСК =====
add_files(SOURCE_FOLDER, TARGET_FOLDER, NAME_PART, FILE_FORMAT, OVERWRITE)
