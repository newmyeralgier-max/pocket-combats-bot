import os

folder = r"C:\bot\tpl\иконки предметов"  # Укажи путь

# 1. Сначала удаляем оригиналы, если у нас есть _full (чтобы не мешали)
#    Здесь наоборот — мы хотим оставить именно версию без _full, поэтому убираем дубликаты _full
for filename in os.listdir(folder):
    filepath = os.path.join(folder, filename)
    if not os.path.isfile(filepath):
        continue

    name, ext = os.path.splitext(filename)
    if not name.endswith("_full"):
        full_name = f"{name}_full{ext}"
        full_path = os.path.join(folder, full_name)
        if os.path.exists(full_path):
            print(f"Удаляю версию с _full: {full_name}")
            os.remove(full_path)

# 2. Теперь переименовываем все файлы с _full → без него
for filename in os.listdir(folder):
    filepath = os.path.join(folder, filename)
    if not os.path.isfile(filepath):
        continue

    name, ext = os.path.splitext(filename)
    if name.endswith("_full"):
        new_name = name[:-5] + ext
        new_path = os.path.join(folder, new_name)
        print(f"Переименовываю: {filename} → {new_name}")
        os.rename(filepath, new_path)
