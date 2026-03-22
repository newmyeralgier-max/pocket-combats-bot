import os

folder = r"C:\bot\tpl\характеристики"  # Укажи путь

# 1. Сначала удаляем лишние _full, если оригинал существует
for filename in os.listdir(folder):
    filepath = os.path.join(folder, filename)
    if not os.path.isfile(filepath):
        continue

    name, ext = os.path.splitext(filename)
    if name.endswith("_full"):
        original_name = name[:-5] + ext
        original_path = os.path.join(folder, original_name)
        if os.path.exists(original_path):
            print(f"Удаляю дубликат: {filename}")
            os.remove(filepath)

# 2. Теперь переименовываем все, у кого нет _full
for filename in os.listdir(folder):
    filepath = os.path.join(folder, filename)
    if not os.path.isfile(filepath):
        continue

    name, ext = os.path.splitext(filename)
    if not name.endswith("_full"):
        new_name = f"{name}_full{ext}"
        new_path = os.path.join(folder, new_name)
        print(f"Переименовываю: {filename} → {new_name}")
        os.rename(filepath, new_path)
