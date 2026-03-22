import json
import re
from pathlib import Path

# === НАСТРОЙКИ ===
mode = "all"  # "json", "files" или "all"
file_ext = ".md"  # расширение создаваемых файлов

# Пути
base_path = Path(r"C:\bot\tools\obsidian")
txt_file = base_path / "obsidian.txt"
json_file = base_path / "obsidian.json"

# Папка для создания файлов
output_dir = Path(r"C:\Obsidian\New-Life\Учеба\Темы для изучения")


def parse_terms_from_txt(file_path):
    """Читает txt и возвращает список уникальных терминов (RU+EN), без дублей по регистру."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Паттерн: слово с заглавной буквы (RU или EN) + возможные слова с маленькой буквы
    pattern = r"(?:[А-ЯЁ][а-яё]+|[A-Z][a-z]+)(?:\s(?:[а-яё]+|[a-z]+))*"
    found_terms = re.findall(pattern, text)

    # Убираем пустые, короткие и дубликаты (без учёта регистра)
    seen_lower = set()
    terms = []
    for term in (t.strip() for t in found_terms if len(t.strip()) > 1):
        lower_term = term.lower()
        if lower_term not in seen_lower:
            seen_lower.add(lower_term)
            terms.append(term)

    return terms


def save_json(terms, file_path):
    """Сохраняет список терминов в JSON."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(terms, f, ensure_ascii=False, indent=4)
    print(f"JSON создан: {file_path} ({len(terms)} терминов)")


def load_json(file_path):
    """Загружает список терминов из JSON."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_files(terms, folder, ext):
    """Создаёт файлы по списку терминов, пропуская уже существующие."""
    folder.mkdir(parents=True, exist_ok=True)
    created_count = 0
    skipped_count = 0

    for term in terms:
        safe_name = "".join(c for c in term if c not in r'\/:*?"<>|').strip()
        file_path = folder / f"{safe_name}{ext}"
        if file_path.exists():
            skipped_count += 1
            continue
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# {term}\n")
        created_count += 1

    print(f"Создано {created_count} файлов, пропущено {skipped_count} (уже существовали)")


# === ЛОГИКА РЕЖИМОВ ===
if mode == "json":
    terms = parse_terms_from_txt(txt_file)
    save_json(terms, json_file)

elif mode == "files":
    if not json_file.exists():
        print("Ошибка: JSON-файл не найден. Сначала запусти в режиме 'json' или 'all'.")
    else:
        terms = load_json(json_file)
        create_files(terms, output_dir, file_ext)

elif mode == "all":
    terms = parse_terms_from_txt(txt_file)
    save_json(terms, json_file)
    create_files(terms, output_dir, file_ext)

else:
    print("Ошибка: неизвестный режим. Используй 'json', 'files' или 'all'.")
