import json

import requests
from bs4 import BeautifulSoup

urls = [
    "https://pocketcombats.fandom.com/ru/wiki/Предметы_и_лут",
    "https://pocketcombats.fandom.com/ru/wiki/Категория:Трофеи",
    "https://pocketcombats.fandom.com/ru/wiki/Категория:Расходные_материалы",
]

items = []

for url in urls:
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")

    # Ищем ссылки на страницы предметов
    for a in soup.select("a.category-page__member-link, a[href*='/wiki/']"):
        name = a.get_text(strip=True)
        if name and not name.startswith("Категория:"):
            # Приводим к формату *_*
            formatted = name.replace(" ", "_")
            if formatted not in items:
                items.append(formatted)

# Сохраняем в JSON
with open("items.json", "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=2)

print(f"Сохранено {len(items)} предметов в items.json")
