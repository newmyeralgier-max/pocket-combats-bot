# pocket-combats-bot

Бот для PocketCombats (FSM: FIND → FIGHT → LOOT → RECOVER) на Python/OpenCV.
Распознавание через template matching на скриншотах устройства; управление через ADB.

## Структура репо

```
script/
├── main/fsm_main.py       # точка входа, главный FSM-цикл
├── core/                  # конфиг, логирование, замеры перфоманса
├── device/adb.py          # тапы/свайпы/скриншоты через ADB
├── screen/capture.py      # абстракция ScreenCapture: AdbCapture / WindowCapture
├── detection/             # поиск предметов и кнопки подбора
├── loot/                  # основной loot-цикл, шаблоны, вкладки
├── fight/                 # бой: MOVES → навык → victory/defeat
└── overlays/              # закрытие всплывающих окон

tools/cfg/
├── config.json            # главный конфиг (пороги, ROI, пути к шаблонам)
├── templates_registry.json# реестр шаблонов (path → group → key)
└── *.json                 # whitelists / not_* отчёты

tpl/                       # PNG-шаблоны (в git не хранятся — слишком большие)
debug/                     # снимки для отладки, создаётся автоматически
```

## Установка

1. Склонируй репо в любую папку (не обязательно `C:\bot` — путь теперь автодетектится).
2. Положи рядом каталог `tpl/` со всеми шаблонами. Структура должна совпадать со ссылками в `tools/cfg/templates_registry.json`:
   - `tpl/имя_предметов/*.png`
   - `tpl/иконки_предметов/*.png`
   - `tpl/имя_рун/*.png`, `tpl/иконки_рун/*.png`
   - `tpl/служебные/*.png` (UI: `attack.png`, `continue.png`, `victory.png`, `items_open.png` и т.д.)
3. Установи зависимости:
   ```bash
   pip install -r requirements.txt
   ```

### Пути к шаблонам

Старые конфиги содержат абсолютные пути вида `C:/bot/tpl/...`. Они продолжают работать: при загрузке конфига префикс `C:/bot/` автоматически заменяется на актуальный корень репозитория (детектится по расположению `script/loot/utils.py`).

Можно явно задать корень через переменную окружения:
```bash
# Linux / macOS
export POCKET_BOT_ROOT=/home/user/pocket-combats-bot

# Windows PowerShell
$env:POCKET_BOT_ROOT = "D:\games\pocket-combats-bot"
```

В новых конфигах можно использовать плейсхолдер `${BOT_ROOT}`:
```json
"ATTACK": "${BOT_ROOT}/tpl/служебные/attack.png"
```

## Запуск

### Телефон по ADB (вариант по умолчанию)

1. Подключи телефон через USB / Wi-Fi ADB (`adb devices` должен видеть устройство).
2. В `tools/cfg/config.json`:
   ```json
   "CAPTURE_METHOD": "adb",
   "device_id": null
   ```
3. Старт:
   ```bash
   python -m script.main.fsm_main
   ```

### Эмулятор на ПК (BlueStacks / MuMu / LDPlayer) — быстрее в 5–10 раз

ADB screencap даёт 300–500 мс на кадр. Захват окна эмулятора через `mss` — 20–50 мс. Это самое дешёвое ускорение в репо.

1. В `tools/cfg/config.json` добавь:
   ```json
   "CAPTURE_METHOD": "window",
   "CAPTURE_WINDOW_TITLE": "BlueStacks App Player"
   ```
   (подставь реальный заголовок окна эмулятора).
2. Для тапов всё равно используется ADB — подключись к эмулятору:
   ```bash
   adb connect 127.0.0.1:5555
   ```
3. Старт как обычно: `python -m script.main.fsm_main`.

## Конфиг — ключевые секции

| ключ | смысл |
|---|---|
| `CAPTURE_METHOD` | `adb` / `window` / `stream` |
| `SCREEN_W`, `SCREEN_H` | физическое разрешение устройства |
| `SAFE_TAP_AREA` | зона, куда можно тапать |
| `FIND.THRESHOLD` | порог сходства для matchTemplate по именам предметов |
| `FIND.ITEM_SCALES` | масштабы поиска |
| `ALLOWED_ITEM_NAMES` | whitelist имён (без этого бот ничего не возьмёт) |
| `ALLOWED_LOOT_DIRS` | где лежат шаблоны имён для подбора |
| `MATCH.PICKUP_*` | пороги кнопки «Подобрать» |
| `FIGHT.SKILLS` | список навыков, которыми бить |
| `FIGHT.THRESHOLDS` | пороги для ui-кнопок боя |
| `THRESHOLDS` | общие пороги (шевроны табов, чата и т.д.) |
| `MAIN.MAX_LOOPS` | ограничение FSM по итерациям |

## Отладка

- Все debug-снимки пишутся в `debug/` (включается `DEBUG: true`, `save_debug_images: true`).
- Структурированный JSON-лог пишется буферизованно (см. `BufferedStructuredLogger`) в файл из `LOG_FILE`.
- Для профилирования горячих точек:
  ```json
  "PERF_REPORT_EVERY_N": 10
  ```
  — раз в N циклов лутания в лог уходит `perf_report` с таймингами.

## Частые грабли

- **«Бот ничего не видит»** — проверь, что папка `tpl/` лежит в корне репо и шрифт/размер в игре совпадают с тем, с которого делались шаблоны. Любое изменение DPI / темы ломает `matchTemplate`.
- **«ADB screencap зависает»** — перезапусти `adb kill-server && adb start-server`, либо перейди на эмулятор + WindowCapture.
- **«Победил, но не залутал»** — проверь, что в `tools/cfg/config.json` секция `ALLOWED_ITEM_NAMES` содержит нужные `*_icon.png` имена. Бот ищет только их на экране победы.
- **«Разрешение не 1080×2460»** — координаты `SCREEN_W/SCREEN_H`, `ITEMS_ROI`, `ROI.VICTORY_DROP_ABS` и тапы `tap_raw(540, 2250)` жёстко привязаны к этому разрешению. Нужно пересобрать шаблоны и ROI под своё.
