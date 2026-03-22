# План оптимизации Game Bot — Полный план работы

> Каждый пункт описан максимально подробно для автономного исполнения.

---

## 1. Исправить двойные значения конфига

**Проблема:** На уровне модуля [auto_bot.py](file:///c:/bot/script/loot/auto_bot.py) (строки 87-362) значения из `CFG` замораживаются в глобальные переменные (`FIND_THRESHOLD`, `PICKUP_SCALES` и т.д.). Но внутри функции [_raw_find_item_names()](file:///c:/bot/script/loot/auto_bot.py#1062-1243) (строки 1129-1136) те же значения перечитываются из `CFG` заново. Получается **два источника правды**.

**Решение:**
1. Открыть [auto_bot.py](file:///c:/bot/script/loot/auto_bot.py)
2. В [_raw_find_item_names()](file:///c:/bot/script/loot/auto_bot.py#1062-1243) убрать повторные `CFG["FIND"].get(...)` и заменить на глобальные переменные, определённые в начале модуля
3. Аналогично для `grp_thr` (строка 1137 и 1193) — есть две дублирующих инициализации
4. Аналогично для `max_lines` (строка 1138 и 1196)
5. Проверить все остальные функции на аналогичный паттерн (`CTRL+F` по `CFG.get` и `CFG[`)
6. Убедиться что значения в начале модуля совпадают с дефолтами внутри функций

**Файлы:** [auto_bot.py](file:///c:/bot/script/loot/auto_bot.py)
**Тест:** `py_compile`, затем grep по `CFG.get` — не должно быть дубликатов с глобальными переменными

---

## 2. Ускорение получения скриншотов

**Проблема:** `adb exec-out screencap -p` → **300-500мс** на скриншот. В одном цикле 5-10 скриншотов = 2-5 секунд чистых задержек.

**Варианты решения (реализовать оба, выбирать через конфиг):**

### Вариант A: BlueStacks / эмулятор на ПК
1. Создать файл `script/screen/capture.py` с абстрактным интерфейсом:
   ```
   class ScreenCapture(ABC):
       def grab(self) -> np.ndarray | None: ...
   ```
2. Реализация `AdbCapture` — текущий метод через ADB (`adb exec-out screencap -p`)
3. Реализация `WindowCapture` — захват окна BlueStacks через Win32 API:
   - Использовать `pywin32` (`win32gui.FindWindow`, `win32ui.CreateDCFromHandle`)
   - Или `mss` (pip install mss) — быстрее, ~20-30мс на скриншот
   - Находить окно по заголовку (настраивается в конфиге: `"capture_window_title": "BlueStacks"`)
4. В [config.json](file:///c:/bot/%D0%A1%D0%BA%D0%B0%D0%BD%D0%B5%D1%80/config.json) добавить: `"CAPTURE_METHOD": "adb"` или `"window"`
5. В [auto_bot.py](file:///c:/bot/script/loot/auto_bot.py) заменить все вызовы [screenshot_bgr()](file:///c:/bot/script/loot/auto_bot.py#610-628) на `capture.grab()`

### Вариант B: Потоковый ADB (если остаёмся на телефоне)
1. Вместо одиночных `adb exec-out screencap` использовать `scrcpy --no-display --record` или `adb exec-out screenrecord --output-format=h264 -` с декодированием через ffmpeg
2. Реализация `StreamCapture` — читает последний фрейм из потока
3. Фоновый поток непрерывно читает фреймы, основной поток берёт последний

**Файлы:** Новый `script/screen/capture.py`, правки [auto_bot.py](file:///c:/bot/script/loot/auto_bot.py) (заменить [screenshot_bgr](file:///c:/bot/script/loot/auto_bot.py#610-628))
**Конфиг:** `"CAPTURE_METHOD": "adb" | "window" | "stream"`
**Ожидаемый результат:** с 300-500мс до 20-50мс на скриншот (при Window capture)

---

## 3. Кэширование шаблонов в памяти

**Проблема:** [get_item_name_templates()](file:///c:/bot/script/loot/auto_bot.py#127-196) каждый раз вызывает [imread_u8()](file:///c:/bot/script/loot/utils.py#108-127) для каждого шаблона. При 30 шаблонах — 30 чтений с диска каждый цикл.

**Решение:**
1. В [auto_bot.py](file:///c:/bot/script/loot/auto_bot.py) создать модульный кэш:
   ```
   _ITEM_TPL_CACHE: Dict[str, Tuple[str, np.ndarray, str]] = {}
   _ITEM_TPL_CACHE_HASH: int = 0  # хеш от списка файлов для инвалидации
   ```
2. В [get_item_name_templates()](file:///c:/bot/script/loot/auto_bot.py#127-196):
   - Построить хеш от `ALLOWED_LOOT_DIRS` + список файлов (имена + mtime)
   - Если хеш совпадает с `_ITEM_TPL_CACHE_HASH` → вернуть кэш
   - Если нет → загрузить с диска, обновить кэш
3. Аналогично для [tpl_loader.py](file:///c:/bot/script/loot/tpl_loader.py) → [load_item_name_templates_unified()](file:///c:/bot/script/loot/tpl_loader.py#265-315) — добавить LRU-кэш
4. Кэшировать **и масштабированные версии** шаблонов (пункт 8) здесь же

**Файлы:** [auto_bot.py](file:///c:/bot/script/loot/auto_bot.py), [tpl_loader.py](file:///c:/bot/script/loot/tpl_loader.py)
**Тест:** Замерить время [get_item_name_templates()](file:///c:/bot/script/loot/auto_bot.py#127-196) до и после (должно быть ~0мс на повторный вызов vs ~50-100мс)

---

## 4. Разбить [auto_bot.py](file:///c:/bot/script/loot/auto_bot.py) на модули

**Проблема:** 1600+ строк, делает всё. Невозможно тестировать и поддерживать.

**План разбиения:**

### 4.1 `script/device/adb.py` (~150 строк)
Переместить:
- [adb_cmd()](file:///c:/bot/script/loot/auto_bot.py#599-608) — выполнение ADB-команд
- [screenshot_bgr()](file:///c:/bot/script/loot/auto_bot.py#610-628) — захват экрана (или обёртка над capture.py)
- [tap()](file:///c:/bot/script/loot/auto_bot.py#630-644), [tap_raw()](file:///c:/bot/script/loot/auto_bot.py#646-659) — тапы
- [device_swipe()](file:///c:/bot/script/loot/auto_bot.py#675-781), `smart_human_swipe()` — свайпы
- [super_swipe_before_loot()](file:///c:/bot/script/loot/auto_bot.py#444-454) — свайп перед лутом
- Константы: `DEVICE_ID`, `MAX_SCREENSHOT_ATTEMPTS`

### 4.2 `script/detection/items.py` (~300 строк)
Переместить:
- [get_item_name_templates()](file:///c:/bot/script/loot/auto_bot.py#127-196) — загрузка шаблонов
- [_collect_hits()](file:///c:/bot/script/loot/auto_bot.py#1026-1045) — обработка совпадений
- [_raw_find_item_names()](file:///c:/bot/script/loot/auto_bot.py#1062-1243) — поиск имён предметов
- [find_item_names()](file:///c:/bot/script/loot/auto_bot.py#1245-1290) — обёртка со стабилизацией
- [compute_items_visible_roi()](file:///c:/bot/script/loot/auto_bot.py#432-442) — ROI для списка предметов
- [merge_same_lines()](file:///c:/bot/script/loot/auto_bot.py#1047-1060) — группировка хитов
- [match_multi_scaled()](file:///c:/bot/script/loot/auto_bot.py#457-512) — мульти-масштабное сопоставление
- Константы: `FIND_THRESHOLD`, `FIND_SCALES`, `SUPPRESS_Y`, `ALLOWED_ITEM_NAMES`, `SERVICE_SKIP`

### 4.3 `script/detection/pickup.py` (~200 строк)
Переместить:
- `detect_pickup_combined()` — комбинированная детекция кнопки подбора
- [verify_item_removed()](file:///c:/bot/script/loot/auto_bot.py#1353-1372) — верификация подбора
- [quick_double_tap_from_slot()](file:///c:/bot/script/loot/auto_bot.py#1438-1451) — быстрый двойной тап
- Константы: `PICKUP_*` пороги

### 4.4 `script/core/logging.py` (~80 строк)
Переместить:
- [log()](file:///c:/bot/script/loot/utils.py#68-75), [structured_log()](file:///c:/bot/script/loot/auto_bot.py#587-594) — логирование
- [snap()](file:///c:/bot/script/loot/auto_bot.py#554-574), [dbg_name()](file:///c:/bot/script/loot/auto_bot.py#527-531), [now_ts()](file:///c:/bot/script/loot/auto_bot.py#245-247) — дебаг-снапшоты
- [ensure_dirs()](file:///c:/bot/script/loot/utils.py#57-59) — создание директорий
- Константы: `LOG_DIR`, `LOG_FILE`, `DEBUG_DIR`

### 4.5 [script/loot/auto_bot.py](file:///c:/bot/script/loot/auto_bot.py) (~остаток, ~400 строк)
Оставить:
- [auto_loot_once()](file:///c:/bot/script/loot/auto_bot.py#1374-1575) — главная функция одного цикла лута
- [ensure_tab_state()](file:///c:/bot/script/loot/auto_bot.py#1292-1351) — управление табами
- [collapse_if_open()](file:///c:/bot/script/loot/auto_bot.py#1004-1013) — сворачивание панели
- [set_victory_targets()](file:///c:/bot/script/loot/auto_bot.py#370-374) / [get_victory_targets()](file:///c:/bot/script/loot/auto_bot.py#376-378) — фильтр победы
- Импорты из новых модулей

**Порядок работы:**
1. Создать файлы-заглушки со всеми импортами
2. Перемещать функции по одной, запуская `py_compile` после каждой
3. В [auto_bot.py](file:///c:/bot/script/loot/auto_bot.py) добавить ре-экспорты (`from script.device.adb import tap, screenshot_bgr`) для обратной совместимости
4. Убедиться что [fsm_main.py](file:///c:/bot/script/main/fsm_main.py) и другие потребители не сломались

---

## 5. Вынести все магические числа в конфиг

**Проблема:** Пороги вроде `0.82`, `0.86`, `0.83` разбросаны по коду.

**Решение:**
1. Собрать ВСЕ хардкоженные числа из [auto_bot.py](file:///c:/bot/script/loot/auto_bot.py), [tab_detector.py](file:///c:/bot/script/loot/tab_detector.py), [victory_drop.py](file:///c:/bot/script/loot/victory_drop.py), [clear_overlays.py](file:///c:/bot/script/overlays/clear_overlays.py), [fight.py](file:///c:/bot/script/fight/fight.py), [quick_tap.py](file:///c:/bot/script/loot/quick_tap.py)
2. Для каждого определить: имя, значение, описание
3. В [config.json](file:///c:/bot/%D0%A1%D0%BA%D0%B0%D0%BD%D0%B5%D1%80/config.json) добавить секцию `"THRESHOLDS"`:
   ```json
   "THRESHOLDS": {
     "SCHARR_FALLBACK_OFFSET": 0.03,
     "SCHARR_FALLBACK_MIN": 0.83,
     "TAB_DETECT_MIN": 0.85,
     "OVERLAY_TEXT_THR": 0.7,
     ...
   }
   ```
4. В [utils.py](file:///c:/bot/script/loot/utils.py) добавить хелпер [thr(name, default)](file:///c:/bot/script/fight/fight.py#33-35) для чтения с дефолтом
5. Заменить все хардкоженные числа на [thr("SCHARR_FALLBACK_MIN", 0.83)](file:///c:/bot/script/fight/fight.py#33-35) и т.д.

**Файлы:** Все скрипты + [config.json](file:///c:/bot/%D0%A1%D0%BA%D0%B0%D0%BD%D0%B5%D1%80/config.json)
**Тест:** `grep -rn "0\.\d\d" script/` — не должно остаться хардкода порогов

---

## 6. Буферизованный structured_log

**Проблема:** Каждый [structured_log()](file:///c:/bot/script/loot/auto_bot.py#587-594) = open → write → close. Десятки I/O операций в секунду.

**Решение:**
1. Создать класс `BufferedStructuredLogger`:
   ```
   class BufferedStructuredLogger:
       _buffer: List[dict]
       _flush_interval: float  # секунды (из конфига, дефолт 2.0)
       _max_buffer_size: int   # максимум записей (дефолт 50)
       _last_flush: float
   ```
2. Метод [log(event, **kwargs)](file:///c:/bot/script/loot/utils.py#68-75): добавляет в буфер, проверяет flush_interval и max_buffer_size
3. Метод `flush()`: пишет весь буфер в файл одним вызовом, очищает буфер
4. Метод `__del__()` / `atexit.register()`: flush при завершении
5. Заменить глобальную функцию [structured_log](file:///c:/bot/script/loot/auto_bot.py#587-594) на экземпляр этого класса
6. Добавить в конфиг: `"LOG_FLUSH_INTERVAL_S": 2.0`, `"LOG_MAX_BUFFER": 50`

**Файлы:** `script/core/logging.py` (новый), [auto_bot.py](file:///c:/bot/script/loot/auto_bot.py)
**Тест:** Замерить время 100 вызовов [structured_log](file:///c:/bot/script/loot/auto_bot.py#587-594) до и после

---

## 7. Оптимизация двойного поиска (основной + Scharr)

**Проблема:** При неудаче основного поиска запускается Scharr-fallback по всем шаблонам — двойная работа.

**Решение:**
1. Перед Scharr-fallback проверить, есть ли вообще текст в ROI:
   - `mean_brightness = np.mean(proc)` — если < 10, экран пустой, не искать
   - `edge_density = np.count_nonzero(_scharr_mag(proc) > 30) / proc.size` — если < 0.01, нет текста
2. Кэшировать `_scharr_mag(proc)` — он уже считается, но внутри цикла для каждого шаблона заново вычисляется `_scharr_mag(tpl)`. Вынести `_scharr_mag(tpl)` в кэш шаблонов (пункт 3)
3. В конфиге добавить `"SCHARR_FALLBACK_ENABLED": true` чтобы можно было отключить

**Файлы:** [auto_bot.py](file:///c:/bot/script/loot/auto_bot.py) (функция [_raw_find_item_names](file:///c:/bot/script/loot/auto_bot.py#1062-1243))

---

## 8. Предвычисление масштабированных шаблонов

**Проблема:** [match_scaled()](file:///c:/bot/script/loot/utils.py#187-213) вызывает `cv2.resize()` для каждого масштаба при каждом вызове.

**Решение:**
1. При загрузке шаблона (пункт 3, кэш) предвычислить все масштабированные gray-версии:
   ```
   for scale in FIND_SCALES:
       scaled = cv2.resize(gray, (int(w*s), int(h*s)), ...)
       cache[name]["scaled"][scale] = scaled
       cache[name]["scharr_scaled"][scale] = _scharr_mag(scaled)  # пункт 7
   ```
2. В [find_all_matches](file:///c:/bot/script/loot/matcher.py#50-89) и [match_scaled](file:///c:/bot/script/loot/utils.py#187-213) принимать предвычисленный шаблон вместо пересчёта
3. Инвалидация: при изменении `FIND_SCALES` в конфиге

**Файлы:** [auto_bot.py](file:///c:/bot/script/loot/auto_bot.py), [matcher.py](file:///c:/bot/script/loot/matcher.py), [utils.py](file:///c:/bot/script/loot/utils.py)
**Ожидание:** Экономия ~50% времени на template matching (resize — дорогая операция)

---

## 9. [frame_hash](file:///c:/bot/script/loot/auto_bot.py#783-791) через numpy

**Проблема:** Python-цикл `for i, v in enumerate(diff.flatten())` ~72 итерации — медленно.

**Решение:**
Заменить текущую реализацию:
```python
def frame_hash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    # Вместо Python-цикла:
    bits = np.packbits(diff.flatten(), bitorder='little')
    return int.from_bytes(bits.tobytes(), byteorder='little')
```

**Файлы:** [auto_bot.py](file:///c:/bot/script/loot/auto_bot.py) (функция [frame_hash](file:///c:/bot/script/loot/auto_bot.py#783-791), строка ~783)
**Тест:** Сравнить результат старой и новой версии на 10 тестовых изображениях — должны совпадать. Замерить скорость — ожидание 50-100x ускорение.

---

## 10. Быстрый предварительный фильтр шаблонов

**Проблема:** 30 шаблонов × `cv2.matchTemplate()` = 30 тяжёлых вызовов.

**Решение:**
1. Добавить быстрый pre-filter перед полным поиском:
   - Уменьшить ROI и шаблон в 4x (`cv2.resize(..., (w//4, h//4))`)
   - Запустить `matchTemplate` на уменьшенных — это в 16x быстрее
   - Если `max_val < threshold - 0.1`, шаблон точно не совпадёт → пропустить
2. Полный поиск запускать только для шаблонов, прошедших pre-filter
3. Добавить в конфиг: `"PREFILTER_ENABLED": true`, `"PREFILTER_SCALE": 0.25`, `"PREFILTER_MARGIN": 0.1`

**Файлы:** [auto_bot.py](file:///c:/bot/script/loot/auto_bot.py) (или `script/detection/items.py` если уже разбили)
**Ожидание:** При 30 шаблонах, если 25 из них "не в кадре" — экономия ~80% времени поиска

---

## 11. Graceful shutdown

**Проблема:** CTRL+C обрывает бота без сохранения состояния.

**Решение:**
1. В [fsm_main.py](file:///c:/bot/script/main/fsm_main.py) обернуть главный цикл:
   ```python
   import signal
   _shutdown = False
   def _handle_signal(sig, frame):
       global _shutdown
       _shutdown = True
       log("[SHUTDOWN] Получен сигнал, завершаюсь после текущего цикла...")
   signal.signal(signal.SIGINT, _handle_signal)
   signal.signal(signal.SIGTERM, _handle_signal)
   ```
2. В главном цикле проверять `if _shutdown: break`
3. После выхода из цикла:
   - `structured_log.flush()` (пункт 6)
   - Сохранить текущее состояние FSM и slot_lifecycle в файл
   - Логировать `"bot_shutdown_clean"`

**Файлы:** [fsm_main.py](file:///c:/bot/script/main/fsm_main.py)

---

## 12. ADB retry с exponential backoff

**Проблема:** [adb_cmd()](file:///c:/bot/script/loot/auto_bot.py#599-608) — один try, один таймаут. ADB часто зависает.

**Решение:**
1. Обернуть [adb_cmd()](file:///c:/bot/script/loot/auto_bot.py#599-608) в retry-логику:
   ```
   max_retries: 3 (из конфига)
   backoff: 0.5s, 1.0s, 2.0s
   ```
2. При `TimeoutExpired` — убить процесс ADB, подождать backoff, повторить
3. При 3 неудачах подряд — вызвать `adb kill-server && adb start-server` и повторить
4. [screenshot_bgr()](file:///c:/bot/script/loot/auto_bot.py#610-628) — отдельный retry: при пустом/битом RAW — повторить без полного backoff
5. Добавить в конфиг: `"ADB_MAX_RETRIES": 3`, `"ADB_BACKOFF_BASE_S": 0.5`

**Файлы:** [auto_bot.py](file:///c:/bot/script/loot/auto_bot.py) (или `script/device/adb.py` если разбили)

---

## 13. Метрики производительности

**Проблема:** Нет замеров времени, непонятно что тормозит.

**Решение:**
1. Создать простой контекстный менеджер:
   ```python
   class PerfTimer:
       _timings: Dict[str, List[float]] = defaultdict(list)
       
       @contextmanager
       def measure(self, label: str):
           t0 = time.perf_counter()
           yield
           self._timings[label].append(time.perf_counter() - t0)
       
       def report(self) -> Dict[str, dict]:
           # avg, min, max, count для каждого label
   ```
2. Обернуть ключевые операции:
   - [screenshot_bgr()](file:///c:/bot/script/loot/auto_bot.py#610-628) → `"screenshot"`
   - [find_all_matches()](file:///c:/bot/script/loot/matcher.py#50-89) → `"template_match"`
   - [_raw_find_item_names()](file:///c:/bot/script/loot/auto_bot.py#1062-1243) → `"item_search"`
   - `detect_pickup_combined()` → `"pickup_detect"`
   - [tap()](file:///c:/bot/script/loot/auto_bot.py#630-644) → `"tap"`
   - `structured_log.flush()` → `"log_flush"`
3. Каждые N циклов (конфиг: `"PERF_REPORT_EVERY_N": 10`) выводить отчёт в лог
4. Добавить в конфиг: `"PERF_ENABLED": true`

**Файлы:** `script/core/perf.py` (новый), правки во всех модулях
**Результат:** В логе появятся строки типа `"perf_report": {"screenshot": {"avg": 0.35, "count": 47}, ...}`

---

## 14. Recovery при неизвестном экране

**Проблема:** Если игра показывает рекламу, дисконнект, краш — бот зависает.

**Решение:**
1. В [fsm_main.py](file:///c:/bot/script/main/fsm_main.py) добавить новое состояние `STATE_UNKNOWN = "unknown_screen"`
2. Отслеживать таймер бездействия:
   - Если бот в одном состоянии дольше `MAX_STATE_DURATION_S` (конфиг, дефолт 60с) — переход в `STATE_UNKNOWN`
3. В `STATE_UNKNOWN`:
   - Сделать скриншот и сохранить в `debug/unknown_screens/`
   - Попробовать закрыть оверлей ([clear_overlays](file:///c:/bot/script/overlays/clear_overlays.py#58-85))
   - Попробовать нажать Back (`adb shell input keyevent 4`)
   - Подождать 3 секунды
   - Сделать новый скриншот
   - Если экран изменился — вернуться в `STATE_FIND`
   - Если нет — повторить ещё 2 раза
   - Если после 3 попыток не помогло — логировать `"bot_stuck"` и подождать 30с
4. Добавить в конфиг: `"MAX_STATE_DURATION_S": 60`, `"UNKNOWN_SCREEN_MAX_RETRIES": 3`

**Файлы:** [fsm_main.py](file:///c:/bot/script/main/fsm_main.py)

---

## Порядок выполнения (рекомендуемый)

| # | Пункт | Зависимости | Сложность |
|---|-------|-------------|-----------|
| 1 | Двойные значения конфига | — | 🟢 Лёгкая |
| 9 | frame_hash через numpy | — | 🟢 Лёгкая |
| 5 | Магические числа в конфиг | — | 🟡 Средняя |
| 6 | Буферизованный лог | — | 🟡 Средняя |
| 3 | Кэш шаблонов | — | 🟡 Средняя |
| 11 | Graceful shutdown | 6 | 🟢 Лёгкая |
| 12 | ADB retry | — | 🟡 Средняя |
| 13 | Метрики | 6 | 🟡 Средняя |
| 7 | Оптимизация Scharr | 3 | 🟡 Средняя |
| 8 | Предвычисление масштабов | 3 | 🟡 Средняя |
| 10 | Pre-filter шаблонов | 3, 8 | 🟡 Средняя |
| 14 | Recovery unknown screen | — | 🟡 Средняя |
| 2 | Быстрый захват экрана | — | 🔴 Сложная |
| 4 | Разбить auto_bot.py | 1-3, 5-6 | 🔴 Сложная |

> [!IMPORTANT]
> Пункт 4 (разбиение) лучше делать ПОСЛЕДНИМ — после всех остальных оптимизаций. Иначе каждое изменение придётся вносить сразу в несколько файлов.
