
[USER_DEVICE_IP] (e.g., 192.168.0.100:5555), [SCREEN_RESOLUTION] (e.g., 1080x2460), [GAME_ACTIVITY] (e.g., com.pocketcombats.GameActivity), [AUTOMATION_SCOPE] (e.g., "Complete daily quests, farm resources, PvE battles").  
Разработчик создает скрипт ADB для автоматизации мобильной игры на устройстве Android (Poco X4 GT) через Wi-Fi ADB. 
Все игровые действия на данный момент
1. Проверить отсутствие оверлеев syschat and monster params, если открыты закрыть close 1, close 2, оверлеи редкость, если нажимать все верно оверлеев мы не увидим, не надо мне 10000 строк и возможностей у закрытия оверлеев
2. Привести вкладки к рабочему виду, "разбросанные вещи" закрыты, монстры" открыты 
3. Листнуть вниз
4. Нажать иконку attack
5. Перейти к бою, нажать moves, когда нажать preferred skill, если не получается, нажать обычную атаку
6. Дальше победа, и кнопка continue
7. Листнуть вниз
8. Закрыть монстры
9. Открыть вещи 
10. Выпадает текстовый список предметов, нажать на имя первого предмета, открывается описание предмета и там высвечивается кнопка "подобрать " если она активная, она коричневая и ее нужно нажать, если не активная, серая, то нужно свернуть предмет нажав на его имя, и остановить действие лутинга и перейти к следующему, если предмет был подобран активной кнопкой, то перейти к следующему до нахождения неактивной кнопки
11. Привести все окна к исходному виду
12. Уточнения: все вкладки кроме оверлеев открываются по нажатию на имя, и закрываются тоже по нажатию по имени


в соответствии с этими папками, путями, названиями
auto bot and tap detector основа нашей конфигурации
Цель: выполнить шаги 1–12 надёжно, без падений, с понятными логами в один 


 Х, и от 400 до 2100 по y, при 1080 на 2460

давай, смотри я щас проверил, те же разбросанные вещи идут до +- 500 x, а стрелка около 1000. те между 500-1000 ничего нет, а разница между предметами 150 по y, 1080 2460 разрешение экрана
1. м-о,в-з
2. м-з,в-о
3. м-з,в-з
4. м-о,в-о
насчет стрелок, close tab open tab это сама вырезка стрелок, но по ним он не может находить, поэтому в chevrons лежат 4 файла с названием вкладки и состоянием стрелки, прямоугольник вырезанный от 0 до 1080 по х, pickup own это та же кнопка что и pickup,насчет имен предметов у меня есть как "янтарь так и yantar, все кроме шевронов в папке my, игра похожа на википедию, также листается, а вкладки похожи на вкладки википедии, также на стрелочке сбоку сворачиваются и открываются, давай точечные патчи, напоминаю я тупой поэтому мне надо все максимально подробно, то что ты предложил я еще не менял мне надо больше конкретики для обьяснения
игра похожа на википедию, также листается, а вкладки похожи на вкладки википедии, также на стрелочке сбоку сворачиваются и открываются
Давай зафиксируем текущее состояние, чтобы видеть прогресс и чётко очертить, что уже реализовано, что работает неидеально, и куда двигаться.

---
{
  "DETECT_TUNING": {
    "SUPPRESS_Y": 35,
    "GROUP_THRESHOLD": 35
  },
  
  "DRY_RUN": false,
  "DEBUG": true,
  "DEBUG_LEVEL": 3,
  "DEBUG_PACK_BY_STEP": true,
  "SCREEN_W": 1080,
  "SCREEN_H": 2460,
  "SAFE_TAP_AREA": {
    "x1": 0,
    "y1": 400,
    "x2": 1080,
    "y2": 2100
  },
  
  "QUICK_TAP": {
    "X_TARGET_MODE": "absolute",
    "X_ABSOLUTE": 950,
    "X_JITTER_PX": 3,
    "Y_OFFSET_PCT": 0.20,
    "ZONE_HEIGHT_PCT": 0.10,
    "ZONE_HEIGHT_MIN_PX": 6,
    "ZONE_HEIGHT_MAX_PX": 12,
    "Y_JITTER_PX": 0,
    "TAPS": 2,
    "DELAY_BETWEEN_TAPS_MS": 40,
    "POST_SERIES_DELAY_MS": 120,
    "DEBUG_OVERLAY": true,
    "DEBUG_DIR": "debug/quick_tap"
  
  
  
  },
  
  "ITEMS_ROI": [
    0,
    400,
    500,
    2100
  ],
  "STEP_DELAY": 0.3,
  "SLEEP_AFTER_TAP": 0.4,
  "DELAY_BETWEEN_LOOPS": 5.0,
  "MATCH": {
    "tab_btn": 0.85,
    "ITEM_NAME_THRESHOLD": 0.83,
    "PICKUP_ACTIVE_THRESHOLD": 0.92,
    "PICKUP_INACTIVE_THRESHOLD": 0.94,
    "PICKUP_TPL_THRESHOLD": 0.86,
    "PICKUP_SCALES": [
      0.9,
      0.95,
      1.0,
      1.05
    ],
    "PICKUP_COLOR_SAT_MIN": 90,
    "PICKUP_COLOR_VAL_MIN": 160,
    "PICKUP_COLOR_RATIO_THRESHOLD": 0.12,
    "PICKUP_METHOD_WEIGHTS": {
      "templ": 0.6,
      "edge": 0.25,
      "color": 0.15
    }
  },
  "ROI": {
    "PICKUP_REL": [0.50, 0.17, 0.98, 0.85],
    "PICKUP_REL_CARD_FALLBACK": [0.50, 0.17, 0.98, 0.85] 
  
  
  },
  "SWIPE": {
    "DURATION_MS": 400,
    "PAUSE_MS": 500
  },
  
  
  
  "FIND": {
    "STABILIZE_FRAMES": 1,
      "DETECT_ONLY_WHITELIST": true,
      "SKIP_SUBSTR": ["tab", "chevron", "hdr", "label"],
      "EXCLUDE_ZONES": [[0, 1900, 500, 2100]]
    
    
  },
  "TIMINGS": {
    "VERIFY_ITEM_REMOVED_1_MS": 1200,
    "VERIFY_ITEM_REMOVED_2_MS": 800,
    "CARD_OPEN_DELAY_MS": 250,
    "POST_SWIPE_DELAY_MS": 250
    
  },
  "ORDER": {
    "RESCAN_AFTER_PICKUP": true
  },
  "LOGIC": {
    "ALLOW_PICKUP_OTHER": false,
    "ABORT_ON_FIRST_INACTIVE": true
  },
  "ALLOWED_ITEM_NAMES": [
    "goreloe_derevo.png",
    "goreloe_derevo_1.png",
    "latnye_perchatki.png",
    "latnye_perchatki_1.png",
    "pautina.png",
    "pautina_1.png",
    "runa_nachitannosti.png",
    "runa_nachitannosti_1.png",
    "sgorevshaya_spichka.png",
    "sgorevshaya_spichka_1.png",
    "ugol.png",
    "yantar.png",
    "yantar_1.png",
    "zelenye_yagody.png",
    "zheleznaya_ruda.png",
    "zheleznaya_ruda_1.png",
    "горелое_дерево.png",
    "горелый_посох.png",
    "зеленые_ягоды.png",
    "латные_перчатки.png",
    "паутина.png",
    "руна_начитанности.png",
    "сгоревшая_спичка.png",
    "тяжелый_цеп.png",
    "уголь.png",
    "янтарь.png"
  ],
  "TEMPLATES": {
    "PICKUP_ACTIVE": [
      "C:/bot/tpl/templates/pickup_active.png"
    ],
    "PICKUP_INACTIVE": "C:/bot/tpl/templates/pickup_inactive.png",
    "SYSCHAT": "C:/bot/tpl/templates/syschat.png",

  "SKIP_ITEM_NAMES": [],
  "WHITELIST_MANAGER": {},
  
    "PICKUP_INACTIVE_THRESHOLD": 0.94,
    "VERIFY_ITEM_REMOVED_1_MS": 1600,
    "VERIFY_ITEM_REMOVED_2_MS": 1000,
    "item_group_threshold": 75

  },
  "OVERLAYS": {
      "THRESHOLDS": {},
      "CALL_SITES": ["on_enter_fight","each_fight_click","on_victory","before_loot"]
  },
    "MAIN": {
      "MAX_LOOPS": 999,
      "DELAY_SEC": 0.5
    },
    "FIGHT": {
      "TIMINGS": {
        "after_click": 0.4,
        "after_victory": 0.6,
        "loop_tick": 0.3
      },
      "THRESHOLDS": {
        "moves_btn": 0.82,
        "skill_btn": 0.82,
        "skill_text_btn": 0.82,
        "attack_btn": 0.82,
        "victory_icon": 0.82,
        "continue_btn": 0.82,
        "close_btn": 0.82
      },
      "TIMEOUTS": {
        "battle_start": 10
      }
    }
    
       
    
  
  
}



