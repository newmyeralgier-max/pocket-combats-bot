import json
import os

import cv2
from detect_states_v4 import (
    annotate,
    classify_by_y_band,
    detect_headers,
    parse_scales,
    save_json,
)


def main():
    img_path = "C:/bot/screens/lobby_2.png"
    tmpl_dir = "C:/bot/tpl/my"
    out_dir = "C:/bot/out"
    templates = {
        "вещи": cv2.imread(os.path.join(tmpl_dir, "items_header.png")),
        "монстры": cv2.imread(os.path.join(tmpl_dir, "monster_tab.png")),
        "стрелка_вверх": cv2.imread(os.path.join(tmpl_dir, "open_tab.png")),
        "стрелка_вниз": cv2.imread(os.path.join(tmpl_dir, "close_tab.png")),
    }
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Не найден скрин: {img_path}")
    scales = parse_scales("0.95,1.0,1.05")
    threshold = 0.88
    tolerance = 20
    states = detect_headers(img, templates, threshold=threshold, scales=scales)
    active, inactive, y_med = classify_by_y_band(
        {k: v for k, v in states.items() if k in ["вещи", "монстры"]}, y_band_tolerance=tolerance
    )
    result = {}
    for name in ["вещи", "монстры"]:
        found = states.get(name, {}).get("found", False)
        tab_center = states.get(name, {}).get("center", [None, None])
        tab_y = tab_center[1] if tab_center[1] else None
        arrow_up = states.get("стрелка_вверх", {}).get("center", [None, None])[1]
        arrow_down = states.get("стрелка_вниз", {}).get("center", [None, None])[1]
        if not found or tab_y is None:
            result[name] = "не найдено"
        elif arrow_up and abs(arrow_up - tab_y) <= tolerance:
            result[name] = "открыта"
        elif arrow_down and abs(arrow_down - tab_y) <= tolerance:
            result[name] = "закрыта"
        else:
            result[name] = "не определено"
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(img_path))[0]
    ann_path = os.path.join(out_dir, f"{base}_annotated.png")
    json_path = os.path.join(out_dir, f"{base}_log.json")
    annotate(img, states, active, ann_path, y_med=y_med, y_tol=tolerance)
    log = {
        "изображение": img_path,
        "шаблоны": list(templates.keys()),
        "порог": threshold,
        "масштабы": scales,
        "состояния": states,
        "видимые вкладки (по Y)": active,
        "результаты": result,
    }
    save_json(json_path, log)
    print(f"\n🟢 Результаты:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print(f"\n📁 Аннотация сохранена: {ann_path}")
    print(f"🧾 Лог сохранён: {json_path}")


if __name__ == "__main__":
    main()
