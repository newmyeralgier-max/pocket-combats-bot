from pathlib import Path

import cv2
import numpy as np

ROOT = Path("C:\\bot")
IN_DIR = ROOT / "runs" / "input"
OUT_DIR = ROOT / "tpl" / "my"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_title(frame):
    h, w = frame.shape[:2]
    x = int(w * 0.08)
    y = int(h * 0.18)
    ww = int(w * 0.84)
    hh = int(h * 0.28)
    crop = frame[y : y + hh, x : x + ww]
    tx = int(ww * 0.04)
    ty = int(hh * 0.04)
    tw = int(ww * 0.92)
    th = int(hh * 0.2)
    title = crop[ty : ty + th, tx : tx + tw]
    return title


def normalize(name):
    return name.strip().lower().replace(" ", "_").replace("ё", "е")


def main():
    files = sorted(IN_DIR.glob("*"))
    if not files:
        print("❌ Нет входных скринов в C:/bot/runs/input")
        return
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            print(f"Пропуск: {p.name}")
            continue
        tile = extract_title(img)
        cv2.imshow("Имя предмета", tile)
        cv2.waitKey(1)
        name = input(f"🔍 Имя для {p.name}: ").strip()
        if not name:
            continue
        out = OUT_DIR / f"{normalize(name)}.png"
        cv2.imwrite(str(out), tile)
        print(f"✅ Сохранил: {out.name}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
