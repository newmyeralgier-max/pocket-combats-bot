import os
from datetime import datetime

import cv2
import numpy as np
import pyautogui

WINDOW_REGION = 200, 120, 900, 520
OUT_DIR = "C:\\bot\\screens"
SHOW_WINDOW = True


def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def grab_window_bgr():
    l, t, w, h = WINDOW_REGION
    im = pyautogui.screenshot(region=(l, t, w, h))
    im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    return im


def analyze(img_bgr):
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    dominant = int(np.argmax(hist))
    print(f"[INFO] Размер: {w}x{h}, доминирующий цвет (gray): {dominant}")
    return dominant


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img = grab_window_bgr()
    fname = f"debug_window_{now_str()}.png"
    path = os.path.join(OUT_DIR, fname)
    cv2.imwrite(path, img)
    print(f"[SAVED] {path}")
    analyze(img)
    if SHOW_WINDOW:
        cv2.imshow("Window Region", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
