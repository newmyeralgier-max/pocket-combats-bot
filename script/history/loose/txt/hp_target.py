import cv2
import numpy as np


def bgr_to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def red_mask_hsv(hsv):
    lower1 = np.array([0, 120, 120], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([160, 120, 120], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def find_hp_bars(bgr, world_roi=None, min_w=120, max_h=20, ar_min=4.0):
    """
    Возвращает список прямоугольников HP полосок в координатах экрана: [(x,y,w,h), ...]
    world_roi: (x,y,w,h) — область игрового мира (исключаем нижнюю панель/кнопки)
    min_w: минимальная ширина полоски
    max_h: максимальная высота (полоска тонкая)
    ar_min: минимальное соотношение сторон w/h
    """
    H, W = bgr.shape[:2]
    if world_roi is None:
        world_roi = 0, 0, W, int(H * 0.75)
    rx, ry, rw, rh = world_roi
    crop = bgr[ry : ry + rh, rx : rx + rw]
    hsv = bgr_to_hsv(crop)
    mask = red_mask_hsv(hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_w or h > max_h:
            continue
        if h == 0:
            continue
        ar = w / float(h)
        if ar < ar_min:
            continue
        area = cv2.contourArea(cnt)
        if area < 0.3 * (w * h):
            continue
        rects.append((x + rx, y + ry, w, h))
    return rects


def choose_target(rects, prefer="widest", screen_size=None):
    """
    Выбор цели:
    - widest: самая широкая полоска (обычно главный текущий таргет/ближайший)
    - center: ближе всего к центру экрана
    """
    if not rects:
        return None
    if prefer == "widest":
        return max(rects, key=lambda r: r[2])
    elif prefer == "center" and screen_size is not None:
        H, W = screen_size
        cx, cy = W // 2, H // 2

        def dist(r):
            x, y, w, h = r
            mx, my = x + w // 2, y + h // 2
            return (mx - cx) ** 2 + (my - cy) ** 2

        return min(rects, key=dist)
    else:
        return rects[0]


def aim_point_from_hp(rect, offset=30):
    """
    Точка прицеливания: чуть ниже центра полоски HP.
    offset: пикселей вниз от низа полоски.
    """
    x, y, w, h = rect
    mx = x + w // 2
    my = y + h + offset
    return mx, my
