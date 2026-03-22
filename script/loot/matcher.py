# C:/bot/script/loot/matcher.py


import os
from typing import List, Tuple

import cv2
import numpy as np


class TemplateCache:
    def __init__(self):
        self._cache = {}  # path -> {mtime, name, gray_proc}

    def get(self, path: str, img_bgr: np.ndarray, mode: str):
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            mtime = None
        key = (path, mode)
        entry = self._cache.get(key)
        if entry and entry.get("mtime") == mtime:
            return entry["gray_proc"]
        gray_proc = preprocess_gray(img_bgr, mode=mode)
        self._cache[key] = {"mtime": mtime, "gray_proc": gray_proc, "scaled": {}}
        return gray_proc

    def get_all_scaled(self, path: str, img_bgr: np.ndarray, mode: str, scales: List[float]) -> dict:
        gray_proc = self.get(path, img_bgr, mode)
        key = (path, mode)
        entry = self._cache[key]
        
        scaled_dict = {}
        th, tw = gray_proc.shape[:2]
        for s in scales:
            if s == 1.0:
                scaled_dict[s] = gray_proc
            elif s in entry["scaled"]:
                scaled_dict[s] = entry["scaled"][s]
            else:
                tws = max(5, int(round(tw * s)))
                ths = max(5, int(round(th * s)))
                tpl_s = cv2.resize(gray_proc, (tws, ths), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)
                entry["scaled"][s] = tpl_s
                scaled_dict[s] = tpl_s
                
        return scaled_dict

    def get_all_scharr_scaled(self, path: str, img_bgr: np.ndarray, mode: str, scales: List[float], scharr_fn) -> dict:
        key = (path, mode)
        entry = self._cache.get(key)
        if not entry or "scharr_scaled" not in entry:
            self.get(path, img_bgr, mode) # ensure entry exists
            entry = self._cache[key]
            entry["scharr_scaled"] = {}
            
        scaled_dict = self.get_all_scaled(path, img_bgr, mode, scales)
        scharr_dict = {}
        for s in scales:
            if s in entry["scharr_scaled"]:
                scharr_dict[s] = entry["scharr_scaled"][s]
            else:
                s_mag = scharr_fn(scaled_dict[s])
                entry["scharr_scaled"][s] = s_mag
                scharr_dict[s] = s_mag
        return scharr_dict


TPL_CACHE = TemplateCache()


def preprocess_gray(gray: np.ndarray, mode: str = "gray") -> np.ndarray:
    # Вход может быть BGR или Gray
    if gray is None:
        return gray
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if mode == "gray":
        return gray
    if mode == "sobel":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(gray)
        g = cv2.bilateralFilter(g, 5, 75, 75)
        grad_x = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
        return cv2.convertScaleAbs(grad)
    return gray


def find_all_matches(
    roi_proc: np.ndarray,
    tpl_scaled_dict: dict,
    *,
    scales: List[float],
    threshold: float,
    min_dy: int,
    method: int = cv2.TM_CCOEFF_NORMED,
) -> List[Tuple[int, int, float, float]]:
    """
    Возвращает список (mx, my, scale, score) со стабилизацией по вертикальному разносу min_dy.
    tpl_scaled_dict — предвычисленные масштабированные шаблоны.
    """
    H, W = roi_proc.shape[:2]
    picks: List[Tuple[int, int, float, float]] = []

    def ok_y(y: int) -> bool:
        return all(abs(y - py) >= min_dy for (px, py, pscale, pscore) in picks)

    for s in scales:
        tpl_s = tpl_scaled_dict.get(s)
        if tpl_s is None:
            continue
            
        ths, tws = tpl_s.shape[:2]
        if ths == 0 or tws == 0 or ths > H or tws > W:
            continue
            
        res = cv2.matchTemplate(roi_proc, tpl_s, method)
        ys, xs = np.where(res >= float(threshold))
        
        # Собираем хиты с вертикальным разносом
        for y, x in zip(ys.tolist(), xs.tolist()):
            if ok_y(y):
                picks.append((int(x), int(y), float(s), float(res[y, x])))

    # Сортировка: сверху-вниз, при равенстве — по score убыв.
    picks.sort(key=lambda m: (m[1], -m[3]))
    return picks
