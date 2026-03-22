import os
import time

import cv2
import numpy as np

DEBUG_VIZ = True
DEBUG_DIR = "loot_debug"


def _ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _now_ms() -> int:
    return int(time.time() * 1000)


def sample_rgb_bgr(frame: np.ndarray, x: int, y: int):
    h, w = frame.shape[:2]
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    b, g, r = frame[y, x].tolist()
    return (r, g, b), (b, g, r)


class VizFrame:

    def __init__(self, frame: np.ndarray, tag: str):
        self.frame = frame.copy()
        self.tag = tag

    def box(self, x1, y1, x2, y2, color=(0, 255, 0), label=None, score=None):
        cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 2)
        if label is not None:
            txt = label if score is None else f"{label} {score:.3f}"
            cv2.putText(self.frame, txt, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    def point(self, x, y, color=(0, 255, 0), radius=6):
        cv2.circle(self.frame, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)

    def save(self, suffix: str):
        if not DEBUG_VIZ:
            return
        _ensure_dir(DEBUG_DIR)
        name = f"{_now_ms()}_{self.tag}_{suffix}.png"
        cv2.imwrite(os.path.join(DEBUG_DIR, name), self.frame)


def debug_pickup_click(
    item_name: str,
    frame_bgr: np.ndarray,
    btn_bbox: tuple,
    btn_score: float,
    tap_func,
    screencap_func,
    log_func=print,
    y_offset_up: int = 12,
):
    """
    Визуализирует детект кнопки, ставит маркер точки клика, логирует RGB,
    сохраняет кадры до/после клика и выполняет сам клик.
    """
    x1, y1, x2, y2 = map(int, btn_bbox)
    cx = (x1 + x2) // 2
    click_y = max(y1 + 4, y2 - y_offset_up)
    click_x = cx
    pre_viz = VizFrame(frame_bgr, tag=f"{item_name}")
    pre_viz.box(x1, y1, x2, y2, color=(0, 255, 255), label="pickup", score=btn_score)
    pre_viz.point(click_x, click_y, color=(0, 200, 0))
    pre_viz.save("before_tap")
    rgb, bgr = sample_rgb_bgr(frame_bgr, click_x, click_y)
    log_func(
        f"[PICKUP:DEBUG] {item_name} tap=({click_x},{click_y}) rgb={rgb} bgr={bgr} btn_bbox=({x1},{y1},{x2},{y2}) score={btn_score:.3f} offset_up={y_offset_up}"
    )
    tap_func(click_x, click_y)
    time.sleep(0.25)
    post_bgr = screencap_func()
    post_viz = VizFrame(post_bgr, tag=f"{item_name}")
    post_viz.point(click_x, click_y, color=(0, 165, 255))
    post_viz.save("after_tap")
    return post_bgr
