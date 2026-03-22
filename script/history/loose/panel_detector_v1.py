from typing import Dict, Tuple

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def detect_panel_state(image_path: str) -> Dict[str, any]:
    """
    Заглушка-детектор панели:
    - Возвращает 'visible' со статичным боксом, чтобы pipeline был целостным.
    - Ты подставишь сюда свою реальную логику позже.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    H, W = img.shape[:2]
    h_bar = max(40, int(0.12 * H))
    box = int(0.04 * W), int(0.03 * H), int(0.92 * W), h_bar
    return {"state": "visible", "box": box, "score": 0.9}


def annotate(image_bgr: np.ndarray, result: Dict[str, any]) -> np.ndarray:
    vis = image_bgr.copy()
    x, y, w, h = result.get("box", (0, 0, 0, 0))
    state = result.get("state", "unknown")
    score = result.get("score", 0.0)
    color = (255, 0, 0) if state == "visible" else (100, 100, 255)
    if w > 0 and h > 0:
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis, f"Panel: {state} ({score:.2f})", (x, max(20, y - 8)), FONT, 0.6, color, 2, cv2.LINE_AA)
    return vis
