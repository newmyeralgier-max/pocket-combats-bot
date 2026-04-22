# -*- coding: utf-8 -*-
"""
Опциональное распознавание имён предметов через OCR.

Смысл: `script/detection/items.py` ищет имена предметов через matchTemplate
по 500+ PNG-шаблонам. Любое изменение шрифта/DPI/темы → предмет теряется,
плюс поддерживать по PNG на каждое имя дорого.

OCR даёт:
  • один проход по ROI вместо N × matchTemplate;
  • устойчивость к мелким изменениям рендера;
  • поддержка новых предметов — просто добавить имя в ALLOWED_ITEM_NAMES,
    без рисования нового шаблона.

Статус: экспериментально, выключено по умолчанию. Включается в конфиге:

    "OCR": {
        "ENABLED": true,
        "ENGINE": "auto",        // "auto" | "rapidocr" | "tesseract"
        "FUZZY_MIN": 0.75,       // минимальный SequenceMatcher.ratio
        "MIN_CONF": 0.5,         // минимальная уверенность OCR-движка
        "DEBUG": false
    }

Движки (устанавливаются отдельно, не в requirements.txt):
  • rapidocr-onnxruntime (pip) — рекомендуется, самый простой и быстрый;
  • pytesseract + tesseract-ocr (+ tesseract-ocr-rus) — если уже стоит.

Smoke-тест на реальном скриншоте: см. tools/debug/ocr_smoke_test.py.
"""

from __future__ import annotations

import difflib
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------
# Lazy-loading движков. При отсутствии зависимости возвращаем None и
# функции detect/run_ocr просто сообщат, что OCR недоступен.
# --------------------------------------------------------------------

_ENGINE_CACHE: Dict[str, Any] = {}


def _load_rapidocr():
    if "rapidocr" in _ENGINE_CACHE:
        return _ENGINE_CACHE["rapidocr"]
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-not-found]
        _ENGINE_CACHE["rapidocr"] = RapidOCR()
    except Exception as e:  # pragma: no cover - зависит от окружения
        _ENGINE_CACHE["rapidocr"] = None
        _ENGINE_CACHE["rapidocr_error"] = str(e)
    return _ENGINE_CACHE["rapidocr"]


def _load_tesseract():
    if "tesseract" in _ENGINE_CACHE:
        return _ENGINE_CACHE["tesseract"]
    try:
        import pytesseract  # type: ignore[import-not-found]
        _ENGINE_CACHE["tesseract"] = pytesseract
    except Exception as e:  # pragma: no cover
        _ENGINE_CACHE["tesseract"] = None
        _ENGINE_CACHE["tesseract_error"] = str(e)
    return _ENGINE_CACHE["tesseract"]


def available_engine(preferred: str = "auto") -> Optional[str]:
    """Возвращает имя доступного движка: 'rapidocr', 'tesseract' или None."""
    if preferred in ("auto", "rapidocr") and _load_rapidocr() is not None:
        return "rapidocr"
    if preferred in ("auto", "tesseract") and _load_tesseract() is not None:
        return "tesseract"
    return None


# --------------------------------------------------------------------
# Нормализация имён для fuzzy-матчинга.
# --------------------------------------------------------------------

_SUFFIX_RE = re.compile(r"_(icon|name|desc)(_\d+)?(\.(png|jpe?g))?$", re.IGNORECASE)
_EXT_RE = re.compile(r"\.(png|jpe?g)$", re.IGNORECASE)
_NON_LETTER_RE = re.compile(r"[\W_]+", re.UNICODE)


def normalize_name(s: str) -> str:
    """Имя шаблона → компактная форма для fuzzy-сравнения.

    'Алая_тиара_icon.png' → 'алаятиара'
    'Апельсин_icon_1.png' → 'апельсин'
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = _SUFFIX_RE.sub("", s)
    s = _EXT_RE.sub("", s)
    s = s.replace("ё", "е")
    s = _NON_LETTER_RE.sub("", s)
    return s


def _best_match(
    ocr_text: str,
    whitelist_norm: Sequence[Tuple[str, str]],
    fuzzy_min: float,
) -> Optional[Tuple[str, float]]:
    """Возвращает (original_name, ratio) или None."""
    needle = normalize_name(ocr_text)
    if not needle:
        return None
    best_name: Optional[str] = None
    best_ratio = 0.0
    for orig, hay in whitelist_norm:
        if not hay:
            continue
        # SequenceMatcher — stdlib, нормально работает с юникодом.
        r = difflib.SequenceMatcher(None, needle, hay).ratio()
        # Бонус за вхождение needle целиком в hay (или наоборот) —
        # частый случай: OCR дал половину имени из-за шума.
        if needle in hay or hay in needle:
            r = max(r, 0.85)
        if r > best_ratio:
            best_ratio = r
            best_name = orig
    if best_name is None or best_ratio < fuzzy_min:
        return None
    return best_name, best_ratio


# --------------------------------------------------------------------
# Запуск OCR.
# --------------------------------------------------------------------

RawHit = Dict[str, Any]


def run_ocr(
    frame_bgr: np.ndarray,
    engine: str = "auto",
    min_conf: float = 0.0,
) -> List[RawHit]:
    """Запускает OCR на изображении. Возвращает список словарей:
        {text: str, score: float, box: (x, y, w, h), corners: [(x,y)*4]}
    """
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return []

    picked = available_engine(engine)
    if picked is None:
        return []

    results: List[RawHit] = []
    if picked == "rapidocr":
        ocr = _load_rapidocr()
        try:
            out, _ = ocr(frame_bgr)
        except Exception:
            return []
        if not out:
            return []
        for item in out:
            # rapidocr: [ [corners x4], text, score ]
            try:
                corners, text, score = item
                score = float(score or 0.0)
            except Exception:
                continue
            if score < min_conf:
                continue
            xs = [int(p[0]) for p in corners]
            ys = [int(p[1]) for p in corners]
            x, y = min(xs), min(ys)
            w, h = max(xs) - x, max(ys) - y
            results.append({
                "text": str(text or ""),
                "score": score,
                "box": (x, y, w, h),
                "corners": [(int(p[0]), int(p[1])) for p in corners],
            })
        return results

    if picked == "tesseract":
        pyt = _load_tesseract()
        try:
            data = pyt.image_to_data(frame_bgr, lang="rus", output_type=pyt.Output.DICT)
        except Exception:
            return []
        n = len(data.get("text", []))
        for i in range(n):
            txt = (data["text"][i] or "").strip()
            if not txt:
                continue
            try:
                conf = float(data["conf"][i]) / 100.0
            except Exception:
                conf = 0.0
            if conf < min_conf:
                continue
            x = int(data["left"][i]); y = int(data["top"][i])
            w = int(data["width"][i]); h = int(data["height"][i])
            results.append({
                "text": txt, "score": conf,
                "box": (x, y, w, h),
                "corners": [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
            })
        return results

    return []


# --------------------------------------------------------------------
# Высокоуровневый детектор: OCR → fuzzy → whitelist, формат как у
# template-ветки (name, score, box, center, slot_hash).
# --------------------------------------------------------------------


def detect_item_names_ocr(
    frame_bgr: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
    whitelist: Optional[Sequence[str]] = None,
    *,
    engine: str = "auto",
    fuzzy_min: float = 0.75,
    min_conf: float = 0.5,
) -> List[Dict[str, Any]]:
    """OCR-детектор имён предметов.

    Args:
      frame_bgr: полный BGR-кадр экрана.
      roi:      (x1, y1, x2, y2) ROI в координатах кадра; None → весь кадр.
      whitelist: список имён шаблонов (обычно CFG['ALLOWED_ITEM_NAMES'],
                 например 'Янтарь_icon.png'). Всё, что OCR выдал и не
                 сматчилось в whitelist по fuzzy_min — отбрасывается.
      engine:   'auto' | 'rapidocr' | 'tesseract'.
      fuzzy_min: нижний порог SequenceMatcher.ratio().
      min_conf:  нижний порог уверенности OCR.

    Returns: список dict'ов того же формата, что и у item-template-детектора:
             {name, score, box:(ax,ay,w,h), center:(cx,cy), slot_hash}.
    """
    if frame_bgr is None:
        return []
    H, W = frame_bgr.shape[:2]
    if roi is None:
        x1, y1, x2, y2 = 0, 0, W, H
    else:
        x1, y1, x2, y2 = [int(v) for v in roi]
        x1 = max(0, min(W - 1, x1)); x2 = max(x1 + 1, min(W, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(y1 + 1, min(H, y2))
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return []

    raw = run_ocr(crop, engine=engine, min_conf=min_conf)
    if not raw:
        return []

    if whitelist:
        whitelist_norm: List[Tuple[str, str]] = [(w, normalize_name(w)) for w in whitelist]
    else:
        whitelist_norm = []

    hits: List[Dict[str, Any]] = []
    for r in raw:
        bx, by, bw, bh = r["box"]
        ax, ay = x1 + bx, y1 + by

        name: str
        score: float
        if whitelist_norm:
            m = _best_match(r["text"], whitelist_norm, fuzzy_min)
            if m is None:
                continue
            name = m[0]
            score = float(m[1])
        else:
            name = r["text"]
            score = float(r["score"])

        hits.append({
            "name": name,
            "score": score,
            "box": (ax, ay, bw, bh),
            "center": (ax + bw // 2, ay + bh // 2),
            "slot_hash": 0,
            "ocr_text": r["text"],
            "ocr_conf": float(r["score"]),
        })
    return hits
