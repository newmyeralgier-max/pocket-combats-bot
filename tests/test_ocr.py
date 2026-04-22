# -*- coding: utf-8 -*-
"""Юнит-тесты для чисто-Python частей script/detection/ocr.py.

Сам rapidocr/pytesseract движок тестируется вручную через
tools/debug/ocr_smoke_test.py на реальном скриншоте игры.
"""

import unittest

import numpy as np

from script.detection import ocr as OCR


class TestNormalizeName(unittest.TestCase):
    def test_strips_suffix(self):
        self.assertEqual(OCR.normalize_name("Янтарь_icon.png"), "янтарь")
        self.assertEqual(OCR.normalize_name("Апельсин_icon_1.png"), "апельсин")
        self.assertEqual(OCR.normalize_name("Алая_тиара_icon.png"), "алаятиара")
        self.assertEqual(OCR.normalize_name("Волчий_клык_name.png"), "волчийклык")

    def test_yo_is_e(self):
        self.assertEqual(OCR.normalize_name("Зелёные_ягоды"),
                         OCR.normalize_name("Зеленые_ягоды"))

    def test_unicode_nfkc_lower(self):
        # NFKC + lowercase
        self.assertEqual(OCR.normalize_name("Янтарь"), "янтарь")
        self.assertEqual(OCR.normalize_name("ЯНТАРЬ"), "янтарь")

    def test_empty_and_none(self):
        self.assertEqual(OCR.normalize_name(""), "")
        self.assertEqual(OCR.normalize_name(None), "")  # type: ignore[arg-type]


class TestBestMatch(unittest.TestCase):
    def setUp(self):
        self.whitelist = [
            ("Янтарь_icon.png", OCR.normalize_name("Янтарь_icon.png")),
            ("Алая_тиара_icon.png", OCR.normalize_name("Алая_тиара_icon.png")),
            ("Зелёные_ягоды_icon.png", OCR.normalize_name("Зелёные_ягоды_icon.png")),
            ("Волчий_клык_icon.png", OCR.normalize_name("Волчий_клык_icon.png")),
        ]

    def test_exact_hit(self):
        m = OCR._best_match("Янтарь", self.whitelist, 0.75)
        self.assertIsNotNone(m)
        assert m is not None
        self.assertEqual(m[0], "Янтарь_icon.png")
        self.assertGreaterEqual(m[1], 0.99)

    def test_yo_vs_e(self):
        m = OCR._best_match("Зеленые ягоды", self.whitelist, 0.75)
        self.assertIsNotNone(m)
        assert m is not None
        self.assertEqual(m[0], "Зелёные_ягоды_icon.png")

    def test_partial_substring_boost(self):
        # OCR выдал только часть имени — всё равно должно замэтчиться
        # благодаря бусту за вхождение.
        m = OCR._best_match("клык", self.whitelist, 0.75)
        self.assertIsNotNone(m)
        assert m is not None
        self.assertEqual(m[0], "Волчий_клык_icon.png")

    def test_below_threshold(self):
        self.assertIsNone(OCR._best_match("Ерунда", self.whitelist, 0.75))

    def test_empty_text(self):
        self.assertIsNone(OCR._best_match("", self.whitelist, 0.75))


class TestRunOcrNoEngine(unittest.TestCase):
    """Если ни один OCR-движок не установлен — run_ocr должен тихо вернуть []."""

    def test_run_ocr_handles_missing_engine(self):
        # Подменяем кеш движков, чтобы симулировать отсутствие.
        saved = dict(OCR._ENGINE_CACHE)
        try:
            OCR._ENGINE_CACHE.clear()
            OCR._ENGINE_CACHE["rapidocr"] = None
            OCR._ENGINE_CACHE["tesseract"] = None
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            self.assertEqual(OCR.run_ocr(img, engine="auto"), [])
            self.assertEqual(OCR.detect_item_names_ocr(img, whitelist=["a.png"]), [])
            self.assertIsNone(OCR.available_engine("auto"))
        finally:
            OCR._ENGINE_CACHE.clear()
            OCR._ENGINE_CACHE.update(saved)


if __name__ == "__main__":
    unittest.main()
