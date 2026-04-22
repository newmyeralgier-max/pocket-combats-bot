# -*- coding: utf-8 -*-
"""Юнит-тесты для script/detection/player_status.py.

Покрывают:
  • detect_dead: нет шаблона в CFG → False; шаблон есть, но файла
    нет на диске → False; корректная работа _cfg_get.
  • compute_hp_ratio: зелёный HP-бар → малое значение; красный → большое;
    пустой ROI → None; нет ROI в конфиге → None.
  • find_heal_button: нет шаблона → None.

Реальная matchTemplate-логика для шаблонов-в-файлах проверяется вручную
на реальных скринах игры — юнит-тесты фокусируются на защите от кривого
конфига и цветовом HP-детекторе.
"""

import unittest

import numpy as np

from script.detection import player_status as PS


class TestCfgGet(unittest.TestCase):
    def test_nested_miss(self):
        self.assertEqual(PS._cfg_get({}, "RECOVER", "DEAD_TPL", default="x"), "x")

    def test_nested_hit(self):
        cfg = {"RECOVER": {"DEAD_TPL": "/tmp/x.png"}}
        self.assertEqual(PS._cfg_get(cfg, "RECOVER", "DEAD_TPL"), "/tmp/x.png")

    def test_wrong_type(self):
        cfg = {"RECOVER": "notadict"}
        self.assertEqual(PS._cfg_get(cfg, "RECOVER", "DEAD_TPL", default=None), None)


class TestDetectDead(unittest.TestCase):
    def test_no_tpl_in_cfg(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertFalse(PS.detect_dead(img, {}))
        self.assertFalse(PS.detect_dead(img, {"RECOVER": {}}))
        self.assertFalse(PS.detect_dead(img, {"RECOVER": {"DEAD_TPL": ""}}))

    def test_missing_file(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cfg = {"RECOVER": {"DEAD_TPL": "/nonexistent/path/death.png"}}
        self.assertFalse(PS.detect_dead(img, cfg))

    def test_none_frame(self):
        self.assertFalse(PS.detect_dead(None, {"RECOVER": {"DEAD_TPL": "/x.png"}}))


class TestComputeHpRatio(unittest.TestCase):
    def _solid_color_hp(self, bgr_color, roi=(10, 10, 90, 30)):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        x1, y1, x2, y2 = roi
        img[y1:y2, x1:x2] = bgr_color
        cfg = {"RECOVER": {"HP_ROI": list(roi)}}
        return PS.compute_hp_ratio(img, cfg)

    def test_green_bar_means_low_ratio(self):
        # Чистый зелёный → 0% «красноты».
        r = self._solid_color_hp((0, 255, 0))
        self.assertIsNotNone(r)
        assert r is not None
        self.assertLess(r, 0.05)

    def test_red_bar_means_high_ratio(self):
        # Чистый красный → ~100% «красноты».
        r = self._solid_color_hp((0, 0, 255))
        self.assertIsNotNone(r)
        assert r is not None
        self.assertGreater(r, 0.95)

    def test_no_roi(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertIsNone(PS.compute_hp_ratio(img, {}))
        self.assertIsNone(PS.compute_hp_ratio(img, {"RECOVER": {}}))
        self.assertIsNone(PS.compute_hp_ratio(img, {"RECOVER": {"HP_ROI": [1, 2, 3]}}))

    def test_empty_roi(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # ROI за пределами / нулевой
        cfg = {"RECOVER": {"HP_ROI": [200, 200, 300, 300]}}
        self.assertIsNone(PS.compute_hp_ratio(img, cfg))

    def test_grey_bar_is_undetermined(self):
        # Серый UI без цветных пикселей — считаем «неизмеримо».
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        cfg = {"RECOVER": {"HP_ROI": [10, 10, 90, 30]}}
        self.assertIsNone(PS.compute_hp_ratio(img, cfg))


class TestIsLowHp(unittest.TestCase):
    def test_high_ratio_is_low_hp(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[10:30, 10:90] = (0, 0, 255)  # red
        cfg = {"RECOVER": {"HP_ROI": [10, 10, 90, 30], "HP_LOW_RATIO": 0.5}}
        self.assertTrue(PS.is_low_hp(img, cfg))

    def test_low_ratio_is_not_low_hp(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[10:30, 10:90] = (0, 255, 0)  # green
        cfg = {"RECOVER": {"HP_ROI": [10, 10, 90, 30], "HP_LOW_RATIO": 0.5}}
        self.assertFalse(PS.is_low_hp(img, cfg))

    def test_unknown_is_none(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        cfg = {"RECOVER": {"HP_ROI": [10, 10, 90, 30]}}
        self.assertIsNone(PS.is_low_hp(img, cfg))


class TestFindHealButton(unittest.TestCase):
    def test_no_cfg(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertIsNone(PS.find_heal_button(img, {}))

    def test_missing_file(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cfg = {"RECOVER": {"HEAL_BTN_TPL": "/no/such/file.png"}}
        self.assertIsNone(PS.find_heal_button(img, cfg))


if __name__ == "__main__":
    unittest.main()
