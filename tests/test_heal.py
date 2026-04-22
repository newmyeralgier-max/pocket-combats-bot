# -*- coding: utf-8 -*-
"""Юнит-тесты для script/overlays/heal.py.

Проверяем, что heal не падает при отсутствии зависимостей (adb,
cv2-шаблонов), корректно предпочитает явные координаты и молча
возвращает False при любых ошибках.
"""

import unittest
from unittest.mock import patch

import numpy as np

from script.overlays import heal as HEAL


class TestCoerceXy(unittest.TestCase):
    def test_list(self):
        self.assertEqual(HEAL._coerce_xy([100, 200]), (100, 200))

    def test_tuple(self):
        self.assertEqual(HEAL._coerce_xy((50.0, 60.0)), (50, 60))

    def test_wrong_len(self):
        self.assertIsNone(HEAL._coerce_xy([1, 2, 3]))

    def test_not_list(self):
        self.assertIsNone(HEAL._coerce_xy("100,200"))

    def test_garbage(self):
        self.assertIsNone(HEAL._coerce_xy(["a", "b"]))


class TestPerformHeal(unittest.TestCase):
    def test_no_cfg(self):
        self.assertFalse(HEAL.perform_heal({}))
        self.assertFalse(HEAL.perform_heal({"RECOVER": {}}))

    def test_xy_takes_priority(self):
        calls = []

        def fake_tap(x, y, reason=None):
            calls.append((x, y, reason))

        cfg = {"RECOVER": {"HEAL_TAP_XY": [500, 600], "HEAL_POST_SLEEP_S": 0.0}}
        with patch("script.device.adb.tap_raw", fake_tap):
            self.assertTrue(HEAL.perform_heal(cfg))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], 500)
        self.assertEqual(calls[0][1], 600)

    def test_xy_failure_returns_false(self):
        def boom(*a, **k):
            raise RuntimeError("adb broken")

        cfg = {"RECOVER": {"HEAL_TAP_XY": [500, 600]}}
        with patch("script.device.adb.tap_raw", boom):
            self.assertFalse(HEAL.perform_heal(cfg))

    def test_tpl_no_match_returns_false(self):
        cfg = {"RECOVER": {"HEAL_BTN_TPL": "/nonexistent.png"}}
        # Нет файла шаблона → find_heal_button вернёт None → False.
        self.assertFalse(HEAL.perform_heal(cfg, frame_bgr=np.zeros((100, 100, 3), dtype=np.uint8)))


if __name__ == "__main__":
    unittest.main()
