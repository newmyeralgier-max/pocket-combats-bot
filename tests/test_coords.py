# -*- coding: utf-8 -*-
"""Юнит-тесты script/core/coords.py (резолюшен-агностичный слой)."""

import unittest

import numpy as np

from script.core import coords as C


class TestCurrentSize(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(C.current_size({}), (1080, 2460))

    def test_from_cfg(self):
        self.assertEqual(C.current_size({"SCREEN_W": 1920, "SCREEN_H": 1080}), (1920, 1080))

    def test_bad_values_fall_back(self):
        # Некорректные значения → дефолты.
        self.assertEqual(C.current_size({"SCREEN_W": "abc", "SCREEN_H": None}), (1080, 2460))


class TestRelPoint(unittest.TestCase):
    def test_half_half(self):
        cfg = {"SCREEN_W": 1080, "SCREEN_H": 2460}
        self.assertEqual(C.rel_point(cfg, 0.5, 0.5), (540, 1230))

    def test_continue_fallback(self):
        # 2250/2460 ≈ 0.9146 → ровно 2250 на эталонном разрешении
        cfg = {"SCREEN_W": 1080, "SCREEN_H": 2460}
        x, y = C.rel_point(cfg, 0.5, 0.9146)
        self.assertEqual(x, 540)
        self.assertEqual(y, 2250)

    def test_continue_fallback_scales(self):
        # На другом разрешении та же относительная точка — пропорционально.
        cfg = {"SCREEN_W": 720, "SCREEN_H": 1640}
        x, y = C.rel_point(cfg, 0.5, 0.9146)
        self.assertEqual(x, 360)
        self.assertEqual(y, 1500)  # 1640*0.9146 ≈ 1500

    def test_corners(self):
        cfg = {"SCREEN_W": 100, "SCREEN_H": 200}
        self.assertEqual(C.rel_point(cfg, 0.0, 0.0), (0, 0))
        self.assertEqual(C.rel_point(cfg, 1.0, 1.0), (100, 200))


class TestRelRoi(unittest.TestCase):
    def test_rel_roi(self):
        cfg = {"SCREEN_W": 1080, "SCREEN_H": 2460}
        self.assertEqual(
            C.rel_roi(cfg, 0.0, 0.163, 0.463, 0.854),
            (0, 401, 500, 2101),  # ≈ [0, 400, 500, 2100]
        )


class TestScaleFromRef(unittest.TestCase):
    def test_identity_same_res(self):
        cfg = {"SCREEN_W": 1080, "SCREEN_H": 2460}
        self.assertEqual(C.scale_xy_from_ref(cfg, 540, 2250), (540, 2250))
        self.assertEqual(
            C.scale_roi_from_ref(cfg, [0, 1000, 1080, 1500]),
            (0, 1000, 1080, 1500),
        )

    def test_scale_to_half_height(self):
        cfg = {"SCREEN_W": 1080, "SCREEN_H": 1230}  # половина высоты
        x, y = C.scale_xy_from_ref(cfg, 540, 2250)
        self.assertEqual(x, 540)
        self.assertEqual(y, 1125)  # 2250/2

    def test_scale_roi(self):
        cfg = {"SCREEN_W": 540, "SCREEN_H": 1230}  # всё в половину
        self.assertEqual(
            C.scale_roi_from_ref(cfg, [0, 1000, 1080, 1500]),
            (0, 500, 540, 750),
        )

    def test_custom_ref(self):
        cfg = {"SCREEN_W": 2160, "SCREEN_H": 4920,
               "REF_SCREEN_W": 1080, "REF_SCREEN_H": 2460}
        self.assertEqual(C.scale_xy_from_ref(cfg, 540, 1230), (1080, 2460))

    def test_none_roi(self):
        self.assertEqual(C.scale_roi_from_ref({}, None), (0, 0, 0, 0))


class TestIsRelativeRoi(unittest.TestCase):
    def test_all_fractions(self):
        self.assertTrue(C.is_relative_roi([0.0, 0.1, 0.9, 1.0]))
        self.assertTrue(C.is_relative_roi((0.0, 0.0, 1.0, 1.0)))

    def test_has_pixel(self):
        self.assertFalse(C.is_relative_roi([0, 400, 500, 2100]))
        self.assertFalse(C.is_relative_roi([0.5, 400, 0.9, 0.95]))

    def test_bad_input(self):
        self.assertFalse(C.is_relative_roi(None))
        self.assertFalse(C.is_relative_roi([1, 2, 3]))
        self.assertFalse(C.is_relative_roi([0.5, "x", 0.9, 1.0]))


class TestAutoRoi(unittest.TestCase):
    def test_relative_goes_through_rel(self):
        cfg = {"SCREEN_W": 1080, "SCREEN_H": 2460}
        self.assertEqual(C.auto_roi(cfg, [0.0, 0.0, 0.5, 0.5]), (0, 0, 540, 1230))

    def test_pixel_goes_through_scale(self):
        cfg = {"SCREEN_W": 540, "SCREEN_H": 1230, "REF_SCREEN_W": 1080, "REF_SCREEN_H": 2460}
        # Пиксели эталонного 1080×2460 → масштабируются в половину
        self.assertEqual(C.auto_roi(cfg, [0, 1000, 1080, 1500]), (0, 500, 540, 750))


class TestDetectAndApply(unittest.TestCase):
    def test_detect_size(self):
        img = np.zeros((1640, 720, 3), dtype=np.uint8)
        self.assertEqual(C.detect_screen_size(img), (720, 1640))
        self.assertIsNone(C.detect_screen_size(None))

    def test_apply_updates_cfg(self):
        cfg = {"SCREEN_W": 1080, "SCREEN_H": 2460}
        img = np.zeros((1640, 720, 3), dtype=np.uint8)
        new_size = C.apply_detected_size(cfg, img)
        self.assertEqual(new_size, (720, 1640))
        self.assertEqual(cfg["SCREEN_W"], 720)
        self.assertEqual(cfg["SCREEN_H"], 1640)

    def test_apply_noop_same_size(self):
        cfg = {"SCREEN_W": 1080, "SCREEN_H": 2460}
        img = np.zeros((2460, 1080, 3), dtype=np.uint8)
        self.assertIsNone(C.apply_detected_size(cfg, img))
        self.assertEqual(cfg["SCREEN_W"], 1080)


if __name__ == "__main__":
    unittest.main()
