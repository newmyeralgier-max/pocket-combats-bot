# -*- coding: utf-8 -*-
import os
import sys
import unittest
from unittest.mock import patch
import numpy as np

# Ensure root (c:\bot) is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from script.loot.victory_drop import (
    _build_registry_index,
    _icon_path_to_item_name_filename,
    scan_victory_drop_targets
)
import script.loot.auto_bot as auto_bot
from script.loot import tpl_loader as TL

class TestVictoryDrop(unittest.TestCase):

    def setUp(self):
        self.mock_cfg = {
            "ROI": {
                "VICTORY_DROP_REL": [0.1, 0.5, 0.9, 0.9]
            },
            "FIND": {
                "PREPROC_MODE": "gray",
                "ITEM_SCALES": [1.0],
                "MIN_DY": 10
            },
            "ALLOWED_ITEM_NAMES": ["item_sword_name.png", "item_shield_name.png"]
        }
        # 1920x1080 empty frame
        self.dummy_frame = np.zeros((1920, 1080, 3), dtype=np.uint8)

    def test_build_registry_index(self):
        reg = {
            "item.icon.sword": {"group": "item.icon", "tpl": "C:/tpls/sword_icon.png"},
            "rune.icon.fire": {"group": "rune.icon", "tpl": "C:/tpls/fire_icon.png"},
            "item.name.sword": {"group": "item.name", "tpl": "C:/tpls/item_sword_name.png"}
        }
        idx = _build_registry_index(reg)
        self.assertIn("C:/tpls/sword_icon.png", idx)
        self.assertEqual(idx["C:/tpls/sword_icon.png"], ("item.icon", "sword"))
        self.assertIn("C:/tpls/fire_icon.png", idx)
        self.assertEqual(idx["C:/tpls/fire_icon.png"], ("rune.icon", "fire"))

    def test_icon_path_to_item_name(self):
        reg = {
            "item.icon.sword": {"group": "item.icon", "tpl": "C:/tpls/sword_icon.png"},
            "item.name.sword": {"group": "item.name", "tpl": "C:/tpls/item_sword_name.png"}
        }
        idx = _build_registry_index(reg)
        
        with patch.object(TL, 'REG', reg):
            TL._nfkc_lower = lambda x: x.lower()
            res = _icon_path_to_item_name_filename("C:/tpls/sword_icon.png", idx)
            self.assertEqual(res, "item_sword_name.png")

    @patch('script.loot.victory_drop.find_all_matches')
    @patch('script.loot.utils.imread_u8')
    @patch('script.loot.auto_bot.ui_tpl_path')
    @patch('script.loot.victory_drop._detect_continue_box')
    def test_scan_victory_drop_targets_empty(self, mock_detect_cont, mock_ui_path, mock_imread, mock_find):
        mock_ui_path.return_value = "dummy_continue.png"
        mock_imread.return_value = np.zeros((50, 200, 3), dtype=np.uint8)
        
        # Step 0: continue found
        # Step 3: no matches inside ROI
        mock_find.side_effect = [
            [(100, 1500, 1.0, 0.9)],
            [] 
        ]
        mock_detect_cont.return_value = (500, 1500, 200, 50)
        
        reg = {
            "item.icon.sword": {"group": "item.icon", "tpl": "C:/tpls/sword_icon.png"}
        }
        with patch.object(TL, 'REG', reg):
            auto_bot.CFG = self.mock_cfg
            res = scan_victory_drop_targets(self.dummy_frame, self.mock_cfg)
            
            self.assertFalse(res["loot_found"])
            self.assertEqual(len(res["allowed_loot_ids"]), 0)
            self.assertEqual(len(res["detected_all"]), 0)

    @patch('script.loot.victory_drop.find_all_matches')
    @patch('script.loot.utils.imread_u8')
    @patch('script.loot.auto_bot.ui_tpl_path')
    @patch('script.loot.victory_drop._detect_continue_box')
    def test_scan_victory_drop_targets_found(self, mock_detect_cont, mock_ui_path, mock_imread, mock_find):
        mock_ui_path.return_value = "dummy_continue.png"
        mock_imread.return_value = np.zeros((50, 200, 3), dtype=np.uint8)
        mock_detect_cont.return_value = (500, 1500, 200, 50)
        
        # We will simulate multiple calls to find_all_matches.
        def mock_find_func(img, tpl, **kwargs):
            if tpl.shape == (50, 200): # dummy continue
                return [(100, 1500, 1.0, 0.9)]
            elif tpl.shape == (50, 50): # dummy sword (whitelist)
                return [(10, 20, 1.0, 0.95)]
            elif tpl.shape == (40, 40): # dummy trash (not whitelist)
                return [(80, 20, 1.0, 0.88)]
            return []
            
        mock_find.side_effect = mock_find_func
        
        reg = {
            "item.icon.sword": {"group": "item.icon", "tpl": "C:/tpls/sword_icon.png"},
            "item.name.sword": {"group": "item.name", "tpl": "C:/tpls/item_sword_name.png"},
            "item.icon.trash": {"group": "item.icon", "tpl": "C:/tpls/trash_icon.png"},
            "item.name.trash": {"group": "item.name", "tpl": "C:/tpls/item_trash_name.png"}
        }
        
        # Add fake images to _IMG_CACHE so imread is bypassed for icons loops
        import script.loot.victory_drop as VD
        VD._IMG_CACHE["C:/tpls/sword_icon.png"] = np.zeros((50, 50, 3), dtype=np.uint8)
        VD._IMG_CACHE["C:/tpls/trash_icon.png"] = np.zeros((40, 40, 3), dtype=np.uint8)
        
        with patch.object(TL, 'REG', reg):
            auto_bot.CFG = self.mock_cfg
            TL._nfkc_lower = lambda x: x.lower()
            res = scan_victory_drop_targets(self.dummy_frame, self.mock_cfg)
            
            self.assertTrue(res["loot_found"])
            self.assertEqual(len(res["detected_all"]), 2)
            self.assertEqual(len(res["allowed_loot_ids"]), 1)
            self.assertIn("item_sword_name.png", res["allowed_loot_ids"])
            self.assertNotIn("item_trash_name.png", res["allowed_loot_ids"])

if __name__ == '__main__':
    unittest.main()
