import json
import logging
import os
from dataclasses import dataclass
from math import floor
from typing import Dict, List, Optional, Tuple


@dataclass
class ItemInfo:
    name: str
    unit_weight: float
    stack_size: int
    price: float
    category: str


@dataclass
class Config:
    reserve: float
    vendor_trigger_percent: float
    min_free_slots: int
    partial_pick: bool
    allow_auto_drop: bool
    drop_floor_value_density: float
    whitelist: List[str]
    blacklist: List[str]
    log_file: str
    log_level: str


@dataclass
class Decision:
    action: str
    take_qty: int = 0
    reason: str = ""
    detail: Optional[dict] = None


def load_config(path: str = "config.json") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    weight = cfg.get("weight", {})
    inv = cfg.get("inventory", {})
    loot = cfg.get("loot", {})
    lists = cfg.get("lists", {})
    logging_cfg = cfg.get("logging", {})
    return Config(
        reserve=weight.get("reserve", 5.0),
        vendor_trigger_percent=weight.get("vendor_trigger_percent", 0.85),
        min_free_slots=inv.get("min_free_slots", 1),
        partial_pick=loot.get("partial_pick", False),
        allow_auto_drop=loot.get("allow_auto_drop", False),
        drop_floor_value_density=loot.get("drop_floor_value_density", 2.0),
        whitelist=lists.get("whitelist", []),
        blacklist=lists.get("blacklist", []),
        log_file=logging_cfg.get("file", "decisions.log"),
        log_level=logging_cfg.get("level", "INFO"),
    )


def load_items_db(path: str = "items_db.json") -> Dict[str, ItemInfo]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    defaults = raw.get("_defaults", {})
    db: Dict[str, ItemInfo] = {}
    for name, v in raw.items():
        if name == "_defaults":
            continue
        cat = v.get("category", "ресурс")
        base = defaults.get(cat, {})
        db[name] = ItemInfo(
            name=name,
            unit_weight=float(v.get("weight", base.get("weight", 0.5))),
            stack_size=int(v.get("stack", base.get("stack", 50))),
            price=float(v.get("price", base.get("price", 1))),
            category=cat,
        )
    return db


def setup_logger(file_path: str, level: str = "INFO"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True) if os.path.dirname(file_path) else None
    logging.basicConfig(
        filename=file_path,
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        encoding="utf-8",
    )


def log_decision(decision: Decision):
    logging.info(f"action={decision.action} reason={decision.reason} detail={decision.detail}")


def decide_pick_simple(
    item_name: str,
    qty_found: int,
    weight_curr: Optional[float],
    weight_max: Optional[float],
    free_slots: int,
    cfg: Config,
    db: Dict[str, ItemInfo],
) -> Decision:
    if item_name in cfg.blacklist:
        d = Decision("skip", reason="blacklist", detail={"item": item_name})
        log_decision(d)
        return d
    if item_name in cfg.whitelist and free_slots > 0:
        d = Decision("take_all", take_qty=qty_found, reason="whitelist", detail={"item": item_name})
        log_decision(d)
        return d
    if weight_curr is not None and weight_max is not None:
        if weight_curr >= weight_max * cfg.vendor_trigger_percent:
            d = Decision("go_vendor", reason="weight_threshold", detail={"curr": weight_curr, "max": weight_max})
            log_decision(d)
            return d
    if free_slots <= 0:
        d = Decision("skip", reason="no_slots", detail={"free_slots": free_slots})
        log_decision(d)
        return d
    d = Decision("take_all", take_qty=qty_found, reason="basic_pick", detail={"item": item_name})
    log_decision(d)
    return d
