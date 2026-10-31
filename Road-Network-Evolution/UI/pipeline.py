"""End-to-end: two satellite images in -> road masks, change map, ranked changes."""
import time
from dataclasses import dataclass

import numpy as np

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "RCRM"))
from road_change_analysis import analyse  # noqa: E402  (RCRM module)
from config import Config

PRIORITY_COLORS = {"High": (220, 38, 38), "Medium": (245, 158, 11), "Low": (34, 139, 34)}


@dataclass
class Result:
    roads_t1: np.ndarray
    roads_t2: np.ndarray
    road_prob_t1: np.ndarray
    road_prob_t2: np.ndarray
    change_prob: np.ndarray
    analysis: object
    change_outputs: list = None      # all LGMM-Net output maps (5), if the model provides them
    seconds_roads: float = 0.0
    seconds_change: float = 0.0

    @property
    def regions(self):
        return self.analysis.regions


def check_pair(t1, t2):
    if t1.ndim != 3 or t2.ndim != 3 or t1.shape[2] != 3 or t2.shape[2] != 3:
        raise ValueError("Both images must be 3-channel RGB.")
    if t1.shape != t2.shape:
        raise ValueError(f"Images must be the same size and aligned: {t1.shape[:2]} vs {t2.shape[:2]}. "
                         "Export both dates from Earth Engine with the same region and scale.")


def run_pipeline(t1, t2, road_fn, change_fn, cfg=None):
    """t1, t2: uint8 RGB arrays of equal size. road_fn / change_fn come from models.py."""
    cfg = cfg or Config()
    check_pair(t1, t2)
    t0 = time.time()
    p1, p2 = road_fn(t1), road_fn(t2)
    t_roads = time.time() - t0
    t0 = time.time()
    cp = change_fn(t1, t2)
    t_change = time.time() - t0
    outs = list(getattr(change_fn, "all_outputs", None) or [cp])
    r1, r2 = p1 >= cfg.road_threshold, p2 >= cfg.road_threshold
    return Result(r1, r2, p1, p2, cp, analyse(r1, r2, cp, cfg), outs, t_roads, t_change)


# ------------------------------------------------------------ rendering ------
def overlay_roads(img, mask, color=(255, 0, 255), alpha=0.6):
    out = img.astype(np.float32).copy()
    c = np.array(color, np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * c
    return out.astype(np.uint8)


def priority_map(base_img, result, alpha=0.55):
    """Colour each kept change region by its priority (red/amber/green)."""
    out = base_img.astype(np.float32).copy()
    for r in result.regions:
        m = result.analysis.labels == r["id"]
        out[m] = (1 - alpha) * out[m] + alpha * np.array(PRIORITY_COLORS[r["priority"]], np.float32)
    return out.astype(np.uint8)
