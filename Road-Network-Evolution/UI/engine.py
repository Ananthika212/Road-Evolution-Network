"""One object for the UI to call:

    engine = Engine(Config(crnet_weights="../weights/crnet_best.pth",
                           lgmm_weights="../weights/lgmm_best.pth"))
    result = engine.run(t1_rgb_uint8, t2_rgb_uint8)
    result.regions          # ranked list of dicts (priority, type, area, length ...)
    priority_map(t2, result)
"""
import numpy as np
from PIL import Image

from config import Config
from inference import pick_device, load_crnet, make_road_fn, load_lgmm, make_change_fn
from pipeline import run_pipeline


def read_rgb(path_or_file):
    return np.array(Image.open(path_or_file).convert("RGB"))


class Engine:
    def __init__(self, cfg=None, device=None):
        self.cfg = cfg or Config()
        self.device = device or pick_device()
        self.road_fn = make_road_fn(self.cfg, load_crnet(self.cfg, self.device), self.device)
        self.change_fn = make_change_fn(self.cfg, load_lgmm(self.cfg, self.device), self.device)

    def run(self, t1, t2, gsd_m=None):
        if gsd_m:
            self.cfg.gsd_m = float(gsd_m)
        return run_pipeline(t1, t2, self.road_fn, self.change_fn, self.cfg)
