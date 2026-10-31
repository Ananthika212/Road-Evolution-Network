"""Glue between the application and the model folders.

CRNet/predict.py and LGMM-Net/predict.py hold the loading and tiled-inference code; this file
loads them by path (both folders contain a file called model.py, so they cannot share sys.path)
and adapts them to the application's Config.
"""
import importlib.util
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _load_predict(folder, alias):
    """Import <ROOT>/<folder>/predict.py as module `alias`, with only that folder on sys.path."""
    path = os.path.join(ROOT, folder)
    if alias in sys.modules:
        return sys.modules[alias]
    # predict.py does `from model import ...`: make sure the right model.py is the one found
    sys.modules.pop("model", None)
    sys.path.insert(0, path)
    try:
        spec = importlib.util.spec_from_file_location(alias, os.path.join(path, "predict.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[alias] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(path)
        sys.modules.pop("model", None)
    return mod


def pick_device():
    return _load_predict("CRNet", "crnet_predict").pick_device()


# ---- CRNet -----------------------------------------------------------------------------
def load_crnet(cfg, device):
    return _load_predict("CRNet", "crnet_predict").load_crnet(cfg.crnet_weights, device)


def make_road_fn(cfg, model, device):
    return _load_predict("CRNet", "crnet_predict").make_road_fn(
        model, device, cfg.crnet_tile, cfg.crnet_overlap)


# ---- LGMM-Net --------------------------------------------------------------------------
def _lgmm():
    # LGMM-Net has its own `models` package; drop any other package called "models" first
    for k in [k for k in sys.modules if k == "models" or k.startswith("models.")]:
        del sys.modules[k]
    return _load_predict("LGMM-Net", "lgmm_predict")


def load_lgmm(cfg, device):
    return _lgmm().load_lgmm(cfg.lgmm_weights, device)


def make_change_fn(cfg, model, device):
    return _lgmm().ChangeModel(model, device, cfg.lgmm_tile, cfg.lgmm_overlap, cfg.lgmm_bgr)
