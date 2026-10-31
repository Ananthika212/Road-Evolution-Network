"""LGMM-Net road-change detection for a pair of images of any size.

    python predict.py --t1 before.png --t2 after.png --weights ../weights/lgmm_best.pth --out change.png

Tiled at the native 256x256 size (overlap 32, Hann blending, no resizing).
Input convention (same as training): BGR channel order, pixel / 255, then mean 0.5 / std 0.5.
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image

from model import build_lgmm_net

TILE, OVERLAP = 256, 32


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _starts(length, tile, step):
    if length <= tile:
        return [0]
    s = list(range(0, length - tile + 1, step))
    if s[-1] != length - tile:
        s.append(length - tile)
    return s


def _blend_window(tile):
    w = np.hanning(tile + 2)[1:-1].astype(np.float32)
    return np.outer(np.maximum(w, 1e-3), np.maximum(w, 1e-3))


def run_tiled(image, predict_tile, tile=TILE, overlap=OVERLAP):
    """Overlapping-tile inference with Hann blending (same routine as CRNet/predict.py)."""
    H, W = image.shape[:2]
    pad_h, pad_w = max(0, tile - H), max(0, tile - W)
    if pad_h or pad_w:
        mode = "reflect" if (H > 1 and W > 1 and pad_h < H and pad_w < W) else "edge"
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
    Hp, Wp = image.shape[:2]
    step, win = tile - overlap, _blend_window(tile)
    acc, wsum, multi = None, np.zeros((Hp, Wp), np.float32), False
    for y in _starts(Hp, tile, step):
        for x in _starts(Wp, tile, step):
            p = predict_tile(image[y:y + tile, x:x + tile])
            if acc is None:
                multi = p.ndim == 3
                acc = np.zeros(((p.shape[0] if multi else 1), Hp, Wp), np.float32)
            acc[:, y:y + tile, x:x + tile] += (p if multi else p[None]) * win
            wsum[y:y + tile, x:x + tile] += win
    out = (acc / np.maximum(wsum, 1e-6))[:, :H, :W]
    return out if multi else out[0]


def load_lgmm(weights, device=None):
    device = device or pick_device()
    model = build_lgmm_net()
    sd = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("model_dict", sd), strict=True)
    return model.to(device).eval()


class ChangeModel:
    """Callable: (T1, T2 uint8 RGB) -> change probability map (H,W) in [0,1].
    After a call, .all_outputs holds every output map of the network (main + deep-supervision)."""

    def __init__(self, model, device=None, tile=TILE, overlap=OVERLAP, bgr=True):
        self.model, self.device = model, device or pick_device()
        self.tile, self.overlap, self.bgr = tile, overlap, bgr
        self.all_outputs = []

    @torch.inference_mode()
    def _predict(self, pair_tile):
        x = torch.from_numpy(np.ascontiguousarray(pair_tile.transpose(2, 0, 1)))[None].to(self.device)
        out = self.model((x[:, :3], x[:, 3:]))
        outs = list(out) if isinstance(out, (list, tuple)) else [out]
        return np.stack([torch.softmax(o.float(), dim=1)[0, 1].cpu().numpy() for o in outs])

    def __call__(self, t1_u8, t2_u8):
        if self.bgr:  # the training data loader used cv2.imread (BGR) without converting
            t1_u8, t2_u8 = t1_u8[..., ::-1], t2_u8[..., ::-1]
        a = (t1_u8.astype(np.float32) / 255.0 - 0.5) / 0.5
        b = (t2_u8.astype(np.float32) / 255.0 - 0.5) / 0.5
        stacked = run_tiled(np.concatenate([a, b], axis=2), self._predict, self.tile, self.overlap)
        self.all_outputs = [stacked[i] for i in range(stacked.shape[0])]
        return self.all_outputs[0]


def main():
    ap = argparse.ArgumentParser(description="LGMM-Net road-change detection")
    ap.add_argument("--t1", required=True, help="earlier image")
    ap.add_argument("--t2", required=True, help="later image (same size, aligned)")
    ap.add_argument("--weights", default=os.path.join("..", "weights", "lgmm_best.pth"))
    ap.add_argument("--out", default="change.png", help="binary change mask (white = road change)")
    ap.add_argument("--prob-out", default="", help="optional: save the probability map too")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()

    t1 = np.array(Image.open(a.t1).convert("RGB"))
    t2 = np.array(Image.open(a.t2).convert("RGB"))
    if t1.shape != t2.shape:
        raise SystemExit(f"Images must be the same size: {t1.shape} vs {t2.shape}")
    dev = pick_device()
    prob = ChangeModel(load_lgmm(a.weights, dev), dev)(t1, t2)
    Image.fromarray(((prob > a.threshold) * 255).astype(np.uint8)).save(a.out)
    if a.prob_out:
        Image.fromarray((prob * 255).astype(np.uint8)).save(a.prob_out)
    print(f"saved {a.out}  (changed pixels: {(prob > a.threshold).mean() * 100:.2f}%)")


if __name__ == "__main__":
    main()
