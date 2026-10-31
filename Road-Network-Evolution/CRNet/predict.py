"""CRNet road extraction on an image of any size (tiled inference at native resolution).

    python predict.py --image area.png --weights ../weights/crnet_best.pth --out roads.png

The image is cut into overlapping 1024x1024 tiles (never resized), each tile is predicted and
the tiles are blended with a 2-D Hann window, so no seams appear.
Input convention (same as training): RGB, pixel / 255.
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image

from model import CRNet

TILE, OVERLAP = 1024, 128


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
    """Run predict_tile over image (H,W,C float32) in overlapping tiles and blend the results.

    predict_tile: (tile,tile,C) -> (tile,tile) map, or (K,tile,tile) for K stacked maps.
    Images smaller than a tile are padded (reflect), never resized.
    """
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


def load_crnet(weights, device=None):
    device = device or pick_device()
    model = CRNet(num_classes=1, pretrained=False)
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def make_road_fn(model, device=None, tile=TILE, overlap=OVERLAP):
    """Return a function: uint8 RGB image (H,W,3) -> road probability map (H,W) in [0,1]."""
    device = device or pick_device()
    use_amp = device.type == "cuda"

    @torch.inference_mode()
    def predict_tile(t):
        x = torch.from_numpy(np.ascontiguousarray(t.transpose(2, 0, 1)))[None].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            y = model(x)
        return y[0, 0].float().cpu().numpy()

    def road_fn(img_u8):
        return run_tiled(img_u8.astype(np.float32) / 255.0, predict_tile, tile, overlap)

    return road_fn


def main():
    ap = argparse.ArgumentParser(description="CRNet road extraction")
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default=os.path.join("..", "weights", "crnet_best.pth"))
    ap.add_argument("--out", default="roads.png", help="binary road mask (white = road)")
    ap.add_argument("--prob-out", default="", help="optional: save the probability map too")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()

    dev = pick_device()
    road_fn = make_road_fn(load_crnet(a.weights, dev), dev)
    prob = road_fn(np.array(Image.open(a.image).convert("RGB")))
    Image.fromarray(((prob > a.threshold) * 255).astype(np.uint8)).save(a.out)
    if a.prob_out:
        Image.fromarray((prob * 255).astype(np.uint8)).save(a.prob_out)
    print(f"saved {a.out}  (road pixels: {(prob > a.threshold).mean() * 100:.1f}%)")


if __name__ == "__main__":
    main()
