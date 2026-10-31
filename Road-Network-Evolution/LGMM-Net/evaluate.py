"""Check that the LGMM-Net inference reproduces the trained model on the LEVIR-MCI test set.

    python evaluate.py --root "D:\\datasets\\LEVIR-MCI-dataset"
    python evaluate.py --root ... --limit 100          # quick trial on the first 100 pairs

Expected folder layout (as released):  <root>/images/test/A, .../B, .../label
(also accepted: <root>/test/A, B, label).  The label values are printed for the first file; if the road class is
not 1, pass --road-value N.  Result: precision / recall / F1 / IoU / accuracy of the road-change class,
to compare with the report (P 86.23, R 70.04, F1 77.30, IoU 63.00, Acc 98.63).
"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

from predict import pick_device, load_lgmm, ChangeModel


def find_dirs(root):
    for base in (os.path.join(root, "images", "test"), os.path.join(root, "test"), root):
        a, b, l = (os.path.join(base, x) for x in ("A", "B", "label"))
        if all(os.path.isdir(p) for p in (a, b, l)):
            return a, b, l
    sys.exit(f"Could not find A, B and label folders under {root} (looked in images/test, test and the root).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--road-value", type=int, default=1, help="label value of the road-change class")
    ap.add_argument("--weights", default=os.path.join("..", "weights", "lgmm_best.pth"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()

    dir_a, dir_b, dir_l = find_dirs(a.root)
    names = sorted(f for f in os.listdir(dir_l) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")))
    if a.limit:
        names = names[:a.limit]
    print(f"{len(names)} test pairs from {dir_a}")

    dev = pick_device()
    change_fn = ChangeModel(load_lgmm(a.weights, dev), dev)

    tp = fp = fn = tn = 0
    t0 = time.time()
    for i, n in enumerate(names, 1):
        t1 = np.array(Image.open(os.path.join(dir_a, n)).convert("RGB"))
        t2 = np.array(Image.open(os.path.join(dir_b, n)).convert("RGB"))
        lab = np.array(Image.open(os.path.join(dir_l, n)))
        if lab.ndim == 3:
            lab = lab[..., 0] if lab[..., 0].max() <= 2 else lab.max(axis=2)
        if i == 1:
            print("label values in the first file:", np.unique(lab).tolist(), "(road class =", a.road_value, ")")
        gt = lab == a.road_value
        pr = change_fn(t1, t2) >= a.threshold
        tp += int((gt & pr).sum()); fp += int((~gt & pr).sum())
        fn += int((gt & ~pr).sum()); tn += int((~gt & ~pr).sum())
        if i % 25 == 0 or i == len(names):
            print(f"  {i}/{len(names)}  {time.time() - t0:.0f}s")

    eps = 1e-9
    P = tp / (tp + fp + eps); R = tp / (tp + fn + eps)
    F1 = 2 * P * R / (P + R + eps); IoU = tp / (tp + fp + fn + eps); Acc = (tp + tn) / (tp + tn + fp + fn)
    print("\nROAD-CHANGE CLASS, pixels pooled over all test pairs")
    print(f"  Precision {100*P:.2f}   Recall {100*R:.2f}   F1 {100*F1:.2f}   IoU {100*IoU:.2f}   Accuracy {100*Acc:.2f}")
    print("  Report (Table 6.2): P 86.23  R 70.04  F1 77.30  IoU 63.00  Acc 98.63")
    print("\nWithin about 1 point of the report = the app reproduces the trained model.")
    print("Much lower = the input convention is wrong (check road value, BGR order, normalisation).")


if __name__ == "__main__":
    main()
