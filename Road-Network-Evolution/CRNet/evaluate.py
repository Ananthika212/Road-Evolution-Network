"""Measure CRNet (DeepGlobe fine-tuned) on the 1,530 held-out DeepGlobe test tiles, including accuracy.

    python evaluate.py --root "D:\\datasets\\deepglobe\\train"            # folder with *_sat.jpg and *_mask.png
    python evaluate.py --root ... --limit 50                               # quick trial

Uses the same split as the training notebook (sorted file list, random.Random(42).shuffle, first 1530 = test),
input = RGB / 255, threshold 0.5, mask > 128. Pixel counts are pooled over all test tiles.
Needs the DeepGlobe fine-tuned CRNet weights (--weights, default ../weights/crnet_best.pth).
"""
import argparse, glob, os, random, sys, time
import numpy as np
from PIL import Image
from predict import pick_device, load_crnet, make_road_fn

SEED, NUM_TEST = 42, 1530

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--weights", default=os.path.join("..", "weights", "crnet_best.pth"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    pairs = []
    for sat in sorted(glob.glob(os.path.join(a.root, "*_sat.jpg"))):
        m = sat.replace("_sat.jpg", "_mask.png")
        if os.path.exists(m): pairs.append((sat, m))
    if not pairs: sys.exit(f"No *_sat.jpg / *_mask.png pairs in {a.root}")
    pairs = sorted(pairs); random.Random(SEED).shuffle(pairs)
    test = pairs[:NUM_TEST]
    if a.limit: test = test[:a.limit]
    print(f"{len(pairs)} pairs found; evaluating {len(test)} test tiles")
    dev = pick_device()
    fn = make_road_fn(load_crnet(a.weights, dev), dev)
    tp = fp = fn_ = tn = 0; t0 = time.time()
    for i, (s, m) in enumerate(test, 1):
        img = np.array(Image.open(s).convert("RGB"))
        gt = np.array(Image.open(m).convert("L")) > 128
        pr = fn(img) > a.threshold
        tp += int((pr & gt).sum()); fp += int((pr & ~gt).sum()); fn_ += int((~pr & gt).sum()); tn += int((~pr & ~gt).sum())
        if i % 50 == 0: print(f"  {i}/{len(test)}  {time.time()-t0:.0f}s")
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn_, 1); F = 2 * P * R / max(P + R, 1e-9)
    IoU = tp / max(tp + fp + fn_, 1); Acc = (tp + tn) / max(tp + tn + fp + fn_, 1)
    print(f"\nTP {tp:,}  FP {fp:,}  FN {fn_:,}  TN {tn:,}")
    print(f"Precision {P*100:.2f}  Recall {R*100:.2f}  F1 {F*100:.2f}  IoU {IoU*100:.2f}  Accuracy {Acc*100:.2f}")
    print("Report values: P 73.67  R 74.52  F1 74.10  IoU 58.85  Acc not recorded")

if __name__ == "__main__":
    main()
