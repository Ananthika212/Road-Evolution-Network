"""
crnet_deepglobe_finetune.py
===========================
Fine-tune CRNet (starting from the Massachusetts checkpoint) on the
DeepGlobe Road Extraction dataset, on Kaggle.

Three modes (set MODE below):
  "zeroshot" - evaluate the Massachusetts checkpoint as-is on the DeepGlobe
               test split (domain-gap baseline, ~10 min).
  "train"    - fine-tune on DeepGlobe. Saves last.pth every epoch (for
               resuming in a new Kaggle session) and best.pth on val IoU.
  "test"     - evaluate best.pth on the DeepGlobe test split.
  "both"     - run "zeroshot" then "test" in one go (baseline vs fine-tuned).

DeepGlobe notes:
  - Only the official train/ folder has masks (valid/ and test/ masks were
    never released), so we split the 6226 labelled tiles ourselves:
    1530 test (standard split used in the literature) + VAL_SIZE val +
    the rest for training. The split is a seeded shuffle of the sorted
    file list, so it is identical in every session.
  - Tiles are already 1024x1024 at 0.5 m/px, which is exactly what SSFRM
    needs and matches LEVIR imagery.
  - Masks are binarised with > 128 because they are not pure 0/255.

Requires model.py (the CRNet definition) next to this file. On Kaggle, upload model.py as a dataset or utility script.
"""

import os
import glob
import json
import math
import time
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# On Kaggle, model.py can live in an input dataset; add its folder to the path
import sys
for _model_file in glob.glob("/kaggle/input/**/model.py", recursive=True):
    sys.path.append(os.path.dirname(_model_file))
    break
from model import CRNet, IMAGE_SIZE


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODE = "test"  # "zeroshot" | "train" | "test" | "both" (zeroshot + test)

DEEPGLOBE_ROOT = "/kaggle/input/deepglobe-road-extraction-dataset/train"
MASS_CKPT = "/kaggle/input/crnet-massachusetts/crnet_best.pth"   # your Massachusetts checkpoint
RESUME_CKPT = ""   # last.pth from a previous session; "" = start fresh from MASS_CKPT
OUT_DIR = "/kaggle/working"

# Must match how the Massachusetts model was trained, or the pretrained
# weights see differently-distributed input.
CHANNEL_ORDER = "RGB"          # "RGB" if you used PIL / converted cv2, "BGR" if raw cv2.imread
NORMALIZE = False               # False if you only divided by 255
MEAN = (0.485, 0.456, 0.406)   # ImageNet stats (in RGB order)
STD = (0.229, 0.224, 0.225)

NUM_TEST = 1530
VAL_SIZE = 300
SEED = 42

EPOCHS = 15
BATCH_SIZE = 2          # 1024x1024 inputs; 2 fits a 16 GB T4/P100
ACCUM_STEPS = 4         # effective batch = BATCH_SIZE * ACCUM_STEPS
BASE_LR = 1e-4          # set to ~1/10 of the LR you used for Massachusetts
WEIGHT_DECAY = 1e-4
FREEZE_EPOCHS = 1       # train decoder only for the first N epochs
PATIENCE = 4            # early stopping on val IoU
USE_AMP = True          # set False if you see NaN losses
NUM_WORKERS = 2
TIME_LIMIT_H = 11.3     # stop cleanly before Kaggle's 12 h session limit

ENCODER_MODULES = ["firstconv", "firstbn", "encoder1", "encoder2", "encoder3", "encoder4"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def locate_deepglobe_train(root):
    # Kaggle mounts datasets under names that don't always match the slug,
    # so fall back to searching /kaggle/input for the folder holding *_mask.png files
    if glob.glob(os.path.join(root, "*_mask.png")):
        return root
    for dirpath, _, filenames in os.walk("/kaggle/input"):
        if any(f.endswith("_mask.png") for f in filenames):
            print(f"DeepGlobe train folder not at {root}; using {dirpath}")
            return dirpath
    raise FileNotFoundError(f"No folder with *_mask.png files found at {root} or under /kaggle/input")


def locate_checkpoint(path, hint="massachusetts"):
    if os.path.exists(path):
        return path
    candidates = [p for p in glob.glob("/kaggle/input/**/*.pth", recursive=True) if hint in p.lower()]
    if not candidates:
        raise FileNotFoundError(f"{path} not found and no .pth with '{hint}' in its path under /kaggle/input")
    print(f"Checkpoint not at {path}; using {candidates[0]} (others: {candidates[1:]})")
    return candidates[0]


def find_pairs(root):
    root = locate_deepglobe_train(root)
    pairs = []
    for sat in sorted(glob.glob(os.path.join(root, "*_sat.jpg"))):
        mask = sat.replace("_sat.jpg", "_mask.png")
        if os.path.exists(mask):
            pairs.append((sat, mask))
    if not pairs:
        raise FileNotFoundError(f"No *_sat.jpg / *_mask.png pairs found in {root}")
    print(f"Found {len(pairs)} DeepGlobe image/mask pairs in {root}")
    return pairs


def make_split(pairs):
    pairs = sorted(pairs)
    random.Random(SEED).shuffle(pairs)
    test = pairs[:NUM_TEST]
    val = pairs[NUM_TEST:NUM_TEST + VAL_SIZE]
    train = pairs[NUM_TEST + VAL_SIZE:]

    with open(os.path.join(OUT_DIR, "deepglobe_split.json"), "w") as f:
        json.dump({k: [os.path.basename(s) for s, _ in v]
                   for k, v in (("train", train), ("val", val), ("test", test))}, f, indent=1)
    print(f"Split: train {len(train)}, val {len(val)}, test {len(test)}")
    return train, val, test


class DeepGlobeRoads(Dataset):
    def __init__(self, pairs, augment=False):
        self.pairs = pairs
        self.augment = augment
        self.mean = np.array(MEAN, dtype=np.float32)
        self.std = np.array(STD, dtype=np.float32)
        if CHANNEL_ORDER == "BGR":
            self.mean, self.std = self.mean[::-1].copy(), self.std[::-1].copy()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sat_path, mask_path = self.pairs[idx]
        img = cv2.imread(sat_path, cv2.IMREAD_COLOR)          # BGR
        if CHANNEL_ORDER == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

        mask = (mask > 128).astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                img, mask = img[::-1], mask[::-1]
            k = random.randint(0, 3)
            img, mask = np.rot90(img, k), np.rot90(mask, k)

        img = img.astype(np.float32) / 255.0
        if self.augment:
            # mild brightness/contrast jitter
            img = np.clip((img - 0.5) * random.uniform(0.9, 1.1) + 0.5 + random.uniform(-0.05, 0.05), 0, 1)
        if NORMALIZE:
            img = (img - self.mean) / self.std

        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        mask = torch.from_numpy(np.ascontiguousarray(mask))[None]
        return img, mask


def make_loader(pairs, train):
    return DataLoader(DeepGlobeRoads(pairs, augment=train), batch_size=BATCH_SIZE if train else 1,
                      shuffle=train, num_workers=NUM_WORKERS, pin_memory=True, drop_last=train)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def extract_state_dict(ckpt):
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
    # strip DataParallel prefix if the checkpoint was saved from nn.DataParallel
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}


def load_weights(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(extract_state_dict(ckpt), strict=True)
    print(f"Loaded weights from {path}")
    return ckpt


def set_encoder_frozen(model, frozen):
    for name in ENCODER_MODULES:
        for p in getattr(model, name).parameters():
            p.requires_grad = not frozen


def freeze_encoder_bn(model):
    # keep BatchNorm running stats fixed while the encoder is frozen
    for name in ENCODER_MODULES:
        getattr(model, name).eval()


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------
def dice_loss(pred, target, eps=1.0):
    num = 2 * (pred * target).sum(dim=(1, 2, 3)) + eps
    den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


def criterion(pred, target):
    # CRNet already applies sigmoid, so use plain BCE on probabilities.
    # Called outside autocast: F.binary_cross_entropy refuses fp16 inputs.
    pred = pred.float().clamp(1e-6, 1 - 1e-6)
    return F.binary_cross_entropy(pred, target) + dice_loss(pred, target)


@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    model.eval()
    tp = fp = fn = 0
    for img, mask in loader:
        img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP and device.type == "cuda"):
            pred = model(img)
        p = pred.float() > threshold
        t = mask > 0.5
        tp += (p & t).sum().item()
        fp += (p & ~t).sum().item()
        fn += (~p & t).sum().item()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return {
        "iou": tp / (tp + fp + fn + 1e-9),
        "f1": 2 * precision * recall / (precision + recall + 1e-9),
        "precision": precision,
        "recall": recall,
    }


def fmt(m):
    return "  ".join(f"{k} {v * 100:.2f}%" for k, v in m.items())


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def evaluate_checkpoint(path, test_pairs, tag):
    model = CRNet(num_classes=1, pretrained=False).to(device)
    load_weights(model, path)
    metrics = evaluate(model, make_loader(test_pairs, train=False))
    print(f"[{tag}] DeepGlobe test: {fmt(metrics)}")
    with open(os.path.join(OUT_DIR, f"metrics_{tag}.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def train(train_pairs, val_pairs):
    start_time = time.time()
    train_loader = make_loader(train_pairs, train=True)
    val_loader = make_loader(val_pairs, train=False)

    model = CRNet(num_classes=1, pretrained=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: 0.5 * (1 + math.cos(math.pi * e / EPOCHS)))
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")

    start_epoch, best_iou, bad_epochs = 0, 0.0, 0
    if RESUME_CKPT:
        ckpt = load_weights(model, RESUME_CKPT)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_iou, bad_epochs = ckpt["epoch"] + 1, ckpt["best_iou"], ckpt["bad_epochs"]
        print(f"Resuming at epoch {start_epoch}, best val IoU so far {best_iou * 100:.2f}%")
    else:
        # fresh optimizer at a lower LR; the Massachusetts optimizer state is not reused
        load_weights(model, MASS_CKPT)

    history_path = os.path.join(OUT_DIR, "history.json")
    history = json.load(open(history_path)) if RESUME_CKPT and os.path.exists(history_path) else []

    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        frozen = epoch < FREEZE_EPOCHS
        set_encoder_frozen(model, frozen)
        model.train()
        if frozen:
            freeze_encoder_bn(model)

        running, n_batches, skipped = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, (img, mask) in enumerate(train_loader):
            img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                pred = model(img)
            loss = criterion(pred, mask)
            if not torch.isfinite(loss):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss / ACCUM_STEPS).backward()

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item()
            n_batches += 1
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {loss.item():.4f}")
        scheduler.step()

        val = evaluate(model, val_loader)
        train_loss = running / max(n_batches, 1)
        print(f"Epoch {epoch} ({'encoder frozen' if frozen else 'full'}) "
              f"train loss {train_loss:.4f} | val {fmt(val)} | {(time.time() - epoch_start) / 60:.1f} min"
              + (f" | skipped {skipped} non-finite batches" if skipped else ""))
        history.append({"epoch": epoch, "train_loss": train_loss, **val})
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        if val["iou"] > best_iou:
            best_iou, bad_epochs = val["iou"], 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val": val},
                       os.path.join(OUT_DIR, "best.pth"))
            print(f"  new best val IoU {best_iou * 100:.2f}% -> best.pth")
        else:
            bad_epochs += 1

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_iou": best_iou,
            "bad_epochs": bad_epochs,
        }, os.path.join(OUT_DIR, "last.pth"))

        if bad_epochs >= PATIENCE:
            print(f"Early stopping: no val improvement for {PATIENCE} epochs")
            break

        elapsed = time.time() - start_time
        if elapsed + 1.1 * (time.time() - epoch_start) > TIME_LIMIT_H * 3600:
            print("Stopping before the Kaggle time limit. Download last.pth and best.pth, "
                  "upload them as a dataset and set RESUME_CKPT to continue.")
            break


def locate_finetuned_checkpoint():
    """Find best.pth: /kaggle/working first, then any attached input."""
    local = os.path.join(OUT_DIR, "best.pth")
    if os.path.exists(local):
        return local
    found = glob.glob("/kaggle/input/**/best.pth", recursive=True)
    if not found:
        raise FileNotFoundError("best.pth not found in /kaggle/working or /kaggle/input - "
                                "add the finished training version's output as an input")
    print("Using", found[0])
    return found[0]


if __name__ == "__main__":
    seed_everything(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    if MODE in ("zeroshot", "both") and not RESUME_CKPT:
        MASS_CKPT = locate_checkpoint(MASS_CKPT)
    elif MODE == "train" and not RESUME_CKPT:
        MASS_CKPT = locate_checkpoint(MASS_CKPT)
    train_pairs, val_pairs, test_pairs = make_split(find_pairs(DEEPGLOBE_ROOT))

    if MODE == "zeroshot":
        evaluate_checkpoint(MASS_CKPT, test_pairs, "zeroshot_massachusetts")
    elif MODE == "train":
        train(train_pairs, val_pairs)
    elif MODE == "test":
        evaluate_checkpoint(locate_finetuned_checkpoint(), test_pairs, "finetuned_deepglobe")
    elif MODE == "both":
        # baseline (Massachusetts weights) and fine-tuned weights on the same test split
        evaluate_checkpoint(MASS_CKPT, test_pairs, "zeroshot_massachusetts")
        evaluate_checkpoint(locate_finetuned_checkpoint(), test_pairs, "finetuned_deepglobe")
    else:
        raise ValueError(f"Unknown MODE {MODE!r}")
