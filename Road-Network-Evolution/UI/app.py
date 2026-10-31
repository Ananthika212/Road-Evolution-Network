"""ROAD-EVO web app:  uvicorn app:app --port 8000   then open http://localhost:8000

Environment variables (all optional, defaults shown):
  ROADEVO_CRNET   ../weights/crnet_best.pth   DeepGlobe CRNet weights
  ROADEVO_LGMM    ../weights/lgmm_best.pth    LGMM-Net weights (LEVIR-MCI road-only)
"""
import base64
import io
import os
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from skimage.morphology import skeletonize

from config import Config

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_SIDE = 4096                       # larger images would take very long on a laptop
COLORS = {"High": (239, 68, 68), "Medium": (245, 176, 36), "Low": (76, 141, 255)}   # as in the slides
DESCRIPTIONS = {
    "added": "New road segment (construction)",
    "removed": "Road removed or no longer visible",
    "modified": "Road modification (widening / realignment)",
    "uncertain": "Road change (type uncertain)",
}

_engine = None
_state = {"status": "loading", "error": ""}
_lock = threading.Lock()              # one analysis at a time


def get_engine():
    global _engine
    if _engine is None:
        from engine import Engine
        cfg = Config(
            crnet_weights=os.environ.get("ROADEVO_CRNET", os.path.join(HERE, "..", "weights", "crnet_best.pth")),
            lgmm_weights=os.environ.get("ROADEVO_LGMM", os.path.join(HERE, "..", "weights", "lgmm_best.pth")),
        )
        _engine = Engine(cfg)
    return _engine


def _preload():
    try:
        with _lock:
            get_engine()
        _state["status"] = "ready"
    except Exception as e:                       # shown in the UI instead of crashing
        _state["status"], _state["error"] = "error", str(e)


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=_preload, daemon=True).start()
    yield


app = FastAPI(title="ROAD-EVO", lifespan=lifespan)


# ------------------------------------------------------------------ helpers --
def _b64(arr, fmt="PNG", quality=88):
    buf = io.BytesIO()
    im = Image.fromarray(arr)
    if fmt == "JPEG":
        im.convert("RGB").save(buf, format="JPEG", quality=quality)
    else:
        im.save(buf, format="PNG", optimize=False)
    mime = "jpeg" if fmt == "JPEG" else "png"
    return f"data:image/{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def _read(upload: UploadFile):
    try:
        return np.array(Image.open(io.BytesIO(upload.file.read())).convert("RGB"))
    except Exception:
        raise HTTPException(400, f"Could not read '{upload.filename}' as an image (use PNG, JPG or TIFF).")


def _rgba(mask, color, alpha=255):
    out = np.zeros(mask.shape + (4,), np.uint8)
    out[mask] = (*color, alpha)
    return out


def _gray(prob, max_side=None):
    g = np.clip(prob * 255, 0, 255).astype(np.uint8)
    if max_side and max(g.shape) > max_side:
        s = max_side / max(g.shape)
        g = cv2.resize(g, (int(g.shape[1] * s), int(g.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return g


def _thin(mask):
    sk = skeletonize(mask)
    return sk, cv2.dilate(sk.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def _thumb(img, labels, region, color):
    x0, y0, x1, y1 = region["bbox"]
    pad = max(24, int(0.35 * max(x1 - x0, y1 - y0)))
    H, W = img.shape[:2]
    ya, yb, xa, xb = max(0, y0 - pad), min(H, y1 + pad), max(0, x0 - pad), min(W, x1 + pad)
    crop = img[ya:yb, xa:xb].copy()
    m = labels[ya:yb, xa:xb] == region["id"]
    crop[m] = (0.35 * crop[m] + 0.65 * np.array(color)).astype(np.uint8)
    s = 240 / max(crop.shape[:2])
    if s != 1:
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                          interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return _b64(crop, "JPEG", 85)


# ---------------------------------------------------------------------- API --
@app.get("/api/health")
def health():
    return {"status": _state["status"], "error": _state["error"]}


@app.post("/api/analyse")
def analyse(t1: UploadFile = File(None), t2: UploadFile = File(None), gsd: float = Form(0.5),
            location: str = Form(""), t1_date: str = Form(""), t2_date: str = Form(""),
            latitude: float = Form(None), longitude: float = Form(None)):
    if t1 is not None and t2 is not None:
        a, b = _read(t1), _read(t2)
    elif latitude is not None and longitude is not None and t1_date and t2_date:
        import satellite                       # fetch both dates from Earth Engine (same region, same size)
        try:
            a, d1, g1 = satellite.fetch_rgb(latitude, longitude, t1_date)
            b, d2, g2 = satellite.fetch_rgb(latitude, longitude, t2_date)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, str(e)[:300])
        gsd, t1_date, t2_date = g1, d1, d2
        location = location or f"{latitude:.5f}, {longitude:.5f}"
    else:
        raise HTTPException(400, "Upload both images, or fetch high-resolution imagery for a location first.")
    if not (0.05 <= gsd <= 30):
        raise HTTPException(400, "Pixel size must be between 0.05 and 30 metres.")
    if a.shape != b.shape:
        raise HTTPException(400, f"The two images must be the same size and aligned "
                                 f"(got {a.shape[1]}×{a.shape[0]} and {b.shape[1]}×{b.shape[0]}). "
                                 "Export both dates from Earth Engine with the same region and scale.")
    if max(a.shape[:2]) > MAX_SIDE:
        raise HTTPException(400, f"Image is too large ({a.shape[1]}×{a.shape[0]}). Maximum side is {MAX_SIDE} px.")
    with _lock:
        t0 = time.time()
        try:
            eng = get_engine()
            _state["status"] = "ready"
        except Exception as e:
            _state["status"], _state["error"] = "error", str(e)
            raise HTTPException(500, f"Models could not be loaded: {e}")
        res = eng.run(a, b, gsd_m=gsd)
        cfg, secs = eng.cfg, time.time() - t0
    return JSONResponse(build_payload(a, b, res, cfg, gsd, secs, str(eng.device),
                                      location[:120], t1_date[:20], t2_date[:20]))


def build_payload(a, b, res, cfg, gsd, secs, device, location, d1, d2):
    H, W = a.shape[:2]
    sk1, line1 = _thin(res.roads_t1)
    sk2, line2 = _thin(res.roads_t2)
    labels = res.analysis.labels
    road_zone = cv2.dilate((res.roads_t1 | res.roads_t2).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)

    prio = np.zeros((H, W, 4), np.uint8)
    regions = []
    for r in res.regions:
        color = COLORS[r["priority"]]
        m = labels == r["id"]
        seg = m & road_zone
        prio[seg if seg.any() else m] = (*color, 215)
        regions.append({
            **r,
            "description": DESCRIPTIONS.get(r["change_type"], DESCRIPTIONS["uncertain"]),
            "length_km": round(r["affected_road_length_m"] / 1000, 3),
            "thumb": _thumb(b, labels, r, color),
        })

    change_mask = res.analysis.change_mask
    heat = _gray(res.change_prob)
    overlay = b.copy()
    overlay[change_mask] = (0.35 * overlay[change_mask] + 0.65 * np.array([239, 68, 68])).astype(np.uint8)
    counts = {k: sum(1 for r in regions if r["priority"] == k) for k in ("High", "Medium", "Low")}
    n_changed = int(change_mask.sum())
    L1, L2 = float(sk1.sum()) * gsd / 1000, float(sk2.sum()) * gsd / 1000

    return {
        "meta": {"width": W, "height": H, "gsd": gsd, "seconds": round(secs, 1), "device": device,
                 "seconds_roads": round(res.seconds_roads, 1), "seconds_change": round(res.seconds_change, 1),
                 "location": location, "t1_date": d1, "t2_date": d2,
                 "model_input": cfg.lgmm_tile, "crnet_tile": cfg.crnet_tile,
                 "n_outputs": len(res.change_outputs or [])},
        "metrics": {
            "changed_pixels": n_changed, "total_pixels": H * W,
            "change_pct": round(100 * n_changed / (H * W), 2),
            "avg_confidence": round(float(np.mean([r["confidence_C"] for r in regions])), 2) if regions else 0,
            "counts": counts, "total_regions": len(regions),
            "road_px_t1": int(res.roads_t1.sum()), "road_px_t2": int(res.roads_t2.sum()),
            "road_km_t1": round(L1, 2), "road_km_t2": round(L2, 2),
            "road_cov_t1": round(100 * float(res.roads_t1.mean()), 2),
            "road_cov_t2": round(100 * float(res.roads_t2.mean()), 2),
        },
        "images": {
            "t1": _b64(a, "JPEG"), "t2": _b64(b, "JPEG"),
            "roads_t1": _b64(_rgba(line1, (255, 255, 255), 235)), "roads_t2": _b64(_rgba(line2, (255, 255, 255), 235)),
            "priority": _b64(prio),
            "heat": _b64(heat), "mask": _b64(change_mask.astype(np.uint8) * 255),
            "overlay": _b64(overlay, "JPEG"),
            "outputs": [_b64(_gray(o, 512)) for o in (res.change_outputs or [])],
        },
        "regions": regions,
    }


DIST = os.path.join(HERE, "dist")
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
if os.path.isdir(os.path.join(DIST, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST, "assets")), name="assets")


class PreviewRequest(BaseModel):
    latitude: float
    longitude: float
    date: str


@app.post("/api/satellite/preview")
def satellite_preview(req: PreviewRequest):
    """Sentinel-2 thumbnail for the dashboard (optional; needs Google Earth Engine, see satellite.py)."""
    from satellite import preview
    try:
        return preview(req.latitude, req.longitude, req.date)
    except Exception as e:  # noqa: BLE001
        return {"success": False, "message": f"Could not fetch imagery: {str(e)[:160]}"}


@app.get("/classic")
def classic():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/")
def index():
    page = os.path.join(DIST, "index.html")
    return FileResponse(page if os.path.exists(page) else os.path.join(HERE, "static", "index.html"))


@app.get("/favicon.svg")
def favicon():
    f = os.path.join(DIST, "favicon.svg")
    if not os.path.exists(f):
        raise HTTPException(404)
    return FileResponse(f)
