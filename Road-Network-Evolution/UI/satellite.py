"""Satellite imagery from Google Earth Engine for the dashboard.

Setup (once):  python -m pip install earthengine-api
               python -c "import ee; ee.Authenticate()"
               set the Google Cloud project:  $env:EE_PROJECT = "your-project-id"

Sources, in order of preference:
  * NAIP (USDA/NAIP/DOQQ): 0.6 m aerial imagery, United States only. This matches the 0.5 m resolution the
    road models were trained on, so these images CAN be analysed (LEVIR-MCI itself is imagery of Texas).
  * Sentinel-2: 10 m, worldwide. Only a visual preview; too coarse for the road models. Upload high-resolution
    images for the analysis instead.
"""
import datetime as dt
import io
import math
import os
import urllib.request

import numpy as np

NAIP_GSD = 0.6
SIZE_PX = 1024                      # fetched analysis images are 1024 x 1024 px  (about 614 m x 614 m at 0.6 m)
WINDOW_DAYS = 3 * 365               # NAIP is flown every 2-3 years: accept the closest image within 3 years


def _ee():
    try:
        import ee
    except ImportError:
        raise RuntimeError("Earth Engine is not installed on this server (python -m pip install earthengine-api). "
                           "You can still upload the T1 and T2 images below.")
    project = os.environ.get("EE_PROJECT") or os.environ.get("ROADEVO_EE_PROJECT")
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Earth Engine is not signed in ({str(e)[:120]}). Run ee.Authenticate() and set EE_PROJECT, "
                           "or upload the T1 and T2 images below.")
    return ee


def _rect(ee, lat, lon, half_m):
    dlat = half_m / 111320.0
    dlon = half_m / (111320.0 * max(0.2, math.cos(math.radians(lat))))
    return ee.Geometry.Rectangle([lon - dlon, lat - dlat, lon + dlon, lat + dlat])


def _naip(ee, lat, lon, date):
    """Closest NAIP image to `date` covering the point, or None."""
    day = dt.date.fromisoformat(date)
    point = ee.Geometry.Point([lon, lat])
    target = ee.Date(day.isoformat())
    col = (ee.ImageCollection("USDA/NAIP/DOQQ").filterBounds(point)
           .filterDate((day - dt.timedelta(days=WINDOW_DAYS)).isoformat(), (day + dt.timedelta(days=WINDOW_DAYS)).isoformat())
           .map(lambda i: i.set("diff", ee.Number(i.date().difference(target, "day")).abs())).sort("diff"))
    if col.size().getInfo() == 0:
        return None
    img = ee.Image(col.first())
    actual = dt.datetime.utcfromtimestamp(img.get("system:time_start").getInfo() / 1000).date().isoformat()
    return img, actual


def _thumb(ee, img, region, dims, bands, vmax):
    return img.select(bands).getThumbURL({"region": region, "dimensions": dims, "min": 0, "max": vmax, "format": "png"})


def fetch_rgb(latitude, longitude, date):
    """High-resolution (NAIP) image for analysis: returns (uint8 RGB array, actual_date, gsd_m)."""
    from PIL import Image
    ee = _ee()
    found = _naip(ee, latitude, longitude, date)
    if not found:
        raise RuntimeError("No high-resolution (NAIP, 0.6 m) imagery exists here in Earth Engine (it covers the United States only). "
                           "Upload your own T1 and T2 images instead.")
    img, actual = found
    region = _rect(ee, latitude, longitude, SIZE_PX * NAIP_GSD / 2)
    url = _thumb(ee, img, region, f"{SIZE_PX}x{SIZE_PX}", ["R", "G", "B"], 255)
    data = urllib.request.urlopen(url, timeout=180).read()
    arr = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    return arr, actual, NAIP_GSD


def preview(latitude, longitude, date, half_window_days=20, half_size_m=1280):
    """Thumbnail for the dashboard: NAIP if available (analysable), else Sentinel-2 (preview only)."""
    try:
        ee = _ee()
    except RuntimeError as e:
        return {"success": False, "message": str(e)}
    found = _naip(ee, latitude, longitude, date)
    if found:
        img, actual = found
        region = _rect(ee, latitude, longitude, SIZE_PX * NAIP_GSD / 2)
        url = _thumb(ee, img, region, "640x640", ["R", "G", "B"], 255)
        return {"success": True, "data": {"thumbnail_url": url, "actual_date": actual, "source": "NAIP 0.6 m",
                                          "analysable": True, "gsd": NAIP_GSD}}
    day = dt.date.fromisoformat(date)
    point = ee.Geometry.Point([longitude, latitude])
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(point)
           .filterDate((day - dt.timedelta(days=half_window_days)).isoformat(), (day + dt.timedelta(days=half_window_days)).isoformat())
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40)).sort("CLOUDY_PIXEL_PERCENTAGE"))
    if col.size().getInfo() == 0:
        return {"success": False, "message": f"No cloud-free Sentinel-2 image within {half_window_days} days of {date} at this location."}
    img = ee.Image(col.first())
    actual = dt.datetime.utcfromtimestamp(img.get("system:time_start").getInfo() / 1000).date().isoformat()
    url = _thumb(ee, img, point.buffer(half_size_m).bounds(), "640x640", ["B4", "B3", "B2"], 3000)
    return {"success": True, "data": {"thumbnail_url": url, "actual_date": actual, "source": "Sentinel-2 10 m (preview only)",
                                      "analysable": False, "gsd": 10}}
