"""
SPHEREx Blink Viewer - backend
Dependencias: pip install fastapi uvicorn numpy astropy astroquery reproject pillow
Ejecutar:     uvicorn main:app --reload
"""
import json
import os
import shutil
import threading
import time
import traceback
import concurrent.futures

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astroquery.ipac.irsa import Irsa
from reproject import reproject_interp
from PIL import Image

PIX = 6.2          # escala de píxel de SPHEREx (arcsec/px)
MAX_FRAMES = 20    # épocas máximas por objetivo, repartidas en toda la línea temporal
CACHE = os.path.join("static", "cache")

os.makedirs(CACHE, exist_ok=True)
app = FastAPI(title="SPHEREx Blink Viewer")
app.mount("/static", StaticFiles(directory="static"), name="static")

_locks, _guard = {}, threading.Lock()
_slots = threading.Semaphore(2)  # máximo 2 procesamientos en vivo a la vez


# ───────────────────────── utilidades ─────────────────────────
def target_wcs(ra, dec, n):
    """Cuadrícula común: TAN centrada en el objetivo, norte arriba, este a la izquierda."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]
    w.wcs.cdelt = [-PIX / 3600, PIX / 3600]
    return w


def query_table(coord, radius):
    for _ in range(3):
        try:
            res = Irsa.query_sia(pos=(coord, radius), collection=["spherex_dr1", "spherex_qr3"])
            if res is not None and len(res) > 0:
                return res
        except Exception:
            time.sleep(2)
    return None


def fetch_frame(row, out_wcs, n):
    """Descarga el FITS y lo reproyecta a la cuadrícula común (alineación entre épocas)."""
    with fits.open(row["access_url"], memmap=False) as hdul:
        hdu = hdul["IMAGE"]
        data = np.asarray(hdu.data, dtype=np.float32)
        arr, _ = reproject_interp((data, WCS(hdu.header, hdul)), out_wcs, shape_out=(n, n))
    arr = arr.astype(np.float32)
    return arr if np.isfinite(arr).mean() > 0.5 else None  # descarta si el objetivo cae fuera


def row_meta(row, i):
    cols, m = row.colnames, {"time": float(i), "date": None, "band": None, "wl_um": None}
    try:
        if "t_min" in cols:
            m["time"] = float(row["t_min"])
            m["date"] = Time(m["time"], format="mjd").iso[:10]
        if "energy_bandpassname" in cols:
            m["band"] = str(row["energy_bandpassname"])
        if "em_min" in cols and "em_max" in cols:  # metros -> micras (verifica estas columnas en tu tabla SIA)
            m["wl_um"] = [round(float(row["em_min"]) * 1e6, 3), round(float(row["em_max"]) * 1e6, 3)]
    except Exception:
        pass
    if not np.isfinite(m["time"]):  # NaN rompería el JSON en el navegador
        m["time"] = float(i)
    return m


def stretch(a, lo, hi):
    t = np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo), 0, 1)
    return np.arcsinh(10 * t) / np.arcsinh(10)


def save_png(a, path):
    Image.fromarray((np.flipud(a) * 255).astype(np.uint8)).save(path)  # flipud: norte arriba


def build(ra, dec, fov, key):
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    table = query_table(coord, (0.05 + fov / 120) * u.deg)
    if table is None:
        raise LookupError("No se encontraron datos de SPHEREx para estas coordenadas.")

    if "energy_bandpassname" in table.colnames:  # un solo detector para comparar épocas homogéneas
        sel = table[table["energy_bandpassname"] == "SPHEREx-D2"]
        if len(sel):
            table = sel
    if "t_min" in table.colnames:
        table.sort("t_min")  # orden cronológico
    if len(table) > MAX_FRAMES:  # repartidas en toda la línea temporal, no solo las primeras
        table = table[np.unique(np.linspace(0, len(table) - 1, MAX_FRAMES).astype(int))]

    n = max(16, int(round(fov * 60 / PIX)))
    out_wcs = target_wcs(ra, dec, n)

    errors = []

    def work(i):
        try:
            return fetch_frame(table[i], out_wcs, n), row_meta(table[i], i)
        except Exception as e:
            traceback.print_exc()  # el detalle completo aparece en la terminal de uvicorn
            errors.append(f"{type(e).__name__}: {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        res = [r for r in ex.map(work, range(len(table))) if r and r[0] is not None]
    if not res:
        raise RuntimeError("No se pudieron procesar los FITS: " + (errors[0] if errors else "ninguna imagen cubre ese punto"))

    stack = np.stack([r[0] for r in res])
    lo, hi = np.nanpercentile(stack, [1, 99.5])          # UN solo estiramiento para toda la serie
    hi = hi if hi > lo else lo + 1.0
    med = np.nanmedian(stack, axis=0)                    # cielo estático
    ds = np.nanpercentile(np.abs(stack - med), 99.5) or 1.0

    final = os.path.join(CACHE, key)
    tmp = final + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)

    frames = []
    for i, (arr, meta) in enumerate(res):
        save_png(stretch(arr, lo, hi), os.path.join(tmp, f"frame_{i:02d}.png"))
        d = np.nan_to_num(arr - med, nan=0.0)
        save_png(0.5 + 0.5 * np.clip(d / ds, -1, 1), os.path.join(tmp, f"diff_{i:02d}.png"))
        frames.append({"index": i, **meta,
                       "url": f"/static/cache/{key}/frame_{i:02d}.png",
                       "diff_url": f"/static/cache/{key}/diff_{i:02d}.png"})

    with open(os.path.join(tmp, "meta.json"), "w") as f:
        json.dump({"frames": frames}, f)
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)  # la caché solo existe cuando está completa
    return frames


# ───────────────────────── API ─────────────────────────
@app.get("/api/target")
def get_target(
    ra: float = Query(..., ge=0, le=360),
    dec: float = Query(..., ge=-90, le=90),
    fov: float = Query(10, ge=2, le=60, description="Campo de visión en arcmin"),
):
    fov = int(round(fov))
    key = f"ra_{ra:.3f}_dec_{dec:.3f}_fov_{fov}".replace(".", "_").replace("-", "m")
    meta_path = os.path.join(CACHE, key, "meta.json")

    def cached():
        with open(meta_path) as f:
            return {"status": "success", "source": "cache", "fov": fov, "pixel_scale": PIX,
                    "frames": json.load(f)["frames"]}

    if os.path.exists(meta_path):
        return cached()

    with _guard:
        lock = _locks.setdefault(key, threading.Lock())
    with lock:                       # evita procesar dos veces el mismo objetivo
        if os.path.exists(meta_path):
            return cached()
        try:
            with _slots:
                frames = build(ra, dec, fov, key)
        except LookupError as e:
            return JSONResponse(status_code=404, content={"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})
    return {"status": "success", "source": "nasa_live_processed", "fov": fov, "pixel_scale": PIX, "frames": frames}


@app.get("/")
def read_root():
    return FileResponse(os.path.join("static", "index.html"))
