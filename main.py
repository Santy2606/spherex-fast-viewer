import os
import time
import concurrent.futures
import numpy as np
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.visualization import ImageNormalize, PercentileInterval, SqrtStretch
from astroquery.ipac.irsa import Irsa

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = FastAPI(title="SPHEREx Ultra-Fast & Dynamic Viewer")

os.makedirs("static/cache", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/api/target")
def get_target_time_series(ra: float = Query(...), dec: float = Query(...)):
    """
    Busca en caché local primero (Instantáneo). 
    Si no existe, activa el motor de procesamiento en vivo para la nueva coordenada.
    """
    safe_name = f"ra_{ra:.3f}_dec_{dec:.3f}".replace(".", "_").replace("-", "m")
    target_dir = os.path.join("static", "cache", safe_name)
    
    # 1. SI YA ESTÁ EN CACHÉ (Instantáneo)
    if os.path.exists(target_dir):
        cached_files = sorted([f for f in os.listdir(target_dir) if f.endswith(".png")])
        if len(cached_files) > 0:
            frames = []
            for i, filename in enumerate(cached_files):
                frames.append({
                    "index": i,
                    "time": float(i * 180),
                    "url": f"/static/cache/{safe_name}/{filename}"
                })
            return {"status": "success", "source": "cache", "frames": frames}

    # 2. SI ES UNA NUEVA COORDENADA: Motor de procesamiento en vivo
    try:
        os.makedirs(target_dir, exist_ok=True)
        coord = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
        search_radius = 0.05 * u.deg
        
        table = None
        for attempt in range(3):
            try:
                results = Irsa.query_sia(pos=(coord, search_radius), collection=['spherex_dr1', 'spherex_qr3'])
                if results is not None and len(results) > 0:
                    table = results
                    break
            except Exception:
                time.sleep(2)

        if table is None or len(table) == 0:
            return JSONResponse(status_code=404, content={"error": "No se encontraron datos en la NASA para estas coordenadas."})

        if 'energy_bandpassname' in table.colnames:
            filtered = table[table['energy_bandpassname'] == 'SPHEREx-D2']
            if len(filtered) > 0:
                table = filtered

        table = table[:15] # Limitar a 15 épocas para asegurar fluidez
        image_paths = []

        def process_and_save(idx, row):
            try:
                url = row["access_url"]
                with fits.open(url, memmap=False) as hdulist:
                    image_data = hdulist["IMAGE"].data
                    image_data = np.nan_to_num(image_data, nan=0.0)
                    
                    norm = ImageNormalize(image_data, interval=PercentileInterval(98), stretch=SqrtStretch())
                    scaled = norm(image_data)
                    
                    filename = f"frame_{idx:02d}.png"
                    filepath = os.path.join(target_dir, filename)
                    
                    fig, ax = plt.subplots(figsize=(6, 6), frameon=False)
                    ax.imshow(scaled, origin='lower', cmap='gray')
                    ax.axis('off')
                    fig.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=120)
                    plt.close(fig)
                    
                    timestamp = row["t_min"] if "t_min" in row.colnames else float(idx)
                    return {"index": idx, "time": float(timestamp), "url": f"/static/cache/{safe_name}/{filename}"}
            except Exception:
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_and_save, i, row) for i, row in enumerate(table)]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    image_paths.append(res)

        image_paths = sorted(image_paths, key=lambda x: x["index"])
        
        if len(image_paths) == 0:
            return JSONResponse(status_code=500, content={"error": "Error al procesar los fotogramas FITS."})

        return {"status": "success", "source": "nasa_live_processed", "frames": image_paths}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def read_root():
    return {"message": "SPHEREx Ultra-Fast & Dynamic Viewer Activo"}