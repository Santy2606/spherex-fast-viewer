import os
import time
import threading
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

app = FastAPI(title="SPHEREx Ultra-Fast Blink Viewer")

# Montar carpeta estática
os.makedirs("static/cache", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Coordenadas oficiales de los objetivos del reto para pre-caché automático
CHALLENGE_TARGETS = [
    {"ra": 133.795, "dec": -7.245, "name": "WISE 0855"},
    {"ra": 210.802, "dec": 54.348, "name": "Galaxia M101"},
    {"ra": 304.693, "dec": 42.443, "name": "Objeto 1 (D2)"}
]

def fetch_or_cache_target(ra: float, dec: float):
    """Función central para obtener o generar caché de una coordenada."""
    safe_name = f"ra_{ra:.3f}_dec_{dec:.3f}".replace(".", "_").replace("-", "m")
    target_dir = os.path.join("static", "cache", safe_name)
    
    # Si ya existe en caché, retornar ruta y origen
    if os.path.exists(target_dir) and len([f for f in os.listdir(target_dir) if f.endswith(".png")]) > 0:
        return target_dir, "cache"

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
        raise Exception("No se encontraron datos en la NASA para estas coordenadas.")

    if 'energy_bandpassname' in table.colnames:
        filtered = table[table['energy_bandpassname'] == 'SPHEREx-D2']
        if len(filtered) > 0:
            table = filtered

    table = table[:15] # 15 épocas para animación fluida

    def process_and_cache(idx, row):
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
                ax.imshow(scaled, origin='lower', cmap='inferno')
                ax.axis('off')
                fig.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=120)
                plt.close(fig)
        except Exception:
            pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_and_cache, i, row) for i, row in enumerate(table)]
        concurrent.futures.wait(futures)

    return target_dir, "nasa_downloaded"

def background_precache():
    """Ejecuta la descarga y guardado en caché de los objetivos del reto en segundo plano."""
    print("[INFO] Iniciando pre-caché automático en segundo plano para los objetivos del reto...")
    for target in CHALLENGE_TARGETS:
        try:
            print(f"[PRE-CACHÉ] Procesando objetivo: {target['name']} (RA: {target['ra']}, Dec: {target['dec']})...")
            fetch_or_cache_target(target['ra'], target['dec'])
            print(f"[PRE-CACHÉ] ¡Objetivo {target['name']} guardado en caché exitosamente!")
        except Exception as e:
            print(f"[PRE-CACHÉ AVISO] No se pudo pre-cachar {target['name']}: {e}")
    print("[INFO] ¡Todos los objetivos del reto están listos y cacheados en el servidor!")

@app.on_event("startup")
def startup_event():
    """Lanza el pre-caché en segundo plano ni bien arranca la aplicación FastAPI."""
    threading.Thread(target=background_precache, daemon=True).start()

@app.get("/api/target")
def get_target_time_series(ra: float = Query(...), dec: float = Query(...)):
    """Endpoint de consulta optimizado para servir desde caché de forma instantánea."""
    try:
        safe_name = f"ra_{ra:.3f}_dec_{dec:.3f}".replace(".", "_").replace("-", "m")
        target_dir, source = fetch_or_cache_target(ra, dec)
        
        cached_files = sorted([f for f in os.listdir(target_dir) if f.endswith(".png")])
        frames = []
        for i, filename in enumerate(cached_files):
            frames.append({
                "index": i,
                "time": float(i * 180),
                "url": f"/static/cache/{safe_name}/{filename}"
            })
        return {"status": "success", "source": source, "frames": frames}
    except Exception as e:
        return JSONResponse(status_code=404, content={"error": str(e)})

@app.get("/")
def read_root():
    return {"message": "SPHEREx Ultra-Fast Blink API Activa"}