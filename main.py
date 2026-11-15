import os
import numpy as np
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

app = FastAPI(title="SPHEREx Ultra-Fast Blink Viewer")

# Asegurar que la ruta estática exista para servir la caché subida a GitHub
os.makedirs("static/cache", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/api/target")
def get_target_time_series(ra: float = Query(...), dec: float = Query(...)):
    """
    Endpoint ultra-rápido optimizado para Vercel: 
    Busca los fotogramas cacheados directamente en la estructura de archivos estáticos.
    """
    try:
        safe_name = f"ra_{ra:.3f}_dec_{dec:.3f}".replace(".", "_").replace("-", "m")
        target_dir = os.path.join("static", "cache", safe_name)
        
        # Verificar si la caché existe en los archivos subidos a Vercel
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
                return {"status": "success", "source": "vercel_cache", "frames": frames}

        return JSONResponse(status_code=404, content={"error": f"No hay caché local para RA={ra}, Dec={dec}. Asegúrate de subir la carpeta static/cache a GitHub."})
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def read_root():
    return {"message": "SPHEREx Blink API en Vercel funcionando correctamente. Visita /static/index.html"}