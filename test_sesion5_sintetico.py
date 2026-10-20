"""
Prueba de humo de sesion5.py con un producto Sentinel-2 L2A FALSO (zip con
bandas .jp2 sintéticas, SCL, XML con BOA_ADD_OFFSET y un parche de nube).
Comprueba que el recorte al AOI, la lectura dentro del zip, la máscara de
nubes y la escritura de salidas funcionan, y encadena la extracción.

Uso: python test_sesion5_sintetico.py
"""

import json
import os
import shutil
import sys
import tempfile
import zipfile

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

import config
import extract_shorelines
import sesion5
import shoreline_utils as utils

PROJ = "_test_sesion5"
X0, Y0, N = 430000.0, 4584000.0, 400          # 4 km x 4 km a 10 m
THETA = np.deg2rad(30)


def write_jp2(path, data, px):
    with rasterio.open(path, "w", driver="JP2OpenJPEG", height=data.shape[0], width=data.shape[1], count=1,
                       dtype=data.dtype, crs="EPSG:32631", transform=from_origin(X0, Y0, px, px),
                       QUALITY=100, REVERSIBLE="YES") as dst:
        dst.write(data, 1)


def main():
    tmp = tempfile.mkdtemp()
    pdir = utils.project_path(PROJ)
    shutil.rmtree(pdir, ignore_errors=True)
    (pdir / "input").mkdir(parents=True)
    # Redirigir carpetas de imágenes a un directorio temporal
    config.IMAGES_DIR = config.BASE_DIR / "Imagenes_test"
    for name in ("ENTRADA", "PROCESADOS", "ERROR", "SALIDA"):
        setattr(config, f"CARPETA_{name}", config.IMAGES_DIR / name.lower())
    config.CSV_LOG = config.IMAGES_DIR / "registro.csv"
    shutil.rmtree(config.IMAGES_DIR, ignore_errors=True)
    config.CARPETA_ENTRADA.mkdir(parents=True)

    # reflectancias sintéticas
    cols, rows = np.meshgrid(np.arange(N), np.arange(N))
    x, y = X0 + (cols + .5) * 10, Y0 - (rows + .5) * 10
    s = (x - (X0 + 2000)) * np.cos(THETA) + (y - (Y0 - 2000)) * np.sin(THETA)
    w = 0.5 * (1 + np.tanh(s / 15))
    G, NIR = 0.10 - 0.04 * w, 0.30 - 0.28 * w
    to_dn = lambda r: np.clip(r * 10000 + 1000, 1, 65535).astype("uint16")   # baseline >= 4: +1000
    scl = np.full((N // 2, N // 2), 5, dtype="uint8")
    scl[100:108, 100:108] = 9                                            # parche de nube dentro del AOI

    name = "S2A_MSIL2A_20260601T103031_N0511_R008_T31TDF_20260601T120000"
    root = os.path.join(tmp, name + ".SAFE")
    r10, r20 = os.path.join(root, "IMG_DATA", "R10m"), os.path.join(root, "IMG_DATA", "R20m")
    os.makedirs(r10)
    os.makedirs(r20)
    for b, arr in {"B02": G, "B03": G, "B04": G, "B08": NIR}.items():
        write_jp2(os.path.join(r10, f"T31TDF_20260601T103031_{b}_10m.jp2"), to_dn(arr), 10)
    write_jp2(os.path.join(r20, "T31TDF_20260601T103031_SCL_20m.jp2"), scl, 20)
    with open(os.path.join(root, "MTD_MSIL2A.xml"), "w") as f:
        f.write('<root><BOA_QUANTIFICATION_VALUE unit="none">10000</BOA_QUANTIFICATION_VALUE>'
                '<BOA_ADD_OFFSET band_id="2">-1000</BOA_ADD_OFFSET></root>')
    zpath = config.CARPETA_ENTRADA / (name + ".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for d, _, files in os.walk(root):
            for fn in files:
                full = os.path.join(d, fn)
                z.write(full, os.path.relpath(full, tmp))

    # AOI: 2 km centrales
    c = [(X0 + 1000, Y0 - 1000), (X0 + 3000, Y0 - 1000), (X0 + 3000, Y0 - 3000), (X0 + 1000, Y0 - 3000)]
    tr = Transformer.from_crs(32631, 4326, always_xy=True)
    ring = [list(tr.transform(*p)) for p in c]
    ring.append(ring[0])
    with open(pdir / "input" / "polygon.geojson", "w") as f:
        json.dump({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
                   "geometry": {"type": "Polygon", "coordinates": [ring]}}]}, f)

    print("== sesion5 ==")
    sesion5.main(PROJ)
    ok = True
    tif = pdir / "output" / "estimated_ndwi" / (name + "_ndwi.tif")
    ok &= tif.exists()
    print("OK " if tif.exists() else "FALLO", "NDWI GeoTIFF creado")
    if tif.exists():
        with rasterio.open(tif) as src:
            a = src.read(1)
            print(f"   recorte: {a.shape} px (la escena completa sería {N}x{N}); NaN (nubes): {int(np.isnan(a).sum())} px; "
                  f"offset BOA: {src.tags().get('BOA_OFFSET')}")
            ok &= a.shape[0] < N and np.isnan(a).sum() > 0 and src.tags().get("BOA_OFFSET") == "-1000"
    moved = (config.CARPETA_PROCESADOS / (name + ".zip")).exists()
    ok &= moved
    print("OK " if moved else "FALLO", "zip movido a procesados")

    print("== extracción ==")
    _, params = extract_shorelines.main(PROJ)
    n = int(params["n_segments"].iloc[0])
    ok &= n >= 1
    print("OK " if n >= 1 else "FALLO", f"segmentos de costa: {n}, longitud {params['total_length_m'].iloc[0]} m, "
          f"umbral {params['ndwi_threshold'].iloc[0]} ({params['threshold_method'].iloc[0]})")

    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(pdir, ignore_errors=True)
    shutil.rmtree(config.IMAGES_DIR, ignore_errors=True)
    print("\nRESULTADO:", "TODO CORRECTO" if ok else "HAY FALLOS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
