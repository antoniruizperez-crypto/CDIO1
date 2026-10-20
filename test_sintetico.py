"""
Prueba de humo con datos sintéticos (no necesita descargar nada).

Crea un proyecto temporal con una costa recta diagonal y tres fechas en las que
la costa se desplaza una cantidad CONOCIDA (+7.3 m mar adentro, 0 m, -12.0 m
tierra adentro). Ejecuta extracción -> transectos -> validación y comprueba que
los resultados coinciden con lo esperado (signo y magnitud, con precisión subpíxel).

Uso:   python test_sintetico.py          (borra el proyecto temporal al acabar)
       python test_sintetico.py --keep   (lo conserva en projects/_test_sintetico)
"""

import json
import shutil
import sys

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

import calculate_erosion
import config
import error_estimator
import extract_shorelines
import shoreline_utils as utils

PROJ = "_test_sintetico"
X0, Y0, PX, N = 430000.0, 4584000.0, 10.0, 200       # esquina sup-izq, píxel, tamaño (200x200 px)
THETA = np.deg2rad(30)                                # normal hacia el agua
NORMAL = np.array([np.cos(THETA), np.sin(THETA)])
CENTER = np.array([X0 + N * PX / 2, Y0 - N * PX / 2])
SCENES = [("20260601", 0.0), ("20260701", 7.3), ("20260801", -12.0)]  # (fecha, desplazamiento m)
rng = np.random.default_rng(0)


def make_ndwi(shift):
    cols, rows = np.meshgrid(np.arange(N), np.arange(N))
    x = X0 + (cols + 0.5) * PX
    y = Y0 - (rows + 0.5) * PX
    s = (x - CENTER[0]) * NORMAL[0] + (y - CENTER[1]) * NORMAL[1] - shift   # >0 agua
    ndwi = 0.2 + 0.6 * np.tanh(s / 20.0) + rng.normal(0, 0.02, s.shape)
    ndwi[40:42, 150:152] = -0.3     # mancha de tierra en el agua (debe eliminarse)
    ndwi[150:152, 40:42] = 0.6      # mancha de agua en tierra (debe eliminarse)
    return ndwi.astype("float32")


def main(keep=False):
    pdir = utils.project_path(PROJ)
    if pdir.exists():
        shutil.rmtree(pdir)
    (pdir / "input").mkdir(parents=True)
    ndwi_dir = pdir / "output" / "estimated_ndwi"
    ndwi_dir.mkdir(parents=True)
    gt_dir = pdir / "ground_truth"
    gt_dir.mkdir()

    # AOI: el raster menos 100 m por cada lado
    corners = [(X0 + 100, Y0 - 100), (X0 + N * PX - 100, Y0 - 100),
               (X0 + N * PX - 100, Y0 - N * PX + 100), (X0 + 100, Y0 - N * PX + 100)]
    tr = Transformer.from_crs(32631, 4326, always_xy=True)
    ring = [list(tr.transform(*c)) for c in corners]
    ring.append(ring[0])
    with open(pdir / "input" / "polygon.geojson", "w") as f:
        json.dump({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
                   "geometry": {"type": "Polygon", "coordinates": [ring]}}]}, f)

    for date, shift in SCENES:
        name = f"S2A_MSIL2A_{date}T103031_N0511_R008_T31TDF_{date}T120000_ndwi.tif"
        with rasterio.open(ndwi_dir / name, "w", driver="GTiff", height=N, width=N, count=1,
                           dtype="float32", crs="EPSG:32631", transform=from_origin(X0, Y0, PX, PX),
                           nodata=np.nan) as dst:
            dst.write(make_ndwi(shift), 1)
            dst.update_tags(CLOUD_PCT_AOI="1.0")

    # Ground truth = costa real del 2026-07-01, con el sentido de recorrido INVERTIDO a propósito
    t = np.arange(-900, 900.1, 5.0)
    tang = np.array([-NORMAL[1], NORMAL[0]])
    gt = CENTER[None, :] + 7.3 * NORMAL[None, :] + t[:, None] * tang[None, :]
    gt = gt[::-1]
    with open(gt_dir / "gt_20260701.csv", "w") as f:
        f.write("x\ty\n")
        for x, y in gt:
            f.write(f"{x:.3f}\t{y:.3f}\n")

    print("== 1. Extracción ==")
    _, params = extract_shorelines.main(PROJ, threshold_mode="fixed", threshold_fixed=0.2)
    print("== 2. Erosión / acreción ==")
    _, summary = calculate_erosion.main(PROJ, show_plot=False)
    print("== 3. Validación contra ground truth ==")
    _, metrics = error_estimator.main(PROJ, gt_dir, "2026-07-01", show_plot=False)

    expected = {"2026-06-01": 0.0, "2026-07-01": 7.3, "2026-08-01": -12.0}
    ok = True
    print("\n== Comprobaciones ==")
    for _, r in summary.iterrows():
        e = expected[r["date"]]
        good = abs(r["avg_distance_m"] - e) < 1.0 and r["coverage_ratio"] > 0.95
        ok &= good
        print(f"{'OK ' if good else 'FALLO'} {r['date']}: medido {r['avg_distance_m']:+.2f} m, esperado {e:+.1f} m, "
              f"cobertura {r['coverage_ratio']:.2f}")
    good = (params["n_segments"] == 1).all()
    ok &= good
    print(f"{'OK ' if good else 'FALLO'} manchas eliminadas: segmentos por fecha = {params['n_segments'].tolist()} (esperado 1)")
    good = abs(metrics["bias_m"]) < 1.0 and metrics["rmse_m"] < 1.5
    ok &= good
    print(f"{'OK ' if good else 'FALLO'} validación: sesgo {metrics['bias_m']:+.2f} m, RMSE {metrics['rmse_m']:.2f} m (esperado ~0)")

    if not keep:
        shutil.rmtree(pdir)
    print("\nRESULTADO:", "TODO CORRECTO" if ok else "HAY FALLOS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(keep="--keep" in sys.argv))
