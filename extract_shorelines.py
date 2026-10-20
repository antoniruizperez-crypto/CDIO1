"""
Extracción de la línea de costa con resolución SUBPÍXEL.

En vez de marcar los píxeles de agua que tocan tierra (error sistemático de
~5 m y puntos desordenados), se calcula el contorno del NDWI continuo en el
nivel del umbral (marching squares). El resultado:
  - posición interpolada dentro del píxel (precisión de pocos metros),
  - polilíneas ya ORDENADAS (no hace falta reconstruir la costa por vecino
    más cercano ni se unen tramos separados),
  - cada segmento orientado con el agua a la izquierda (ver shoreline_utils).

Limpieza de ruido: se eliminan manchas pequeñas de agua EN tierra y de tierra
EN agua por ÁREA (no por morfología), de modo que no se borran espigones
estrechos pero largos. El closing clásico es opcional (--closing-radius).

Entrada : projects/<p>/output/estimated_ndwi/*.tif   (los genera sesion5.py)
Salida  : projects/<p>/output/estimated_waterbodies_edges/
            all_shorelines.csv, extraction_params.csv, shorelines_meta.json,
            <escena>.kmz/.csv/_xyz.csv/.geojson
          projects/<p>/output/estimated_waterbodies_images_clean/<escena>_clean.tif
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask
from scipy import ndimage as ndi
from shapely.geometry import LineString
from skimage.measure import find_contours
from skimage.morphology import closing, disk
from tqdm import tqdm

import config
import shoreline_utils as utils


# ---------------------------------------------------------------------------
# Limpieza de la máscara
# ---------------------------------------------------------------------------

def drop_small_components(binary, min_px, protected):
    """Elimina componentes True con menos de min_px píxeles, salvo las que tocan 'protected'."""
    if min_px <= 1 or not binary.any():
        return binary
    lab, n = ndi.label(binary, structure=np.ones((3, 3), bool))
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    remove = sizes < min_px
    remove[0] = False
    keep_ids = np.unique(lab[protected & (lab > 0)])
    remove[keep_ids] = False
    return binary & ~remove[lab]


def clean_water_mask(ndwi, thr, valid, min_area_px, closing_radius):
    """Devuelve (agua_cruda, agua_limpia) como arrays booleanos."""
    ndwi_f = np.where(valid, ndwi, -9.0)
    water_raw = valid & (ndwi_f > thr)
    water = water_raw.copy()
    if closing_radius > 0:
        water = closing(water, disk(closing_radius)) & valid

    # Los componentes que tocan el borde del recorte o píxeles inválidos pueden
    # ser trozos de algo grande: no se tocan.
    protected = np.zeros(valid.shape, bool)
    protected[0, :] = protected[-1, :] = protected[:, 0] = protected[:, -1] = True
    protected |= ndi.binary_dilation(~valid, iterations=1)

    water = drop_small_components(water, min_area_px, protected)      # agua suelta en tierra
    land = drop_small_components(valid & ~water, min_area_px, protected)  # tierra suelta en agua
    water_clean = valid & ~land
    return water_raw, water_clean


def adjust_ndwi(ndwi, thr, water_raw, water_clean, margin):
    """NDWI corregido: los píxeles cuya clase cambió pasan al lado correcto del umbral."""
    adj = ndwi.copy()
    adj[water_clean & ~water_raw] = thr + margin
    adj[~water_clean & water_raw] = thr - margin
    return adj


# ---------------------------------------------------------------------------
# Contornos
# ---------------------------------------------------------------------------

def contour_to_xy(contour, transform):
    """(fila, col) subpíxel de find_contours -> coordenadas del mapa (centro de píxel = índice entero)."""
    rows, cols = contour[:, 0] + 0.5, contour[:, 1] + 0.5
    x = transform.c + transform.a * cols + transform.b * rows
    y = transform.f + transform.d * cols + transform.e * rows
    return np.column_stack([x, y])


def orient_water_left(coords, state, transform, offset_m, max_samples=25):
    """
    Devuelve (coords, ambiguo). Orienta el segmento para que el agua quede a la
    izquierda, muestreando la máscara limpia a ambos lados (voto por mayoría).
    state: 1 agua, 0 tierra, -1 inválido.
    """
    n = len(coords)
    if n < 5:
        return coords, True
    inv = ~transform
    h, w = state.shape
    idx = np.unique(np.linspace(1, n - 2, min(max_samples, n - 2)).astype(int))
    score = 0

    def sample(p):
        col = inv.a * p[0] + inv.b * p[1] + inv.c
        row = inv.d * p[0] + inv.e * p[1] + inv.f
        r, c = int(np.floor(row)), int(np.floor(col))
        return int(state[r, c]) if (0 <= r < h and 0 <= c < w) else -1

    for i in idx:
        t = coords[min(i + 2, n - 1)] - coords[max(i - 2, 0)]
        norm = np.hypot(*t)
        if norm == 0:
            continue
        nrm = np.array([-t[1], t[0]]) / norm
        left, right = sample(coords[i] + offset_m * nrm), sample(coords[i] - offset_m * nrm)
        if left == 1 and right == 0:
            score += 1
        elif left == 0 and right == 1:
            score -= 1
    if score < 0:
        return np.ascontiguousarray(coords[::-1]), False
    return coords, score == 0


# ---------------------------------------------------------------------------
# Procesado de una escena
# ---------------------------------------------------------------------------

def scene_stem(path):
    stem = Path(path).stem
    return stem[:-5] if stem.endswith("_ndwi") else stem


def save_mask_tif(state, output_path, profile):
    """Máscara limpia: 0 tierra, 1 agua, 255 inválido."""
    p = profile.copy()
    p.update(driver="GTiff", dtype=rasterio.uint8, count=1, nodata=255)
    out = np.where(state < 0, 255, state).astype(np.uint8)
    with rasterio.open(output_path, "w", **p) as dst:
        dst.write(out, 1)


def extract_scene(ndwi_file, aoi_gdf, clean_dir, params):
    with rasterio.open(ndwi_file) as src:
        ndwi = src.read(1).astype(np.float32)
        transform, crs, profile, tags = src.transform, src.crs, src.profile, src.tags()

    aoi_proj = aoi_gdf.to_crs(crs)
    aoi_poly = aoi_proj.union_all() if hasattr(aoi_proj, "union_all") else aoi_proj.unary_union
    valid = np.isfinite(ndwi)
    aoi_mask = geometry_mask(list(aoi_proj.geometry), transform=transform, invert=True, out_shape=ndwi.shape)

    thr, method = utils.compute_ndwi_threshold(
        ndwi[aoi_mask & valid], params["threshold_mode"], params["threshold_fixed"]
    )
    water_raw, water_clean = clean_water_mask(
        ndwi, thr, valid, params["min_area_px"], params["closing_radius"]
    )
    adj = adjust_ndwi(ndwi, thr, water_raw, water_clean, config.FLIP_MARGIN)
    work = np.where(valid, adj, np.nan)
    state = np.where(valid, water_clean.astype(np.int8), -1).astype(np.int8)

    if params["save_clean"]:
        save_mask_tif(state, clean_dir / f"{scene_stem(ndwi_file)}_clean.tif", profile)

    pixel = abs(transform.a)
    segments, ambiguous = [], 0
    for c in find_contours(work, level=thr, mask=valid):
        if len(c) < 3:
            continue
        xy = contour_to_xy(c, transform)
        for part in utils.flatten_lines(LineString(xy).intersection(aoi_poly)):
            if part.length < params["min_length_m"]:
                continue
            coords, amb = orient_water_left(
                utils.dedupe_consecutive(np.asarray(part.coords)), state, transform, 1.5 * pixel
            )
            ambiguous += int(amb)
            segments.append(coords)
    segments.sort(key=lambda s: -LineString(s).length)

    n_aoi = max(int(aoi_mask.sum()), 1)
    info = {
        "ndwi_threshold": round(thr, 4),
        "threshold_method": method,
        "water_fraction_aoi": round(float((water_clean & aoi_mask).sum()) / n_aoi, 4),
        "pixels_changed_by_cleaning": int((water_raw != water_clean).sum()),
        "n_segments": len(segments),
        "total_length_m": round(float(sum(LineString(s).length for s in segments)), 1),
        "ambiguous_orientation": ambiguous,
        "cloud_pct_aoi": tags.get("CLOUD_PCT_AOI", ""),
    }
    return segments, crs, transform, info


def write_scene_outputs(stem, date, segments, crs, aoi_gdf, output_dir):
    """Escribe KMZ, CSV lat/lon, CSV x/y y GeoJSON de una escena. Devuelve el DataFrame x/y."""
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lines_ll = []
    for s in segments:
        lon, lat = tr.transform(s[:, 0], s[:, 1])
        lines_ll.append(list(zip(lon, lat)))

    utils.export_lines_to_kmz(lines_ll, output_dir / f"{stem}.kmz")

    with open(output_dir / f"{stem}.csv", "w") as f:
        f.write("lat,lon,date,segment\n")
        for k, line in enumerate(lines_ll):
            for lon, lat in line:
                f.write(f"{lat},{lon},{date},{k}\n")

    rows = []
    for k, s in enumerate(segments):
        rows.append(
            pd.DataFrame({"x": s[:, 0], "y": s[:, 1], "date": date, "segment": k,
                          "order": np.arange(len(s)), "scene": stem})
        )
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(output_dir / f"{stem}_xyz.csv", index=False)

    features = [
        {"type": "Feature", "properties": {"date": date, "segment": k},
         "geometry": {"type": "LineString", "coordinates": [list(p) for p in line]}}
        for k, line in enumerate(lines_ll)
    ]
    features.append({"type": "Feature", "properties": {"name": "AOI"},
                     "geometry": aoi_gdf.geometry.iloc[0].__geo_interface__})
    with open(output_dir / f"{stem}.geojson", "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    return df


def choose_scenes(ndwi_files):
    """Una escena por fecha: si hay varias (teselas solapadas) se queda la de menos nubes."""
    by_date = {}
    for f in ndwi_files:
        d = utils.extract_date(f.name)
        if d is None:
            raise ValueError(f"No se pudo extraer la fecha (YYYYMMDD) de: {f.name}")
        by_date.setdefault(d.isoformat(), []).append(f)

    chosen, skipped = {}, []
    for d, files in by_date.items():
        def cloud(path):
            with rasterio.open(path) as s:
                return float(s.tags().get("CLOUD_PCT_AOI", 100))
        best = min(files, key=cloud)
        chosen[d] = best
        skipped.extend(f.name for f in files if f != best)
    return chosen, skipped


def main(project, ndwi_dir=None, polygon_path=None, clean_dir=None, threshold_mode=None,
         threshold_fixed=None, min_area_px=None, min_length_m=None, closing_radius=None):
    project_dir = utils.project_path(project)
    output_dir = project_dir / "output" / "estimated_waterbodies_edges"
    output_dir.mkdir(parents=True, exist_ok=True)

    ndwi_dir = Path(ndwi_dir) if ndwi_dir else project_dir / "output" / "estimated_ndwi"
    polygon_path = Path(polygon_path) if polygon_path else project_dir / "input" / "polygon.geojson"
    clean_dir = Path(clean_dir) if clean_dir else project_dir / "output" / "estimated_waterbodies_images_clean"

    params = {
        "threshold_mode": threshold_mode or config.MODO_UMBRAL,
        "threshold_fixed": config.UMBRAL_NDWI_FIJO if threshold_fixed is None else threshold_fixed,
        "min_area_px": config.MIN_AREA_PX if min_area_px is None else min_area_px,
        "min_length_m": config.MIN_SEGMENT_LENGTH_M if min_length_m is None else min_length_m,
        "closing_radius": config.CLOSING_RADIUS if closing_radius is None else closing_radius,
        "save_clean": config.SAVE_CLEAN_MASKS,
    }
    if params["save_clean"]:
        clean_dir.mkdir(parents=True, exist_ok=True)

    if not ndwi_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de NDWI: {ndwi_dir} (ejecuta sesion5.py antes)")
    if not polygon_path.is_file():
        raise FileNotFoundError(f"No existe el polígono del AOI: {polygon_path}")
    ndwi_files = sorted(ndwi_dir.glob("*.tif"))
    if not ndwi_files:
        raise FileNotFoundError(f"No hay .tif de NDWI en {ndwi_dir}")

    aoi_gdf = gpd.read_file(polygon_path).set_crs(4326, allow_override=True)
    chosen, skipped = choose_scenes(ndwi_files)

    # Se borran resultados de ejecuciones anteriores para no mezclarlos
    for old in output_dir.glob("*_xyz.csv"):
        old.unlink()

    frames, infos, no_coast, crs_used = [], [], [], None
    for date, ndwi_file in tqdm(sorted(chosen.items()), desc="Extrayendo costas"):
        segments, crs, _, info = extract_scene(ndwi_file, aoi_gdf, clean_dir, params)
        if crs_used is not None and crs != crs_used:
            raise ValueError(f"CRS distinto en {ndwi_file.name}: {crs} != {crs_used}")
        crs_used = crs
        stem = scene_stem(ndwi_file)
        info.update({"date": date, "scene": stem})
        infos.append(info)
        if not segments:
            no_coast.append(ndwi_file.name)
            continue
        frames.append(write_scene_outputs(stem, date, segments, crs, aoi_gdf, output_dir))

    if not frames:
        raise RuntimeError("No se encontró costa en ninguna escena.")

    all_df = pd.concat(frames, ignore_index=True)
    all_path = output_dir / "all_shorelines.csv"
    all_df.to_csv(all_path, index=False)
    pd.DataFrame(infos).to_csv(output_dir / "extraction_params.csv", index=False)
    with open(output_dir / "shorelines_meta.json", "w") as f:
        json.dump({"crs": crs_used.to_string(), **{k: params[k] for k in
                   ("threshold_mode", "min_area_px", "min_length_m", "closing_radius")}}, f, indent=2)

    print(f"Escenas procesadas: {len(frames)} de {len(chosen)}")
    if skipped:
        print(f"Escenas duplicadas por fecha descartadas (más nubes): {', '.join(skipped)}")
    if no_coast:
        print(f"Sin costa en {len(no_coast)} escena(s): {', '.join(no_coast)}")
    print(f"Escrito {all_path}")
    return all_df, pd.DataFrame(infos)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrae la costa (contornos subpíxel) de los NDWI.")
    parser.add_argument("-p", "--project", default=config.NOMBRE_PROYECTO)
    parser.add_argument("--ndwi-dir", help="Carpeta con los NDWI .tif (por defecto, la del proyecto).")
    parser.add_argument("--polygon", help="GeoJSON del AOI (por defecto, el del proyecto).")
    parser.add_argument("--clean-dir", help="Dónde guardar las máscaras limpias.")
    parser.add_argument("--threshold-mode", choices=["otsu", "fixed"], default=None)
    parser.add_argument("--threshold", type=float, default=None, help="Umbral fijo (modo fixed o fallback).")
    parser.add_argument("--min-area-px", type=int, default=None)
    parser.add_argument("--min-length-m", type=float, default=None)
    parser.add_argument("--closing-radius", type=int, default=None, help="0 = desactivado.")
    a = parser.parse_args()
    main(a.project, a.ndwi_dir, a.polygon, a.clean_dir, a.threshold_mode, a.threshold,
         a.min_area_px, a.min_length_m, a.closing_radius)
