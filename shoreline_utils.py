"""
Funciones compartidas por todos los scripts.

Convención de signo (IMPORTANTE)
--------------------------------
Todas las líneas de costa se orientan de modo que el AGUA queda a la IZQUIERDA
del sentido de recorrido. La normal de cada transecto es (-ty, tx), es decir,
apunta a la izquierda = hacia el mar. Por tanto:

    distancia > 0  ->  la costa medida está MAR ADENTRO respecto a la referencia (acreción)
    distancia < 0  ->  la costa medida está TIERRA ADENTRO respecto a la referencia (erosión)
"""

import re
import zipfile
from datetime import datetime
from pathlib import Path
from xml.dom.minidom import Document

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.interpolate import splev, splprep
from shapely.geometry import LineString, Point
from skimage.filters import threshold_otsu

import config

DATE_RE = re.compile(r"(?<!\d)(\d{8})(?!\d)")


def project_path(project):
    return config.project_path(project)


# ---------------------------------------------------------------------------
# Fechas y ficheros
# ---------------------------------------------------------------------------

def extract_date(name: str):
    match = DATE_RE.search(name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def closest_ground_truth_file(ground_truth_dir, target_date):
    """Devuelve (fecha, ruta) del CSV de referencia con fecha más cercana a target_date."""
    base = Path(ground_truth_dir).expanduser().resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"{base} no es un directorio")

    dated = []
    for csv_file in base.glob("*.csv"):
        d = extract_date(csv_file.name)
        if d is not None:
            dated.append((d, csv_file))
    if not dated:
        raise FileNotFoundError(f"No hay CSV con fecha (YYYYMMDD) en el nombre en {base}")
    return min(dated, key=lambda row: abs(row[0] - target_date))


# ---------------------------------------------------------------------------
# Lectura / escritura de coordenadas
# ---------------------------------------------------------------------------

def load_xy_csv(csv_path):
    """
    Lee un CSV de coordenadas x,y (separador tabulador, espacio o coma; líneas
    que empiecen por % ignoradas; cabeceras de texto ignoradas). Devuelve (N,2).
    Es la ÚNICA función de lectura de referencias externas, para que todos los
    scripts interpreten el archivo igual.
    """
    df = pd.read_csv(
        csv_path,
        sep=r"\s+|,|\t",
        engine="python",
        comment="%",
        names=["x", "y"],
        usecols=[0, 1],
        header=None,
    )
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"])
    if df.empty:
        raise ValueError(f"No se encontraron coordenadas numéricas en {csv_path}")
    return df[["x", "y"]].to_numpy(float)


def transform_xy(xy, source_crs, target_crs):
    xy = np.asarray(xy, float)
    if str(source_crs) == str(target_crs):
        return xy
    tr = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    x, y = tr.transform(xy[:, 0], xy[:, 1])
    return np.column_stack([x, y])


def load_coords_csv(csv_path, source_crs=32631, target_crs=4326):
    """Lee un CSV x,y y lo devuelve como lista de (lon, lat)."""
    ll = transform_xy(load_xy_csv(csv_path), source_crs, target_crs)
    return list(zip(ll[:, 0], ll[:, 1]))


def _kml_document():
    doc = Document()
    kml = doc.createElement("kml")
    kml.setAttribute("xmlns", "http://www.opengis.net/kml/2.2")
    doc.appendChild(kml)
    document = doc.createElement("Document")
    kml.appendChild(document)
    return doc, document


def _write_kmz(doc, output_kmz_path):
    kml_str = doc.toprettyxml(indent="  ", encoding="UTF-8")
    with zipfile.ZipFile(output_kmz_path, "w", zipfile.ZIP_DEFLATED) as kmz:
        kmz.writestr("doc.kml", kml_str)


def export_coords_to_kmz(coords, output_kmz_path):
    """Puntos (lon, lat) -> KMZ."""
    doc, document = _kml_document()
    for lon, lat in coords:
        placemark = doc.createElement("Placemark")
        point = doc.createElement("Point")
        coord = doc.createElement("coordinates")
        coord.appendChild(doc.createTextNode(f"{lon},{lat},0"))
        point.appendChild(coord)
        placemark.appendChild(point)
        document.appendChild(placemark)
    _write_kmz(doc, output_kmz_path)


def export_lines_to_kmz(lines_lonlat, output_kmz_path):
    """Lista de polilíneas [(lon, lat), ...] -> KMZ con un Placemark por segmento."""
    doc, document = _kml_document()
    for i, line in enumerate(lines_lonlat):
        placemark = doc.createElement("Placemark")
        name = doc.createElement("name")
        name.appendChild(doc.createTextNode(f"segmento_{i}"))
        placemark.appendChild(name)
        ls = doc.createElement("LineString")
        coord = doc.createElement("coordinates")
        coord.appendChild(doc.createTextNode(" ".join(f"{lon},{lat},0" for lon, lat in line)))
        ls.appendChild(coord)
        placemark.appendChild(ls)
        document.appendChild(placemark)
    _write_kmz(doc, output_kmz_path)


def read_shorelines(csv_path):
    """
    Lee all_shorelines.csv y devuelve {fecha: [array(N,2) por segmento]}, con los
    puntos de cada segmento ya en su orden a lo largo de la costa.
    """
    df = pd.read_csv(csv_path)
    out = {}
    for date, g in df.groupby("date"):
        segs = []
        for _, s in g.groupby("segment"):
            s = s.sort_values("order")
            segs.append(s[["x", "y"]].to_numpy(float))
        out[str(date)] = segs
    return out


def load_project_aoi(project_dir, crs):
    """Polígono del AOI como geometría shapely en el CRS indicado."""
    import geopandas as gpd

    path = Path(project_dir) / "input" / "polygon.geojson"
    if not path.is_file():
        raise FileNotFoundError(f"No existe el polígono del AOI: {path}")
    gdf = gpd.read_file(path).set_crs(4326, allow_override=True).to_crs(crs)
    return gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union


# ---------------------------------------------------------------------------
# Umbral de NDWI
# ---------------------------------------------------------------------------

def compute_ndwi_threshold(values, mode=None, fixed=None, lo=None, hi=None):
    """
    Devuelve (umbral, método). 'otsu' calcula un umbral por imagen sobre los
    píxeles del AOI; si el histograma no es bimodal (casi todo agua o casi todo
    tierra) o hay pocos datos, cae al umbral fijo.
    """
    mode = config.MODO_UMBRAL if mode is None else mode
    fixed = config.UMBRAL_NDWI_FIJO if fixed is None else fixed
    lo = config.OTSU_MIN if lo is None else lo
    hi = config.OTSU_MAX if hi is None else hi

    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if mode == "fixed" or values.size < 100:
        return float(fixed), "fixed"
    try:
        thr = float(threshold_otsu(values))
    except ValueError:
        return float(fixed), "fallback_fixed"
    frac_water = float(np.mean(values > thr))
    if not (0.03 <= frac_water <= 0.97):
        return float(fixed), "fallback_fixed"
    return float(np.clip(thr, lo, hi)), "otsu"


# ---------------------------------------------------------------------------
# Geometría
# ---------------------------------------------------------------------------

def flatten_lines(geom):
    """Extrae todas las LineString de cualquier geometría (Multi*, Collection)."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if hasattr(geom, "geoms"):
        out = []
        for g in geom.geoms:
            out.extend(flatten_lines(g))
        return out
    return []


def dedupe_consecutive(coords, tol=1e-6):
    c = np.asarray(coords, float)[:, :2]
    if len(c) < 2:
        return c
    keep = np.r_[True, np.hypot(*np.diff(c, axis=0).T) > tol]
    return c[keep]


def order_segments(segments):
    """Ordena segmentos a lo largo del eje principal de la costa (ids estables)."""
    if len(segments) < 2:
        return list(segments)
    cents = np.array([np.mean(s, axis=0) for s in segments])
    c = cents - cents.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    axis = vt[0]
    if axis[1] < 0 or (axis[1] == 0 and axis[0] < 0):
        axis = -axis
    return [segments[i] for i in np.argsort(c @ axis)]


def orient_like(coords, reference_lines, step=50.0):
    """
    Invierte 'coords' si su sentido de recorrido es opuesto al de las líneas de
    referencia cercanas. Se usa para que una referencia externa (cuyo sentido
    se desconoce) quede con el agua a la izquierda, igual que las costas extraídas.
    """
    coords = np.asarray(coords, float)
    line = LineString(coords)
    if not reference_lines or line.length == 0:
        return coords
    score = 0.0
    n = max(int(line.length / step), 3)
    for s in np.linspace(0, line.length, n + 2)[1:-1]:
        p = line.interpolate(s)
        a = line.interpolate(max(s - 5, 0))
        b = line.interpolate(min(s + 5, line.length))
        t = np.array([b.x - a.x, b.y - a.y])
        ref = min(reference_lines, key=lambda r: r.distance(p))
        if ref.distance(p) > 100:
            continue
        sr = ref.project(p)
        ra = ref.interpolate(max(sr - 5, 0))
        rb = ref.interpolate(min(sr + 5, ref.length))
        tr = np.array([rb.x - ra.x, rb.y - ra.y])
        score += float(np.dot(t, tr))
    return coords if score >= 0 else np.ascontiguousarray(coords[::-1])


def resample_segment(coords, spacing, sigma):
    """
    Remuestrea un segmento ORDENADO a puntos ~equiespaciados con una spline de
    suavizado. Devuelve array (n, 4): x, y, dx, dy (tangente), o None.
    sigma = desviación típica (m) que se tolera entre spline y puntos.
    """
    pts = dedupe_consecutive(coords)
    if len(pts) < 2:
        return None
    length = float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))
    n = max(int(round(length / spacing)) + 1, 2)

    if len(pts) >= 4:
        try:
            tck, _ = splprep([pts[:, 0], pts[:, 1]], s=len(pts) * sigma**2, k=3)
            unew = np.linspace(0, 1, n)
            xs, ys = splev(unew, tck)
            dx, dy = splev(unew, tck, der=1)
            return np.column_stack([xs, ys, dx, dy])
        except (ValueError, TypeError):
            pass

    line = LineString(pts)
    q = [line.interpolate(v) for v in np.linspace(0, length, n)]
    xs = np.array([p.x for p in q])
    ys = np.array([p.y for p in q])
    return np.column_stack([xs, ys, np.gradient(xs), np.gradient(ys)])


def build_transects(segments, spacing, sigma, half_length):
    """
    Genera transectos perpendiculares a cada segmento de referencia.
    Cada transecto es un dict con: id, segment, x, y, nx, ny (normal hacia la
    izquierda = mar) y line (LineString de longitud 2*half_length).
    """
    transects = []
    tid = 0
    for seg_id, seg in enumerate(segments):
        r = resample_segment(seg, spacing, sigma)
        if r is None:
            continue
        for x, y, dx, dy in r:
            norm = float(np.hypot(dx, dy))
            if norm < 1e-12:
                continue
            nx, ny = -dy / norm, dx / norm
            transects.append(
                {
                    "id": tid,
                    "segment": seg_id,
                    "x": float(x),
                    "y": float(y),
                    "nx": nx,
                    "ny": ny,
                    "line": LineString(
                        [(x - nx * half_length, y - ny * half_length),
                         (x + nx * half_length, y + ny * half_length)]
                    ),
                }
            )
            tid += 1
    return transects


def mean_transect_spacing(transects):
    d = [
        np.hypot(b["x"] - a["x"], b["y"] - a["y"])
        for a, b in zip(transects[:-1], transects[1:])
        if a["segment"] == b["segment"]
    ]
    return float(np.mean(d)) if d else float("nan")


def _iter_geoms(geom):
    if hasattr(geom, "geoms"):
        for g in geom.geoms:
            yield from _iter_geoms(g)
    else:
        yield geom


def _signed_candidates(inter, x, y, nx, ny):
    """Distancias con signo (a lo largo de la normal) de todos los cruces."""
    if inter.is_empty:
        return []
    origin = Point(x, y)
    out = []
    for g in _iter_geoms(inter):
        if isinstance(g, Point):
            px, py = g.x, g.y
        elif isinstance(g, LineString):  # solape colineal: punto más cercano al origen
            p = g.interpolate(g.project(origin))
            px, py = p.x, p.y
        else:
            continue
        out.append((px - x) * nx + (py - y) * ny)
    return out


def measure_transects(transects, geom):
    """
    Mide, para cada transecto, la distancia con signo al cruce más cercano con
    'geom' (LineString o MultiLineString). Devuelve lista de (distancia, n_cruces);
    distancia = NaN si no hay cruce.
    """
    out = []
    for t in transects:
        cand = _signed_candidates(t["line"].intersection(geom), t["x"], t["y"], t["nx"], t["ny"])
        out.append((min(cand, key=abs) if cand else float("nan"), len(cand)))
    return out
