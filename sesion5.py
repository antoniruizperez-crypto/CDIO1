"""
Funciones auxiliares del proyecto de costa y de calibración.

Dependencias:
    pip install numpy pandas pyproj scipy shapely rasterio

Este archivo se importa desde otros scripts.
No es necesario ejecutarlo directamente.

Las funciones de calibración no identifican automáticamente
el centro del objetivo ni sustituyen la inspección de las bandas.
"""

import hashlib
import multiprocessing as mp
import re
import zipfile
from datetime import datetime
from pathlib import Path
from xml.dom.minidom import Document

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.spatial.distance import cdist
from shapely.geometry import LineString
from shapely.ops import linemerge


DATE_RE = re.compile(r"(?<!\d)(\d{8})(?!\d)")

# Carpeta del proyecto (donde viven los scripts). Todas las rutas se
# calculan desde aquí, no desde el directorio de trabajo, para que los
# scripts funcionen igual lanzados desde el panel, la terminal o un IDE.
BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Fechas y rutas
# ---------------------------------------------------------------------------

def extract_date(name: str):
    match = DATE_RE.search(str(name))

    if not match:
        return None

    try:
        return datetime.strptime(
            match.group(1), "%Y%m%d"
        ).date()
    except ValueError:
        return None


def project_path(project):
    return BASE_DIR / "projects" / project


def images_path(*parts):
    """
    Carpeta de imágenes descargadas y de productos intermedios
    (zips, resultados visuales, calibración): BASE_DIR/Imagenes/...
    Se mantiene fuera de projects/ porque los zips pesan mucho.
    """
    return BASE_DIR / "Imagenes" / Path(*parts) if parts else BASE_DIR / "Imagenes"


# ---------------------------------------------------------------------------
# Exportación y lectura de coordenadas
# ---------------------------------------------------------------------------

def export_coords_to_kmz(coords, output_kmz_path):
    output_kmz_path = Path(output_kmz_path)
    output_kmz_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    doc = Document()

    kml = doc.createElement("kml")
    kml.setAttribute(
        "xmlns",
        "http://www.opengis.net/kml/2.2",
    )
    doc.appendChild(kml)

    document = doc.createElement("Document")
    kml.appendChild(document)

    for lon, lat in coords:
        placemark = doc.createElement("Placemark")
        point = doc.createElement("Point")
        coord = doc.createElement("coordinates")

        coord.appendChild(
            doc.createTextNode(f"{lon},{lat},0")
        )

        point.appendChild(coord)
        placemark.appendChild(point)
        document.appendChild(placemark)

    kml_str = doc.toprettyxml(
        indent="  ",
        encoding="UTF-8",
    )

    with zipfile.ZipFile(
        output_kmz_path,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as kmz:
        kmz.writestr("doc.kml", kml_str)


def load_coords_csv(
    csv_path,
    source_crs=32631,
    target_crs=4326,
):
    """
    Lee las dos primeras columnas como coordenadas.

    Se mantiene EPSG:32631 por compatibilidad con el proyecto
    original. Para otras zonas, indicar source_crs explícitamente.
    """
    df = pd.read_csv(
        csv_path,
        sep=r"\s+|,",
        engine="python",
        comment="%",
        names=["x", "y"],
        usecols=[0, 1],
        header=None,
    )

    df["x"] = pd.to_numeric(
        df["x"], errors="coerce"
    )
    df["y"] = pd.to_numeric(
        df["y"], errors="coerce"
    )

    df = df.dropna(subset=["x", "y"])

    valores = df[["x", "y"]].to_numpy(
        dtype=float
    )
    valores = valores[
        np.isfinite(valores).all(axis=1)
    ]

    if len(valores) == 0:
        raise ValueError(
            "El CSV no contiene coordenadas numéricas válidas."
        )

    transformer = Transformer.from_crs(
        source_crs,
        target_crs,
        always_xy=True,
    )

    xs, ys = transformer.transform(
        valores[:, 0],
        valores[:, 1],
        errcheck=True,
    )

    return list(zip(xs, ys))


# ---------------------------------------------------------------------------
# Ordenación de puntos y línea de costa
# ---------------------------------------------------------------------------

def order_points(n, dist_matrix, starting_point):
    if n < 1:
        raise ValueError(
            "Debe existir al menos un punto."
        )

    if not 0 <= starting_point < n:
        raise ValueError(
            "Índice inicial fuera de rango."
        )

    order = []
    visited = np.zeros(n, dtype=bool)
    current = starting_point

    order.append(current)
    visited[current] = True

    sum_distance = 0.0

    for _ in range(n - 1):
        row = dist_matrix[current].copy()
        row[visited] = np.inf

        next_idx = int(np.argmin(row))
        sum_distance += float(
            dist_matrix[current, next_idx]
        )

        order.append(next_idx)
        visited[next_idx] = True
        current = next_idx

    return order, sum_distance


def _compute_for_range(
    coords,
    dist_matrix,
    start_range,
):
    results = []
    n = len(coords)

    for starting_point in start_range:
        _, sum_distance = order_points(
            n,
            dist_matrix,
            starting_point,
        )

        results.append(
            (starting_point, sum_distance)
        )

    return results


def build_ordered_linestring(coords, ordered=False):
    coords = np.asarray(
        coords, dtype=float
    )

    if (
        coords.ndim != 2
        or coords.shape[1] != 2
    ):
        raise ValueError(
            "Las coordenadas deben tener forma (N, 2)."
        )

    if len(coords) < 2:
        raise ValueError(
            "Se necesitan al menos dos puntos "
            "para construir una línea de costa."
        )

    if not np.isfinite(coords).all():
        raise ValueError(
            "Las coordenadas contienen valores no válidos."
        )

    if ordered:
        return LineString(coords)

    n = len(coords)

    print(f"Number of points to order: {n}")

    dist_matrix = cdist(
        coords,
        coords,
        metric="euclidean",
    )

    num_procs = min(
        n,
        max(1, mp.cpu_count() - 1),
    )

    chunk_size = (
        n + num_procs - 1
    ) // num_procs

    ranges = [
        range(i, min(i + chunk_size, n))
        for i in range(0, n, chunk_size)
    ]

    if num_procs == 1:
        results = [
            _compute_for_range(
                coords,
                dist_matrix,
                ranges[0],
            )
        ]
    else:
        with mp.Pool(
            processes=num_procs
        ) as pool:
            results = pool.starmap(
                _compute_for_range,
                [
                    (
                        coords,
                        dist_matrix,
                        start_range,
                    )
                    for start_range in ranges
                ],
            )

    sum_distances = {
        starting_point: distance
        for group in results
        for starting_point, distance in group
    }

    best_starting_point = min(
        sum_distances,
        key=sum_distances.get,
    )

    ordered_indices, _ = order_points(
        n,
        dist_matrix,
        best_starting_point,
    )

    return LineString(
        coords[ordered_indices]
    )


# ---------------------------------------------------------------------------
# Transformación de coordenadas para calibración
# ---------------------------------------------------------------------------

def transform_coordinates(
    x,
    y,
    source_crs,
    target_crs,
):
    """
    Transforma una coordenada.

    En CRS geográficos:
        x = longitud
        y = latitud
    """
    if not (
        np.isfinite(x)
        and np.isfinite(y)
    ):
        raise ValueError(
            "Las coordenadas deben ser finitas."
        )

    transformer = Transformer.from_crs(
        source_crs,
        target_crs,
        always_xy=True,
    )

    x_out, y_out = transformer.transform(
        x,
        y,
        errcheck=True,
    )

    if not (
        np.isfinite(x_out)
        and np.isfinite(y_out)
    ):
        raise ValueError(
            "La transformación produjo valores no válidos."
        )

    return float(x_out), float(y_out)


def validate_metric_crs(crs):
    """
    Comprueba que un CRS sea proyectado y utilice metros.
    """
    parsed = CRS.from_user_input(crs)

    if not parsed.is_projected:
        raise ValueError(
            "Para errores en metros se necesita un CRS proyectado."
        )

    axes = parsed.axis_info[:2]

    if len(axes) != 2:
        raise ValueError(
            "No se pudieron comprobar las unidades del CRS."
        )

    for axis in axes:
        factor = axis.unit_conversion_factor

        if (
            factor is None
            or not np.isclose(
                factor,
                1.0,
                rtol=0,
                atol=1e-12,
            )
        ):
            raise ValueError(
                "El CRS proyectado debe utilizar metros."
            )

    return parsed


# ---------------------------------------------------------------------------
# Lectura de GeoTIFF y relación píxel-coordenadas
# ---------------------------------------------------------------------------

def read_geotiff(raster_path, band=1):
    """
    Devuelve:
        datos, mascara_valida, perfil

    Respeta tanto nodata como la máscara del raster.
    """
    import rasterio

    with rasterio.open(raster_path) as src:
        if not 1 <= band <= src.count:
            raise ValueError(
                "Número de banda fuera de rango."
            )

        array = src.read(
            band,
            masked=True,
        )

        datos = np.asarray(
            array.data
        ).copy()

        validos = (
            ~np.ma.getmaskarray(array)
            & np.isfinite(datos)
        )

        perfil = src.profile.copy()

    return datos, validos, perfil


def pixel_to_coordinates(
    raster_path,
    row,
    col,
    target_crs=None,
):
    """
    Devuelve el centro de un píxel.

    No interpreta ese centro como el centro observado
    de un objetivo de calibración.
    """
    import rasterio

    if (
        int(row) != row
        or int(col) != col
    ):
        raise ValueError(
            "Fila y columna deben ser índices enteros."
        )

    row = int(row)
    col = int(col)

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(
                "El raster no tiene CRS."
            )

        if not (
            0 <= row < src.height
            and 0 <= col < src.width
        ):
            raise ValueError(
                "Píxel fuera del raster."
            )

        x, y = src.xy(
            row,
            col,
            offset="center",
        )

        source_crs = src.crs

    if target_crs is not None:
        return transform_coordinates(
            x,
            y,
            source_crs,
            target_crs,
        )

    return float(x), float(y)


def coordinates_to_pixel(
    raster_path,
    x,
    y,
    source_crs=None,
):
    """
    Devuelve la fila y columna que contienen una coordenada.

    Si source_crs es None, x e y se interpretan
    directamente en el CRS del raster.
    """
    import rasterio

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(
                "El raster no tiene CRS."
            )

        if source_crs is not None:
            x, y = transform_coordinates(
                x,
                y,
                source_crs,
                src.crs,
            )

        if not (
            np.isfinite(x)
            and np.isfinite(y)
        ):
            raise ValueError(
                "Las coordenadas deben ser finitas."
            )

        row, col = src.index(x, y)

        if not (
            0 <= row < src.height
            and 0 <= col < src.width
        ):
            raise ValueError(
                "La coordenada queda fuera del raster."
            )

    return int(row), int(col)


# ---------------------------------------------------------------------------
# Errores geométricos y RMSE
# ---------------------------------------------------------------------------

def geometry_error(
    observed_x,
    observed_y,
    reference_x,
    reference_y,
    crs,
):
    """
    Error de una medición en un CRS proyectado en metros.

    Todas las coordenadas deben estar en el mismo CRS.

    Convención:
        dx = observado - referencia
        dy = observado - referencia
    """
    validate_metric_crs(crs)

    valores = np.asarray(
        [
            observed_x,
            observed_y,
            reference_x,
            reference_y,
        ],
        dtype=float,
    )

    if not np.isfinite(valores).all():
        raise ValueError(
            "Las coordenadas contienen valores no válidos."
        )

    dx = float(
        observed_x - reference_x
    )
    dy = float(
        observed_y - reference_y
    )

    return {
        "dx_m": dx,
        "dy_m": dy,
        "error_m": float(np.hypot(dx, dy)),
    }


def geometry_rmse(dx_m, dy_m):
    """
    Resume errores ya calculados en metros.

    RMSE global 2D:
        sqrt(mean(dx² + dy²))

    No mezcla ni transforma sistemas de referencia.
    """
    dx = np.asarray(
        dx_m, dtype=float
    )
    dy = np.asarray(
        dy_m, dtype=float
    )

    if (
        dx.ndim != 1
        or dy.ndim != 1
        or dx.shape != dy.shape
    ):
        raise ValueError(
            "dx_m y dy_m deben ser vectores de igual longitud."
        )

    if dx.size == 0:
        raise ValueError(
            "No hay mediciones para calcular el RMSE."
        )

    if not (
        np.isfinite(dx).all()
        and np.isfinite(dy).all()
    ):
        raise ValueError(
            "Los errores contienen valores no válidos."
        )

    return {
        "n": int(dx.size),
        "bias_x_m": float(np.mean(dx)),
        "bias_y_m": float(np.mean(dy)),
        "rmse_x_m": float(
            np.sqrt(np.mean(dx ** 2))
        ),
        "rmse_y_m": float(
            np.sqrt(np.mean(dy ** 2))
        ),
        "rmse_global_m": float(
            np.sqrt(
                np.mean(dx ** 2 + dy ** 2)
            )
        ),
    }
