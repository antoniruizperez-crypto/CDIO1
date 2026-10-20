import multiprocessing as mp
import os
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
from pathlib import Path
from xml.dom.minidom import Document

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial.distance import cdist
from shapely.geometry import LineString


DATE_RE = re.compile(r"(?<!\d)(\d{8})(?!\d)")

# Tope duro de procesos simultáneos en todo el proyecto, aunque se pida más.
MAX_WORKERS_HARD = 4


def _ram_total_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return 8  # valor conservador si no se puede consultar


def default_workers(gb_per_process, cap=2):
    """
    Procesos paralelos por defecto: el menor entre `cap`, (núcleos - 1) y
    (RAM total / gb_per_process). Siempre al menos 1 y nunca más de
    MAX_WORKERS_HARD.
    """
    cores = os.cpu_count() or 1
    ram_based = int(_ram_total_gb() // gb_per_process)
    return max(1, min(cap, cores - 1, ram_based, MAX_WORKERS_HARD))


def _pool_pass(func, jobs, names, workers, results, errors, desc):
    """
    Una pasada de procesamiento en un pool. Devuelve los nombres cuyo worker
    murió (BrokenProcessPool, p.ej. por falta de RAM). Los errores normales
    lanzados dentro de `func` se guardan en `errors`.
    """
    ctx = mp.get_context("spawn")
    died = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, max_tasks_per_child=1) as pool:
        futures = {pool.submit(func, *jobs[name]): name for name in names}
        iterator = as_completed(futures)
        if desc:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=len(futures), desc=desc)
        for future in iterator:
            name = futures[future]
            try:
                results[name] = future.result()
            except BrokenProcessPool:
                died.append(name)
            except Exception as e:
                errors[name] = e
    return died


def run_parallel(func, jobs, workers, desc=None):
    """
    Ejecuta func(*args) para cada trabajo, con como máximo `workers` procesos
    a la vez. `jobs` es un dict {nombre: tupla_de_argumentos}; `func` debe
    ser una función de nivel de módulo (para poder enviarse a otro proceso).

    - Cada proceso hace UN trabajo y muere: devuelve toda su memoria al sistema.
    - Si un worker muere (p.ej. el sistema lo mata por falta de RAM), el pool
      entero se rompe y fallan también los trabajos que ni habían empezado.
      Esos se reintentan de uno en uno, cada uno en un pool propio.

    Devuelve (results, errors, failed):
      results: {nombre: valor devuelto}
      errors:  {nombre: excepción lanzada dentro de func}
      failed:  [nombres cuyo worker murió incluso en solitario]
    """
    workers = max(1, min(workers, MAX_WORKERS_HARD, len(jobs)))
    results, errors, failed = {}, {}, []

    if workers == 1:
        # Sin pool: más fácil de depurar. (Si aquí se acaba la RAM, el
        # programa entero cae; no hay forma de aislarlo sin un proceso aparte.)
        for name, args in jobs.items():
            try:
                results[name] = func(*args)
            except Exception as e:
                errors[name] = e
        return results, errors, failed

    died = _pool_pass(func, jobs, list(jobs), workers, results, errors, desc)
    if died:
        print(f"\n{len(died)} trabajo(s) afectados por una caída de worker; reintentando de uno en uno...")
        for name in died:
            if _pool_pass(func, jobs, [name], 1, results, errors, None):
                failed.append(name)
    return results, errors, failed


def extract_date(name: str):
    match = DATE_RE.search(name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def export_coords_to_kmz(coords, output_kmz_path):
    doc = Document()
    kml = doc.createElement("kml")
    kml.setAttribute("xmlns", "http://www.opengis.net/kml/2.2")
    doc.appendChild(kml)

    document = doc.createElement("Document")
    kml.appendChild(document)

    for lon, lat in coords:
        placemark = doc.createElement("Placemark")
        point = doc.createElement("Point")
        coord = doc.createElement("coordinates")
        coord.appendChild(doc.createTextNode(f"{lon},{lat},0"))
        point.appendChild(coord)
        placemark.appendChild(point)
        document.appendChild(placemark)

    kml_str = doc.toprettyxml(indent="  ", encoding="UTF-8")

    with zipfile.ZipFile(output_kmz_path, "w", zipfile.ZIP_DEFLATED) as kmz:
        kmz.writestr("doc.kml", kml_str)


def load_coords_csv(csv_path, source_crs=32631, target_crs=4326):
    df = pd.read_csv(
        csv_path,
        sep=r"\s+|,",
        engine="python",
        comment="%",
        names=["x", "y"],
        usecols=[0, 1],
        header=None,
    )
    df = df.dropna(subset=["x", "y"])

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    lons, lats = transformer.transform(df["x"].to_numpy(), df["y"].to_numpy())
    return list(zip(lons, lats))


def order_points(n, dist_matrix, starting_point):
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
        sum_distance += dist_matrix[current, next_idx]
        order.append(next_idx)
        visited[next_idx] = True
        current = next_idx

    return order, sum_distance


# Matriz de distancias compartida con los procesos hijos. Se asigna ANTES de
# crear el pool con "fork": los hijos la heredan sin copiarla ni enviarla por
# pickle (antes se enviaba una copia n x n a cada trozo, y con miles de puntos
# eso agotaba la RAM).
_DIST_MATRIX = None


def _compute_for_range(start_range):
    results = []
    n = _DIST_MATRIX.shape[0]
    for starting_point in start_range:
        _, sum_distance = order_points(n, _DIST_MATRIX, starting_point)
        results.append((starting_point, sum_distance))
    return results


def build_ordered_linestring(coords, ordered=False):
    global _DIST_MATRIX

    if ordered:
        return LineString(coords)

    coords = np.asarray(coords)
    n = len(coords)
    if n < 2:
        raise ValueError("At least two points are required to build a shoreline.")

    print(f"Number of points to order: {n}")

    # La matriz n x n de float64 es lo que más RAM consume. Mejor un error
    # claro ahora que un cierre brusco del sistema a mitad de cálculo.
    matrix_gb = n * n * 8 / 1e9
    if matrix_gb > 0.5 * _ram_total_gb():
        raise MemoryError(
            f"{n} puntos necesitan una matriz de distancias de {matrix_gb:.1f} GB, "
            f"demasiado para esta máquina ({_ram_total_gb():.0f} GB de RAM). "
            "Reduce los puntos de la costa (p.ej. submuestrear la máscara) o acota el polígono AOI."
        )

    dist_matrix = cdist(coords, coords, metric="euclidean")
    num_procs = max(1, min(MAX_WORKERS_HARD, mp.cpu_count() - 1))

    try:
        _DIST_MATRIX = dist_matrix
        if num_procs > 1 and "fork" in mp.get_all_start_methods():
            chunk_size = (n + num_procs - 1) // num_procs
            ranges = [range(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]
            with mp.get_context("fork").Pool(processes=num_procs) as pool:
                results = pool.map(_compute_for_range, ranges)
        else:
            results = [_compute_for_range(range(n))]
    finally:
        _DIST_MATRIX = None

    sum_distances = dict(sum(results, []))
    best_starting_point = min(sum_distances, key=sum_distances.get)
    ordered_indices, _ = order_points(n, dist_matrix, best_starting_point)
    return LineString(coords[ordered_indices])


def project_path(project):
    return Path(__file__).resolve().parent / "projects" / project
