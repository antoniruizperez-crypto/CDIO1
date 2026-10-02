import multiprocessing as mp
import re
import zipfile
from datetime import datetime
from pathlib import Path
from xml.dom.minidom import Document

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial.distance import cdist
from shapely.geometry import LineString


DATE_RE = re.compile(r"(?<!\d)(\d{8})(?!\d)")


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


def _compute_for_range(coords, dist_matrix, start_range):
    results = []
    n = len(coords)
    for starting_point in start_range:
        _, sum_distance = order_points(n, dist_matrix, starting_point)
        results.append((starting_point, sum_distance))
    return results


def build_ordered_linestring(coords, ordered=False):
    if ordered:
        return LineString(coords)

    coords = np.asarray(coords)
    n = len(coords)
    if n < 2:
        raise ValueError("At least two points are required to build a shoreline.")

    print(f"Number of points to order: {n}")
    dist_matrix = cdist(coords, coords, metric="euclidean")

    num_procs = max(1, mp.cpu_count() - 1)
    chunk_size = (n + num_procs - 1) // num_procs
    ranges = [range(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]

    with mp.Pool(processes=num_procs) as pool:
        results = pool.starmap(
            _compute_for_range,
            [(coords, dist_matrix.copy(), start_range) for start_range in ranges],
        )

    sum_distances = dict(sum(results, []))
    best_starting_point = min(sum_distances, key=sum_distances.get)
    ordered_indices, _ = order_points(n, dist_matrix, best_starting_point)
    return LineString(coords[ordered_indices])


def project_path(project):
    return Path(__file__).resolve().parent / "projects" / project