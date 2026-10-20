import argparse
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask
from scipy.ndimage import binary_erosion, convolve

import shoreline_utils as utils


# Cada proceso mantiene en RAM la máscara (uint8), la máscara del AOI, el
# resultado de la convolución y varios arrays booleanos: del orden de 1 GB
# para un tile de 10 m. Estimación; ajústala si mides otro pico.
GB_POR_PROCESO = 2
MAX_PROCESOS_DEFECTO = 4


def reproject_coords_to_latlon(coords, src_crs):
    # Una sola llamada vectorizada: antes se llamaba a pyproj una vez por punto
    # en un bucle de Python, que era mucho más lento con miles de puntos.
    if len(coords) == 0:
        return []
    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    arr = np.asarray(coords, dtype=float)
    lons, lats = transformer.transform(arr[:, 0], arr[:, 1])
    return list(zip(np.asarray(lons).tolist(), np.asarray(lats).tolist()))


def export_coords_to_csv_latlon(coords, output_csv_path, date):
    with open(output_csv_path, "w") as f:
        for lon, lat in coords:
            f.write(f"{lat},{lon},{date}\n")


def export_coords_to_csv_xy(coords, output_csv_path, date):
    with open(output_csv_path, "w") as f:
        for x, y in coords:
            f.write(f"{x},{y},{date}\n")


def extract_date_from_mask(mask_path):
    date = utils.extract_date(mask_path.name)
    if date is None:
        raise ValueError(
            f"Could not extract an YYYYMMDD date from mask filename: {mask_path.name}"
        )
    return date.isoformat()


def read_mask_band(src):
    if src.count != 1:
        raise ValueError(f"Expected a single-band mask GeoTIFF, got {src.count} bands: {src.name}")
    return src.read()[0]


def estimate_mask_edges(mask_file, aoi_proj, aoi_gdf, output_dir):
    with rasterio.open(mask_file) as src:
        mask_image = read_mask_band(src)
        aoi_mask = geometry_mask(
            [geom for geom in aoi_proj.geometry],
            transform=src.transform,
            invert=True,
            out_shape=mask_image.shape,
        )

        kernel = np.array(
            [
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ]
        )

        neighbor_sum = convolve(mask_image.astype(np.uint8), kernel, mode="constant", cval=0)
        transition_mask = np.logical_and(mask_image, neighbor_sum < 8)
        eroded_aoi_mask = binary_erosion(aoi_mask, structure=kernel)
        final_coast_mask = np.logical_and(transition_mask, eroded_aoi_mask)

        rows, cols = np.nonzero(final_coast_mask)
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        coords_xy = list(zip(xs, ys))

        if len(coords_xy) < 2:
            print(f"No coastline found in {mask_file}.")
            return False

        coords_latlon = reproject_coords_to_latlon(coords_xy, src.crs)

    date = extract_date_from_mask(mask_file)
    stem = mask_file.stem

    kmz_path = output_dir / f"{stem}.kmz"
    utils.export_coords_to_kmz(coords_latlon, kmz_path)

    csv_latlon_path = output_dir / f"{stem}.csv"
    export_coords_to_csv_latlon(coords_latlon, csv_latlon_path, date)

    csv_xy_path = output_dir / f"{stem}_xyz.csv"
    export_coords_to_csv_xy(coords_xy, csv_xy_path, date)

    geojson_path = output_dir / f"{stem}.geojson"
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            *[
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": coord},
                    "properties": {"date": date},
                }
                for coord in coords_latlon
            ],
            {
                "type": "Feature",
                "geometry": aoi_gdf.geometry[0].__geo_interface__,
                "properties": {"name": "AOI"},
            },
        ],
    }
    with open(geojson_path, "w") as f:
        json.dump(geojson_data, f, indent=2)

    return True


def merge_edge_csvs(output_dir):
    all_edges_csv_path = output_dir / "all_shorelines.csv"
    edge_files = [
        output_dir / filename
        for filename in os.listdir(output_dir)
        if filename.endswith("_xyz.csv")
    ]

    with open(all_edges_csv_path, "w") as out:
        out.write("x,y,date\n")
        for edge_file in edge_files:
            with open(edge_file) as f:
                out.writelines(f.readlines())

    return all_edges_csv_path


def _procesar_mascara(mask_file, aoi_proj, aoi_gdf, output_dir):
    """Función de nivel de módulo (necesario para enviarla a otro proceso)."""
    return estimate_mask_edges(mask_file, aoi_proj, aoi_gdf, output_dir)


def main(project, mask_dir=None, polygon_path=None, workers=None):
    project_dir = utils.project_path(project)
    input_dir = project_dir / "input"
    output_dir = project_dir / "output" / "estimated_waterbodies_edges"
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_dir = Path(mask_dir) if mask_dir else project_dir / "output" / "estimated_waterbodies_images"
    polygon_path = Path(polygon_path) if polygon_path else input_dir / "polygon.geojson"

    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")
    if not polygon_path.is_file():
        raise FileNotFoundError(f"AOI polygon does not exist: {polygon_path}")

    mask_files = sorted(mask_dir.glob("*.tif"))
    if not mask_files:
        raise FileNotFoundError(f"No .tif mask files found in {mask_dir}")

    aoi_gdf = gpd.read_file(polygon_path).set_crs(4326, allow_override=True)
    with rasterio.open(mask_files[0]) as src:
        raster_crs = src.crs
    aoi_proj = aoi_gdf.to_crs(raster_crs)

    # Las máscaras que ya tienen su _xyz.csv se saltan (como antes).
    processed = 0
    pending = {}
    for mask_file in mask_files:
        csv_xy_path = output_dir / f"{mask_file.stem}_xyz.csv"
        if csv_xy_path.exists() and csv_xy_path.stat().st_size > 0:
            processed += 1
        else:
            pending[mask_file.name] = (mask_file, aoi_proj, aoi_gdf, output_dir)

    if pending:
        workers = workers or utils.default_workers(GB_POR_PROCESO, MAX_PROCESOS_DEFECTO)
        workers = max(1, min(workers, utils.MAX_WORKERS_HARD, len(pending)))
        print(f"Extracting shoreline edges from {len(pending)} masks, {workers} at a time...")

        results, errors, failed = utils.run_parallel(
            _procesar_mascara, pending, workers, desc="Extracting shoreline edges"
        )

        processed += sum(1 for ok in results.values() if ok)
        for name, error in errors.items():
            print(f"ERROR in {name}: {error}")
        if failed:
            print(
                f"{len(failed)} mask(s) could not be processed (worker killed, probably out of RAM): "
                f"{', '.join(failed)}. Try --workers 1 or free memory."
            )

    all_edges_csv_path = merge_edge_csvs(output_dir)
    print(f"Processed {processed} mask files.")
    print(f"Wrote {all_edges_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract shoreline edge points from binary water/land mask GeoTIFFs."
    )
    parser.add_argument(
        "-p",
        "--project",
        required=True,
        help="Project name under coastline_estimator/projects.",
    )
    parser.add_argument(
        "--mask-dir",
        help="Directory containing binary mask GeoTIFFs. Defaults to the project output mask directory.",
    )
    parser.add_argument(
        "--polygon",
        help="AOI polygon GeoJSON. Defaults to the project input polygon.geojson.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help=f"Masks processed at once (default: automatic, max {MAX_PROCESOS_DEFECTO}; hard limit {utils.MAX_WORKERS_HARD}). Use 1 to run serially.",
    )
    args = parser.parse_args()
    main(args.project, args.mask_dir, args.polygon, args.workers)
