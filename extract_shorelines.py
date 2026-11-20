"""
Extracción de puntos de costa desde máscaras binarias GeoTIFF.

Dependencias:
    pip install geopandas numpy pandas rasterio pyproj scipy \
        scikit-image tqdm shapely

Utiliza shoreline_utils.py del proyecto.

Entrada:
    GeoTIFF de una banda:
        0 = tierra
        1 = agua

    Se respeta nodata y la máscara de validez del raster.

Salida:
    CSV combinado:
        x,y,date,case,source_file,crs

    No incluye point_id porque los puntos del borde no están
    ordenados a lo largo de la costa.

Uso habitual:
    python extract_shorelines.py -p costa_principal --case original

Otro conjunto de máscaras:
    python extract_shorelines.py \
        -p costa_principal \
        --mask-dir ruta/mascaras_distorsionadas \
        --case distorsionado

Importante:
- No genera las distorsiones ni las máscaras de agua.
- Extrae puntos de transición agua/tierra.
- El closing puede modificar la geometría de los bordes.
- Para comparar casos, utilizar el mismo AOI y los mismos parámetros.
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
from scipy.ndimage import binary_erosion, convolve
from skimage.morphology import closing, disk
from tqdm import tqdm

import shoreline_utils as utils


CLOSING_RADIUS = 1
SAVE_CLEAN_MASKS = True

CSV_COLUMNS = [
    "x",
    "y",
    "date",
    "case",
    "source_file",
    "crs",
]


# ---------------------------------------------------------------------------
# Coordenadas y fechas
# ---------------------------------------------------------------------------

def reproject_coords_to_latlon(coords, src_crs):
    transformer = Transformer.from_crs(
        src_crs,
        "EPSG:4326",
        always_xy=True,
    )

    xs, ys = np.asarray(
        coords, dtype=float
    ).T

    lons, lats = transformer.transform(
        xs, ys, errcheck=True
    )

    return [
        (float(lon), float(lat))
        for lon, lat in zip(lons, lats)
    ]


def extract_date_from_mask(mask_path):
    extracted = utils.extract_date(
        mask_path.name
    )

    if extracted is None:
        raise ValueError(
            "No se puede extraer una fecha YYYYMMDD de "
            f"{mask_path.name}."
        )

    return extracted.isoformat()


# ---------------------------------------------------------------------------
# Lectura y limpieza de máscaras
# ---------------------------------------------------------------------------

def read_mask_band(src):
    if src.count != 1:
        raise ValueError(
            f"Se esperaba una banda: {src.name}"
        )

    array = src.read(1, masked=True)
    values = np.asarray(array.data)

    valid = (
        ~np.ma.getmaskarray(array)
        & np.isfinite(values)
    )

    if not valid.any():
        raise ValueError(
            f"La máscara no tiene datos válidos: {src.name}"
        )

    if not np.isin(
        values[valid], [0, 1]
    ).all():
        raise ValueError(
            f"La máscara debe contener valores 0/1: {src.name}"
        )

    water = (values == 1) & valid

    return water, valid


def denoise_mask(
    mask_image,
    valid,
    radius=CLOSING_RADIUS,
):
    """
    Aplica closing sin considerar fiables sus resultados
    cerca de zonas sin datos.

    El área segura depende del soporte de las dos operaciones
    morfológicas: dilatación y erosión.
    """
    binary = mask_image.astype(bool) & valid

    if radius <= 0:
        return binary, valid.copy()

    cleaned = closing(
        binary, disk(radius)
    ).astype(bool)

    safe_valid = binary_erosion(
        valid,
        structure=disk(2 * radius),
        border_value=0,
    )

    cleaned &= valid

    return cleaned, safe_valid


def save_mask_tif(
    mask,
    output_path,
    src_profile,
    valid,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    profile = {
        "driver": "GTiff",
        "height": mask.shape[0],
        "width": mask.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": src_profile["crs"],
        "transform": src_profile["transform"],
        "nodata": None,
        "compress": "deflate",
    }

    with rasterio.Env(
        GDAL_TIFF_INTERNAL_MASK=True
    ):
        with rasterio.open(
            output_path, "w", **profile
        ) as dst:
            dst.write(
                mask.astype(np.uint8), 1
            )
            dst.write_mask(
                valid.astype(np.uint8) * 255
            )


# ---------------------------------------------------------------------------
# Extracción y exportación
# ---------------------------------------------------------------------------

def estimate_mask_edges(
    mask_file,
    aoi_gdf,
    output_dir,
    clean_dir,
    case,
    closing_radius,
    save_clean,
):
    date = extract_date_from_mask(
        mask_file
    )

    with rasterio.open(mask_file) as src:
        if src.crs is None:
            raise ValueError(
                f"El raster no tiene CRS: {mask_file}"
            )

        raw_mask, valid = read_mask_band(src)

        mask_image, safe_valid = denoise_mask(
            raw_mask,
            valid,
            closing_radius,
        )

        if save_clean:
            save_mask_tif(
                mask_image,
                clean_dir / f"{mask_file.stem}_closed.tif",
                src.profile,
                safe_valid,
            )

        # Cada raster puede tener su propio CRS.
        aoi_proj = aoi_gdf.to_crs(src.crs)

        aoi_mask = geometry_mask(
            list(aoi_proj.geometry),
            transform=src.transform,
            invert=True,
            out_shape=mask_image.shape,
        )

        neighborhood = np.ones(
            (3, 3), dtype=bool
        )

        # Excluye límites del AOI, del raster y zonas sin datos.
        safe_domain = binary_erosion(
            aoi_mask & safe_valid,
            structure=neighborhood,
            border_value=0,
        )

        kernel = np.ones(
            (3, 3), dtype=np.uint8
        )
        kernel[1, 1] = 0

        neighbor_sum = convolve(
            mask_image.astype(np.uint8),
            kernel,
            mode="constant",
            cval=0,
        )

        transition_mask = (
            mask_image & (neighbor_sum < 8)
        )

        final_coast_mask = (
            transition_mask & safe_domain
        )

        rows, cols = np.nonzero(
            final_coast_mask
        )

        if len(rows) < 2:
            return pd.DataFrame(
                columns=CSV_COLUMNS
            )

        xs, ys = rasterio.transform.xy(
            src.transform,
            rows,
            cols,
            offset="center",
        )

        coords_xy = [
            (float(x), float(y))
            for x, y in zip(xs, ys)
        ]

        coords_latlon = reproject_coords_to_latlon(
            coords_xy, src.crs
        )

        crs_text = src.crs.to_string()

    stem = mask_file.stem

    frame = pd.DataFrame({
        "x": [p[0] for p in coords_xy],
        "y": [p[1] for p in coords_xy],
        "date": date,
        "case": case,
        "source_file": mask_file.name,
        "crs": crs_text,
    })

    # CSV con cabecera, utilizable por error_estimator.py.
    frame.to_csv(
        output_dir / f"{stem}_points.csv",
        index=False,
    )

    # Exportaciones auxiliares con el formato anterior, sin cabecera.
    pd.DataFrame([
        (x, y, date)
        for x, y in coords_xy
    ]).to_csv(
        output_dir / f"{stem}_xyz.csv",
        index=False,
        header=False,
    )

    pd.DataFrame([
        (lat, lon, date)
        for lon, lat in coords_latlon
    ]).to_csv(
        output_dir / f"{stem}.csv",
        index=False,
        header=False,
    )

    utils.export_coords_to_kmz(
        coords_latlon,
        output_dir / f"{stem}.kmz",
    )

    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],
            },
            "properties": {
                "date": date,
                "case": case,
                "source_file": mask_file.name,
            },
        }
        for lon, lat in coords_latlon
    ]

    aoi_latlon = aoi_gdf.to_crs(
        "EPSG:4326"
    )

    for geometry in aoi_latlon.geometry:
        features.append({
            "type": "Feature",
            "geometry": geometry.__geo_interface__,
            "properties": {"name": "AOI"},
        })

    with (
        output_dir / f"{stem}.geojson"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            {
                "type": "FeatureCollection",
                "features": features,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    return frame


# ---------------------------------------------------------------------------
# Flujo principal
# ---------------------------------------------------------------------------

def main(
    project,
    mask_dir=None,
    polygon_path=None,
    clean_dir=None,
    closing_radius=CLOSING_RADIUS,
    case="original",
    output_dir=None,
    save_clean=SAVE_CLEAN_MASKS,
):
    if closing_radius < 0:
        raise ValueError(
            "El radio del closing no puede ser negativo."
        )

    project_dir = utils.project_path(
        project
    )

    root_output = (
        project_dir
        / "output"
        / "estimated_waterbodies_edges"
    )

    # Conserva la ruta histórica para el caso original.
    default_output = (
        root_output
        if case == "original"
        else root_output / case
    )

    output_dir = (
        Path(output_dir)
        if output_dir
        else default_output
    )

    mask_dir = (
        Path(mask_dir)
        if mask_dir
        else (
            project_dir
            / "output"
            / "estimated_waterbodies_images"
        )
    )

    polygon_path = (
        Path(polygon_path)
        if polygon_path
        else (
            project_dir
            / "input"
            / "polygon.geojson"
        )
    )

    clean_dir = (
        Path(clean_dir)
        if clean_dir
        else (
            project_dir
            / "output"
            / "estimated_waterbodies_images_closed"
            / case
        )
    )

    if not mask_dir.is_dir():
        raise FileNotFoundError(
            f"No existe la carpeta de máscaras: {mask_dir}"
        )

    if not polygon_path.is_file():
        raise FileNotFoundError(
            f"No existe el AOI: {polygon_path}"
        )

    if save_clean and clean_dir.resolve() == mask_dir.resolve():
        raise ValueError(
            "La carpeta de máscaras limpias debe ser "
            "distinta de la carpeta de entrada."
        )

    mask_files = sorted(
        p for p in mask_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".tif", ".tiff"}
    )

    if not mask_files:
        raise FileNotFoundError(
            f"No hay GeoTIFF en {mask_dir}"
        )

    aoi_gdf = gpd.read_file(
        polygon_path
    )

    if aoi_gdf.empty:
        raise ValueError(
            "El archivo AOI está vacío."
        )

    if aoi_gdf.crs is None:
        # GeoJSON sin CRS explícito: se asume lon/lat.
        aoi_gdf = aoi_gdf.set_crs(
            "EPSG:4326"
        )

    aoi_gdf = aoi_gdf.loc[
        aoi_gdf.geometry.notna()
        & ~aoi_gdf.geometry.is_empty
    ].copy()

    if aoi_gdf.empty:
        raise ValueError(
            "El AOI no tiene geometrías utilizables."
        )

    if not aoi_gdf.geometry.is_valid.all():
        raise ValueError(
            "El AOI contiene geometrías inválidas."
        )

    if not aoi_gdf.geom_type.isin(
        ["Polygon", "MultiPolygon"]
    ).all():
        raise ValueError(
            "El AOI debe contener polígonos."
        )

    output_dir.mkdir(
        parents=True, exist_ok=True
    )

    if save_clean:
        clean_dir.mkdir(
            parents=True, exist_ok=True
        )

    frames = []
    no_coast = []
    errors = []

    for mask_file in tqdm(
        mask_files,
        desc=f"Extrayendo costas: {case}",
    ):
        try:
            frame = estimate_mask_edges(
                mask_file,
                aoi_gdf,
                output_dir,
                clean_dir,
                case,
                closing_radius,
                save_clean,
            )

            if frame.empty:
                no_coast.append(
                    mask_file.name
                )
            else:
                frames.append(frame)

        except Exception as error:
            errors.append({
                "source_file": mask_file.name,
                "error": str(error),
            })

    combined = (
        pd.concat(
            frames, ignore_index=True
        )
        if frames
        else pd.DataFrame(
            columns=CSV_COLUMNS
        )
    )

    # No se escanean CSV antiguos de la carpeta.
    combined_path = (
        output_dir / "all_shorelines.csv"
    )

    combined.to_csv(
        combined_path,
        index=False,
    )

    report = {
        "case": case,
        "mask_dir": str(mask_dir.resolve()),
        "polygon": str(polygon_path.resolve()),
        "closing_radius_pixels": closing_radius,
        "n_masks": len(mask_files),
        "n_processed": len(frames),
        "n_points": len(combined),
        "no_coast": no_coast,
        "errors": errors,
        "crs_values": (
            sorted(
                combined["crs"].dropna().unique().tolist()
            )
            if not combined.empty else []
        ),
    }

    with (
        output_dir / "extraction_report.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nMáscaras con costa: "
        f"{len(frames)}/{len(mask_files)}"
    )
    print(
        f"Máscaras sin costa: {len(no_coast)}"
    )
    print(
        f"Máscaras con error: {len(errors)}"
    )
    print(f"CSV combinado: {combined_path}")

    for item in errors:
        print(
            f"ERROR en {item['source_file']}: "
            f"{item['error']}"
        )

    if errors or combined.empty:
        raise SystemExit(1)

    return combined_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extraer puntos de costa desde "
            "máscaras binarias GeoTIFF."
        )
    )

    parser.add_argument(
        "-p", "--project",
        required=True,
    )
    parser.add_argument(
        "--mask-dir",
    )
    parser.add_argument(
        "--polygon",
    )
    parser.add_argument(
        "--clean-dir",
    )
    parser.add_argument(
        "--output-dir",
    )
    parser.add_argument(
        "--closing-radius",
        type=int,
        default=CLOSING_RADIUS,
    )
    parser.add_argument(
        "--case",
        choices=[
            "original",
            "distorsionado",
            "corregido",
        ],
        default="original",
    )
    parser.add_argument(
        "--no-save-clean",
        action="store_true",
    )

    args = parser.parse_args()

    try:
        main(
            project=args.project,
            mask_dir=args.mask_dir,
            polygon_path=args.polygon,
            clean_dir=args.clean_dir,
            closing_radius=args.closing_radius,
            case=args.case,
            output_dir=args.output_dir,
            save_clean=not args.no_save_clean,
        )
    except (ValueError, OSError) as error:
        parser.exit(
            status=1,
            message=f"Error: {error}\n",
        )
