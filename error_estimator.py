"""
Validación de líneas de costa mediante transectos.

Dependencias:
    pip install numpy pandas matplotlib shapely pyproj scipy

Utiliza shoreline_utils.py del proyecto.

Entrada estimada:
    CSV con columnas:
        date,x,y

    Opcionalmente:
        case
        point_id

    - case permite separar original/distorsionado/corregido.
    - point_id permite indicar el orden de los puntos.
    - Sin point_id, se utiliza la ordenación de shoreline_utils.

Referencia:
    CSV con las dos primeras columnas X e Y.
    Se admiten comas, tabulaciones o espacios y una cabecera.
    Sus puntos deben estar ordenados a lo largo de la costa.

IMPORTANTE:
    Los CSV no contienen necesariamente información del CRS.
    --crs declara el CRS común de ambos archivos:
    no transforma ni verifica automáticamente sus coordenadas.

Uso habitual:
    python error_estimator.py \
        -p costa_principal \
        -g ruta/referencias \
        -d 2017-06-01 \
        --crs EPSG:32631

Con un CSV estimado específico:
    python error_estimator.py \
        -p costa_principal \
        --estimated-csv ruta/original.csv \
        --reference-file ruta/referencia.csv \
        -d 2017-06-01 \
        --case original \
        --crs EPSG:32631 \
        --no-plot
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS
from shapely.geometry import LineString, Point

import shoreline_utils as utils


SPATIAL_SPACING = 10.0
TRANSECT_LENGTH = 100.0


# ---------------------------------------------------------------------------
# Comprobaciones y lectura
# ---------------------------------------------------------------------------

def validate_metric_crs(crs):
    parsed = CRS.from_user_input(crs)

    if not parsed.is_projected:
        raise ValueError(
            "El CRS debe ser proyectado, no latitud/longitud."
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
                factor, 1.0, rtol=0, atol=1e-12
            )
        ):
            raise ValueError(
                "El CRS debe utilizar metros."
            )

    return parsed


def closest_ground_truth_file(ground_truth_dir, target_date):
    base = Path(ground_truth_dir).expanduser().resolve()

    if not base.is_dir():
        raise NotADirectoryError(
            f"No existe la carpeta de referencias: {base}"
        )

    dated_files = []

    for csv_file in sorted(base.glob("*.csv")):
        file_date = utils.extract_date(csv_file.name)

        if file_date is not None:
            dated_files.append((file_date, csv_file))

    if not dated_files:
        raise FileNotFoundError(
            f"No hay CSV con fecha reconocible en {base}."
        )

    return min(
        dated_files,
        key=lambda row: (
            abs((row[0] - target_date).days),
            row[0],
            row[1].name,
        ),
    )


def read_reference_coords(csv_path):
    """
    Lee las dos primeras columnas numéricas sin alterar
    el orden original.

    Elimina solo duplicados consecutivos.
    """
    df = pd.read_csv(
        csv_path,
        sep=r"\s+|,",
        engine="python",
        comment="%",
        header=None,
        usecols=[0, 1],
    )

    df = df.apply(
        pd.to_numeric, errors="coerce"
    )

    coords = df.to_numpy(dtype=float)
    coords = coords[
        np.isfinite(coords).all(axis=1)
    ]

    if len(coords) < 2:
        raise ValueError(
            "La referencia necesita al menos dos puntos válidos."
        )

    keep = np.concatenate([
        [True],
        np.any(np.diff(coords, axis=0) != 0, axis=1),
    ])

    coords = coords[keep]

    if len(coords) < 2:
        raise ValueError(
            "La referencia no contiene dos puntos distintos."
        )

    return coords


def read_estimated_line(
    csv_path,
    target_date,
    case,
    ordered=False,
):
    df = pd.read_csv(csv_path)

    required = {"date", "x", "y"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"{csv_path} debe contener date, x e y."
        )

    normalized_dates = pd.to_datetime(
        df["date"], errors="coerce", utc=True
    ).dt.strftime("%Y-%m-%d")

    df = df.loc[
        normalized_dates == target_date
    ].copy()

    if df.empty:
        raise ValueError(
            f"No hay puntos para {target_date} en {csv_path}."
        )

    if "case" in df.columns:
        df = df.loc[
            df["case"].astype(str) == case
        ].copy()

        if df.empty:
            raise ValueError(
                f"No hay puntos del caso '{case}' "
                f"para {target_date}."
            )

    # Evita mezclar productos si el CSV los identifica.
    for column in ("product_id", "product", "source_file"):
        if (
            column in df.columns
            and df[column].nunique(dropna=False) > 1
        ):
            raise ValueError(
                f"Hay varios valores de '{column}' "
                "en la selección. Utiliza un CSV separado "
                "para cada línea que quieras evaluar."
            )

    if "point_id" in df.columns:
        df["point_id"] = pd.to_numeric(
            df["point_id"], errors="raise"
        )

        if df["point_id"].duplicated().any():
            raise ValueError(
                "Hay point_id duplicados. "
                "Comprueba que no se han mezclado varias líneas."
            )

        df = df.sort_values("point_id")
        ordered = True

    df["x"] = pd.to_numeric(
        df["x"], errors="coerce"
    )
    df["y"] = pd.to_numeric(
        df["y"], errors="coerce"
    )

    coords = df[["x", "y"]].to_numpy(dtype=float)
    coords = coords[
        np.isfinite(coords).all(axis=1)
    ]

    if len(coords) < 2:
        raise ValueError(
            "La costa estimada necesita al menos dos puntos."
        )

    if ordered:
        keep = np.concatenate([
            [True],
            np.any(np.diff(coords, axis=0) != 0, axis=1),
        ])
        coords = coords[keep]
    else:
        # Elimina duplicados sin cambiar el orden de primera aparición.
        _, indices = np.unique(
            coords, axis=0, return_index=True
        )
        coords = coords[np.sort(indices)]

    if len(coords) < 2:
        raise ValueError(
            "La costa estimada no contiene dos puntos distintos."
        )

    return utils.build_ordered_linestring(
        coords, ordered=ordered
    )


# ---------------------------------------------------------------------------
# Transectos e intersecciones
# ---------------------------------------------------------------------------

def build_transects(reference_line, spacing, length):
    if reference_line.length <= 0:
        raise ValueError(
            "La referencia tiene longitud nula."
        )

    positions = np.arange(
        0.0,
        reference_line.length,
        spacing,
    )

    transects = []

    # Se aproxima la tangente con dos puntos próximos
    # sobre la referencia, sin aplicar una spline.
    delta = min(
        spacing / 2.0,
        reference_line.length / 2.0,
    )

    for transect_id, distance_along in enumerate(positions):
        center = reference_line.interpolate(
            float(distance_along)
        )

        before = reference_line.interpolate(
            max(0.0, distance_along - delta)
        )
        after = reference_line.interpolate(
            min(
                reference_line.length,
                distance_along + delta,
            )
        )

        tangent = np.array([
            after.x - before.x,
            after.y - before.y,
        ])

        norm = np.linalg.norm(tangent)

        if norm <= 1e-12:
            continue

        normal = np.array([
            -tangent[1], tangent[0]
        ]) / norm

        origin = np.array([
            center.x, center.y
        ])

        start = origin - normal * length / 2.0
        end = origin + normal * length / 2.0

        transects.append({
            "transect_id": transect_id,
            "chainage_m": float(distance_along),
            "reference_x": center.x,
            "reference_y": center.y,
            "normal": normal,
            "geometry": LineString([start, end]),
        })

    if not transects:
        raise ValueError(
            "No se pudieron generar transectos válidos."
        )

    return transects


def intersection_points(geometry):
    """
    Devuelve puntos de intersección.

    Los solapamientos lineales se consideran ambiguos:
    no se convierten artificialmente en una distancia puntual.
    """
    if geometry.is_empty:
        return [], False

    if isinstance(geometry, Point):
        return [geometry], False

    if geometry.geom_type in {
        "MultiPoint", "GeometryCollection"
    }:
        points = []
        overlap = False

        for part in geometry.geoms:
            part_points, part_overlap = intersection_points(part)
            points.extend(part_points)
            overlap = overlap or part_overlap

        return points, overlap

    return [], True


def signed_intersection_distance(
    intersection,
    px,
    py,
    normal,
):
    points, overlap = intersection_points(intersection)

    if overlap or not points:
        return np.nan

    origin = np.array([px, py], dtype=float)

    signed_distances = [
        float(np.dot(
            np.array([point.x, point.y]) - origin,
            normal,
        ))
        for point in points
    ]

    return min(signed_distances, key=abs)


# ---------------------------------------------------------------------------
# Evaluación principal
# ---------------------------------------------------------------------------

def main(
    project,
    ground_truth,
    date,
    show_plot=True,
    estimated_csv=None,
    reference_file=None,
    case="original",
    crs=None,
    spacing=SPATIAL_SPACING,
    transect_length=TRANSECT_LENGTH,
    ordered=False,
):
    if crs is None:
        raise ValueError(
            "Indica el CRS común con --crs."
        )

    metric_crs = validate_metric_crs(crs)

    if (
        not np.isfinite(spacing)
        or not np.isfinite(transect_length)
        or spacing <= 0
        or transect_length <= 0
    ):
        raise ValueError(
            "El espaciado y la longitud deben ser positivos y finitos."
        )

    target_date = datetime.strptime(
        date, "%Y-%m-%d"
    ).date()

    project_dir = utils.project_path(project)

    output_dir = (
        project_dir
        / "output"
        / "estimated_error"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = (
        Path(estimated_csv)
        if estimated_csv
        else (
            project_dir
            / "output"
            / "estimated_waterbodies_edges"
            / "all_shorelines.csv"
        )
    )

    line = read_estimated_line(
        csv_path,
        date,
        case,
        ordered=ordered,
    )

    if reference_file:
        gt_path = Path(reference_file)
        reference_date = utils.extract_date(
            gt_path.name
        )
    else:
        if not ground_truth:
            raise ValueError(
                "Indica --ground-truth o --reference-file."
            )

        reference_date, gt_path = closest_ground_truth_file(
            ground_truth, target_date
        )

    reference_coords = read_reference_coords(
        gt_path
    )
    reference_line = LineString(
        reference_coords
    )

    transects = build_transects(
        reference_line,
        spacing,
        transect_length,
    )

    results = []

    for transect in transects:
        intersection = transect["geometry"].intersection(line)

        distance = signed_intersection_distance(
            intersection,
            transect["reference_x"],
            transect["reference_y"],
            transect["normal"],
        )

        results.append({
            "date": date,
            "case": case,
            "transect_id": transect["transect_id"],
            "chainage_m": transect["chainage_m"],
            "reference_x": transect["reference_x"],
            "reference_y": transect["reference_y"],
            "distance_m": distance,
        })

    distance_df = pd.DataFrame(results)
    valid = distance_df["distance_m"].dropna()

    safe_case = re.sub(
        r"[^A-Za-z0-9_-]+", "_", case
    ).strip("_") or "caso"

    prefix = f"{date}_{safe_case}"

    distance_path = (
        output_dir / f"{prefix}_distances.csv"
    )
    distance_df.to_csv(
        distance_path, index=False
    )

    summary = {
        "date": date,
        "case": case,
        "estimated_csv": str(csv_path.resolve()),
        "reference_csv": str(gt_path.resolve()),
        "reference_date": (
            reference_date.isoformat()
            if reference_date else None
        ),
        "reference_date_difference_days": (
            (target_date - reference_date).days
            if reference_date else None
        ),
        "declared_common_crs": metric_crs.to_string(),
        "spacing_m": spacing,
        "transect_length_m": transect_length,
        "method": "reference_polyline_local_tangent",
        "n_transects": len(distance_df),
        "n_valid": len(valid),
        "rmse_m": (
            float(np.sqrt(np.mean(valid.to_numpy() ** 2)))
            if len(valid) else None
        ),
        "mean_signed_distance_m": (
            float(valid.mean())
            if len(valid) else None
        ),
        "mae_m": (
            float(valid.abs().mean())
            if len(valid) else None
        ),
        "sign_note": (
            "El signo depende del orden de la referencia; "
            "no identifica automáticamente erosión/acreción."
        ),
    }

    summary_path = (
        output_dir / f"{prefix}_summary.json"
    )

    with summary_path.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    # Importación diferida para permitir --no-plot sin interfaz gráfica.
    import matplotlib

    if not show_plot:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(
        *reference_line.xy,
        "b-",
        label="Referencia",
    )
    ax.plot(
        *line.xy,
        "k--",
        label=f"Estimada: {case}",
    )

    for transect in transects:
        ax.plot(
            *transect["geometry"].xy,
            "g-",
            alpha=0.25,
        )

    ax.set_title(
        f"Validación de costa: {date} — {case}"
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    ax.legend()

    fig.savefig(
        output_dir / f"{prefix}_validation.png",
        dpi=150,
        bbox_inches="tight",
    )

    if show_plot:
        plt.show()

    plt.close(fig)

    print(f"Referencia: {gt_path}")
    print(f"Fecha de referencia: {reference_date}")
    print(
        f"Transectos válidos: {len(valid)}/{len(distance_df)}"
    )

    if len(valid) == 0:
        raise ValueError(
            "No hay intersecciones válidas para calcular el RMSE. "
            "Se han guardado los archivos de diagnóstico. "
            "Revisa CRS, referencia, orden y longitud de transectos."
        )

    print(f"RMSE: {summary['rmse_m']:.3f} m")
    print(f"Distancias: {distance_path}")
    print(f"Resumen: {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validar costas mediante transectos."
    )

    parser.add_argument(
        "-p", "--project", required=True
    )
    parser.add_argument(
        "-g", "--ground-truth",
        help="Carpeta con referencias fechadas."
    )
    parser.add_argument(
        "--reference-file",
        help="Referencia concreta; tiene prioridad sobre -g."
    )
    parser.add_argument(
        "-d", "--date", required=True,
        help="Fecha YYYY-MM-DD."
    )
    parser.add_argument(
        "--estimated-csv",
        help="CSV estimado; por defecto all_shorelines.csv."
    )
    parser.add_argument(
        "--case", default="original",
        help="Caso evaluado; también filtra la columna case si existe."
    )
    parser.add_argument(
        "--crs", required=True,
        help="CRS común real de ambos CSV, por ejemplo EPSG:32631."
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=SPATIAL_SPACING,
    )
    parser.add_argument(
        "--transect-length",
        type=float,
        default=TRANSECT_LENGTH,
    )
    parser.add_argument(
        "--ordered",
        action="store_true",
        help="Los puntos estimados ya siguen el recorrido de la costa."
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
    )

    args = parser.parse_args()

    try:
        main(
            project=args.project,
            ground_truth=args.ground_truth,
            date=args.date,
            show_plot=not args.no_plot,
            estimated_csv=args.estimated_csv,
            reference_file=args.reference_file,
            case=args.case,
            crs=args.crs,
            spacing=args.spacing,
            transect_length=args.transect_length,
            ordered=args.ordered,
        )
    except (ValueError, OSError) as error:
        parser.exit(
            status=1,
            message=f"Error: {error}\n",
        )
