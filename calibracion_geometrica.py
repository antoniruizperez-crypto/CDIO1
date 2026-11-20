"""
Calibración geométrica de Sentinel-2 en Shadnagar.

Compatible con:
- Los recortes y metadatos.json generados por sesion5.py.
- utils.py del profesor, con show_geotiff().
- shoreline_utils.py original: no es necesario modificarlo.

Dependencias:
    pip install numpy pandas rasterio matplotlib pyproj

Estructura esperada:
    carpeta_entrada/
        nombre_producto/
            B02.tif
            B03.tif
            B04.tif
            B08.tif
            metadatos.json

Uso:
    python calibracion_geometrica.py \
        --entrada Imagenes/calibracion/recortes_01km2

Solo regenerar tablas:
    python calibracion_geometrica.py \
        --entrada Imagenes/calibracion/recortes_01km2 \
        --solo-resumen

Corrección geométrica:
    python calibracion_geometrica.py \
        --entrada Imagenes/calibracion/recortes_01km2 \
        --solo-resumen \
        --calibration-dates 2025-01-07 2025-02-16 \
        --export-corrected

Notas:
- Las fechas anteriores son solo ejemplos de formato.
- No importa sesion5.py: evita activar su backend Agg.
- La identificación del objetivo es manual.
- No se identifica el objetivo usando la coordenada de referencia.
- Los clics del visor corresponden a centros de píxel.
- Una traslación no corrige rotación, escala ni deformaciones.
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from rasterio.transform import Affine


BANDAS = ["B02", "B03", "B04", "B08"]

NOMBRES_BANDAS = {
    "B02": "Azul",
    "B03": "Verde",
    "B04": "Rojo",
    "B08": "NIR",
}

ENTRADA_DEFAULT = "Imagenes/calibracion/recortes_01km2"
SALIDA_DEFAULT = "Imagenes/calibracion/evaluacion_geometrica"

LAT_DEFAULT = 17.034261
LON_DEFAULT = 78.183078

COLUMNAS = [
    "file",
    "signature",
    "product",
    "date",
    "satellite",
    "level",
    "band",
    "crs",
    "reference_lon",
    "reference_lat",
    "reference_x",
    "reference_y",
    "observed_x",
    "observed_y",
    "pixel_row",
    "pixel_col",
    "pixel_value",
    "dx_m",
    "dy_m",
    "error_m",
]

GROUP_MODEL = ["level", "satellite", "band", "crs"]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def save_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temporary, index=False)
    temporary.replace(path)


def validate_metric_crs(crs):
    parsed = CRS.from_user_input(crs)

    if not parsed.is_projected:
        raise ValueError(
            "El GeoTIFF debe tener un CRS proyectado."
        )

    axes = parsed.axis_info[:2]

    if len(axes) != 2:
        raise ValueError(
            "No se pueden comprobar las unidades del CRS."
        )

    for axis in axes:
        factor = axis.unit_conversion_factor

        if factor is None or not math.isclose(
            factor, 1.0, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError(
                "El CRS debe estar expresado en metros."
            )

    return parsed


def file_signature(path):
    stat = path.stat()

    text = (
        f"{path.resolve()}|"
        f"{stat.st_size}|"
        f"{stat.st_mtime_ns}"
    )

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def metadata_identity(meta):
    product = str(meta.get("producto", ""))

    satellite_match = re.search(
        r"S2([ABC])", product
    )

    if satellite_match:
        satellite = f"S2{satellite_match.group(1)}"
    else:
        text = str(meta.get("satelite", "")).upper()
        satellite = next(
            (
                f"S2{letter}"
                for letter in "ABC"
                if (
                    f"2{letter}" in text
                    or f"2-{letter}" in text
                )
            ),
            "",
        )

    date_match = re.search(
        r"_(\d{8})T\d{6}", product
    )

    if date_match:
        raw = date_match.group(1)
        acquisition_date = (
            f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
        )
    else:
        raw = meta.get("fecha_adquisicion")

        if not raw:
            raise ValueError(
                f"No se puede identificar la fecha de {product}."
            )

        acquisition_date = pd.to_datetime(
            raw, utc=True
        ).strftime("%Y-%m-%d")

    level = str(meta.get("nivel", ""))

    if level not in {"L1C", "L2A"}:
        raise ValueError(
            f"Nivel no válido en {product}: {level}"
        )

    if satellite not in {"S2A", "S2B", "S2C"}:
        raise ValueError(
            f"No se puede identificar el satélite de {product}."
        )

    if not product:
        raise ValueError(
            "Falta 'producto' en metadatos.json."
        )

    return product, acquisition_date, satellite, level


# ---------------------------------------------------------------------------
# Descubrimiento de bandas
# ---------------------------------------------------------------------------

def discover_jobs(entrada, latitude, longitude, levels):
    jobs = []
    problems = []

    for metadata_path in sorted(
        entrada.rglob("metadatos.json")
    ):
        try:
            with metadata_path.open(
                encoding="utf-8"
            ) as file:
                meta = json.load(file)

            product, date, satellite, level = metadata_identity(
                meta
            )

            if level not in levels:
                continue

            reference = meta.get(
                "objetivo_referencia", {}
            )

            lat = (
                latitude
                if latitude is not None
                else reference.get("latitud", LAT_DEFAULT)
            )
            lon = (
                longitude
                if longitude is not None
                else reference.get("longitud", LON_DEFAULT)
            )

            lat = float(lat)
            lon = float(lon)

            if not (
                -90 <= lat <= 90
                and -180 <= lon <= 180
            ):
                raise ValueError(
                    "Coordenadas de referencia no válidas."
                )

            for band in BANDAS:
                path = metadata_path.parent / f"{band}.tif"

                if not path.is_file():
                    problems.append({
                        "file": str(path),
                        "error": "Falta la banda.",
                    })
                    continue

                jobs.append({
                    "path": path,
                    "file": str(path.resolve()),
                    "signature": file_signature(path),
                    "product": product,
                    "date": date,
                    "satellite": satellite,
                    "level": level,
                    "band": band,
                    "reference_lon": lon,
                    "reference_lat": lat,
                })

        except Exception as error:
            problems.append({
                "file": str(metadata_path),
                "error": str(error),
            })

    return jobs, problems


# ---------------------------------------------------------------------------
# Medición interactiva
# ---------------------------------------------------------------------------

def measure_job(job):
    """
    Devuelve:
        ("accepted", medicion)
        ("skip", None)
        ("quit", None)
    """
    # Importación diferida: --solo-resumen no necesita abrir ventanas.
    from utils import show_geotiff

    path = job["path"]

    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(
                "Se esperaba una banda por GeoTIFF."
            )

        if src.crs is None:
            raise ValueError(
                "El raster no tiene CRS."
            )

        crs = validate_metric_crs(src.crs)

        if src.width < 3 or src.height < 3:
            raise ValueError(
                "Recorte demasiado pequeño para identificar "
                "visualmente el objetivo. Regenera los recortes "
                "con --area-m2 100000."
            )

        transformer = Transformer.from_crs(
            "EPSG:4326",
            crs,
            always_xy=True,
        )

        reference_x, reference_y = transformer.transform(
            job["reference_lon"],
            job["reference_lat"],
            errcheck=True,
        )

    while True:
        print(
            f"\n{job['product']}\n"
            f"Fecha: {job['date']} | "
            f"Satélite: {job['satellite']} | "
            f"Nivel: {job['level']} | "
            f"Banda: {job['band']} "
            f"({NOMBRES_BANDAS[job['band']]})"
        )
        print(
            "Haz clic sobre el centro observado del objetivo.\n"
            "Si no lo reconoces, cierra la ventana sin seleccionar."
        )

        selected = show_geotiff(
            str(path),
            title=(
                f"{job['date']} | {job['satellite']} | "
                f"{job['level']} | {job['band']}"
            ),
        )

        if selected is None:
            choice = input(
                "Sin selección: [s] saltar, [r] repetir, "
                "[q] guardar y salir: "
            ).strip().lower()

            if choice == "r":
                continue

            return (
                ("quit", None)
                if choice == "q"
                else ("skip", None)
            )

        x, y, value = map(float, selected)

        if not np.isfinite([x, y, value]).all():
            print("Selección no válida. Repite.")
            continue

        with rasterio.open(path) as src:
            row, col = src.index(x, y)

            if not (
                0 <= row < src.height
                and 0 <= col < src.width
            ):
                print("Selección fuera del raster. Repite.")
                continue

            mask = src.read_masks(
                1,
                window=((row, row + 1), (col, col + 1)),
            )

            if mask[0, 0] == 0:
                print(
                    "El píxel seleccionado no tiene datos válidos."
                )
                continue

        choice = input(
            "¿Aceptar? [Enter] sí, [r] repetir, "
            "[s] saltar, [q] guardar y salir: "
        ).strip().lower()

        if choice == "r":
            continue

        if choice == "q":
            return "quit", None

        if choice == "s":
            return "skip", None

        if choice != "":
            print("Opción no reconocida. Repite la selección.")
            continue

        dx = x - reference_x
        dy = y - reference_y

        record = {
            column: job.get(column)
            for column in COLUMNAS
        }

        record.update({
            "crs": crs.to_string(),
            "reference_x": float(reference_x),
            "reference_y": float(reference_y),
            "observed_x": x,
            "observed_y": y,
            "pixel_row": int(row),
            "pixel_col": int(col),
            "pixel_value": value,
            "dx_m": float(dx),
            "dy_m": float(dy),
            "error_m": float(np.hypot(dx, dy)),
        })

        print(
            f"Medición guardada. Error 2D: "
            f"{record['error_m']:.3f} m."
        )

        return "accepted", record


# ---------------------------------------------------------------------------
# Resúmenes
# ---------------------------------------------------------------------------

def statistics(group, dx_col="dx_m", dy_col="dy_m"):
    dx = group[dx_col].to_numpy(dtype=float)
    dy = group[dy_col].to_numpy(dtype=float)

    return {
        "n": len(group),
        "n_dates": group["date"].nunique(),
        "bias_x_m": float(np.mean(dx)),
        "bias_y_m": float(np.mean(dy)),
        "rmse_x_m": float(np.sqrt(np.mean(dx ** 2))),
        "rmse_y_m": float(np.sqrt(np.mean(dy ** 2))),
        "rmse_2d_m": float(
            np.sqrt(np.mean(dx ** 2 + dy ** 2))
        ),
    }


def grouped_statistics(df, columns):
    rows = []

    grouper = columns[0] if len(columns) == 1 else columns

    for keys, group in df.groupby(
        grouper, sort=True, dropna=False
    ):
        if len(columns) == 1:
            keys = (keys,)

        row = dict(zip(columns, keys))
        row.update(statistics(group))
        rows.append(row)

    return pd.DataFrame(rows)


def write_summaries(df, output):
    tables = {
        "rmse_por_nivel.csv": ["level"],
        "rmse_por_banda.csv": ["level", "band"],
        "rmse_por_satelite.csv": ["level", "satellite"],
        "rmse_por_satelite_banda.csv": [
            "level", "satellite", "band"
        ],
        "evolucion_temporal.csv": [
            "level", "date", "satellite"
        ],
    }

    for filename, columns in tables.items():
        save_csv(
            grouped_statistics(df, columns),
            output / filename,
        )

    global_summary = statistics(df)
    global_summary["note"] = (
        "Resumen de todas las mediciones disponibles. "
        "Puede mezclar L1C y L2A, que son observaciones "
        "relacionadas de las mismas adquisiciones. "
        "Consultar también los resúmenes separados por nivel."
    )

    with (output / "rmse_global.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            global_summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    coverage = (
        df.groupby(["level", "satellite"])
        .agg(
            n_measurements=("band", "size"),
            n_dates=("date", "nunique"),
        )
        .reset_index()
    )

    save_csv(
        coverage,
        output / "cobertura_mediciones.csv",
    )

    print("\nRMSE por nivel:")
    print(
        grouped_statistics(
            df, ["level"]
        ).to_string(index=False)
    )

    for level, group in df.groupby("level"):
        dates = group["date"].nunique()
        satellites = set(group["satellite"])
        missing = sorted(
            {"S2A", "S2B", "S2C"} - satellites
        )

        complete = []
        for product, product_group in group.groupby("product"):
            if set(product_group["band"]) == set(BANDAS):
                complete.append(product_group.iloc[0]["date"])

        print(
            f"\n{level}: {dates} fechas medidas; "
            f"{len(set(complete))} fechas con al menos "
            "un producto medido en las cuatro bandas."
        )

        if dates < 10 or missing:
            print(
                "AVISO: cobertura aún incompleta para "
                "10 fechas y los tres satélites. "
                f"Satélites pendientes: {missing}"
            )


# ---------------------------------------------------------------------------
# Corrección geométrica básica
# ---------------------------------------------------------------------------

def export_corrected_raster(record, destination):
    source = Path(record["file"])
    destination.parent.mkdir(
        parents=True, exist_ok=True
    )

    with rasterio.open(source) as src:
        profile = src.profile.copy()

        profile.update(
            driver="GTiff",
            transform=(
                Affine.translation(
                    -float(record["bias_x_m"]),
                    -float(record["bias_y_m"]),
                )
                * src.transform
            ),
        )

        data = src.read()
        valid = src.dataset_mask()
        tags = src.tags()
        tags.update({
            "geometry_correction": "translation",
            "applied_shift_x_m": str(
                -float(record["bias_x_m"])
            ),
            "applied_shift_y_m": str(
                -float(record["bias_y_m"])
            ),
        })

        with rasterio.Env(
            GDAL_TIFF_INTERNAL_MASK=True
        ):
            with rasterio.open(
                destination, "w", **profile
            ) as dst:
                dst.write(data)
                dst.write_mask(valid)
                dst.update_tags(**tags)


def run_correction(
    df,
    calibration_dates,
    output,
    export_corrected,
):
    selected_dates = set(calibration_dates)

    available_dates = set(df["date"])
    unknown = selected_dates - available_dates

    if unknown:
        raise ValueError(
            "Fechas de ajuste sin mediciones: "
            f"{sorted(unknown)}"
        )

    training = df[
        df["date"].isin(selected_dates)
    ].copy()

    evaluation = df[
        ~df["date"].isin(selected_dates)
    ].copy()

    if training.empty or evaluation.empty:
        raise ValueError(
            "La corrección necesita fechas de ajuste "
            "y otras fechas distintas para evaluación."
        )

    model = (
        training.groupby(GROUP_MODEL)
        .agg(
            bias_x_m=("dx_m", "mean"),
            bias_y_m=("dy_m", "mean"),
            n_training=("dx_m", "size"),
            n_training_dates=("date", "nunique"),
        )
        .reset_index()
    )

    save_csv(
        model,
        output / "modelo_traslacion.csv",
    )

    corrected = evaluation.merge(
        model,
        on=GROUP_MODEL,
        how="left",
        validate="many_to_one",
    )

    unavailable = corrected[
        corrected["bias_x_m"].isna()
    ].copy()

    save_csv(
        unavailable,
        output / "evaluacion_sin_modelo.csv",
    )

    corrected = corrected[
        corrected["bias_x_m"].notna()
    ].copy()

    if corrected.empty:
        raise ValueError(
            "Ninguna medición de evaluación tiene un modelo "
            "del mismo nivel, satélite, banda y CRS. "
            "Revisa las fechas de ajuste."
        )

    corrected["corrected_x"] = (
        corrected["observed_x"]
        - corrected["bias_x_m"]
    )
    corrected["corrected_y"] = (
        corrected["observed_y"]
        - corrected["bias_y_m"]
    )

    corrected["corrected_dx_m"] = (
        corrected["dx_m"]
        - corrected["bias_x_m"]
    )
    corrected["corrected_dy_m"] = (
        corrected["dy_m"]
        - corrected["bias_y_m"]
    )

    corrected["corrected_error_m"] = np.hypot(
        corrected["corrected_dx_m"],
        corrected["corrected_dy_m"],
    )

    save_csv(
        corrected,
        output / "evaluacion_corregida.csv",
    )

    comparison = []

    for level, group in corrected.groupby("level"):
        before = statistics(group)
        after = statistics(
            group,
            "corrected_dx_m",
            "corrected_dy_m",
        )

        comparison.append({
            "level": level,
            "n_evaluation": len(group),
            "n_dates": group["date"].nunique(),
            "rmse_before_m": before["rmse_2d_m"],
            "rmse_after_m": after["rmse_2d_m"],
        })

    save_csv(
        pd.DataFrame(comparison),
        output / "comparacion_antes_despues.csv",
    )

    if export_corrected:
        for record in corrected.to_dict("records"):
            destination = (
                output
                / "recortes_corregidos"
                / record["product"]
                / f"{record['band']}.tif"
            )

            export_corrected_raster(
                record, destination
            )

    print(
        "\nCorrección evaluada únicamente en fechas "
        "no utilizadas para ajustar el modelo."
    )
    print(
        pd.DataFrame(comparison).to_string(index=False)
    )

    if not unavailable.empty:
        print(
            f"AVISO: {len(unavailable)} mediciones "
            "sin modelo compatible; consultar "
            "evaluacion_sin_modelo.csv."
        )


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Medición del objetivo y calibración geométrica."
        )
    )

    parser.add_argument(
        "--entrada",
        default=ENTRADA_DEFAULT,
    )
    parser.add_argument(
        "--salida",
        default=SALIDA_DEFAULT,
    )
    parser.add_argument(
        "--niveles",
        nargs="+",
        choices=["L1C", "L2A"],
        default=["L1C", "L2A"],
    )
    parser.add_argument(
        "--latitud",
        type=float,
        help="Sustituir la referencia de los metadatos.",
    )
    parser.add_argument(
        "--longitud",
        type=float,
        help="Sustituir la referencia de los metadatos.",
    )
    parser.add_argument(
        "--solo-resumen",
        action="store_true",
    )
    parser.add_argument(
        "--repetir",
        action="store_true",
        help="Volver a medir las bandas ya registradas.",
    )
    parser.add_argument(
        "--calibration-dates",
        nargs="+",
        help="Fechas YYYY-MM-DD para ajustar la traslación.",
    )
    parser.add_argument(
        "--export-corrected",
        action="store_true",
        help="Exportar recortes de evaluación corregidos.",
    )

    args = parser.parse_args()

    if args.export_corrected and not args.calibration_dates:
        parser.error(
            "--export-corrected requiere --calibration-dates."
        )

    if args.calibration_dates:
        for value in args.calibration_dates:
            try:
                parsed = pd.to_datetime(
                    value, format="%Y-%m-%d"
                )
                if parsed.strftime("%Y-%m-%d") != value:
                    raise ValueError
            except ValueError:
                parser.error(
                    f"Fecha no válida: {value}. Usa YYYY-MM-DD."
                )

    entrada = Path(args.entrada)
    output = Path(args.salida)

    if not entrada.is_dir():
        parser.error(
            f"No existe la carpeta de entrada: {entrada}"
        )

    output.mkdir(
        parents=True, exist_ok=True
    )

    jobs, problems = discover_jobs(
        entrada,
        args.latitud,
        args.longitud,
        args.niveles,
    )

    if not jobs:
        parser.error(
            "No se encontraron bandas con metadatos compatibles."
        )

    measurements_path = (
        output / "mediciones.csv"
    )

    if measurements_path.exists():
        stored = pd.read_csv(
            measurements_path,
            dtype={
                "file": str,
                "signature": str,
                "date": str,
            },
        )

        if not set(COLUMNAS).issubset(stored.columns):
            parser.error(
                "mediciones.csv tiene otro formato. "
                "Utiliza otra carpeta de salida."
            )

        records = stored[COLUMNAS].to_dict("records")
    else:
        records = []

    def key(record):
        return (
            record["file"],
            record["signature"],
            float(record["reference_lon"]),
            float(record["reference_lat"]),
        )

    known_keys = {
        key(record) for record in records
    }

    if not args.solo_resumen:
        try:
            for index, job in enumerate(jobs, start=1):
                print(
                    f"\nBanda {index}/{len(jobs)}"
                )

                if key(job) in known_keys and not args.repetir:
                    print(
                        "Ya medida: se conserva el registro."
                    )
                    continue

                try:
                    status, record = measure_job(job)
                except Exception as error:
                    problems.append({
                        "file": job["file"],
                        "error": str(error),
                    })
                    print(f"ERROR: {error}")
                    continue

                if status == "quit":
                    break

                if status == "accepted":
                    records = [
                        old
                        for old in records
                        if key(old) != key(record)
                    ]

                    records.append(record)
                    known_keys.add(key(record))

                    save_csv(
                        pd.DataFrame(
                            records,
                            columns=COLUMNAS,
                        ),
                        measurements_path,
                    )

        except KeyboardInterrupt:
            print(
                "\nSesión interrumpida. "
                "Las mediciones aceptadas ya están guardadas."
            )

    save_csv(
        pd.DataFrame(
            problems,
            columns=["file", "error"],
        ),
        output / "incidencias.csv",
    )

    current_keys = {
        key(job) for job in jobs
    }

    active_records = [
        record
        for record in records
        if key(record) in current_keys
    ]

    if not active_records:
        print(
            "\nTodavía no hay mediciones válidas "
            "para las imágenes actuales."
        )
        return

    active = pd.DataFrame(
        active_records,
        columns=COLUMNAS,
    )

    numeric_columns = [
        "dx_m",
        "dy_m",
        "observed_x",
        "observed_y",
        "reference_x",
        "reference_y",
    ]

    for column in numeric_columns:
        active[column] = pd.to_numeric(
            active[column], errors="raise"
        )

    if not np.isfinite(
        active[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Las mediciones guardadas contienen valores no válidos."
        )

    save_csv(
        active,
        output / "mediciones_activas.csv",
    )

    write_summaries(
        active, output
    )

    if args.calibration_dates:
        run_correction(
            active,
            args.calibration_dates,
            output,
            args.export_corrected,
        )

    print(
        f"\nResultados guardados en: {output.resolve()}"
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(
            f"Error: {error}"
        )
