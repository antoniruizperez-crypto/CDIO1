"""
Validación: compara la costa estimada de UNA fecha con una referencia externa
(ground truth) y calcula métricas de error.

Signo: positivo = la costa estimada está mar adentro de la referencia.

Métricas: sesgo (media con signo), MAE, RMSE, desviación típica (RMSE sin el
sesgo), mediana, P90 de |d| y cobertura (fracción de transectos con medida).
Los transectos sin cruce NO entran en las métricas; si la cobertura es baja, el
error real puede ser mayor que el calculado (aumenta --half-length).

Salidas (projects/<p>/output/estimated_error/):
  shoreline_distances.csv, error_metrics.csv, error_map.png
"""

import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

import calculate_erosion as ce
import config
import shoreline_utils as utils


def main(project, ground_truth, date, show_plot=True, spacing=None, half_length=None, sigma=None):
    spacing = spacing or config.SPATIAL_SPACING
    half_length = half_length or config.TRANSECT_HALF_LENGTH
    sigma = config.SPLINE_SMOOTH_M if sigma is None else sigma

    target = datetime.strptime(date, "%Y-%m-%d").date()
    project_dir = utils.project_path(project)
    edges_dir = project_dir / "output" / "estimated_waterbodies_edges"
    output_dir = project_dir / "output" / "estimated_error"
    output_dir.mkdir(parents=True, exist_ok=True)

    shorelines = utils.read_shorelines(edges_dir / "all_shorelines.csv")
    if date not in shorelines:
        raise ValueError(f"La fecha {date} no está en all_shorelines.csv. Disponibles: {sorted(shorelines)}")
    crs = ce.load_meta(edges_dir)["crs"]

    ref_date, ref_segments = ce.reference_from_ground_truth(
        project_dir, ground_truth, date, shorelines, crs, output_dir
    )
    gap_days = abs((datetime.strptime(ref_date, "%Y-%m-%d").date() - target).days)

    transects = utils.build_transects(ref_segments, spacing, sigma, half_length)
    if not transects:
        raise ValueError("No se pudieron generar transectos con la referencia externa.")

    geom = ce.date_geometry(shorelines[date])
    res = utils.measure_transects(transects, geom)
    df = pd.DataFrame({
        "date": date,
        "transect_id": [t["id"] for t in transects],
        "ref_segment": [t["segment"] for t in transects],
        "reference_x": [t["x"] for t in transects],
        "reference_y": [t["y"] for t in transects],
        "distance_m": [r[0] for r in res],
        "n_crossings": [r[1] for r in res],
    })
    d = df["distance_m"].dropna().to_numpy()
    if d.size == 0:
        raise ValueError("Ningún transecto cruza la costa estimada: revisa fechas, AOI y --half-length.")

    metrics = {
        "date": date, "reference_date": ref_date, "days_between": gap_days,
        "valid_samples": int(d.size), "transects": len(transects),
        "coverage_ratio": d.size / len(transects),
        "bias_m": float(d.mean()),
        "mae_m": float(np.abs(d).mean()),
        "rmse_m": float(np.sqrt((d**2).mean())),
        "std_m": float(d.std(ddof=1)) if d.size > 1 else float("nan"),
        "median_m": float(np.median(d)),
        "p90_abs_m": float(np.percentile(np.abs(d), 90)),
    }
    print(f"Referencia: {ref_date} ({gap_days} días de diferencia con {date})")
    if gap_days > 0:
        print("  Aviso: parte del error puede ser cambio real de la costa entre ambas fechas.")
    for k in ("valid_samples", "coverage_ratio", "bias_m", "mae_m", "rmse_m", "std_m", "median_m", "p90_abs_m"):
        v = metrics[k]
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    if metrics["coverage_ratio"] < 0.9:
        print("  Aviso: cobertura < 90 %. Las métricas ignoran los transectos sin cruce.")

    df.to_csv(output_dir / "shoreline_distances.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "error_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.add_collection(LineCollection([list(t["line"].coords) for t in transects],
                                     colors="green", alpha=0.25, linewidths=0.6))
    for k, s in enumerate(ref_segments):
        ax.plot(s[:, 0], s[:, 1], "b", alpha=0.8, label=f"Referencia externa ({ref_date})" if k == 0 else None)
    for k, s in enumerate(shorelines[date]):
        ax.plot(s[:, 0], s[:, 1], "r--", label=f"Costa estimada ({date})" if k == 0 else None)
    ax.set_title(f"Validación: RMSE = {metrics['rmse_m']:.2f} m, sesgo = {metrics['bias_m']:.2f} m")
    ax.set_xlabel("X (m, proyectado)")
    ax.set_ylabel("Y (m, proyectado)")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale()
    ax.grid(True)
    fig.savefig(output_dir / "error_map.png", dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)
    return df, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Error de la costa estimada frente a una referencia externa.")
    parser.add_argument("-p", "--project", default=config.NOMBRE_PROYECTO)
    parser.add_argument("-g", "--ground-truth", required=True, help="Carpeta con CSV de referencia (x,y en UTM).")
    parser.add_argument("-d", "--date", required=True, help="Fecha a comparar, YYYY-MM-DD.")
    parser.add_argument("--spacing", type=float)
    parser.add_argument("--half-length", type=float)
    parser.add_argument("--no-plot", action="store_true")
    a = parser.parse_args()
    main(a.project, a.ground_truth, a.date, show_plot=not a.no_plot, spacing=a.spacing, half_length=a.half_length)
