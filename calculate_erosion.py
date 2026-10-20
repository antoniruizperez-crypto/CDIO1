"""
Medición de erosión / acreción respecto a una costa de referencia.

1. Se elige la referencia: la primera fecha del dataset, o una costa externa
   (ground truth) con --ground-truth.
2. Cada segmento de referencia se remuestrea con una spline de suavizado y se
   generan transectos perpendiculares cada SPATIAL_SPACING metros.
3. Para cada fecha se mide dónde cruza su costa cada transecto.

Signo: ver shoreline_utils. Positivo = mar adentro (acreción), negativo = tierra
adentro (erosión), siempre respecto a la referencia.

Salidas (projects/<p>/output/estimated_erosion_accretion/):
  shoreline_distances.csv, distances_summary.csv, transects_map.png
"""

import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from shapely.geometry import LineString, MultiLineString

import config
import shoreline_utils as utils


def load_meta(edges_dir):
    import json
    with open(edges_dir / "shorelines_meta.json") as f:
        return json.load(f)


def reference_from_first_date(shorelines, first_date):
    segs = [s for s in shorelines[first_date] if LineString(s).length >= config.REF_MIN_SEGMENT_M]
    if not segs:  # costa muy fragmentada: se usa el segmento más largo
        segs = [max(shorelines[first_date], key=lambda s: LineString(s).length)]
    return first_date, utils.order_segments(segs)


def reference_from_ground_truth(project_dir, ground_truth, first_date, shorelines, crs, output_dir):
    target = datetime.strptime(first_date, "%Y-%m-%d").date()
    ref_date, gt_path = utils.closest_ground_truth_file(ground_truth, target)
    xy = utils.transform_xy(utils.load_xy_csv(gt_path), config.GT_CRS, crs)

    ll = utils.transform_xy(xy, crs, 4326)
    utils.export_lines_to_kmz([list(map(tuple, ll))], output_dir / f"{gt_path.stem}_reference.kmz")

    aoi = utils.load_project_aoi(project_dir, crs)
    parts = utils.flatten_lines(LineString(utils.dedupe_consecutive(xy)).intersection(aoi))
    parts = [p for p in parts if p.length >= config.REF_MIN_SEGMENT_M]
    if not parts:
        raise ValueError("La referencia externa no tiene tramos útiles dentro del AOI.")

    est_lines = [LineString(s) for s in shorelines[first_date]]
    segs = [utils.orient_like(np.asarray(p.coords)[:, :2], est_lines) for p in parts]
    return ref_date.isoformat(), utils.order_segments(segs)


def date_geometry(segments):
    return MultiLineString([s for s in segments if len(s) >= 2])


def plot_map(ref_segments, transects, shorelines, dates, ref_label, out_png, show):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.add_collection(LineCollection([list(t["line"].coords) for t in transects],
                                     colors="green", alpha=0.25, linewidths=0.6))
    for k, s in enumerate(ref_segments):
        ax.plot(s[:, 0], s[:, 1], "k--", alpha=0.6, label=f"Referencia ({ref_label})" if k == 0 else None)
    cmap = plt.cm.tab10.colors
    for i, d in enumerate(dates):
        for k, s in enumerate(shorelines[d]):
            ax.plot(s[:, 0], s[:, 1], "--", color=cmap[i % 10], lw=1, label=d if k == 0 else None)
    ax.set_title("Referencia, transectos (verde) y costas por fecha")
    ax.set_xlabel("X (m, proyectado)")
    ax.set_ylabel("Y (m, proyectado)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale()
    ax.grid(True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main(project, ground_truth=None, show_plot=True, spacing=None, half_length=None, sigma=None):
    spacing = spacing or config.SPATIAL_SPACING
    half_length = half_length or config.TRANSECT_HALF_LENGTH
    sigma = config.SPLINE_SMOOTH_M if sigma is None else sigma

    project_dir = utils.project_path(project)
    edges_dir = project_dir / "output" / "estimated_waterbodies_edges"
    output_dir = project_dir / "output" / "estimated_erosion_accretion"
    output_dir.mkdir(parents=True, exist_ok=True)

    shorelines = utils.read_shorelines(edges_dir / "all_shorelines.csv")
    crs = load_meta(edges_dir)["crs"]
    dates = sorted(shorelines)

    if ground_truth is None:
        ref_label, ref_segments = reference_from_first_date(shorelines, dates[0])
    else:
        ref_label, ref_segments = reference_from_ground_truth(
            project_dir, ground_truth, dates[0], shorelines, crs, output_dir
        )

    transects = utils.build_transects(ref_segments, spacing, sigma, half_length)
    if not transects:
        raise ValueError("No se pudieron generar transectos con la referencia elegida.")
    print(f"Referencia: {ref_label} | segmentos: {len(ref_segments)} | transectos: {len(transects)}")
    print(f"Espaciado medio entre transectos: {utils.mean_transect_spacing(transects):.2f} m")

    rows = []
    for d in dates:
        geom = date_geometry(shorelines[d])
        for t, (dist, ncross) in zip(transects, utils.measure_transects(transects, geom)):
            rows.append({"date": d, "transect_id": t["id"], "ref_segment": t["segment"],
                         "reference_x": t["x"], "reference_y": t["y"],
                         "distance_m": dist, "n_crossings": ncross})
    distance_df = pd.DataFrame(rows)
    total = len(transects)

    summary_df = (
        distance_df.groupby("date")
        .agg(avg_distance_m=("distance_m", "mean"),
             median_distance_m=("distance_m", "median"),
             std_distance_m=("distance_m", "std"),
             max_distance_m=("distance_m", "max"),
             min_distance_m=("distance_m", "min"),
             valid_points=("distance_m", lambda v: int(v.notna().sum())),
             multi_crossing_points=("n_crossings", lambda v: int((v > 1).sum())))
        .reset_index()
    )
    summary_df["coverage_ratio"] = summary_df["valid_points"] / total
    summary_df["range_distance_m"] = summary_df["max_distance_m"] - summary_df["min_distance_m"]

    params_path = edges_dir / "extraction_params.csv"
    if params_path.exists():
        p = pd.read_csv(params_path)[["date", "ndwi_threshold", "cloud_pct_aoi"]]
        p["date"] = p["date"].astype(str)
        summary_df = summary_df.merge(p, on="date", how="left")

    distance_df.to_csv(output_dir / "shoreline_distances.csv", index=False)
    summary_df.to_csv(output_dir / "distances_summary.csv", index=False)

    print(f"Fechas: {len(dates)} ({dates[0]} -> {dates[-1]})")
    print(f"Transectos con al menos una medida válida: "
          f"{distance_df.loc[distance_df['distance_m'].notna(), 'transect_id'].nunique()} de {total}")
    print(summary_df[["date", "avg_distance_m", "std_distance_m", "coverage_ratio"]].to_string(index=False))

    plot_map(ref_segments, transects, shorelines, dates, ref_label,
             output_dir / "transects_map.png", show_plot)
    return distance_df, summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estima erosión y acreción respecto a una referencia.")
    parser.add_argument("-p", "--project", default=config.NOMBRE_PROYECTO)
    parser.add_argument("-g", "--ground-truth", help="Carpeta con CSV de referencia externa (x,y en UTM).")
    parser.add_argument("--spacing", type=float, help="Separación entre transectos (m).")
    parser.add_argument("--half-length", type=float, help="Longitud del transecto a cada lado (m).")
    parser.add_argument("--no-plot", action="store_true", help="No abrir ventanas (el PNG se guarda igual).")
    a = parser.parse_args()
    main(a.project, a.ground_truth, show_plot=not a.no_plot, spacing=a.spacing, half_length=a.half_length)
