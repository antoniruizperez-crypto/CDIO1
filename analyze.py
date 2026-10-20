"""
Gráficas de los resultados de calculate_erosion.py (se guardan como PNG en
projects/<p>/output/estimated_erosion_accretion/ y se muestran si no se usa --no-plot):

  1. distance_by_transect.png : distancia por transecto y fecha (qué tramos cambian más).
                                Los huecos sin medida se dejan en blanco, no se unen.
  2. distance_over_time.png   : media ± desviación de todos los transectos por fecha, y cobertura.

Signo: positivo = mar adentro (acreción), negativo = tierra adentro (erosión).
"""

import argparse
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import shoreline_utils as utils


def main(project, show_plot=True):
    out_dir = utils.project_path(project) / "output" / "estimated_erosion_accretion"
    df = pd.read_csv(out_dir / "shoreline_distances.csv", parse_dates=["date"])
    if df["distance_m"].notna().sum() == 0:
        raise SystemExit("No hay distancias válidas que representar.")

    colors = plt.cm.tab10.colors
    markers = itertools.cycle(("o", "s", "^", "D", "v", "<", ">", "p", "*", "h"))

    fig1, ax = plt.subplots(figsize=(12, 6))
    for i, (date, g) in enumerate(df.groupby("date")):
        mk = next(markers)
        for k, (_, seg) in enumerate(g.groupby("ref_segment")):
            seg = seg.sort_values("transect_id")
            # los NaN se conservan: matplotlib deja un hueco en vez de unir tramos sin medida
            ax.plot(seg["transect_id"], seg["distance_m"], marker=mk, ms=3, lw=1,
                    color=colors[i % 10], label=date.strftime("%Y-%m-%d") if k == 0 else None)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("ID de transecto (a lo largo de la costa)")
    ax.set_ylabel("Distancia a la referencia (m)  [+ mar adentro, − tierra adentro]")
    ax.set_title("Distancia por transecto y fecha")
    ax.grid(True)
    ax.legend(title="Fecha", fontsize=8)
    fig1.tight_layout()
    fig1.savefig(out_dir / "distance_by_transect.png", dpi=150)

    s = (df.groupby("date")
           .agg(mean=("distance_m", "mean"), std=("distance_m", "std"),
                valid=("distance_m", lambda v: v.notna().sum()), total=("distance_m", "size"))
           .reset_index())
    s["coverage"] = s["valid"] / s["total"]

    fig2, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    a1.errorbar(s["date"], s["mean"], yerr=s["std"], marker="o", capsize=3, color="tab:blue")
    a1.axhline(0, color="k", lw=0.8)
    a1.set_ylabel("Distancia media ± desv. (m)")
    a1.set_title("Evolución de la distancia media a la referencia")
    a1.grid(True)
    a2.bar(s["date"], s["coverage"], width=3, color="tab:gray")
    a2.set_ylim(0, 1.05)
    a2.set_ylabel("Cobertura")
    a2.set_xlabel("Fecha")
    a2.grid(True)
    fig2.autofmt_xdate()
    fig2.tight_layout()
    fig2.savefig(out_dir / "distance_over_time.png", dpi=150)

    if len(s) >= 3:
        t = (s["date"] - s["date"].min()).dt.days.to_numpy() / 365.25
        slope, _ = np.polyfit(t, s["mean"].to_numpy(), 1)
        print(f"Tendencia lineal de la distancia media: {slope:+.2f} m/año con {len(s)} fechas.")
        if len(s) < 6:
            print("  Aviso: serie muy corta; no es una tasa fiable ni permite hablar de tendencias estacionales o climáticas.")

    if show_plot:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grafica los resultados de erosión/acreción.")
    parser.add_argument("-p", "--project", default=config.NOMBRE_PROYECTO)
    parser.add_argument("--no-plot", action="store_true", help="Guardar los PNG sin abrir ventanas.")
    a = parser.parse_args()
    main(a.project, show_plot=not a.no_plot)
