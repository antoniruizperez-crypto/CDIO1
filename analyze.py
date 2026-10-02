import argparse
import itertools

import matplotlib.pyplot as plt
import pandas as pd

import shoreline_utils as utils


def main(project):
    csv_path = (
        utils.project_path(project)
        / "output"
        / "estimated_erosion_accretion"
        / "shoreline_distances.csv"
    )
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.dropna(subset=["distance_m"])

    colors = plt.cm.tab10.colors
    markers = itertools.cycle(("o", "s", "^", "D", "v", "<", ">", "p", "*", "h"))

    plt.figure(figsize=(12, 6))
    for i, (date, group) in enumerate(df.groupby("date")):
        plt.plot(
            group["transect_id"],
            group["distance_m"],
            label=date.strftime("%Y-%m-%d"),
            marker=next(markers),
            color=colors[i % len(colors)],
            linestyle="-",
        )

    plt.xlabel("Transect ID")
    plt.ylabel("Distance (m)")
    plt.title("Shoreline Distance by Transect and Date")
    plt.grid(True)
    plt.legend(title="Date")
    plt.tight_layout()
    plt.show()

    avg_df = df.groupby("date")["distance_m"].mean().reset_index()
    plt.figure(figsize=(10, 5))
    plt.plot(avg_df["date"], avg_df["distance_m"], marker="o", linestyle="-", color="tab:blue")
    plt.xlabel("Date")
    plt.ylabel("Average Distance (m)")
    plt.title("Average Shoreline Distance Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot erosion and accretion outputs.")
    parser.add_argument(
        "-p",
        "--project",
        required=True,
        help="Project name under coastline_estimator/projects.",
    )
    args = parser.parse_args()
    main(args.project)