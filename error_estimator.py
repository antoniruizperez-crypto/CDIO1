import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import splev, splprep
from shapely.geometry import LineString, MultiPoint, Point

import shoreline_utils as utils


SPATIAL_SPACING = 10
TRANSECT_LENGTH = 100


def closest_ground_truth_file(ground_truth_dir, target_date):
    base = Path(ground_truth_dir).expanduser().resolve()
    if not base.is_dir():
        print(f"Error: {base} is not a directory", file=sys.stderr)
        sys.exit(1)

    dated_files = []
    for csv_file in base.glob("*.csv"):
        date = utils.extract_date(csv_file.name)
        if date is not None:
            dated_files.append((date, csv_file))

    if not dated_files:
        raise FileNotFoundError(f"No dated CSV files found in {base}")

    return min(dated_files, key=lambda row: abs(row[0] - target_date))


def signed_intersection_distance(intersection, px, py, normal):
    if isinstance(intersection, MultiPoint):
        distances = [Point(px, py).distance(point) for point in intersection.geoms]
        return distances[int(np.argmin(distances))]

    if isinstance(intersection, Point):
        signed_vector = np.array([intersection.x - px, intersection.y - py])
        distance = np.linalg.norm(signed_vector)
        if np.dot(signed_vector, normal) < 0:
            distance = -distance
        return distance

    return np.nan


def main(project, ground_truth, date, show_plot=True):
    target_date = datetime.strptime(date, "%Y-%m-%d").date()
    project_dir = utils.project_path(project)
    output_dir = project_dir / "output" / "estimated_error"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = project_dir / "output" / "estimated_waterbodies_edges" / "all_shorelines.csv"
    df = pd.read_csv(csv_path)
    shorelines = {date: group[["x", "y"]].values for date, group in df.groupby("date")}

    if date not in shorelines:
        raise ValueError(f"Date {date} is not present in {csv_path}")

    reference_date, gt_csv_path = closest_ground_truth_file(ground_truth, target_date)
    reference_coords = np.loadtxt(gt_csv_path, delimiter="\t", skiprows=1)
    gt_coords = utils.load_coords_csv(gt_csv_path)
    utils.export_coords_to_kmz(gt_coords, gt_csv_path.with_suffix(".kmz"))

    x = reference_coords[:, 0]
    y = reference_coords[:, 1]
    xy_unique = np.unique(np.column_stack((x, y)), axis=0)
    x_unique = xy_unique[:, 0]
    y_unique = xy_unique[:, 1]

    reference_line = utils.build_ordered_linestring(np.column_stack((x, y)), ordered=True)
    shoreline_length = reference_line.length
    n_points = int(shoreline_length / SPATIAL_SPACING)

    tck, _ = splprep([x_unique, y_unique], s=0)
    unew = np.linspace(0, 1, num=n_points)
    x_smooth, y_smooth = splev(unew, tck)
    dx, dy = splev(unew, tck, der=1)

    fig, ax = plt.subplots(figsize=(12, 8))
    x_ref, y_ref = reference_line.xy
    ax.plot(x_ref, y_ref, "k--", label=f"Reference shoreline ({reference_date})", alpha=0.5)
    ax.plot(x_smooth, y_smooth, "b", label=f"Smoothed shoreline ({reference_date})", alpha=0.7)

    for i in range(len(unew)):
        px, py = x_smooth[i], y_smooth[i]
        tx, ty = dx[i], dy[i]
        normal = np.array([-ty, tx])
        normal /= np.linalg.norm(normal)
        start = (px - normal[0] * TRANSECT_LENGTH / 2, py - normal[1] * TRANSECT_LENGTH / 2)
        end = (px + normal[0] * TRANSECT_LENGTH / 2, py + normal[1] * TRANSECT_LENGTH / 2)
        ax.plot([start[0], end[0]], [start[1], end[1]], "g-", alpha=0.3)

    print(f"Ordering shoreline for {date}...")
    line = utils.build_ordered_linestring(shorelines[date], False)
    print("Done")

    distance_results = []
    x_line, y_line = line.xy
    ax.plot(x_line, y_line, "k--", label=date)

    for i in range(len(unew)):
        px, py = x_smooth[i], y_smooth[i]
        tx, ty = dx[i], dy[i]
        normal = np.array([-ty, tx])
        normal /= np.linalg.norm(normal)
        start = (px - normal[0] * TRANSECT_LENGTH / 2, py - normal[1] * TRANSECT_LENGTH / 2)
        end = (px + normal[0] * TRANSECT_LENGTH / 2, py + normal[1] * TRANSECT_LENGTH / 2)
        transect = LineString([start, end])

        distance = signed_intersection_distance(transect.intersection(line), px, py, normal)
        distance_results.append(
            {
                "date": date,
                "transect_id": i,
                "reference_x": px,
                "reference_y": py,
                "distance_m": distance,
            }
        )

    distance_df = pd.DataFrame(distance_results)
    rmse = np.sqrt((distance_df["distance_m"].dropna() ** 2).mean())
    print(f"RMSE: {rmse}")
    print(f"Valid samples: {distance_df['distance_m'].notna().sum()}")
    print(f"Reference date: {reference_date}")
    distance_df.to_csv(output_dir / "shoreline_distances.csv", index=False)

    ax.set_title("Shoreline Geometry and Transect Intersections")
    ax.set_xlabel("X (meters, projected)")
    ax.set_ylabel("Y (meters, projected)")
    ax.legend()
    ax.axis("equal")
    plt.grid(True)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate shoreline error against ground truth data.")
    parser.add_argument(
        "-p",
        "--project",
        required=True,
        help="Project name under coastline_estimator/projects.",
    )
    parser.add_argument(
        "-g",
        "--ground-truth",
        required=True,
        help="Path to a folder containing ground truth CSV files.",
    )
    parser.add_argument("-d", "--date", required=True, help="Date to compare, formatted as YYYY-MM-DD.")
    parser.add_argument("--no-plot", action="store_true", help="Write CSV outputs without opening plots.")
    args = parser.parse_args()
    main(args.project, args.ground_truth, args.date, show_plot=not args.no_plot)