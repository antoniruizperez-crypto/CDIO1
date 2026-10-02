import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
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


def load_reference(project_dir, shorelines, sorted_dates_str, ground_truth):
    if ground_truth is None:
        reference_date = sorted_dates_str[0]
        reference_coords = shorelines[reference_date]
        return reference_date, reference_coords, 1, False

    first_date = datetime.strptime(sorted_dates_str[0], "%Y-%m-%d").date()
    reference_date, gt_csv_path = closest_ground_truth_file(ground_truth, first_date)
    reference_coords = np.loadtxt(gt_csv_path, delimiter="\t", skiprows=1)

    gt_coords = utils.load_coords_csv(gt_csv_path)
    utils.export_coords_to_kmz(gt_coords, gt_csv_path.with_suffix(".kmz"))

    return reference_date.isoformat(), reference_coords, 0, True


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


def load_project_aoi(project_dir):
    geojson_path = project_dir / "input" / "polygon.geojson"
    mask_dir = project_dir / "output" / "estimated_waterbodies_images"
    first_mask = next(mask_dir.glob("*.tif"), None)
    if first_mask is None or not geojson_path.exists():
        return None

    aoi = gpd.read_file(geojson_path).set_crs(4326, allow_override=True)
    with rasterio.open(first_mask) as src:
        return aoi.to_crs(src.crs)


def main(project, ground_truth=None, show_plot=True):
    project_dir = utils.project_path(project)
    output_dir = project_dir / "output" / "estimated_erosion_accretion"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = project_dir / "output" / "estimated_waterbodies_edges" / "all_shorelines.csv"
    df = pd.read_csv(csv_path)

    shorelines = {date: group[["x", "y"]].values for date, group in df.groupby("date")}
    sorted_dates_str = sorted(shorelines.keys())

    reference_date, reference_coords, start_index, ordered = load_reference(
        project_dir, shorelines, sorted_dates_str, ground_truth
    )

    x = reference_coords[:, 0]
    y = reference_coords[:, 1]
    xy_unique = np.unique(np.column_stack((x, y)), axis=0)
    x_unique = xy_unique[:, 0]
    y_unique = xy_unique[:, 1]

    reference_date_underscores = str(reference_date).replace("-", "_")
    line_path = output_dir / f"ref_{reference_date_underscores}.pkl"
    if line_path.exists() and line_path.stat().st_size > 0:
        with open(line_path, "rb") as f:
            reference_line = pickle.load(f)
    else:
        reference_line = utils.build_ordered_linestring(np.column_stack((x, y)), ordered)
        if ground_truth is not None:
            aoi = load_project_aoi(project_dir)
            if aoi is not None:
                reference_line = reference_line.intersection(aoi.unary_union)
        with open(line_path, "wb") as f:
            pickle.dump(reference_line, f)

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

    distance_results = []
    colors = ["r", "m", "c", "y"]
    dist_reference = 0

    for idx, date in enumerate(sorted_dates_str[start_index:]):
        coords = shorelines[date]
        if len(coords) < 2:
            print(f"Skipping shoreline for {date} due to insufficient points.")
            continue

        print(f"Ordering shoreline for {date}...")
        date_underscores = date.replace("-", "_")
        line_cache_path = output_dir / f"line_{date_underscores}.pkl"
        if line_cache_path.exists() and line_cache_path.stat().st_size > 0:
            with open(line_cache_path, "rb") as f:
                line = pickle.load(f)
        else:
            line = utils.build_ordered_linestring(coords, False)
            with open(line_cache_path, "wb") as f:
                pickle.dump(line, f)
        print("Done")

        for i in range(len(unew)):
            px, py = x_smooth[i], y_smooth[i]
            if i > 0:
                dist_reference += np.sqrt((px - x_smooth[i - 1]) ** 2 + (py - y_smooth[i - 1]) ** 2)

            x_line, y_line = line.xy
            ax.plot(
                x_line,
                y_line,
                "--",
                label=date if i == 0 else None,
                color=colors[idx % len(colors)],
            )

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

    if distance_results:
        print(f"Average distance between transects: {dist_reference / max(1, len(unew) - 1):.2f} meters")

    distance_df = pd.DataFrame(distance_results)
    total_transects = distance_df["transect_id"].nunique()

    print(f"Number of transects computed: {len(unew)}")
    print(
        "Number of valid transects with at least one valid distance: "
        f"{distance_df.loc[distance_df['distance_m'].notna(), 'transect_id'].nunique()}"
    )
    print(f"First date in dataset: {distance_df['date'].min()}")
    print(f"Last date in dataset: {distance_df['date'].max()}")
    print(f"Number of distinct dates in dataset: {distance_df['date'].nunique()}")

    summary_df = (
        distance_df.groupby("date")
        .agg(
            avg_distance_m=("distance_m", "mean"),
            median_distance_m=("distance_m", "median"),
            std_distance_m=("distance_m", "std"),
            max_distance_m=("distance_m", "max"),
            min_distance_m=("distance_m", "min"),
            valid_points=("distance_m", lambda values: values.notna().sum()),
        )
        .reset_index()
    )

    summary_df["coverage_ratio"] = summary_df["valid_points"] / total_transects
    summary_df["range_distance_m"] = summary_df["max_distance_m"] - summary_df["min_distance_m"]

    summary_df.to_csv(output_dir / "distances_summary.csv", index=False)
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
    parser = argparse.ArgumentParser(description="Estimate erosion and accretion from shoreline data.")
    parser.add_argument(
        "-p",
        "--project",
        required=True,
        help="Project name under coastline_estimator/projects.",
    )
    parser.add_argument(
        "-g",
        "--ground-truth",
        help="Path to a folder containing ground truth CSV files to use as a reference.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Write CSV outputs without opening plots.")
    args = parser.parse_args()
    main(args.project, args.ground_truth, show_plot=not args.no_plot)