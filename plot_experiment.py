"""Generate thesis figures from a KrakenScope experiment CSV.

The GUI exports one CSV row per synchronized localization sample. This script
turns that file into SVG plots that compare simulator ground truth with the
algorithm output.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import statistics
from collections import defaultdict
from pathlib import Path


WIDTH = 980
HEIGHT = 640
PADDING = 74
COLORS = {
    "truth": "#22c55e",
    "tracked": "#f59e0b",
    "raw": "#ef4444",
    "grid": "#d7dde8",
    "axis": "#26364f",
    "text": "#172033",
    "muted": "#64748b",
    "background": "#ffffff",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Ground Truth vs KrakenScope algorithm figures."
    )
    parser.add_argument("csv_path", type=Path, help="CSV exported from KrakenScope")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output folder. Defaults to <csv name>_plots next to the CSV.",
    )
    args = parser.parse_args()

    rows = read_rows(args.csv_path)
    if not rows:
        raise SystemExit("No rows found in the CSV.")

    output_dir = args.out or args.csv_path.with_suffix("").with_name(
        f"{args.csv_path.stem}_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = defaultdict(list)
    for row in rows:
        targets[row["target"]].append(row)

    summary_lines = [
        "# KrakenScope experiment summary",
        "",
        f"Source CSV: `{args.csv_path.name}`",
        "",
        "| Target | Samples | Mode | Mean error | RMSE | Median | P95 | Max |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]

    for target_name, target_rows in sorted(targets.items()):
        target_rows.sort(key=lambda item: ffloat(item["elapsed_s"]))
        safe_name = slug(target_name)
        trajectory_svg = render_trajectory(target_name, target_rows)
        error_svg = render_error_plot(target_name, target_rows)
        (output_dir / f"{safe_name}_trajectory.svg").write_text(
            trajectory_svg,
            encoding="utf-8",
        )
        (output_dir / f"{safe_name}_error.svg").write_text(
            error_svg,
            encoding="utf-8",
        )

        tracked_errors = [
            ffloat(row["tracked_error_m"])
            for row in target_rows
            if row.get("tracked_error_m") not in {"", None}
        ]
        mode = most_common(row.get("mode", "—") for row in target_rows)
        summary_lines.append(
            "| {target} | {samples} | {mode} | {mean:.1f} m | {rmse:.1f} m | "
            "{median:.1f} m | {p95:.1f} m | {maximum:.1f} m |".format(
                target=target_name,
                samples=len(tracked_errors),
                mode=mode,
                mean=mean(tracked_errors),
                rmse=rmse(tracked_errors),
                median=statistics.median(tracked_errors) if tracked_errors else 0.0,
                p95=percentile(tracked_errors, 0.95),
                maximum=max(tracked_errors) if tracked_errors else 0.0,
            )
        )

    (output_dir / "summary.md").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote plots and summary to: {output_dir}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def render_trajectory(target_name: str, rows: list[dict[str, str]]) -> str:
    truth_points = [
        (ffloat(row["truth_latitude"]), ffloat(row["truth_longitude"]))
        for row in rows
    ]
    tracked_points = [
        (ffloat(row["tracked_latitude"]), ffloat(row["tracked_longitude"]))
        for row in rows
    ]
    raw_points = [
        (ffloat(row["raw_latitude"]), ffloat(row["raw_longitude"]))
        for row in rows
        if row.get("raw_latitude") and row.get("raw_longitude")
    ]

    reference = truth_points[0]
    truth_xy = [latlon_to_xy(*point, *reference) for point in truth_points]
    tracked_xy = [latlon_to_xy(*point, *reference) for point in tracked_points]
    raw_xy = [latlon_to_xy(*point, *reference) for point in raw_points]
    all_xy = truth_xy + tracked_xy + raw_xy
    bounds = padded_bounds(all_xy)

    def project(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        min_x, max_x, min_y, max_y = bounds
        sx = PADDING + (x - min_x) / (max_x - min_x) * (WIDTH - 2 * PADDING)
        sy = HEIGHT - PADDING - (y - min_y) / (max_y - min_y) * (HEIGHT - 2 * PADDING)
        return sx, sy

    elements = svg_header(f"{target_name}: Ground Truth vs Algorithm")
    elements += axes("East offset (m)", "North offset (m)")
    elements += grid_labels(bounds, x_label="m", y_label="m")
    elements += polyline(truth_xy, project, COLORS["truth"], 4, "Ground truth")
    elements += polyline(tracked_xy, project, COLORS["tracked"], 4, "Tracked estimate")
    if raw_xy:
        elements += polyline(raw_xy, project, COLORS["raw"], 2.5, "Raw estimate", dash=True)
    elements += legend(
        [
            ("Ground truth", COLORS["truth"], False),
            ("Tracked estimate", COLORS["tracked"], False),
            ("Raw estimate", COLORS["raw"], True),
        ]
    )
    elements += "</svg>\n"
    return "".join(elements)


def render_error_plot(target_name: str, rows: list[dict[str, str]]) -> str:
    tracked = [
        (ffloat(row["elapsed_s"]), ffloat(row["tracked_error_m"]))
        for row in rows
        if row.get("tracked_error_m")
    ]
    raw = [
        (ffloat(row["elapsed_s"]), ffloat(row["raw_error_m"]))
        for row in rows
        if row.get("raw_error_m")
    ]
    all_points = tracked + raw
    min_x = 0.0
    max_x = max((point[0] for point in all_points), default=1.0)
    max_y = max((point[1] for point in all_points), default=1.0)
    max_x = max(1.0, max_x)
    max_y = max(10.0, max_y * 1.12)

    def project(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        sx = PADDING + (x - min_x) / (max_x - min_x) * (WIDTH - 2 * PADDING)
        sy = HEIGHT - PADDING - y / max_y * (HEIGHT - 2 * PADDING)
        return sx, sy

    elements = svg_header(f"{target_name}: Localization error over time")
    elements += axes("Time (s)", "Error (m)")
    elements += numeric_grid(min_x, max_x, 0.0, max_y)
    elements += polyline(tracked, project, COLORS["tracked"], 4, "Tracked error")
    if raw:
        elements += polyline(raw, project, COLORS["raw"], 2.5, "Raw error", dash=True)
    elements += legend(
        [
            ("Tracked error", COLORS["tracked"], False),
            ("Raw error", COLORS["raw"], True),
        ]
    )
    elements += "</svg>\n"
    return "".join(elements)


def svg_header(title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">\n',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{COLORS["background"]}"/>\n',
        f'<text x="{PADDING}" y="42" font-family="Arial, sans-serif" '
        f'font-size="26" font-weight="700" fill="{COLORS["text"]}">'
        f"{html.escape(title)}</text>\n",
    ]


def axes(x_label: str, y_label: str) -> list[str]:
    left = PADDING
    right = WIDTH - PADDING
    top = PADDING
    bottom = HEIGHT - PADDING
    return [
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" '
        f'stroke="{COLORS["axis"]}" stroke-width="2"/>\n',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" '
        f'stroke="{COLORS["axis"]}" stroke-width="2"/>\n',
        f'<text x="{WIDTH / 2}" y="{HEIGHT - 18}" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="16" fill="{COLORS["muted"]}">'
        f"{html.escape(x_label)}</text>\n",
        f'<text x="22" y="{HEIGHT / 2}" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="16" fill="{COLORS["muted"]}" '
        f'transform="rotate(-90 22 {HEIGHT / 2})">{html.escape(y_label)}</text>\n',
    ]


def numeric_grid(min_x: float, max_x: float, min_y: float, max_y: float) -> list[str]:
    elements = []
    for index in range(6):
        ratio = index / 5
        x = PADDING + ratio * (WIDTH - 2 * PADDING)
        y = HEIGHT - PADDING - ratio * (HEIGHT - 2 * PADDING)
        x_value = min_x + ratio * (max_x - min_x)
        y_value = min_y + ratio * (max_y - min_y)
        elements.append(
            f'<line x1="{x:.1f}" y1="{PADDING}" x2="{x:.1f}" y2="{HEIGHT - PADDING}" '
            f'stroke="{COLORS["grid"]}" stroke-width="1"/>\n'
        )
        elements.append(
            f'<line x1="{PADDING}" y1="{y:.1f}" x2="{WIDTH - PADDING}" y2="{y:.1f}" '
            f'stroke="{COLORS["grid"]}" stroke-width="1"/>\n'
        )
        elements.append(axis_text(x, HEIGHT - PADDING + 23, f"{x_value:.0f}", "middle"))
        elements.append(axis_text(PADDING - 12, y + 5, f"{y_value:.0f}", "end"))
    return elements


def grid_labels(
    bounds: tuple[float, float, float, float],
    x_label: str,
    y_label: str,
) -> list[str]:
    min_x, max_x, min_y, max_y = bounds
    return numeric_grid(min_x, max_x, min_y, max_y)


def polyline(
    points: list[tuple[float, float]],
    project,
    color: str,
    width: float,
    label: str,
    dash: bool = False,
) -> list[str]:
    if not points:
        return []
    path_points = " ".join(
        f"{x:.1f},{y:.1f}" for x, y in (project(point) for point in points)
    )
    dash_attr = ' stroke-dasharray="9 8"' if dash else ""
    first_x, first_y = project(points[0])
    last_x, last_y = project(points[-1])
    return [
        f'<polyline points="{path_points}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"'
        f'{dash_attr}/>\n',
        f'<circle cx="{first_x:.1f}" cy="{first_y:.1f}" r="5" fill="{color}" opacity="0.65"/>\n',
        f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="7" fill="{color}"/>\n',
        f'<title>{html.escape(label)}</title>\n',
    ]


def legend(items: list[tuple[str, str, bool]]) -> list[str]:
    elements = []
    x = WIDTH - 330
    y = 35
    elements.append(
        f'<rect x="{x - 15}" y="{y - 22}" width="300" height="{28 * len(items) + 16}" '
        'rx="12" fill="#f8fafc" stroke="#dbe3ef"/>\n'
    )
    for index, (label, color, dash) in enumerate(items):
        item_y = y + index * 28
        dash_attr = ' stroke-dasharray="8 7"' if dash else ""
        elements.append(
            f'<line x1="{x}" y1="{item_y}" x2="{x + 38}" y2="{item_y}" '
            f'stroke="{color}" stroke-width="4"{dash_attr}/>\n'
        )
        elements.append(
            f'<text x="{x + 50}" y="{item_y + 5}" font-family="Arial, sans-serif" '
            f'font-size="15" fill="{COLORS["text"]}">{html.escape(label)}</text>\n'
        )
    return elements


def axis_text(x: float, y: float, text: str, anchor: str) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-size="13" fill="{COLORS["muted"]}">'
        f"{html.escape(text)}</text>\n"
    )


def latlon_to_xy(
    latitude: float,
    longitude: float,
    reference_latitude: float,
    reference_longitude: float,
) -> tuple[float, float]:
    earth_radius = 6_371_000.0
    x = math.radians(longitude - reference_longitude) * earth_radius * math.cos(
        math.radians(reference_latitude)
    )
    y = math.radians(latitude - reference_latitude) * earth_radius
    return x, y


def padded_bounds(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)
    pad_x = span_x * 0.12
    pad_y = span_y * 0.12
    return min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y


def ffloat(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def rmse(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else 0.0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def most_common(values) -> str:
    counts = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    return max(counts.items(), key=lambda item: item[1])[0] if counts else "—"


def slug(value: str) -> str:
    return "".join(
        char.lower() if char.isalnum() else "_"
        for char in value.strip()
    ).strip("_") or "target"


if __name__ == "__main__":
    main()
