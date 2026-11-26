"""Run reproducible KrakenScope simulation experiments without real-time waiting.

The runner intentionally drives the same simulator channel model and the same
KrakenScope GUI localization pipeline used interactively.  It only replaces
UDP transport and wall-clock sleeping with direct in-process packet delivery.
Ground truth is included in packets solely for the existing post-localization
evaluation path.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


# Both Qt applications are created but never displayed.  This must be set
# before importing PySide6 or either GUI module.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")

ROOT = Path(__file__).resolve().parent
SIMULATOR_ROOT = ROOT.parent / "KrakenSDR_Simulator-main"
if str(SIMULATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_ROOT))

from PySide6.QtWidgets import QApplication  # noqa: E402

from main import EXPERIMENT_FIELDS, MainWindow as KrakenScopeWindow  # noqa: E402
from plot_experiment import (  # noqa: E402
    percentile,
    render_error_plot,
    render_trajectory,
)
from simulator_qt import (  # noqa: E402
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    MainWindow as SimulatorWindow,
    RECEIVER_BIAS_STD_DEGREES,
    destination_point,
)


SUMMARY_FIELDS = [
    "scenario",
    "title",
    "category",
    "repeats",
    "samples",
    "duration_s",
    "interval_s",
    "doa_noise_deg",
    "rssi_noise_db",
    "speed_mps",
    "tracked_mean_m",
    "tracked_rmse_m",
    "tracked_rmse_run_mean_m",
    "tracked_rmse_run_std_m",
    "tracked_rmse_95ci_m",
    "steady_tracked_rmse_m",
    "tracked_median_m",
    "tracked_p95_m",
    "tracked_max_m",
    "raw_mean_m",
    "raw_rmse_m",
    "steady_raw_rmse_m",
    "rmse_improvement_percent",
    "auto_mode_accuracy_percent",
    "mean_start_detection_delay_s",
    "mean_stop_detection_delay_s",
    "mean_mode_switches",
    "total_false_mode_switches",
    "total_missed_mode_transitions",
    "mean_geometry_score",
]

RUN_SUMMARY_FIELDS = [
    "scenario",
    "repeat",
    "seed",
    "samples",
    "tracked_mean_m",
    "tracked_rmse_m",
    "steady_tracked_rmse_m",
    "tracked_median_m",
    "tracked_p95_m",
    "tracked_max_m",
    "raw_mean_m",
    "raw_rmse_m",
    "steady_raw_rmse_m",
    "rmse_improvement_percent",
    "auto_mode_accuracy_percent",
    "start_detection_delay_s",
    "stop_detection_delay_s",
    "mode_switches",
    "false_mode_switches",
    "missed_mode_transitions",
    "mean_geometry_score",
]


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    category: str
    motion: str
    duration_s: float
    interval_s: float
    doa_noise_deg: float
    rssi_noise_db: float
    baseline_m: float = 300.0
    target_north_m: float = 220.0
    speed_mps: float = 0.0
    heading_degrees: float = 90.0
    radius_m: float = 120.0
    description: str = ""


SCENARIOS = (
    Scenario(
        key="static_good",
        title="Static target — good geometry",
        category="static",
        motion="static",
        duration_s=90.0,
        interval_s=0.20,
        doa_noise_deg=1.5,
        rssi_noise_db=1.0,
        description="Static convergence under the standard realistic channel.",
    ),
    Scenario(
        key="static_hard_noise",
        title="Static target — hard noise",
        category="static",
        motion="static",
        duration_s=120.0,
        interval_s=0.20,
        doa_noise_deg=2.0,
        rssi_noise_db=1.5,
        description="Static robustness with stronger DoA and RSSI noise.",
    ),
    Scenario(
        key="moving_straight",
        title="Moving target — straight route",
        category="moving",
        motion="straight",
        duration_s=90.0,
        interval_s=0.20,
        doa_noise_deg=1.5,
        rssi_noise_db=1.0,
        speed_mps=6.0,
        description="Constant-speed route through useful two-receiver geometry.",
    ),
    Scenario(
        key="moving_hard_noise",
        title="Moving target — hard noise",
        category="moving",
        motion="straight",
        duration_s=90.0,
        interval_s=0.20,
        doa_noise_deg=2.0,
        rssi_noise_db=1.5,
        speed_mps=10.0,
        description="Faster motion with stronger noise and channel outliers.",
    ),
    Scenario(
        key="moving_curve",
        title="Moving target — curved route",
        category="moving",
        motion="circle",
        duration_s=120.0,
        interval_s=0.20,
        doa_noise_deg=1.5,
        rssi_noise_db=1.0,
        speed_mps=6.0,
        target_north_m=300.0,
        radius_m=120.0,
        description="Continuous heading changes test EKF model mismatch.",
    ),
    Scenario(
        key="stop_move_stop",
        title="Stop–move–stop target",
        category="transition",
        motion="stop_move_stop",
        duration_s=100.0,
        interval_s=0.20,
        doa_noise_deg=1.5,
        rssi_noise_db=1.0,
        speed_mps=6.0,
        description="Tests automatic STATIC/TRACK transitions.",
    ),
    Scenario(
        key="bad_geometry",
        title="Static target — poor geometry",
        category="geometry",
        motion="static",
        duration_s=120.0,
        interval_s=0.20,
        doa_noise_deg=1.5,
        rssi_noise_db=1.0,
        target_north_m=1500.0,
        description="Shallow bearing intersection demonstrates geometric limits.",
    ),
)


def scenario_by_key(key: str) -> Scenario:
    for scenario in SCENARIOS:
        if scenario.key == key:
            return scenario
    raise KeyError(key)


def local_offset(
    latitude: float,
    longitude: float,
    east_meters: float,
    north_meters: float,
) -> tuple[float, float]:
    distance = math.hypot(east_meters, north_meters)
    if distance <= 0.0:
        return latitude, longitude
    heading = math.degrees(math.atan2(east_meters, north_meters)) % 360.0
    return destination_point(latitude, longitude, heading, distance)


def receiver_layout(scenario: Scenario) -> list[dict[str, object]]:
    half_baseline = scenario.baseline_m / 2.0
    west = local_offset(DEFAULT_LATITUDE, DEFAULT_LONGITUDE, -half_baseline, 0.0)
    east = local_offset(DEFAULT_LATITUDE, DEFAULT_LONGITUDE, half_baseline, 0.0)
    return [
        {
            "name": "Kraken-1",
            "latitude": west[0],
            "longitude": west[1],
            "host": "127.0.0.1",
            "port": 4200,
        },
        {
            "name": "Kraken-2",
            "latitude": east[0],
            "longitude": east[1],
            "host": "127.0.0.1",
            "port": 4200,
        },
    ]


def truth_state(
    scenario: Scenario,
    elapsed_s: float,
) -> tuple[float, float, float, float]:
    """Return latitude, longitude, speed and heading for one scenario time."""
    north = scenario.target_north_m
    if scenario.motion == "static":
        east = 0.0
        speed = 0.0
        heading = scenario.heading_degrees
    elif scenario.motion == "straight":
        east = -250.0 + scenario.speed_mps * elapsed_s
        speed = scenario.speed_mps
        heading = 90.0
    elif scenario.motion == "circle":
        angular_speed = scenario.speed_mps / scenario.radius_m
        angle = angular_speed * elapsed_s
        east = scenario.radius_m * math.cos(angle)
        north = scenario.target_north_m + scenario.radius_m * math.sin(angle)
        speed = scenario.speed_mps
        heading = (-math.degrees(angle)) % 360.0
    elif scenario.motion == "stop_move_stop":
        moving_start = 20.0
        moving_end = 70.0
        moving_elapsed = max(0.0, min(elapsed_s, moving_end) - moving_start)
        east = -180.0 + scenario.speed_mps * moving_elapsed
        speed = scenario.speed_mps if moving_start <= elapsed_s < moving_end else 0.0
        heading = 90.0
    else:
        raise ValueError(f"Unknown motion model: {scenario.motion}")
    latitude, longitude = local_offset(
        DEFAULT_LATITUDE,
        DEFAULT_LONGITUDE,
        east,
        north,
    )
    return latitude, longitude, speed, heading


def expected_moving(scenario: Scenario, elapsed_s: float) -> bool:
    return truth_state(scenario, elapsed_s)[2] > 0.05


def configure_simulator(
    simulator: SimulatorWindow,
    scenario: Scenario,
) -> None:
    simulator.krakens = receiver_layout(scenario)
    latitude, longitude, speed, heading = truth_state(scenario, 0.0)
    simulator.targets = [
        {
            "name": "Target-1",
            "latitude": latitude,
            "longitude": longitude,
            "frequency": 433.92,
            "active": True,
            "moving": speed > 0.05,
            "speed_mps": speed,
            "_current_speed_mps": speed,
            "heading_degrees": heading,
            "random_speed": False,
        }
    ]
    simulator.interval.setValue(scenario.interval_s)
    simulator.realistic_noise.setChecked(True)
    simulator.evaluation_mode.setChecked(True)
    simulator.doa_noise.setValue(scenario.doa_noise_deg)
    simulator.rssi_noise.setValue(scenario.rssi_noise_db)
    simulator.receiver_sequences = {
        receiver["name"]: 0 for receiver in simulator.krakens
    }
    simulator.receiver_biases = {
        receiver["name"]: random.gauss(0.0, RECEIVER_BIAS_STD_DEGREES)
        for receiver in simulator.krakens
    }
    simulator.link_shadow_offsets = {}
    simulator.multipath_bursts = {}


def configure_scope(scope: KrakenScopeWindow) -> None:
    scope.clear_session_state(clear_packet=True)
    scope.config["cycle_window"] = 0.5
    scope.config["timeout"] = 3.0
    # Avoid GUI painting and table work during high-speed offline execution.
    # Localization and evaluation methods remain the production methods.
    scope.update_map = lambda: None
    scope.refresh = lambda: None


def run_scenario(
    scope: KrakenScopeWindow,
    simulator: SimulatorWindow,
    scenario: Scenario,
    seed: int,
    duration_scale: float = 1.0,
) -> list[dict[str, object]]:
    random.seed(seed)
    configure_scope(scope)
    configure_simulator(simulator, scenario)

    duration_s = max(scenario.interval_s * 2.0, scenario.duration_s * duration_scale)
    cycle_count = int(math.floor(duration_s / scenario.interval_s)) + 1
    base_timestamp_ms = 1_700_000_000_000.0 + (seed % 100_000) * 1_000_000.0
    target = simulator.targets[0]

    for cycle_index in range(cycle_count):
        elapsed_s = cycle_index * scenario.interval_s
        latitude, longitude, speed, heading = truth_state(scenario, elapsed_s)
        target.update(
            {
                "latitude": latitude,
                "longitude": longitude,
                "moving": speed > 0.05,
                "speed_mps": speed,
                "_current_speed_mps": speed,
                "heading_degrees": heading,
            }
        )
        timestamp_ms = base_timestamp_ms + elapsed_s * 1000.0
        for receiver_index, receiver in enumerate(simulator.krakens):
            packet = simulator._build_message(receiver, timestamp_ms)
            scope.handle_datagram(
                json.dumps(packet, ensure_ascii=False).encode("utf-8"),
                ("127.0.0.1", 50000 + receiver_index),
                # The offline runner performs no wall-clock sleeping. Still
                # refresh receiver liveness on every injected packet so plot
                # generation or a slow host cannot trigger the GUI's real-time
                # three-second offline timeout between simulation cycles.
                time.time(),
            )
    return [dict(row) for row in scope.evaluation_samples]


def finite_values(rows: list[dict[str, object]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value in {None, ""}:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def root_mean_square(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def improvement_percent(tracked_rmse: float, raw_rmse: float) -> float:
    if raw_rmse <= 1e-12:
        return 0.0
    return (raw_rmse - tracked_rmse) / raw_rmse * 100.0


def mode_accuracy(rows: list[dict[str, object]], scenario: Scenario) -> float:
    # Give the radio-only detector an initial evidence period. Later mode
    # changes remain part of the metric and are not hidden.
    considered = [
        row
        for row in rows
        if float(row.get("elapsed_s", 0.0) or 0.0) >= 15.0
    ]
    if not considered:
        considered = rows
    if not considered:
        return 0.0
    correct = 0
    for row in considered:
        elapsed_s = float(row.get("elapsed_s", 0.0) or 0.0)
        detected = bool(int(float(row.get("moving", 0) or 0)))
        correct += int(detected == expected_moving(scenario, elapsed_s))
    return correct / len(considered) * 100.0


def expected_mode_transitions(scenario: Scenario) -> list[tuple[float, bool]]:
    """Return the truth-defined mode transitions used only for evaluation."""
    if scenario.motion in {"straight", "circle"}:
        return [(0.0, True)]
    if scenario.motion == "stop_move_stop":
        return [(20.0, True), (70.0, False)]
    return []


def mode_transition_metrics(
    rows: list[dict[str, object]],
    scenario: Scenario,
    minimum_stable_seconds: float = 2.0,
) -> dict[str, float | int | None]:
    """Measure transition latency separately from steady-state accuracy.

    A detected transition must persist for at least ``minimum_stable_seconds``.
    This prevents a one-sample mode flicker from being credited as successful
    detection. Ground truth is consulted here only after localization.
    """
    samples = sorted(
        (
            float(row.get("elapsed_s", 0.0) or 0.0),
            bool(int(float(row.get("moving", 0) or 0))),
        )
        for row in rows
    )
    if not samples:
        return {
            "start_detection_delay_s": None,
            "stop_detection_delay_s": None,
            "mode_switches": 0,
            "false_mode_switches": 0,
            "missed_mode_transitions": len(expected_mode_transitions(scenario)),
        }

    segments: list[dict[str, float | bool | int]] = []
    for timestamp, moving in samples:
        if not segments or bool(segments[-1]["moving"]) != moving:
            segments.append(
                {
                    "moving": moving,
                    "start": timestamp,
                    "last": timestamp,
                    "index": len(segments),
                }
            )
        else:
            segments[-1]["last"] = timestamp

    matched_switches: set[int] = set()
    delays: dict[bool, list[float]] = {True: [], False: []}
    missed = 0
    for expected_time, expected_mode in expected_mode_transitions(scenario):
        match = next(
            (
                segment
                for segment in segments
                if bool(segment["moving"]) == expected_mode
                and float(segment["start"]) >= expected_time
                and float(segment["last"]) - float(segment["start"])
                >= minimum_stable_seconds
            ),
            None,
        )
        if match is None:
            missed += 1
            continue
        delays[expected_mode].append(float(match["start"]) - expected_time)
        segment_index = int(match["index"])
        if segment_index > 0:
            matched_switches.add(segment_index)

    actual_switches = max(0, len(segments) - 1)
    false_switches = max(0, actual_switches - len(matched_switches))
    return {
        "start_detection_delay_s": delays[True][0] if delays[True] else None,
        "stop_detection_delay_s": delays[False][0] if delays[False] else None,
        "mode_switches": actual_switches,
        "false_mode_switches": false_switches,
        "missed_mode_transitions": missed,
    }


def summarize_run(
    rows: list[dict[str, object]],
    scenario: Scenario,
    repeat: int,
    seed: int,
) -> dict[str, object]:
    tracked = finite_values(rows, "tracked_error_m")
    raw = finite_values(rows, "raw_error_m")
    steady_rows = [
        row for row in rows if float(row.get("elapsed_s", 0.0) or 0.0) >= 15.0
    ]
    if not steady_rows:
        steady_rows = rows
    steady_tracked = finite_values(steady_rows, "tracked_error_m")
    steady_raw = finite_values(steady_rows, "raw_error_m")
    geometry = finite_values(rows, "geometry_score")
    tracked_rmse = root_mean_square(tracked)
    raw_rmse = root_mean_square(raw)
    transition_metrics = mode_transition_metrics(rows, scenario)
    return {
        "scenario": scenario.key,
        "repeat": repeat,
        "seed": seed,
        "samples": len(tracked),
        "tracked_mean_m": mean(tracked),
        "tracked_rmse_m": tracked_rmse,
        "steady_tracked_rmse_m": root_mean_square(steady_tracked),
        "tracked_median_m": statistics.median(tracked) if tracked else 0.0,
        "tracked_p95_m": percentile(tracked, 0.95),
        "tracked_max_m": max(tracked, default=0.0),
        "raw_mean_m": mean(raw),
        "raw_rmse_m": raw_rmse,
        "steady_raw_rmse_m": root_mean_square(steady_raw),
        "rmse_improvement_percent": improvement_percent(tracked_rmse, raw_rmse),
        "auto_mode_accuracy_percent": mode_accuracy(rows, scenario),
        **transition_metrics,
        "mean_geometry_score": mean(geometry),
    }


def summarize_scenario(
    scenario: Scenario,
    run_summaries: list[dict[str, object]],
    run_rows: list[list[dict[str, object]]],
    duration_scale: float,
) -> dict[str, object]:
    all_rows = [row for rows in run_rows for row in rows]
    tracked = finite_values(all_rows, "tracked_error_m")
    raw = finite_values(all_rows, "raw_error_m")
    steady_rows = [
        row
        for row in all_rows
        if float(row.get("elapsed_s", 0.0) or 0.0) >= 15.0
    ]
    if not steady_rows:
        steady_rows = all_rows
    steady_tracked = finite_values(steady_rows, "tracked_error_m")
    steady_raw = finite_values(steady_rows, "raw_error_m")
    geometry = finite_values(all_rows, "geometry_score")
    run_rmses = [float(item["tracked_rmse_m"]) for item in run_summaries]
    tracked_rmse = root_mean_square(tracked)
    raw_rmse = root_mean_square(raw)
    start_delays = [
        float(item["start_detection_delay_s"])
        for item in run_summaries
        if item.get("start_detection_delay_s") is not None
    ]
    stop_delays = [
        float(item["stop_detection_delay_s"])
        for item in run_summaries
        if item.get("stop_detection_delay_s") is not None
    ]
    return {
        "scenario": scenario.key,
        "title": scenario.title,
        "category": scenario.category,
        "repeats": len(run_summaries),
        "samples": len(tracked),
        "duration_s": scenario.duration_s * duration_scale,
        "interval_s": scenario.interval_s,
        "doa_noise_deg": scenario.doa_noise_deg,
        "rssi_noise_db": scenario.rssi_noise_db,
        "speed_mps": scenario.speed_mps,
        "tracked_mean_m": mean(tracked),
        "tracked_rmse_m": tracked_rmse,
        "tracked_rmse_run_mean_m": mean(run_rmses),
        "tracked_rmse_run_std_m": (
            statistics.pstdev(run_rmses) if len(run_rmses) > 1 else 0.0
        ),
        "tracked_rmse_95ci_m": (
            1.96 * statistics.stdev(run_rmses) / math.sqrt(len(run_rmses))
            if len(run_rmses) > 1
            else 0.0
        ),
        "steady_tracked_rmse_m": root_mean_square(steady_tracked),
        "tracked_median_m": statistics.median(tracked) if tracked else 0.0,
        "tracked_p95_m": percentile(tracked, 0.95),
        "tracked_max_m": max(tracked, default=0.0),
        "raw_mean_m": mean(raw),
        "raw_rmse_m": raw_rmse,
        "steady_raw_rmse_m": root_mean_square(steady_raw),
        "rmse_improvement_percent": improvement_percent(tracked_rmse, raw_rmse),
        "auto_mode_accuracy_percent": mean(
            [float(item["auto_mode_accuracy_percent"]) for item in run_summaries]
        ),
        "mean_start_detection_delay_s": (
            mean(start_delays) if start_delays else None
        ),
        "mean_stop_detection_delay_s": (
            mean(stop_delays) if stop_delays else None
        ),
        "mean_mode_switches": mean(
            [float(item["mode_switches"]) for item in run_summaries]
        ),
        "total_false_mode_switches": sum(
            int(item["false_mode_switches"]) for item in run_summaries
        ),
        "total_missed_mode_transitions": sum(
            int(item["missed_mode_transitions"]) for item in run_summaries
        ),
        "mean_geometry_score": mean(geometry),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_representative_plots(
    output_dir: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        return
    string_rows = [
        {key: str(value) for key, value in row.items()}
        for row in rows
    ]
    target_name = str(string_rows[0].get("target", "Target-1"))
    (output_dir / "trajectory.svg").write_text(
        render_trajectory(target_name, string_rows),
        encoding="utf-8",
    )
    (output_dir / "error.svg").write_text(
        render_error_plot(target_name, string_rows),
        encoding="utf-8",
    )


def render_comparison(summary_rows: list[dict[str, object]]) -> str:
    width = 1180
    height = 680
    left = 90
    right = 40
    top = 90
    bottom = 165
    chart_width = width - left - right
    chart_height = height - top - bottom
    maximum = max(
        (
            max(float(row["tracked_rmse_m"]), float(row["raw_rmse_m"]))
            for row in summary_rows
        ),
        default=1.0,
    )
    maximum = max(10.0, maximum * 1.15)
    group_width = chart_width / max(1, len(summary_rows))
    bar_width = min(38.0, group_width * 0.28)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="90" y="45" font-family="Arial" font-size="27" font-weight="700" '
        'fill="#172033">KrakenScope scenario comparison</text>',
        '<text x="90" y="72" font-family="Arial" font-size="15" fill="#64748b">'
        'Pooled RMSE across all deterministic seeds</text>',
    ]
    for index in range(6):
        ratio = index / 5.0
        y = top + chart_height * (1.0 - ratio)
        value = maximum * ratio
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" '
            'stroke="#e2e8f0"/>'
        )
        parts.append(
            f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-family="Arial" '
            f'font-size="13" fill="#64748b">{value:.0f} m</text>'
        )
    for index, row in enumerate(summary_rows):
        center_x = left + group_width * (index + 0.5)
        tracked = float(row["tracked_rmse_m"])
        raw = float(row["raw_rmse_m"])
        tracked_height = tracked / maximum * chart_height
        raw_height = raw / maximum * chart_height
        tracked_x = center_x - bar_width - 3
        raw_x = center_x + 3
        parts.extend(
            [
                f'<rect x="{tracked_x:.1f}" y="{top+chart_height-tracked_height:.1f}" '
                f'width="{bar_width:.1f}" height="{tracked_height:.1f}" rx="4" fill="#f59e0b"/>',
                f'<rect x="{raw_x:.1f}" y="{top+chart_height-raw_height:.1f}" '
                f'width="{bar_width:.1f}" height="{raw_height:.1f}" rx="4" fill="#ef4444"/>',
                f'<text x="{center_x:.1f}" y="{height-bottom+28}" text-anchor="end" '
                f'transform="rotate(-38 {center_x:.1f} {height-bottom+28})" '
                f'font-family="Arial" font-size="13" fill="#334155">{html.escape(str(row["scenario"]))}</text>',
            ]
        )
    parts.extend(
        [
            '<rect x="850" y="28" width="16" height="16" rx="3" fill="#f59e0b"/>',
            '<text x="875" y="41" font-family="Arial" font-size="14" fill="#334155">Tracked</text>',
            '<rect x="965" y="28" width="16" height="16" rx="3" fill="#ef4444"/>',
            '<text x="990" y="41" font-family="Arial" font-size="14" fill="#334155">Raw</text>',
            f'<line x1="{left}" y1="{top+chart_height}" x2="{width-right}" '
            f'y2="{top+chart_height}" stroke="#26364f" stroke-width="2"/>',
            '</svg>\n',
        ]
    )
    return "".join(parts)


def write_summary_markdown(
    path: Path,
    summary_rows: list[dict[str, object]],
    repeats: int,
    seed: int,
) -> None:
    lines = [
        "# KrakenScope automated experiment summary",
        "",
        f"Base seed: `{seed}` · repetitions per scenario: `{repeats}`",
        "",
        "| Scenario | Samples | Tracked RMSE | Steady RMSE | Raw RMSE | Improvement | P95 | Mode accuracy | Start delay | Stop delay | False switches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {scenario} | {samples} | {tracked:.1f} m | {steady:.1f} m | {raw:.1f} m | "
            "{improvement:+.1f}% | {p95:.1f} m | {mode:.1f}% | {start} | {stop} | {false_switches} |".format(
                scenario=row["scenario"],
                samples=row["samples"],
                tracked=float(row["tracked_rmse_m"]),
                steady=float(row["steady_tracked_rmse_m"]),
                raw=float(row["raw_rmse_m"]),
                improvement=float(row["rmse_improvement_percent"]),
                p95=float(row["tracked_p95_m"]),
                mode=float(row["auto_mode_accuracy_percent"]),
                start=(
                    "—"
                    if row.get("mean_start_detection_delay_s") is None
                    else f"{float(row['mean_start_detection_delay_s']):.1f} s"
                ),
                stop=(
                    "—"
                    if row.get("mean_stop_detection_delay_s") is None
                    else f"{float(row['mean_stop_detection_delay_s']):.1f} s"
                ),
                false_switches=int(row["total_false_mode_switches"]),
            )
        )
    lines.extend(
        [
            "",
            "`Tracked` is the KrakenScope output. `Raw` is the instantaneous two-bearing intersection.",
            "Ground truth is used only after localization to calculate these metrics.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible KrakenScope simulation experiments."
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[scenario.key for scenario in SCENARIOS],
        help="Scenario key. Repeat the option to select multiple; default is all.",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument(
        "--duration-scale",
        type=float,
        default=1.0,
        help="Scale every scenario duration, useful for smoke tests.",
    )
    parser.add_argument(
        "--plots",
        choices=("first", "all", "none"),
        default="first",
        help="Generate plots for the first, every, or no repetition.",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--list", action="store_true", help="List scenarios and exit.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if not math.isfinite(args.duration_scale) or args.duration_scale <= 0.0:
        raise SystemExit("--duration-scale must be a positive number")


def main() -> None:
    args = parse_args()
    if args.list:
        for scenario in SCENARIOS:
            print(
                f"{scenario.key:22} {scenario.duration_s:5.0f}s  "
                f"DoA {scenario.doa_noise_deg:.1f}°  {scenario.title}"
            )
        return
    validate_args(args)

    selected = (
        [scenario_by_key(key) for key in args.scenario]
        if args.scenario
        else list(SCENARIOS)
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.out or ROOT / "experiments" / "automated" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication([])
    simulator = SimulatorWindow()
    # The offline runner must never claim UDP port 4200. Temporarily suppress
    # the constructor's normal listener startup, then keep the production
    # method intact for all other uses.
    normal_start_listener = KrakenScopeWindow.start_listener
    KrakenScopeWindow.start_listener = lambda self: None
    try:
        scope = KrakenScopeWindow()
    finally:
        KrakenScopeWindow.start_listener = normal_start_listener
    scope.refresh_timer.stop()
    scope.map.animation_timer.stop()
    simulator.map.animation_timer.stop()

    all_run_summaries: list[dict[str, object]] = []
    scenario_summaries: list[dict[str, object]] = []
    try:
        for scenario_index, scenario in enumerate(selected):
            print(f"[{scenario_index + 1}/{len(selected)}] {scenario.key}")
            scenario_dir = output_dir / scenario.key
            scenario_dir.mkdir(parents=True, exist_ok=True)
            scenario_run_summaries = []
            scenario_run_rows = []
            for repeat_index in range(args.repeats):
                seed = args.seed + scenario_index * 10_000 + repeat_index
                rows = run_scenario(
                    scope,
                    simulator,
                    scenario,
                    seed,
                    duration_scale=args.duration_scale,
                )
                repeat_number = repeat_index + 1
                run_path = scenario_dir / f"run_{repeat_number:03d}.csv"
                write_csv(run_path, EXPERIMENT_FIELDS, rows)
                run_summary = summarize_run(
                    rows,
                    scenario,
                    repeat_number,
                    seed,
                )
                scenario_run_summaries.append(run_summary)
                all_run_summaries.append(run_summary)
                scenario_run_rows.append(rows)
                if args.plots == "all" or (
                    args.plots == "first" and repeat_index == 0
                ):
                    plot_dir = scenario_dir / f"run_{repeat_number:03d}_plots"
                    plot_dir.mkdir(parents=True, exist_ok=True)
                    write_representative_plots(plot_dir, rows)
                print(
                    "  run {run:02d} seed={seed} tracked RMSE={tracked:.1f} m "
                    "raw={raw:.1f} m".format(
                        run=repeat_number,
                        seed=seed,
                        tracked=float(run_summary["tracked_rmse_m"]),
                        raw=float(run_summary["raw_rmse_m"]),
                    )
                )
            write_csv(
                scenario_dir / "runs_summary.csv",
                RUN_SUMMARY_FIELDS,
                scenario_run_summaries,
            )
            scenario_summaries.append(
                summarize_scenario(
                    scenario,
                    scenario_run_summaries,
                    scenario_run_rows,
                    args.duration_scale,
                )
            )
    finally:
        scope.stop_listener()
        scope.close()
        simulator.close()

    write_csv(output_dir / "runs_summary.csv", RUN_SUMMARY_FIELDS, all_run_summaries)
    write_csv(output_dir / "summary.csv", SUMMARY_FIELDS, scenario_summaries)
    write_summary_markdown(
        output_dir / "summary.md",
        scenario_summaries,
        args.repeats,
        args.seed,
    )
    (output_dir / "comparison.svg").write_text(
        render_comparison(scenario_summaries),
        encoding="utf-8",
    )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_seed": args.seed,
        "repeats": args.repeats,
        "duration_scale": args.duration_scale,
        "plots": args.plots,
        "scenarios": [asdict(scenario) for scenario in selected],
        "pipeline": {
            "simulator": "KrakenSDR Simulator realistic channel model",
            "localization": "KrakenScope production GUI localization pipeline",
            "ground_truth_usage": "post-localization metrics only",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nCompleted. Results: {output_dir}")


if __name__ == "__main__":
    main()
