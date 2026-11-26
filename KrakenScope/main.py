"""KrakenScope simulator localization dashboard."""

from __future__ import annotations

import json
import math
import os
import socket
import stat
import statistics
import sys
import time
import csv
from datetime import datetime
from dataclasses import dataclass, field, replace
from pathlib import Path

# Stale Qt variables from Conda can override PySide6's bundled macOS plugins.
os.environ.pop("QT_PLUGIN_PATH", None)
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

# iCloud Drive can mark Qt plugins as hidden. Qt then misses the Cocoa plugin.
_qt_platform_dir = (
    Path(sys.prefix)
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
    / "PySide6"
    / "Qt"
    / "plugins"
    / "platforms"
)
if _qt_platform_dir.is_dir() and hasattr(os, "chflags"):
    for _plugin in _qt_platform_dir.glob("*.dylib"):
        try:
            os.chflags(_plugin, _plugin.stat().st_flags & ~stat.UF_HIDDEN)
        except OSError:
            pass

from PySide6.QtCore import QPointF, QProcess, QRectF, QSettings, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtNetwork import QNetworkDiskCache
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core import (
    BearingMeasurement,
    BearingTrackState,
    EstimatedTarget,
    MotionAnalysis,
    PacketError,
    ReceiverState,
    analyze_observed_motion,
    analyze_bearing_motion,
    bucket_timestamped_bearings,
    continuous_time_alpha,
    distance_meters,
    estimate_static_target,
    estimate_target,
    initialize_bearing_track,
    parse_packet,
    receiver_key,
    update_bearing_track,
)


ROOT = Path(__file__).resolve().parent
SIMULATOR_ROOT = ROOT.parent / "KrakenSDR_Simulator-main"
SIMULATOR_GUI = SIMULATOR_ROOT / "simulator_gui.py"
if str(SIMULATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_ROOT))

from simulator_qt import DEFAULT_LATITUDE, DEFAULT_LONGITUDE, SlippyMap  # noqa: E402


SIMULATOR_TILE_CACHE = (
    Path.home()
    / "Library"
    / "Caches"
    / "University of Cyprus"
    / "KrakenSDR Simulator"
    / "KrakenSDRSimulator"
    / "map-tiles"
)

EXPERIMENT_FIELDS = [
    "run_iso",
    "target",
    "timestamp_ms",
    "elapsed_s",
    "mode",
    "moving",
    "speed_mps",
    "heading_degrees",
    "bearing_angular_rate_dps",
    "bearing_motion_confidence",
    "motion_scatter_m",
    "tracker_recovered",
    "raw_reference_gap_m",
    "tracker_reference_gap_m",
    "truth_latitude",
    "truth_longitude",
    "tracked_latitude",
    "tracked_longitude",
    "raw_latitude",
    "raw_longitude",
    "tracked_error_m",
    "raw_error_m",
    "filter_delta_m",
    "residual_m",
    "geometry_score",
    "geometry_label",
    "mean_rssi_db",
    "measurement_count",
    "accepted_samples",
    "rejected_samples",
    "active_receivers",
    "cycle_window_s",
]


@dataclass
class EvaluationStats:
    samples: int = 0
    current_error: float = 0.0
    current_raw_error: float | None = None
    sum_error: float = 0.0
    sum_squared_error: float = 0.0
    maximum_error: float = 0.0
    last_timestamp: float | None = None
    errors: list[float] = field(default_factory=list)

    def add(
        self,
        error_meters: float,
        timestamp: float,
        raw_error_meters: float | None = None,
    ) -> None:
        if self.last_timestamp == timestamp:
            return
        self.samples += 1
        self.current_error = error_meters
        self.current_raw_error = raw_error_meters
        self.sum_error += error_meters
        self.sum_squared_error += error_meters * error_meters
        self.maximum_error = max(self.maximum_error, error_meters)
        self.last_timestamp = timestamp
        self.errors.append(error_meters)

    @property
    def current_filter_delta(self) -> float | None:
        if self.current_raw_error is None:
            return None
        return self.current_error - self.current_raw_error

    @property
    def average_error(self) -> float:
        return self.sum_error / self.samples if self.samples else 0.0

    @property
    def rmse(self) -> float:
        return math.sqrt(self.sum_squared_error / self.samples) if self.samples else 0.0

    @property
    def median_error(self) -> float:
        return statistics.median(self.errors) if self.errors else 0.0

    @property
    def standard_deviation(self) -> float:
        return statistics.pstdev(self.errors) if len(self.errors) > 1 else 0.0

    @property
    def percentile_95(self) -> float:
        if not self.errors:
            return 0.0
        ordered = sorted(self.errors)
        index = math.ceil(0.95 * len(ordered)) - 1
        return ordered[max(0, index)]

    @property
    def quality(self) -> str:
        if self.rmse < 10.0:
            return "EXCELLENT"
        if self.rmse < 25.0:
            return "GOOD"
        if self.rmse < 50.0:
            return "FAIR"
        return "POOR"


class LiveExperimentPlot(QWidget):
    """Small dependency-free live plot for thesis experiment monitoring."""

    def __init__(self, plot_kind: str, title: str, parent=None):
        super().__init__(parent)
        self.plot_kind = plot_kind
        self.title = title
        self.samples: list[dict[str, object]] = []
        self.setMinimumHeight(250)

    def set_samples(self, samples: list[dict[str, object]]) -> None:
        self.samples = list(samples[-600:])
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        outer = QRectF(self.rect()).adjusted(4, 4, -4, -4)
        painter.setPen(QPen(QColor("#1e3657"), 1.2))
        painter.setBrush(QColor("#07111f"))
        painter.drawRoundedRect(outer, 14, 14)

        painter.setPen(QColor("#e5edf8"))
        painter.drawText(
            outer.adjusted(16, 13, -16, -12),
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
            self.title,
        )
        plot_rect = outer.adjusted(52, 48, -22, -42)
        if len(self.samples) < 2:
            painter.setPen(QColor("#64748b"))
            painter.drawText(
                plot_rect,
                Qt.AlignmentFlag.AlignCenter,
                "Waiting for ground-truth evaluation samples…",
            )
            painter.end()
            return

        if self.plot_kind == "trajectory":
            self._paint_trajectory(painter, plot_rect)
        else:
            self._paint_error(painter, plot_rect)
        painter.end()

    def _paint_trajectory(self, painter: QPainter, rect: QRectF) -> None:
        rows = [
            row for row in self.samples
            if self._has_values(
                row,
                "truth_latitude",
                "truth_longitude",
                "tracked_latitude",
                "tracked_longitude",
            )
        ]
        if len(rows) < 2:
            return
        reference = (
            self._float(rows[0]["truth_latitude"]),
            self._float(rows[0]["truth_longitude"]),
        )
        truth = [
            self._latlon_to_xy(
                self._float(row["truth_latitude"]),
                self._float(row["truth_longitude"]),
                reference,
            )
            for row in rows
        ]
        tracked = [
            self._latlon_to_xy(
                self._float(row["tracked_latitude"]),
                self._float(row["tracked_longitude"]),
                reference,
            )
            for row in rows
        ]
        bounds = self._bounds(truth + tracked)
        self._paint_grid(painter, rect, bounds, "m", "m")
        self._draw_polyline(painter, rect, bounds, truth, QColor("#22c55e"), 2.8)
        self._draw_polyline(painter, rect, bounds, tracked, QColor("#f59e0b"), 2.8)
        self._paint_legend(
            painter,
            rect,
            [
                ("Ground truth", QColor("#22c55e")),
                ("Tracked", QColor("#f59e0b")),
            ],
        )

    def _paint_error(self, painter: QPainter, rect: QRectF) -> None:
        tracked = [
            (self._float(row["elapsed_s"]), self._float(row["tracked_error_m"]))
            for row in self.samples
            if self._has_values(row, "elapsed_s", "tracked_error_m")
        ]
        if len(tracked) < 2:
            return
        max_x = max(1.0, max(point[0] for point in tracked))
        max_y = max(10.0, max(point[1] for point in tracked) * 1.15)
        bounds = (0.0, max_x, 0.0, max_y)
        self._paint_grid(painter, rect, bounds, "s", "m")
        self._draw_polyline(painter, rect, bounds, tracked, QColor("#f59e0b"), 2.8)
        self._paint_legend(
            painter,
            rect,
            [
                ("Tracked error", QColor("#f59e0b")),
            ],
        )

    def _paint_grid(
        self,
        painter: QPainter,
        rect: QRectF,
        bounds: tuple[float, float, float, float],
        x_unit: str,
        y_unit: str,
    ) -> None:
        min_x, max_x, min_y, max_y = bounds
        painter.setPen(QPen(QColor("#18304f"), 1))
        for index in range(5):
            ratio = index / 4
            x = rect.left() + ratio * rect.width()
            y = rect.bottom() - ratio * rect.height()
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            painter.setPen(QColor("#7890ad"))
            painter.drawText(
                QPointF(x - 18, rect.bottom() + 22),
                f"{min_x + ratio * (max_x - min_x):.0f}{x_unit}",
            )
            painter.drawText(
                QPointF(rect.left() - 48, y + 5),
                f"{min_y + ratio * (max_y - min_y):.0f}{y_unit}",
            )
            painter.setPen(QPen(QColor("#18304f"), 1))
        painter.setPen(QPen(QColor("#3b5578"), 1.5))
        painter.drawRect(rect)

    def _paint_legend(
        self,
        painter: QPainter,
        rect: QRectF,
        items: list[tuple[str, QColor]],
    ) -> None:
        x = rect.left() + 12
        y = rect.top() + 14
        for index, (label, color) in enumerate(items):
            item_y = y + index * 21
            painter.setPen(QPen(color, 3))
            painter.drawLine(QPointF(x, item_y), QPointF(x + 24, item_y))
            painter.setPen(QColor("#cbd8ea"))
            painter.drawText(QPointF(x + 34, item_y + 5), label)

    def _draw_polyline(
        self,
        painter: QPainter,
        rect: QRectF,
        bounds: tuple[float, float, float, float],
        points: list[tuple[float, float]],
        color: QColor,
        width: float,
        dashed: bool = False,
    ) -> None:
        if len(points) < 2:
            return
        pen = QPen(color, width)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        projected = [self._project(rect, bounds, point) for point in points]
        for first, second in zip(projected, projected[1:]):
            painter.drawLine(first, second)
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(projected[-1], 4.5, 4.5)

    def _project(
        self,
        rect: QRectF,
        bounds: tuple[float, float, float, float],
        point: tuple[float, float],
    ) -> QPointF:
        min_x, max_x, min_y, max_y = bounds
        x, y = point
        sx = rect.left() + (x - min_x) / max(1e-9, max_x - min_x) * rect.width()
        sy = rect.bottom() - (y - min_y) / max(1e-9, max_y - min_y) * rect.height()
        return QPointF(sx, sy)

    @staticmethod
    def _bounds(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)
        span_x = max(1.0, max_x - min_x)
        span_y = max(1.0, max_y - min_y)
        return (
            min_x - span_x * 0.12,
            max_x + span_x * 0.12,
            min_y - span_y * 0.12,
            max_y + span_y * 0.12,
        )

    @staticmethod
    def _latlon_to_xy(
        latitude: float,
        longitude: float,
        reference: tuple[float, float],
    ) -> tuple[float, float]:
        reference_latitude, reference_longitude = reference
        earth_radius = 6_371_000.0
        x = math.radians(longitude - reference_longitude) * earth_radius * math.cos(
            math.radians(reference_latitude)
        )
        y = math.radians(latitude - reference_latitude) * earth_radius
        return x, y

    @staticmethod
    def _has_values(row: dict[str, object], *keys: str) -> bool:
        return all(str(row.get(key, "")).strip() not in {"", "None"} for key in keys)

    @staticmethod
    def _float(value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


TEXT = {
    "title": "KrakenScope — Simulator Localization",
    "settings": "Settings",
    "receiver": "Receiver",
    "status": "Status",
    "address": "Address",
    "lat": "Latitude",
    "lon": "Longitude",
    "packets": "Packets",
    "sequence": "Seq",
    "targets": "Targets",
    "target": "Target",
    "measurements": "Measurements",
    "rssi": "Mean RSSI",
    "residual": "Residual",
    "samples": "Samples",
    "current_error": "Current",
    "average_error": "Average",
    "rmse": "RMSE",
    "median_error": "Median",
    "standard_deviation": "Std dev",
    "percentile_95": "P95",
    "maximum_error": "Maximum",
    "quality": "Quality",
    "online": "ONLINE",
    "offline": "OFFLINE",
    "bind_ip": "Listener bind IP",
    "port": "UDP port",
    "timeout": "Offline timeout (s)",
    "cycle_window": "Cycle grouping window (s)",
    "listening": "Listening on {bind}:{port}",
    "stopped": "Listener stopped",
    "bind_error": "Could not listen on {bind}:{port}: {error}",
    "sim_error": "Could not open the simulator GUI.",
    "no_location": "Packet from {name} has no receiver latitude/longitude, so it cannot be localized.",
    "localized": "{targets} localized target(s) from {receivers} receiver(s)",
    "waiting": "Waiting for simulator packets...",
}


class UdpListener(QThread):
    datagram = Signal(bytes, object, float)
    failed = Signal(str)

    def __init__(self, bind_ip: str, port: int):
        super().__init__()
        self.bind_ip = bind_ip
        self.port = port
        self._running = True
        self._socket: socket.socket | None = None

    def run(self) -> None:
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind((self.bind_ip, self.port))
            self._socket.settimeout(0.25)
            while self._running:
                try:
                    data, address = self._socket.recvfrom(65535)
                    self.datagram.emit(data, address, time.time())
                except socket.timeout:
                    continue
                except OSError:
                    if self._running:
                        raise
        except OSError as exc:
            self.failed.emit(str(exc))
        finally:
            if self._socket:
                self._socket.close()

    def stop(self) -> None:
        self._running = False
        if self._socket:
            self._socket.close()
        self.wait(1000)


class SettingsDialog(QDialog):
    def __init__(self, values: dict, tr, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("settings"))
        form = QFormLayout(self)
        self.bind_ip = QLineEdit(values["bind_ip"])
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(values["port"])
        self.timeout = QDoubleSpinBox()
        self.timeout.setRange(0.5, 60.0)
        self.timeout.setSingleStep(0.5)
        self.timeout.setValue(values["timeout"])
        self.cycle_window = QDoubleSpinBox()
        self.cycle_window.setRange(0.05, 10.0)
        self.cycle_window.setSingleStep(0.05)
        self.cycle_window.setValue(values["cycle_window"])
        self.real_receiver_1_enabled = QCheckBox("Enable manual Real Kraken 1")
        self.real_receiver_1_enabled.setChecked(values["real_receiver_1_enabled"])
        self.real_receiver_1_name = QLineEdit(values["real_receiver_1_name"])
        self.real_receiver_1_source = QLineEdit(values["real_receiver_1_source"])
        self.real_receiver_1_source.setPlaceholderText("optional: 192.168.1.50 or 192.168.1.50:12345")
        self.real_receiver_1_latitude = self._coordinate_spin(-90.0, 90.0, values["real_receiver_1_latitude"])
        self.real_receiver_1_longitude = self._coordinate_spin(-180.0, 180.0, values["real_receiver_1_longitude"])

        self.real_receiver_2_enabled = QCheckBox("Enable manual Real Kraken 2")
        self.real_receiver_2_enabled.setChecked(values["real_receiver_2_enabled"])
        self.real_receiver_2_name = QLineEdit(values["real_receiver_2_name"])
        self.real_receiver_2_source = QLineEdit(values["real_receiver_2_source"])
        self.real_receiver_2_source.setPlaceholderText("optional: 192.168.1.51 or 192.168.1.51:12345")
        self.real_receiver_2_latitude = self._coordinate_spin(-90.0, 90.0, values["real_receiver_2_latitude"])
        self.real_receiver_2_longitude = self._coordinate_spin(-180.0, 180.0, values["real_receiver_2_longitude"])
        form.addRow(tr("bind_ip"), self.bind_ip)
        form.addRow(tr("port"), self.port)
        form.addRow(tr("timeout"), self.timeout)
        form.addRow(tr("cycle_window"), self.cycle_window)
        form.addRow(QLabel("REAL / LEGACY KRAKEN RECEIVERS"))
        form.addRow(self.real_receiver_1_enabled)
        form.addRow("Real Kraken 1 name", self.real_receiver_1_name)
        form.addRow("Real Kraken 1 source", self.real_receiver_1_source)
        form.addRow("Real Kraken 1 latitude", self.real_receiver_1_latitude)
        form.addRow("Real Kraken 1 longitude", self.real_receiver_1_longitude)
        form.addRow(self.real_receiver_2_enabled)
        form.addRow("Real Kraken 2 name", self.real_receiver_2_name)
        form.addRow("Real Kraken 2 source", self.real_receiver_2_source)
        form.addRow("Real Kraken 2 latitude", self.real_receiver_2_latitude)
        form.addRow("Real Kraken 2 longitude", self.real_receiver_2_longitude)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> dict:
        return {
            "bind_ip": self.bind_ip.text().strip() or "0.0.0.0",
            "port": self.port.value(),
            "timeout": self.timeout.value(),
            "cycle_window": self.cycle_window.value(),
            "real_receiver_1_enabled": self.real_receiver_1_enabled.isChecked(),
            "real_receiver_1_name": self.real_receiver_1_name.text().strip() or "Real-Kraken-1",
            "real_receiver_1_source": self.real_receiver_1_source.text().strip(),
            "real_receiver_1_latitude": self.real_receiver_1_latitude.value(),
            "real_receiver_1_longitude": self.real_receiver_1_longitude.value(),
            "real_receiver_2_enabled": self.real_receiver_2_enabled.isChecked(),
            "real_receiver_2_name": self.real_receiver_2_name.text().strip() or "Real-Kraken-2",
            "real_receiver_2_source": self.real_receiver_2_source.text().strip(),
            "real_receiver_2_latitude": self.real_receiver_2_latitude.value(),
            "real_receiver_2_longitude": self.real_receiver_2_longitude.value(),
        }

    @staticmethod
    def _coordinate_spin(minimum: float, maximum: float, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(7)
        spin.setSingleStep(0.0001)
        spin.setValue(value)
        return spin


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("UCY", "KrakenScope")
        self.config = {
            "bind_ip": self.settings.value("bind_ip", "0.0.0.0"),
            "port": int(self.settings.value("port", 4200)),
            "timeout": float(self.settings.value("timeout", 3.0)),
            "cycle_window": float(self.settings.value("cycle_window", 1.5)),
            "real_receiver_1_enabled": bool(
                self.settings.value(
                    "real_receiver_1_enabled",
                    self.settings.value("real_receiver_enabled", False, type=bool),
                    type=bool,
                )
            ),
            "real_receiver_1_name": str(
                self.settings.value(
                    "real_receiver_1_name",
                    self.settings.value("real_receiver_name", "Real-Kraken-1"),
                )
            ),
            "real_receiver_1_source": str(self.settings.value("real_receiver_1_source", "")),
            "real_receiver_1_latitude": float(
                self.settings.value(
                    "real_receiver_1_latitude",
                    self.settings.value("real_receiver_latitude", DEFAULT_LATITUDE),
                )
            ),
            "real_receiver_1_longitude": float(
                self.settings.value(
                    "real_receiver_1_longitude",
                    self.settings.value("real_receiver_longitude", DEFAULT_LONGITUDE),
                )
            ),
            "real_receiver_2_enabled": bool(
                self.settings.value("real_receiver_2_enabled", False, type=bool)
            ),
            "real_receiver_2_name": str(self.settings.value("real_receiver_2_name", "Real-Kraken-2")),
            "real_receiver_2_source": str(self.settings.value("real_receiver_2_source", "")),
            "real_receiver_2_latitude": float(
                self.settings.value("real_receiver_2_latitude", DEFAULT_LATITUDE)
            ),
            "real_receiver_2_longitude": float(
                self.settings.value("real_receiver_2_longitude", DEFAULT_LONGITUDE)
            ),
        }
        self.receivers: dict[str, ReceiverState] = {}
        self.estimates: list[EstimatedTarget] = []
        self.raw_estimates: dict[str, EstimatedTarget] = {}
        self.display_estimates: dict[str, EstimatedTarget] = {}
        self.smoothed_estimates: dict[str, EstimatedTarget] = {}
        self.estimate_measurements: dict[str, list[BearingMeasurement]] = {}
        self.sample_rejections: dict[str, tuple[int, int]] = {}
        self.target_motion: dict[str, dict] = {}
        self.target_observation_history: dict[str, list[tuple[float, float, float]]] = {}
        self.target_bearing_history: dict[
            str, dict[str, list[tuple[float, float]]]
        ] = {}
        self.target_bearing_motion_evidence: dict[
            str, list[tuple[float, float, float]]
        ] = {}
        self.target_motion_velocity_history: dict[
            str, list[tuple[float, float, float]]
        ] = {}
        self.motion_mode_candidates: dict[str, tuple[bool, float]] = {}
        self.auto_motion_modes: dict[str, bool] = {}
        self.static_history_started_ms: dict[str, float] = {}
        self.track_states: dict[str, BearingTrackState] = {}
        self.track_seed_estimates: dict[str, EstimatedTarget] = {}
        self.track_recovery_candidates: dict[
            str, tuple[float, float, float]
        ] = {}
        self.static_filter_timestamps: dict[str, float] = {}
        self.last_localization_packets: dict[str, dict[str, int]] = {}
        self.estimate_trails: dict[str, list[tuple[float, float]]] = {}
        self.ground_truths: dict[str, tuple[float, float]] = {}
        self.evaluation_stats: dict[str, EvaluationStats] = {}
        self.evaluation_samples: list[dict[str, object]] = []
        self._evaluation_sample_keys: set[tuple[str, float]] = set()
        self._evaluation_first_timestamps: dict[str, float] = {}
        self._experiment_run_iso = datetime.now().isoformat(timespec="seconds")
        self.show_raw_debug = bool(self.settings.value("show_raw_debug", False, type=bool))
        self.theme_mode = str(self.settings.value("theme_mode", "dark"))
        if self.theme_mode not in {"dark", "light"}:
            self.theme_mode = "dark"
        self.listener: UdpListener | None = None
        self.simulator = QProcess(self)
        self._auto_frame_done = False
        self._auto_frame_pending = False
        self._build_ui()
        self.apply_theme()
        self.retranslate()
        self.start_listener()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh)
        self.refresh_timer.start(250)

    def tr_text(self, key: str, **kwargs) -> str:
        return TEXT.get(key, key).format(**kwargs)

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("appRoot")
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 16, 18, 14)
        root.setSpacing(14)

        header = QFrame()
        header.setObjectName("header")
        header.setMaximumHeight(84)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 12, 14, 12)

        brand = QVBoxLayout()
        brand.setSpacing(0)
        self.app_title = QLabel("KRAKEN<span style='color:#38bdf8'>SCOPE</span>")
        self.app_title.setObjectName("appTitle")
        brand.addWidget(self.app_title)
        header_layout.addLayout(brand)
        header_layout.addSpacing(22)

        self.listener_status = QLabel("●  LISTENER STARTING")
        self.listener_status.setObjectName("statusBadge")
        self.listener_status.setFixedHeight(32)
        self.listener_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(self.listener_status)
        header_layout.addStretch()

        self.theme_button = QPushButton()
        self.theme_button.setObjectName("secondaryButton")
        self.theme_button.clicked.connect(self.toggle_theme)
        self.settings_button = QPushButton()
        self.settings_button.setObjectName("secondaryButton")
        self.settings_button.clicked.connect(self.open_settings)
        self.clear_button = QPushButton()
        self.clear_button.setObjectName("secondaryButton")
        self.clear_button.clicked.connect(self.clear_session)
        self.listener_button = QPushButton()
        self.listener_button.setObjectName("primaryButton")
        self.listener_button.clicked.connect(self.toggle_listener)
        self.simulator_button = QPushButton()
        self.simulator_button.setObjectName("secondaryButton")
        self.simulator_button.clicked.connect(self.toggle_simulator)
        for widget in (
            self.theme_button,
            self.settings_button,
            self.clear_button,
            self.simulator_button,
            self.listener_button,
        ):
            header_layout.addWidget(widget)
        root.addWidget(header)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(12)
        metrics.setVerticalSpacing(0)
        self.receivers_card, self.receivers_metric, self.receivers_metric_hint = self._metric_card(
            "◉", "ACTIVE RECEIVERS", "0", "Waiting for data", "#38bdf8"
        )
        self.targets_card, self.targets_metric, self.targets_metric_hint = self._metric_card(
            "◇", "LOCALIZED TARGETS", "0", "No solution yet", "#a78bfa"
        )
        self.mode_card, self.mode_metric, self.mode_metric_hint = self._metric_card(
            "⇄", "LOCALIZATION MODE", "WAITING", "No target mode yet", "#14b8a6"
        )
        self.rmse_card, self.rmse_metric, self.rmse_metric_hint = self._metric_card(
            "⌁", "CURRENT ERROR", "—", "Evaluation inactive", "#22c55e"
        )
        self.packets_card, self.packets_metric, self.packets_metric_hint = self._metric_card(
            "⇣", "PACKETS RECEIVED", "0", "UDP port 4200", "#f59e0b"
        )
        for column, card in enumerate(
            (
                self.receivers_card,
                self.targets_card,
                self.mode_card,
                self.rmse_card,
                self.packets_card,
            )
        ):
            metrics.addWidget(card, 0, column)
            metrics.setColumnStretch(column, 1)
        root.addLayout(metrics)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        map_panel = QFrame()
        map_panel.setObjectName("panel")
        map_layout = QVBoxLayout(map_panel)
        map_layout.setContentsMargins(7, 7, 7, 7)
        map_layout.setSpacing(0)

        map_stack = QWidget()
        map_stack.setObjectName("mapStack")
        map_stack_layout = QGridLayout(map_stack)
        map_stack_layout.setContentsMargins(0, 0, 0, 0)

        self.map = SlippyMap()
        self.map.setObjectName("liveMap")
        self.use_simulator_tile_cache()
        self.map.set_theme(self.theme_mode)
        self.map.locked = False
        self.map.set_view(DEFAULT_LATITUDE, DEFAULT_LONGITUDE, 15)
        map_stack_layout.addWidget(self.map, 0, 0)

        map_identity = QFrame()
        map_identity.setObjectName("floatingPanel")
        identity_layout = QVBoxLayout(map_identity)
        identity_layout.setContentsMargins(12, 9, 12, 9)
        identity_layout.setSpacing(5)
        self.map_title = QLabel("LIVE OPERATIONS MAP")
        self.map_title.setObjectName("floatingTitle")
        self.map_legend = QHBoxLayout()
        self.map_legend.setSpacing(12)
        self.raw_legend_label: QLabel | None = None
        for text, color in (
            ("▲ Receiver", "#38bdf8"),
            ("◆ Tracked", "#f59e0b"),
            ("× Raw", "#ef4444"),
            ("● Ground truth", "#22c55e"),
        ):
            label = QLabel(text)
            label.setObjectName("legendLabel")
            label.setProperty("accent", color)
            if "Raw" in text:
                self.raw_legend_label = label
                label.setVisible(self.show_raw_debug)
            self.map_legend.addWidget(label)
        self.raw_debug_toggle = QCheckBox("Show raw debug")
        self.raw_debug_toggle.setObjectName("rawDebugToggle")
        self.raw_debug_toggle.setChecked(self.show_raw_debug)
        self.raw_debug_toggle.toggled.connect(self.toggle_raw_debug)
        self.recenter_map_button = QPushButton("Recenter map")
        self.recenter_map_button.setObjectName("compactButton")
        self.recenter_map_button.clicked.connect(self.recenter_map)
        identity_layout.addWidget(self.map_title)
        identity_layout.addLayout(self.map_legend)
        identity_layout.addWidget(self.raw_debug_toggle)
        identity_layout.addWidget(self.recenter_map_button)
        map_stack_layout.addWidget(
            map_identity, 0, 0,
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
        )

        self.auto_frame_status = QLabel("●  AUTO FRAME")
        self.auto_frame_status.setObjectName("mapStatus")
        map_stack_layout.addWidget(
            self.auto_frame_status, 0, 0,
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
        )
        map_layout.addWidget(map_stack, 1)
        splitter.addWidget(map_panel)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(430)
        sidebar.setMaximumWidth(570)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(8)

        side_header = QHBoxLayout()
        self.inspector_title = QLabel("LIVE INSPECTOR")
        self.inspector_title.setObjectName("panelTitle")
        self.inspector_hint = QLabel("● LIVE DATA")
        self.inspector_hint.setObjectName("panelHint")
        side_header.addWidget(self.inspector_title)
        side_header.addStretch()
        side_header.addWidget(self.inspector_hint)
        sidebar_layout.addLayout(side_header)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)
        overview_layout.setContentsMargins(4, 10, 4, 4)
        overview_layout.setSpacing(8)
        self.estimate_title = QLabel()
        self.estimate_title.setObjectName("sectionTitle")
        overview_layout.addWidget(self.estimate_title)
        self.estimate_table = QTableWidget(0, 7)
        self._configure_table(self.estimate_table)
        overview_layout.addWidget(self.estimate_table, 1)

        evaluation_header = QHBoxLayout()
        self.evaluation_title = QLabel()
        self.evaluation_title.setObjectName("sectionTitle")
        self.reset_evaluation_button = QPushButton()
        self.reset_evaluation_button.setObjectName("compactButton")
        self.reset_evaluation_button.clicked.connect(self.reset_evaluation)
        self.export_evaluation_button = QPushButton()
        self.export_evaluation_button.setObjectName("compactButton")
        self.export_evaluation_button.clicked.connect(self.export_evaluation_csv)
        evaluation_header.addWidget(self.evaluation_title)
        evaluation_header.addStretch()
        evaluation_header.addWidget(self.export_evaluation_button)
        evaluation_header.addWidget(self.reset_evaluation_button)
        overview_layout.addLayout(evaluation_header)
        self.evaluation_table = QTableWidget(0, 6)
        self._configure_table(self.evaluation_table)
        overview_layout.addWidget(self.evaluation_table, 1)
        self.evaluation_details = QLabel("No evaluation samples yet")
        self.evaluation_details.setObjectName("metricsStrip")
        self.evaluation_details.setWordWrap(True)
        overview_layout.addWidget(self.evaluation_details)

        plots_tab = QWidget()
        plots_layout = QVBoxLayout(plots_tab)
        plots_layout.setContentsMargins(4, 10, 4, 4)
        plots_layout.setSpacing(10)
        self.live_trajectory_plot = LiveExperimentPlot(
            "trajectory",
            "LIVE TRAJECTORY · GROUND TRUTH VS ALGORITHM",
        )
        self.live_error_plot = LiveExperimentPlot(
            "error",
            "LIVE LOCALIZATION ERROR",
        )
        plots_layout.addWidget(self.live_trajectory_plot, 1)
        plots_layout.addWidget(self.live_error_plot, 1)

        receivers_tab = QWidget()
        receivers_layout = QVBoxLayout(receivers_tab)
        receivers_layout.setContentsMargins(4, 10, 4, 4)
        self.receiver_title = QLabel()
        self.receiver_title.setObjectName("sectionTitle")
        receivers_layout.addWidget(self.receiver_title)
        self.receiver_table = QTableWidget(0, 8)
        self._configure_table(self.receiver_table)
        receivers_layout.addWidget(self.receiver_table)

        packet_tab = QWidget()
        packet_layout = QVBoxLayout(packet_tab)
        packet_layout.setContentsMargins(4, 10, 4, 4)
        self.latest_title = QLabel()
        self.latest_title.setObjectName("sectionTitle")
        packet_layout.addWidget(self.latest_title)
        self.latest_packet = QTextEdit()
        self.latest_packet.setReadOnly(True)
        self.latest_packet.setPlaceholderText("Waiting for the first simulator packet…")
        packet_layout.addWidget(self.latest_packet)

        self.tabs.addTab(overview_tab, "  Overview  ")
        self.tabs.addTab(plots_tab, "  Live plots  ")
        self.tabs.addTab(receivers_tab, "  Receivers  ")
        self.tabs.addTab(packet_tab, "  Raw packet  ")
        sidebar_layout.addWidget(self.tabs)
        splitter.addWidget(sidebar)
        splitter.setSizes([1080, 470])

        root.addWidget(splitter)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().hide()
        self.resize(1550, 920)

    def _metric_card(
        self, icon: str, title: str, value: str, hint: str, accent: str
    ) -> tuple[QFrame, QLabel, QLabel]:
        card = QFrame()
        card.setObjectName("metricCard")
        card.setProperty("accent", accent)
        card.setMinimumHeight(104)
        card.setMaximumHeight(120)
        layout = QGridLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(1)
        icon_label = QLabel(icon)
        icon_label.setObjectName("metricIcon")
        icon_label.setProperty("accent", accent)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label = QLabel(title)
        title_label.setObjectName("metricTitle")
        value_label = QLabel(value)
        value_label.setObjectName("metricValue")
        value_label.setProperty("accent", accent)
        hint_label = QLabel(hint)
        hint_label.setObjectName("metricHint")
        layout.addWidget(icon_label, 0, 0, 2, 1)
        layout.addWidget(title_label, 0, 1)
        layout.addWidget(value_label, 1, 1)
        layout.addWidget(hint_label, 2, 0, 1, 2)
        layout.setColumnStretch(1, 1)
        return card, value_label, hint_label

    @staticmethod
    def _configure_table(table: QTableWidget) -> None:
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(34)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setHighlightSections(False)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.setShowGrid(False)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

    def use_simulator_tile_cache(self) -> None:
        """Use the exact same on-disk map tile cache as the simulator GUI."""
        try:
            SIMULATOR_TILE_CACHE.mkdir(parents=True, exist_ok=True)
            cache = QNetworkDiskCache(self.map.network)
            cache.setCacheDirectory(str(SIMULATOR_TILE_CACHE))
            cache.setMaximumCacheSize(120 * 1024 * 1024)
            self.map.network.setCache(cache)
            self.map._krakenscope_shared_cache = cache
        except OSError:
            pass

    def retranslate(self) -> None:
        self.setWindowTitle(self.tr_text("title"))
        self.theme_button.setText("Dark mode" if self.theme_mode == "light" else "Light mode")
        self.settings_button.setText("⚙  Settings")
        self.clear_button.setText("↺  Clear")
        self.listener_button.setText("■  Stop listener" if self.listener else "▶  Start listener")
        self.simulator_button.setText(
            "■  Stop simulator"
            if self.simulator.state() != QProcess.ProcessState.NotRunning
            else "↗  Launch simulator"
        )
        self.map_title.setText("LIVE OPERATIONS MAP")
        self.receiver_title.setText("RECEIVER STATUS")
        self.estimate_title.setText("LOCALIZATION SOLUTIONS")
        self.evaluation_title.setText("GROUND-TRUTH EVALUATION")
        self.reset_evaluation_button.setText("Reset")
        self.export_evaluation_button.setText("Export CSV")
        self.latest_title.setText("LATEST UDP PACKET")
        self.receiver_table.setHorizontalHeaderLabels([
            self.tr_text("receiver"),
            self.tr_text("status"),
            self.tr_text("address"),
            self.tr_text("lat"),
            self.tr_text("lon"),
            self.tr_text("packets"),
            self.tr_text("sequence"),
            self.tr_text("targets"),
        ])
        self.estimate_table.setHorizontalHeaderLabels([
            self.tr_text("target"),
            "Position",
            "Samples",
            self.tr_text("rssi"),
            self.tr_text("residual"),
            "Geometry",
            "A / R",
        ])
        self.evaluation_table.setHorizontalHeaderLabels([
            self.tr_text("target"),
            "Tracked",
            "Raw",
            "Δ",
            self.tr_text("rmse"),
            self.tr_text("quality"),
        ])
        if self.listener:
            self.statusBar().showMessage(self.tr_text("listening", bind=self.config["bind_ip"], port=self.config["port"]))
            self.listener_status.setText(f"●  LISTENING  {self.config['bind_ip']}:{self.config['port']}")
            self.listener_status.setProperty("online", True)
        else:
            self.statusBar().showMessage(self.tr_text("stopped"))
            self.listener_status.setText("●  LISTENER OFFLINE")
            self.listener_status.setProperty("online", False)
        self.listener_status.style().unpolish(self.listener_status)
        self.listener_status.style().polish(self.listener_status)

    def start_listener(self) -> None:
        if self.listener:
            self.listener.stop()
        self.clear_session_state(clear_packet=True)
        self.listener = UdpListener(self.config["bind_ip"], self.config["port"])
        self.listener.datagram.connect(self.handle_datagram)
        self.listener.failed.connect(self.listener_failed)
        self.listener.start()
        self.retranslate()

    def clear_session_state(self, clear_packet: bool = True) -> None:
        self.receivers.clear()
        self.estimates.clear()
        self.raw_estimates.clear()
        self.display_estimates.clear()
        self.smoothed_estimates.clear()
        self.estimate_measurements.clear()
        self.sample_rejections.clear()
        self.target_motion.clear()
        self.target_observation_history.clear()
        self.target_bearing_history.clear()
        self.target_bearing_motion_evidence.clear()
        self.target_motion_velocity_history.clear()
        self.motion_mode_candidates.clear()
        self.auto_motion_modes.clear()
        self.static_history_started_ms.clear()
        self.track_states.clear()
        self.track_seed_estimates.clear()
        self.track_recovery_candidates.clear()
        self.static_filter_timestamps.clear()
        self.last_localization_packets.clear()
        self.estimate_trails.clear()
        self.ground_truths.clear()
        self.evaluation_stats.clear()
        self.evaluation_samples.clear()
        self._evaluation_sample_keys.clear()
        self._evaluation_first_timestamps.clear()
        self._experiment_run_iso = datetime.now().isoformat(timespec="seconds")
        self._auto_frame_done = False
        self._auto_frame_pending = False
        if hasattr(self, "map"):
            self.map.krakens = []
            self.map.targets = []
            self.map.signal_paths = []
            self.map.target_trails = {}
            self.map.update()
        if clear_packet and hasattr(self, "latest_packet"):
            self.latest_packet.clear()
            self.latest_packet.setPlaceholderText("Waiting for the first simulator packet…")
        if hasattr(self, "auto_frame_status"):
            self.auto_frame_status.setText("●  AUTO FRAME")

    def clear_session(self) -> None:
        self.clear_session_state(clear_packet=True)
        self.refresh()
        self.statusBar().showMessage("Session cleared")

    def stop_listener(self) -> None:
        if self.listener:
            self.listener.stop()
            self.listener = None
        self.retranslate()

    def toggle_listener(self) -> None:
        if self.listener:
            self.stop_listener()
        else:
            self.start_listener()

    def fit_map_to_data(self, points: list[tuple[float, float]] | None = None) -> None:
        if points is None:
            points = [(item["latitude"], item["longitude"]) for item in self.map.krakens + self.map.targets]
        if not points:
            return
        latitude = sum(point[0] for point in points) / len(points)
        longitude = sum(point[1] for point in points) / len(points)
        lat_span = max(point[0] for point in points) - min(point[0] for point in points)
        lon_span = max(point[1] for point in points) - min(point[1] for point in points)
        span = max(lat_span, lon_span)
        if span < 0.002:
            zoom = 18
        elif span < 0.006:
            zoom = 16
        elif span < 0.02:
            zoom = 15
        elif span < 0.08:
            zoom = 13
        elif span < 0.25:
            zoom = 11
        else:
            zoom = 9
        self.map.set_view(latitude, longitude, zoom)

    def map_frame_points(self) -> list[tuple[float, float]]:
        receiver_points = [
            (item["latitude"], item["longitude"])
            for item in self.map.krakens
        ]
        truth_points = list(self.ground_truths.values())
        tracked_points = [
            (estimate.latitude, estimate.longitude)
            for estimate in self.estimates
        ]
        target_points = truth_points if truth_points else tracked_points
        return receiver_points + target_points

    def recenter_map(self) -> None:
        points = self.map_frame_points()
        if not points:
            self.map.set_view(DEFAULT_LATITUDE, DEFAULT_LONGITUDE, 15)
            self._auto_frame_done = False
            self._auto_frame_pending = False
            self.auto_frame_status.setText("●  AUTO FRAME")
            return
        self._auto_frame_pending = False
        self.fit_map_to_data(points)
        self._auto_frame_done = True
        self.auto_frame_status.setText("●  FRAME LOCKED")

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.config, self.tr_text, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        old_endpoint = (self.config["bind_ip"], self.config["port"])
        self.config.update(dialog.values())
        for key, value in self.config.items():
            self.settings.setValue(key, value)
        if self.listener and old_endpoint != (self.config["bind_ip"], self.config["port"]):
            self.start_listener()
        self.retranslate()

    def toggle_raw_debug(self, enabled: bool) -> None:
        self.show_raw_debug = enabled
        self.settings.setValue("show_raw_debug", enabled)
        if self.raw_legend_label is not None:
            self.raw_legend_label.setVisible(enabled)
        self.update_map()

    def toggle_simulator(self) -> None:
        if self.simulator.state() != QProcess.ProcessState.NotRunning:
            self.simulator.terminate()
            QTimer.singleShot(1500, self._kill_simulator_if_needed)
            self.retranslate()
            return
        if not SIMULATOR_GUI.exists():
            QMessageBox.critical(self, "KrakenScope", self.tr_text("sim_error"))
            return
        self.simulator.start(sys.executable, [str(SIMULATOR_GUI)])
        if not self.simulator.waitForStarted(1500):
            QMessageBox.critical(self, "KrakenScope", self.tr_text("sim_error"))
        self.simulator.finished.connect(lambda *_args: self.retranslate())
        self.retranslate()

    def _kill_simulator_if_needed(self) -> None:
        if self.simulator.state() != QProcess.ProcessState.NotRunning:
            self.simulator.kill()
        self.retranslate()

    def handle_datagram(self, data: bytes, address: tuple[str, int], received_at: float) -> None:
        try:
            packet = parse_packet(data)
        except PacketError as exc:
            self.statusBar().showMessage(str(exc), 4000)
            return

        packet = self._with_manual_real_receiver_location(packet, address)
        key = receiver_key(packet, address[0], address[1])
        state = self.receivers.get(key)
        if state is None:
            state = ReceiverState(key=key, display_name=key)
            self.receivers[key] = state
        elif self._packet_starts_new_scenario(state, packet):
            self.reset_measurement_history()
        state.add_packet(packet, address, received_at)

        try:
            pretty = json.dumps(json.loads(data.decode("utf-8")), indent=2, ensure_ascii=False)
        except (UnicodeDecodeError, json.JSONDecodeError):
            pretty = data[:2000].decode("utf-8", errors="replace")
        self.latest_packet.setPlainText(pretty)

        if packet.receiver_latitude is None or packet.receiver_longitude is None:
            self.statusBar().showMessage(self.tr_text("no_location", name=key), 5000)
        self.recompute_localization()
        self.refresh()

    def _with_manual_real_receiver_location(self, packet, address: tuple[str, int]):
        if packet.receiver_latitude is not None and packet.receiver_longitude is not None:
            return packet
        slot = self._matching_manual_receiver_slot(packet, address)
        if slot is None:
            return packet
        return replace(
            packet,
            kraken_name=slot["name"],
            receiver_latitude=slot["latitude"],
            receiver_longitude=slot["longitude"],
        )

    def _matching_manual_receiver_slot(self, packet, address: tuple[str, int]) -> dict | None:
        slots = self._manual_receiver_slots()
        if not slots:
            return None

        source_ip = address[0].strip().lower()
        source_full = f"{address[0]}:{address[1]}".strip().lower()
        packet_names = {
            str(value).strip().lower()
            for value in (packet.kraken_name, packet.source)
            if value
        }
        for slot in slots:
            slot_source = slot["source"].strip().lower()
            slot_name = slot["name"].strip().lower()
            if slot_name in packet_names:
                return slot
            if slot_source and slot_source in {source_ip, source_full}:
                return slot

        unfiltered_slots = [slot for slot in slots if not slot["source"].strip()]
        if len(slots) == 1 and len(unfiltered_slots) == 1:
            return unfiltered_slots[0]
        return None

    def _manual_receiver_slots(self) -> list[dict]:
        slots = []
        for index in (1, 2):
            if not self.config.get(f"real_receiver_{index}_enabled", False):
                continue
            latitude = float(self.config.get(f"real_receiver_{index}_latitude", DEFAULT_LATITUDE))
            longitude = float(self.config.get(f"real_receiver_{index}_longitude", DEFAULT_LONGITUDE))
            if not math.isfinite(latitude) or not math.isfinite(longitude):
                continue
            slots.append(
                {
                    "name": str(
                        self.config.get(f"real_receiver_{index}_name", f"Real-Kraken-{index}")
                    ).strip()
                    or f"Real-Kraken-{index}",
                    "source": str(self.config.get(f"real_receiver_{index}_source", "")).strip(),
                    "latitude": latitude,
                    "longitude": longitude,
                }
            )
        return slots

    def _packet_starts_new_scenario(self, state: ReceiverState, packet) -> bool:
        if state.latitude is not None and state.longitude is not None:
            if packet.receiver_latitude is not None and packet.receiver_longitude is not None:
                if distance_meters(
                    state.latitude,
                    state.longitude,
                    packet.receiver_latitude,
                    packet.receiver_longitude,
                ) > 1.0:
                    return True
        if state.targets and len(state.targets) != len(packet.targets):
            return True
        for index, new_target in enumerate(packet.targets):
            if index >= len(state.targets):
                continue
            previous_target = state.targets[index]
            previous_name = previous_target.name or f"Target-{index + 1}"
            new_name = new_target.name or f"Target-{index + 1}"
            if previous_name != new_name:
                return True
        return False

    def reset_measurement_history(self) -> None:
        for receiver in self.receivers.values():
            receiver.history.clear()
        self.estimates.clear()
        self.raw_estimates.clear()
        self.display_estimates.clear()
        self.smoothed_estimates.clear()
        self.estimate_measurements.clear()
        self.sample_rejections.clear()
        self.target_motion.clear()
        self.target_observation_history.clear()
        self.target_bearing_history.clear()
        self.target_bearing_motion_evidence.clear()
        self.target_motion_velocity_history.clear()
        self.motion_mode_candidates.clear()
        self.auto_motion_modes.clear()
        self.static_history_started_ms.clear()
        self.track_states.clear()
        self.track_seed_estimates.clear()
        self.track_recovery_candidates.clear()
        self.static_filter_timestamps.clear()
        self.last_localization_packets.clear()
        self.estimate_trails.clear()
        self.ground_truths.clear()
        self.evaluation_stats.clear()
        self.evaluation_samples.clear()
        self._evaluation_sample_keys.clear()
        self._evaluation_first_timestamps.clear()
        self._experiment_run_iso = datetime.now().isoformat(timespec="seconds")
        self._auto_frame_done = False
        self._auto_frame_pending = False
        if hasattr(self, "auto_frame_status"):
            self.auto_frame_status.setText("●  AUTO FRAME")

    def recompute_localization(self) -> None:
        now = time.time()
        usable = [
            state
            for state in self.receivers.values()
            if state.latitude is not None
            and state.longitude is not None
            and state.targets
            and state.online(now, self.config["timeout"])
        ]
        newest_timestamp = self._newest_target_timestamp(usable)
        max_targets = max((len(state.targets) for state in usable), default=0)
        estimates: list[EstimatedTarget] = []
        raw_cycle_estimates: dict[str, EstimatedTarget] = {}
        estimate_measurements: dict[str, list[BearingMeasurement]] = {}
        sample_rejections: dict[str, tuple[int, int]] = {}
        target_motion: dict[str, dict] = {}
        completed_cycles: dict[str, float] = {}

        for target_index in range(max_targets):
            current_measurements: list[BearingMeasurement] = []
            cycle_packet_tokens: dict[str, int] = {}
            target_name = f"Target-{target_index + 1}"
            target_timestamps: list[float] = []
            for state in usable:
                if target_index >= len(state.targets):
                    continue
                target = state.targets[target_index]
                if target.name:
                    target_name = target.name
                if newest_timestamp is not None and target.timestamp_ms is not None:
                    delta_seconds = abs(target.timestamp_ms - newest_timestamp) / 1000.0
                    if delta_seconds > self.config["cycle_window"]:
                        continue
                    target_timestamps.append(target.timestamp_ms)
                current_measurements.append(
                    BearingMeasurement(
                        receiver_name=state.display_name,
                        receiver_latitude=state.latitude,
                        receiver_longitude=state.longitude,
                        doa_degrees=target.doa_degrees % 360.0,
                        rssi_db=target.rssi_db,
                    )
                )
                cycle_packet_tokens[state.key] = state.packet_count
            cycle_is_complete = (
                newest_timestamp is None
                or (
                    len(current_measurements) == len(usable)
                    and (
                        not target_timestamps
                        or len(target_timestamps) == len(current_measurements)
                    )
                )
            )
            previous_cycle_tokens = self.last_localization_packets.get(target_name, {})
            cycle_is_fresh = bool(cycle_packet_tokens) and all(
                packet_count > previous_cycle_tokens.get(receiver_key, -1)
                for receiver_key, packet_count in cycle_packet_tokens.items()
            )
            # Do not feed mixed old/new receiver packets into localization.
            # Every active receiver must have supplied a new packet since the
            # previous solution. This prevents one mixed old/new solution and
            # a second duplicate solution being generated for the same cycle.
            # Exact timestamps are intentionally not required because hybrid
            # real/simulated receivers use independent clocks.
            if not cycle_is_complete or not cycle_is_fresh:
                continue
            raw_cycle_estimate = estimate_target(target_name, current_measurements)
            if raw_cycle_estimate is None:
                continue
            raw_cycle_estimates[target_name] = raw_cycle_estimate
            previous_mode = self.auto_motion_modes.get(target_name, False)
            motion_analysis = self._auto_detect_target_motion(
                target_name,
                newest_timestamp,
                raw_cycle_estimate,
                current_measurements,
            )
            moving = motion_analysis.moving
            if previous_mode is not None and previous_mode != moving:
                previous_filtered = self.smoothed_estimates.get(target_name)
                self.track_states.pop(target_name, None)
                self.track_recovery_candidates.pop(target_name, None)
                self.static_filter_timestamps.pop(target_name, None)
                self.estimate_trails.pop(target_name, None)
                if moving:
                    self.smoothed_estimates.pop(target_name, None)
                    if previous_filtered is not None:
                        self.track_seed_estimates[target_name] = previous_filtered
                    self.static_history_started_ms.pop(target_name, None)
                else:
                    self.track_seed_estimates.pop(target_name, None)
                    self.static_history_started_ms[target_name] = (
                        newest_timestamp or time.time() * 1000.0
                    )
            target_motion[target_name] = {
                "moving": moving,
                "speed_mps": motion_analysis.speed_mps,
                "heading_degrees": motion_analysis.heading_degrees,
                "confidence": motion_analysis.confidence,
                "auto_detected": True,
                "scatter_meters": motion_analysis.scatter_meters,
                "reference_latitude": motion_analysis.projected_latitude,
                "reference_longitude": motion_analysis.projected_longitude,
                "bearing_angular_rate_dps": motion_analysis.bearing_angular_rate_dps,
                "bearing_motion_confidence": motion_analysis.bearing_motion_confidence,
            }
            measurements: list[BearingMeasurement] = []
            accepted_sample_count = 0
            rejected_sample_count = 0
            if moving:
                # Moving localization uses only the current synchronized DoA
                # cycle. Old bearings describe old target positions.
                measurements = current_measurements
                accepted_sample_count = len(measurements)
            else:
                for state in usable:
                    receiver_measurements, accepted, rejected = (
                        self._static_bearing_measurements(
                            state,
                            target_index,
                            self.static_history_started_ms.get(target_name),
                        )
                    )
                    measurements.extend(receiver_measurements)
                    accepted_sample_count += accepted
                    rejected_sample_count += rejected
            estimate = (
                raw_cycle_estimate
                if moving
                else estimate_static_target(target_name, measurements)
            )
            if estimate is not None:
                estimate = EstimatedTarget(
                    name=estimate.name,
                    latitude=estimate.latitude,
                    longitude=estimate.longitude,
                    measurement_count=accepted_sample_count,
                    mean_rssi_db=estimate.mean_rssi_db,
                    residual_meters=estimate.residual_meters,
                    geometry_score=estimate.geometry_score,
                    geometry_label=estimate.geometry_label,
                )
                estimates.append(estimate)
                # Keep only the current bearing lines on the map. The static
                # batch may contain hundreds of historical angular samples.
                estimate_measurements[estimate.name] = current_measurements
                sample_rejections[estimate.name] = (
                    accepted_sample_count,
                    rejected_sample_count,
                )
                if newest_timestamp is not None:
                    completed_cycles[estimate.name] = newest_timestamp
                self.last_localization_packets[estimate.name] = cycle_packet_tokens

        if newest_timestamp is not None and not estimates:
            return
        self.raw_estimates = raw_cycle_estimates
        self.target_motion = target_motion
        self.estimate_measurements = estimate_measurements
        # Install the per-cycle counts before filtering so the moving EKF can
        # replace them with its actual innovation-gating decisions.
        self.sample_rejections = sample_rejections
        self.estimates = self.filter_estimates(estimates, newest_timestamp)
        # Evaluation metadata is deliberately collected only after the radio-
        # measurement localization pipeline has completed.
        cycle_truths = self._collect_evaluation_truths(usable, newest_timestamp)
        self.ground_truths = cycle_truths
        self._update_estimate_trails()
        for estimate in self.estimates:
            truth = cycle_truths.get(estimate.name)
            timestamp = completed_cycles.get(estimate.name)
            if truth is None or timestamp is None:
                continue
            tracked_error = distance_meters(
                estimate.latitude,
                estimate.longitude,
                truth[0],
                truth[1],
            )
            raw_estimate = self.raw_estimates.get(estimate.name)
            raw_error = (
                distance_meters(
                    raw_estimate.latitude,
                    raw_estimate.longitude,
                    truth[0],
                    truth[1],
                )
                if raw_estimate is not None
                else None
            )
            self.evaluation_stats.setdefault(estimate.name, EvaluationStats()).add(
                tracked_error,
                timestamp,
                raw_error,
            )
            self._record_evaluation_sample(
                estimate=estimate,
                raw_estimate=raw_estimate,
                truth=truth,
                timestamp_ms=timestamp,
                tracked_error=tracked_error,
                raw_error=raw_error,
                active_receiver_count=len(usable),
            )
        self.update_map()

    def _collect_evaluation_truths(
        self,
        states: list[ReceiverState],
        newest_timestamp: float | None,
    ) -> dict[str, tuple[float, float]]:
        truths: dict[str, tuple[float, float]] = {}
        truth_timestamps: dict[str, float] = {}
        for state in states:
            for index, target in enumerate(state.targets):
                if target.truth_latitude is None or target.truth_longitude is None:
                    continue
                if newest_timestamp is not None and target.timestamp_ms is not None:
                    if abs(target.timestamp_ms - newest_timestamp) / 1000.0 > self.config["cycle_window"]:
                        continue
                target_name = target.name or f"Target-{index + 1}"
                truth_timestamp = (
                    target.timestamp_ms
                    if target.timestamp_ms is not None
                    else newest_timestamp or 0.0
                )
                if truth_timestamp < truth_timestamps.get(target_name, -math.inf):
                    continue
                truths[target_name] = (
                    target.truth_latitude,
                    target.truth_longitude,
                )
                truth_timestamps[target_name] = truth_timestamp
        return truths

    def _record_evaluation_sample(
        self,
        estimate: EstimatedTarget,
        raw_estimate: EstimatedTarget | None,
        truth: tuple[float, float],
        timestamp_ms: float,
        tracked_error: float,
        raw_error: float | None,
        active_receiver_count: int,
    ) -> None:
        key = (estimate.name, timestamp_ms)
        if key in self._evaluation_sample_keys:
            return
        self._evaluation_sample_keys.add(key)
        first_timestamp = self._evaluation_first_timestamps.setdefault(
            estimate.name,
            timestamp_ms,
        )
        elapsed_seconds = max(0.0, (timestamp_ms - first_timestamp) / 1000.0)
        motion = self.target_motion.get(estimate.name, {})
        moving = bool(motion.get("moving", False))
        accepted, rejected = self.sample_rejections.get(
            estimate.name,
            (estimate.measurement_count, 0),
        )
        row = {
            "run_iso": self._experiment_run_iso,
            "target": estimate.name,
            "timestamp_ms": f"{timestamp_ms:.3f}",
            "elapsed_s": f"{elapsed_seconds:.3f}",
            "mode": "AUTO TRACK" if moving else "AUTO STATIC",
            "moving": int(moving),
            "speed_mps": f"{float(motion.get('speed_mps', 0.0) or 0.0):.3f}",
            "heading_degrees": f"{float(motion.get('heading_degrees', 0.0) or 0.0):.3f}",
            "bearing_angular_rate_dps": f"{float(motion.get('bearing_angular_rate_dps', 0.0) or 0.0):.4f}",
            "bearing_motion_confidence": f"{float(motion.get('bearing_motion_confidence', 0.0) or 0.0):.4f}",
            "motion_scatter_m": f"{float(motion.get('scatter_meters', 0.0) or 0.0):.3f}",
            "tracker_recovered": int(bool(motion.get("tracker_recovered", False))),
            "raw_reference_gap_m": f"{float(motion.get('raw_reference_gap_m', 0.0) or 0.0):.3f}",
            "tracker_reference_gap_m": f"{float(motion.get('tracker_reference_gap_m', 0.0) or 0.0):.3f}",
            "truth_latitude": f"{truth[0]:.9f}",
            "truth_longitude": f"{truth[1]:.9f}",
            "tracked_latitude": f"{estimate.latitude:.9f}",
            "tracked_longitude": f"{estimate.longitude:.9f}",
            "raw_latitude": "" if raw_estimate is None else f"{raw_estimate.latitude:.9f}",
            "raw_longitude": "" if raw_estimate is None else f"{raw_estimate.longitude:.9f}",
            "tracked_error_m": f"{tracked_error:.3f}",
            "raw_error_m": "" if raw_error is None else f"{raw_error:.3f}",
            "filter_delta_m": "" if raw_error is None else f"{tracked_error - raw_error:.3f}",
            "residual_m": f"{estimate.residual_meters:.3f}",
            "geometry_score": f"{estimate.geometry_score:.3f}",
            "geometry_label": estimate.geometry_label,
            "mean_rssi_db": f"{estimate.mean_rssi_db:.3f}",
            "measurement_count": estimate.measurement_count,
            "accepted_samples": accepted,
            "rejected_samples": rejected,
            "active_receivers": active_receiver_count,
            "cycle_window_s": f"{self.config['cycle_window']:.3f}",
        }
        self.evaluation_samples.append(row)

    def _auto_detect_target_motion(
        self,
        target_name: str,
        timestamp_ms: float | None,
        raw_estimate: EstimatedTarget,
        measurements: list[BearingMeasurement],
    ) -> MotionAnalysis:
        # Ground truth and simulator motion metadata are deliberately excluded.
        # Mode, speed and heading come only from the sequence of raw DoA-based
        # localization solutions, which is also available with real Krakens.
        observation_timestamp = (
            timestamp_ms
            if timestamp_ms is not None and math.isfinite(timestamp_ms)
            else time.monotonic() * 1000.0
        )
        history = self.target_observation_history.setdefault(target_name, [])
        bearing_histories = self.target_bearing_history.setdefault(target_name, {})
        for measurement in measurements:
            receiver_history = bearing_histories.setdefault(
                measurement.receiver_name,
                [],
            )
            if not receiver_history or observation_timestamp > receiver_history[-1][0]:
                receiver_history.append(
                    (observation_timestamp, measurement.doa_degrees % 360.0)
                )
            bearing_cutoff = observation_timestamp - 20_000.0
            receiver_history[:] = [
                sample for sample in receiver_history if sample[0] >= bearing_cutoff
            ]
        if history and observation_timestamp <= history[-1][0]:
            return self._motion_consensus_analysis(target_name, observation_timestamp)

        if history:
            elapsed_seconds = max(0.001, (observation_timestamp - history[-1][0]) / 1000.0)
            if elapsed_seconds > max(5.0, self.config["timeout"] * 2.0):
                history.clear()
                bearing_histories.clear()
                self.target_bearing_motion_evidence.pop(target_name, None)
                self.target_motion_velocity_history.pop(target_name, None)
                self.motion_mode_candidates.pop(target_name, None)
                self.auto_motion_modes[target_name] = False
            jump_meters = distance_meters(
                history[-1][1] if history else raw_estimate.latitude,
                history[-1][2] if history else raw_estimate.longitude,
                raw_estimate.latitude,
                raw_estimate.longitude,
            )
            # Reject an isolated physically implausible raw jump from the mode
            # detector. It remains available as Raw and is still gated by the
            # tracker, but cannot flip STATIC/TRACK on its own.
            if history and jump_meters > max(40.0, 45.0 * elapsed_seconds):
                return self._motion_consensus_analysis(target_name, observation_timestamp)

        history.append(
            (
                observation_timestamp,
                raw_estimate.latitude,
                raw_estimate.longitude,
            )
        )
        cutoff = observation_timestamp - 12_000.0
        history[:] = [sample for sample in history if sample[0] >= cutoff]
        return self._motion_consensus_analysis(target_name, observation_timestamp)

    def _motion_consensus_analysis(
        self,
        target_name: str,
        observation_timestamp: float,
    ) -> MotionAnalysis:
        history = self.target_observation_history.get(target_name, [])
        current_mode = self.auto_motion_modes.get(target_name, False)
        base = analyze_observed_motion(history, False, lookback_seconds=8.0)
        bearing_motion = analyze_bearing_motion(
            self.target_bearing_history.get(target_name, {}),
            lookback_seconds=8.0 if current_mode else 12.0,
        )
        bearing_evidence = self.target_bearing_motion_evidence.setdefault(
            target_name,
            [],
        )
        if bearing_motion.span_seconds >= 4.0:
            bearing_evidence.append(
                (
                    observation_timestamp,
                    bearing_motion.angular_rate_dps,
                    bearing_motion.confidence,
                )
            )
        bearing_evidence_cutoff = observation_timestamp - (
            10_000.0 if current_mode else 8_000.0
        )
        bearing_evidence[:] = [
            sample for sample in bearing_evidence if sample[0] >= bearing_evidence_cutoff
        ]
        evidence = self.target_motion_velocity_history.setdefault(target_name, [])
        if base.span_seconds >= 3.5 and (
            not evidence or observation_timestamp > evidence[-1][0]
        ):
            heading_radians = math.radians(base.heading_degrees)
            evidence.append(
                (
                    observation_timestamp,
                    base.speed_mps * math.sin(heading_radians),
                    base.speed_mps * math.cos(heading_radians),
                )
            )
        # Entry is deliberately conservative; once TRACK is established, a
        # shorter evidence window recognizes that the target has stopped.
        evidence_window_ms = 10_000.0 if current_mode else 20_000.0
        evidence_cutoff = observation_timestamp - evidence_window_ms
        evidence[:] = [sample for sample in evidence if sample[0] >= evidence_cutoff]

        if len(evidence) < 2:
            return MotionAnalysis(current_mode, span_seconds=base.span_seconds)

        # Reduce overlapping velocity estimates to fixed time buckets so a
        # shorter packet interval cannot gain extra voting power.
        buckets: dict[int, list[tuple[float, float, float]]] = {}
        for timestamp, east_velocity, north_velocity in evidence:
            bucket = int(math.floor(timestamp / 750.0))
            buckets.setdefault(bucket, []).append(
                (timestamp, east_velocity, north_velocity)
            )
        velocity_points = sorted(
            (
                statistics.median(sample[0] for sample in samples),
                statistics.median(sample[1] for sample in samples),
                statistics.median(sample[2] for sample in samples),
            )
            for samples in buckets.values()
        )
        east_velocity = statistics.median(point[1] for point in velocity_points)
        north_velocity = statistics.median(point[2] for point in velocity_points)
        speed_mps = math.hypot(east_velocity, north_velocity)
        heading_degrees = math.degrees(math.atan2(east_velocity, north_velocity)) % 360.0

        bearing_buckets: dict[int, list[tuple[float, float, float]]] = {}
        for timestamp, angular_rate, bearing_confidence in bearing_evidence:
            bearing_buckets.setdefault(int(math.floor(timestamp / 750.0)), []).append(
                (timestamp, angular_rate, bearing_confidence)
            )
        bearing_votes = sorted(
            (
                statistics.median(sample[0] for sample in samples),
                statistics.median(sample[1] for sample in samples),
                statistics.median(sample[2] for sample in samples),
            )
            for samples in bearing_buckets.values()
        )
        bearing_vote_span = (
            (bearing_votes[-1][0] - bearing_votes[0][0]) / 1000.0
            if len(bearing_votes) >= 2
            else 0.0
        )
        entry_fraction = (
            sum(rate >= 0.75 and item_confidence >= 0.50 for _time, rate, item_confidence in bearing_votes)
            / len(bearing_votes)
            if bearing_votes
            else 0.0
        )
        stay_fraction = (
            sum(rate >= 0.55 and item_confidence >= 0.20 for _time, rate, item_confidence in bearing_votes)
            / len(bearing_votes)
            if bearing_votes
            else 0.0
        )
        if current_mode:
            has_enough_evidence = (
                bearing_motion.receiver_count >= 2
                and bearing_vote_span >= 4.0
                and len(bearing_votes) >= 6
            )
            candidate_mode = (
                stay_fraction >= 0.35 if has_enough_evidence else current_mode
            )
        else:
            has_enough_evidence = (
                bearing_motion.receiver_count >= 2
                and bearing_vote_span >= 4.0
                and len(bearing_votes) >= 6
            )
            candidate_mode = (
                entry_fraction >= 0.50 if has_enough_evidence else current_mode
            )
        confidence = max(
            base.confidence,
            bearing_motion.confidence,
        )

        if candidate_mode == current_mode:
            self.motion_mode_candidates.pop(target_name, None)
        else:
            candidate = self.motion_mode_candidates.get(target_name)
            if candidate is None or candidate[0] != candidate_mode:
                self.motion_mode_candidates[target_name] = (
                    candidate_mode,
                    observation_timestamp,
                )
            else:
                hold_seconds = 2.0
                held_seconds = (observation_timestamp - candidate[1]) / 1000.0
                if held_seconds >= hold_seconds:
                    current_mode = candidate_mode
                    self.auto_motion_modes[target_name] = current_mode
                    self.motion_mode_candidates.pop(target_name, None)
                    self.target_bearing_motion_evidence.pop(target_name, None)

        return MotionAnalysis(
            moving=current_mode,
            speed_mps=speed_mps if current_mode else 0.0,
            heading_degrees=heading_degrees if current_mode else 0.0,
            confidence=confidence,
            span_seconds=base.span_seconds,
            scatter_meters=base.scatter_meters,
            displacement_meters=base.displacement_meters,
            endpoint_progress_meters=base.endpoint_progress_meters,
            projected_latitude=base.projected_latitude,
            projected_longitude=base.projected_longitude,
            bearing_angular_rate_dps=bearing_motion.angular_rate_dps,
            bearing_motion_confidence=bearing_motion.confidence,
        )

    def _static_bearing_measurements(
        self,
        state: ReceiverState,
        target_index: int,
        history_started_ms: float | None = None,
    ) -> tuple[list[BearingMeasurement], int, int]:
        samples = [
            (
                target.timestamp_ms / 1000.0
                if target.timestamp_ms is not None
                else received_at,
                target.doa_degrees % 360.0,
                target.rssi_db,
            )
            for received_at, index, target in state.history
            if index == target_index
            and (
                history_started_ms is None
                or (
                    target.timestamp_ms
                    if target.timestamp_ms is not None
                    else received_at * 1000.0
                )
                >= history_started_ms
            )
        ]
        if not samples or state.latitude is None or state.longitude is None:
            return [], 0, 0
        bucketed = bucket_timestamped_bearings(samples)
        measurements = [
            BearingMeasurement(
                receiver_name=state.display_name,
                receiver_latitude=state.latitude,
                receiver_longitude=state.longitude,
                doa_degrees=bucket.doa_degrees,
                rssi_db=bucket.rssi_db,
            )
            for bucket in bucketed
        ]
        accepted = sum(bucket.accepted_samples for bucket in bucketed)
        rejected = sum(bucket.rejected_samples for bucket in bucketed)
        return measurements, accepted, rejected

    def _update_estimate_trails(self) -> None:
        active_names = {estimate.name for estimate in self.estimates}
        for stale_name in set(self.estimate_trails) - active_names:
            del self.estimate_trails[stale_name]
        for estimate in self.estimates:
            if not self.target_motion.get(estimate.name, {}).get("moving", False):
                self.estimate_trails.pop(estimate.name, None)
                continue
            trail = self.estimate_trails.setdefault(estimate.name, [])
            point = (estimate.latitude, estimate.longitude)
            if not trail or distance_meters(*trail[-1], *point) >= 1.0:
                trail.append(point)
                del trail[:-250]

    def reset_evaluation(self) -> None:
        self.evaluation_stats.clear()
        self.evaluation_samples.clear()
        self._evaluation_sample_keys.clear()
        self._evaluation_first_timestamps.clear()
        self._experiment_run_iso = datetime.now().isoformat(timespec="seconds")
        self._refresh_evaluation()

    def export_evaluation_csv(self) -> None:
        if not self.evaluation_samples:
            QMessageBox.information(
                self,
                "KrakenScope",
                "No ground-truth evaluation samples have been recorded yet.",
            )
            return
        experiments_dir = ROOT / "experiments"
        experiments_dir.mkdir(exist_ok=True)
        default_name = (
            "krakenscope_experiment_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export experiment CSV",
            str(experiments_dir / default_name),
            "CSV files (*.csv)",
        )
        if not path:
            return
        output_path = Path(path)
        if output_path.suffix.lower() != ".csv":
            output_path = output_path.with_suffix(".csv")
        try:
            with output_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=EXPERIMENT_FIELDS)
                writer.writeheader()
                writer.writerows(self.evaluation_samples)
        except OSError as exc:
            QMessageBox.critical(
                self,
                "KrakenScope",
                f"Could not export experiment CSV:\n{exc}",
            )
            return
        QMessageBox.information(
            self,
            "KrakenScope",
            f"Exported {len(self.evaluation_samples)} samples to:\n{output_path}",
        )

    def filter_estimates(
        self, estimates: list[EstimatedTarget], timestamp_ms: float | None
    ) -> list[EstimatedTarget]:
        smoothed: list[EstimatedTarget] = []
        active_names = {estimate.name for estimate in estimates}
        for stale_name in set(self.smoothed_estimates) - active_names:
            del self.smoothed_estimates[stale_name]
        for stale_name in set(self.track_states) - active_names:
            del self.track_states[stale_name]
        for stale_name in set(self.track_seed_estimates) - active_names:
            del self.track_seed_estimates[stale_name]
        for stale_name in set(self.track_recovery_candidates) - active_names:
            del self.track_recovery_candidates[stale_name]
        for stale_name in set(self.static_filter_timestamps) - active_names:
            del self.static_filter_timestamps[stale_name]
        for estimate in estimates:
            if self.target_motion.get(estimate.name, {}).get("moving", False):
                self.static_filter_timestamps.pop(estimate.name, None)
                current = self._track_moving_estimate(estimate, timestamp_ms)
                self.smoothed_estimates[estimate.name] = current
                smoothed.append(current)
                continue
            self.track_states.pop(estimate.name, None)
            self.track_seed_estimates.pop(estimate.name, None)
            self.track_recovery_candidates.pop(estimate.name, None)
            previous = self.smoothed_estimates.get(estimate.name)
            if previous is None:
                current = estimate
            elif (
                estimate.name in self.static_history_started_ms
                and estimate.measurement_count < 6
            ):
                # On TRACK -> STATIC, keep the last stable tracked position
                # until the new static batch has at least three bearings per
                # receiver. This prevents an underdetermined first-cycle batch
                # from causing a visible kilometre-scale jump.
                current = previous
            else:
                current_timestamp = (
                    timestamp_ms
                    if timestamp_ms is not None and math.isfinite(timestamp_ms)
                    else time.monotonic() * 1000.0
                )
                previous_timestamp = self.static_filter_timestamps.get(estimate.name)
                if previous_timestamp is None:
                    delta_time = 1.0
                else:
                    delta_time = max(0.0, (current_timestamp - previous_timestamp) / 1000.0)
                # A 1.2 s time constant gives the same smoothing per real
                # second whether packets arrive at 10 Hz, 5 Hz or 1 Hz.
                alpha = continuous_time_alpha(delta_time, 1.2)
                if alpha <= 0.0:
                    smoothed.append(previous)
                    continue
                current = EstimatedTarget(
                    name=estimate.name,
                    latitude=previous.latitude * (1.0 - alpha) + estimate.latitude * alpha,
                    longitude=previous.longitude * (1.0 - alpha) + estimate.longitude * alpha,
                    measurement_count=estimate.measurement_count,
                    mean_rssi_db=previous.mean_rssi_db * (1.0 - alpha) + estimate.mean_rssi_db * alpha,
                    residual_meters=previous.residual_meters * (1.0 - alpha) + estimate.residual_meters * alpha,
                    geometry_score=estimate.geometry_score,
                    geometry_label=estimate.geometry_label,
                )
            current_timestamp = (
                timestamp_ms
                if timestamp_ms is not None and math.isfinite(timestamp_ms)
                else time.monotonic() * 1000.0
            )
            self.static_filter_timestamps[estimate.name] = current_timestamp
            self.smoothed_estimates[estimate.name] = current
            smoothed.append(current)
        return smoothed

    def _track_moving_estimate(
        self, estimate: EstimatedTarget, timestamp_ms: float | None
    ) -> EstimatedTarget:
        measurements = self.estimate_measurements.get(estimate.name, [])
        if len(measurements) < 2:
            return estimate
        current_timestamp = (
            timestamp_ms
            if timestamp_ms is not None and math.isfinite(timestamp_ms)
            else time.monotonic() * 1000.0
        )
        motion = self.target_motion.get(estimate.name, {})
        motion["tracker_recovered"] = False
        observed_speed = max(0.0, float(motion.get("speed_mps", 0.0) or 0.0))
        observed_heading = math.radians(
            float(motion.get("heading_degrees", 0.0) or 0.0) % 360.0
        )
        observed_east_velocity = min(15.0, observed_speed) * math.sin(
            observed_heading
        )
        observed_north_velocity = min(15.0, observed_speed) * math.cos(
            observed_heading
        )
        reference_estimate: EstimatedTarget | None = None
        reference_latitude = motion.get("reference_latitude")
        reference_longitude = motion.get("reference_longitude")
        if (
            reference_latitude is not None
            and reference_longitude is not None
            and math.isfinite(float(reference_latitude))
            and math.isfinite(float(reference_longitude))
        ):
            reference_estimate = EstimatedTarget(
                name=estimate.name,
                latitude=float(reference_latitude),
                longitude=float(reference_longitude),
                measurement_count=estimate.measurement_count,
                mean_rssi_db=estimate.mean_rssi_db,
                residual_meters=estimate.residual_meters,
                geometry_score=estimate.geometry_score,
                geometry_label=estimate.geometry_label,
            )
        state = self.track_states.get(estimate.name)
        if state is None:
            seed_estimate = self.track_seed_estimates.pop(estimate.name, None)
            if reference_estimate is not None:
                # Prefer the robust projected radio history when it agrees
                # with the last stable STATIC solution. A disagreement of
                # hundreds of metres means the raw-motion fit was corrupted.
                if seed_estimate is None or distance_meters(
                    seed_estimate.latitude,
                    seed_estimate.longitude,
                    reference_estimate.latitude,
                    reference_estimate.longitude,
                ) <= 250.0:
                    seed_estimate = reference_estimate
            if seed_estimate is None:
                seed_estimate = estimate
            state = initialize_bearing_track(
                seed_estimate,
                measurements,
                current_timestamp,
                east_velocity_mps=observed_east_velocity,
                north_velocity_mps=observed_north_velocity,
            )
            self.track_states[estimate.name] = state
            self.sample_rejections[estimate.name] = (len(measurements), 0)
            return seed_estimate

        delta_time = max(0.0, (current_timestamp - state.timestamp_ms) / 1000.0)
        acceleration_sigma = max(
            1.0,
            min(4.0, 1.0 + 0.04 * math.hypot(
                state.east_velocity_mps,
                state.north_velocity_mps,
            )),
        )
        updated, tracked, accepted, rejected = update_bearing_track(
            state,
            estimate.name,
            measurements,
            current_timestamp,
            acceleration_sigma_mps2=acceleration_sigma,
            maximum_speed_mps=35.0,
        )
        scatter_meters = max(
            0.0,
            float(motion.get("scatter_meters", 0.0) or 0.0),
        )
        reference_is_consistent = (
            reference_estimate is not None
            and estimate.geometry_score >= 45.0
            and scatter_meters <= 80.0
            and distance_meters(
                estimate.latitude,
                estimate.longitude,
                reference_estimate.latitude,
                reference_estimate.longitude,
            ) <= 150.0
        )
        raw_reference_gap = (
            distance_meters(
                estimate.latitude,
                estimate.longitude,
                reference_estimate.latitude,
                reference_estimate.longitude,
            )
            if reference_estimate is not None
            else 0.0
        )
        tracker_reference_gap = (
            distance_meters(
                tracked.latitude,
                tracked.longitude,
                reference_estimate.latitude,
                reference_estimate.longitude,
            )
            if reference_estimate is not None
            else 0.0
        )
        motion["raw_reference_gap_m"] = raw_reference_gap
        motion["tracker_reference_gap_m"] = tracker_reference_gap
        if (
            observed_speed <= 35.0
            and float(motion.get("confidence", 0.0) or 0.0) >= 0.15
            and delta_time > 0.0
        ):
            velocity_alpha = continuous_time_alpha(delta_time, 3.0)
            updated.east_velocity_mps = (
                updated.east_velocity_mps * (1.0 - velocity_alpha)
                + observed_east_velocity * velocity_alpha
            )
            updated.north_velocity_mps = (
                updated.north_velocity_mps * (1.0 - velocity_alpha)
                + observed_north_velocity * velocity_alpha
            )

        reference_limit = max(
            120.0,
            min(
                250.0,
                80.0
                + 3.0 * scatter_meters
                + 3.0 * min(20.0, observed_speed),
            ),
        )
        strong_recovery_reference = (
            reference_estimate is not None
            and estimate.geometry_score >= 65.0
            and scatter_meters <= 60.0
            and distance_meters(
                estimate.latitude,
                estimate.longitude,
                reference_estimate.latitude,
                reference_estimate.longitude,
            ) <= 150.0
        )
        diverged_from_strong_reference = (
            strong_recovery_reference
            and reference_estimate is not None
            and distance_meters(
                tracked.latitude,
                tracked.longitude,
                reference_estimate.latitude,
                reference_estimate.longitude,
            ) > reference_limit
        )
        sustained_divergence = False
        if diverged_from_strong_reference and reference_estimate is not None:
            candidate = self.track_recovery_candidates.get(estimate.name)
            if candidate is None:
                self.track_recovery_candidates[estimate.name] = (
                    current_timestamp,
                    reference_estimate.latitude,
                    reference_estimate.longitude,
                )
            else:
                held_seconds = max(0.0, (current_timestamp - candidate[0]) / 1000.0)
                candidate_consistent = distance_meters(
                    candidate[1],
                    candidate[2],
                    reference_estimate.latitude,
                    reference_estimate.longitude,
                ) <= 50.0 + 20.0 * held_seconds
                if not candidate_consistent:
                    self.track_recovery_candidates[estimate.name] = (
                        current_timestamp,
                        reference_estimate.latitude,
                        reference_estimate.longitude,
                    )
                else:
                    required_hold_seconds = (
                        0.2
                        if estimate.geometry_score >= 85.0
                        and scatter_meters <= 30.0
                        else 0.8
                    )
                    sustained_divergence = held_seconds >= required_hold_seconds
        else:
            self.track_recovery_candidates.pop(estimate.name, None)

        if (
            updated.coast_seconds >= 4.0 and reference_is_consistent
        ) or sustained_divergence:
            # Recover after a prolonged sequence of gated measurements without
            # importing any external target position. Recovery is deliberately
            # forbidden in weak geometry. A non-coasting tracker is reset only
            # after a strong radio-only reference disagrees consistently for
            # at least two cycles in excellent geometry, and longer in
            # merely adequate geometry; never after one attractive-looking
            # raw solution.
            recovery_estimate = reference_estimate
            updated = initialize_bearing_track(
                recovery_estimate,
                measurements,
                current_timestamp,
                east_velocity_mps=observed_east_velocity,
                north_velocity_mps=observed_north_velocity,
            )
            tracked = recovery_estimate
            accepted = len(measurements)
            rejected = 0
            motion["tracker_recovered"] = True
            self.track_recovery_candidates.pop(estimate.name, None)
        self.track_states[estimate.name] = updated
        self.sample_rejections[estimate.name] = (accepted, rejected)

        speed = math.hypot(
            updated.east_velocity_mps,
            updated.north_velocity_mps,
        )
        if motion is not None:
            motion["speed_mps"] = speed
            if speed > 0.05:
                motion["heading_degrees"] = math.degrees(
                    math.atan2(
                        updated.east_velocity_mps,
                        updated.north_velocity_mps,
                    )
                ) % 360.0
        return EstimatedTarget(
            name=tracked.name,
            latitude=tracked.latitude,
            longitude=tracked.longitude,
            measurement_count=estimate.measurement_count,
            mean_rssi_db=tracked.mean_rssi_db,
            residual_meters=tracked.residual_meters,
            geometry_score=tracked.geometry_score,
            geometry_label=tracked.geometry_label,
        )

    def _newest_target_timestamp(self, states: list[ReceiverState]) -> float | None:
        timestamps = [
            target.timestamp_ms
            for state in states
            for target in state.targets
            if target.timestamp_ms is not None
        ]
        return max(timestamps) if timestamps else None

    def update_map(self) -> None:
        now = time.time()
        self.map.krakens = [
            {
                "name": state.display_name,
                "latitude": state.latitude,
                "longitude": state.longitude,
            }
            for state in self.receivers.values()
            if state.latitude is not None and state.longitude is not None and state.online(now, self.config["timeout"])
        ]
        display_estimates = [self._display_estimate(estimate) for estimate in self.estimates]
        self.display_estimates = {
            estimate.name: estimate for estimate in display_estimates
        }
        estimated_markers = [
            {
                "name": f"{estimate.name} tracked",
                "latitude": estimate.latitude,
                "longitude": estimate.longitude,
                "active": True,
                "color": "#ff9500",
                "symbol": "◆",
            }
            for estimate in display_estimates
        ]
        raw_markers = [
            {
                "name": f"{name} raw",
                "latitude": estimate.latitude,
                "longitude": estimate.longitude,
                "active": True,
                "color": "#ef4444",
                "symbol": "×",
            }
            for name, estimate in self.raw_estimates.items()
            if self.show_raw_debug and self.target_motion.get(name, {}).get("moving", False)
        ]
        truth_markers = [
            {
                "name": f"{name} truth",
                "latitude": latitude,
                "longitude": longitude,
                "active": True,
                "color": "#22c55e",
                "symbol": "●",
            }
            for name, (latitude, longitude) in self.ground_truths.items()
        ]
        self.map.targets = estimated_markers + raw_markers + truth_markers
        signal_paths = []
        for estimate in display_estimates:
            for measurement in self.estimate_measurements.get(estimate.name, []):
                signal_paths.append(
                    {
                        "receiver_name": measurement.receiver_name,
                        "receiver_lat": measurement.receiver_latitude,
                        "receiver_lon": measurement.receiver_longitude,
                        "target_name": estimate.name,
                        "target_lat": estimate.latitude,
                        "target_lon": estimate.longitude,
                        "doa": measurement.doa_degrees,
                        "rssi": measurement.rssi_db,
                    }
                )
        self.map.signal_paths = signal_paths
        self.map.target_trails = {
            name: list(points)
            for name, points in self.estimate_trails.items()
        }
        self._auto_frame_map()
        self._follow_moving_target(display_estimates)
        self.map.update()

    def _display_estimate(self, estimate: EstimatedTarget) -> EstimatedTarget:
        if not self.target_motion.get(estimate.name, {}).get("moving", False):
            self.display_estimates.pop(estimate.name, None)
        # The tracked estimate is already smoothed by the time-aware Kalman
        # filter. A second per-refresh interpolation added visible lag at short
        # simulator intervals and made the map disagree with the error metrics.
        return estimate

    def _follow_moving_target(self, display_estimates: list[EstimatedTarget]) -> None:
        if not self._auto_frame_done:
            return
        moving_estimates = [
            estimate
            for estimate in display_estimates
            if self.target_motion.get(estimate.name, {}).get("moving", False)
        ]
        if not moving_estimates:
            return
        inner_view = self.map.rect().adjusted(110, 90, -110, -90)
        if all(
            inner_view.contains(
                self.map.geo_to_screen(estimate.latitude, estimate.longitude).toPoint()
            )
            for estimate in moving_estimates
        ):
            return
        latitude = sum(estimate.latitude for estimate in moving_estimates) / len(moving_estimates)
        longitude = sum(estimate.longitude for estimate in moving_estimates) / len(moving_estimates)
        self.map.set_view(latitude, longitude, self.map.zoom)
        self.auto_frame_status.setText("●  TRACKING TARGET")

    def _auto_frame_map(self) -> None:
        if self._auto_frame_done or self._auto_frame_pending:
            return
        receiver_points = sorted(
            (round(item["latitude"], 7), round(item["longitude"], 7))
            for item in self.map.krakens
        )
        truth_points = sorted(
            (round(latitude, 7), round(longitude, 7))
            for latitude, longitude in self.ground_truths.values()
        )
        estimate_points = sorted(
            (round(estimate.latitude, 5), round(estimate.longitude, 5))
            for estimate in self.estimates
        )
        target_points = truth_points if truth_points else estimate_points
        if len(receiver_points) < 2 or not target_points:
            return
        # Give the remaining UDP datagrams in the synchronized simulator cycle
        # a brief moment to arrive, then frame exactly once for this listener
        # session. Localization noise can never alter the viewport afterwards.
        self._auto_frame_pending = True
        QTimer.singleShot(350, self._perform_initial_auto_frame)

    def _perform_initial_auto_frame(self) -> None:
        self._auto_frame_pending = False
        if self._auto_frame_done:
            return
        points = self.map_frame_points()
        if len(self.map.krakens) < 2 or len(points) <= len(self.map.krakens):
            return
        self.fit_map_to_data(points)
        self._auto_frame_done = True
        self.auto_frame_status.setText("●  FRAME LOCKED")

    def refresh(self) -> None:
        now = time.time()
        self._refresh_receivers(now)
        self._refresh_estimates()
        self._refresh_evaluation()
        self._refresh_metrics(now)
        self.update_map()
        if self.estimates:
            self.statusBar().showMessage(
                self.tr_text("localized", targets=len(self.estimates), receivers=len(self.map.krakens))
            )
        elif self.listener:
            self.statusBar().showMessage(self.tr_text("waiting"))

    def _refresh_metrics(self, now: float) -> None:
        online_receivers = sum(
            state.online(now, self.config["timeout"]) for state in self.receivers.values()
        )
        total_packets = sum(state.packet_count for state in self.receivers.values())
        self.receivers_metric.setText(str(online_receivers))
        self.receivers_metric_hint.setText(
            "Ready for localization" if online_receivers >= 2 else "At least 2 required"
        )
        self.targets_metric.setText(str(len(self.estimates)))
        self.targets_metric_hint.setText(
            self._localization_summary()
            if self.estimates else "No solution yet"
        )
        mode_label, mode_hint, mode_color = self._mode_summary()
        self.mode_metric.setText(mode_label)
        self.mode_metric_hint.setText(mode_hint)
        self.mode_metric.setProperty("accent", mode_color)
        self.mode_metric.setStyleSheet(f"color: {mode_color};")
        self.packets_metric.setText(f"{total_packets:,}")
        self.packets_metric_hint.setText(f"UDP {self.config['bind_ip']}:{self.config['port']}")
        current_errors = [
            stats.current_error
            for stats in self.evaluation_stats.values()
            if stats.samples
        ]
        if not current_errors:
            self.rmse_metric.setText("—")
            self.rmse_metric_hint.setText("Evaluation inactive")
        else:
            current_error = sum(current_errors) / len(current_errors)
            self.rmse_metric.setText(f"{current_error:.1f} m")
            if len(current_errors) == 1:
                target_name, stats = next(
                    (name, stats)
                    for name, stats in self.evaluation_stats.items()
                    if stats.samples
                )
                raw_hint = (
                    f" · raw {stats.current_raw_error:.1f} m"
                    if stats.current_raw_error is not None
                    else ""
                )
                self.rmse_metric_hint.setText(f"{target_name} · tracked{raw_hint}")
            else:
                self.rmse_metric_hint.setText(
                    f"Mean tracked error across {len(current_errors)} targets"
                )

    def _refresh_receivers(self, now: float) -> None:
        states = sorted(self.receivers.values(), key=lambda item: item.display_name)
        self.receiver_table.setRowCount(len(states))
        for row, state in enumerate(states):
            online = state.online(now, self.config["timeout"])
            values = [
                state.display_name,
                self.tr_text("online") if online else self.tr_text("offline"),
                state.source_address,
                "—" if state.latitude is None else f"{state.latitude:.7f}",
                "—" if state.longitude is None else f"{state.longitude:.7f}",
                str(state.packet_count),
                "—" if state.sequence is None else str(state.sequence),
                str(len(state.targets)),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if column == 1:
                    item.setForeground(QColor("#22c55e" if online else "#ef4444"))
                self.receiver_table.setItem(row, column, item)

    def _refresh_estimates(self) -> None:
        self.estimate_table.setRowCount(len(self.estimates))
        for row, estimate in enumerate(self.estimates):
            accepted, rejected = self.sample_rejections.get(
                estimate.name, (estimate.measurement_count, 0)
            )
            motion = self.target_motion.get(estimate.name, {})
            target_label = estimate.name
            if motion.get("moving", False):
                mode = "AUTO TRACK" if motion.get("auto_detected", False) else "TRACK"
                target_label += f"  {mode} · {motion.get('speed_mps', 0.0):.1f} m/s"
            elif motion.get("auto_detected", False):
                target_label += "  AUTO STATIC"
            values = [
                target_label,
                f"{estimate.latitude:.5f}, {estimate.longitude:.5f}",
                str(estimate.measurement_count),
                f"{estimate.mean_rssi_db:.1f} dB",
                f"{estimate.residual_meters:.1f} m",
                f"{estimate.geometry_label} · {estimate.geometry_score:.0f}",
                f"{accepted} / {rejected}",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if column == 5:
                    item.setForeground(QColor(self._geometry_color(estimate.geometry_label)))
                self.estimate_table.setItem(row, column, item)

    def _sample_summary(self) -> str:
        accepted = sum(values[0] for values in self.sample_rejections.values())
        rejected = sum(values[1] for values in self.sample_rejections.values())
        return f"{accepted} accepted · {rejected} rejected"

    def _localization_summary(self) -> str:
        if not self.estimates:
            return self._sample_summary()
        weakest = min(self.estimates, key=lambda estimate: estimate.geometry_score)
        return f"Geometry {weakest.geometry_label} · {weakest.geometry_score:.0f} · {self._sample_summary()}"

    def _mode_summary(self) -> tuple[str, str, str]:
        if not self.estimates:
            return "WAITING", "No localized target yet", "#94a3b8"

        modes = []
        for estimate in self.estimates:
            motion = self.target_motion.get(estimate.name, {})
            if motion.get("moving", False):
                modes.append(("AUTO TRACK", "#f59e0b"))
            else:
                modes.append(("AUTO STATIC", "#22c55e"))

        unique_modes = {mode for mode, _color in modes}
        if len(unique_modes) > 1:
            moving_count = sum(1 for mode, _color in modes if mode == "AUTO TRACK")
            static_count = sum(1 for mode, _color in modes if mode == "AUTO STATIC")
            return "MIXED", f"{static_count} static · {moving_count} tracking", "#a78bfa"

        label, color = modes[0]
        if label == "AUTO TRACK":
            speeds = [
                self.target_motion.get(estimate.name, {}).get("speed_mps", 0.0)
                for estimate in self.estimates
                if self.target_motion.get(estimate.name, {}).get("moving", False)
            ]
            speed_hint = f" · {sum(speeds) / len(speeds):.1f} m/s" if speeds else ""
            return label, f"Kalman tracking{speed_hint}", color
        return label, "Cumulative static correction", color

    @staticmethod
    def _geometry_color(label: str) -> str:
        return {
            "GOOD": "#22c55e",
            "FAIR": "#f59e0b",
            "POOR": "#ef4444",
        }.get(label, "#94a3b8")

    def _refresh_evaluation(self) -> None:
        rows = sorted(self.evaluation_stats.items())
        self.evaluation_table.setRowCount(len(rows))
        detail_parts = []
        for row, (name, stats) in enumerate(rows):
            raw_error = (
                f"{stats.current_raw_error:.1f} m"
                if stats.current_raw_error is not None
                else "—"
            )
            filter_delta = (
                f"{stats.current_filter_delta:+.1f} m"
                if stats.current_filter_delta is not None
                else "—"
            )
            values = [
                name,
                f"{stats.current_error:.1f} m",
                raw_error,
                filter_delta,
                f"{stats.rmse:.1f} m",
                stats.quality,
            ]
            detail_parts.append(
                f"{name}  ·  {stats.samples} samples  ·  Avg {stats.average_error:.1f} m"
                f"  ·  Median {stats.median_error:.1f} m  ·  P95 {stats.percentile_95:.1f} m"
                f"  ·  Std {stats.standard_deviation:.1f} m  ·  Max {stats.maximum_error:.1f} m"
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if column == 3 and stats.current_filter_delta is not None:
                    item.setForeground(
                        QColor("#22c55e" if stats.current_filter_delta <= 0 else "#f59e0b")
                    )
                if column == 5:
                    quality_color = {
                        "EXCELLENT": "#22c55e",
                        "GOOD": "#84cc16",
                        "FAIR": "#f59e0b",
                        "POOR": "#ef4444",
                    }[stats.quality]
                    item.setForeground(QColor(quality_color))
                self.evaluation_table.setItem(row, column, item)
        self.evaluation_details.setText(
            "\n".join(detail_parts) if detail_parts else "No evaluation samples yet"
        )
        if hasattr(self, "live_trajectory_plot"):
            self.live_trajectory_plot.set_samples(self.evaluation_samples)
            self.live_error_plot.set_samples(self.evaluation_samples)

    def listener_failed(self, error: str) -> None:
        QMessageBox.critical(
            self,
            "KrakenScope",
            self.tr_text("bind_error", bind=self.config["bind_ip"], port=self.config["port"], error=error),
        )
        self.listener = None
        self.retranslate()

    def toggle_theme(self) -> None:
        self.theme_mode = "light" if self.theme_mode == "dark" else "dark"
        self.settings.setValue("theme_mode", self.theme_mode)
        self.apply_theme()
        self.retranslate()

    def apply_theme(self) -> None:
        palettes = {
            "dark": {
                "background": "#070b14",
                "panel": "#101827",
                "panel_alt": "#0c1320",
                "field": "#0b1220",
                "hover": "#162235",
                "pressed": "#15334a",
                "border": "#243247",
                "border_strong": "#314158",
                "text": "#e8eef8",
                "soft_text": "#b7c6da",
                "muted": "#7f8da3",
                "primary": "#0ea5e9",
                "primary_hover": "#38bdf8",
                "primary_text": "#ffffff",
                "selection": "#15334a",
                "metric_hover": "#121d2e",
                "metric_badge": "#0d1726",
                "metric_badge_border": "#2b3b52",
                "floating_bg": "rgba(7, 11, 20, 225)",
                "danger_bg": "#261319",
                "danger_text": "#ef4444",
                "danger_border": "#57232d",
                "success_bg": "#10271d",
                "success_text": "#22c55e",
                "success_border": "#24583c",
                "map_status_bg": "rgba(7, 25, 38, 230)",
                "map_status_border": "#1b5875",
                "map_status_text": "#38bdf8",
                "metrics_strip_bg": "#0a111d",
                "metrics_strip_text": "#93a4ba",
                "tooltip_bg": "#101827",
                "dialog_bg": "#101827",
                "scrollbar": "#314158",
            },
            "light": {
                "background": "#eef3f8",
                "panel": "#ffffff",
                "panel_alt": "#f8fafc",
                "field": "#ffffff",
                "hover": "#e8f4fb",
                "pressed": "#dbeafe",
                "border": "#cbd5e1",
                "border_strong": "#94a3b8",
                "text": "#0f172a",
                "soft_text": "#334155",
                "muted": "#64748b",
                "primary": "#0284c7",
                "primary_hover": "#0ea5e9",
                "primary_text": "#ffffff",
                "selection": "#bfdbfe",
                "metric_hover": "#f1f7fc",
                "metric_badge": "#f0f9ff",
                "metric_badge_border": "#bae6fd",
                "floating_bg": "rgba(255, 255, 255, 235)",
                "danger_bg": "#fff1f2",
                "danger_text": "#dc2626",
                "danger_border": "#fecdd3",
                "success_bg": "#ecfdf5",
                "success_text": "#047857",
                "success_border": "#a7f3d0",
                "map_status_bg": "rgba(240, 249, 255, 235)",
                "map_status_border": "#7dd3fc",
                "map_status_text": "#0369a1",
                "metrics_strip_bg": "#f8fafc",
                "metrics_strip_text": "#475569",
                "tooltip_bg": "#ffffff",
                "dialog_bg": "#ffffff",
                "scrollbar": "#94a3b8",
            },
        }
        theme = palettes.get(self.theme_mode, palettes["dark"])
        QApplication.instance().setStyleSheet(
            """
            QWidget {
                background: %(background)s;
                color: %(text)s;
                font-family: "Avenir Next", "SF Pro Display", Arial;
                font-size: 13px;
            }
            QWidget#appRoot { background: %(background)s; }
            QFrame#header, QFrame#metricCard, QFrame#panel, QFrame#sidebar {
                background: %(panel)s;
                border: 1px solid %(border)s;
                border-radius: 16px;
            }
            QFrame#header { border-radius: 18px; }
            QFrame#metricCard:hover {
                border-color: %(border_strong)s;
                background: %(metric_hover)s;
            }
            QWidget#mapStack { background: transparent; }
            QFrame#floatingPanel {
                background: %(floating_bg)s;
                border: 1px solid %(border_strong)s;
                border-radius: 12px;
                margin: 14px;
            }
            QLabel { background: transparent; border: none; }
            QLabel#appTitle { font-size: 21px; font-weight: 800; }
            QLabel#appSubtitle, QLabel#panelHint, QLabel#metricTitle {
                color: %(muted)s;
                font-size: 9px;
                font-weight: 700;
            }
            QLabel#panelHint { color: %(success_text)s; }
            QLabel#statusBadge {
                color: %(danger_text)s;
                background: %(danger_bg)s;
                border: 1px solid %(danger_border)s;
                border-radius: 11px;
                padding: 6px 10px;
                font-size: 10px;
                font-weight: 800;
            }
            QLabel#statusBadge[online="true"] {
                color: %(success_text)s;
                background: %(success_bg)s;
                border-color: %(success_border)s;
            }
            QLabel#mapStatus {
                color: %(map_status_text)s;
                background: %(map_status_bg)s;
                border: 1px solid %(map_status_border)s;
                border-radius: 11px;
                padding: 6px 10px;
                margin: 14px;
                font-size: 9px;
                font-weight: 800;
            }
            QLabel#floatingTitle {
                color: %(text)s;
                font-size: 11px;
                font-weight: 800;
            }
            QLabel#legendLabel { font-size: 9px; font-weight: 700; }
            QLabel#metricsStrip {
                color: %(metrics_strip_text)s;
                background: %(metrics_strip_bg)s;
                border: 1px solid %(border)s;
                border-radius: 9px;
                padding: 8px 10px;
                font-size: 10px;
            }
            QLabel#metricIcon {
                min-width: 36px;
                max-width: 36px;
                min-height: 36px;
                max-height: 36px;
                border-radius: 10px;
                font-size: 18px;
                font-weight: 800;
            }
            QLabel#metricValue { font-size: 27px; font-weight: 800; }
            QLabel#metricHint { color: %(muted)s; font-size: 10px; }
            QLabel#panelTitle { font-size: 13px; font-weight: 800; }
            QLabel#sectionTitle {
                color: %(muted)s;
                font-size: 10px;
                font-weight: 800;
                padding: 4px 2px;
            }
            QPushButton, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
                background: %(field)s;
                color: %(text)s;
                padding: 7px 10px;
                border: 1px solid %(border)s;
                border-radius: 8px;
            }
            QPushButton:hover, QComboBox:hover {
                background: %(hover)s;
                border-color: %(primary_hover)s;
            }
            QPushButton:pressed { background: %(pressed)s; }
            QPushButton#primaryButton {
                background: %(primary)s;
                color: %(primary_text)s;
                border-color: %(primary_hover)s;
                font-weight: 750;
                padding-left: 14px;
                padding-right: 14px;
            }
            QPushButton#primaryButton:hover { background: %(primary_hover)s; }
            QPushButton#compactButton { padding: 5px 8px; font-size: 11px; }
            QCheckBox {
                spacing: 7px;
                color: %(muted)s;
                min-height: 24px;
            }
            QCheckBox::indicator { width: 15px; height: 15px; }
            QTabWidget::pane {
                background: %(panel_alt)s;
                border: 1px solid %(border)s;
                border-radius: 12px;
                top: -1px;
            }
            QTabBar::tab {
                background: transparent;
                color: %(muted)s;
                padding: 10px 12px;
                border-bottom: 2px solid transparent;
                font-weight: 700;
            }
            QTabBar::tab:selected {
                color: %(primary_hover)s;
                background: %(hover)s;
                border-bottom-color: %(primary_hover)s;
            }
            QHeaderView::section {
                padding: 7px;
                background: %(panel_alt)s;
                color: %(muted)s;
                border: none;
                border-bottom: 1px solid %(border)s;
                font-size: 10px;
                font-weight: 700;
            }
            QTableWidget, QTextEdit {
                background: %(field)s;
                color: %(text)s;
                alternate-background-color: %(panel_alt)s;
                selection-background-color: %(selection)s;
                selection-color: %(text)s;
                border: 1px solid %(border)s;
                border-radius: 9px;
            }
            QTableWidget::item { padding: 5px; border: none; }
            QTextEdit {
                color: %(soft_text)s;
                font-family: Menlo;
                font-size: 11px;
                padding: 8px;
            }
            QMenu {
                background: %(panel)s;
                color: %(text)s;
                border: 1px solid %(border)s;
                border-radius: 9px;
                padding: 5px;
            }
            QMenu::item {
                padding: 7px 28px 7px 12px;
                border-radius: 6px;
            }
            QMenu::item:selected {
                background: %(primary)s;
                color: %(primary_text)s;
            }
            QDialog, QMessageBox {
                background: %(dialog_bg)s;
                color: %(text)s;
            }
            QDialogButtonBox { background: transparent; }
            QToolTip {
                background: %(tooltip_bg)s;
                color: %(text)s;
                border: 1px solid %(border)s;
                padding: 5px;
            }
            QSplitter::handle { background: transparent; width: 10px; }
            QWidget#liveMap { border-radius: 12px; }
            QStatusBar { background: transparent; color: %(muted)s; }
            QStatusBar::item { border: none; }
            QScrollBar:vertical {
                background: transparent;
                width: 10px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: %(scrollbar)s;
                min-height: 24px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            """ % theme
        )
        self._apply_accent_styles(theme)
        if hasattr(self, "map"):
            self.map.set_theme(self.theme_mode)

    def _apply_accent_styles(self, theme: dict[str, str] | None = None) -> None:
        if theme is None:
            theme = {
                "metric_badge": "#0d1726",
                "metric_badge_border": "#2b3b52",
                "primary_hover": "#38bdf8",
            }
        metric_badge = theme["metric_badge"]
        metric_badge_border = theme["metric_badge_border"]
        for label in self.findChildren(QLabel):
            accent = label.property("accent")
            if label.objectName() == "metricIcon":
                accent = accent or theme["primary_hover"]
                label.setStyleSheet(
                    f"color: {accent}; background: {metric_badge}; border: 1px solid {metric_badge_border};"
                )
            elif label.objectName() in {"metricValue", "legendLabel"}:
                accent = accent or theme["primary_hover"]
                label.setStyleSheet(f"color: {accent};")

    def closeEvent(self, event) -> None:
        if self.listener:
            self.listener.stop()
        if self.simulator.state() != QProcess.ProcessState.NotRunning:
            self.simulator.kill()
            self.simulator.waitForFinished(1000)
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("KrakenScope")
    app.setOrganizationName("UCY")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
