"""Native cross-platform Qt GUI for the multi-KrakenSDR simulator."""

from __future__ import annotations

import json
import math
import random
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import (
    QByteArray,
    QPoint,
    QPointF,
    QRect,
    QSettings,
    QStandardPaths,
    Qt,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import QAction, QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkDiskCache, QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


DEFAULT_LATITUDE = 35.1856
DEFAULT_LONGITUDE = 33.3823
TILE_SIZE = 256
MAX_MERCATOR_LAT = 85.05112878
TILE_URLS = {
    "dark": "https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png",
    "light": "https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png",
}
DOA_NOISE_STD_DEGREES = 1.5
RSSI_AT_10_METERS_DB = -30.0
RSSI_NOISE_STD_DB = 1.0
RECEIVER_BIAS_STD_DEGREES = 0.75
MULTIPATH_PROBABILITY = 0.06
PATH_LOSS_EXPONENT = 2.25
SHADOW_FADING_STD_DB = 4.0
SHADOW_FADING_MEMORY = 0.985
SLOW_BIAS_DRIFT_STD_DEGREES = 0.015
NLOS_PROBABILITY = 0.015
DEFAULT_RANDOM_SPEED_MIN_MPS = 2.0
DEFAULT_RANDOM_SPEED_MAX_MPS = 20.0


def bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    x = math.sin(delta_lon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def destination_point(
    latitude: float, longitude: float, heading_degrees: float, distance_meters_value: float
) -> tuple[float, float]:
    """Move a point along a great-circle path."""
    if distance_meters_value <= 0:
        return latitude, longitude
    radius = 6_371_000.0
    angular_distance = distance_meters_value / radius
    heading = math.radians(heading_degrees)
    latitude_1 = math.radians(latitude)
    longitude_1 = math.radians(longitude)
    latitude_2 = math.asin(
        math.sin(latitude_1) * math.cos(angular_distance)
        + math.cos(latitude_1) * math.sin(angular_distance) * math.cos(heading)
    )
    longitude_2 = longitude_1 + math.atan2(
        math.sin(heading) * math.sin(angular_distance) * math.cos(latitude_1),
        math.cos(angular_distance) - math.sin(latitude_1) * math.sin(latitude_2),
    )
    return math.degrees(latitude_2), ((math.degrees(longitude_2) + 180.0) % 360.0) - 180.0


def offset_point(
    latitude: float,
    longitude: float,
    forward_meters: float,
    right_meters: float,
    heading_degrees: float,
) -> tuple[float, float]:
    """Offset a point in coordinates relative to a route heading."""
    heading = math.radians(heading_degrees)
    east = forward_meters * math.sin(heading) + right_meters * math.cos(heading)
    north = forward_meters * math.cos(heading) - right_meters * math.sin(heading)
    distance = math.hypot(east, north)
    if distance <= 0:
        return latitude, longitude
    bearing = (math.degrees(math.atan2(east, north)) + 360.0) % 360.0
    return destination_point(latitude, longitude, bearing, distance)


def route_template_points(
    template_name: str,
    latitude: float,
    longitude: float,
    size_meters: float,
    heading_degrees: float,
) -> list[list[float]]:
    """Generate a route whose first point is the target's current position."""
    size = max(1.0, float(size_meters))
    local_points: list[tuple[float, float]]
    if template_name == "Straight line":
        local_points = [(0.0, 0.0), (size, 0.0)]
    elif template_name == "Circle":
        local_points = [
            (size * math.sin(angle), size * (1.0 - math.cos(angle)))
            for angle in (2.0 * math.pi * index / 32 for index in range(32))
        ]
    elif template_name == "Square":
        local_points = [(0.0, 0.0), (size, 0.0), (size, size), (0.0, size)]
    elif template_name == "Rectangle":
        local_points = [(0.0, 0.0), (size * 1.5, 0.0), (size * 1.5, size), (0.0, size)]
    elif template_name == "Triangle":
        local_points = [
            (0.0, 0.0),
            (size, 0.0),
            (size * 0.5, size * math.sqrt(3.0) / 2.0),
        ]
    elif template_name == "Figure eight":
        local_points = [
            (size * math.sin(angle), size * 0.55 * math.sin(2.0 * angle))
            for angle in (2.0 * math.pi * index / 40 for index in range(40))
        ]
    elif template_name == "Zigzag":
        local_points = [(0.0, 0.0)]
        for index in range(1, 7):
            local_points.append((size * index / 3.0, size * (0.45 if index % 2 else -0.45)))
    else:
        raise ValueError(f"Unknown route template: {template_name}")
    return [
        list(offset_point(latitude, longitude, forward, right, heading_degrees))
        for forward, right in local_points
    ]


def reset_target_route_progress(target: dict, *, move_to_start: bool = False) -> None:
    """Reset transient route state after a route is created, edited, or restarted."""
    target.pop("_route_index", None)
    target["_route_finished"] = False

    route = target.get("route", [])
    if not target.get("moving", False) or len(route) < 2:
        return

    if move_to_start:
        target["latitude"] = float(route[0][0])
        target["longitude"] = float(route[0][1])
    target["_route_index"] = 1


def realistic_doa_noise_std(rssi_db: float, base_std: float, distance_meters_value: float) -> float:
    """Approximate Kraken-like bearing uncertainty from signal strength.

    This is intentionally a simulator model rather than a hardware promise:
    strong, clean signals stay around 1–2 degrees; medium signals move into
    the 3–5 degree range; weak signals can easily become 8–12+ degrees.
    """
    distance_factor = max(0.0, min(1.0, (distance_meters_value - 120.0) / 900.0))
    if rssi_db >= -45.0:
        std = max(base_std, 1.0)
    elif rssi_db >= -60.0:
        std = max(base_std, 1.0 + (-45.0 - rssi_db) / 15.0 * 2.0)
    elif rssi_db >= -75.0:
        std = max(base_std, 3.0 + (-60.0 - rssi_db) / 15.0 * 4.0)
    elif rssi_db >= -90.0:
        std = max(base_std, 7.0 + (-75.0 - rssi_db) / 15.0 * 6.0)
    else:
        std = max(base_std, 15.0)
    return min(28.0, std + distance_factor * 2.5)


def realistic_doa_error(
    rssi_db: float,
    base_std: float,
    receiver_bias: float,
    distance_meters_value: float,
    multipath_active: bool,
) -> float:
    std = realistic_doa_noise_std(rssi_db, base_std, distance_meters_value)
    error = receiver_bias + random.gauss(0.0, std)
    weak_signal_factor = max(0.0, min(1.0, (-55.0 - rssi_db) / 35.0))
    distance_factor = max(0.0, min(1.0, (distance_meters_value - 150.0) / 850.0))
    multipath_probability = MULTIPATH_PROBABILITY * (0.45 + weak_signal_factor + 0.7 * distance_factor)
    if multipath_active or random.random() < multipath_probability:
        error += random.choice((-1.0, 1.0)) * random.uniform(8.0, 38.0)
    if random.random() < NLOS_PROBABILITY * (0.5 + weak_signal_factor):
        error += random.choice((-1.0, 1.0)) * random.uniform(25.0, 70.0)
    return error


def realistic_rssi(
    distance_meters_value: float,
    receiver_shadow_db: float,
    base_noise_db: float,
    multipath_active: bool,
) -> float:
    distance = max(distance_meters_value, 10.0)
    path_loss = 10.0 * PATH_LOSS_EXPONENT * math.log10(distance / 10.0)
    small_scale_fading = random.gauss(0.0, base_noise_db)
    if multipath_active:
        small_scale_fading += random.uniform(-10.0, 3.0)
    rssi = RSSI_AT_10_METERS_DB - path_loss + receiver_shadow_db + small_scale_fading
    return max(-118.0, min(-15.0, rssi))


def update_shadow_fading(previous_db: float | None) -> float:
    if previous_db is None:
        return random.gauss(0.0, SHADOW_FADING_STD_DB)
    innovation_std = SHADOW_FADING_STD_DB * math.sqrt(1.0 - SHADOW_FADING_MEMORY**2)
    return previous_db * SHADOW_FADING_MEMORY + random.gauss(0.0, innovation_std)


def geo_to_world(latitude: float, longitude: float, zoom: int) -> QPointF:
    latitude = max(-MAX_MERCATOR_LAT, min(MAX_MERCATOR_LAT, latitude))
    scale = TILE_SIZE * (2**zoom)
    x = (longitude + 180.0) / 360.0 * scale
    sin_lat = math.sin(math.radians(latitude))
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * scale
    return QPointF(x, y)


def world_to_geo(x: float, y: float, zoom: int) -> tuple[float, float]:
    scale = TILE_SIZE * (2**zoom)
    longitude = x / scale * 360.0 - 180.0
    n = math.pi - 2.0 * math.pi * y / scale
    latitude = math.degrees(math.atan(math.sinh(n)))
    return latitude, longitude


class SlippyMap(QWidget):
    pointClicked = Signal(str, float, float)
    viewChanged = Signal(float, float, int)
    cursorMoved = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 360)
        self.setObjectName("mapCanvas")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.center_lat = DEFAULT_LATITUDE
        self.center_lon = DEFAULT_LONGITUDE
        self.zoom = 15
        self.locked = True
        self.placement_mode = "none"
        self.krakens: list[dict] = []
        self.targets: list[dict] = []
        self.target_trails: dict[str, list[tuple[float, float]]] = {}
        self.route_draft: list[tuple[float, float]] = []
        self.route_draft_name = ""
        self.route_draft_loop = False
        self.user_location: tuple[float, float] | None = None
        self.signal_paths: list[dict] = []
        self.animation_phase = 0.0
        self.tiles: dict[tuple[int, int, int], QPixmap] = {}
        self.pending: set[tuple[int, int, int]] = set()
        self.failed_tiles: dict[tuple[int, int, int], float] = {}
        self.drag_start: QPoint | None = None
        self.drag_center: QPointF | None = None
        self.dragged = False
        self.theme_mode = "dark"
        self.tile_url = TILE_URLS[self.theme_mode]
        self.placeholder_color = QColor("#1c1c1e")
        self.map_background_color = QColor("#151517")
        self.attribution_color = QColor("#8e8e93")

        self.network = QNetworkAccessManager(self)
        cache = QNetworkDiskCache(self.network)
        cache_root = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation))
        cache_directory = cache_root / "KrakenSDRSimulator" / "map-tiles"
        try:
            cache_directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            cache_directory = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.TempLocation)) / "KrakenSDRSimulator-map-tiles"
            cache_directory.mkdir(parents=True, exist_ok=True)
        cache.setCacheDirectory(str(cache_directory))
        cache.setMaximumCacheSize(80 * 1024 * 1024)
        self.network.setCache(cache)
        self.network.finished.connect(self._tile_finished)
        self.animation_timer = QTimer(self)
        self.animation_timer.setInterval(40)
        self.animation_timer.timeout.connect(self._animate_signals)
        self.animation_timer.start()

    def set_theme(self, theme_mode: str):
        theme_mode = "light" if theme_mode == "light" else "dark"
        if self.theme_mode == theme_mode:
            return
        self.theme_mode = theme_mode
        self.tile_url = TILE_URLS[theme_mode]
        if theme_mode == "light":
            self.placeholder_color = QColor("#eef3f8")
            self.map_background_color = QColor("#f8fafc")
            self.attribution_color = QColor("#64748b")
        else:
            self.placeholder_color = QColor("#1c1c1e")
            self.map_background_color = QColor("#151517")
            self.attribution_color = QColor("#8e8e93")
        self.tiles.clear()
        self.pending.clear()
        self.failed_tiles.clear()
        self.update()

    def set_view(self, latitude: float, longitude: float, zoom: int | None = None):
        self.center_lat = max(-MAX_MERCATOR_LAT, min(MAX_MERCATOR_LAT, latitude))
        self.center_lon = ((longitude + 180.0) % 360.0) - 180.0
        if zoom is not None:
            self.zoom = max(2, min(19, int(zoom)))
        self.update()
        self.viewChanged.emit(self.center_lat, self.center_lon, self.zoom)

    def screen_to_geo(self, point: QPointF) -> tuple[float, float]:
        center = geo_to_world(self.center_lat, self.center_lon, self.zoom)
        world_x = center.x() + point.x() - self.width() / 2
        world_y = center.y() + point.y() - self.height() / 2
        return world_to_geo(world_x, world_y, self.zoom)

    def geo_to_screen(self, latitude: float, longitude: float) -> QPointF:
        center = geo_to_world(self.center_lat, self.center_lon, self.zoom)
        world = geo_to_world(latitude, longitude, self.zoom)
        return QPointF(self.width() / 2 + world.x() - center.x(), self.height() / 2 + world.y() - center.y())

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.map_background_color)
        self._paint_tiles(painter)
        self._paint_routes(painter)
        self._paint_target_trails(painter)
        self._paint_signal_paths(painter)
        self._paint_markers(painter)
        painter.setPen(self.attribution_color)
        painter.setFont(QFont("Arial", 8))
        painter.drawText(12, self.height() - 12, "© OpenStreetMap contributors  © CARTO")

    def _paint_tiles(self, painter: QPainter):
        center = geo_to_world(self.center_lat, self.center_lon, self.zoom)
        left = center.x() - self.width() / 2
        top = center.y() - self.height() / 2
        first_x = math.floor(left / TILE_SIZE)
        first_y = math.floor(top / TILE_SIZE)
        last_x = math.floor((left + self.width()) / TILE_SIZE)
        last_y = math.floor((top + self.height()) / TILE_SIZE)
        tile_count = 2**self.zoom
        for raw_x in range(first_x, last_x + 1):
            for tile_y in range(first_y, last_y + 1):
                if not 0 <= tile_y < tile_count:
                    continue
                tile_x = raw_x % tile_count
                key = (self.zoom, tile_x, tile_y)
                target = QRect(
                    round(raw_x * TILE_SIZE - left),
                    round(tile_y * TILE_SIZE - top),
                    TILE_SIZE,
                    TILE_SIZE,
                )
                pixmap = self.tiles.get(key)
                if pixmap:
                    painter.drawPixmap(target, pixmap)
                else:
                    painter.fillRect(target, self.placeholder_color)
                    self._request_tile(key)

    def _request_tile(self, key: tuple[int, int, int]):
        if key in self.pending:
            return
        last_failure = self.failed_tiles.get(key)
        if last_failure is not None and time.monotonic() - last_failure < 30.0:
            return
        self.pending.add(key)
        zoom, tile_x, tile_y = key
        request = QNetworkRequest(QUrl(self.tile_url.format(z=zoom, x=tile_x, y=tile_y)))
        request.setRawHeader(QByteArray(b"User-Agent"), QByteArray(b"KrakenSDR-Simulator/2.0 thesis project"))
        reply = self.network.get(request)
        reply.setProperty("tile_key", key)

    def _tile_finished(self, reply: QNetworkReply):
        key = reply.property("tile_key")
        if key:
            key = tuple(key)
            self.pending.discard(key)
            if reply.error() == QNetworkReply.NetworkError.NoError:
                pixmap = QPixmap()
                if pixmap.loadFromData(reply.readAll()):
                    self.tiles[key] = pixmap
                    self.failed_tiles.pop(key, None)
                    if len(self.tiles) > 350:
                        self.tiles.pop(next(iter(self.tiles)))
                    self.update()
            else:
                self.failed_tiles[key] = time.monotonic()
        reply.deleteLater()

    def _animate_signals(self):
        if not self.signal_paths:
            return
        self.animation_phase = (self.animation_phase + 1.0) % 1000.0
        self.update()

    def _paint_signal_paths(self, painter: QPainter):
        """Draw animated RF/data paths from targets toward Kraken receivers."""
        for index, path in enumerate(self.signal_paths):
            receiver = self.geo_to_screen(path["receiver_lat"], path["receiver_lon"])
            target = self.geo_to_screen(path["target_lat"], path["target_lon"])
            dx = receiver.x() - target.x()
            dy = receiver.y() - target.y()
            length = math.hypot(dx, dy)
            if length < 2:
                continue

            # Quiet base connection and a moving dashed data layer.
            painter.setPen(QPen(QColor(0, 122, 255, 48), 4.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(target, receiver)
            moving_pen = QPen(QColor(0, 122, 255, 180), 2.0, Qt.PenStyle.CustomDashLine, Qt.PenCapStyle.RoundCap)
            moving_pen.setDashPattern([4.0, 6.0])
            moving_pen.setDashOffset(-self.animation_phase * 0.28 - index * 2.5)
            painter.setPen(moving_pen)
            painter.drawLine(target, receiver)

            # Three pulses travel from the RF target toward the receiver.
            for pulse_index in range(3):
                progress = (self.animation_phase * 0.012 + pulse_index / 3 + index * 0.07) % 1.0
                point = QPointF(target.x() + dx * progress, target.y() + dy * progress)
                painter.setPen(QPen(QColor(255, 255, 255, 225), 2.0))
                painter.setBrush(QColor(0, 122, 255, 225))
                painter.drawEllipse(point, 4.0, 4.0)

            # Compact live measurement badge at the middle of each path.
            midpoint = QPointF((target.x() + receiver.x()) / 2, (target.y() + receiver.y()) / 2)
            text = f'{path["doa"]:.1f}°  {path["rssi"]:.0f} dB'
            badge_width = max(82, len(text) * 6 + 16)
            badge = QRect(round(midpoint.x() - badge_width / 2), round(midpoint.y() - 12), badge_width, 24)
            painter.setPen(QPen(QColor(82, 82, 88, 220), 1.0))
            painter.setBrush(QColor(36, 36, 39, 242))
            painter.drawRoundedRect(badge, 10, 10)
            painter.setPen(QColor("#f5f5f7"))
            painter.setFont(QFont("Arial", 8, QFont.Weight.DemiBold))
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_routes(self, painter: QPainter):
        for target in self.targets:
            route = target.get("route", [])
            if len(route) >= 2:
                self._draw_route(
                    painter,
                    route,
                    QColor(255, 159, 10, 185),
                    close_route=bool(target.get("loop_route", False)),
                )
        if len(self.route_draft) >= 1:
            self._draw_route(
                painter,
                self.route_draft,
                QColor(10, 132, 255, 235),
                show_points=True,
                close_route=self.route_draft_loop,
            )

    def _draw_route(
        self,
        painter: QPainter,
        route,
        color: QColor,
        show_points: bool = False,
        close_route: bool = False,
    ):
        screen_points = [self.geo_to_screen(float(point[0]), float(point[1])) for point in route]
        line_points = list(screen_points)
        if close_route and len(line_points) >= 2 and route[-1] != route[0]:
            line_points.append(line_points[0])
        if len(line_points) >= 2:
            base_pen = QPen(QColor(0, 0, 0, 150), 6.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(base_pen)
            for start, end in zip(line_points, line_points[1:]):
                painter.drawLine(start, end)
            route_pen = QPen(color, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(route_pen)
            for start, end in zip(line_points, line_points[1:]):
                painter.drawLine(start, end)
        if show_points:
            for index, point in enumerate(screen_points):
                painter.setPen(QPen(QColor("#f5f5f7"), 2.0))
                painter.setBrush(color)
                painter.drawEllipse(point, 6.0, 6.0)
                if index:
                    painter.setPen(QColor("#f5f5f7"))
                    painter.setFont(QFont("Arial", 7, QFont.Weight.Bold))
                    painter.drawText(
                        QRect(round(point.x() - 8), round(point.y() - 8), 16, 16),
                        Qt.AlignmentFlag.AlignCenter,
                        str(index),
                    )

    def _paint_target_trails(self, painter: QPainter):
        for points in self.target_trails.values():
            if len(points) < 2:
                continue
            screen_points = [self.geo_to_screen(latitude, longitude) for latitude, longitude in points]
            painter.setPen(
                QPen(QColor(255, 69, 58, 135), 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            )
            for start, end in zip(screen_points, screen_points[1:]):
                painter.drawLine(start, end)

    def _paint_markers(self, painter: QPainter):
        if self.user_location:
            self._draw_marker(painter, *self.user_location, "You", QColor("#af52de"), "●")
        for receiver in self.krakens:
            self._draw_marker(
                painter, receiver["latitude"], receiver["longitude"], receiver["name"], QColor("#007aff"), "▲"
            )
        for target in self.targets:
            color = QColor(target.get("color", "#ff3b30" if target["active"] else "#8e8e93"))
            if target["active"] and target.get("moving", False):
                self._draw_heading_indicator(
                    painter,
                    target["latitude"],
                    target["longitude"],
                    target.get("heading_degrees", 0.0),
                    color,
                )
            self._draw_marker(
                painter,
                target["latitude"],
                target["longitude"],
                target["name"],
                color,
                target.get("symbol", "●"),
            )

    def _draw_heading_indicator(self, painter, latitude, longitude, heading, color):
        point = self.geo_to_screen(latitude, longitude)
        angle = math.radians(heading)
        start = QPointF(point.x() + math.sin(angle) * 10, point.y() - math.cos(angle) * 10)
        end = QPointF(point.x() + math.sin(angle) * 28, point.y() - math.cos(angle) * 28)
        painter.setPen(QPen(QColor(255, 255, 255, 220), 5.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(start, end)
        painter.setPen(QPen(color, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(start, end)

    def _draw_marker(self, painter, latitude, longitude, label, color, symbol):
        point = self.geo_to_screen(latitude, longitude)
        if not self.rect().adjusted(-80, -40, 80, 40).contains(point.toPoint()):
            return
        painter.setPen(QPen(QColor("#f5f5f7"), 3))
        painter.setBrush(color)
        painter.drawEllipse(point, 11, 11)
        painter.setPen(QColor("#f5f5f7"))
        painter.setFont(QFont("Arial", 8, QFont.Weight.DemiBold))
        painter.drawText(QRect(round(point.x() - 10), round(point.y() - 10), 20, 20), Qt.AlignmentFlag.AlignCenter, symbol)
        label_rect = QRect(round(point.x() + 14), round(point.y() - 13), max(70, len(label) * 8 + 14), 26)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(36, 36, 39, 242))
        painter.drawRoundedRect(label_rect, 9, 9)
        painter.setPen(QColor("#f5f5f7"))
        painter.drawText(label_rect.adjusted(7, 0, -5, 0), Qt.AlignmentFlag.AlignVCenter, label)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start = event.position().toPoint()
            self.drag_center = geo_to_world(self.center_lat, self.center_lon, self.zoom)
            self.dragged = False

    def mouseMoveEvent(self, event):
        latitude, longitude = self.screen_to_geo(event.position())
        self.cursorMoved.emit(latitude, longitude)
        if self.drag_start is None or self.locked:
            return
        delta = event.position().toPoint() - self.drag_start
        if delta.manhattanLength() > 3:
            self.dragged = True
        latitude, longitude = world_to_geo(
            self.drag_center.x() - delta.x(), self.drag_center.y() - delta.y(), self.zoom
        )
        self.center_lat, self.center_lon = latitude, longitude
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if not self.dragged and self.placement_mode != "none":
            latitude, longitude = self.screen_to_geo(event.position())
            self.pointClicked.emit(self.placement_mode, latitude, longitude)
        self.drag_start = None
        self.drag_center = None
        self.viewChanged.emit(self.center_lat, self.center_lon, self.zoom)

    def wheelEvent(self, event):
        if self.locked:
            return
        step = 1 if event.angleDelta().y() > 0 else -1
        new_zoom = max(2, min(19, self.zoom + step))
        if new_zoom != self.zoom:
            self.zoom = new_zoom
            self.update()
            self.viewChanged.emit(self.center_lat, self.center_lon, self.zoom)


class ItemDialog(QDialog):
    def __init__(self, kind: str, initial: dict, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.setWindowTitle("Kraken receiver" if kind == "kraken" else "RF target")
        self.setMinimumWidth(370)
        form = QFormLayout(self)
        self.name = QLineEdit(initial.get("name", ""))
        form.addRow("Name" if kind == "kraken" else "Internal ID", self.name)
        if kind == "kraken":
            self.host = QLineEdit(initial.get("host", "127.0.0.1"))
            self.port = QSpinBox()
            self.port.setRange(1, 65535)
            self.port.setValue(initial.get("port", 4200))
            form.addRow("Destination hostname / IP", self.host)
            form.addRow("UDP port", self.port)
        else:
            self.route = [list(point) for point in initial.get("route", [])]
            self.frequency = QDoubleSpinBox()
            self.frequency.setRange(0.001, 100000.0)
            self.frequency.setDecimals(6)
            self.frequency.setValue(initial.get("frequency", 446.1))
            self.active = QCheckBox("Active")
            self.active.setChecked(initial.get("active", True))
            form.addRow("Frequency (MHz)", self.frequency)
            form.addRow("", self.active)
            self.moving = QCheckBox("Move during simulation")
            self.moving.setChecked(initial.get("moving", False))
            initial_speed = float(initial.get("speed_mps", 10.0))
            self.speed = QDoubleSpinBox()
            self.speed.setRange(0.0, 500.0)
            self.speed.setDecimals(1)
            self.speed.setSingleStep(1.0)
            self.speed.setSuffix(" m/s")
            self.speed.setValue(initial_speed)
            self.random_speed = QCheckBox("Random speed")
            self.random_speed.setChecked(initial.get("random_speed", False))
            self.min_speed = QDoubleSpinBox()
            self.min_speed.setRange(0.0, 500.0)
            self.min_speed.setDecimals(1)
            self.min_speed.setSingleStep(1.0)
            self.min_speed.setSuffix(" m/s")
            self.min_speed.setValue(
                initial.get(
                    "min_speed_mps",
                    max(0.0, initial_speed * 0.5, DEFAULT_RANDOM_SPEED_MIN_MPS),
                )
            )
            self.max_speed = QDoubleSpinBox()
            self.max_speed.setRange(0.0, 500.0)
            self.max_speed.setDecimals(1)
            self.max_speed.setSingleStep(1.0)
            self.max_speed.setSuffix(" m/s")
            self.max_speed.setValue(
                initial.get(
                    "max_speed_mps",
                    min(500.0, max(initial_speed * 1.5, DEFAULT_RANDOM_SPEED_MAX_MPS)),
                )
            )
            self.heading = QDoubleSpinBox()
            self.heading.setRange(0.0, 359.9)
            self.heading.setDecimals(1)
            self.heading.setSingleStep(5.0)
            self.heading.setSuffix(" °")
            self.heading.setValue(initial.get("heading_degrees", 90.0))
            self.loop_route = QCheckBox("Loop route")
            self.loop_route.setChecked(initial.get("loop_route", False))
            self.moving.toggled.connect(self._sync_motion_controls)
            self.random_speed.toggled.connect(self._sync_motion_controls)
            form.addRow("", self.moving)
            form.addRow("Speed", self.speed)
            form.addRow("", self.random_speed)
            form.addRow("Min speed", self.min_speed)
            form.addRow("Max speed", self.max_speed)
            form.addRow("Heading", self.heading)
            form.addRow("", self.loop_route)
            self._sync_motion_controls()
        self.latitude = QDoubleSpinBox()
        self.latitude.setRange(-90.0, 90.0)
        self.latitude.setDecimals(7)
        self.latitude.setValue(initial.get("latitude", DEFAULT_LATITUDE))
        self.longitude = QDoubleSpinBox()
        self.longitude.setRange(-180.0, 180.0)
        self.longitude.setDecimals(7)
        self.longitude.setValue(initial.get("longitude", DEFAULT_LONGITUDE))
        coordinate_prefix = "" if kind == "kraken" else "Ground-truth "
        form.addRow(f"{coordinate_prefix}latitude", self.latitude)
        form.addRow(f"{coordinate_prefix}longitude", self.longitude)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _accept_if_valid(self):
        if not self.name.text().strip():
            QMessageBox.warning(self, "Invalid name", "Name cannot be empty.")
            return
        if self.kind == "kraken" and not self.host.text().strip():
            QMessageBox.warning(self, "Invalid destination", "Hostname/IP cannot be empty.")
            return
        if (
            self.kind == "target"
            and self.moving.isChecked()
            and self.random_speed.isChecked()
            and self.max_speed.value() <= self.min_speed.value()
        ):
            QMessageBox.warning(self, "Invalid speed range", "Max speed must be greater than min speed.")
            return
        self.accept()

    def _sync_motion_controls(self):
        if self.kind != "target":
            return
        moving = self.moving.isChecked()
        random_speed = self.random_speed.isChecked()
        self.speed.setEnabled(moving and not random_speed)
        self.random_speed.setEnabled(moving)
        self.min_speed.setEnabled(moving and random_speed)
        self.max_speed.setEnabled(moving and random_speed)
        self.heading.setEnabled(moving)
        self.loop_route.setEnabled(moving)

    def result_data(self) -> dict:
        data = {
            "name": self.name.text().strip(),
            "latitude": self.latitude.value(),
            "longitude": self.longitude.value(),
        }
        if self.kind == "kraken":
            data.update(host=self.host.text().strip(), port=self.port.value())
        else:
            data.update(
                frequency=self.frequency.value(),
                active=self.active.isChecked(),
                moving=self.moving.isChecked(),
                speed_mps=self.speed.value(),
                random_speed=self.random_speed.isChecked(),
                min_speed_mps=self.min_speed.value(),
                max_speed_mps=self.max_speed.value(),
                heading_degrees=self.heading.value(),
                loop_route=self.loop_route.isChecked(),
                route=self.route,
            )
        return data


class RouteTemplateDialog(QDialog):
    TEMPLATES = (
        "Straight line",
        "Circle",
        "Square",
        "Rectangle",
        "Triangle",
        "Figure eight",
        "Zigzag",
    )

    def __init__(self, target: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Route template")
        self.setMinimumWidth(360)
        form = QFormLayout(self)
        self.template = QComboBox()
        self.template.addItems(self.TEMPLATES)
        self.size = QDoubleSpinBox()
        self.size.setRange(5.0, 100_000.0)
        self.size.setDecimals(1)
        self.size.setSingleStep(25.0)
        self.size.setValue(250.0)
        self.size.setSuffix(" m")
        self.heading = QDoubleSpinBox()
        self.heading.setRange(0.0, 359.9)
        self.heading.setDecimals(1)
        self.heading.setSingleStep(5.0)
        self.heading.setValue(target.get("heading_degrees", 0.0))
        self.heading.setSuffix(" °")
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.1, 500.0)
        self.speed.setDecimals(1)
        self.speed.setValue(max(0.1, target.get("speed_mps", 10.0)))
        self.speed.setSuffix(" m/s")
        self.loop_route = QCheckBox("Loop route")
        self.loop_route.setChecked(True)
        self.template.currentTextChanged.connect(self._template_changed)
        form.addRow("Shape", self.template)
        form.addRow("Size / radius", self.size)
        form.addRow("Start heading", self.heading)
        form.addRow("Target speed", self.speed)
        form.addRow("", self.loop_route)
        note = QLabel("The route starts from the target's current position.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        form.addRow(note)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Create route")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self._template_changed(self.template.currentText())

    def _template_changed(self, template_name: str):
        self.loop_route.setChecked(template_name not in {"Straight line", "Zigzag"})

    def values(self) -> tuple[str, float, float, float, bool]:
        return (
            self.template.currentText(),
            self.size.value(),
            self.heading.value(),
            self.speed.value(),
            self.loop_route.isChecked(),
        )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KrakenSDR Simulator")
        self.resize(1500, 920)
        self.krakens: list[dict] = []
        self.targets: list[dict] = []
        self.receiver_sequences: dict[str, int] = {}
        self.receiver_biases: dict[str, float] = {}
        self.link_shadow_offsets: dict[tuple[str, str], float] = {}
        self.multipath_bursts: dict[tuple[str, str], int] = {}
        self.packet_count = 0
        self.last_motion_time: float | None = None
        self.route_target_index: int | None = None
        self.udp_socket: socket.socket | None = None
        self.configuration_controls: list[QWidget] = []
        self.settings = QSettings("UCY", "KrakenSDRSimulator")
        self.theme_mode = str(self.settings.value("theme_mode", "dark"))
        if self.theme_mode not in {"dark", "light"}:
            self.theme_mode = "dark"
        self._build_ui()
        self._apply_style()
        self._refresh()

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("appRoot")
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 16, 18, 14)
        root.setSpacing(14)

        header = QFrame()
        header.setObjectName("header")
        header.setMaximumHeight(68)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 12, 14, 12)

        brand = QVBoxLayout()
        brand.setSpacing(0)
        self.app_title = QLabel("SIMULATOR")
        self.app_title.setObjectName("appTitle")
        brand.addWidget(self.app_title)
        header_layout.addLayout(brand)
        header_layout.addSpacing(22)

        self.header_status = QLabel("●  CONFIGURATION")
        self.header_status.setObjectName("statusBadge")
        self.header_status.setFixedHeight(32)
        self.header_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(self.header_status)
        header_layout.addStretch()
        self.theme_button = QPushButton("Light mode")
        self.theme_button.setObjectName("secondaryButton")
        self.theme_button.clicked.connect(self.toggle_theme)
        header_layout.addWidget(self.theme_button)
        root.addWidget(header)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(12)
        metrics.setVerticalSpacing(0)
        self.krakens_metric_card, self.krakens_metric, self.krakens_metric_hint = self._metric_card(
            "◉", "CONFIGURED RECEIVERS", "0", "Add at least 2", "#38bdf8"
        )
        self.targets_metric_card, self.targets_metric, self.targets_metric_hint = self._metric_card(
            "◇", "RF TARGETS", "0", "No targets configured", "#a78bfa"
        )
        self.active_metric_card, self.active_metric, self.active_metric_hint = self._metric_card(
            "⇄", "ACTIVE EMITTERS", "0", "Ready to transmit", "#14b8a6"
        )
        self.packets_metric_card, self.packets_metric, self.packets_metric_hint = self._metric_card(
            "⇣", "PACKETS SENT", "0", "UDP output idle", "#f59e0b"
        )
        for column, card in enumerate(
            (
                self.krakens_metric_card,
                self.targets_metric_card,
                self.active_metric_card,
                self.packets_metric_card,
            )
        ):
            metrics.addWidget(card, 0, column)
            metrics.setColumnStretch(column, 1)
        root.addLayout(metrics)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(420)
        sidebar.setMaximumWidth(540)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(10, 10, 10, 10)
        side_layout.setSpacing(8)
        side_header = QHBoxLayout()
        title = QLabel("SIMULATOR INSPECTOR")
        title.setObjectName("panelTitle")
        side_header.addWidget(title)
        side_header.addStretch()
        subtitle = QLabel("● CONFIG")
        subtitle.setWordWrap(True)
        subtitle.setObjectName("panelHint")
        side_header.addWidget(subtitle)
        side_layout.addLayout(side_header)

        self.project_button = QPushButton("Project")
        self.project_button.setObjectName("secondaryButton")
        project_menu = QMenu(self.project_button)
        new_action = QAction("New configuration", self)
        new_action.triggered.connect(self.new_configuration)
        open_action = QAction("Open…", self)
        open_action.triggered.connect(self.load_configuration)
        save_action = QAction("Save…", self)
        save_action.triggered.connect(self.save_configuration)
        project_menu.addAction(new_action)
        project_menu.addSeparator()
        project_menu.addAction(open_action)
        project_menu.addAction(save_action)
        self.project_button.setMenu(project_menu)
        side_layout.addWidget(self.project_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.configuration_controls.append(self.project_button)

        self.tabs = QTabWidget()
        receiver_tab, self.kraken_table = self._make_table_tab(
            ["Name", "Destination", "Lat", "Lon"], self.add_kraken, self.edit_kraken, self.remove_kraken
        )
        target_tab, self.target_table = self._make_table_tab(
            ["Target", "MHz", "Motion", "Active"], self.add_target, self.edit_target, self.remove_target
        )
        self.route_button = QPushButton("Route")
        route_menu = QMenu(self.route_button)
        template_action = QAction("Choose template…", self)
        template_action.triggered.connect(self.apply_route_template)
        draw_action = QAction("Draw custom route", self)
        draw_action.triggered.connect(self.begin_route_drawing)
        clear_action = QAction("Clear route", self)
        clear_action.triggered.connect(self.clear_selected_route)
        route_menu.addAction(template_action)
        route_menu.addAction(draw_action)
        route_menu.addSeparator()
        route_menu.addAction(clear_action)
        self.route_button.setMenu(route_menu)
        target_tab.layout().addWidget(self.route_button)
        self.configuration_controls.append(self.route_button)
        self.tabs.addTab(receiver_tab, "Krakens")
        self.tabs.addTab(target_tab, "Targets")
        side_layout.addWidget(self.tabs, 4)

        simulation = QFrame()
        simulation.setObjectName("card")
        sim_layout = QVBoxLayout(simulation)
        simulation_title = QLabel("SIMULATION")
        simulation_title.setObjectName("sectionLabel")
        sim_layout.addWidget(simulation_title)
        row = QHBoxLayout()
        row.addWidget(QLabel("Interval"))
        self.interval = QDoubleSpinBox()
        self.interval.setRange(0.05, 60.0)
        self.interval.setSingleStep(0.05)
        self.interval.setValue(1.0)
        self.interval.setSuffix(" s")
        self.configuration_controls.append(self.interval)
        row.addWidget(self.interval)
        sim_layout.addLayout(row)
        self.advanced_button = QToolButton()
        self.advanced_button.setText("Advanced settings")
        self.advanced_button.setCheckable(True)
        self.advanced_button.setObjectName("advancedButton")
        sim_layout.addWidget(self.advanced_button)
        advanced_settings = QFrame()
        advanced_settings.setObjectName("advancedSettings")
        advanced_layout = QVBoxLayout(advanced_settings)
        advanced_layout.setContentsMargins(0, 4, 0, 2)
        self.realistic_noise = QCheckBox("Realistic Kraken noise")
        self.realistic_noise.setChecked(True)
        self.configuration_controls.append(self.realistic_noise)
        advanced_layout.addWidget(self.realistic_noise)
        self.evaluation_mode = QCheckBox("Evaluation mode (send target ground truth)")
        self.evaluation_mode.setChecked(True)
        self.configuration_controls.append(self.evaluation_mode)
        advanced_layout.addWidget(self.evaluation_mode)
        noise_row = QHBoxLayout()
        noise_row.addWidget(QLabel("Base DoA noise"))
        self.doa_noise = QDoubleSpinBox()
        self.doa_noise.setRange(0.0, 20.0)
        self.doa_noise.setSingleStep(0.1)
        self.doa_noise.setDecimals(2)
        self.doa_noise.setValue(DOA_NOISE_STD_DEGREES)
        self.doa_noise.setSuffix(" °")
        noise_row.addWidget(self.doa_noise)
        noise_row.addWidget(QLabel("RSSI noise"))
        self.rssi_noise = QDoubleSpinBox()
        self.rssi_noise.setRange(0.0, 30.0)
        self.rssi_noise.setSingleStep(0.1)
        self.rssi_noise.setDecimals(2)
        self.rssi_noise.setValue(RSSI_NOISE_STD_DB)
        self.rssi_noise.setSuffix(" dB")
        noise_row.addWidget(self.rssi_noise)
        self.configuration_controls.extend([self.doa_noise, self.rssi_noise])
        advanced_layout.addLayout(noise_row)
        advanced_settings.setVisible(False)
        self.advanced_button.toggled.connect(advanced_settings.setVisible)
        self.advanced_button.toggled.connect(
            lambda expanded: self.advanced_button.setText(
                "Hide advanced settings" if expanded else "Advanced settings"
            )
        )
        sim_layout.addWidget(advanced_settings)
        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.toggle_simulation)
        sim_layout.addWidget(self.start_button)
        self.status = QLabel("Ready")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        sim_layout.addWidget(self.status)
        side_layout.addWidget(simulation)

        self.packet_text = QTextEdit()
        self.packet_text.setReadOnly(True)
        self.packet_text.setPlaceholderText("The latest send cycle will appear here.")
        packet_label = QLabel("LATEST CYCLE")
        packet_label.setObjectName("sectionLabel")
        side_layout.addWidget(packet_label)
        self.packet_text.setMinimumHeight(56)
        side_layout.addWidget(self.packet_text, 2)
        splitter.addWidget(sidebar)

        map_panel = QFrame()
        map_panel.setObjectName("panel")
        map_layout = QVBoxLayout(map_panel)
        map_layout.setContentsMargins(10, 10, 10, 10)
        map_layout.setSpacing(8)

        map_toolbar = QFrame()
        map_toolbar.setObjectName("mapToolbar")
        toolbar_layout = QHBoxLayout(map_toolbar)
        toolbar_layout.setContentsMargins(12, 8, 12, 8)
        toolbar_layout.setSpacing(8)
        map_title = QLabel("SIMULATION MAP")
        map_title.setObjectName("mapTitle")
        toolbar_layout.addWidget(map_title)
        toolbar_layout.addStretch()
        self.map_settings_button = QToolButton()
        self.map_settings_button.setText("Settings")
        self.map_settings_button.setCheckable(True)
        toolbar_layout.addWidget(self.map_settings_button)
        set_position = QPushButton("Set my position")
        set_position.setObjectName("compactButton")
        set_position.clicked.connect(lambda: self.set_placement_mode("user"))
        toolbar_layout.addWidget(set_position)
        self.map_lock = QCheckBox("Lock")
        self.map_lock.setChecked(True)
        toolbar_layout.addWidget(self.map_lock)
        map_layout.addWidget(map_toolbar)

        self.map_settings_panel = QFrame()
        self.map_settings_panel.setObjectName("mapSettingsPanel")
        settings_layout = QHBoxLayout(self.map_settings_panel)
        settings_layout.setContentsMargins(12, 8, 12, 8)
        settings_layout.setSpacing(8)
        self.lat_input = QDoubleSpinBox()
        self.lat_input.setRange(-90, 90)
        self.lat_input.setDecimals(7)
        self.lat_input.setValue(DEFAULT_LATITUDE)
        self.lat_input.setMinimumWidth(126)
        self.lon_input = QDoubleSpinBox()
        self.lon_input.setRange(-180, 180)
        self.lon_input.setDecimals(7)
        self.lon_input.setValue(DEFAULT_LONGITUDE)
        self.lon_input.setMinimumWidth(126)
        self.zoom_input = QSpinBox()
        self.zoom_input.setRange(2, 19)
        self.zoom_input.setValue(15)
        self.zoom_input.setMinimumWidth(72)
        settings_layout.addWidget(QLabel("Latitude"))
        settings_layout.addWidget(self.lat_input)
        settings_layout.addWidget(QLabel("Longitude"))
        settings_layout.addWidget(self.lon_input)
        settings_layout.addWidget(QLabel("Zoom"))
        settings_layout.addWidget(self.zoom_input)
        set_map = QPushButton("Go")
        set_map.setObjectName("compactButton")
        set_map.clicked.connect(self.set_map)
        settings_layout.addWidget(set_map)
        settings_layout.addStretch()
        self.map_settings_panel.setVisible(False)
        self.map_settings_button.toggled.connect(self.map_settings_panel.setVisible)
        self.map_settings_button.toggled.connect(
            lambda visible: self.map_settings_button.setText("Hide settings" if visible else "Settings")
        )
        self.configuration_controls.extend([
            self.lat_input, self.lon_input, self.zoom_input, set_map, set_position,
        ])
        map_layout.addWidget(self.map_settings_panel)

        place_toolbar = QFrame()
        place_toolbar.setObjectName("placeToolbar")
        place_layout = QHBoxLayout(place_toolbar)
        place_layout.setContentsMargins(12, 8, 12, 8)
        place_layout.setSpacing(8)
        place_label = QLabel("PLACE")
        place_label.setObjectName("sectionLabel")
        place_layout.addWidget(place_label)
        self.place_group = QButtonGroup(self)
        for label, mode in (("Kraken", "kraken"), ("Target", "target"), ("Explore", "none")):
            button = QToolButton()
            button.setText(label)
            button.setCheckable(True)
            button.setProperty("mode", mode)
            button.clicked.connect(lambda checked=False, m=mode: self.set_placement_mode(m))
            self.place_group.addButton(button)
            place_layout.addWidget(button)
            if mode != "none":
                self.configuration_controls.append(button)
            if mode == "none":
                button.setChecked(True)
        place_layout.addStretch()
        self.route_hint = QLabel("")
        self.route_hint.setObjectName("routeHint")
        self.route_hint.setVisible(False)
        place_layout.addWidget(self.route_hint)
        self.undo_route_button = QPushButton("Undo")
        self.undo_route_button.clicked.connect(self.undo_route_waypoint)
        self.undo_route_button.setVisible(False)
        place_layout.addWidget(self.undo_route_button)
        self.route_loop_checkbox = QCheckBox("Loop")
        self.route_loop_checkbox.setToolTip("Return from the last waypoint to the starting point")
        self.route_loop_checkbox.toggled.connect(self._set_route_draft_loop)
        self.route_loop_checkbox.setVisible(False)
        place_layout.addWidget(self.route_loop_checkbox)
        self.finish_route_button = QPushButton("Done")
        self.finish_route_button.setObjectName("primary")
        self.finish_route_button.clicked.connect(self.finish_route_drawing)
        self.finish_route_button.setVisible(False)
        place_layout.addWidget(self.finish_route_button)
        self.cancel_route_button = QPushButton("Cancel")
        self.cancel_route_button.clicked.connect(self.cancel_route_drawing)
        self.cancel_route_button.setVisible(False)
        place_layout.addWidget(self.cancel_route_button)
        map_layout.addWidget(place_toolbar)

        map_stack = QWidget()
        map_stack.setObjectName("mapStack")
        map_stack_layout = QGridLayout(map_stack)
        map_stack_layout.setContentsMargins(0, 0, 0, 0)
        self.map = SlippyMap()
        self.map.setObjectName("liveMap")
        self.map.locked = self.map_lock.isChecked()
        self.map.pointClicked.connect(self.place_from_map)
        self.map.viewChanged.connect(self._sync_map_fields)
        self.map.cursorMoved.connect(self._cursor_moved)
        self.map_lock.toggled.connect(lambda checked: setattr(self.map, "locked", checked))
        map_stack_layout.addWidget(self.map, 0, 0)
        self.map_mode_status = QLabel("●  DESIGN MODE")
        self.map_mode_status.setObjectName("mapStatus")
        map_stack_layout.addWidget(
            self.map_mode_status, 0, 0,
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
        )
        map_layout.addWidget(map_stack, 1)
        splitter.addWidget(map_panel)
        splitter.setSizes([470, 1030])
        root.addWidget(splitter)
        self.setCentralWidget(central)

        self.footer_message = QLabel("Ready")
        self.footer_counts = QLabel("0 Krakens  ·  0 Targets")
        self.footer_coordinates = QLabel("—")
        self.statusBar().addWidget(self.footer_message, 1)
        self.statusBar().addPermanentWidget(self.footer_counts)
        self.statusBar().addPermanentWidget(self.footer_coordinates)

        self.simulation_timer = QTimer(self)
        self.simulation_timer.timeout.connect(self.send_cycle)
        self.motion_timer = QTimer(self)
        self.motion_timer.setInterval(50)
        self.motion_timer.timeout.connect(self._advance_targets)

    def _metric_card(
        self, icon: str, title: str, value: str, hint: str, accent: str
    ) -> tuple[QFrame, QLabel, QLabel]:
        card = QFrame()
        card.setObjectName("metricCard")
        card.setMinimumHeight(82)
        card.setMaximumHeight(96)
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

    def _make_table_tab(self, headers, add_callback, edit_callback, remove_callback):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setHighlightSections(False)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.setShowGrid(False)
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(36)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.doubleClicked.connect(lambda _index: edit_callback())
        layout.addWidget(table)
        buttons = QHBoxLayout()
        add_button = QPushButton("＋ Add")
        add_button.clicked.connect(add_callback)
        buttons.addWidget(add_button)
        more_button = QPushButton("More")
        more_menu = QMenu(more_button)
        edit_action = QAction("Edit selected", self)
        edit_action.triggered.connect(edit_callback)
        remove_action = QAction("Remove selected", self)
        remove_action.triggered.connect(remove_callback)
        more_menu.addAction(edit_action)
        more_menu.addSeparator()
        more_menu.addAction(remove_action)
        more_button.setMenu(more_menu)
        buttons.addWidget(more_button)
        self.configuration_controls.extend([add_button, more_button])
        layout.addLayout(buttons)
        return tab, table

    def _cursor_moved(self, latitude: float, longitude: float):
        self.footer_coordinates.setText(f"{latitude:.6f}, {longitude:.6f}  ·  z{self.map.zoom}")

    def toggle_theme(self):
        self.theme_mode = "light" if self.theme_mode == "dark" else "dark"
        self.settings.setValue("theme_mode", self.theme_mode)
        self._apply_style()

    def _set_status(self, message: str):
        self.status.setText(message)
        self.footer_message.setText(message)
        if hasattr(self, "header_status") and not self.simulation_timer.isActive():
            self.header_status.setText("●  CONFIGURATION")

    def _set_configuration_enabled(self, enabled: bool):
        for control in self.configuration_controls:
            control.setEnabled(enabled)

    def configuration_snapshot(self) -> dict:
        return {
            "format": "krakensdr-simulator-config",
            "version": 1,
            "map": {
                "latitude": self.map.center_lat,
                "longitude": self.map.center_lon,
                "zoom": self.map.zoom,
                "locked": self.map_lock.isChecked(),
                "user_location": list(self.map.user_location) if self.map.user_location else None,
            },
            "simulation": {
                "interval_seconds": self.interval.value(),
                "realistic_noise": self.realistic_noise.isChecked(),
                "doa_noise_degrees": self.doa_noise.value(),
                "rssi_noise_db": self.rssi_noise.value(),
                "evaluation_mode": self.evaluation_mode.isChecked(),
            },
            "krakens": self.krakens,
            "targets": [
                {key: value for key, value in target.items() if not key.startswith("_")}
                for target in self.targets
            ],
        }

    def save_configuration(self):
        path, _selected_filter = QFileDialog.getSaveFileName(
            self, "Save simulator configuration", "krakensdr-simulation.json", "JSON files (*.json)"
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            Path(path).write_text(json.dumps(self.configuration_snapshot(), indent=2), encoding="utf-8")
            self._set_status(f"Saved {Path(path).name}")
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def load_configuration(self):
        path, _selected_filter = QFileDialog.getOpenFileName(
            self, "Open simulator configuration", "", "JSON files (*.json)"
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            krakens, targets, map_data, simulation = self._validated_configuration(data)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "Invalid configuration", str(exc))
            return
        self.krakens = krakens
        self.targets = targets
        self.interval.setValue(simulation["interval_seconds"])
        self.realistic_noise.setChecked(simulation["realistic_noise"])
        self.doa_noise.setValue(simulation["doa_noise_degrees"])
        self.rssi_noise.setValue(simulation["rssi_noise_db"])
        self.evaluation_mode.setChecked(simulation["evaluation_mode"])
        self.map_lock.setChecked(bool(map_data.get("locked", True)))
        self.map.user_location = tuple(map_data["user_location"]) if map_data.get("user_location") else None
        self.map.target_trails = {}
        self.cancel_route_drawing(update_status=False)
        self.map.set_view(map_data["latitude"], map_data["longitude"], map_data["zoom"])
        self._refresh()
        self._set_status(f"Opened {Path(path).name}")

    @staticmethod
    def _validated_configuration(data: dict) -> tuple[list[dict], list[dict], dict, dict]:
        if not isinstance(data, dict) or data.get("format") != "krakensdr-simulator-config":
            raise ValueError("This is not a KrakenSDR simulator configuration file.")
        krakens = data.get("krakens")
        targets = data.get("targets")
        map_data = data.get("map")
        if not isinstance(krakens, list) or not isinstance(targets, list) or not isinstance(map_data, dict):
            raise ValueError("Configuration lists or map settings are missing.")
        normalized_krakens = []
        for receiver in krakens:
            if not isinstance(receiver, dict) or not receiver.get("name") or not receiver.get("host"):
                raise ValueError("Every Kraken requires a name and destination.")
            if not 1 <= int(receiver["port"]) <= 65535:
                raise ValueError("A Kraken UDP port is outside 1–65535.")
            if not -90 <= float(receiver["latitude"]) <= 90 or not -180 <= float(receiver["longitude"]) <= 180:
                raise ValueError("A Kraken has invalid coordinates.")
            normalized_krakens.append({
                "name": str(receiver["name"]).strip(),
                "host": str(receiver["host"]).strip(),
                "port": int(receiver["port"]),
                "latitude": float(receiver["latitude"]),
                "longitude": float(receiver["longitude"]),
            })
        normalized_targets = []
        for target in targets:
            if not isinstance(target, dict) or not target.get("name"):
                raise ValueError("Every internal target requires an identifier.")
            if float(target["frequency"]) <= 0:
                raise ValueError("A target frequency must be positive.")
            if not -90 <= float(target["latitude"]) <= 90 or not -180 <= float(target["longitude"]) <= 180:
                raise ValueError("A target has invalid coordinates.")
            if not isinstance(target.get("active"), bool):
                raise ValueError("A target active state must be true or false.")
            route = target.get("route", [])
            if not isinstance(route, list):
                raise ValueError("A target route must be a list of coordinates.")
            normalized_route = []
            for point in route:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ValueError("A target route contains an invalid waypoint.")
                point_latitude, point_longitude = float(point[0]), float(point[1])
                if not -90 <= point_latitude <= 90 or not -180 <= point_longitude <= 180:
                    raise ValueError("A target route contains invalid coordinates.")
                normalized_route.append([point_latitude, point_longitude])
            random_speed = bool(target.get("random_speed", False))
            min_speed = float(target.get("min_speed_mps", DEFAULT_RANDOM_SPEED_MIN_MPS))
            max_speed = float(target.get("max_speed_mps", DEFAULT_RANDOM_SPEED_MAX_MPS))
            normalized_targets.append({
                "name": str(target["name"]).strip(),
                "frequency": float(target["frequency"]),
                "active": target["active"],
                "latitude": float(target["latitude"]),
                "longitude": float(target["longitude"]),
                "moving": bool(target.get("moving", False)),
                "speed_mps": float(target.get("speed_mps", 10.0)),
                "random_speed": random_speed,
                "min_speed_mps": min_speed,
                "max_speed_mps": max_speed,
                "heading_degrees": float(target.get("heading_degrees", 90.0)) % 360.0,
                "loop_route": bool(target.get("loop_route", False)),
                "route": normalized_route,
            })
            if not 0.0 <= normalized_targets[-1]["speed_mps"] <= 500.0:
                raise ValueError("A target speed is outside 0–500 m/s.")
            if not 0.0 <= min_speed <= 500.0 or not 0.0 <= max_speed <= 500.0:
                raise ValueError("A random target speed is outside 0–500 m/s.")
            if random_speed and max_speed <= min_speed:
                raise ValueError("A random target max speed must be greater than min speed.")
        kraken_names = [receiver["name"] for receiver in normalized_krakens]
        target_names = [target["name"] for target in normalized_targets]
        if len(kraken_names) != len(set(kraken_names)) or len(target_names) != len(set(target_names)):
            raise ValueError("Kraken and internal target identifiers must be unique.")
        latitude = float(map_data["latitude"])
        longitude = float(map_data["longitude"])
        zoom = int(map_data["zoom"])
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180 or not 2 <= zoom <= 19:
            raise ValueError("Map center or zoom is invalid.")
        user_location = map_data.get("user_location")
        if user_location is not None:
            if not isinstance(user_location, list) or len(user_location) != 2:
                raise ValueError("Saved user location is invalid.")
            user_latitude, user_longitude = float(user_location[0]), float(user_location[1])
            if not -90 <= user_latitude <= 90 or not -180 <= user_longitude <= 180:
                raise ValueError("Saved user location is invalid.")
            map_data = dict(map_data, user_location=[user_latitude, user_longitude])
        simulation_data = data.get("simulation", {})
        interval = float(simulation_data.get("interval_seconds", 1.0))
        if not 0.05 <= interval <= 60:
            raise ValueError("Simulation interval is outside 0.05–60 seconds.")
        doa_noise = float(simulation_data.get("doa_noise_degrees", DOA_NOISE_STD_DEGREES))
        rssi_noise = float(simulation_data.get("rssi_noise_db", RSSI_NOISE_STD_DB))
        if not 0.0 <= doa_noise <= 20.0 or not 0.0 <= rssi_noise <= 30.0:
            raise ValueError("Saved noise settings are outside their supported ranges.")
        simulation = {
            "interval_seconds": interval,
            "realistic_noise": bool(simulation_data.get("realistic_noise", True)),
            "doa_noise_degrees": doa_noise,
            "rssi_noise_db": rssi_noise,
            "evaluation_mode": bool(simulation_data.get("evaluation_mode", True)),
        }
        map_data = dict(map_data, latitude=latitude, longitude=longitude, zoom=zoom)
        return normalized_krakens, normalized_targets, map_data, simulation

    def new_configuration(self):
        if (self.krakens or self.targets) and QMessageBox.question(
            self, "New configuration", "Clear all receivers, targets, and live paths?"
        ) != QMessageBox.StandardButton.Yes:
            return
        self.krakens = []
        self.targets = []
        self.map.user_location = None
        self.map.signal_paths = []
        self.map.target_trails = {}
        self.cancel_route_drawing(update_status=False)
        self._refresh()
        self._set_status("New configuration")

    def set_placement_mode(self, mode):
        if self.route_target_index is not None and mode != "route":
            self.cancel_route_drawing(update_status=False)
        self.map.placement_mode = mode
        if hasattr(self, "map_mode_status"):
            mode_label = {
                "kraken": "●  PLACE KRAKEN",
                "target": "●  PLACE TARGET",
                "user": "●  SET POSITION",
                "route": "●  ROUTE DRAWING",
                "none": "●  DESIGN MODE",
            }.get(mode, "●  DESIGN MODE")
            if self.simulation_timer.isActive():
                mode_label = "●  LIVE TRANSMISSION"
            self.map_mode_status.setText(mode_label)
        if mode == "user":
            self._set_status("Click your real position on the map")
        else:
            self._set_status("Click the map to place an object" if mode != "none" else "Map inspection mode")

    def set_map(self):
        self.map.set_view(self.lat_input.value(), self.lon_input.value(), self.zoom_input.value())
        self.map.user_location = None
        self._set_status("Fixed map center updated")

    def _sync_map_fields(self, latitude, longitude, zoom):
        self.lat_input.setValue(latitude)
        self.lon_input.setValue(longitude)
        self.zoom_input.setValue(zoom)

    def place_from_map(self, kind, latitude, longitude):
        if self.simulation_timer.isActive():
            return
        if kind == "route":
            self.map.route_draft.append((latitude, longitude))
            self.route_hint.setText(f"{len(self.map.route_draft) - 1} waypoint(s)")
            self.map.update()
            self._set_status("Add another waypoint or press Done")
            return
        if kind == "user":
            self.map.user_location = (latitude, longitude)
            self.map.update()
            self.set_placement_mode("none")
            for button in self.place_group.buttons():
                if button.property("mode") == "none":
                    button.setChecked(True)
            self._set_status(f"My position: {latitude:.7f}, {longitude:.7f}")
            return
        if kind == "kraken":
            initial = {
                "name": f"Kraken-{len(self.krakens) + 1}", "host": "127.0.0.1", "port": 4200,
                "latitude": latitude, "longitude": longitude,
            }
            dialog = ItemDialog(kind, initial, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.krakens.append(dialog.result_data())
        else:
            initial = {
                "name": f"Target-{len(self.targets) + 1}", "frequency": 446.1 + len(self.targets) * 0.1,
                "active": True, "moving": False, "speed_mps": 10.0, "heading_degrees": 90.0,
                "random_speed": False,
                "min_speed_mps": DEFAULT_RANDOM_SPEED_MIN_MPS,
                "max_speed_mps": DEFAULT_RANDOM_SPEED_MAX_MPS,
                "loop_route": False, "route": [],
                "latitude": latitude, "longitude": longitude,
            }
            dialog = ItemDialog(kind, initial, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.targets.append(dialog.result_data())
        self._refresh()

    def add_kraken(self):
        self.place_from_map("kraken", self.map.center_lat, self.map.center_lon)

    def add_target(self):
        self.place_from_map("target", self.map.center_lat, self.map.center_lon)

    def begin_route_drawing(self):
        if self.simulation_timer.isActive():
            return
        row = self._selected_row(self.target_table)
        if row is None:
            QMessageBox.information(self, "Draw route", "Select a target first.")
            return
        target = self.targets[row]
        self.route_target_index = row
        self.map.route_draft = [(target["latitude"], target["longitude"])]
        self.map.route_draft_name = target["name"]
        self.route_loop_checkbox.setChecked(bool(target.get("loop_route", False)))
        self.map.route_draft_loop = self.route_loop_checkbox.isChecked()
        self.set_placement_mode("route")
        self.route_hint.setText(f"{target['name']} · click waypoints")
        self.route_hint.setVisible(True)
        self.undo_route_button.setVisible(True)
        self.route_loop_checkbox.setVisible(True)
        self.finish_route_button.setVisible(True)
        self.cancel_route_button.setVisible(True)
        self.start_button.setEnabled(False)
        self._set_status(f"Drawing route for {target['name']}")
        self.map.update()

    def apply_route_template(self):
        if self.simulation_timer.isActive():
            return
        row = self._selected_row(self.target_table)
        if row is None:
            QMessageBox.information(self, "Route templates", "Select a target first.")
            return
        target = self.targets[row]
        dialog = RouteTemplateDialog(target, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        template_name, size, heading, speed, loop_route = dialog.values()
        target["route"] = route_template_points(
            template_name,
            target["latitude"],
            target["longitude"],
            size,
            heading,
        )
        target["moving"] = True
        target["speed_mps"] = speed
        target["heading_degrees"] = heading
        target["loop_route"] = loop_route
        reset_target_route_progress(target, move_to_start=True)
        self._refresh()
        self._set_status(f"{template_name} route created for {target['name']}")

    def finish_route_drawing(self):
        if self.route_target_index is None:
            return
        if len(self.map.route_draft) < 2:
            QMessageBox.information(self, "Draw route", "Add at least one waypoint on the map.")
            return
        target = self.targets[self.route_target_index]
        target["route"] = [[latitude, longitude] for latitude, longitude in self.map.route_draft]
        target["moving"] = True
        target["loop_route"] = self.route_loop_checkbox.isChecked()
        target["heading_degrees"] = bearing_degrees(*target["route"][0], *target["route"][1])
        reset_target_route_progress(target, move_to_start=True)
        route_name = target["name"]
        waypoint_count = len(target["route"]) - 1
        self._close_route_editor()
        self._refresh()
        self._set_status(f"Route saved for {route_name} · {waypoint_count} waypoint(s)")

    def undo_route_waypoint(self):
        if self.route_target_index is None or len(self.map.route_draft) <= 1:
            return
        self.map.route_draft.pop()
        self.route_hint.setText(f"{len(self.map.route_draft) - 1} waypoint(s)")
        self.map.update()

    def _set_route_draft_loop(self, checked: bool):
        self.map.route_draft_loop = checked
        self.map.update()

    def cancel_route_drawing(self, _checked=False, update_status=True):
        if self.route_target_index is None and not self.map.route_draft:
            return
        self._close_route_editor()
        if update_status:
            self._set_status("Route drawing cancelled")

    def _close_route_editor(self):
        self.route_target_index = None
        self.map.route_draft = []
        self.map.route_draft_name = ""
        self.map.route_draft_loop = False
        self.set_placement_mode("none")
        self.route_hint.setVisible(False)
        self.undo_route_button.setVisible(False)
        self.route_loop_checkbox.setVisible(False)
        self.finish_route_button.setVisible(False)
        self.cancel_route_button.setVisible(False)
        if not self.simulation_timer.isActive():
            self.start_button.setEnabled(True)
        for button in self.place_group.buttons():
            if button.property("mode") == "none":
                button.setChecked(True)
        self.map.update()

    def clear_selected_route(self):
        row = self._selected_row(self.target_table)
        if row is None:
            QMessageBox.information(self, "Clear route", "Select a target first.")
            return
        self.targets[row]["route"] = []
        reset_target_route_progress(self.targets[row])
        self._refresh()
        self._set_status(f"Route cleared for {self.targets[row]['name']}")

    def _selected_row(self, table):
        rows = table.selectionModel().selectedRows()
        return rows[0].row() if rows else None

    def edit_kraken(self):
        if self.simulation_timer.isActive():
            return
        row = self._selected_row(self.kraken_table)
        if row is None:
            return
        dialog = ItemDialog("kraken", self.krakens[row], self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.krakens[row] = dialog.result_data()
            self._refresh()

    def edit_target(self):
        if self.simulation_timer.isActive():
            return
        row = self._selected_row(self.target_table)
        if row is None:
            return
        dialog = ItemDialog("target", self.targets[row], self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            updated = dialog.result_data()
            if updated.get("route"):
                updated["route"][0] = [updated["latitude"], updated["longitude"]]
            self.targets[row] = updated
            self._refresh()

    def remove_kraken(self):
        row = self._selected_row(self.kraken_table)
        if row is not None:
            del self.krakens[row]
            self._refresh()

    def remove_target(self):
        row = self._selected_row(self.target_table)
        if row is not None:
            del self.targets[row]
            self._refresh()

    def _refresh(self):
        self.kraken_table.setRowCount(len(self.krakens))
        for row, item in enumerate(self.krakens):
            values = [item["name"], f'{item["host"]}:{item["port"]}', f'{item["latitude"]:.6f}', f'{item["longitude"]:.6f}']
            for column, value in enumerate(values):
                self.kraken_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.target_table.setRowCount(len(self.targets))
        for row, item in enumerate(self.targets):
            if item.get("moving", False) and len(item.get("route", [])) >= 2:
                if item.get("random_speed", False):
                    motion = (
                        f'Route · {item.get("min_speed_mps", DEFAULT_RANDOM_SPEED_MIN_MPS):.1f}-'
                        f'{item.get("max_speed_mps", DEFAULT_RANDOM_SPEED_MAX_MPS):.1f} m/s'
                    )
                else:
                    motion = f'Route · {len(item["route"]) - 1} pts'
            elif item.get("moving", False):
                if item.get("random_speed", False):
                    motion = (
                        f'{item.get("min_speed_mps", DEFAULT_RANDOM_SPEED_MIN_MPS):.1f}-'
                        f'{item.get("max_speed_mps", DEFAULT_RANDOM_SPEED_MAX_MPS):.1f} m/s · '
                        f'{item.get("heading_degrees", 90.0):.0f}°'
                    )
                else:
                    motion = f'{item.get("speed_mps", 10.0):.1f} m/s · {item.get("heading_degrees", 90.0):.0f}°'
            else:
                motion = "Stationary"
            values = [
                item["name"],
                f'{item["frequency"]:.3f}',
                motion,
                "On" if item["active"] else "Off",
            ]
            for column, value in enumerate(values):
                self.target_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.map.krakens = self.krakens
        self.map.targets = self.targets
        self.map.update()
        active_count = sum(1 for target in self.targets if target["active"])
        kraken_label = "Kraken" if len(self.krakens) == 1 else "Krakens"
        target_label = "Target" if len(self.targets) == 1 else "Targets"
        self.footer_counts.setText(
            f"{len(self.krakens)} {kraken_label}  ·  {len(self.targets)} {target_label} ({active_count} active)"
        )
        self.krakens_metric.setText(str(len(self.krakens)))
        self.targets_metric.setText(str(len(self.targets)))
        self.active_metric.setText(str(active_count))
        self.packets_metric.setText(str(self.packet_count))
        self.krakens_metric_hint.setText("Ready" if len(self.krakens) >= 2 else "Add at least 2")
        self.targets_metric_hint.setText("Configured" if self.targets else "No targets configured")
        self.active_metric_hint.setText("Ready to transmit" if active_count else "No active emitters")
        self.packets_metric_hint.setText("UDP output idle" if not self.simulation_timer.isActive() else "Live UDP stream")

    def start_simulation(self):
        self.cancel_route_drawing(update_status=False)
        if len(self.krakens) < 2:
            return QMessageBox.warning(self, "Configuration", "Add at least 2 Kraken receivers.")
        if not any(target["active"] for target in self.targets):
            return QMessageBox.warning(self, "Configuration", "Add at least 1 active target.")
        kraken_names = [item["name"] for item in self.krakens]
        target_names = [item["name"] for item in self.targets]
        if len(set(kraken_names)) != len(kraken_names) or len(set(target_names)) != len(target_names):
            return QMessageBox.warning(self, "Configuration", "Kraken and target names must be unique.")
        self.receiver_sequences = {receiver["name"]: 0 for receiver in self.krakens}
        self.receiver_biases = {
            receiver["name"]: random.gauss(0.0, RECEIVER_BIAS_STD_DEGREES)
            for receiver in self.krakens
        }
        self.link_shadow_offsets = {}
        self.multipath_bursts = {}
        self.packet_count = 0
        self.last_motion_time = time.monotonic()
        for target in self.targets:
            target["_current_speed_mps"] = self._target_motion_speed(target)
            reset_target_route_progress(target, move_to_start=True)
        self.map.target_trails = {
            target["name"]: [(target["latitude"], target["longitude"])]
            for target in self.targets
            if target.get("moving", False)
        }
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.simulation_timer.start(round(self.interval.value() * 1000))
        self.motion_timer.start()
        self.start_button.setText("Stop")
        self.start_button.setObjectName("stopButton")
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)
        self._set_configuration_enabled(False)
        self.set_placement_mode("none")
        for button in self.place_group.buttons():
            if button.property("mode") == "none":
                button.setChecked(True)
        self.header_status.setText("●  TRANSMITTING")
        self.header_status.setProperty("online", True)
        self.header_status.style().unpolish(self.header_status)
        self.header_status.style().polish(self.header_status)
        self.map_mode_status.setText("●  LIVE TRANSMISSION")
        self._set_status("Simulation running")
        self.send_cycle()

    def stop_simulation(self):
        self.simulation_timer.stop()
        self.motion_timer.stop()
        if self.udp_socket:
            self.udp_socket.close()
            self.udp_socket = None
        self.start_button.setText("Start")
        self.start_button.setObjectName("primary")
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)
        self._set_configuration_enabled(True)
        self._set_status("Simulation stopped")
        self.header_status.setProperty("online", False)
        self.header_status.style().unpolish(self.header_status)
        self.header_status.style().polish(self.header_status)
        self.map_mode_status.setText("●  DESIGN MODE")
        self.last_motion_time = None
        self.map.signal_paths = []
        self.map.update()
        self._refresh()

    def toggle_simulation(self):
        if self.simulation_timer.isActive():
            self.stop_simulation()
        else:
            self.start_simulation()

    def _target_motion_speed(self, target: dict) -> float:
        if not target.get("moving", False):
            target["_current_speed_mps"] = 0.0
            return 0.0
        if target.get("random_speed", False):
            min_speed = max(0.0, min(500.0, float(target.get("min_speed_mps", DEFAULT_RANDOM_SPEED_MIN_MPS))))
            max_speed = max(0.0, min(500.0, float(target.get("max_speed_mps", DEFAULT_RANDOM_SPEED_MAX_MPS))))
            if max_speed < min_speed:
                min_speed, max_speed = max_speed, min_speed
            speed = random.uniform(min_speed, max_speed)
        else:
            speed = max(0.0, min(500.0, float(target.get("speed_mps", 0.0))))
        target["_current_speed_mps"] = speed
        return speed

    def send_cycle(self):
        self._advance_targets()
        signal_paths = []
        errors = []
        cycle_packets = []
        sent_this_cycle = 0
        active_targets = [target for target in self.targets if target["active"]]
        cycle_timestamp = datetime.now(timezone.utc).timestamp() * 1000
        for receiver in self.krakens:
            try:
                message = self._build_message(receiver, cycle_timestamp)
                encoded = json.dumps(message, ensure_ascii=False).encode("utf-8")
                self.udp_socket.sendto(encoded, (receiver["host"], receiver["port"]))
                self.packet_count += 1
                sent_this_cycle += 1
                cycle_packets.append({
                    "destination": f'{receiver["host"]}:{receiver["port"]}',
                    "packet": message,
                })
                for target, measurement in zip(active_targets, message["targets"]):
                    signal_paths.append({
                        "receiver_name": receiver["name"],
                        "receiver_lat": receiver["latitude"],
                        "receiver_lon": receiver["longitude"],
                        "target_name": target["name"],
                        "target_lat": target["latitude"],
                        "target_lon": target["longitude"],
                        "doa": measurement["doa"],
                        "rssi": measurement["rssi"],
                    })
            except OSError as exc:
                errors.append(f'{receiver["name"]}: {exc}')
        if cycle_packets:
            self.packet_text.setPlainText(self._format_live_cycle(cycle_timestamp, cycle_packets))
        self.packets_metric.setText(str(self.packet_count))
        self.map.signal_paths = signal_paths
        self.map.update()
        if sent_this_cycle == 0 and errors:
            self.stop_simulation()
            QMessageBox.critical(self, "UDP error", "No receiver accepted a packet.\n\n" + "\n".join(errors))
        elif errors:
            self._set_status(f"Sent {self.packet_count} packets · {len(errors)} receiver error(s)")
        else:
            self._set_status(f"Live · {self.packet_count} packets sent")

    def _advance_targets(self):
        """Advance moving targets using elapsed wall-clock time."""
        now = time.monotonic()
        if self.last_motion_time is None:
            self.last_motion_time = now
            return
        elapsed_seconds = min(now - self.last_motion_time, 5.0)
        self.last_motion_time = now
        if elapsed_seconds <= 0:
            return
        moved = False
        for target in self.targets:
            if (
                not target["active"]
                or not target.get("moving", False)
                or target.get("_route_finished", False)
            ):
                continue
            speed = self._target_motion_speed(target)
            distance = speed * elapsed_seconds
            if len(target.get("route", [])) >= 2:
                self._advance_target_on_route(target, distance)
            else:
                target["latitude"], target["longitude"] = destination_point(
                    target["latitude"],
                    target["longitude"],
                    target.get("heading_degrees", 0.0),
                    distance,
                )
            trail = self.map.target_trails.setdefault(target["name"], [])
            if not trail or distance_meters(
                trail[-1][0], trail[-1][1], target["latitude"], target["longitude"]
            ) >= 1.0:
                trail.append((target["latitude"], target["longitude"]))
            if len(trail) > 300:
                del trail[:-300]
            moved = True
        if moved:
            targets_by_name = {target["name"]: target for target in self.targets}
            for path in self.map.signal_paths:
                target = targets_by_name.get(path["target_name"])
                if target:
                    path["target_lat"] = target["latitude"]
                    path["target_lon"] = target["longitude"]
            self.map.update()

    def _advance_target_on_route(self, target: dict, distance_to_move: float):
        route = target["route"]
        waypoint_index = int(target.get("_route_index", 1))
        remaining = max(0.0, distance_to_move)
        safety_count = 0
        while remaining > 0.0 and safety_count < len(route) * 3 + 3:
            safety_count += 1
            if waypoint_index >= len(route):
                if target.get("loop_route", False):
                    waypoint_index = 0
                else:
                    target["_route_finished"] = True
                    break
            waypoint_latitude = float(route[waypoint_index][0])
            waypoint_longitude = float(route[waypoint_index][1])
            segment_distance = distance_meters(
                target["latitude"],
                target["longitude"],
                waypoint_latitude,
                waypoint_longitude,
            )
            if segment_distance < 0.05:
                target["latitude"], target["longitude"] = waypoint_latitude, waypoint_longitude
                waypoint_index += 1
                continue
            heading = bearing_degrees(
                target["latitude"],
                target["longitude"],
                waypoint_latitude,
                waypoint_longitude,
            )
            target["heading_degrees"] = heading
            if remaining < segment_distance:
                target["latitude"], target["longitude"] = destination_point(
                    target["latitude"],
                    target["longitude"],
                    heading,
                    remaining,
                )
                remaining = 0.0
            else:
                target["latitude"], target["longitude"] = waypoint_latitude, waypoint_longitude
                remaining -= segment_distance
                waypoint_index += 1
        target["_route_index"] = waypoint_index

    def _format_live_cycle(self, timestamp: float, cycle_packets: list[dict]) -> str:
        payload = {
            "cycle_timestamp": timestamp,
            "kraken_packets": len(cycle_packets),
            "packets": cycle_packets,
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    def _build_message(self, receiver, timestamp=None):
        receiver_name = receiver["name"]
        self.receiver_sequences[receiver_name] = self.receiver_sequences.get(receiver_name, 0) + 1
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).timestamp() * 1000
        payload = []
        for target in self.targets:
            if not target["active"]:
                continue
            doa = bearing_degrees(
                receiver["latitude"], receiver["longitude"], target["latitude"], target["longitude"]
            )
            distance = distance_meters(
                receiver["latitude"], receiver["longitude"], target["latitude"], target["longitude"]
            )
            link_key = (receiver_name, target["name"])
            if self.realistic_noise.isChecked():
                self.link_shadow_offsets[link_key] = update_shadow_fading(
                    self.link_shadow_offsets.get(link_key)
                )
                burst_remaining = self.multipath_bursts.get(link_key, 0)
                if burst_remaining <= 0 and random.random() < 0.012:
                    burst_remaining = random.randint(4, 16)
                multipath_active = burst_remaining > 0
                if multipath_active:
                    burst_remaining -= 1
                self.multipath_bursts[link_key] = burst_remaining
                simulated_rssi = realistic_rssi(
                    distance,
                    self.link_shadow_offsets[link_key],
                    self.rssi_noise.value(),
                    multipath_active,
                )
            else:
                attenuation = 20.0 * math.log10(max(distance, 10.0) / 10.0)
                simulated_rssi = RSSI_AT_10_METERS_DB - attenuation + random.gauss(0.0, self.rssi_noise.value())
                simulated_rssi = max(-115.0, min(-15.0, simulated_rssi))
                multipath_active = False
            if receiver_name not in self.receiver_biases:
                self.receiver_biases[receiver_name] = random.gauss(0.0, RECEIVER_BIAS_STD_DEGREES)
            if self.realistic_noise.isChecked():
                self.receiver_biases[receiver_name] += random.gauss(0.0, SLOW_BIAS_DRIFT_STD_DEGREES)
                self.receiver_biases[receiver_name] = max(-5.0, min(5.0, self.receiver_biases[receiver_name]))
                doa_error = realistic_doa_error(
                    simulated_rssi,
                    self.doa_noise.value(),
                    self.receiver_biases[receiver_name],
                    distance,
                    multipath_active,
                )
            else:
                doa_error = random.gauss(0.0, self.doa_noise.value())
            current_speed = float(target.get("_current_speed_mps", target.get("speed_mps", 0.0)))
            random_speed = bool(target.get("random_speed", False))
            measurement = {
                "timestamp": timestamp, "frequency": target["frequency"],
                "rssi": round(simulated_rssi, 2),
                "doa": round((doa + doa_error) % 360, 2),
                "motion": {
                    "moving": bool(target.get("moving", False)),
                    "speed_mps": round(current_speed, 3),
                    "configured_speed_mps": float(target.get("speed_mps", 0.0)),
                    "random_speed": random_speed,
                    "min_speed_mps": float(target.get("min_speed_mps", DEFAULT_RANDOM_SPEED_MIN_MPS)),
                    "max_speed_mps": float(target.get("max_speed_mps", DEFAULT_RANDOM_SPEED_MAX_MPS)),
                    "heading_degrees": float(target.get("heading_degrees", 0.0)) % 360.0,
                },
                "channel": {
                    "distance_meters": round(distance, 2),
                    "multipath_active": bool(multipath_active),
                },
            }
            if self.evaluation_mode.isChecked():
                measurement["evaluation"] = {
                    "target_name": target["name"],
                    "latitude": target["latitude"],
                    "longitude": target["longitude"],
                    "moving": bool(target.get("moving", False)),
                    "speed_mps": round(current_speed, 3),
                    "configured_speed_mps": float(target.get("speed_mps", 0.0)),
                    "random_speed": random_speed,
                    "min_speed_mps": float(target.get("min_speed_mps", DEFAULT_RANDOM_SPEED_MIN_MPS)),
                    "max_speed_mps": float(target.get("max_speed_mps", DEFAULT_RANDOM_SPEED_MAX_MPS)),
                    "heading_degrees": float(target.get("heading_degrees", 0.0)) % 360.0,
                }
            payload.append(measurement)
        return {
            "schema_version": 1,
            "kraken_name": receiver_name, "sequence": self.receiver_sequences[receiver_name],
            "receiver": {"name": receiver_name, "latitude": receiver["latitude"], "longitude": receiver["longitude"]},
            "targets": payload,
        }

    def closeEvent(self, event):
        self.stop_simulation()
        event.accept()

    def _apply_style(self):
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
                "danger_bg": "#261319",
                "danger_text": "#ef4444",
                "danger_border": "#57232d",
                "danger_hover": "#371821",
                "success_bg": "#10271d",
                "success_text": "#22c55e",
                "success_border": "#24583c",
                "map_status_bg": "rgba(7, 25, 38, 230)",
                "map_status_border": "#1b5875",
                "map_status_text": "#38bdf8",
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
                "danger_bg": "#fff1f2",
                "danger_text": "#dc2626",
                "danger_border": "#fecdd3",
                "danger_hover": "#ffe4e6",
                "success_bg": "#ecfdf5",
                "success_text": "#047857",
                "success_border": "#a7f3d0",
                "map_status_bg": "rgba(240, 249, 255, 235)",
                "map_status_border": "#7dd3fc",
                "map_status_text": "#0369a1",
                "tooltip_bg": "#ffffff",
                "dialog_bg": "#ffffff",
                "scrollbar": "#94a3b8",
            },
        }
        theme = palettes.get(self.theme_mode, palettes["dark"])
        self.setStyleSheet("""
            QMainWindow, QWidget {
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
            #sidebar {
                background: %(panel)s;
                border: 1px solid %(border)s;
                border-radius: 16px;
            }
            QWidget#mapStack {
                background: transparent;
            }
            #sidebar QLabel, #card QLabel, #mapToolbar QLabel, #placeToolbar QLabel, #floatingPanel QLabel {
                background: transparent;
                color: %(text)s;
            }
            QLabel {
                background: transparent;
                border: none;
            }
            QLabel#appTitle {
                font-size: 21px;
                font-weight: 800;
            }
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
            QLabel#metricIcon {
                min-width: 36px;
                max-width: 36px;
                min-height: 36px;
                max-height: 36px;
                border-radius: 10px;
                font-size: 18px;
                font-weight: 800;
            }
            QLabel#metricValue {
                font-size: 27px;
                font-weight: 800;
            }
            QLabel#metricHint {
                color: %(muted)s;
                font-size: 10px;
            }
            QLabel#panelTitle {
                font-size: 13px;
                font-weight: 800;
            }
            #muted { color: %(muted)s; padding-bottom: 10px; }
            #sectionLabel {
                color: %(muted)s;
                font-size: 10px;
                font-weight: 800;
                padding: 5px 2px 0 2px;
            }
            #status, #routeHint {
                color: %(primary_hover)s;
                font-size: 12px;
                font-weight: 600;
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
            QStatusBar {
                background: transparent;
                border-top: none;
                color: %(muted)s;
            }
            QStatusBar QLabel {
                background: transparent;
                color: %(muted)s;
                padding: 0 8px;
                font-size: 11px;
            }
            #card, #mapToolbar, #mapSettingsPanel, #placeToolbar {
                background: %(panel_alt)s;
                border: 1px solid %(border)s;
                border-radius: 12px;
            }
            #mapTitle {
                font-size: 11px;
                font-weight: 800;
                color: %(text)s;
            }
            #advancedSettings { background: transparent; border: none; }
            QToolButton#advancedButton {
                background: transparent;
                color: %(primary_hover)s;
                border: none;
                padding: 4px 0;
                text-align: left;
            }
            QToolButton#advancedButton:hover { background: transparent; color: %(primary)s; }
            QTabWidget::pane {
                border: 1px solid %(border)s;
                background: %(panel_alt)s;
                border-radius: 12px;
                top: -1px;
            }
            QTabBar::tab {
                background: transparent;
                color: %(muted)s;
                padding: 10px 12px;
                border: none;
                border-bottom: 2px solid transparent;
                font-weight: 700;
            }
            QTabBar::tab:selected {
                color: %(primary_hover)s;
                background: %(hover)s;
                border-bottom: 2px solid %(primary_hover)s;
            }
            QTableWidget {
                background: %(field)s;
                color: %(text)s;
                border: 1px solid %(border)s;
                border-radius: 9px;
                selection-background-color: %(selection)s;
                selection-color: %(text)s;
                alternate-background-color: %(panel_alt)s;
            }
            QTableWidget::item {
                padding: 5px;
                border: none;
            }
            QHeaderView::section {
                background: %(panel_alt)s;
                color: %(muted)s;
                padding: 7px;
                border: none;
                border-bottom: 1px solid %(border)s;
                font-size: 10px;
                font-weight: 700;
            }
            QTextEdit {
                background: %(field)s;
                color: %(soft_text)s;
                border: 1px solid %(border)s;
                border-radius: 9px;
                padding: 8px;
                font-family: Menlo;
                font-size: 11px;
                selection-background-color: %(selection)s;
            }
            QPushButton, QToolButton {
                background: %(field)s;
                color: %(text)s;
                border: 1px solid %(border)s;
                border-radius: 8px;
                padding: 7px 10px;
            }
            QPushButton:hover, QToolButton:hover {
                background: %(hover)s;
                border-color: %(primary_hover)s;
            }
            QPushButton:pressed, QToolButton:pressed { background: %(pressed)s; }
            QPushButton#primary, QPushButton#primaryButton {
                background: %(primary)s;
                color: %(primary_text)s;
                border-color: %(primary_hover)s;
                font-weight: 750;
                padding-left: 14px;
                padding-right: 14px;
            }
            QPushButton#primary:hover, QPushButton#primaryButton:hover { background: %(primary_hover)s; }
            QPushButton#stopButton {
                background: %(danger_bg)s;
                color: %(danger_text)s;
                border: 1px solid %(danger_border)s;
                font-weight: 750;
                padding: 9px 16px;
            }
            QPushButton#stopButton:hover { background: %(danger_hover)s; }
            QPushButton#quietButton, QPushButton#secondaryButton {
                background: %(field)s;
                color: %(text)s;
                border: 1px solid %(border)s;
                padding: 5px 8px;
            }
            QPushButton#quietButton:hover, QPushButton#secondaryButton:hover {
                background: %(hover)s;
                border-color: %(primary_hover)s;
            }
            QPushButton#compactButton {
                padding: 5px 8px;
                font-size: 11px;
            }
            QToolButton:checked {
                background: %(hover)s;
                color: %(primary_hover)s;
                border: 1px solid %(primary_hover)s;
                font-weight: 700;
            }
            QPushButton:disabled, QToolButton:disabled {
                color: %(muted)s;
                background: %(panel_alt)s;
                border-color: %(border)s;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background: %(field)s;
                color: %(text)s;
                border: 1px solid %(border)s;
                border-radius: 8px;
                padding: 7px 10px;
                selection-background-color: %(selection)s;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 1px solid %(primary_hover)s;
            }
            QComboBox QAbstractItemView {
                background: %(field)s;
                color: %(text)s;
                border: 1px solid %(border)s;
                selection-background-color: %(selection)s;
                selection-color: %(text)s;
                outline: none;
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
            QMenu::item:selected { background: %(primary)s; color: %(primary_text)s; }
            QMenu::separator {
                height: 1px;
                background: %(border)s;
                margin: 5px 8px;
            }
            QCheckBox {
                spacing: 7px;
                color: %(muted)s;
                min-height: 24px;
            }
            QCheckBox::indicator { width: 15px; height: 15px; }
            QDialog, QMessageBox { background: %(dialog_bg)s; color: %(text)s; }
            QDialogButtonBox { background: transparent; }
            QToolTip {
                background: %(tooltip_bg)s;
                color: %(text)s;
                border: 1px solid %(border)s;
                padding: 5px;
            }
            QSplitter::handle { background: transparent; width: 10px; }
            QWidget#liveMap { border-radius: 12px; }
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
        """ % theme)
        metric_badge = theme["metric_badge"]
        metric_badge_border = theme["metric_badge_border"]
        for label in self.findChildren(QLabel):
            if label.objectName() == "metricIcon":
                accent = label.property("accent") or theme["primary_hover"]
                label.setStyleSheet(
                    f"color: {accent}; background: {metric_badge}; border: 1px solid {metric_badge_border};"
                )
            elif label.objectName() == "metricValue":
                accent = label.property("accent") or theme["primary_hover"]
                label.setStyleSheet(f"color: {accent};")
        if hasattr(self, "theme_button"):
            self.theme_button.setText("Dark mode" if self.theme_mode == "light" else "Light mode")
        if hasattr(self, "map"):
            self.map.set_theme(self.theme_mode)


def main() -> int:
    application = QApplication(sys.argv)
    application.setApplicationName("KrakenSDR Simulator")
    application.setOrganizationName("University of Cyprus")
    app_font = application.font()
    app_font.setPointSize(13)
    application.setFont(app_font)
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
