"""Protocol parsing and state models for KrakenScope."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from typing import Any


class PacketError(ValueError):
    pass


@dataclass(frozen=True)
class Target:
    timestamp_ms: float | None
    frequency_mhz: float
    rssi_db: float
    doa_degrees: float
    name: str | None = None
    truth_latitude: float | None = None
    truth_longitude: float | None = None
    moving: bool = False
    speed_mps: float | None = None
    heading_degrees: float | None = None


@dataclass(frozen=True)
class Packet:
    source: str | None
    sequence: int | None
    kraken_name: str | None
    receiver_latitude: float | None
    receiver_longitude: float | None
    targets: list[Target]


@dataclass(frozen=True)
class BearingMeasurement:
    receiver_name: str
    receiver_latitude: float
    receiver_longitude: float
    doa_degrees: float
    rssi_db: float


@dataclass(frozen=True)
class EstimatedTarget:
    name: str
    latitude: float
    longitude: float
    measurement_count: int
    mean_rssi_db: float
    residual_meters: float
    geometry_score: float = 0.0
    geometry_label: str = "POOR"


@dataclass(frozen=True)
class MotionAnalysis:
    """Motion inferred exclusively from observed localization positions."""

    moving: bool
    speed_mps: float = 0.0
    heading_degrees: float = 0.0
    confidence: float = 0.0
    span_seconds: float = 0.0
    scatter_meters: float = 0.0
    displacement_meters: float = 0.0
    endpoint_progress_meters: float = 0.0
    projected_latitude: float | None = None
    projected_longitude: float | None = None
    bearing_angular_rate_dps: float = 0.0
    bearing_motion_confidence: float = 0.0


@dataclass(frozen=True)
class BearingMotionAnalysis:
    """Motion evidence derived directly from receiver DoA time series."""

    moving: bool
    angular_rate_dps: float = 0.0
    confidence: float = 0.0
    span_seconds: float = 0.0
    receiver_count: int = 0


@dataclass(frozen=True)
class BucketedBearing:
    timestamp_seconds: float
    doa_degrees: float
    rssi_db: float
    accepted_samples: int
    rejected_samples: int


@dataclass
class BearingTrackState:
    """Four-state bearing-only EKF state in a local ENU coordinate frame."""

    reference_latitude: float
    reference_longitude: float
    east_meters: float
    north_meters: float
    east_velocity_mps: float
    north_velocity_mps: float
    covariance: list[list[float]]
    timestamp_ms: float
    coast_seconds: float = 0.0


@dataclass
class ReceiverState:
    key: str
    display_name: str
    last_seen: float | None = None
    source_address: str = "—"
    packet_count: int = 0
    error_count: int = 0
    latitude: float | None = None
    longitude: float | None = None
    sequence: int | None = None
    targets: list[Target] = field(default_factory=list)
    history: list[tuple[float, int, Target]] = field(default_factory=list)

    def add_packet(self, packet: Packet, address: tuple[str, int], received_at: float) -> None:
        self.last_seen = received_at
        self.source_address = f"{address[0]}:{address[1]}"
        self.packet_count += 1
        self.sequence = packet.sequence
        if packet.receiver_latitude is not None and packet.receiver_longitude is not None:
            self.latitude = packet.receiver_latitude
            self.longitude = packet.receiver_longitude
        self.targets = packet.targets
        self.history.extend((received_at, index, target) for index, target in enumerate(packet.targets))

    def online(self, now: float, timeout: float = 3.0) -> bool:
        return self.last_seen is not None and now - self.last_seen <= timeout


def parse_packet(data: bytes) -> Packet:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PacketError(f"Invalid UTF-8 JSON: {exc}") from exc

    source: str | None = None
    sequence: int | None = None
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("targets")
        source_value = payload.get("source")
        source = str(source_value).strip() if source_value is not None else None
        sequence = _optional_int(payload.get("sequence"))
        receiver = payload.get("receiver") if isinstance(payload.get("receiver"), dict) else {}
        kraken_value = payload.get("kraken_name") or receiver.get("name") or source
        kraken_name = str(kraken_value).strip() if kraken_value is not None else None
        receiver_latitude = _optional_finite(receiver.get("latitude"))
        receiver_longitude = _optional_finite(receiver.get("longitude"))
    else:
        raise PacketError("Payload must be an array or metadata object")

    if not isinstance(items, list):
        raise PacketError("The targets field must be an array")

    targets: list[Target] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise PacketError(f"Target {index} must be an object")
        evaluation = item.get("evaluation") if isinstance(item.get("evaluation"), dict) else {}
        motion = item.get("motion") if isinstance(item.get("motion"), dict) else evaluation
        target_name_value = evaluation.get("target_name")
        target = Target(
            timestamp_ms=_optional_finite(item.get("timestamp")),
            frequency_mhz=_required_finite(item.get("frequency"), index, "frequency"),
            rssi_db=_required_finite(item.get("rssi"), index, "rssi"),
            doa_degrees=_required_finite(item.get("doa"), index, "doa") % 360.0,
            name=str(target_name_value).strip() if target_name_value is not None else None,
            truth_latitude=_optional_finite(evaluation.get("latitude")),
            truth_longitude=_optional_finite(evaluation.get("longitude")),
            moving=bool(motion.get("moving", False)),
            speed_mps=_optional_finite(motion.get("speed_mps")),
            heading_degrees=_optional_finite(motion.get("heading_degrees")),
        )
        targets.append(target)
    if isinstance(payload, list):
        kraken_name = None
        receiver_latitude = None
        receiver_longitude = None
    return Packet(
        source=source,
        sequence=sequence,
        kraken_name=kraken_name,
        receiver_latitude=receiver_latitude,
        receiver_longitude=receiver_longitude,
        targets=targets,
    )


def receiver_key(packet: Packet, source_ip: str, source_port: int) -> str:
    if packet.kraken_name:
        return packet.kraken_name
    if packet.source:
        return packet.source
    return f"{source_ip}:{source_port}"


def distance_meters(latitude_1: float, longitude_1: float, latitude_2: float, longitude_2: float) -> float:
    """Return great-circle distance between two geographic points."""
    earth_radius = 6_371_000.0
    phi_1 = math.radians(latitude_1)
    phi_2 = math.radians(latitude_2)
    delta_phi = math.radians(latitude_2 - latitude_1)
    delta_lambda = math.radians(longitude_2 - longitude_1)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return earth_radius * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def circular_mean_degrees(angles: list[float]) -> float:
    """Average navigation bearings without breaking at the 0°/360° boundary."""
    if not angles:
        raise ValueError("At least one angle is required")
    sine = sum(math.sin(math.radians(angle)) for angle in angles)
    cosine = sum(math.cos(math.radians(angle)) for angle in angles)
    if abs(sine) < 1e-12 and abs(cosine) < 1e-12:
        return angles[-1] % 360.0
    return math.degrees(math.atan2(sine, cosine)) % 360.0


def angular_distance_degrees(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def robust_weighted_bearing(
    samples: list[tuple[float, float]],
) -> tuple[float, float, int, int]:
    """Return robust RSSI-weighted DoA/RSSI means and accepted/rejected counts.

    A circular medoid provides an outlier-resistant center. Samples far from
    that center are rejected using a median absolute deviation threshold, and
    the remaining angles are averaged with stronger RSSI receiving more weight.
    """
    if not samples:
        raise ValueError("At least one DoA/RSSI sample is required")

    angles = [angle % 360.0 for angle, _rssi in samples]
    center = min(
        angles,
        key=lambda candidate: sum(
            angular_distance_degrees(candidate, other) for other in angles
        ),
    )
    deviations = sorted(angular_distance_degrees(angle, center) for angle in angles)
    median_deviation = deviations[len(deviations) // 2]
    threshold = min(35.0, max(8.0, 3.5 * median_deviation))
    accepted = [
        (angle, rssi)
        for angle, rssi in samples
        if angular_distance_degrees(angle, center) <= threshold
    ]
    if not accepted:
        accepted = [min(samples, key=lambda sample: angular_distance_degrees(sample[0], center))]

    weighted_sine = 0.0
    weighted_cosine = 0.0
    weighted_rssi = 0.0
    total_weight = 0.0
    for angle, rssi in accepted:
        weight = _rssi_weight(rssi)
        weighted_sine += weight * math.sin(math.radians(angle))
        weighted_cosine += weight * math.cos(math.radians(angle))
        weighted_rssi += weight * rssi
        total_weight += weight
    mean_angle = math.degrees(math.atan2(weighted_sine, weighted_cosine)) % 360.0
    mean_rssi = weighted_rssi / total_weight
    return mean_angle, mean_rssi, len(accepted), len(samples) - len(accepted)


def robust_time_bucketed_bearing(
    samples: list[tuple[float, float, float]],
    bucket_seconds: float = 0.5,
) -> tuple[float, float, int, int]:
    """Average an unlimited static history without depending on packet rate.

    ``samples`` contains ``(time_seconds, doa_degrees, rssi_db)`` values.  Each
    fixed-duration bucket first produces one robust representative bearing;
    those representatives are then combined.  Consequently, a receiver that
    happens to send five packets in a time slice does not get five times the
    influence of a receiver/rate that sends one, while every packet still
    contributes to its time bucket.
    """
    bucketed = bucket_timestamped_bearings(samples, bucket_seconds)
    representatives = [(item.doa_degrees, item.rssi_db) for item in bucketed]
    within_bucket_rejected = sum(item.rejected_samples for item in bucketed)

    angle, rssi, accepted_buckets, rejected_buckets = robust_weighted_bearing(
        representatives
    )
    # Packet counts remain meaningful UI metrics. A rejected time bucket is
    # conservatively counted as one additional rejected observation.
    rejected = min(len(samples), within_bucket_rejected + rejected_buckets)
    return angle, rssi, len(samples) - rejected, rejected


def bucket_timestamped_bearings(
    samples: list[tuple[float, float, float]],
    bucket_seconds: float = 0.5,
) -> list[BucketedBearing]:
    """Return one robust bearing per fixed time bucket.

    This is the common packet-rate normalization used by the static estimator:
    ten packets in half a second have the same temporal influence as one packet
    in another half-second bucket.
    """
    if not samples:
        raise ValueError("At least one timestamped DoA/RSSI sample is required")
    if not math.isfinite(bucket_seconds) or bucket_seconds <= 0.0:
        raise ValueError("bucket_seconds must be a positive finite value")

    ordered = sorted(samples, key=lambda item: item[0])
    origin = ordered[0][0]
    buckets: dict[int, list[tuple[float, float, float]]] = {}
    for timestamp, angle, rssi in ordered:
        bucket = int(math.floor(max(0.0, timestamp - origin) / bucket_seconds))
        buckets.setdefault(bucket, []).append((timestamp, angle, rssi))

    result: list[BucketedBearing] = []
    for bucket_samples in buckets.values():
        angle, rssi, accepted, rejected = robust_weighted_bearing(
            [(sample[1], sample[2]) for sample in bucket_samples]
        )
        result.append(
            BucketedBearing(
                timestamp_seconds=statistics.median(
                    sample[0] for sample in bucket_samples
                ),
                doa_degrees=angle,
                rssi_db=rssi,
                accepted_samples=accepted,
                rejected_samples=rejected,
            )
        )
    return result


def continuous_time_alpha(delta_seconds: float, time_constant_seconds: float) -> float:
    """Convert a time constant to a packet-rate-independent smoothing gain."""
    if not math.isfinite(delta_seconds) or delta_seconds <= 0.0:
        return 0.0
    if not math.isfinite(time_constant_seconds) or time_constant_seconds <= 0.0:
        return 1.0
    return 1.0 - math.exp(-delta_seconds / time_constant_seconds)


def analyze_observed_motion(
    history: list[tuple[float, float, float]],
    previously_moving: bool,
    lookback_seconds: float | None = None,
) -> MotionAnalysis:
    """Infer STATIC/TRACK from noisy raw solutions without ground truth.

    Timestamps are milliseconds. Raw positions are first reduced to one robust
    median point per fixed-duration bucket, then a Theil-Sen-style velocity is
    estimated from pairwise slopes. The decision compares sustained progress
    with the observed scatter so isolated localization jumps do not look like
    target motion. Hysteresis uses a shorter window when already tracking.
    """
    if len(history) < 2:
        return MotionAnalysis(previously_moving)
    ordered = sorted(history, key=lambda item: item[0])
    last = ordered[-1]
    window = lookback_seconds or (4.0 if previously_moving else 8.0)
    cutoff = last[0] - max(2.0, window) * 1000.0
    recent = [item for item in ordered if item[0] >= cutoff]
    if len(recent) < 2:
        return MotionAnalysis(previously_moving)

    reference_latitude = statistics.median(item[1] for item in recent)
    reference_longitude = statistics.median(item[2] for item in recent)
    origin_timestamp = recent[0][0]
    buckets: dict[int, list[tuple[float, float, float]]] = {}
    bucket_milliseconds = 750.0
    for timestamp, latitude, longitude in recent:
        x, y = _latlon_to_xy(
            latitude,
            longitude,
            reference_latitude,
            reference_longitude,
        )
        bucket = int(math.floor(timestamp / bucket_milliseconds))
        buckets.setdefault(bucket, []).append(
            ((timestamp - origin_timestamp) / 1000.0, x, y)
        )
    points = sorted(
        (
            statistics.median(sample[0] for sample in samples),
            statistics.median(sample[1] for sample in samples),
            statistics.median(sample[2] for sample in samples),
        )
        for samples in buckets.values()
    )
    if len(points) < 3:
        return MotionAnalysis(previously_moving)
    span_seconds = points[-1][0] - points[0][0]
    minimum_span = 2.25 if previously_moving else 3.5
    if span_seconds < minimum_span:
        return MotionAnalysis(previously_moving, span_seconds=span_seconds)

    east_slopes: list[float] = []
    north_slopes: list[float] = []
    for index, first in enumerate(points):
        for second in points[index + 1 :]:
            delta_time = second[0] - first[0]
            if delta_time < 0.70:
                continue
            east_slopes.append((second[1] - first[1]) / delta_time)
            north_slopes.append((second[2] - first[2]) / delta_time)
    if not east_slopes:
        return MotionAnalysis(previously_moving, span_seconds=span_seconds)

    east_velocity = statistics.median(east_slopes)
    north_velocity = statistics.median(north_slopes)
    speed_mps = math.hypot(east_velocity, north_velocity)
    heading_degrees = math.degrees(math.atan2(east_velocity, north_velocity)) % 360.0
    east_intercept = statistics.median(
        x - east_velocity * timestamp for timestamp, x, _y in points
    )
    north_intercept = statistics.median(
        y - north_velocity * timestamp for timestamp, _x, y in points
    )
    residuals = [
        math.hypot(
            x - (east_intercept + east_velocity * timestamp),
            y - (north_intercept + north_velocity * timestamp),
        )
        for timestamp, x, y in points
    ]
    scatter_meters = max(1.0, statistics.median(residuals))
    fitted_displacement = speed_mps * span_seconds

    if speed_mps > 0.05:
        unit_east = east_velocity / speed_mps
        unit_north = north_velocity / speed_mps
        endpoint_progress = (
            (points[-1][1] - points[0][1]) * unit_east
            + (points[-1][2] - points[0][2]) * unit_north
        )
    else:
        endpoint_progress = 0.0
    signal_to_scatter = fitted_displacement / scatter_meters
    confidence = max(0.0, min(1.0, (signal_to_scatter - 0.8) / 2.5))
    projected_x = east_intercept + east_velocity * points[-1][0]
    projected_y = north_intercept + north_velocity * points[-1][0]
    projected_latitude, projected_longitude = _xy_to_latlon(
        projected_x,
        projected_y,
        reference_latitude,
        reference_longitude,
    )

    if previously_moving:
        moving = (
            speed_mps >= 0.45
            and fitted_displacement >= max(2.5, 0.85 * scatter_meters)
            and endpoint_progress >= max(1.5, 0.55 * scatter_meters)
        )
    else:
        moving = (
            speed_mps >= 0.80
            and fitted_displacement >= max(5.0, 1.60 * scatter_meters)
            and endpoint_progress >= max(4.0, 1.20 * scatter_meters)
        )
    return MotionAnalysis(
        moving=moving,
        speed_mps=speed_mps,
        heading_degrees=heading_degrees,
        confidence=confidence,
        span_seconds=span_seconds,
        scatter_meters=scatter_meters,
        displacement_meters=fitted_displacement,
        endpoint_progress_meters=endpoint_progress,
        projected_latitude=projected_latitude,
        projected_longitude=projected_longitude,
    )


def analyze_bearing_motion(
    histories: dict[str, list[tuple[float, float]]],
    lookback_seconds: float = 12.0,
) -> BearingMotionAnalysis:
    """Detect sustained motion directly from multiple DoA histories.

    Each receiver is reduced to robust 0.75-second angular buckets. Angles are
    unwrapped and a Theil-Sen median slope is fitted, so isolated multipath
    bearings do not look like movement. The strongest reliable receiver is
    used because motion can be radial (near-zero angular rate) relative to one
    receiver while remaining visible from the other.
    """
    newest_timestamp = max(
        (timestamp for samples in histories.values() for timestamp, _angle in samples),
        default=None,
    )
    if newest_timestamp is None:
        return BearingMotionAnalysis(False)
    cutoff = newest_timestamp - max(4.0, lookback_seconds) * 1000.0
    receiver_results: list[tuple[float, float, float]] = []

    for samples in histories.values():
        recent = sorted(sample for sample in samples if sample[0] >= cutoff)
        if len(recent) < 4:
            continue
        buckets: dict[int, list[tuple[float, float]]] = {}
        for timestamp, angle in recent:
            buckets.setdefault(int(math.floor(timestamp / 750.0)), []).append(
                (timestamp, angle % 360.0)
            )
        bucket_points = []
        for bucket_samples in buckets.values():
            angles = [angle for _timestamp, angle in bucket_samples]
            medoid = min(
                angles,
                key=lambda candidate: sum(
                    angular_distance_degrees(candidate, other) for other in angles
                ),
            )
            bucket_points.append(
                (
                    statistics.median(timestamp for timestamp, _angle in bucket_samples),
                    medoid,
                )
            )
        bucket_points.sort()
        if len(bucket_points) < 5:
            continue
        origin = bucket_points[0][0]
        unwrapped: list[tuple[float, float]] = []
        previous_wrapped = bucket_points[0][1]
        previous_unwrapped = previous_wrapped
        for timestamp, wrapped in bucket_points:
            if unwrapped:
                delta = (wrapped - previous_wrapped + 180.0) % 360.0 - 180.0
                previous_unwrapped += delta
                previous_wrapped = wrapped
            unwrapped.append(((timestamp - origin) / 1000.0, previous_unwrapped))
        span_seconds = unwrapped[-1][0] - unwrapped[0][0]
        if span_seconds < 4.0:
            continue
        slopes = []
        for index, first in enumerate(unwrapped):
            for second in unwrapped[index + 1 :]:
                delta_time = second[0] - first[0]
                if delta_time >= 0.70:
                    slopes.append((second[1] - first[1]) / delta_time)
        if not slopes:
            continue
        signed_rate = statistics.median(slopes)
        intercept = statistics.median(
            angle - signed_rate * timestamp for timestamp, angle in unwrapped
        )
        residuals = [
            abs(angle - (intercept + signed_rate * timestamp))
            for timestamp, angle in unwrapped
        ]
        scatter_degrees = max(0.5, statistics.median(residuals))
        angular_rate = abs(signed_rate)
        trend_degrees = angular_rate * span_seconds
        signal_ratio = trend_degrees / max(1.0, 2.0 * scatter_degrees)
        confidence = max(0.0, min(1.0, (signal_ratio - 0.5) / 2.0))
        receiver_results.append((angular_rate, confidence, span_seconds))

    if not receiver_results:
        return BearingMotionAnalysis(False)
    reliable = max(
        receiver_results,
        key=lambda item: item[0] * (0.25 + 0.75 * item[1]),
    )
    angular_rate, confidence, span_seconds = reliable
    moving = angular_rate >= 0.75 and confidence >= 0.50
    return BearingMotionAnalysis(
        moving=moving,
        angular_rate_dps=angular_rate,
        confidence=confidence,
        span_seconds=span_seconds,
        receiver_count=len(receiver_results),
    )


def estimate_static_target(
    name: str,
    measurements: list[BearingMeasurement],
) -> EstimatedTarget | None:
    """Robust batch bearing estimator for a stationary target.

    Unlike averaging already-triangulated positions, this optimizer keeps the
    observations in their native angular domain. A Huber M-estimator limits
    multipath/outlier influence and the RSSI contribution remains deliberately
    weak until real receiver-specific noise calibration is available.
    """
    receiver_groups: dict[str, list[BearingMeasurement]] = {}
    for measurement in measurements:
        receiver_groups.setdefault(measurement.receiver_name, []).append(measurement)
    if len(receiver_groups) < 2:
        return None

    filtered: list[BearingMeasurement] = []
    representative_measurements: list[BearingMeasurement] = []
    for receiver_measurements in receiver_groups.values():
        angles = [item.doa_degrees % 360.0 for item in receiver_measurements]
        center = min(
            angles,
            key=lambda candidate: sum(
                angular_distance_degrees(candidate, other) for other in angles
            ),
        )
        deviations = sorted(
            angular_distance_degrees(angle, center) for angle in angles
        )
        median_deviation = deviations[len(deviations) // 2]
        threshold = min(15.0, max(3.0, 4.0 * median_deviation))
        accepted = [
            item
            for item in receiver_measurements
            if angular_distance_degrees(item.doa_degrees, center) <= threshold
        ]
        if not accepted:
            accepted = [
                min(
                    receiver_measurements,
                    key=lambda item: angular_distance_degrees(
                        item.doa_degrees, center
                    ),
                )
            ]
        filtered.extend(accepted)
        mean_doa, mean_rssi, _accepted, _rejected = robust_weighted_bearing(
            [(item.doa_degrees, item.rssi_db) for item in accepted]
        )
        template = accepted[-1]
        representative_measurements.append(
            BearingMeasurement(
                receiver_name=template.receiver_name,
                receiver_latitude=template.receiver_latitude,
                receiver_longitude=template.receiver_longitude,
                doa_degrees=mean_doa,
                rssi_db=mean_rssi,
            )
        )

    initial = estimate_target(name, representative_measurements)
    if initial is None:
        return None
    reference_latitude = statistics.mean(
        item.receiver_latitude for item in representative_measurements
    )
    reference_longitude = statistics.mean(
        item.receiver_longitude for item in representative_measurements
    )
    x, y = _latlon_to_xy(
        initial.latitude,
        initial.longitude,
        reference_latitude,
        reference_longitude,
    )

    for _iteration in range(15):
        normal_00 = normal_01 = normal_11 = 0.0
        rhs_0 = rhs_1 = 0.0
        for item in filtered:
            receiver_x, receiver_y = _latlon_to_xy(
                item.receiver_latitude,
                item.receiver_longitude,
                reference_latitude,
                reference_longitude,
            )
            delta_x = x - receiver_x
            delta_y = y - receiver_y
            range_squared = max(25.0, delta_x * delta_x + delta_y * delta_y)
            predicted = math.atan2(delta_x, delta_y)
            observed = math.radians(item.doa_degrees)
            residual = _wrapped_radians(observed - predicted)
            jacobian_x = delta_y / range_squared
            jacobian_y = -delta_x / range_squared
            sigma = _bearing_sigma_radians(item.rssi_db)
            standardized = abs(residual) / sigma
            huber_weight = 1.0 if standardized <= 2.5 else 2.5 / standardized
            weight = _rssi_weight(item.rssi_db) * huber_weight / (sigma * sigma)
            normal_00 += weight * jacobian_x * jacobian_x
            normal_01 += weight * jacobian_x * jacobian_y
            normal_11 += weight * jacobian_y * jacobian_y
            rhs_0 += weight * jacobian_x * residual
            rhs_1 += weight * jacobian_y * residual
        normal_00 += 1e-9
        normal_11 += 1e-9
        determinant = normal_00 * normal_11 - normal_01 * normal_01
        if abs(determinant) < 1e-15:
            break
        step_x = (rhs_0 * normal_11 - normal_01 * rhs_1) / determinant
        step_y = (normal_00 * rhs_1 - rhs_0 * normal_01) / determinant
        step_length = math.hypot(step_x, step_y)
        if step_length > 150.0:
            scale = 150.0 / step_length
            step_x *= scale
            step_y *= scale
        x += step_x
        y += step_y
        if math.hypot(step_x, step_y) < 0.01:
            break

    latitude, longitude = _xy_to_latlon(
        x,
        y,
        reference_latitude,
        reference_longitude,
    )
    cross_track_residuals: list[float] = []
    for item in filtered:
        receiver_x, receiver_y = _latlon_to_xy(
            item.receiver_latitude,
            item.receiver_longitude,
            reference_latitude,
            reference_longitude,
        )
        theta = math.radians(item.doa_degrees)
        cross_track_residuals.append(
            abs((x - receiver_x) * math.cos(theta) - (y - receiver_y) * math.sin(theta))
        )
    return EstimatedTarget(
        name=name,
        latitude=latitude,
        longitude=longitude,
        measurement_count=len(filtered),
        mean_rssi_db=statistics.mean(item.rssi_db for item in filtered),
        residual_meters=statistics.mean(cross_track_residuals),
        geometry_score=initial.geometry_score,
        geometry_label=initial.geometry_label,
    )


def initialize_bearing_track(
    estimate: EstimatedTarget,
    measurements: list[BearingMeasurement],
    timestamp_ms: float,
    east_velocity_mps: float = 0.0,
    north_velocity_mps: float = 0.0,
) -> BearingTrackState:
    """Initialize a real/simulator-agnostic bearing-only EKF."""
    reference_latitude = statistics.mean(
        item.receiver_latitude for item in measurements
    )
    reference_longitude = statistics.mean(
        item.receiver_longitude for item in measurements
    )
    east, north = _latlon_to_xy(
        estimate.latitude,
        estimate.longitude,
        reference_latitude,
        reference_longitude,
    )
    geometry_uncertainty = max(25.0, 100.0 - estimate.geometry_score)
    position_variance = geometry_uncertainty * geometry_uncertainty
    return BearingTrackState(
        reference_latitude=reference_latitude,
        reference_longitude=reference_longitude,
        east_meters=east,
        north_meters=north,
        east_velocity_mps=east_velocity_mps,
        north_velocity_mps=north_velocity_mps,
        covariance=[
            [position_variance, 0.0, 0.0, 0.0],
            [0.0, position_variance, 0.0, 0.0],
            [0.0, 0.0, 100.0, 0.0],
            [0.0, 0.0, 0.0, 100.0],
        ],
        timestamp_ms=timestamp_ms,
    )


def update_bearing_track(
    state: BearingTrackState,
    name: str,
    measurements: list[BearingMeasurement],
    timestamp_ms: float,
    acceleration_sigma_mps2: float = 3.0,
    maximum_speed_mps: float = 35.0,
) -> tuple[BearingTrackState, EstimatedTarget, int, int]:
    """Predict and correct a constant-velocity EKF directly with DoA values."""
    raw = estimate_target(name, measurements)
    delta_time = (timestamp_ms - state.timestamp_ms) / 1000.0
    if delta_time <= 0.0:
        latitude, longitude = _xy_to_latlon(
            state.east_meters,
            state.north_meters,
            state.reference_latitude,
            state.reference_longitude,
        )
        template = raw or EstimatedTarget(name, latitude, longitude, 0, -115.0, 0.0)
        return (
            state,
            EstimatedTarget(
                name=name,
                latitude=latitude,
                longitude=longitude,
                measurement_count=len(measurements),
                mean_rssi_db=template.mean_rssi_db,
                residual_meters=template.residual_meters,
                geometry_score=template.geometry_score,
                geometry_label=template.geometry_label,
            ),
            0,
            0,
        )
    delta_time = min(3.0, max(0.02, delta_time))
    state_vector = [
        state.east_meters + state.east_velocity_mps * delta_time,
        state.north_meters + state.north_velocity_mps * delta_time,
        state.east_velocity_mps,
        state.north_velocity_mps,
    ]
    transition = [
        [1.0, 0.0, delta_time, 0.0],
        [0.0, 1.0, 0.0, delta_time],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    covariance = _matrix_multiply(
        _matrix_multiply(transition, state.covariance),
        _matrix_transpose(transition),
    )
    acceleration_variance = max(0.25, acceleration_sigma_mps2**2)
    dt2 = delta_time * delta_time
    dt3 = dt2 * delta_time
    dt4 = dt2 * dt2
    process_noise = [
        [0.25 * dt4, 0.0, 0.5 * dt3, 0.0],
        [0.0, 0.25 * dt4, 0.0, 0.5 * dt3],
        [0.5 * dt3, 0.0, dt2, 0.0],
        [0.0, 0.5 * dt3, 0.0, dt2],
    ]
    covariance = [
        [
            covariance[row][column]
            + acceleration_variance * process_noise[row][column]
            for column in range(4)
        ]
        for row in range(4)
    ]

    accepted = 0
    rejected = 0
    for item in sorted(measurements, key=lambda measurement: measurement.receiver_name):
        receiver_east, receiver_north = _latlon_to_xy(
            item.receiver_latitude,
            item.receiver_longitude,
            state.reference_latitude,
            state.reference_longitude,
        )
        delta_east = state_vector[0] - receiver_east
        delta_north = state_vector[1] - receiver_north
        range_squared = max(25.0, delta_east * delta_east + delta_north * delta_north)
        predicted_bearing = math.atan2(delta_east, delta_north)
        innovation = _wrapped_radians(
            math.radians(item.doa_degrees) - predicted_bearing
        )
        observation = [
            delta_north / range_squared,
            -delta_east / range_squared,
            0.0,
            0.0,
        ]
        covariance_observation = [
            sum(covariance[row][column] * observation[column] for column in range(4))
            for row in range(4)
        ]
        measurement_variance = _bearing_sigma_radians(item.rssi_db) ** 2
        innovation_variance = (
            sum(observation[index] * covariance_observation[index] for index in range(4))
            + measurement_variance
        )
        if innovation_variance <= 1e-12:
            rejected += 1
            continue
        normalized_innovation = innovation * innovation / innovation_variance
        if normalized_innovation > 16.0:
            rejected += 1
            continue
        if normalized_innovation > 6.25:
            innovation = math.copysign(
                2.5 * math.sqrt(innovation_variance), innovation
            )
        gain = [value / innovation_variance for value in covariance_observation]
        state_vector = [
            state_vector[index] + gain[index] * innovation for index in range(4)
        ]

        # Joseph covariance update preserves symmetry and positive variance.
        identity_minus_gain_h = [
            [
                (1.0 if row == column else 0.0)
                - gain[row] * observation[column]
                for column in range(4)
            ]
            for row in range(4)
        ]
        covariance = _matrix_multiply(
            _matrix_multiply(identity_minus_gain_h, covariance),
            _matrix_transpose(identity_minus_gain_h),
        )
        covariance = [
            [
                covariance[row][column]
                + gain[row] * measurement_variance * gain[column]
                for column in range(4)
            ]
            for row in range(4)
        ]
        accepted += 1

    speed = math.hypot(state_vector[2], state_vector[3])
    speed_limit = max(1.0, float(maximum_speed_mps))
    if speed > speed_limit:
        scale = speed_limit / speed
        state_vector[2] *= scale
        state_vector[3] *= scale
    coast_seconds = 0.0 if accepted else state.coast_seconds + delta_time
    updated = BearingTrackState(
        reference_latitude=state.reference_latitude,
        reference_longitude=state.reference_longitude,
        east_meters=state_vector[0],
        north_meters=state_vector[1],
        east_velocity_mps=state_vector[2],
        north_velocity_mps=state_vector[3],
        covariance=_symmetrize_covariance(covariance),
        timestamp_ms=timestamp_ms,
        coast_seconds=coast_seconds,
    )
    latitude, longitude = _xy_to_latlon(
        updated.east_meters,
        updated.north_meters,
        updated.reference_latitude,
        updated.reference_longitude,
    )
    if raw is None:
        mean_rssi = statistics.mean(item.rssi_db for item in measurements)
        residual_meters = 0.0
        geometry_score = 0.0
        geometry_label = "POOR"
    else:
        mean_rssi = raw.mean_rssi_db
        residual_meters = raw.residual_meters
        geometry_score = raw.geometry_score
        geometry_label = raw.geometry_label
    return (
        updated,
        EstimatedTarget(
            name=name,
            latitude=latitude,
            longitude=longitude,
            measurement_count=len(measurements),
            mean_rssi_db=mean_rssi,
            residual_meters=residual_meters,
            geometry_score=geometry_score,
            geometry_label=geometry_label,
        ),
        accepted,
        rejected,
    )


def estimate_target(name: str, measurements: list[BearingMeasurement]) -> EstimatedTarget | None:
    """Estimate a target position from two or more receiver DoA bearings.

    The solver uses a local tangent-plane approximation around the receiver
    centroid and finds the least-squares point closest to all bearing lines.
    This is accurate enough for the simulator-scale maps we use in the thesis
    GUI, and it remains stable when more than two receivers are present.
    """
    if len(measurements) < 2:
        return None

    reference_latitude = sum(item.receiver_latitude for item in measurements) / len(measurements)
    reference_longitude = sum(item.receiver_longitude for item in measurements) / len(measurements)
    normal_matrix = [[0.0, 0.0], [0.0, 0.0]]
    rhs = [0.0, 0.0]
    projected_lines: list[tuple[float, float, float, float, float]] = []

    for item in measurements:
        px, py = _latlon_to_xy(item.receiver_latitude, item.receiver_longitude, reference_latitude, reference_longitude)
        theta = math.radians(item.doa_degrees)
        dx = math.sin(theta)
        dy = math.cos(theta)
        weight = _rssi_weight(item.rssi_db)
        projector = ((1.0 - dx * dx, -dx * dy), (-dx * dy, 1.0 - dy * dy))
        normal_matrix[0][0] += weight * projector[0][0]
        normal_matrix[0][1] += weight * projector[0][1]
        normal_matrix[1][0] += weight * projector[1][0]
        normal_matrix[1][1] += weight * projector[1][1]
        rhs[0] += weight * (projector[0][0] * px + projector[0][1] * py)
        rhs[1] += weight * (projector[1][0] * px + projector[1][1] * py)
        projected_lines.append((px, py, dx, dy, weight))

    determinant = normal_matrix[0][0] * normal_matrix[1][1] - normal_matrix[0][1] * normal_matrix[1][0]
    if abs(determinant) < 1e-6:
        return None

    x = (rhs[0] * normal_matrix[1][1] - normal_matrix[0][1] * rhs[1]) / determinant
    y = (normal_matrix[0][0] * rhs[1] - rhs[0] * normal_matrix[1][0]) / determinant
    robust_pairwise = _pairwise_intersection_estimate(projected_lines)
    if robust_pairwise is not None:
        x, y = robust_pairwise
    latitude, longitude = _xy_to_latlon(x, y, reference_latitude, reference_longitude)
    geometry_score = _geometry_score(projected_lines)
    residuals = []
    for px, py, dx, dy, _weight in projected_lines:
        cross_track = abs((x - px) * dy - (y - py) * dx)
        residuals.append(cross_track)
    mean_rssi = sum(item.rssi_db for item in measurements) / len(measurements)
    return EstimatedTarget(
        name=name,
        latitude=latitude,
        longitude=longitude,
        measurement_count=len(measurements),
        mean_rssi_db=mean_rssi,
        residual_meters=sum(residuals) / len(residuals),
        geometry_score=geometry_score,
        geometry_label=_geometry_label(geometry_score),
    )


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _required_finite(value: Any, index: int, field_name: str) -> float:
    result = _optional_finite(value)
    if result is None:
        raise PacketError(f"Target {index} has invalid {field_name}")
    return result


def _latlon_to_xy(latitude: float, longitude: float, reference_latitude: float, reference_longitude: float) -> tuple[float, float]:
    earth_radius = 6_371_000.0
    x = math.radians(longitude - reference_longitude) * earth_radius * math.cos(math.radians(reference_latitude))
    y = math.radians(latitude - reference_latitude) * earth_radius
    return x, y


def _xy_to_latlon(x: float, y: float, reference_latitude: float, reference_longitude: float) -> tuple[float, float]:
    earth_radius = 6_371_000.0
    latitude = reference_latitude + math.degrees(y / earth_radius)
    longitude = reference_longitude + math.degrees(x / (earth_radius * math.cos(math.radians(reference_latitude))))
    return latitude, longitude


def _wrapped_radians(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _bearing_sigma_radians(rssi_db: float) -> float:
    """RSSI-dependent DoA uncertainty used by the robust estimators.

    Two-receiver bearing tracking becomes unstable when medium/weak signals
    are treated as if they were laboratory-quality 1–2 degree bearings.  The
    piecewise curve deliberately matches the order of uncertainty produced by
    the project's realistic channel model, without reading simulator state.
    """
    rssi = max(-115.0, min(-15.0, rssi_db))
    if rssi >= -45.0:
        sigma_degrees = 1.5
    elif rssi >= -60.0:
        sigma_degrees = 1.5 + (-45.0 - rssi) / 15.0 * 1.5
    elif rssi >= -75.0:
        sigma_degrees = 3.0 + (-60.0 - rssi) / 15.0 * 2.5
    elif rssi >= -90.0:
        sigma_degrees = 5.5 + (-75.0 - rssi) / 15.0 * 3.5
    else:
        sigma_degrees = 10.0
    return math.radians(max(1.5, min(10.0, sigma_degrees)))


def _matrix_multiply(first: list[list[float]], second: list[list[float]]) -> list[list[float]]:
    return [
        [
            sum(first[row][index] * second[index][column] for index in range(len(second)))
            for column in range(len(second[0]))
        ]
        for row in range(len(first))
    ]


def _matrix_transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(column) for column in zip(*matrix)]


def _symmetrize_covariance(covariance: list[list[float]]) -> list[list[float]]:
    result = [[0.0] * 4 for _ in range(4)]
    for row in range(4):
        for column in range(4):
            result[row][column] = 0.5 * (
                covariance[row][column] + covariance[column][row]
            )
        result[row][row] = max(1e-9, result[row][row])
    return result


def _rssi_weight(rssi_db: float) -> float:
    # RSSI in this project is normally around -115..-15 dB. Stronger signals
    # receive slightly more influence, but weak signals are never ignored.
    normalized = (max(-115.0, min(-15.0, rssi_db)) + 115.0) / 100.0
    return 0.25 + 0.75 * normalized


def _pairwise_intersection_estimate(lines: list[tuple[float, float, float, float, float]]) -> tuple[float, float] | None:
    intersections: list[tuple[float, float, float]] = []
    for index, first in enumerate(lines):
        p1x, p1y, d1x, d1y, w1 = first
        for p2x, p2y, d2x, d2y, w2 in lines[index + 1:]:
            determinant = d1x * d2y - d1y * d2x
            angle_quality = abs(determinant)
            # Nearly parallel bearings produce huge unstable intersections.
            if angle_quality < 0.12:
                continue
            qpx = p2x - p1x
            qpy = p2y - p1y
            t = (qpx * d2y - qpy * d2x) / determinant
            x = p1x + t * d1x
            y = p1y + t * d1y
            intersections.append((x, y, angle_quality * angle_quality * (w1 + w2) / 2.0))
    if not intersections:
        return None

    def weighted_mean(points: list[tuple[float, float, float]]) -> tuple[float, float]:
        total_weight = sum(weight for _x, _y, weight in points)
        return (
            sum(x * weight for x, _y, weight in points) / total_weight,
            sum(y * weight for _x, y, weight in points) / total_weight,
        )

    x, y = weighted_mean(intersections)
    if len(intersections) < 4:
        return x, y

    distances = sorted(math.hypot(ix - x, iy - y) for ix, iy, _weight in intersections)
    median_distance = distances[len(distances) // 2]
    keep_radius = max(25.0, median_distance * 2.5)
    filtered = [
        (ix, iy, weight)
        for ix, iy, weight in intersections
        if math.hypot(ix - x, iy - y) <= keep_radius
    ]
    if len(filtered) >= 2:
        return weighted_mean(filtered)
    return x, y


def _geometry_score(lines: list[tuple[float, float, float, float, float]]) -> float:
    """Return a 0..100 score for how well bearing lines constrain a target.

    Bearings that cross close to 90 degrees produce stable localization. Nearly
    parallel bearings have poor geometry because tiny DoA noise creates large
    position jumps. The score is weighted by RSSI-derived line weights and by
    receiver baseline so that distant, well-separated receivers are favored.
    """
    if len(lines) < 2:
        return 0.0

    total_weight = 0.0
    weighted_quality = 0.0
    for index, first in enumerate(lines):
        p1x, p1y, d1x, d1y, w1 = first
        for p2x, p2y, d2x, d2y, w2 in lines[index + 1:]:
            angle_quality = abs(d1x * d2y - d1y * d2x)
            baseline_meters = math.hypot(p2x - p1x, p2y - p1y)
            baseline_quality = min(1.0, baseline_meters / 250.0)
            weight = ((w1 + w2) / 2.0) * (0.35 + 0.65 * baseline_quality)
            weighted_quality += angle_quality * weight
            total_weight += weight
    if total_weight <= 0.0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * weighted_quality / total_weight))


def _geometry_label(score: float) -> str:
    if score >= 70.0:
        return "GOOD"
    if score >= 40.0:
        return "FAIR"
    return "POOR"
