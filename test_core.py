import math
import random
import statistics

import pytest

from core import (
    BearingMeasurement,
    PacketError,
    analyze_observed_motion,
    bucket_timestamped_bearings,
    circular_mean_degrees,
    continuous_time_alpha,
    distance_meters,
    estimate_static_target,
    estimate_target,
    initialize_bearing_track,
    parse_packet,
    receiver_key,
    robust_time_bucketed_bearing,
    robust_weighted_bearing,
    update_bearing_track,
)


def test_legacy_packet():
    packet = parse_packet(b'[{"timestamp":1,"frequency":446.1,"rssi":-45,"doa":-10}]')
    assert packet.source is None
    assert packet.kraken_name is None
    assert packet.targets[0].doa_degrees == 350


def test_metadata_packet_and_source():
    packet = parse_packet(b'{"source":"simulator","sequence":7,"targets":[{"timestamp":1,"frequency":446.1,"rssi":-45,"doa":20}]}')
    assert packet.sequence == 7
    assert receiver_key(packet, "10.0.0.9", 4200) == "simulator"


def test_receiver_packet_identity():
    packet = parse_packet(
        b'{"schema_version":1,"kraken_name":"Kraken-A","sequence":3,'
        b'"receiver":{"name":"Kraken-A","latitude":35.0,"longitude":33.0},'
        b'"targets":[{"timestamp":1,"frequency":446.1,"rssi":-45,"doa":20}]}'
    )
    assert packet.kraken_name == "Kraken-A"
    assert packet.receiver_latitude == 35.0
    assert packet.receiver_longitude == 33.0
    assert receiver_key(packet, "127.0.0.1", 4200) == "Kraken-A"


def test_evaluation_ground_truth_is_optional_and_parsed():
    packet = parse_packet(
        b'{"kraken_name":"Kraken-A","receiver":{"latitude":35,"longitude":33},'
        b'"targets":[{"timestamp":10,"frequency":446.1,"rssi":-45,"doa":20,'
        b'"evaluation":{"target_name":"Drone","latitude":35.001,"longitude":33.002,'
        b'"moving":true,"speed_mps":12.5,"heading_degrees":90}}]}'
    )
    target = packet.targets[0]
    assert target.name == "Drone"
    assert target.truth_latitude == 35.001
    assert target.truth_longitude == 33.002
    assert target.moving is True
    assert target.speed_mps == 12.5
    assert target.heading_degrees == 90.0


def test_source_address_fallback():
    packet = parse_packet(b'[]')
    assert receiver_key(packet, "172.17.0.1", 4200) == "172.17.0.1:4200"


def test_motion_metadata_does_not_require_ground_truth():
    packet = parse_packet(
        b'{"targets":[{"frequency":446.1,"rssi":-50,"doa":45,'
        b'"motion":{"moving":true,"speed_mps":8,"heading_degrees":180}}]}'
    )
    target = packet.targets[0]
    assert target.moving is True
    assert target.speed_mps == 8.0
    assert target.heading_degrees == 180.0
    assert target.truth_latitude is None


def test_invalid_target_rejected():
    with pytest.raises(PacketError):
        parse_packet(b'[{"frequency":null,"rssi":-45,"doa":20}]')


def test_localization_from_two_bearings():
    estimate = estimate_target(
        "Target-1",
        [
            BearingMeasurement("A", 35.0, 33.0, 45.0, -40.0),
            BearingMeasurement("B", 35.0, 33.01, 315.0, -41.0),
        ],
    )
    assert estimate is not None
    assert estimate.latitude > 35.0
    assert math.isclose(estimate.longitude, 33.005, abs_tol=0.0002)
    assert estimate.geometry_score > 40.0
    assert estimate.geometry_label in {"FAIR", "GOOD"}


def test_distance_meters():
    assert 90.0 < distance_meters(35.0, 33.0, 35.0, 33.001) < 92.0


def test_circular_mean_wraps_across_north():
    result = circular_mean_degrees([358.0, 359.0, 0.0, 1.0, 2.0])
    assert result < 0.01 or result > 359.99


def test_robust_weighted_bearing_rejects_extreme_outlier():
    angle, rssi, accepted, rejected = robust_weighted_bearing(
        [(359.0, -45.0), (1.0, -44.0), (0.0, -46.0), (120.0, -90.0)]
    )
    assert angle < 2.0 or angle > 358.0
    assert -47.0 < rssi < -43.0
    assert accepted == 3
    assert rejected == 1


def test_time_bucketed_static_bearing_is_packet_rate_independent():
    sparse = [
        (0.0, 9.0, -50.0),
        (0.5, 11.0, -50.0),
        (1.0, 9.0, -50.0),
        (1.5, 11.0, -50.0),
    ]
    dense = [
        (timestamp + offset, angle, rssi)
        for timestamp, angle, rssi in sparse
        for offset in (0.0, 0.08, 0.16, 0.24)
    ]
    sparse_angle, _rssi, _accepted, _rejected = robust_time_bucketed_bearing(sparse)
    dense_angle, _rssi, _accepted, _rejected = robust_time_bucketed_bearing(dense)
    assert math.isclose(sparse_angle, dense_angle, abs_tol=0.05)


def test_continuous_alpha_has_same_effect_across_packet_intervals():
    one_second_gain = continuous_time_alpha(1.0, 1.2)
    short_gain = continuous_time_alpha(0.2, 1.2)
    composed_gain = 1.0 - (1.0 - short_gain) ** 5
    assert math.isclose(one_second_gain, composed_gain, rel_tol=1e-12)


@pytest.mark.parametrize("interval_seconds", [0.2, 0.5, 1.0])
def test_motion_detection_uses_elapsed_time_not_sample_count(interval_seconds):
    latitude = 35.0
    longitude = 33.0
    history = []
    elapsed = 0.0
    while elapsed <= 9.0 + 1e-9:
        north_meters = elapsed * 2.0
        history.append(
            (elapsed * 1000.0, latitude + north_meters / 111_320.0, longitude)
        )
        elapsed += interval_seconds
    analysis = analyze_observed_motion(history, previously_moving=False)
    assert analysis.moving
    assert 1.5 < analysis.speed_mps < 2.5


@pytest.mark.parametrize("interval_seconds", [0.2, 0.5, 1.0])
def test_static_detection_uses_elapsed_time_not_sample_count(interval_seconds):
    history = [
        (elapsed * 1000.0, 35.0, 33.0)
        for elapsed in [index * interval_seconds for index in range(8)]
    ]
    assert not analyze_observed_motion(history, previously_moving=False).moving


@pytest.mark.parametrize("interval_seconds", [0.2, 0.5, 1.0])
def test_observed_motion_rejects_static_localization_jitter(interval_seconds):
    jitter = [(-2.0, 1.0), (1.5, -1.5), (0.5, 2.0), (-1.0, -0.5)]
    history = []
    elapsed = 0.0
    index = 0
    while elapsed <= 10.0 + 1e-9:
        east, north = jitter[index % len(jitter)]
        history.append(
            (
                elapsed * 1000.0,
                35.0 + north / 111_320.0,
                33.0 + east / (111_320.0 * math.cos(math.radians(35.0))),
            )
        )
        elapsed += interval_seconds
        index += 1
    analysis = analyze_observed_motion(history, previously_moving=False)
    assert not analysis.moving


def _bearing_degrees(start_latitude, start_longitude, end_latitude, end_longitude):
    phi_1 = math.radians(start_latitude)
    phi_2 = math.radians(end_latitude)
    delta_lambda = math.radians(end_longitude - start_longitude)
    return math.degrees(
        math.atan2(
            math.sin(delta_lambda) * math.cos(phi_2),
            math.cos(phi_1) * math.sin(phi_2)
            - math.sin(phi_1) * math.cos(phi_2) * math.cos(delta_lambda),
        )
    ) % 360.0


def test_bucketed_bearings_normalize_packet_rate():
    samples = [
        (0.00, 10.0, -60.0),
        (0.10, 11.0, -60.0),
        (0.20, 9.0, -60.0),
        (0.60, 12.0, -60.0),
    ]
    buckets = bucket_timestamped_bearings(samples, bucket_seconds=0.5)
    assert len(buckets) == 2
    assert buckets[0].accepted_samples == 3
    assert buckets[1].accepted_samples == 1


def test_static_bearing_estimator_rejects_outliers_and_converges():
    random.seed(5)
    truth = (35.004, 33.005)
    receivers = [("A", 35.0, 33.0), ("B", 35.0, 33.01)]
    measurements = []
    for sample_index in range(60):
        for receiver_index, (name, latitude, longitude) in enumerate(receivers):
            noise = random.gauss(0.0, 2.0)
            if receiver_index == 1 and sample_index in {13, 41}:
                noise += 25.0
            measurements.append(
                BearingMeasurement(
                    name,
                    latitude,
                    longitude,
                    _bearing_degrees(latitude, longitude, *truth) + noise,
                    -60.0 + random.gauss(0.0, 1.5),
                )
            )
    estimate = estimate_static_target("Target-1", measurements)
    assert estimate is not None
    assert distance_meters(estimate.latitude, estimate.longitude, *truth) < 3.0


def test_bearing_ekf_improves_moving_target_over_raw_intersections():
    random.seed(6)
    receivers = [("A", 35.0, 33.0), ("B", 35.0, 33.01)]
    state = None
    tracked_errors = []
    raw_errors = []
    for sample_index in range(201):
        elapsed = sample_index * 0.2
        truth_latitude = 35.004
        truth_longitude = 33.005 + 10.0 * elapsed / (
            111_320.0 * math.cos(math.radians(truth_latitude))
        )
        measurements = [
            BearingMeasurement(
                name,
                latitude,
                longitude,
                _bearing_degrees(
                    latitude,
                    longitude,
                    truth_latitude,
                    truth_longitude,
                )
                + random.gauss(0.0, 2.0),
                -60.0 + random.gauss(0.0, 1.5),
            )
            for name, latitude, longitude in receivers
        ]
        raw = estimate_target("Target-1", measurements)
        assert raw is not None
        if state is None:
            state = initialize_bearing_track(
                raw,
                measurements,
                elapsed * 1000.0,
            )
            tracked = raw
        else:
            state, tracked, _accepted, _rejected = update_bearing_track(
                state,
                "Target-1",
                measurements,
                elapsed * 1000.0,
            )
        raw_errors.append(
            distance_meters(
                raw.latitude,
                raw.longitude,
                truth_latitude,
                truth_longitude,
            )
        )
        tracked_errors.append(
            distance_meters(
                tracked.latitude,
                tracked.longitude,
                truth_latitude,
                truth_longitude,
            )
        )
    assert statistics.mean(tracked_errors[-100:]) < 10.0
    assert statistics.mean(tracked_errors[-100:]) < statistics.mean(raw_errors[-100:]) * 0.4
