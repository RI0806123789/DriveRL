"""contracts.py の検証・変換（マップ・シミュレータを使わない純粋な部分）。"""
from __future__ import annotations

import math

import pytest

from app import config
from app.contracts import (
    TAXI_PHASE_RIDING,
    MapData,
    ParamPatchResult,
    SimParams,
    TaxiStatus,
    VehicleSnapshot,
    coerce_bool,
    validate_hidden_sizes,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        (2.5, True),
        (0.0, False),
        (float("nan"), None),
        (float("inf"), None),
        ("true", True),
        (" Yes ", True),
        ("ON", True),
        ("1", True),
        ("off", False),
        ("0", False),
        ("no", False),
        ("maybe", None),
        ("", None),
        (None, None),
        ([], None),
    ],
)
def test_coerce_bool(value: object, expected: bool | None) -> None:
    assert coerce_bool(value) is expected


class TestSimParamsApplyWire:
    def test_defaults_round_trip_without_change(self) -> None:
        params = SimParams()
        result = params.apply_wire(
            params.to_wire(),
            max_vehicles=config.MAX_VEHICLES,
            max_pedestrians=config.MAX_PEDESTRIANS,
        )
        assert result == ParamPatchResult()
        assert not result.has_problem

    def test_unknown_key_is_ignored(self) -> None:
        params = SimParams()
        result = params.apply_wire({"noSuchParam": 1, "type": "set_params"})
        assert result == ParamPatchResult()
        assert params == SimParams()

    @pytest.mark.parametrize(
        ("wire", "value", "expected"),
        [
            ("simSpeed", 100.0, 8.0),
            ("simSpeed", 0.0, 0.25),
            ("gamma", 1.5, 0.9999),
            ("rewardCollision", 5.0, 0.0),
            ("weatherFog", -1.0, 0.0),
        ],
    )
    def test_out_of_range_is_clamped(self, wire: str, value: float, expected: float) -> None:
        params = SimParams()
        result = params.apply_wire({wire: value})
        assert params.to_wire()[wire] == expected
        assert result.clamped == [wire]
        assert result.has_problem

    def test_vehicle_and_pedestrian_counts_follow_server_limits(self) -> None:
        params = SimParams()
        result = params.apply_wire(
            {"vehicleCount": 99, "pedestrianCount": 500}, max_vehicles=8, max_pedestrians=64
        )
        assert (params.vehicle_count, params.pedestrian_count) == (8, 64)
        assert sorted(result.clamped) == ["pedestrianCount", "vehicleCount"]

    def test_vehicle_count_upper_bound_is_at_least_one(self) -> None:
        params = SimParams()
        params.apply_wire({"vehicleCount": 5}, max_vehicles=0)
        assert params.vehicle_count == 1

    def test_integer_fields_are_rounded(self) -> None:
        params = SimParams()
        result = params.apply_wire({"rolloutLength": 100.6})
        assert params.rollout_length == 101
        assert isinstance(params.rollout_length, int)
        assert result.clamped == []

    def test_numeric_string_is_accepted(self) -> None:
        params = SimParams()
        params.apply_wire({"gamma": "0.95"})
        assert params.gamma == pytest.approx(0.95)

    def test_bool_into_number_field_counts_as_integer(self) -> None:
        params = SimParams()
        params.apply_wire({"vehicleCount": True})
        assert params.vehicle_count == 1

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), "abc", None, [1], {"v": 1}])
    def test_non_numbers_are_rejected_without_change(self, value: object) -> None:
        params = SimParams()
        result = params.apply_wire({"learningRate": value})
        assert result.rejected == ["learningRate"]
        assert result.changed == []
        assert params.learning_rate == SimParams().learning_rate

    def test_bool_field_accepts_words(self) -> None:
        params = SimParams()
        result = params.apply_wire({"obeySignals": "off", "weatherAuto": 1})
        assert params.obey_signals is False
        assert params.weather_auto is True
        assert sorted(result.changed) == ["obey_signals", "weather_auto"]

    def test_bool_field_rejects_unknown_words(self) -> None:
        params = SimParams()
        result = params.apply_wire({"safetyAssist": "maybe"})
        assert result.rejected == ["safetyAssist"]
        assert params.safety_assist is SimParams().safety_assist

    def test_same_value_is_not_reported_as_changed(self) -> None:
        params = SimParams()
        result = params.apply_wire({"gamma": params.gamma, "obeySignals": params.obey_signals})
        assert result.changed == []


class TestValidateHiddenSizes:
    def test_accepts_boundaries(self) -> None:
        low = [config.PPO_HIDDEN_MIN_WIDTH] * config.PPO_HIDDEN_MIN_LAYERS
        high = [config.PPO_HIDDEN_MAX_WIDTH] * config.PPO_HIDDEN_MAX_LAYERS
        assert validate_hidden_sizes(low) == (low, "")
        assert validate_hidden_sizes(tuple(high)) == (high, "")

    def test_float_width_becomes_int(self) -> None:
        sizes, why = validate_hidden_sizes([64.0, 32])
        assert sizes == [64, 32]
        assert all(isinstance(w, int) for w in sizes)
        assert why == ""

    @pytest.mark.parametrize(
        "value",
        [
            "128",
            128,
            None,
            [],
            [64] * (config.PPO_HIDDEN_MAX_LAYERS + 1),
            [True],
            ["128"],
            [float("nan")],
            [float("inf")],
            [config.PPO_HIDDEN_MIN_WIDTH - 1],
            [config.PPO_HIDDEN_MAX_WIDTH + 1],
        ],
    )
    def test_rejects_with_reason(self, value: object) -> None:
        sizes, why = validate_hidden_sizes(value)
        assert sizes is None
        assert why


class TestWireRounding:
    def test_vehicle_route_only_when_present(self) -> None:
        base = dict(
            id=0,
            active=True,
            x=1.23456,
            y=-7.89012,
            heading=math.pi,
            speed=3.33333,
            steer=0.123456,
            collided=False,
            reached_goal=False,
            goal=(10.0, 20.0),
        )
        assert "route" not in VehicleSnapshot(**base).to_wire()
        wire = VehicleSnapshot(**base, route=[(0.123, 4.567)]).to_wire()
        assert wire["route"] == [[0.12, 4.57]]
        assert (wire["x"], wire["y"]) == (1.235, -7.89)
        assert wire["heading"] == 3.1416

    def test_taxi_route_only_when_asked(self) -> None:
        status = TaxiStatus(
            phase=TAXI_PHASE_RIDING,
            vehicle_id=2,
            pickup=(1.0, 2.0),
            dropoff=None,
            route=[(0.0, 0.0), (5.554, 6.666)],
        )
        assert "route" not in status.to_wire()
        wire = status.to_wire(include_route=True)
        assert wire["route"] == [[0.0, 0.0], [5.55, 6.67]]
        assert wire["pickup"] == [1.0, 2.0]
        assert wire["dropoff"] is None

    def test_map_keeps_ids_and_rounds_coordinates(self, tiny_map: MapData) -> None:
        wire = tiny_map.to_wire()
        assert [n["id"] for n in wire["nodes"]] == [0, 1]
        edge = wire["edges"][0]
        assert (edge["u"], edge["v"]) == (0, 1)
        assert edge["polyline"][0] == [0.0, 0.0]
        assert len(wire["signals"]) == 1 and len(wire["signs"]) == 1
        assert "length" not in edge
