"""車線変更の配信が判断と実介入を分け、JSONと任意の欄の契約を保つことを確かめる。"""

import json
import math

import pytest

from app.contracts import LaneChangeDecision, VehicleSnapshot


@pytest.mark.parametrize("source", ["camera", "background", "road_event"])
@pytest.mark.parametrize("imposed", [False, True])
def test_lane_change_keeps_judgment_and_intervention_separate(source: str, imposed: bool) -> None:
    decision = LaneChangeDecision(
        state="returning", reason="obstacle", source=source, direction=1, imposed=imposed,
        min_gap=8.12345, rear_decel=1.23456, lateral_accel=-0.34567, lateral_jerk=0.45678, completed=2,
    )
    assert decision.to_wire() == {
        "state": "returning", "reason": "obstacle", "source": source, "direction": 1, "imposed": imposed,
        "minGap": 8.123, "rearDecel": 1.235, "lateralAccel": -0.346, "lateralJerk": 0.457, "completed": 2,
    }


@pytest.mark.parametrize("gap", [math.inf, -math.inf, math.nan])
def test_unknown_lane_gap_is_json_null(gap: float) -> None:
    wire = LaneChangeDecision(min_gap=gap).to_wire()
    assert wire["minGap"] is None
    json.dumps(wire, allow_nan=False)


def test_vehicle_omits_absent_lane_change_and_clears_previous_judgment() -> None:
    vehicle = VehicleSnapshot(
        id=0, active=True, x=0.0, y=0.0, heading=0.0, speed=0.0, steer=0.0,
        collided=False, reached_goal=False, goal=(10.0, 0.0),
    )
    assert "laneChange" not in vehicle.to_wire()
    vehicle.lane_change = LaneChangeDecision(reason="visibility")
    assert vehicle.to_wire()["laneChange"]["reason"] == "visibility"
    vehicle.lane_change = None
    assert "laneChange" not in vehicle.to_wire()
