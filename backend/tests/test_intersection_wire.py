"""交差点判断の配信が操作用の内部値を漏らさず、判断と介入を区別することを確かめる。"""

import json
import math

import pytest

from app.contracts import IntersectionDecision, VehicleSnapshot


@pytest.mark.parametrize("source", ["camera", "background"])
@pytest.mark.parametrize("imposed", [False, True])
def test_intersection_keeps_judgment_and_intervention_separate(source: str, imposed: bool) -> None:
    decision = IntersectionDecision(
        node_id=12,
        state="waiting",
        reason="oncoming",
        source=source,
        waiting_seconds=3.23456,
        gap_seconds=1.23456,
        speed_cap=0.0,
        imposed=imposed,
    )
    assert decision.to_wire() == {
        "nodeId": 12,
        "state": "waiting",
        "reason": "oncoming",
        "source": source,
        "waitingSeconds": 3.235,
        "gapSeconds": 1.235,
        "imposed": imposed,
    }


@pytest.mark.parametrize("gap", [None, math.inf, -math.inf, math.nan])
def test_unknown_gap_is_json_null(gap: float | None) -> None:
    wire = IntersectionDecision(node_id=12, gap_seconds=gap).to_wire()
    assert wire["gapSeconds"] is None
    json.dumps(wire, allow_nan=False)


def test_vehicle_omits_absent_intersection_and_clears_previous_judgment() -> None:
    vehicle = VehicleSnapshot(
        id=0, active=True, x=0.0, y=0.0, heading=0.0, speed=0.0, steer=0.0,
        collided=False, reached_goal=False, goal=(10.0, 0.0),
    )
    assert "intersection" not in vehicle.to_wire()
    vehicle.intersection = IntersectionDecision(node_id=12, reason="visibility")
    assert vehicle.to_wire()["intersection"]["reason"] == "visibility"
    vehicle.intersection = None
    assert "intersection" not in vehicle.to_wire()
