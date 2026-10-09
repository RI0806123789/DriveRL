"""遅れて届いた AI 応答が別の配車へ操作を混ぜないことを検査する。"""

from __future__ import annotations

import ast
import asyncio
import functools
import logging
import math
import threading
import time
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import orjson
import pytest
from fastapi.responses import JSONResponse

from app.contracts import SimParams
from app.runtime import concierge
from app.runtime.taxi import TaxiService
from app.sim.env import SimulationEnv
from tests.test_concierge import _grid_index


def load_function(path: Path, name: str, namespace: dict):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.fixture
def ride_engine():
    params = SimParams(vehicle_count=3, pedestrian_count=0)
    env = SimulationEnv(_grid_index(), params, seed=0)
    env.reset_all()
    env.autopilot_all = True
    taxi = TaxiService()
    slot = int(np.flatnonzero(env.world.fleet.active)[0])
    pickup = (float(env.world.fleet.x[slot]), float(env.world.fleet.y[slot]))
    dropoff = (360.0 if pickup[0] < 180 else 0.0, 360.0 if pickup[1] < 180 else 0.0)
    assert taxi.request(env, pickup, dropoff) is None
    commands, notices = [], []
    engine = SimpleNamespace(
        _env=env, _taxi=taxi, _lock=threading.Lock(), _practical_mode=True,
        _notify=notices.append, _publish_taxi=lambda: None, _apply_player_pose=lambda at: None,
        taxi_command=lambda action, payload=None: commands.append((action, payload or {})),
        practical_mode=lambda: engine._practical_mode,
    )
    path = Path(__file__).resolve().parents[1] / "app" / "runtime" / "engine.py"
    handler = load_function(path, "_handle_taxi", {"Any": Any, "logger": logging.getLogger(__name__)})
    engine._handle_taxi = MethodType(handler, engine)

    def situation():
        done = threading.Event()
        done.set()
        return SimpleNamespace(done=done, situation={**taxi.describe(env), "practicalMode": engine._practical_mode})

    engine.request_taxi_situation = situation
    return engine, commands, notices, pickup, dropoff


def endpoint(engine):
    path = Path(__file__).resolve().parents[1] / "app" / "main.py"
    return load_function(path, "concierge_endpoint", {
        "asyncio": asyncio, "functools": functools, "orjson": orjson, "time": time,
        "Request": Any, "JSONResponse": JSONResponse, "engine": engine,
        "logger": logging.getLogger(__name__), "_concierge_lock": asyncio.Lock(),
        "_concierge_last_call": -math.inf, "CONCIERGE_MIN_INTERVAL_SEC": 0.0,
        "CONCIERGE_MAX_BODY_BYTES": 8192, "CONCIERGE_SITUATION_TIMEOUT_SEC": 1.0,
        "CONCIERGE_BUSY_MESSAGE": "混雑しています", "CONCIERGE_FAILED_MESSAGE": "応答できませんでした",
    })


class RequestBody:
    headers = {"content-type": "application/json"}

    def __init__(self, **body):
        self.body = body

    async def stream(self):
        yield orjson.dumps(self.body)


@pytest.mark.parametrize("tool", ["set_driving_mode", "request_emergency_stop"])
@pytest.mark.parametrize("change", ["new", "end", "mode", "same", "handover"])
def test_delayed_reply_applies_only_to_its_ride(ride_engine, monkeypatch, tool, change):
    engine, commands, notices, pickup, dropoff = ride_engine
    taxi, env = engine._taxi, engine._env
    ride_id = taxi.status.ride_id
    entered, release = threading.Event(), threading.Event()
    call = concierge.ToolCall(tool, {"mode": "comfort"} if tool == "set_driving_mode" else {})

    def ask(*args, **kwargs):
        entered.set()
        assert release.wait(5.0)
        return concierge.ConciergeReply("承知しました。", [call])

    monkeypatch.setattr(concierge, "available", lambda: True)
    monkeypatch.setattr(concierge, "ask", ask)

    async def run():
        task = asyncio.create_task(endpoint(engine)(RequestBody(message="お願いします", rideId=ride_id)))
        try:
            assert await asyncio.to_thread(entered.wait, 5.0)
            if change in ("new", "end", "mode"):
                taxi.cancel(env, "テスト")
            if change == "new":
                assert taxi.request(env, pickup, dropoff) is None
                assert taxi.status.ride_id != ride_id
            elif change == "mode":
                engine._practical_mode = False
            elif change == "handover":
                old_slot = taxi.vehicle_id
                assert taxi._handover(env, halt=False)
                assert taxi.vehicle_id != old_slot and taxi.status.ride_id == ride_id
        finally:
            release.set()
        response = await task
        assert response.status_code == 200

    asyncio.run(run())
    assert len(commands) == 1 and commands[0][1]["rideId"] == ride_id
    status_before = taxi.status.to_wire(include_route=True)
    speed_before = env.world.fleet.speed.copy()
    for action, payload in commands:
        engine._handle_taxi(action, payload)
    if change in ("new", "end", "mode"):
        assert taxi.status.to_wire(include_route=True) == status_before
        np.testing.assert_array_equal(env.world.fleet.speed, speed_before)
        assert notices
    elif tool == "set_driving_mode":
        assert taxi.status.drive_mode == "comfort"
    else:
        assert not taxi.busy and taxi.status.ride_id is None


def test_wrong_client_ride_is_rejected_before_ai(ride_engine, monkeypatch):
    engine, commands, *_ = ride_engine
    monkeypatch.setattr(concierge, "available", lambda: True)
    monkeypatch.setattr(concierge, "ask", lambda *args, **kwargs: pytest.fail("古い依頼が AI に届きました"))
    response = asyncio.run(endpoint(engine)(RequestBody(message="お願いします", rideId="old-ride")))
    assert response.status_code == 409 and commands == []


def test_regular_quick_command_remains_bound_to_ride(ride_engine, monkeypatch):
    engine, commands, *_ = ride_engine
    monkeypatch.setattr(concierge, "available", lambda: True)
    response = asyncio.run(endpoint(engine)(RequestBody(action="hurry")))
    assert response.status_code == 200
    assert commands == [("drive_mode", {"mode": "hurry", "rideId": engine._taxi.status.ride_id})]
    engine._handle_taxi(*commands[0])
    assert engine._taxi.status.drive_mode == "hurry"


def test_alighting_and_rejected_request_do_not_reuse_ride_id(ride_engine):
    engine, _, _, pickup, dropoff = ride_engine
    taxi, env = engine._taxi, engine._env
    ride_id = taxi.status.ride_id
    assert taxi.request(env, pickup, dropoff) is not None
    assert taxi.status.ride_id == ride_id
    assert taxi.board(env) is None
    assert taxi.status.ride_id == ride_id
    assert taxi.alight(env) is None
    assert taxi.status.ride_id is None
    assert taxi.request(env, pickup, dropoff) is None
    assert taxi.status.ride_id != ride_id
