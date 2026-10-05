"""contracts.py・protocol.ts・docs/protocol.md・main.py の 4 つが同じ契約を書いているかを検査する。"""
from __future__ import annotations

import ast
import re
from typing import Any

import numpy as np
import pytest

from app import config
from app.contracts import (
    AutotuneBest,
    AutotuneSnapshot,
    AutotuneTrial,
    FrameSnapshot,
    MapData,
    MetricsSnapshot,
    ObstacleSnapshot,
    PedestrianSnapshot,
    SimParams,
    TaxiStatus,
    VehicleSnapshot,
)
from app.map.presets import list_presets
from app.percep.occlusion import RAY_ANGLES, RAY_HALF_WIDTH, CameraInput, evaluate_occlusion
from app.percep.types import CAMERA_RIG, DetClass, Detection, PerceptionResult

from .conftest import BACKEND_DIR, TsInterface

MAIN_PY = BACKEND_DIR / "app" / "main.py"


def assert_wire_matches(
    wire: dict[str, Any], iface: TsInterface, *, added_by_sender: frozenset[str] = frozenset()
) -> None:
    emitted = set(wire) | added_by_sender
    unknown = emitted - iface.fields
    missing = iface.required - emitted
    assert not unknown, f"{iface.name} に宣言の無いキーを送っている: {sorted(unknown)}"
    assert not missing, f"{iface.name} の必須の欄を送っていない: {sorted(missing)}"


TYPE_ONLY = frozenset({"type"})


def _vehicle() -> VehicleSnapshot:
    return VehicleSnapshot(
        id=0,
        active=True,
        x=0.0,
        y=0.0,
        heading=0.0,
        speed=0.0,
        steer=0.0,
        collided=False,
        reached_goal=False,
        goal=(1.0, 1.0),
        # 省略できる欄も載せる（載せないと、その欄が protocol.ts に宣言されているかを確かめられない）
        v2x_links=[2],
        current_option="STOP",
        route=[(0.0, 0.0), (1.0, 1.0)],
    )


def _occlusion_wire() -> dict[str, Any]:
    """車両の陰（動的）と建物の陰（静的）と角の両方が出る、見通しと死角の見本。省略できる欄も載る。"""
    free = np.full(config.OBS_FREESPACE_DIM, config.OBS_FREESPACE_MAX_DISTANCE, dtype=np.float32)
    free[3] = 8.0
    car = Detection(DetClass.VEHICLE, 0.45, 0.4, 0.55, 0.8, 0.9, distance=12.0)
    views = [
        CameraInput(spec, PerceptionResult(0, [car] if spec.key == "front" else []), free)
        for spec in CAMERA_RIG
    ]
    result = evaluate_occlusion(views)
    assert result.corner_distance is not None
    return result.to_wire(RAY_ANGLES, RAY_HALF_WIDTH)


class TestWireKeysMatchProtocolTs:
    def test_sim_params(self, ts_interfaces: dict[str, TsInterface]) -> None:
        wire = SimParams().to_wire()
        iface = ts_interfaces["SimParams"]
        assert set(wire) == iface.fields == iface.required

    def test_map_presets(self, ts_interfaces: dict[str, TsInterface]) -> None:
        for preset in list_presets():
            assert_wire_matches(preset.to_wire(), ts_interfaces["MapPreset"])

    def test_map_message_and_items(
        self, tiny_map: MapData, ts_interfaces: dict[str, TsInterface]
    ) -> None:
        wire = tiny_map.to_wire()
        assert_wire_matches(wire, ts_interfaces["MapMessage"], added_by_sender=TYPE_ONLY)
        assert_wire_matches(wire["bounds"], ts_interfaces["MapBounds"])
        for key, name in [
            ("nodes", "MapNode"),
            ("edges", "MapEdge"),
            ("buildings", "MapBuilding"),
            ("signals", "MapSignal"),
            ("signs", "MapSign"),
        ]:
            assert wire[key], f"最小のマップに {key} が無い（検査にならない）"
            for item in wire[key]:
                assert_wire_matches(item, ts_interfaces[name])

    def test_frame_and_items(self, ts_interfaces: dict[str, TsInterface]) -> None:
        occlusion = _occlusion_wire()
        frame = FrameSnapshot(
            tick=1,
            sim_time=0.05,
            vehicles=[_vehicle()],
            obstacles=[ObstacleSnapshot(id=0, x=1.0, y=2.0, radius=0.5)],
            pedestrians=[PedestrianSnapshot(id=0, x=1.0, y=2.0, heading=0.0, stride=0.3)],
            signals=[0, 2],
            detections={0: []},
            surround={0: {"rear": []}},
            occlusion={0: occlusion},
            weather={"rain": 0.0, "fog": 0.2, "visibility": 120.0},
        )
        wire = frame.to_wire()
        assert_wire_matches(wire, ts_interfaces["FrameMessage"], added_by_sender=TYPE_ONLY)
        assert_wire_matches(wire["vehicles"][0], ts_interfaces["VehicleState"])
        assert_wire_matches(wire["obstacles"][0], ts_interfaces["ObstacleState"])
        assert_wire_matches(wire["pedestrians"][0], ts_interfaces["NpcPedestrianState"])
        view = wire["occlusion"]["0"]
        assert_wire_matches(view, ts_interfaces["OcclusionView"])
        assert view["cameras"] and view["shadows"], "見本に、カメラと死角（動的・静的）の両方が載っていない"
        for camera in view["cameras"]:
            assert_wire_matches(camera, ts_interfaces["OcclusionCamera"])
        for shadow in view["shadows"]:
            assert_wire_matches(shadow, ts_interfaces["OcclusionShadow"])
        assert {s["kind"] for s in view["shadows"]} == {"dynamic", "static"}

    def test_frame_omits_empty_optionals(self, ts_interfaces: dict[str, TsInterface]) -> None:
        wire = FrameSnapshot(tick=0, sim_time=0.0, vehicles=[], obstacles=[]).to_wire()
        assert_wire_matches(wire, ts_interfaces["FrameMessage"], added_by_sender=TYPE_ONLY)
        optional = ts_interfaces["FrameMessage"].fields - ts_interfaces["FrameMessage"].required
        assert not (set(wire) & optional)

    def test_metrics(self, ts_interfaces: dict[str, TsInterface]) -> None:
        assert_wire_matches(
            MetricsSnapshot().to_wire(), ts_interfaces["MetricsMessage"], added_by_sender=TYPE_ONLY
        )

    def test_autotune(self, ts_interfaces: dict[str, TsInterface]) -> None:
        wire = AutotuneSnapshot(
            available=True,
            running=True,
            phase="running",
            study_name="live-ginza",
            trial=3,
            trial_progress=0.5,
            trial_steps=1536,
            finished_trials=2,
            prior_trials=1,
            tuned_keys=["learningRate"],
            current={"learningRate": 1e-4},
            best=AutotuneBest(trial=1, score=0.3, params={"learningRate": 1e-4}),
            history=[AutotuneTrial(trial=1, score=0.3, outcome="complete")],
            message="試行 #3 を走らせています",
        ).to_wire()
        assert_wire_matches(wire, ts_interfaces["AutotuneMessage"], added_by_sender=TYPE_ONLY)
        assert_wire_matches(wire["best"], ts_interfaces["AutotuneBest"])
        assert_wire_matches(wire["history"][0], ts_interfaces["AutotuneTrial"])
        idle = AutotuneSnapshot().to_wire()
        assert_wire_matches(idle, ts_interfaces["AutotuneMessage"], added_by_sender=TYPE_ONLY)

    @pytest.mark.parametrize("include_route", [False, True])
    def test_taxi(self, include_route: bool, ts_interfaces: dict[str, TsInterface]) -> None:
        wire = TaxiStatus(route=[(0.0, 0.0)]).to_wire(include_route=include_route)
        assert ("route" in wire) is include_route
        assert_wire_matches(wire, ts_interfaces["TaxiMessage"], added_by_sender=TYPE_ONLY)


def test_protocol_version_agrees(protocol_ts_source: str, protocol_md_source: str) -> None:
    found: dict[str, list[int]] = {
        "config.PROTOCOL_VERSION": [config.PROTOCOL_VERSION],
        "protocol.ts の PROTOCOL_VERSION": [
            int(v) for v in re.findall(r"export const PROTOCOL_VERSION = (\d+)", protocol_ts_source)
        ],
        "protocol.ts の冒頭（docs/protocol.md vN）": [
            int(v) for v in re.findall(r"docs/protocol\.md v(\d+)", protocol_ts_source)
        ],
        "protocol.md の題": [
            int(v) for v in re.findall(r"^# WebSocket プロトコル仕様 v(\d+)", protocol_md_source, re.M)
        ],
        "protocol.md の本文": [
            int(v) for v in re.findall(r"`protocolVersion` は \*\*(\d+)\*\*", protocol_md_source)
        ],
        "protocol.md の例": [
            int(v) for v in re.findall(r'"protocolVersion":\s*(\d+)', protocol_md_source)
        ],
    }
    for where, values in found.items():
        assert values, f"{where} から版番号を読めない（書き方が変わった？）"
    mismatched = {k: v for k, v in found.items() if set(v) != {config.PROTOCOL_VERSION}}
    assert not mismatched, f"版番号が {config.PROTOCOL_VERSION} でない所がある: {mismatched}"


def _message_types(
    union: str, ts_unions: dict[str, list[str]], ts_interfaces: dict[str, TsInterface]
) -> set[str]:
    names = ts_unions[union]
    literals = {name: ts_interfaces[name].type_literal for name in names}
    missing = [name for name, lit in literals.items() if lit is None]
    assert not missing, f"{union} の {missing} に type の値が無い"
    return {lit for lit in literals.values() if lit is not None}


def _section(markdown: str, start: str, end: str) -> str:
    head = markdown.index(start)
    return markdown[head : markdown.index(end, head)]


def _accepted_by_main() -> set[str]:
    """main.py の handle_client_message が `kind == "x"` / `kind in 集合` で受け付ける種別。"""
    tree = ast.parse(MAIN_PY.read_text(encoding="utf-8"))
    named: dict[str, set[str]] = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        items = value.keys if isinstance(value, ast.Dict) else getattr(value, "elts", [])
        strings = {i.value for i in items if isinstance(i, ast.Constant) and isinstance(i.value, str)}
        if strings:
            named[target.id] = strings

    func = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "handle_client_message"
    )
    kinds: set[str] = set()
    for node in ast.walk(func):
        if not (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "kind"):
            continue
        for op, comp in zip(node.ops, node.comparators):
            if isinstance(op, ast.Eq) and isinstance(comp, ast.Constant):
                kinds.add(comp.value)
            elif isinstance(op, ast.In) and isinstance(comp, ast.Name):
                kinds |= named[comp.id]
    return kinds


class TestMessageTypes:
    def test_client_types_are_documented(
        self,
        ts_unions: dict[str, list[str]],
        ts_interfaces: dict[str, TsInterface],
        protocol_md_source: str,
    ) -> None:
        client = _message_types("ClientMessage", ts_unions, ts_interfaces)
        server = _message_types("ServerMessage", ts_unions, ts_interfaces)
        section = _section(protocol_md_source, "## 3. クライアント → サーバー", "## 4. ")
        documented = set(re.findall(r'"type":\s*"(\w+)"', section))
        assert client - documented == set(), "protocol.md の 3 章に載っていない送信"
        # 3 章には応答（pong など）も例として出てくる
        assert documented - client - server == set(), "protocol.ts に無い種別が 3 章にある"

    def test_client_types_are_accepted_by_main(
        self, ts_unions: dict[str, list[str]], ts_interfaces: dict[str, TsInterface]
    ) -> None:
        client = _message_types("ClientMessage", ts_unions, ts_interfaces)
        accepted = _accepted_by_main()
        assert client - accepted == set(), "main.py に受け口が無い（未知の種別として断られる）"
        assert accepted - client == set(), "main.py は受け付けるが protocol.ts の ClientMessage に無い"

    def test_server_types_are_documented_and_sent(
        self,
        ts_unions: dict[str, list[str]],
        ts_interfaces: dict[str, TsInterface],
        protocol_md_source: str,
    ) -> None:
        server = _message_types("ServerMessage", ts_unions, ts_interfaces)
        main_source = MAIN_PY.read_text(encoding="utf-8")
        undocumented = {
            t
            for t in server
            if f"`{t}`" not in protocol_md_source
            and not re.search(rf'"type":\s*"{t}"', protocol_md_source)
        }
        unsent = {t for t in server if not re.search(rf'"type":\s*"{t}"', main_source)}
        assert undocumented == set(), "protocol.md に載っていない受信"
        assert unsent == set(), "main.py が一度も送らない受信"

    def test_sim_params_keys_are_documented(self, protocol_md_source: str) -> None:
        section = _section(protocol_md_source, "### 2.5 `params`", "### 2.6 ")
        missing = [k for k in SimParams().to_wire() if k not in section]
        assert missing == [], "protocol.md 2.5 に載っていない params のキー"
