# -*- coding: utf-8 -*-
"""配車と frame の経路が、配信の取りこぼしで届かなくなっていないかを検証する。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
for st in (sys.stdout, sys.stderr):
    try:
        st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


import numpy as np

from app.map import build_map_index, get_preset, load_map
from app.runtime.engine import SimulationEngine
from app.sim.world import World

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if ok else 'NG  '}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def client_route(wire: dict, state: dict) -> list:
    """`store/simStore.ts` の taxi ケースと同じ規則で経路を持ち直す。"""
    if "route" in wire:
        state["route"] = wire["route"]
    elif wire["routeRevision"] != state["routeRevision"]:
        state["route"] = []
    state["routeRevision"] = wire["routeRevision"]
    return state["route"]


print("=" * 72)
print("配車（taxi）— 最新の 1 件しか配らない配信に、差分の経路を載せている")
print("=" * 72)

engine = SimulationEngine()
status = engine._taxi.status
status.phase = "approaching"
status.vehicle_id = 2
status.route = [(0.0, 0.0), (10.0, 0.0), (20.0, 5.0)]
status.route_revision = 1

state = {"routeRevision": 0, "route": [(1.0, 1.0)]}

# エンジンスレッドの同じ反復で 2 回作られる並び（`_drain_inbox` → `_step_once`）
engine._publish_taxi()
engine._publish_taxi()
seq, wire = engine.take_taxi(-1)
route = client_route(wire, state)
check(
    "取られる前に上書きされても、次の通に経路が載る",
    "route" in wire and len(route) == 3,
    f"{len(route)} 点",
)

engine._publish_taxi()
seq, wire = engine.take_taxi(seq)
route = client_route(wire, state)
check(
    "配り終えた経路は載せ直さない（数百点を毎通送らない）",
    "route" not in wire and len(route) == 3,
    f"{len(route)} 点",
)

status.route = [(0.0, 0.0), (5.0, 5.0)]
status.route_revision = 2
engine._publish_taxi()
seq, wire = engine.take_taxi(seq)
route = client_route(wire, state)
check("版が上がれば載せ直す", "route" in wire and len(route) == 2, f"{len(route)} 点")

print()
print("=" * 72)
print("frame — 経路は変化があったスロットだけ載せている")
print("=" * 72)

preset = get_preset("ginza")
index = build_map_index(load_map(preset))
world = World(index, np.random.default_rng(0))
world.set_active_count(2)

first = world.snapshot(0, 0.0, include_routes=False)
check(
    "起こしたスロットの経路が載る",
    len(first.routed_slots) >= 2,
    f"{len(first.routed_slots)} スロット",
)

again = world.snapshot(1, 0.05, include_routes=False)
check(
    "配信できていない間は載り続ける（`route_dirty` を作った時点で落とさない）",
    set(again.routed_slots) == set(first.routed_slots),
    f"{len(again.routed_slots)} スロット",
)

world.clear_route_dirty(again.routed_slots)
after = world.snapshot(2, 0.10, include_routes=False)
check(
    "配信できたら落とす（毎フレーム数千点を送らない）",
    len(after.routed_slots) == 0,
    f"{len(after.routed_slots)} スロット",
)

full = world.snapshot(3, 0.15, include_routes=True)
check(
    "新規接続向けの全経路は変化に関係なく載る",
    len(full.routed_slots) == len(full.vehicles),
    f"{len(full.routed_slots)} スロット",
)

print()
print("=" * 72)
if FAILURES:
    print(f"結果: {len(FAILURES)} 件の不合格")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("結果: すべて合格")
