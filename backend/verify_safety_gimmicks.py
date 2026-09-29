# -*- coding: utf-8 -*-
"""周囲カメラ（後方・左右）と、検出枠に連動する安全ギミック（切り返し・後退 AEB・巻き込み防止）を検証する。"""
from __future__ import annotations

import argparse
import gc
import math
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
for st in (sys.stdout, sys.stderr):
    try:
        st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch

from app import config
from app.contracts import InterventionEvent, SimParams
from app.map import build_map_index, get_preset, load_map
from app.map.loader import MapLoadError
from app.percep.camera import LBL_OBSTACLE, PALETTE, PseudoCamera
from app.percep.encoder import OBS_OFFSETS, local_xy
from app.percep.groundtruth import detect_ground_truth_views
from app.percep.types import CAMERA_RIG, SURROUND_CAMERAS, DetClass
from app.rl.buffer import RolloutBuffer
from app.rl.ppo import PPOTrainer
from app.rl.warmstart import collect_expert
from app.sim.env import SimulationEnv
from app.sim.signals import GREEN

FAILURES: list[str] = []
ACT = np.zeros((config.MAX_VEHICLES, config.ACTION_DIM), dtype=np.float32)
#: 周囲カメラの写り方を見るときに置くパイロンの距離 [m]（カメラから）
VIEW_TEST_M = 8.0
#: 推定した位置と置いた位置の許容差 [m]（単眼の距離推定と箱の中心の方位から出すため）
LOCATE_TOL_M = 1.5
#: 切り返しの試験で、パイロンを置く位置（出発点から経路に沿って）と、抜けたとみなす位置 [m]
PYLON_AHEAD_M = 30.0
PASSED_AFTER_M = 12.0
#: 後退 AEB の試験で、後退を始めた直後に後ろのバンパーからこれだけ離して置く [m]
AEB_BEHIND_M = 2.6
#: 後退 AEB で止まったときに、バンパーと物体の面の間に残っていてほしい距離 [m]
AEB_MIN_GAP_M = 0.5
BUDGET_MS = 50.0
#: 巻き込み防止の試験で、歩行者が離れてから発進を待つ最長 [s]（赤信号で並んでいたときは青まで待つ）
RESUME_WAIT_SEC = 60.0


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if ok else 'NG  '}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def make_env(index, *, vehicles: int = 1, walkers: int = 0, seed: int = 0, assist: bool = False) -> SimulationEnv:
    """真値で走る env（認識器は使わない。枠と位置が決まっているほうが判定できる）。"""
    params = SimParams()
    params.vehicle_count = vehicles
    params.pedestrian_count = walkers
    params.safety_assist = assist
    env = SimulationEnv(index, params, seed=seed)
    env._ensure_percep()
    env._detector = None
    env._camera = None
    return env


def straight_start(env: SimulationEnv, slot: int, need: float) -> float | None:
    """経路の先 `need` m がまっすぐ続く弧長。"""
    state = env.world.slots[slot]
    cum, route = state.route_cum, state.route
    for arc in np.arange(10.0, float(cum[-1]) - need, 4.0):
        marks = np.arange(arc, arc + need, 2.0)
        xs = np.interp(marks, cum, route[:, 0])
        ys = np.interp(marks, cum, route[:, 1])
        h = np.arctan2(np.diff(ys), np.diff(xs))
        if np.max(np.abs(np.arctan2(np.sin(h - h[0]), np.cos(h - h[0])))) < math.radians(8):
            return float(arc)
    return None


def point_at(env: SimulationEnv, slot: int, arc: float) -> tuple[float, float, float]:
    state = env.world.slots[slot]
    cum, route = state.route_cum, state.route
    x = float(np.interp(arc, cum, route[:, 0]))
    y = float(np.interp(arc, cum, route[:, 1]))
    x2 = float(np.interp(arc + 1.0, cum, route[:, 0]))
    y2 = float(np.interp(arc + 1.0, cum, route[:, 1]))
    return x, y, math.atan2(y2 - y, x2 - x)


def place_on_route(env: SimulationEnv, slot: int, arc: float) -> None:
    """車を経路の弧長 `arc` の所へ、経路の向きで置き直す。"""
    x, y, h = point_at(env, slot, arc)
    fleet = env.world.fleet
    fleet.x[slot], fleet.y[slot], fleet.heading[slot] = x, y, h
    env.world.slots[slot].route_progress = int(np.searchsorted(env.world.slots[slot].route_cum, arc))
    env.world.slots[slot].arc_position = arc
    env.world.project_all()


def ego_to_world(env: SimulationEnv, slot: int, fx: float, fy: float) -> tuple[float, float]:
    fleet = env.world.fleet
    h = float(fleet.heading[slot])
    return (
        float(fleet.x[slot]) + fx * math.cos(h) - fy * math.sin(h),
        float(fleet.y[slot]) + fx * math.sin(h) + fy * math.cos(h),
    )


def verify_views(index) -> None:
    print("周囲カメラの写り方（真値の枠・画・推定位置が同じものを指すか）")
    env = make_env(index, seed=3)
    world = env.world
    slot = int(np.flatnonzero(world.fleet.active)[0])
    camera = PseudoCamera(index)
    color = PALETTE[LBL_OBSTACLE]
    for cam in SURROUND_CAMERAS:
        world.clear_obstacles()
        # カメラの視線の先 8m に置く（取り付け位置から、車体の向き + カメラの向きへ）
        look = math.radians(cam.yaw_deg)
        fx = cam.forward + VIEW_TEST_M * math.cos(look)
        fy = -cam.right + VIEW_TEST_M * math.sin(look)
        ox, oy = ego_to_world(env, slot, fx, fy)
        world.add_obstacle(ox, oy, config.OBSTACLE_RADIUS)
        views = detect_ground_truth_views(world, slot, CAMERA_RIG)
        by_key = {spec.key: view for spec, view in zip(CAMERA_RIG, views)}
        found = [d for d in by_key[cam.key].detections if d.cls is DetClass.OBSTACLE]
        others = [
            key
            for key, view in by_key.items()
            if key != cam.key and any(d.cls is DetClass.OBSTACLE for d in view.detections)
        ]
        ok = len(found) == 1
        detail = f"{len(found)} 件"
        if ok:
            ex, ey, _d = local_xy(found[0], cam)
            miss = math.hypot(ex - fx, ey - fy)
            image = camera.render(world, np.array([slot]), spec=cam)[0]
            d = found[0]
            x0, x1 = int(d.x0 * cam.width), int(math.ceil(d.x1 * cam.width))
            y0, y1 = int(d.y0 * cam.height), int(math.ceil(d.y1 * cam.height))
            inside = int(np.all(image[y0:y1, x0:x1] == color, axis=-1).sum())
            outside = int(np.all(image == color, axis=-1).sum()) - inside
            ok = miss <= LOCATE_TOL_M and inside > 0 and outside == 0
            detail = f"推定位置のずれ {miss:.2f}m / 枠の中の画素 {inside} / 枠の外 {outside}"
        check(f"{cam.key}: 正面 {VIEW_TEST_M:.0f}m のパイロンを 1 件だけ検出し、画と枠と位置が合う", ok, detail)
        check(f"{cam.key}: そのパイロンは他のカメラに写らない", not others, f"写ったカメラ {others}")
    lanes = sum(1 for view in views[1:] for d in view.detections if d.cls is DetClass.LANE)
    check("周囲カメラの検出に車線は入らない", lanes == 0, f"{lanes} 件")


def verify_gear() -> None:
    print("\n後退ギアの物理")
    from app.sim.vehicle import VehicleFleet

    fleet = VehicleFleet(1)
    fleet.active[0] = True
    fleet.speed[0] = 3.0
    check("走っている間はギアを入れ替えない", not fleet.set_gear(0, -1) and int(fleet.gear[0]) == 1)
    fleet.speed[0] = 0.0
    check("止まっていれば後退へ入れられる", fleet.set_gear(0, -1) and int(fleet.gear[0]) == -1)
    x0 = float(fleet.x[0])
    for _ in range(80):
        fleet.step(np.array([1.0]), np.array([0.0]), config.DT, config.MAX_SPEED)
    check(
        "後退ギアで踏むと後ろへ進み、速さは上限で頭打ち",
        float(fleet.x[0]) < x0 and abs(float(fleet.speed[0]) + config.REVERSE_MAX_SPEED) < 1e-5,
        f"x {x0:.1f} → {float(fleet.x[0]):.1f}m / 速度 {float(fleet.speed[0]):.2f}m/s",
    )
    for _ in range(40):
        fleet.step(np.array([-1.0]), np.array([0.0]), config.DT, config.MAX_SPEED)
    check("後退中の制動は 0 で止まり、前へは走らない", float(fleet.speed[0]) == 0.0)


def verify_stuck(index, preset: str, trials: int, *, behind: bool) -> None:
    label = "後退 AEB（後退中に真後ろへパイロンが現れる）" if behind else "立ち往生からの切り返し（経路上のパイロン）"
    print(f"\n{label}")
    passed = collided = gave_up = skipped = 0
    gaps: list[float] = []
    order_ok = True
    aeb_stops = 0
    for trial in range(trials):
        env = make_env(index, seed=11 + trial)
        env.autopilot_all = True
        slot = 0
        start = straight_start(env, slot, PYLON_AHEAD_M + 40.0)
        if start is None:
            skipped += 1
            continue
        place_on_route(env, slot, start)
        px, py, _h = point_at(env, slot, start + PYLON_AHEAD_M)
        env.world.add_obstacle(px, py, config.OBSTACLE_RADIUS)
        seen: list[str] = []
        placed = False
        gap_min = math.inf
        outcome = "gave_up"
        for _ in range(int(60.0 / config.DT)):
            res = env.step(ACT)
            assist = env.safety.assist(slot)
            if assist and (not seen or seen[-1] != assist):
                seen.append(assist)
            if behind and not placed and assist == "reversing" and int(env.world.fleet.gear[slot]) < 0:
                bx, by = ego_to_world(env, slot, -config.VEHICLE_LENGTH / 2 - AEB_BEHIND_M, 0.0)
                env.world.add_obstacle(bx, by, config.OBSTACLE_RADIUS)
                placed = True
            if placed:
                ob = env.world.obstacles[-1]
                h = float(env.world.fleet.heading[slot])
                lon = (ob.x - float(env.world.fleet.x[slot])) * math.cos(h) + (ob.y - float(env.world.fleet.y[slot])) * math.sin(h)
                gap_min = min(gap_min, -lon - config.VEHICLE_LENGTH / 2 - ob.radius)
            if bool(res.dones[slot]):
                reason = next((e.reason for e in res.episodes if e.slot == slot), "")
                outcome = "collided" if reason == "collision" else "done"
                break
            if float(env.world.arc[slot]) > start + PYLON_AHEAD_M + PASSED_AFTER_M:
                outcome = "passed"
                break
        passed += int(outcome == "passed")
        collided += int(outcome == "collided")
        gave_up += int(outcome not in ("passed", "collided"))
        if behind and placed:
            gaps.append(gap_min)
            aeb_stops += int("reverse_stop" in seen)
        if not behind and outcome == "passed":
            want = ["front_hold", "reverse_check", "reversing", "detour"]
            order_ok &= [a for a in seen if a in want][:4] == want
    tried = trials - skipped
    check(f"{preset}: 障害物にも建物にも当たらない", collided == 0, f"{tried} 回のうち {collided} 回")
    if behind:
        check(
            f"{preset}: 後退 AEB で止まり、バンパーから {AEB_MIN_GAP_M}m 以上手前に残る",
            bool(gaps) and aeb_stops == len(gaps) and min(gaps) >= AEB_MIN_GAP_M,
            f"{len(gaps)} 回・最小 {min(gaps) if gaps else float('nan'):.2f}m・自動停止 {aeb_stops} 回",
        )
    else:
        check(
            f"{preset}: 止まる → 確認 → 後退 → 脇を抜ける、の順に進んで抜けられる",
            tried > 0 and passed >= max(1, int(tried * 0.8)) and order_ok,
            f"{tried} 回のうち抜けた {passed} 回・あきらめた {gave_up} 回（幅が足りない道では下がらずにあきらめる）",
        )


def verify_turns(index, preset: str) -> None:
    print("\n曲がり角（巻き込み防止が見る所）")
    env = make_env(index, seed=7)
    world = env.world
    routes = turns = off_junction = 0
    for _ in range(40):
        plan = world._random_route()
        if plan is None or not plan.legs:
            continue
        routes += 1
        cum = world._cumulative_length(plan.points)
        starts, _ends, _sides = world._junction_turns(plan.points, cum, plan.legs)
        exits = {float(np.float32(leg.end_arc)): int(leg.exit_node) for leg in plan.legs}
        turns += int(starts.size)
        off_junction += sum(1 for a in starts if not index.is_intersection(exits[float(a)]))
    check(
        f"{preset}: 曲がり角は交差点（3 方向以上に分かれるノード）だけに置く",
        turns > 0 and off_junction == 0,
        f"経路 {routes} 本に {turns} か所・交差点でない所 {off_junction} か所",
    )


def verify_blind_spot(index, preset: str, trials: int) -> None:
    print("\n巻き込み防止（曲がる側を並走する歩行者）")
    decel = abs(config.MAX_DECEL)
    for side, name in ((-1, "左折"), (1, "右折")):
        held = resumed = hits = found = entered = red_waits = 0
        for trial in range(trials):
            env = make_env(index, seed=100 + trial)
            env.autopilot_all = True
            slot = 0
            ok = False
            for _ in range(int(120.0 / config.DT)):
                env.step(ACT)
                if not env.world.fleet.active[slot]:
                    break
                dist, turn = env.world.turn_ahead(slot)
                gap = dist - config.VEHICLE_LENGTH / 2
                speed = abs(float(env.world.fleet.speed[slot]))
                # 並んだ瞬間に止まれない配置は試さない（止まるのに要る距離 + 1m より手前で並ぶ）
                if (
                    turn == side
                    and int(env.world.turn_signal[slot]) == side
                    and speed * speed / (2.0 * decel) + 1.0 <= gap <= 8.0
                ):
                    ok = True
                    break
            if not ok:
                continue
            found += 1
            turn_arc = float(env.world.arc[slot]) + env.world.turn_ahead(slot)[0]
            lat = -side * (config.VEHICLE_WIDTH / 2 + 1.2)
            stopped = 0.0
            hit = False
            far = -math.inf
            for _ in range(int(6.0 / config.DT)):
                env.set_player_pose(ego_to_world(env, slot, 0.3, lat))
                env.step(ACT)
                if abs(float(env.world.fleet.speed[slot])) < 0.3 and env.safety.assist(slot) == "blind_spot":
                    stopped += config.DT
                hit |= bool(env.world.pedestrian_hits[slot])
                far = max(far, float(env.world.arc[slot]))
            env.set_player_pose(None)
            arc0 = float(env.world.arc[slot])
            red = False
            # 赤信号で並んでいたときは青になるまで待つ（赤は最長 47 秒）
            for _ in range(int(RESUME_WAIT_SEC / config.DT)):
                env.step(ACT)
                hit |= bool(env.world.pedestrian_hits[slot])
                to_line, phase = env.world.next_signal(slot)
                red |= int(phase) != GREEN and float(to_line) < 10.0
                if float(env.world.arc[slot]) - arc0 > 5.0:
                    break
            held += int(stopped >= 2.0)
            entered += int(far > turn_arc + 0.6)
            resumed += int(float(env.world.arc[slot]) - arc0 > 5.0)
            red_waits += int(red)
            hits += int(hit)
        check(
            f"{preset} {name}: 並走している間は曲がり角の手前で止まり、離れたら曲がり始める・歩行者に当たらない",
            found > 0 and held == found and entered == 0 and resumed == found and hits == 0,
            f"{found} 回のうち止まった {held} 回・曲がり角へ入った {entered} 回・発進した {resumed} 回"
            f"（うち信号待ち {red_waits} 回）・接触 {hits} 回",
        )


def verify_learning_mask(index) -> None:
    print("\n開発モード（safetyAssist）: 操作を引き受けたステップは学習に使わない")
    masked = mismatched = 0
    # 幅が足りず下がらない道もあるので、引き受けが起きる配置が見つかるまで乱数種を変える
    for seed in range(11, 23):
        env = make_env(index, seed=seed, assist=True)
        slot = 0
        start = straight_start(env, slot, PYLON_AHEAD_M + 40.0)
        if start is None:
            continue
        place_on_route(env, slot, start)
        px, py, _h = point_at(env, slot, start + PYLON_AHEAD_M)
        env.world.add_obstacle(px, py, config.OBSTACLE_RADIUS)
        for _ in range(int(40.0 / config.DT)):
            overriding = env.safety.commands[slot].override
            # 車線に沿って走る方策の代わりに経路追従の操作を入れる（方策の出来に結果を左右させない）
            act = ACT.copy()
            act[slot] = env._autopilot(slot, 8.0)
            res = env.step(act)
            masked += int(not bool(res.learn[slot]))
            mismatched += int(bool(res.learn[slot]) == overriding)
        if masked:
            break
    check(
        "学習から外すのは、安全ギミックが操作を引き受けていたステップと一致する",
        masked > 0 and mismatched == 0,
        f"外したステップ {masked} / 食い違い {mismatched}",
    )


def _gae_returns(learn: np.ndarray | None) -> tuple[np.ndarray, int]:
    """報酬 -0.1・価値 5.0 の 5 ステップ（1 台）の価値目標と、学習に使うサンプル数。"""
    buf = RolloutBuffer(5, 1, 1, 1)
    for t in range(5):
        buf.add(
            np.zeros(1), np.zeros(1), np.zeros(1), np.full(1, 5.0), np.full(1, -0.1),
            np.zeros(1, dtype=bool), np.ones(1, dtype=bool),
            learn=None if learn is None else learn[t : t + 1],
        )
    buf.compute_returns_and_advantages(np.full(1, 5.0), np.ones(1, dtype=bool), 0.99, 0.95)
    data = buf.flat_dataset()
    return buf.returns[:5, 0].copy(), 0 if data is None else int(data["returns"].shape[0])


def verify_gae_mask() -> None:
    print("\n学習から外したステップの手前の価値目標（GAE）")
    full, full_n = _gae_returns(None)
    learn = np.array([True, True, False, True, True])
    masked, masked_n = _gae_returns(learn)
    check(
        "外したステップの手前も「この先の価値 0」にせず、次の状態の価値で補う",
        bool(np.all(masked[:2] > 0.8 * full[:2])),
        f"全部学習 {[round(float(x), 3) for x in full]} / t=2 を外す {[round(float(x), 3) for x in masked]}",
    )
    check(
        "手前は外したステップの価値で打ち切る（r + γV = 4.85）",
        abs(float(masked[1]) - (-0.1 + 0.99 * 5.0)) < 1e-4,
        f"t=1 の価値目標 {float(masked[1]):.4f}",
    )
    check(
        "外したステップは学習データに入らない",
        full_n == 5 and masked_n == 4,
        f"全部学習 {full_n} 件 / t=2 を外す {masked_n} 件",
    )


def verify_expert_mask(index) -> None:
    print("\nウォームスタートの教師: 安全ギミックが引き受けたステップの操作は教えない")
    taught = expected = skipped = 0
    for seed in range(11, 23):
        env = make_env(index, seed=seed)
        slot = 0
        start = straight_start(env, slot, PYLON_AHEAD_M + 40.0)
        if start is None:
            continue
        place_on_route(env, slot, start)
        px, py, _h = point_at(env, slot, start + PYLON_AHEAD_M)
        env.world.add_obstacle(px, py, config.OBSTACLE_RADIUS)
        step = env.step
        counts = [0, 0]

        def spy(actions, step=step, counts=counts):
            res = step(actions)
            learn = res.active if res.learn is None else res.learn
            counts[0] += int(np.count_nonzero(res.active & learn))
            counts[1] += int(np.count_nonzero(res.active & ~learn))
            return res

        env.step = spy
        data = collect_expert(env, int(40.0 / config.DT))
        taught, expected, skipped = len(data), counts[0], counts[1]
        if skipped:
            break
    check(
        "教師のサンプル数 = 引き受けていなかったステップの数",
        skipped > 0 and taught == expected,
        f"教師 {taught} 件 / 引き受けなし {expected} 件 / 外した {skipped} 件",
    )


def verify_observation(index) -> None:
    print("\n観測の周囲カメラの欄（Late Fusion）")
    env = make_env(index, seed=5)
    slot = int(np.flatnonzero(env.world.fleet.active)[0])
    env.world.clear_obstacles()
    env.apply_event(type("E", (), {"kind": "clear_obstacles", "payload": {}})())
    base = OBS_OFFSETS["surround"]
    empty = env.observations[slot, base : base + config.OBS_SURROUND_DIM]
    bx, by = ego_to_world(env, slot, -config.VEHICLE_LENGTH / 2 - 4.0, 0.0)
    env.world.add_obstacle(bx, by, config.OBSTACLE_RADIUS)
    obs = env._compute_observations()[slot, base : base + config.OBS_SURROUND_DIM]
    rear = obs[0:3]
    check("何も写っていなければ 0", bool(np.all(empty == 0.0)), f"{np.round(empty, 2).tolist()}")
    check(
        "後ろ 4m のパイロンが後方カメラの欄に入る（前後は負・信頼度 1）",
        rear[0] < 0.0 and abs(rear[1]) < 0.1 and rear[2] > 0.99,
        f"{np.round(rear, 3).tolist()}",
    )
    check(
        "周囲カメラの欄は V2X の欄の手前（後から足した欄は末尾へ足していく）",
        base + config.OBS_SURROUND_DIM == OBS_OFFSETS["v2x"]
        and OBS_OFFSETS["v2x"] + config.OBS_V2X_DIM == config.OBS_DIM,
        f"OBS_DIM {config.OBS_DIM} / 欄の先頭 {base}",
    )


def verify_event_batch(index) -> None:
    print("\n介入をまとめて適用したときの観測の作り直し")
    env = make_env(index, vehicles=4, seed=6)
    fleet = env.world.fleet
    calls = [0]
    compute = env._compute_observations

    def counted():
        calls[0] += 1
        return compute()

    env._compute_observations = counted
    for k in range(10):
        slot = int(np.flatnonzero(fleet.active)[k % int(fleet.active.sum())])
        h = float(fleet.heading[slot])
        env.apply_event(
            InterventionEvent(
                "add_obstacle",
                {"x": float(fleet.x[slot]) + (25.0 + k) * math.cos(h), "y": float(fleet.y[slot]) + (25.0 + k) * math.sin(h)},
            )
        )
    applied = calls[0]
    env.observations
    env.observations
    check(
        "介入 10 件を続けて適用しても、観測は読むときに 1 回だけ作り直す",
        applied == 0 and calls[0] == 1,
        f"適用中 {applied} 回 / 読んだ後 {calls[0]} 回",
    )


def verify_surround_respawn(index) -> None:
    print("\n再スポーンした車の周囲カメラ（CNN で撮り直していない間）")
    env = make_env(index, seed=5)
    slot = int(np.flatnonzero(env.world.fleet.active)[0])
    base = OBS_OFFSETS["surround"]
    env.world.clear_obstacles()
    bx, by = ego_to_world(env, slot, -config.VEHICLE_LENGTH / 2 - 4.0, 0.0)
    env.world.add_obstacle(bx, by, config.OBSTACLE_RADIUS)
    env._compute_observations()
    idx = np.flatnonzero(env.world.fleet.active)
    # 撮り直さなかったステップと同じ形で取り込む（経路が同じなら残す）
    env._merge_surround(idx, {}, {}, False)
    kept = env._encode_last_perception()[slot, base : base + 3]
    moved = env.world.try_respawn(slot)
    env._merge_surround(idx, {}, {}, False)
    after = env._encode_last_perception()[slot, base : base + config.OBS_SURROUND_DIM]
    check(
        "経路が同じなら撮り直すまで前の結果を使う",
        kept[2] > 0.99,
        f"後方の欄 {[round(float(x), 3) for x in kept]}",
    )
    check(
        "再スポーンした車の周囲カメラの欄に、前の場所で写したものを残さない",
        moved and slot not in env.latest_surround and bool(np.all(after == 0.0)),
        f"再スポーン {moved} / 残った結果 {sorted(env.latest_surround.get(slot, {}))}",
    )


def verify_checkpoint() -> None:
    print("\n旧い観測（66 次元）の重みの読み込み")
    params = SimParams()
    old = PPOTrainer(config.OBS_DIM_BEFORE_SURROUND, config.ACTION_DIM, params, config.MAX_VEHICLES, seed=3)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "old.pt"
        old.save(path)
        new = PPOTrainer(config.OBS_DIM, config.ACTION_DIM, params, config.MAX_VEHICLES, seed=9)
        loaded = new.load(path)
    rng = np.random.default_rng(0)
    x_old = rng.uniform(-1, 1, size=(64, config.OBS_DIM_BEFORE_SURROUND)).astype(np.float32)
    x_new = np.concatenate(
        [
            x_old,
            rng.uniform(-1, 1, size=(64, config.OBS_DIM - config.OBS_DIM_BEFORE_SURROUND)).astype(np.float32),
        ],
        axis=1,
    )
    with torch.no_grad():
        d_old, v_old = old.policy.forward(torch.from_numpy(x_old))
        d_new, v_new = new.policy.forward(torch.from_numpy(x_new))
    diff = max(float((d_old.mean - d_new.mean).abs().max()), float((v_old - v_new).abs().max()))
    check(
        "66 次元の重みを 0 埋めで読み込み、方策と価値の出力が元と同じ",
        loaded and new.widened_from == config.OBS_DIM_BEFORE_SURROUND and diff == 0.0,
        f"読み込み {loaded} / 最大差 {diff:.3e}",
    )


def verify_budget(index, preset: str) -> None:
    print("\n1 ステップの時間（真値・8 台・経路追従・4 カメラと安全ギミック込み）")
    env = make_env(index, vehicles=8, walkers=16, seed=0)
    env.autopilot_all = True
    for _ in range(20):
        env.step(ACT)
    runs = []
    gc.disable()
    try:
        for _ in range(3):
            started = time.perf_counter()
            for _ in range(60):
                env.step(ACT)
            runs.append((time.perf_counter() - started) / 60.0 * 1000.0)
    finally:
        gc.enable()
    check(f"{preset}: {BUDGET_MS:.0f}ms 以内", min(runs) <= BUDGET_MS, f"3 回の最小 {min(runs):.1f}ms（{[round(r, 1) for r in runs]}）")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", default="ginza,umeda,sakae")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--kanazawa", action="store_true", help="金沢でも 1 ステップの時間を測る（重い）")
    args = parser.parse_args()

    verify_gear()
    verify_checkpoint()
    verify_gae_mask()
    indexes = {}
    for preset in [p for p in args.presets.split(",") if p]:
        try:
            indexes[preset] = build_map_index(load_map(get_preset(preset)))
        except MapLoadError as exc:
            check(f"{preset}: 地図を読める", False, str(exc))
    if not indexes:
        sys.exit(1)
    first = next(iter(indexes.values()))
    print()
    verify_views(first)
    verify_observation(first)
    verify_surround_respawn(first)
    verify_event_batch(first)
    verify_learning_mask(first)
    verify_expert_mask(first)
    for preset, index in indexes.items():
        print(f"\n===== {preset} =====")
        verify_stuck(index, preset, args.trials, behind=False)
        verify_stuck(index, preset, max(2, args.trials // 2), behind=True)
        verify_turns(index, preset)
        verify_blind_spot(index, preset, args.trials)
        verify_budget(index, preset)
    if args.kanazawa:
        verify_budget(build_map_index(load_map(get_preset("kanazawa"))), "kanazawa")

    print("=" * 72)
    if FAILURES:
        print(f"結果: {len(FAILURES)} 件の不合格")
        sys.exit(1)
    print("結果: すべて合格")


if __name__ == "__main__":
    main()
