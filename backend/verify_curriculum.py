# -*- coding: utf-8 -*-
"""ヒヤリハットのオートカリキュラム（難易度の昇降格・飛び出し・前走車の急制動）を検証する。マップのキャッシュは読まない（合成の道路で走らせる）。"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
for st in (sys.stdout, sys.stderr):
    try:
        st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np

from app import config
from app.contracts import Bounds, MapData, MapEdge, MapNode, SimParams
from app.map.index import build_map_index
from app.sim.curriculum import (
    CURRICULUM_COOLDOWN_EPISODES,
    CURRICULUM_WINDOW,
    INCIDENT_MAX_PROB,
    INCIDENT_WINDOW_STEPS,
    JAYWALK_SPEED_MPS,
    KIND_JAYWALK,
    KIND_LEADER_BRAKE,
    LEADER_BRAKE_STEPS,
    LEVEL_STEP,
    CurriculumManager,
)
from app.sim.env import SimulationEnv
from app.sim.pedestrians import PedestrianCrowd

FAILURES: list[str] = []
N = config.MAX_VEHICLES
A = config.ACTION_DIM


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if ok else 'NG  '}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def two_way(nodes: list[MapNode], pairs: list[tuple[int, int]], width: float = 7.0) -> list[MapEdge]:
    edges: list[MapEdge] = []
    for i, j in pairs:
        a, b = nodes[i], nodes[j]
        length = math.hypot(b.x - a.x, b.y - a.y)
        for u, v in ((a, b), (b, a)):
            edges.append(MapEdge(len(edges), u.id, v.id, 1, width, False, 11.1, [(u.x, u.y), (v.x, v.y)], length))
    return edges


def grid_map(k: int = 3, step: float = 120.0) -> MapData:
    """建物も信号も無い k×k の碁盤の目。"""
    nodes = [MapNode(i * k + j, j * step, i * step) for i in range(k) for j in range(k)]
    pairs = [
        (i * k + j, (i + di) * k + j + dj)
        for i in range(k)
        for j in range(k)
        for di, dj in ((0, 1), (1, 0))
        if i + di < k and j + dj < k
    ]
    edge = (k - 1) * step
    return MapData("grid", "grid", 0.0, 0.0, edge, Bounds(-20.0, -20.0, edge + 20.0, edge + 20.0), nodes, two_way(nodes, pairs), [])


def line_map() -> MapData:
    """東西にまっすぐな 750m の道（両方向 1 車線ずつ）。"""
    nodes = [MapNode(i, i * 150.0, 0.0) for i in range(6)]
    return MapData("line", "line", 0.0, 0.0, 750.0, Bounds(-20.0, -40.0, 770.0, 40.0), nodes, two_way(nodes, [(i, i + 1) for i in range(5)]), [])


GRID = build_map_index(grid_map())
LINE = build_map_index(line_map())


def make_env(index, vehicles: int, walkers: int, seed: int = 0, *, enabled: bool = True) -> SimulationEnv:
    params = SimParams()
    params.vehicle_count = vehicles
    params.pedestrian_count = walkers
    params.incident_curriculum = enabled
    env = SimulationEnv(index, params, seed=seed)
    env.reset_all()
    return env


# ---------------------------------------------------------------------------
section("1. 難易度 0 では何も起こさない")
cm = CurriculumManager(np.random.default_rng(0))
state_before = cm.rng.bit_generator.state
fired = [cm.should_trigger_pedestrian_jaywalk(12.0, 3.0, 8.0) for _ in range(2000)]
fired += [cm.should_trigger_leader_braking(6.0, 8.0) for _ in range(2000)]
check("L = 0 では飛び出しも急制動も常に False", not any(fired))
check("L = 0 の間は乱数を 1 つも引かない（ほかの乱数の並びを変えない）", cm.rng.bit_generator.state == state_before)

cm.level = 1.0
p1 = float(np.mean([cm.should_trigger_pedestrian_jaywalk(12.0, 3.0, 8.0) for _ in range(20000)]))
check(f"L = 1 で条件がそろえば確率 {INCIDENT_MAX_PROB} で起こす", abs(p1 - INCIDENT_MAX_PROB) < 0.01, f"実際 {p1:.3f}")
check("条件の外（前方 20m）では L = 1 でも起こさない", not any(cm.should_trigger_pedestrian_jaywalk(20.0, 3.0, 8.0) for _ in range(500)))
check("条件の外（横 1m）では起こさない", not any(cm.should_trigger_pedestrian_jaywalk(12.0, 1.0, 8.0) for _ in range(500)))
check("条件の外（4m/s）では起こさない", not any(cm.should_trigger_pedestrian_jaywalk(12.0, 3.0, 4.0) for _ in range(500)))
check("急制動は車間 12m 以上では起こさない", not any(cm.should_trigger_leader_braking(12.5, 8.0) for _ in range(500)))

# ---------------------------------------------------------------------------
section("2. 難易度の昇降格")
cm = CurriculumManager(np.random.default_rng(0))
levels = []
for _ in range(CURRICULUM_WINDOW):
    cm.record_episode_end(True, False)
    levels.append(cm.level)
check(f"{CURRICULUM_WINDOW} 回続けて到達すると L が上がる", levels[-1] == LEVEL_STEP and all(v == 0.0 for v in levels[:-1]), f"L = {levels[-1]}")
for _ in range(60):
    cm.record_episode_end(True, False)
    levels.append(cm.level)
steps = np.diff([0.0, *levels])
check("上がるのは 1 回 0.05 ずつで、下がらない", bool(np.all((np.abs(steps) < 1e-9) | (np.abs(steps - LEVEL_STEP) < 1e-9))))
check(
    f"変えた後は {CURRICULUM_COOLDOWN_EPISODES} エピソード据え置く（段階的に上がる）",
    math.isclose(cm.level, LEVEL_STEP * (1 + 60 // CURRICULUM_COOLDOWN_EPISODES)),
    f"80 エピソードで L = {cm.level}",
)
cm2 = CurriculumManager(np.random.default_rng(0))
for _ in range(400):
    cm2.record_episode_end(True, False)
check("上限は 1.0", cm2.level == 1.0)

cm.level = 0.5
before = cm.level
for i in range(CURRICULUM_WINDOW):
    cm.record_episode_end(i % 5 >= 2, i % 5 < 2)
check("20 回中 8 回衝突（40%）で降格する", cm.level < before, f"{before} → {cm.level}")
cm3 = CurriculumManager(np.random.default_rng(0))
cm3.level = 0.4
for i in range(CURRICULUM_WINDOW):
    cm3.record_episode_end(i % 10 != 0, False)
check("到達 90%・衝突 0% なら昇格、到達 80% なら据え置き", cm3.level == 0.45)
cm4 = CurriculumManager(np.random.default_rng(0))
cm4.level = 0.4
for i in range(CURRICULUM_WINDOW):
    cm4.record_episode_end(i % 5 != 0, False)
check("到達 80%・衝突 0% は据え置き", cm4.level == 0.4)
cm5 = CurriculumManager(np.random.default_rng(0))
for _ in range(CURRICULUM_WINDOW * 2):
    cm5.record_episode_end(False, True)
check("下限は 0.0", cm5.level == 0.0)

# ---------------------------------------------------------------------------
section("3. 飛び出しの動き（合成の碁盤の目）")
crowd = PedestrianCrowd(GRID, np.random.default_rng(1))
crowd.set_count(8)
slot = int(np.flatnonzero(crowd.active)[0])
crowd.edge[slot] = 0
crowd.arc[slot] = 60.0
crowd.side[slot] = 1
crowd.dir[slot] = 1
crowd.crossing[slot] = False
crowd.waiting[slot] = False
crowd._refresh_pose()
_cx, _cy, tx, ty = crowd.net.sample(np.array([0]), np.array([60.0]))
tangent = np.array([float(tx[0]), float(ty[0])])
normal = np.array([-tangent[1], tangent[0]])
start = np.array([crowd.x[slot], crowd.y[slot]])
check("歩道を歩いている人は飛び出させられる", crowd.force_cross_street(slot, JAYWALK_SPEED_MPS[1]))
check("すでに横断中の人は飛び出させない", not crowd.force_cross_street(slot, JAYWALK_SPEED_MPS[1]))
dt = config.DT
path = [start.copy()]
for _ in range(400):
    crowd.step(dt)
    path.append(np.array([crowd.x[slot], crowd.y[slot]]))
    if not crowd.crossing[slot]:
        break
path_arr = np.array(path)
moves = np.diff(path_arr, axis=0)
speeds = np.hypot(moves[:, 0], moves[:, 1]) / dt
along = np.abs(moves @ tangent)
across = moves @ normal
check(f"速さは小走りの上限 {JAYWALK_SPEED_MPS[1]} m/s 以内", float(speeds.max()) <= JAYWALK_SPEED_MPS[1] + 1e-6, f"最大 {speeds.max():.3f} m/s")
check("小走りで渡る（1.2m/s 以上）", float(np.median(speeds[speeds > 0])) >= JAYWALK_SPEED_MPS[0] - 1e-6, f"中央値 {np.median(speeds[speeds > 0]):.3f} m/s")
check("道路に直角に動く（道なりには進まない）", float(along.max()) < 1e-6, f"道なりの成分の最大 {along.max():.2e} m")
check("車道の反対側へ向かって動く", bool(np.all(across < 1e-9)), f"最初の横の動き {across[0]:.4f} m")
check("渡りきったら飛び出しの印は外れ、反対側の歩道に立つ", not crowd.jaywalking[slot] and int(crowd.side[slot]) == -1)
width = abs(float((path_arr[-1] - start) @ normal))
check("歩道から反対側の歩道まで渡る", abs(width - 2.0 * float(crowd.net.walk_offset[0])) < 1e-6, f"{width:.2f} m")
cm6 = CurriculumManager(np.random.default_rng(5))
dashes = [cm6.dash_speed() for _ in range(2000)]
check("小走りの速さは 1.2〜1.8 m/s から選ぶ", min(dashes) >= JAYWALK_SPEED_MPS[0] and max(dashes) <= JAYWALK_SPEED_MPS[1])

# ---------------------------------------------------------------------------
section("4. 環境の中での飛び出し（碁盤の目・8 台・歩行者 64 人・お手本で運転）")
env = make_env(GRID, N, 64, seed=0)
env.curriculum.level = 1.0
env.curriculum.max_probability = 1.0
hook_times: list[float] = []
orig_update = env._update_incidents


def timed_update(*args, **kwargs) -> None:
    started = time.perf_counter()
    orig_update(*args, **kwargs)
    hook_times.append(time.perf_counter() - started)


env._update_incidents = timed_update
geometry_ok = True
jaywalk_seen = 0
last_triggered = 0
for _ in range(1500):
    env.step(np.zeros((N, A), dtype=np.float32), expert=env.active_mask.copy())
    if env.curriculum.triggered_by_kind[KIND_JAYWALK] > jaywalk_seen:
        jaywalk_seen = env.curriculum.triggered_by_kind[KIND_JAYWALK]
        runners = np.flatnonzero(env.world.crowd.jaywalking & (env.world.crowd.cross < 0.05))
        geometry_ok &= runners.size > 0
check("L = 1 なら飛び出しが起きる", jaywalk_seen > 0, f"{jaywalk_seen} 件")
check("起こした直後の歩行者は飛び出しの印が付いて渡り始めている", geometry_ok)
check("1 台に 8 秒は続けて起こさない（1,500 ステップ・8 台で 75 件以下）", env.curriculum.triggered <= 75, f"{env.curriculum.triggered} 件")
median_ms = float(np.median(hook_times)) * 1000.0
p95_ms = float(np.percentile(hook_times, 95)) * 1000.0
check("見届けと判定は 1 ステップ 2ms 未満（8 台・64 人）", p95_ms < 2.0, f"中央値 {median_ms:.3f} ms / 95% {p95_ms:.3f} ms")
check(
    "お手本が運転していた車のヒヤリハットは「自力で回避」に数えない",
    env.curriculum.avoided_rate == 0.0,
    f"{env.curriculum.avoided_rate}",
)

env_off = make_env(GRID, N, 64, seed=0, enabled=False)
env_off.curriculum.level = 1.0
env_off.curriculum.max_probability = 1.0
for _ in range(300):
    env_off.step(np.zeros((N, A), dtype=np.float32), expert=env_off.active_mask.copy())
check("incidentCurriculum が false なら L = 1 でも起こさない", env_off.curriculum.triggered == 0)

env_pr = make_env(GRID, N, 64, seed=0)
env_pr.autopilot_all = True
env_pr.curriculum.level = 1.0
env_pr.curriculum.max_probability = 1.0
for _ in range(300):
    env_pr.step(np.zeros((N, A), dtype=np.float32))
check("実用モード（全車が経路追従）では起こさない", env_pr.curriculum.triggered == 0)

env_a = make_env(GRID, N, 64, seed=3)
env_b = make_env(GRID, N, 64, seed=3, enabled=False)
rng = np.random.default_rng(4)
same = True
for _ in range(200):
    act = rng.uniform(-1, 1, (N, A)).astype(np.float32)
    ra = env_a.step(act)
    rb = env_b.step(act)
    same &= np.array_equal(ra.obs, rb.obs) and np.array_equal(ra.rewards, rb.rewards) and np.array_equal(ra.learn, rb.learn)
check("難易度 0 のままなら、切ったときと観測・報酬・学習の印が完全に一致する", same)

# ---------------------------------------------------------------------------
section("5. 前走車の急制動（まっすぐな道・2 台）")


def follow_scene(follower_expert: bool, follower_accel: float = 0.0) -> tuple[SimulationEnv, list[dict[str, object]]]:
    env = make_env(LINE, 2, 0, seed=0)
    env.relocate_vehicle(0, (100.0, 1.75))
    env.relocate_vehicle(1, (125.0, 1.75))
    env.curriculum.level = 1.0
    env.curriculum.max_probability = 1.0
    expert = np.zeros(N, dtype=bool)
    expert[0] = follower_expert
    log: list[dict[str, object]] = []
    for _ in range(420):
        act = np.zeros((N, A), dtype=np.float32)
        act[1, 0] = 0.15
        act[0, 0] = follower_accel
        braking_before = env.curriculum.braking_mask().copy()
        result = env.step(act, expert=expert)
        log.append(
            {
                "forced": bool(braking_before[1]),
                "learn": bool(result.learn[1]),
                "throttle": float(env.world.throttle[1]),
                "lamp": bool(env.world.braking[1]),
                "assisted1": bool(result.assisted[1]),
                "episodes": [e.reason for e in result.episodes],
            }
        )
        if result.episodes:
            break
    return env, log


env_f, log_f = follow_scene(True)
forced_steps = [row for row in log_f if row["forced"]]
check("前走車に付いた瞬間に急制動が起きる", env_f.curriculum.triggered_by_kind[KIND_LEADER_BRAKE] > 0)
runs: list[int] = []
current = 0
for row in log_f:
    if row["forced"]:
        current += 1
    elif current:
        runs.append(current)
        current = 0
check(
    f"急制動は {LEADER_BRAKE_STEPS} ステップ（1.5 秒）ずつ（観測の終わりで途中のものは除く）",
    len(runs) > 0 and all(r == LEADER_BRAKE_STEPS for r in runs) and current <= LEADER_BRAKE_STEPS,
    f"続いたステップ {runs}" + (f" + 途中 {current}" if current else ""),
)
check("急制動の間、前走車のアクセルは -1.0", all(row["throttle"] == -1.0 for row in forced_steps))
check("急制動の間、前走車の制動灯が点く", all(row["lamp"] for row in forced_steps))
check("急制動の間の前走車のステップは学習に使わない（learn が偽）", all(not row["learn"] for row in forced_steps))
check("急制動が終われば前走車は学習に戻る", any(row["learn"] for row in log_f if not row["forced"]))
check("お手本が運転する後続車は追突しない", all(not row["episodes"] for row in log_f))

env_p, log_p = follow_scene(False, follower_accel=0.6)
check("自分で止まらない後続車は、急制動で前走車に追突する", any("collision" in row["episodes"] for row in log_p), f"{[r['episodes'] for r in log_p if r['episodes']]}")
check("追突したヒヤリハットは「回避できなかった」に数える", env_p.curriculum.avoided_rate == 0.0, f"{env_p.curriculum.avoided_rate}")

# ---------------------------------------------------------------------------
section("6. 見届け")
cm7 = CurriculumManager(np.random.default_rng(0), num_agents=4)
none = np.zeros(4, dtype=bool)
alive = np.ones(4, dtype=bool)
cm7.start(0, KIND_JAYWALK)
cm7.start(1, KIND_JAYWALK)
cm7.start(2, KIND_LEADER_BRAKE, leader=3)
check("見届け中の車には新しく起こさない", not cm7.can_start(0))
check("急制動を掛けた前走車は braking_mask に出る", bool(cm7.braking_mask()[3]))
crash = none.copy()
crash[1] = True
assisted = none.copy()
assisted[2] = True
cm7.advance(crash, crash, assisted, alive)
for _ in range(INCIDENT_WINDOW_STEPS):
    cm7.advance(none, none, none, alive)
check("1 件は衝突・1 件はお手本・1 件は自力で回避 → 回避率 1/3", cm7.avoided_rate is not None and math.isclose(cm7.avoided_rate, 1 / 3))
check("見届けが済んでも 8 秒の間隔が明けるまでは起こさない", not cm7.can_start(0))
for _ in range(200):
    cm7.advance(none, none, none, alive)
check("間隔が明ければまた起こせる", cm7.can_start(0))
cm8 = CurriculumManager(np.random.default_rng(0), num_agents=2)
check("まだ見届けていなければ回避率は None（0 ではない）", cm8.avoided_rate is None)
cm8.start(0, KIND_JAYWALK)
ended = np.array([True, False])
cm8.advance(np.zeros(2, dtype=bool), ended, np.zeros(2, dtype=bool), np.ones(2, dtype=bool))
check("見届け中に到達・打ち切りで終わったら回避に数える", cm8.avoided_rate == 1.0)
cm8.level = 0.7
cm8.start(1, KIND_LEADER_BRAKE, leader=0)
cm8.reset_incidents()
check("reset_incidents は見届けと急制動を捨て、難易度は残す", not cm8.braking_mask().any() and not cm8.incident_active().any() and cm8.level == 0.7)

print()
if FAILURES:
    print(f"NG: {len(FAILURES)} 件")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("OK: すべての検査に通りました")
