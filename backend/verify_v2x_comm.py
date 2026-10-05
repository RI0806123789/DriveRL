# -*- coding: utf-8 -*-
"""車車間通信（V2X）のメッセージ・近傍の選び方・観測の末尾 4 次元・旧い重みの読み込みを検証する。マップのキャッシュは読まない（合成の碁盤の目で走らせる）。"""
from __future__ import annotations

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
from app.contracts import Bounds, MapData, MapEdge, MapNode, SimParams
from app.map.index import build_map_index
from app.percep.encoder import OBS_OFFSETS, encode_observations
from app.percep.types import DEFAULT_CAMERA, DetClass, Detection, PerceptionResult
from app.rl.importer import CheckpointImportError, inspect_checkpoint
from app.rl.policy import ActorCritic
from app.rl.ppo import PPOTrainer
from app.sim.env import SimulationEnv
from app.sim.v2x import V2XMessageRouter, _nearest_danger, vehicle_message

FAILURES: list[str] = []
N = config.MAX_VEHICLES
A = config.ACTION_DIM
V = config.OBS_V2X_DIM


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if ok else 'NG  '}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def grid_map(k: int = 3, step: float = 120.0) -> MapData:
    """建物も信号も無い k×k の碁盤の目（両方向 1 車線ずつ）。"""
    nodes = [MapNode(i * k + j, j * step, i * step) for i in range(k) for j in range(k)]
    edges: list[MapEdge] = []
    for i in range(k):
        for j in range(k):
            for di, dj in ((0, 1), (1, 0)):
                if i + di >= k or j + dj >= k:
                    continue
                a, b = nodes[i * k + j], nodes[(i + di) * k + j + dj]
                for u, v in ((a, b), (b, a)):
                    edges.append(MapEdge(len(edges), u.id, v.id, 1, 7.0, False, 11.1, [(u.x, u.y), (v.x, v.y)], step))
    edge = (k - 1) * step
    return MapData("grid", "grid", 0.0, 0.0, edge, Bounds(-20.0, -20.0, edge + 20.0, edge + 20.0), nodes, edges, [])


GRID = build_map_index(grid_map())


def make_env(vehicles: int, walkers: int = 0, seed: int = 0, *, v2x: bool = True) -> SimulationEnv:
    params = SimParams()
    params.vehicle_count = vehicles
    params.pedestrian_count = walkers
    params.v2x_comm = v2x
    env = SimulationEnv(GRID, params, seed=seed)
    env.reset_all()
    return env


def placed(*xs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x 軸上に車を並べる。メッセージは見分けられるよう車ごとに違う値にする。"""
    xy = np.zeros((N, 2))
    active = np.zeros(N, dtype=bool)
    messages = np.zeros((N, V), dtype=np.float32)
    for slot, x in enumerate(xs):
        xy[slot] = (x, 0.0)
        active[slot] = True
        messages[slot] = [0.1 * (slot + 1), (-1) ** slot, 0.05 * slot, 1.0 - 0.1 * slot]
    return xy, active, messages


router = V2XMessageRouter()

# ---------------------------------------------------------------------------
section("1. 近傍の選び方と平均")
xy, active, msg = placed(0.0, 15.0)
inbox, links = router.route_and_aggregate(xy, active, msg)
check("15m 離れた 2 台は互いに相手のメッセージを受け取る", np.allclose(inbox[0], msg[1]) and np.allclose(inbox[1], msg[0]))
check("受け取った相手が links に出る", links == {0: (1,), 1: (0,)}, f"{links}")

xy, active, msg = placed(0.0, 50.0)
inbox, links = router.route_and_aggregate(xy, active, msg)
check("50m 離れていれば届かず、双方 [0, 0, 0, 0]", bool(np.all(inbox == 0.0)) and links == {})

xy, active, msg = placed(0.0, config.V2X_RANGE_M)
inbox, links = router.route_and_aggregate(xy, active, msg)
check(f"ちょうど {config.V2X_RANGE_M:.0f}m は届く", links.get(0) == (1,))

xy, active, msg = placed(0.0, 5.0, 12.0, 20.0)
inbox, links = router.route_and_aggregate(xy, active, msg)
check(
    "3 台以上が近いときは、近い 2 台だけを平均する",
    links[0] == (1, 2) and np.allclose(inbox[0], (msg[1] + msg[2]) / 2.0),
    f"車 0 の相手 {links[0]}",
)
check("相手の数は最大 2", all(len(p) <= config.V2X_MAX_PEERS for p in links.values()))

xy, active, msg = placed(0.0, 10.0)
active[1] = False
inbox, links = router.route_and_aggregate(xy, active, msg)
check("走っていない車とはつながらない", links == {} and bool(np.all(inbox == 0.0)))

xy, active, msg = placed(0.0)
inbox, links = router.route_and_aggregate(xy, active, msg)
check("1 台だけならゼロ埋め（単独走行でも観測が壊れない）", bool(np.all(inbox == 0.0)) and links == {})

# ---------------------------------------------------------------------------
section("2. メッセージの 4 要素")
m = vehicle_message(0.5, 1, 15.0, 6.0)
check("車速は v / v_max", math.isclose(float(m[0]), 0.5))
check("右折は +1・左折は -1・直進は 0", float(m[1]) == 1.0 and float(vehicle_message(0, -1, 99, 99)[1]) == -1.0 and float(vehicle_message(0, 0, 99, 99)[1]) == 0.0)
check("危険は max(0, 1 - d/30)", math.isclose(float(m[2]), 0.5))
check("交差点への近さは max(0, 1 - d/30)", math.isclose(float(m[3]), 0.8, rel_tol=1e-6))
far = vehicle_message(1.7, 0, float("inf"), float("inf"))
check("遠ければ 0（inf でも NaN にならない）・車速は 1 で頭打ち", float(far[0]) == 1.0 and float(far[2]) == 0.0 and float(far[3]) == 0.0 and bool(np.all(np.isfinite(far))))
check("後退中（負の速さ）は 0", float(vehicle_message(-0.1, 0, 99, 99)[0]) == 0.0)
check("どの要素も -1..1", bool(np.all(np.abs(vehicle_message(9.0, 3, -5.0, -5.0)) <= 1.0)))


def box_at(distance: float, cls: DetClass) -> Detection:
    return Detection(cls, 0.45, 0.4, 0.55, 0.8, 0.9, distance=distance)


front = PerceptionResult(slot=0, detections=[box_at(40.0, DetClass.PEDESTRIAN), box_at(12.0, DetClass.OBSTACLE), box_at(3.0, DetClass.VEHICLE)])
d = _nearest_danger(front, DEFAULT_CAMERA, None)
check("危険は自車のカメラに写った歩行者・障害物のうち近いもの（車両は数えない）", 11.0 < d < 13.5, f"{d:.2f} m")
check("何も写っていなければ inf", _nearest_danger(PerceptionResult(slot=0), DEFAULT_CAMERA, None) == float("inf"))

# ---------------------------------------------------------------------------
section("3. 観測の V2X の 4 次元（合成の碁盤の目）")
base = OBS_OFFSETS["v2x"]
tail = base + V
check(
    f"V2X の欄は観測の 75〜78 で、後ろには死角の欄だけが続く（観測は {config.OBS_DIM} 次元）",
    base == 75 and tail == OBS_OFFSETS["occlusion"] and tail + config.OBS_OCCLUSION_DIM == config.OBS_DIM,
    f"欄の先頭 {base}",
)
env = make_env(N, walkers=16)
consistent = True
linked = 0
finite = True
for _ in range(400):
    env.step(np.zeros((N, A), dtype=np.float32), expert=env.active_mask.copy())
    obs = env.observations
    fleet = env.world.fleet
    messages = env.v2x.compute_messages(env.world, float(env.params.max_speed), env.latest_perception, env.latest_surround, DEFAULT_CAMERA)
    inbox, links = env.v2x.route_and_aggregate(np.column_stack((fleet.x, fleet.y)), fleet.active, messages)
    consistent &= bool(np.allclose(obs[:, base:tail], inbox, atol=1e-6)) and links == env.v2x_links
    finite &= bool(np.all(np.isfinite(obs[:, base:tail]))) and bool(np.all(np.abs(obs[:, base:tail]) <= 1.0 + 1e-6))
    linked += len(links)
    for slot in np.flatnonzero(fleet.active):
        s = int(slot)
        signal_m, _ = env.world.next_signal(s)
        want = min(1.0, max(0.0, 1.0 - min(env.world.next_junction(s), signal_m) / 30.0))
        consistent &= math.isclose(float(messages[s, 3]), want, abs_tol=1e-6)
        consistent &= float(messages[s, 1]) == float(np.sign(env.world.turn_signal[s]))
check("観測の V2X の 4 次元は、近傍から受け取ったメッセージの平均と一致する", consistent)
check("V2X の欄は有限で -1..1", finite)
check("8 台で走らせるとリンクができる", linked > 0, f"400 ステップで延べ {linked} 台分")
frame = env.snapshot(0, 0.0)
wire = [v.to_wire() for v in frame.vehicles if v.active]
with_links = [w for w in wire if "v2xConnectedIds" in w]
check(
    "frame の車両に v2xConnectedIds が載り、links と一致する",
    all(w.get("v2xConnectedIds", []) == list(env.v2x_links.get(w["id"], ())) for w in wire)
    and len(with_links) == len(env.v2x_links),
    f"{len(with_links)} 台",
)
check("リンクの無い車には v2xConnectedIds を載せない（転送量を増やさない）", all(("v2xConnectedIds" in w) == (w["id"] in env.v2x_links) for w in wire))

lonely = make_env(1)
for _ in range(50):
    lonely.step(np.zeros((N, A), dtype=np.float32), expert=lonely.active_mask.copy())
check("単独走行なら V2X の欄は 0", bool(np.all(lonely.observations[:, base:tail] == 0.0)) and lonely.v2x_links == {})

on = make_env(N, seed=4)
off = make_env(N, seed=4, v2x=False)
rng = np.random.default_rng(5)
head_same = True
tail_zero = True
for _ in range(150):
    act = rng.uniform(-1, 1, (N, A)).astype(np.float32)
    ra = on.step(act)
    rb = off.step(act)
    head_same &= (
        np.array_equal(ra.obs[:, :base], rb.obs[:, :base])
        and np.array_equal(ra.obs[:, tail:], rb.obs[:, tail:])
        and np.array_equal(ra.rewards, rb.rewards)
    )
    tail_zero &= bool(np.all(rb.obs[:, base:tail] == 0.0))
check("v2xComm を切ると V2X の 4 次元は 0、リンクも無い", tail_zero and off.v2x_links == {})
check("V2X は自分の 4 次元だけを変え、ほかの欄と報酬は切ったときと完全に一致する", head_same)

perc = env.latest_perception
enc_with = encode_observations(env.world, env.params, perc, v2x=np.full((N, V), 0.25, dtype=np.float32))
enc_without = encode_observations(env.world, env.params, perc)
active_rows = env.world.fleet.active
check(
    "encode_observations は v2x を自分の欄へ入れるだけ（ほかの欄は変えない）",
    np.array_equal(enc_with[:, :base], enc_without[:, :base]) and np.array_equal(enc_with[:, tail:], enc_without[:, tail:]),
)
check("走っていない車の行は 0 のまま", bool(np.all(enc_with[~active_rows] == 0.0)))

# ---------------------------------------------------------------------------
section(f"4. {config.OBS_DIM} 次元の方策と、旧い重みの読み込み")
policy = ActorCritic(config.OBS_DIM, A)
x = torch.randn(32, config.OBS_DIM)
dist, value = policy.forward(x)
loss = dist.log_prob(torch.zeros(32, A)).sum() + value.sum()
loss.backward()
grads_ok = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in policy.parameters())
check(f"{config.OBS_DIM} 次元で順伝播・逆伝播が通る（NaN なし）", grads_ok and dist.mean.shape == (32, A))

params = SimParams()
for old_dim in config.OBS_WIDENABLE_DIMS:
    old = PPOTrainer(old_dim, A, params, N, seed=3)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"old{old_dim}.pt"
        old.save(path)
        try:
            info = inspect_checkpoint(path, expected_obs_dim=config.OBS_DIM, expected_action_dim=A, expected_hidden_sizes=old.hidden_sizes)
            accepted = info.obs_dim == old_dim
        except CheckpointImportError:
            accepted = False
        new = PPOTrainer(config.OBS_DIM, A, params, N, seed=9)
        loaded = new.load(path)
    rng = np.random.default_rng(0)
    x_old = rng.uniform(-1, 1, size=(64, old_dim)).astype(np.float32)
    x_new = np.concatenate([x_old, rng.uniform(-1, 1, size=(64, config.OBS_DIM - old_dim)).astype(np.float32)], axis=1)
    with torch.no_grad():
        d_old, v_old = old.policy.forward(torch.from_numpy(x_old))
        d_new, v_new = new.policy.forward(torch.from_numpy(x_new))
    diff = max(float((d_old.mean - d_new.mean).abs().max()), float((v_old - v_new).abs().max()))
    # 足した入力の重みは 0 だが、行列積の足し算の順序が入力の幅で変わるので float32 の丸めの差は残る
    check(
        f"{old_dim} 次元の重みを 0 埋めで読み込み、方策と価値の出力が元と同じ（丸めの差 1e-6 以内）",
        loaded and new.widened_from == old_dim and diff <= 1e-6,
        f"読み込み {loaded} / 最大差 {diff:.3e}",
    )
    check(f"{old_dim} 次元のファイルは書き出しからの読み込み（importer）も受け付ける", accepted)

odd = PPOTrainer(config.OBS_DIM - 1, A, params, N, seed=1)
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "odd.pt"
    odd.save(path)
    rejected = not PPOTrainer(config.OBS_DIM, A, params, N, seed=2).load(path)
check(f"移行の対象でない次元（{config.OBS_DIM - 1}）は読み込まない", rejected)

trainer = PPOTrainer(config.OBS_DIM, A, SimParams(rollout_length=16), N, seed=0)
env2 = make_env(N, seed=7)
stats = None
while stats is None:
    obs = env2.observations
    act, lp, val = trainer.act(obs, env2.active_mask)
    r = env2.step(act)
    trainer.store(obs=obs, actions=act, log_probs=lp, values=val, rewards=r.rewards, dones=r.dones, active=r.active, learn=r.learn)
    stats = trainer.maybe_update(r.obs, r.active)
check(f"{config.OBS_DIM} 次元の観測で PPO の更新が最後まで回る（NaN なし）", all(math.isfinite(v) for v in stats.values()), f"{ {k: round(v, 4) for k, v in stats.items()} }")

# ---------------------------------------------------------------------------
section("5. 所要時間（8 台）")
route_times: list[float] = []
exchange_times: list[float] = []
xy_all = np.column_stack((env.world.fleet.x, env.world.fleet.y))
for _ in range(2000):
    started = time.perf_counter()
    router.route_and_aggregate(xy_all, env.world.fleet.active, messages)
    route_times.append(time.perf_counter() - started)
for _ in range(500):
    started = time.perf_counter()
    env._exchange_v2x(env.latest_perception, DEFAULT_CAMERA)
    exchange_times.append(time.perf_counter() - started)
route_ms = float(np.median(route_times)) * 1000.0
exchange_ms = float(np.median(exchange_times)) * 1000.0
check("近傍の選び方と平均は 0.2ms 未満（issue の予算）", route_ms < 0.2, f"中央値 {route_ms:.3f} ms")
check("メッセージ作りを含めても 1 ステップ 1ms 未満", exchange_ms < 1.0, f"中央値 {exchange_ms:.3f} ms")

print()
if FAILURES:
    print(f"NG: {len(FAILURES)} 件")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("OK: すべての検査に通りました")
