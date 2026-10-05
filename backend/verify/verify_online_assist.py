# -*- coding: utf-8 -*-
"""オンライン模倣（経路追従の割り込みと模倣の損失）を検証する。マップのキャッシュは読まない（合成の碁盤の目で走らせる）。"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for st in (sys.stdout, sys.stderr):
    try:
        st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch

from app import config
from app.contracts import AssistDanger, Bounds, MapData, MapEdge, MapNode, SimParams
from app.map.index import build_map_index
from app.rl.buffer import RolloutBuffer
from app.rl.online_assist import (
    ASSIST_OVERRIDE_HOLD_STEPS,
    ASSIST_P_MIN,
    ASSIST_SEGMENT_STEPS,
    ASSIST_WARMUP_STEPS,
    STALL_HOLD_STEPS,
    STALL_STEPS,
    OnlineAssistController,
)
from app.rl.ppo import PPOTrainer
from app.sim.env import SimulationEnv

FAILURES: list[str] = []
N = config.MAX_VEHICLES
A = config.ACTION_DIM


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if ok else 'NG  '}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def grid_map(k: int = 5, step: float = 120.0) -> MapData:
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
                    edges.append(
                        MapEdge(len(edges), u.id, v.id, 1, 7.0, False, 11.1, [(u.x, u.y), (v.x, v.y)], step)
                    )
    edge = (k - 1) * step
    return MapData("grid", "grid", 0.0, 0.0, edge, Bounds(-20.0, -20.0, edge + 20.0, edge + 20.0), nodes, edges, [])


def make_env(seed: int = 0) -> SimulationEnv:
    params = SimParams()
    params.vehicle_count = N
    params.pedestrian_count = 0
    env = SimulationEnv(MAP_INDEX, params, seed=seed)
    env.reset_all()
    return env


MAP_INDEX = build_map_index(grid_map())

# ---------------------------------------------------------------------------
section("1. 割り込みの確率の下がり方")
ctl = OnlineAssistController(N, seed=0)
check("t = 0 で 1.0", ctl.get_assist_probability(0) == 1.0)
check(
    f"t = T_warmup（{ASSIST_WARMUP_STEPS}）で下限 {ASSIST_P_MIN}",
    math.isclose(ctl.get_assist_probability(ASSIST_WARMUP_STEPS), ASSIST_P_MIN),
)
check("それより先も下限のまま", ctl.get_assist_probability(ASSIST_WARMUP_STEPS * 10) == ASSIST_P_MIN)
curve = [ctl.get_assist_probability(t) for t in range(0, ASSIST_WARMUP_STEPS + 1, 100)]
diffs = np.diff(curve)
check("単調に減る（増えない）", bool(np.all(diffs <= 0.0)))
check("滑らか（100 ステップで 0.6% 以上は跳ばない）", float(np.max(np.abs(diffs))) <= 0.006, f"最大 {np.max(np.abs(diffs)):.4f}")
check("負の t は 0 と同じ", ctl.get_assist_probability(-50) == 1.0)

# ---------------------------------------------------------------------------
section("2. 危険なら確率に関わらず割り込む")
late = ASSIST_WARMUP_STEPS * 100
all_active = np.ones(N, dtype=bool)


def fresh(p_min: float = 0.0) -> OnlineAssistController:
    return OnlineAssistController(N, seed=1, p_min=p_min)


def danger(**kw: object) -> AssistDanger:
    d = AssistDanger.safe(N)
    d.speed[:] = 8.0
    for key, (slot, value) in kw.items():
        getattr(d, key)[slot] = value
    return d


c = fresh()
safe_mask = c.decide(late, all_active, danger())
check("危険が無く確率 0 なら割り込まない", not safe_mask.any())
c = fresh()
m = c.decide(late, all_active, danger(lane_offset=(2, 2.0)))
check("車線中心から 2.0m ずれたスロットだけ割り込む", bool(m[2]) and int(m.sum()) == 1)
c = fresh()
m = c.decide(late, all_active, danger(ttc=(3, 0.8)))
check("TTC 0.8 秒で割り込む", bool(m[3]) and int(m.sum()) == 1)
c = fresh()
m = c.decide(late, all_active, danger(ttc=(3, 1.5)))
check("TTC 1.5 秒では割り込まない", not m.any())
c = fresh()
m = c.decide(late, all_active, danger(red_distance=(4, 3.0)))
check("赤信号の 3m 手前・8m/s で割り込む", bool(m[4]))
c = fresh()
d = danger(red_distance=(4, 3.0))
d.speed[4] = 1.0
check("赤信号の 3m 手前でも 1m/s なら割り込まない", not c.decide(late, all_active, d)[4])
c = fresh()
d = danger(lane_offset=(1, 2.0))
inactive = all_active.copy()
inactive[1] = False
check("走っていないスロットには割り込まない", not c.decide(late, inactive, d)[1])

c = fresh()
first = c.decide(late, all_active, danger(lane_offset=(5, 2.0)))
held = [bool(c.decide(late, all_active, danger())[5]) for _ in range(ASSIST_OVERRIDE_HOLD_STEPS + 5)]
check(
    f"危険が去っても {ASSIST_OVERRIDE_HOLD_STEPS} ステップは運転し続け、その後は返す",
    bool(first[5]) and all(held[: ASSIST_OVERRIDE_HOLD_STEPS - 1]) and not any(held[ASSIST_OVERRIDE_HOLD_STEPS:]),
    f"続いたステップ {sum(held)}",
)

c = fresh()
stall = AssistDanger.safe(N)
stall.held[:] = False
stall.speed[:] = 0.0
trig = [bool(c.decide(late, all_active, stall)[0]) for _ in range(STALL_STEPS + 2)]
check(
    f"理由なく止まったまま {STALL_STEPS} ステップで割り込む（固まりの防止）",
    not any(trig[: STALL_STEPS - 1]) and trig[STALL_STEPS - 1],
    f"最初に割り込んだステップ {trig.index(True) + 1 if any(trig) else '-'}",
)
kept = [bool(c.decide(late, all_active, stall)[0]) for _ in range(STALL_HOLD_STEPS + 5)]
driven = sum(trig[STALL_STEPS - 1 :]) + sum(kept)
check(
    f"固まりから起こしたら {STALL_HOLD_STEPS} ステップは運転し続ける",
    driven == STALL_HOLD_STEPS and not kept[-1],
    f"続いたステップ {driven}",
)
c = fresh()
stall.held[:] = True
check("赤信号などに止められているなら固まりとみなさない", not any(bool(c.decide(late, all_active, stall)[0]) for _ in range(STALL_STEPS * 2)))

# ---------------------------------------------------------------------------
section("3. 確率での割り込みは区間ごとに決める")
c = OnlineAssistController(N, seed=3)
half = ASSIST_WARMUP_STEPS // 2
p_half = c.get_assist_probability(half)
history = np.array([c.decide(half, all_active, AssistDanger.safe(N)) for _ in range(4000)])
rate = float(history.mean())
check(f"割り込んだ割合が確率 {p_half:.3f} に近い", abs(rate - p_half) < 0.05, f"実際 {rate:.3f}")
switches = np.flatnonzero(np.any(history[1:] != history[:-1], axis=1)) + 1
check(
    f"運転者が入れ替わるのは {ASSIST_SEGMENT_STEPS} ステップの区切りだけ",
    bool(np.all(switches % ASSIST_SEGMENT_STEPS == 0)),
    f"入れ替わり {switches.size} 回",
)

# ---------------------------------------------------------------------------
section("4. バッファ")
rng = np.random.default_rng(0)


def fill(buffer: RolloutBuffer, assisted_prob: float, seed: int) -> None:
    r = np.random.default_rng(seed)
    for _ in range(buffer.capacity):
        assisted = r.random(N) < assisted_prob if assisted_prob > 0 else None
        buffer.add(
            obs=r.standard_normal((N, config.OBS_DIM)).astype(np.float32),
            actions=r.uniform(-1, 1, (N, A)).astype(np.float32),
            log_probs=r.standard_normal(N).astype(np.float32),
            values=r.standard_normal(N).astype(np.float32),
            rewards=r.standard_normal(N).astype(np.float32),
            dones=r.random(N) < 0.02,
            active=np.ones(N, dtype=bool),
            expert_actions=None if assisted is None else r.uniform(-1, 1, (N, A)).astype(np.float32),
            assisted=assisted,
        )
    buffer.compute_returns_and_advantages(np.zeros(N), np.ones(N, dtype=bool), 0.99, 0.95)


b = RolloutBuffer(64, N, config.OBS_DIM, A)
fill(b, 0.4, seed=5)
data = b.flat_dataset()
assert data is not None
assisted = data["assisted"].numpy()
adv = data["advantages"].numpy()
check("expert_actions の形は [容量, 台数, 行動次元]", b.expert_actions.shape == (64, N, A))
check("assisted の形は [容量, 台数]", b.assisted.shape == (64, N))
check("アシストされたステップも学習データに残る（価値と模倣に使う）", 0 < int(assisted.sum()) < assisted.size)
check(
    "正規化はエキスパートが運転していないステップだけで行う",
    abs(float(adv[~assisted].mean())) < 1e-5 and abs(float(adv[~assisted].std()) - 1.0) < 1e-3,
)
check("アシストされたステップの GAE も計算される（価値の目標になる）", bool(np.any(b.advantages[:64][b.assisted[:64]] != 0.0)))

plain = RolloutBuffer(64, N, config.OBS_DIM, A)
fill(plain, 0.0, seed=6)
pd = plain.flat_dataset()
assert pd is not None
raw = plain.advantages[:64].reshape(-1)
old = (raw - raw.mean()) / (raw.std() + 1e-8)
check("アシストが無ければ正規化は以前と 1 ビットも変わらない", np.array_equal(pd["advantages"].numpy(), old.astype(np.float32)))
check("アシストが無ければ assisted はすべて偽", not bool(pd["assisted"].any()))

# ---------------------------------------------------------------------------
section("5. PPO の更新（補助の模倣の損失）")
params = SimParams()
params.rollout_length = 64


def run_update(
    trainer: PPOTrainer,
    assisted_prob: float,
    seed: int,
    expert: np.ndarray | None = None,
    *,
    reward_scale: float = 1.0,
    explicit_none: bool = False,
) -> dict[str, float]:
    r = np.random.default_rng(seed)
    torch.manual_seed(seed)
    obs = r.standard_normal((N, config.OBS_DIM)).astype(np.float32)
    active = np.ones(N, dtype=bool)
    stats = None
    while stats is None:
        actions, log_probs, values = trainer.act(obs, active)
        assisted = (r.random(N) < assisted_prob) if assisted_prob > 0 else None
        target = None
        if explicit_none:
            assisted = np.zeros(N, dtype=bool)
            target = np.zeros((N, A), dtype=np.float32)
        elif assisted is not None:
            target = expert if expert is not None else r.uniform(-1, 1, (N, A)).astype(np.float32)
        trainer.store(
            obs=obs,
            actions=actions,
            log_probs=log_probs,
            values=values,
            rewards=(r.standard_normal(N) * reward_scale).astype(np.float32),
            dones=np.zeros(N, dtype=bool),
            active=active,
            expert_actions=target,
            assisted=assisted,
        )
        obs = r.standard_normal((N, config.OBS_DIM)).astype(np.float32)
        stats = trainer.maybe_update(obs, active)
    return stats


t = PPOTrainer(config.OBS_DIM, A, params, N, seed=0)
stats = run_update(t, 0.5, seed=7)
finite = all(math.isfinite(v) for v in stats.values())
weights_ok = all(bool(torch.isfinite(p).all()) for p in t.policy.parameters())
check("アシスト 50% で更新しても NaN・無限大が出ない", finite and weights_ok, f"{ {k: round(v, 4) for k, v in stats.items()} }")
check("模倣の損失が統計に出る（正の有限値）", stats["bc_loss"] > 0.0 and math.isfinite(stats["bc_loss"]))

t_all = PPOTrainer(config.OBS_DIM, A, params, N, seed=0)
stats_all = run_update(t_all, 1.0, seed=8)
check("すべてアシストなら方策の代理損失は 0（勾配に入れない）", stats_all["policy_loss"] == 0.0, f"{stats_all['policy_loss']}")
check("すべてアシストでも価値の損失は学習される", stats_all["value_loss"] > 0.0)

target = np.tile(np.array([[0.6, -0.3]], dtype=np.float32), (N, 1))
t_bc = PPOTrainer(config.OBS_DIM, A, params, N, seed=0)
probe = torch.from_numpy(np.random.default_rng(9).standard_normal((256, config.OBS_DIM)).astype(np.float32))
with torch.no_grad():
    before = float(((t_bc.policy.mean_action(probe) - torch.from_numpy(target[:1])) ** 2).sum(-1).mean())
for k in range(20):
    run_update(t_bc, 1.0, seed=20 + k, expert=target, reward_scale=0.0)
with torch.no_grad():
    after = float(((t_bc.policy.mean_action(probe) - torch.from_numpy(target[:1])) ** 2).sum(-1).mean())
check("模倣の損失で方策の平均が教師の操作へ近づく（20 更新で半分未満）", after < before * 0.5, f"二乗誤差 {before:.3f} → {after:.3f}")

t_old = PPOTrainer(config.OBS_DIM, A, params, N, seed=0)
t_new = PPOTrainer(config.OBS_DIM, A, params, N, seed=0)
s_old = run_update(t_old, 0.0, seed=11)
s_new = run_update(t_new, 0.0, seed=11, explicit_none=True)
same = all(torch.equal(a, b) for a, b in zip(t_old.policy.parameters(), t_new.policy.parameters()))
check("assisted を渡さないのと「すべて偽」を渡すので、更新の結果が完全に一致する", same and s_old == s_new)
check("アシストが無ければ bc_loss は 0", s_old["bc_loss"] == 0.0)
check(
    "experience_steps は更新回数 × ロールアウト長 + 収集中の分",
    t_old.experience_steps == t_old.updates * t_old.rollout_length + t_old.buffer.size,
    f"{t_old.experience_steps}",
)

# ---------------------------------------------------------------------------
section("6. 環境（合成の碁盤の目・8 台）")
env_a = make_env(seed=0)
env_b = make_env(seed=0)
zeros = np.zeros(N, dtype=bool)
act_rng = np.random.default_rng(12)
identical = True
for _ in range(120):
    a = act_rng.uniform(-1, 1, (N, A)).astype(np.float32)
    ra = env_a.step(a)
    rb = env_b.step(a, expert=zeros)
    identical &= np.array_equal(ra.obs, rb.obs) and np.array_equal(ra.rewards, rb.rewards)
check("expert が全部偽なら、渡さないときと観測・報酬が完全に一致する", identical)
check("expert を渡さなければ assisted はすべて偽", not bool(ra.assisted.any()))


def travel(env: SimulationEnv, steps: int, accel: float, expert_fn) -> tuple[float, int, int]:
    moved = 0.0
    assisted_steps = 0
    crashes = 0
    for _ in range(steps):
        a = np.zeros((N, A), dtype=np.float32)
        a[:, 0] = accel
        before = np.stack([env.world.fleet.x, env.world.fleet.y], axis=1).astype(np.float64)
        active = env.active_mask
        result = env.step(a, expert=expert_fn(env, active))
        after = np.stack([env.world.fleet.x, env.world.fleet.y], axis=1).astype(np.float64)
        step_moved = np.hypot(*(after - before).T)
        moved += float(np.sum(np.where(result.active & env.active_mask & (step_moved < 5.0), step_moved, 0.0)))
        assisted_steps += int(result.assisted.sum())
        crashes += sum(1 for e in result.episodes if e.reason in ("collision", "offroad"))
    return moved / N, assisted_steps, crashes


frozen, _, _ = travel(make_env(seed=1), 400, -1.0, lambda env, active: None)
check("フルブレーキに固まった方策だけでは動かない（前提の確認）", frozen < 1.0, f"{frozen:.1f} m/台")

env_e = make_env(seed=1)
expert_move, expert_steps, expert_crash = travel(env_e, 400, -1.0, lambda env, active: active.copy())
check("全台エキスパートなら固まった方策の代わりに走る", expert_move > 60.0, f"{expert_move:.1f} m/台 / 20 秒")
check("エキスパートが運転したステップが assisted に出る", expert_steps > 0.9 * 400 * N, f"{expert_steps} / {400 * N}")
check("エキスパートの運転で衝突・逸脱しない", expert_crash == 0, f"{expert_crash} 回")

env_c = make_env(seed=2)
ctl_env = OnlineAssistController(N, seed=0)
danger_times: list[float] = []


def controlled(env: SimulationEnv, active: np.ndarray) -> np.ndarray:
    started = time.perf_counter()
    mask = ctl_env.decide(late, active, env.assist_danger(active))
    danger_times.append(time.perf_counter() - started)
    return mask


stall_move, stall_steps, stall_crash = travel(env_c, 600, -1.0, controlled)
check(
    "確率が下限でも、固まった方策は固まりの判定で起こされて走る",
    stall_move > 20.0 and stall_steps > 0,
    f"{stall_move:.1f} m/台 / 30 秒・割り込み {stall_steps} ステップ",
)
median_ms = float(np.median(danger_times)) * 1000.0
check("危険の判定と割り込みの決定は 1 ステップ 2ms 未満（8 台）", median_ms < 2.0, f"中央値 {median_ms:.3f} ms")

env_d = make_env(seed=3)
for _ in range(60):
    r = env_d.step(np.zeros((N, A), dtype=np.float32), expert=env_d.active_mask.copy())
ea = r.expert_actions[r.assisted]
check("エキスパートの操作は -1..1 に収まる", bool(np.all(np.abs(ea) <= 1.0)) and ea.shape[0] > 0)
check("assisted は learn の部分集合", bool(np.all(~r.assisted | r.learn)))
d = env_d.assist_danger(env_d.active_mask)
check(
    "assist_danger の値が有限（TTC と赤信号の距離は inf 可）",
    bool(np.all(np.isfinite(d.lane_offset)) and np.all(np.isfinite(d.speed)) and not np.any(np.isnan(d.ttc))),
)

print()
if FAILURES:
    print(f"NG: {len(FAILURES)} 件")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("OK: すべての検査に通りました")
