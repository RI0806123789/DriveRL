# -*- coding: utf-8 -*-
"""階層型の方策（上位: 意図の選択 / 下位: 連続値の操作）を検証する。マップのキャッシュは読まない（合成の碁盤の目で走らせる）。"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for st in (sys.stdout, sys.stderr):
    try:
        st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import torch

from app import config
from app.contracts import Bounds, DriveState, MapData, MapEdge, MapNode, SimParams
from app.map.index import build_map_index
from app.rl.buffer import RolloutBuffer
from app.rl.export import InferencePolicy, _build_keras_model
from app.rl.hierarchical_policy import (
    NUM_OPTIONS,
    OPTION_CRUISE,
    OPTION_FOLLOW,
    OPTION_STOP,
    OPTION_YIELD,
    HierarchicalActorCritic,
    OptionScheduler,
    sub_reward_shaping,
)
from app.rl.importer import inspect_checkpoint
from app.rl.ppo import PPOTrainer
from app.sim.env import SimulationEnv
from app.sim.signals import GREEN, RED

FAILURES: list[str] = []
N = config.MAX_VEHICLES
A = config.ACTION_DIM
D = config.OBS_DIM
PERIOD = config.HRL_OPTION_STEPS


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


def drive(speed: float = 0.0, lateral: float = 0.0, lead: float = math.nan, delta_sq: float = 0.0) -> DriveState:
    return DriveState(
        speed=np.full(N, speed),
        lateral=np.full(N, lateral),
        lead_speed=np.full(N, lead),
        action_delta_sq=np.full(N, delta_sq),
        jerk=np.zeros(N),
    )


torch.manual_seed(0)
params = SimParams()
all_active = np.ones(N, dtype=bool)

# ---------------------------------------------------------------------------
section("1. 上位方策が 4 つの意図を正しい確率分布で出す")
policy = HierarchicalActorCritic(D, A)
x = torch.randn(256, D)
dist, meta_values = policy.meta(x)
probs = dist.probs
check("意図は 4 つ（CRUISE / FOLLOW / YIELD / STOP）", NUM_OPTIONS == 4 and config.HRL_OPTIONS == ("CRUISE", "FOLLOW", "YIELD", "STOP"))
check("確率は各行で和が 1・すべて正", bool(torch.allclose(probs.sum(-1), torch.ones(256))) and bool((probs > 0).all()))
check("上位の価値は (B,)", tuple(meta_values.shape) == (256,))
opts, lp, _v = policy.select_option(x)
check("select_option は 0〜3 を返し、対数確率は分布と一致", bool(((opts >= 0) & (opts < 4)).all()) and bool(torch.allclose(lp, dist.log_prob(opts))))
seen = torch.bincount(policy.select_option(torch.randn(4000, D))[0], minlength=4)
check("初期の上位方策はどの意図も引く（偏りすぎない）", bool((seen > 600).all()), f"{seen.tolist()}")
check("下位の入力は観測 + one-hot = 83 次元", policy.sub_in_dim == D + 4 and policy.policy_trunk[0].in_features == D + 4)

# ---------------------------------------------------------------------------
section("2. 同じ観測でも意図を変えると下位方策の操作が変わる")
obs = torch.randn(64, D)
with torch.no_grad():
    accel = {o: policy.mean_action(obs, torch.full((64,), o)).mean(0)[0].item() for o in range(4)}
check("STOP のときアクセルの平均は負", accel[OPTION_STOP] < 0.0, f"{accel[OPTION_STOP]:+.3f}")
check("CRUISE のときアクセルの平均は正", accel[OPTION_CRUISE] > 0.0, f"{accel[OPTION_CRUISE]:+.3f}")
check(
    "CRUISE > FOLLOW > YIELD > STOP の順",
    accel[OPTION_CRUISE] > accel[OPTION_FOLLOW] > accel[OPTION_YIELD] > accel[OPTION_STOP],
    " / ".join(f"{config.HRL_OPTIONS[o]} {accel[o]:+.3f}" for o in range(4)),
)
# 模倣で意図ごとの操作を教えると、意図の one-hot から操作を引き分ける
trained = HierarchicalActorCritic(D, A)
with torch.no_grad():
    trained.option_bias.zero_()
opt = torch.optim.Adam(trained.parameters(), lr=3e-3)
target_accel = torch.tensor([0.6, 0.2, -0.3, -0.8])
for _ in range(300):
    ob = torch.randn(128, D)
    o = torch.randint(0, 4, (128,))
    loss = ((trained.mean_action(ob, o)[:, 0] - target_accel[o]) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
with torch.no_grad():
    learned = [trained.mean_action(obs, torch.full((64,), o)).mean(0)[0].item() for o in range(4)]
check(
    "偏りを 0 にしても、意図ごとの操作を one-hot から学べる（STOP は負・CRUISE は正）",
    learned[OPTION_STOP] < -0.5 and learned[OPTION_CRUISE] > 0.4,
    " / ".join(f"{v:+.2f}" for v in learned),
)
with torch.no_grad():
    greedy = policy.greedy_options(obs)
    same = torch.allclose(policy.mean_action(obs), policy.mean_action(obs, greedy))
check("意図を省くと上位方策のいちばん確率の高い意図を使う", bool(same))

# ---------------------------------------------------------------------------
section("3. 意図は 20 ステップ保たれ、下位方策だけが毎ステップ動く")
sched = OptionScheduler(N, PERIOD)
check("最初のステップは全員選ぶ", bool(sched.due(all_active).all()))
trainer = PPOTrainer(D, A, SimParams(rollout_length=200), N, seed=0)
rng = np.random.default_rng(0)
history_opts = []
history_start = []
done_at = {3: 45}
for t in range(120):
    o = rng.standard_normal((N, D)).astype(np.float32)
    act, lp, val = trainer.act(o, all_active)
    history_opts.append(trainer.current_options)
    dones = np.zeros(N, dtype=bool)
    for slot, step in done_at.items():
        if t == step:
            dones[slot] = True
    trainer.store(o, act, lp, val, np.zeros(N, np.float32), dones, all_active)
    history_start.append(trainer.buffer.option_start[t].copy())
starts = np.array(history_start)
opts_hist = np.array(history_opts)
regular = [s for s in range(N) if s not in done_at]
intervals = {int(v) for s in regular for v in np.diff(np.flatnonzero(starts[:, s]))}
check(f"選び直しの間隔は {PERIOD} ステップちょうど", intervals == {PERIOD}, f"{sorted(intervals)}")
changed = (np.diff(opts_hist, axis=0) != 0)
check("意図が変わるのは選び直したステップだけ", not bool((changed & ~starts[1:]).any()))
check("エピソードが終わった車は次のステップで選び直す", bool(starts[done_at[3] + 1, 3]))
restart_iv = np.diff(np.flatnonzero(starts[done_at[3] + 1 :, 3]))
check("選び直した後はそこから 20 ステップ保つ", bool(restart_iv.size > 0 and np.all(restart_iv == PERIOD)), f"{restart_iv.tolist()}")
check("バッファに意図と選んだ印が積まれる", bool(np.array_equal(trainer.buffer.options[:120], opts_hist)))
sub_changes = np.abs(np.diff(trainer.buffer.actions[:120], axis=0)).sum()
check("下位方策の操作は毎ステップ引く（同じ意図の間も動く）", float(sub_changes) > 0.0)
inactive = all_active.copy()
inactive[5] = False
trainer.act(rng.standard_normal((N, D)).astype(np.float32), inactive)
trainer._last_meta = None
revived = trainer.scheduler.due(all_active)
check("止まっていた車が走り出したら選び直す", bool(revived[5]) and int(revived.sum()) == 1)

# ---------------------------------------------------------------------------
section("4. 上位の GAE（意図 1 つを 1 手として数える）")
gamma, lam = 0.9, 0.8
buf = RolloutBuffer(6, 1, 2, A)
rewards = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
starts_small = [True, False, False, True, False, False]
meta_v = [10.0, 0.0, 0.0, 20.0, 0.0, 0.0]
for t in range(6):
    buf.add(
        np.zeros((1, 2)), np.zeros((1, A)), np.zeros(1), np.zeros(1), np.array([rewards[t]]),
        np.array([False]), np.array([True]), option_start=np.array([starts_small[t]]),
        meta_values=np.array([meta_v[t]]), sub_rewards=np.array([0.5]),
    )
buf.compute_returns_and_advantages(np.zeros(1), np.array([True]), gamma, lam, last_meta_values=np.array([30.0]))
g2 = 4 + gamma * 5 + gamma**2 * 6 + gamma**3 * 30.0
a2 = g2 - 20.0
g1 = 1 + gamma * 2 + gamma**2 * 3 + gamma**3 * 20.0
a1 = g1 - 10.0 + lam * gamma**3 * a2
got = buf.meta_advantages[:, 0]
check("2 つめの意図の advantage は区間の割引報酬 + γ^3 V − V", math.isclose(float(got[3]), a2, rel_tol=1e-5), f"{got[3]:.4f} / {a2:.4f}")
check("1 つめは次の意図の advantage を γ^3 λ で連ねる", math.isclose(float(got[0]), a1, rel_tol=1e-5), f"{got[0]:.4f} / {a1:.4f}")
check("意図を選んでいないステップの上位の advantage は 0", bool(np.all(got[[1, 2, 4, 5]] == 0.0)))
check("下位の GAE は整形を足した報酬（sub_rewards）で数える", math.isclose(float(buf.returns[5, 0]), 0.5, rel_tol=1e-6))
buf_done = RolloutBuffer(4, 1, 2, A)
for t in range(4):
    buf_done.add(
        np.zeros((1, 2)), np.zeros((1, A)), np.zeros(1), np.zeros(1), np.array([1.0]),
        np.array([t == 1]), np.array([True]), option_start=np.array([t in (0, 2)]),
        meta_values=np.array([5.0]),
    )
buf_done.compute_returns_and_advantages(np.zeros(1), np.array([True]), gamma, lam, last_meta_values=np.array([5.0]))
check(
    "エピソードの終わりで区間を切り、次のエピソードの価値を借りない",
    math.isclose(float(buf_done.meta_advantages[0, 0]), 1 + gamma * 1 - 5.0, rel_tol=1e-6),
    f"{buf_done.meta_advantages[0, 0]:.4f}",
)

# ---------------------------------------------------------------------------
section("5. PPO の更新で上位・下位の両方に勾配が流れる")
upd = PPOTrainer(D, A, SimParams(rollout_length=64), N, seed=1)
before = {k: v.detach().clone() for k, v in upd.policy.state_dict().items()}
grads: dict[str, float] = {}
orig_step = upd.optimizer.step


def spy_step(*args: object, **kwargs: object) -> object:
    for name, prm in upd.policy.named_parameters():
        if prm.grad is not None:
            grads[name] = grads.get(name, 0.0) + float(prm.grad.norm())
    return orig_step(*args, **kwargs)


upd.optimizer.step = spy_step  # type: ignore[method-assign]
stats = None
for t in range(64):
    o = rng.standard_normal((N, D)).astype(np.float32)
    act, lp, val = upd.act(o, all_active)
    assisted = np.zeros(N, dtype=bool)
    assisted[:2] = t % 3 == 0
    upd.store(
        o, act, lp, val, rng.standard_normal(N).astype(np.float32), np.zeros(N, bool), all_active,
        expert_actions=np.tile([0.5, 0.0], (N, 1)).astype(np.float32), assisted=assisted,
        expert_options=np.where(assisted, OPTION_FOLLOW, -1), drive=drive(speed=4.0, delta_sq=0.1),
    )
while stats is None:
    stats = upd.maybe_update(np.zeros((N, D), np.float32), all_active)
finite = all(math.isfinite(float(v)) for v in stats.values())
check("更新の統計がすべて有限", finite, ", ".join(f"{k} {v:.3g}" for k, v in stats.items()))
for group in ("meta_trunk", "meta_head", "meta_value_trunk", "meta_value_head", "policy_trunk", "mu_head", "value_trunk", "value_head", "option_bias", "log_std"):
    g = sum(v for k, v in grads.items() if k.startswith(group))
    moved = sum(float((upd.policy.state_dict()[k] - before[k]).abs().sum()) for k in before if k.startswith(group))
    check(f"{group} に勾配が流れて重みが動く", g > 0.0 and math.isfinite(g) and moved > 0.0, f"勾配 {g:.3g}")
check("上位のエントロピーは ln 4 付近（初期）", abs(stats["meta_entropy"] - math.log(4)) < 0.05, f"{stats['meta_entropy']:.4f}")

# 意図の模倣: エキスパートが運転したステップの状況の意図（ラベル）を上位方策が学ぶ
bc = PPOTrainer(D, A, SimParams(rollout_length=64), N, seed=2)
probe = torch.from_numpy(rng.standard_normal((32, D)).astype(np.float32))
with torch.no_grad():
    p_before = float(bc.policy.meta(probe)[0].probs[:, OPTION_YIELD].mean())
for _update in range(12):
    for t in range(64):
        o = rng.standard_normal((N, D)).astype(np.float32)
        act, lp, val = bc.act(o, all_active)
        bc.store(
            o, act, lp, val, np.zeros(N, np.float32), np.zeros(N, bool), all_active,
            expert_actions=np.zeros((N, A), np.float32), assisted=all_active,
            expert_options=np.full(N, OPTION_YIELD),
        )
    while bc.maybe_update(np.zeros((N, D), np.float32), all_active) is None:
        pass
with torch.no_grad():
    p_after = float(bc.policy.meta(probe)[0].probs[:, OPTION_YIELD].mean())
check("エキスパートの状況の意図を教えると、その意図の確率が上がる", p_after > p_before + 0.05, f"YIELD {p_before:.3f} → {p_after:.3f}")
data = None
b2 = RolloutBuffer(4, 2, D, A)
for t in range(4):
    b2.add(
        np.zeros((2, D)), np.zeros((2, A)), np.zeros(2), np.zeros(2), np.array([1.0, -1.0]) * t,
        np.zeros(2, bool), np.ones(2, bool), option_start=np.array([True, True]),
        assisted=np.array([True, False]), expert_actions=np.zeros((2, A)), expert_options=np.array([2, 2]),
    )
b2.compute_returns_and_advantages(np.zeros(2), np.ones(2, bool), 0.9, 0.9, last_meta_values=np.zeros(2))
data = b2.flat_dataset()
assert data is not None
own_meta = data["option_start"] & ~data["assisted"]
check("エキスパートが運転したステップで選んだ意図は、上位の方策の勾配に入れない", bool((data["meta_advantages"][data["assisted"]] == 0).all()) and bool(own_meta.any()))
check("意図の教師はエキスパートが運転したステップだけ", bool((data["expert_options"][~data["assisted"]] == -1).all()))

# ---------------------------------------------------------------------------
section("6. 下位方策の報酬の整形（加加速度・車線維持・意図の速度帯）")
cruise = np.full(N, OPTION_CRUISE)
smooth = sub_reward_shaping(cruise, drive(speed=8.0, delta_sq=0.01), all_active)
jerky = sub_reward_shaping(cruise, drive(speed=8.0, delta_sq=4.0), all_active)
check("操作が急に変わるほど罰が大きい", float(jerky[0]) < float(smooth[0]) < 0.0, f"{smooth[0]:.4f} / {jerky[0]:.4f}")
check("罰は係数 × 操作の変化の二乗", math.isclose(float(smooth[0] - jerky[0]), config.HRL_JERK_COEF * 3.99, rel_tol=1e-4))
off = sub_reward_shaping(cruise, drive(speed=8.0, lateral=2.0), all_active)
check("車線中心から離れるほど罰が大きい（二乗）", math.isclose(float(off[0]), -config.HRL_LANE_COEF * 4.0, rel_tol=1e-5))
stop = np.full(N, OPTION_STOP)
check("STOP で走っていると罰・止まっていれば罰なし", float(sub_reward_shaping(stop, drive(speed=6.0), all_active)[0]) < 0.0 and float(sub_reward_shaping(stop, drive(speed=0.2), all_active)[0]) == 0.0)
check("CRUISE で止まっていると罰", float(sub_reward_shaping(cruise, drive(speed=0.0), all_active)[0]) < 0.0)
yld = np.full(N, OPTION_YIELD)
check("YIELD は 10km/h 以下なら罰なし", float(sub_reward_shaping(yld, drive(speed=2.5), all_active)[0]) == 0.0 and float(sub_reward_shaping(yld, drive(speed=5.0), all_active)[0]) < 0.0)
follow = np.full(N, OPTION_FOLLOW)
check("FOLLOW は前走車の速さに合わせれば罰なし", float(sub_reward_shaping(follow, drive(speed=6.0, lead=6.0), all_active)[0]) == 0.0 and float(sub_reward_shaping(follow, drive(speed=10.0, lead=6.0), all_active)[0]) < 0.0)
worst = sub_reward_shaping(stop, drive(speed=13.9), all_active)
check("意図の外れの罰は頭打ち", math.isclose(float(worst[0]), -config.HRL_CONSISTENCY_COEF, rel_tol=1e-5))
check("走っていない車は 0", float(sub_reward_shaping(stop, drive(speed=13.9, delta_sq=3.0), np.zeros(N, bool)).sum()) == 0.0)

# ---------------------------------------------------------------------------
section("7. 階層型にする前（平らな方策）の重みの読み込み")
flat_src = PPOTrainer(D, A, params, N, seed=5)
flat_state = {}
for key, value in flat_src.policy.state_dict().items():
    if key.startswith(("meta_", "option_bias")):
        continue
    if key in ("policy_trunk.0.weight", "value_trunk.0.weight"):
        value = value[:, :D]
    flat_state[key] = value.clone()
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "flat.pt"
    torch.save(
        {"format": "autoware-sim-ppo-1", "obs_dim": D, "action_dim": A, "hidden_sizes": list(flat_src.hidden_sizes), "updates": 4321, "policy": flat_state},
        path,
    )
    info = inspect_checkpoint(path, expected_obs_dim=D, expected_action_dim=A, expected_hidden_sizes=flat_src.hidden_sizes)
    up = PPOTrainer(D, A, params, N, seed=6)
    loaded = up.load(path)
xf = torch.randn(64, D)
with torch.no_grad():
    flat_mu = torch.tanh(
        torch.nn.functional.linear(
            torch.tanh(torch.nn.functional.linear(torch.tanh(torch.nn.functional.linear(xf, flat_state["policy_trunk.0.weight"], flat_state["policy_trunk.0.bias"])), flat_state["policy_trunk.2.weight"], flat_state["policy_trunk.2.bias"])),
            flat_state["mu_head.weight"], flat_state["mu_head.bias"],
        )
    )
    diffs = [float((up.policy.mean_action(xf, torch.full((64,), o)) - flat_mu).abs().max()) for o in range(4)]
check("平らな方策のチェックポイントを読み込める（更新回数も引き継ぐ）", loaded and up.upgraded_flat and up.updates == 4321)
# 意図の入力の重みは 0 だが、行列積の足し算の順序が入力の幅で変わるので float32 の丸めの差は残る
check("読み込んだ直後は、どの意図でも元の操作と同じ（意図の入力と偏りは 0。丸めの差 1e-6 以内）", max(diffs) <= 1e-6, f"最大差 {max(diffs):.2e}")
check("書き出しからの読み込み（importer）も受け付ける", info.obs_dim == D)
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "hier.pt"
    up.save(path)
    again = PPOTrainer(D, A, params, N, seed=7)
    re_loaded = again.load(path)
    same_state = all(torch.equal(again.policy.state_dict()[k], v) for k, v in up.policy.state_dict().items())
check("保存し直すと階層型のまま読める（移し替えは 1 度だけ）", re_loaded and not again.upgraded_flat and same_state)

# ---------------------------------------------------------------------------
section("8. 書き出したモデルは方策と同じ操作を出す")
xe = torch.randn(128, D)
with torch.no_grad():
    o_ref = trainer.policy.greedy_options(xe)
    a_ref = trainer.policy.mean_action(xe, o_ref)
    v_ref = trainer.policy.sub_value(xe, o_ref)
scripted = torch.jit.script(InferencePolicy(trainer.policy))
a_ts, v_ts = scripted(xe)
check("TorchScript の操作と価値が方策と一致", float((a_ts - a_ref).abs().max()) == 0.0 and float((v_ts - v_ref).abs().max()) == 0.0)
def _keras_gap(policy_trainer, obs: torch.Tensor) -> tuple[float, float]:
    """Keras へ書き出した操作・価値と方策の差（そのままと、保存して読み直した後）。"""
    import keras  # noqa: PLC0415

    with torch.no_grad():
        o = policy_trainer.policy.greedy_options(obs)
        a = policy_trainer.policy.mean_action(obs, o).numpy()
        v = policy_trainer.policy.sub_value(obs, o).numpy()
    model = _build_keras_model(policy_trainer)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "policy.keras"
        model.save(path)
        loaded = keras.saving.load_model(path)
        gaps = []
        for m in (model, loaded):
            a_k, v_k = m.predict(obs.numpy(), verbose=0)
            gaps.append(max(float(np.abs(a_k - a).max()), float(np.abs(v_k - v).max())))
    return gaps[0], gaps[1]


try:
    diff_k, diff_loaded = _keras_gap(trainer, xe)
    check("Keras の操作と価値が方策と一致（float32 の丸めの差だけ）", diff_k < 1e-5, f"最大差 {diff_k:.2e}")
    check("保存して読み直した Keras も一致", diff_loaded < 1e-5, f"最大差 {diff_loaded:.2e}")
    # 意図が同点・近接のとき。torch の argmax は同点なら番号の小さい意図を選ぶ（Keras も同じ規則で選ぶこと）
    fresh = PPOTrainer(D, A, params, N, seed=0)
    zero_gap, _ = _keras_gap(fresh, torch.zeros(4, D))
    check("初期の方策・ゼロ観測（上位の出力層が 0 なので 4 つの意図が同点）でも一致", zero_gap < 1e-5, f"最大差 {zero_gap:.2e}")
    for label, logits in (
        ("完全な同点 [1, 1, 1, 1]", [1.0, 1.0, 1.0, 1.0]),
        ("後ろ 2 つが同点 [0, 2, 2, -1]", [0.0, 2.0, 2.0, -1.0]),
        ("近接 [3, 3 - 1e-6, 3, 2]", [3.0, 3.0 - 1e-6, 3.0, 2.0]),
        ("近接 [0.5, 0.5 + 1e-4, 0.5, 0.5]", [0.5, 0.5 + 1e-4, 0.5, 0.5]),
    ):
        tie = PPOTrainer(D, A, params, N, seed=0)
        with torch.no_grad():
            tie.policy.meta_head.weight.zero_()
            tie.policy.meta_head.bias.copy_(torch.tensor(logits))
        gap, gap_loaded = _keras_gap(tie, torch.randn(32, D))
        check(f"意図が{label}でも一致（保存して読み直した後も）", max(gap, gap_loaded) < 1e-5, f"最大差 {max(gap, gap_loaded):.2e}")
except Exception as exc:  # noqa: BLE001
    check("Keras の操作と価値が方策と一致", False, f"{type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------
section("9. 環境: 意図の教師・走りの真値・frame の currentOption")
env = SimulationEnv(build_map_index(grid_map()), SimParams(vehicle_count=N, pedestrian_count=0), seed=0)
env.reset_all()
world = env.world
orig_signal, orig_hold, orig_ped, orig_junc, orig_lead = world.next_signal, env.traffic_hold, env._pedestrian_gap, world.next_junction, env._lead_gap


def label_with(signal=(math.inf, GREEN), hold=False, ped=math.inf, junc=math.inf, lead=math.inf) -> int:
    world.next_signal = lambda s: signal  # type: ignore[method-assign]
    env.traffic_hold = lambda s: hold  # type: ignore[method-assign]
    env._pedestrian_gap = lambda s: ped  # type: ignore[method-assign]
    world.next_junction = lambda s: junc  # type: ignore[method-assign]
    env._lead_gap = lambda s: lead  # type: ignore[method-assign]
    mask = np.zeros(N, dtype=bool)
    mask[0] = True
    return int(env.option_labels(mask)[0])


cases = [
    ("何も無ければ CRUISE", {}, OPTION_CRUISE),
    ("前走車が 20m 先なら FOLLOW", {"lead": 20.0}, OPTION_FOLLOW),
    ("信号の無い交差点の 10m 手前なら YIELD", {"junc": 10.0, "lead": 20.0}, OPTION_YIELD),
    ("歩行者が 15m 先なら YIELD", {"ped": 15.0}, OPTION_YIELD),
    ("赤信号の 25m 手前なら STOP（前走車・交差点より優先）", {"signal": (25.0, RED), "lead": 20.0, "junc": 10.0}, OPTION_STOP),
    ("止められていれば STOP", {"hold": True}, OPTION_STOP),
    ("青信号なら STOP にしない", {"signal": (5.0, GREEN)}, OPTION_CRUISE),
]
for label, kw, expected in cases:
    got_label = label_with(**kw)
    check(label, got_label == expected, f"{config.HRL_OPTIONS[got_label] if got_label >= 0 else got_label}")
world.next_signal, env.traffic_hold, env._pedestrian_gap, world.next_junction, env._lead_gap = orig_signal, orig_hold, orig_ped, orig_junc, orig_lead
check("slots に入っていない車は -1", bool(np.all(env.option_labels(np.zeros(N, bool)) == -1)))

env.reset_all()
first = env.step(np.tile([0.5, 0.0], (N, 1)).astype(np.float32))
check("エピソードの最初のステップは操作の変化 0", first.drive is not None and float(first.drive.action_delta_sq.sum()) == 0.0)
second = env.step(np.tile([-0.5, 0.2], (N, 1)).astype(np.float32))
alive = second.active & first.active
check("操作を変えたら変化の二乗が入る", bool(np.all(second.drive.action_delta_sq[alive] > 0.5)), f"{second.drive.action_delta_sq[alive][:3]}")
check("加加速度も入る（加速から減速へ）", bool(np.all(second.drive.jerk[alive] < 0.0)))
expert = np.zeros(N, dtype=bool)
expert[:3] = True
res = env.step(np.zeros((N, A), np.float32), expert=expert)
labels = res.expert_options
check("エキスパートが運転した車にだけ意図の教師が付く", labels is not None and bool(np.all(labels[res.assisted] >= 0)) and bool(np.all(labels[~res.assisted] == -1)))

env.current_options = np.full(N, OPTION_YIELD)
frame = env.snapshot(0, 0.0)
wire = [v.to_wire() for v in frame.vehicles if v.active]
check("frame の走っている車に currentOption が載る", bool(wire) and all(w.get("currentOption") == "YIELD" for w in wire))
env.current_options = None
check("意図を渡していなければ currentOption を省く", all("currentOption" not in v.to_wire() for v in env.snapshot(0, 0.0).vehicles))
env.current_options = np.full(N, OPTION_STOP)
env.autopilot_all = True
check("実用モード（全車が経路追従）では currentOption を省く", all("currentOption" not in v.to_wire() for v in env.snapshot(0, 0.0).vehicles))
env.autopilot_all = False

# ---------------------------------------------------------------------------
section("10. 環境と学習器をつないで回す（エキスパートの割り込みあり）")
loop = PPOTrainer(D, A, SimParams(rollout_length=128), N, seed=3)
env.reset_all()
times = []
updates = 0
for t in range(400):
    o = env.observations
    active = env.active_mask
    started = time.perf_counter()
    act, lp, val = loop.act(o, active)
    times.append(time.perf_counter() - started)
    ex = active & (np.arange(N) % 2 == 0) & ((t // 40) % 2 == 0)
    r = env.step(act, expert=ex)
    loop.store(
        o, act, lp, val, r.rewards, r.dones, r.active, r.truncated, r.final_obs, r.learn,
        r.expert_actions, r.assisted, r.expert_options, r.drive,
    )
    s = loop.maybe_update(r.obs, r.active)
    if s is not None:
        updates += 1
        ok = all(math.isfinite(float(v)) for v in s.values())
        if not ok:
            break
check("400 ステップで PPO の更新が回り、統計が有限", updates >= 2, f"更新 {updates} 回")
# 絶対時間は PC の混み具合で揺れるので、同じプロセスで測った下位だけの推論（平らな方策と同じ量）と比べる
probe_obs = np.asarray(env.observations, dtype=np.float32)
probe_t = torch.from_numpy(probe_obs)
probe_opts = torch.from_numpy(loop.current_options)


def median_ms(fn, runs: int = 300) -> float:
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return float(np.median(samples)) * 1000.0


act_ms = min(median_ms(lambda: loop.act(probe_obs, all_active)) for _ in range(3))
loop._last_meta = None
with torch.no_grad():
    base_ms = min(median_ms(lambda: loop.policy.get_action(probe_t, probe_opts)) for _ in range(3))
check(
    "act（意図の保持と、選び直すときだけ上位を通す）は下位だけの推論の 2 倍 + 0.3ms 以内",
    act_ms <= 2.0 * base_ms + 0.3,
    f"act {act_ms:.3f}ms / 下位だけ {base_ms:.3f}ms（ループ中の中央値 {float(np.median(times)) * 1000.0:.3f}ms）",
)

print()
if FAILURES:
    print(f"NG: {len(FAILURES)} 件")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("すべて OK")
