# -*- coding: utf-8 -*-
"""log_std が可動域から出て固まらないことを検証する。"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
for st in (sys.stdout, sys.stderr):
    try:
        st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


import torch

from app import config
from app.rl.policy import ActorCritic

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if ok else 'NG  '}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


print("=" * 72)
print("1. 可動域の設定")
print("=" * 72)
lo, hi = config.PPO_LOG_STD_MIN, config.PPO_LOG_STD_MAX
print(f"  PPO_LOG_STD_MIN = {lo}  → std {math.exp(lo):.4f}")
print(f"  PPO_LOG_STD_MAX = {hi}  → std {math.exp(hi):.4f}")
print(f"  PPO_LOG_STD_INIT = {config.PPO_LOG_STD_INIT}  → std {math.exp(config.PPO_LOG_STD_INIT):.4f}")
print()
check("上限の std が行動範囲の全幅（2）未満", math.exp(hi) < 2.0, f"std {math.exp(hi):.3f}")
check("初期値が可動域の中にある", lo <= config.PPO_LOG_STD_INIT <= hi)

policy = ActorCritic(config.OBS_DIM, config.ACTION_DIM)
check("初期化直後の log_std が可動域内",
      bool(((policy.log_std >= lo) & (policy.log_std <= hi)).all()),
      str([round(float(x), 4) for x in policy.log_std.detach()]))


print()
print("=" * 72)
print("2. 範囲外では勾配が消えること（これが不具合の正体）")
print("=" * 72)
obs = torch.randn(64, config.OBS_DIM)

with torch.no_grad():
    policy.log_std.fill_(hi + 0.5)
policy.zero_grad(set_to_none=True)
_, entropy, _ = policy.evaluate(obs, torch.zeros(64, config.ACTION_DIM))
entropy.mean().backward()
grad_outside = float(policy.log_std.grad.abs().sum()) if policy.log_std.grad is not None else 0.0
print(f"  log_std = {hi + 0.5}（上限の外）のときの勾配の大きさ = {grad_outside:.8f}")
check("範囲外では勾配が 0 になる（＝戻れない）", grad_outside == 0.0)

with torch.no_grad():
    policy.log_std.fill_(hi - 0.5)
policy.zero_grad(set_to_none=True)
_, entropy, _ = policy.evaluate(obs, torch.zeros(64, config.ACTION_DIM))
entropy.mean().backward()
grad_inside = float(policy.log_std.grad.abs().sum()) if policy.log_std.grad is not None else 0.0
print(f"  log_std = {hi - 0.5}（範囲内）のときの勾配の大きさ = {grad_inside:.8f}")
check("範囲内では勾配が流れる", grad_inside > 0.0)

print()
print("=" * 72)
print("3. clamp_log_std() が範囲へ戻すこと")
print("=" * 72)
with torch.no_grad():
    policy.log_std.copy_(torch.tensor([hi + 3.0, lo - 3.0]))
before = [round(float(x), 4) for x in policy.log_std.detach()]
policy.clamp_log_std()
after = [round(float(x), 6) for x in policy.log_std.detach()]
print(f"  {before} → {after}")
eps = 1e-2
check("上限超えが上限のわずか内側へ丸められる",
      hi - eps < float(policy.log_std[0]) < hi,
      f"{float(policy.log_std[0]):.6f}（上限 {hi}）")
check("下限割れが下限のわずか内側へ丸められる",
      lo < float(policy.log_std[1]) < lo + eps,
      f"{float(policy.log_std[1]):.6f}（下限 {lo}）")


policy.zero_grad(set_to_none=True)
_, entropy, _ = policy.evaluate(obs, torch.zeros(64, config.ACTION_DIM))
entropy.mean().backward()
grad_after = float(policy.log_std.grad.abs().sum()) if policy.log_std.grad is not None else 0.0
check("丸めた後も勾配が流れる（復帰できる）", grad_after > 0.0, f"{grad_after:.8f}")

print()
print("=" * 72)
print("4. 強いエントロピー報酬で回しても上限を越えないこと")
print("=" * 72)
policy = ActorCritic(config.OBS_DIM, config.ACTION_DIM)
opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
strong = 0.5
print(f"  entropy_coef = {strong}（既定 0.001 の {int(strong / 0.001)} 倍）で 300 回更新")
traj = []
for step in range(300):
    opt.zero_grad(set_to_none=True)
    _, entropy, _ = policy.evaluate(obs, torch.zeros(64, config.ACTION_DIM))
    loss = -strong * entropy.mean()
    loss.backward()
    opt.step()
    policy.clamp_log_std()
    if step % 60 == 0 or step == 299:
        traj.append((step, float(policy.log_std[0])))
for step, v in traj:
    print(f"    {step:>4} 回目: log_std = {v:+.6f}")
final = policy.log_std.detach()
check("300 回更新しても上限を越えない",
      bool((final <= hi + 1e-6).all()), str([round(float(x), 6) for x in final]))
check("上限に張り付いても勾配は生きている（下がる余地がある）",
      bool((final >= lo).all() and (final <= hi + 1e-6).all()))


print()
print("  上限に達した状態から、逆にノイズを減らす方向へ 300 回")
start = float(policy.log_std[0])
for _ in range(300):
    opt.zero_grad(set_to_none=True)
    _, entropy, _ = policy.evaluate(obs, torch.zeros(64, config.ACTION_DIM))
    loss = +strong * entropy.mean()
    loss.backward()
    opt.step()
    policy.clamp_log_std()
end = float(policy.log_std[0])
print(f"    log_std: {start:+.6f} → {end:+.6f}")
check("上限からでも下げられる（一方通行にならない）", end < start - 0.1,
      f"{start:.4f} → {end:.4f}")


print()
print("=" * 72)
if FAILURES:
    print(f"結果: {len(FAILURES)} 件の不合格")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("結果: すべて合格")
