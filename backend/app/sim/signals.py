"""交通信号機の現示（灯色）制御。

日本の信号制御に合わせた点:

- 灯色は **青 → 黄 → 赤** の順に変わる。青から直接赤にはならない
  （道路交通法施行令 2 条の「黄色の灯火」は「停止位置を越えて進行してはならない」であり、
   停止のための猶予として必ず青と赤の間に入る）
- 黄時間は 3 秒。実務では設計速度 40〜60km/h の一般道でこの値が使われる
- 黄の後に **全赤（クリアランス）** を挟んでから交差方向を青にする。
  これが無いと交差点内に残った車と発進車が衝突する
- 交差する流れが同時に青になることはない（`MapSignal.group` が直交軸で分かれている）

現示は `sim_time` だけの関数として計算し、内部状態を持たない。
こうしておくと一時停止やエピソードのリセットと自然に整合し、
どのクライアントが見ても同じ時刻に同じ灯色になる。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from app import config
from app.contracts import MapSignal

__all__ = [
    "SignalController",
    "GREEN",
    "YELLOW",
    "RED",
    "signal_speed_limit",
    "constrain_accel",
]

#: 灯色。docs/protocol.md の frame.signals と一致させること
GREEN = 0
YELLOW = 1
RED = 2

#: 青・黄・全赤の秒数。**実体は config.py にある**（docs/protocol.md に
#: 数値が明記されているため。code_review B-22）。ここは別名。
#: 市街地の一般的なサイクル長（60〜150 秒）の下限に合わせてある。
DEFAULT_GREEN_SEC = config.SIGNAL_GREEN_SEC
DEFAULT_YELLOW_SEC = config.SIGNAL_YELLOW_SEC
DEFAULT_ALL_RED_SEC = config.SIGNAL_ALL_RED_SEC


class SignalController:
    """信号機の集合に対して、時刻から灯色を決める。

    グループ 0 が青のあいだグループ 1 は赤で、半サイクルずれて入れ替わる。
    """

    def __init__(
        self,
        signals: Sequence[MapSignal],
        *,
        green_sec: float = DEFAULT_GREEN_SEC,
        yellow_sec: float = DEFAULT_YELLOW_SEC,
        all_red_sec: float = DEFAULT_ALL_RED_SEC,
    ) -> None:
        self.green = float(green_sec)
        self.yellow = float(yellow_sec)
        self.all_red = float(all_red_sec)
        self.half_cycle = self.green + self.yellow + self.all_red
        self.cycle = self.half_cycle * 2.0

        self._count = len(signals)
        self._groups = np.array([s.group for s in signals], dtype=np.int8)

        # 交差点ごとに位相をずらす。街じゅうの信号が一斉に変わると不自然なので、
        # node_id から決まる固定のオフセットを与える（再現性のため乱数は使わない）。
        offsets = np.array(
            [(s.node_id * 7919) % max(1, int(self.cycle)) for s in signals],
            dtype=np.float64,
        )
        self._offsets = offsets

        # 計算結果を使い回すバッファ
        self._buffer = np.full(self._count, RED, dtype=np.uint8)

    def __len__(self) -> int:
        return self._count

    @property
    def count(self) -> int:
        return self._count

    def phases(self, sim_time: float) -> list[int]:
        """各信号の灯色を返す。並びは `MapData.signals` と同じ。"""
        if self._count == 0:
            return []

        t = (float(sim_time) + self._offsets) % self.cycle
        # グループ 1 は半サイクルずらす＝グループ 0 が赤のあいだに青になる
        local = np.where(self._groups == 0, t, (t + self.half_cycle) % self.cycle)

        out = self._buffer
        out.fill(RED)
        out[local < self.green] = GREEN
        np.putmask(
            out,
            (local >= self.green) & (local < self.green + self.yellow),
            YELLOW,
        )
        return out.tolist()

    def describe(self) -> str:
        """ログ用の説明。"""
        return (
            f"信号 {self._count} 基 / サイクル {self.cycle:.0f} 秒"
            f"（青 {self.green:.0f} + 黄 {self.yellow:.0f} + 全赤 {self.all_red:.0f}）"
        )


# ---------------------------------------------------------------------------
# 信号に従うための速度・加速度の制約（道路交通法施行令 2 条）
# ---------------------------------------------------------------------------

#: 停止線のどれだけ手前で止まるか [m]
STOP_MARGIN_M = 1.0

#: 停止のために使う減速度の割合。最大減速度をすべて使うと余裕が無いので少し残す
BRAKE_USE_RATIO = 0.8


def signal_speed_limit(
    distance: np.ndarray,
    phase: np.ndarray,
    speed: np.ndarray,
    max_decel_abs: float,
    dt: float = 0.05,
) -> np.ndarray:
    """信号の色ごとに許される速度の上限を返す。

    道路交通法施行令 2 条の「信号の意味」をそのまま速度の制約に翻訳したもの。

    - **青色の灯火**: 進行できる。制限なし（`inf`）
    - **赤色の灯火**: 停止位置を越えて進行してはならない。
      停止線の手前で止まりきれる速度に制限する
    - **黄色の灯火**: 停止位置を越えて進行してはならない。
      **ただし停止位置に近接していて安全に停止できない場合を除く**。
      止まれるなら赤と同じ制限、止まれないなら制限なし

    Args:
        distance: 停止線までの距離 [m]。前方に信号が無ければ `inf`
        phase: 灯色（0=青 / 1=黄 / 2=赤）
        speed: 現在の速度 [m/s]
        max_decel_abs: 最大減速度の絶対値 [m/s^2]
    Returns:
        許される速度の上限 [m/s]。制限が無いところは `inf`
    """
    brake = max(1e-3, float(max_decel_abs) * BRAKE_USE_RATIO)
    dt = max(float(dt), 1e-6)

    room = np.maximum(0.0, distance - STOP_MARGIN_M)

    # 停止線の手前で止まりきれる速度。
    #
    # 連続時間なら v = sqrt(2*a*room) だが、それだと **このステップで進む分**
    # を見込んでいないため、毎ステップ v*dt ずつ食い込んで停止線を越えてしまう。
    # 「次ステップの速度 u で dt だけ進み、そこから止まりきれる」を満たす u を解く:
    #     u^2 / (2a) + u*dt <= room
    #  => u <= -a*dt + sqrt((a*dt)^2 + 2*a*room)
    at = brake * dt
    stop_limit = -at + np.sqrt(at * at + 2.0 * brake * room, dtype=np.float64)

    # 黄色で「安全に停止できない」なら、そのまま進んでよい（施行令 2 条ただし書き）。
    # 判定にも走行中に進む分を含める。
    v = speed.astype(np.float64)
    stopping_distance = (v * v) / (2.0 * brake) + v * dt
    can_stop = stopping_distance <= room

    limit = np.full(distance.shape, np.inf, dtype=np.float64)
    limit = np.where(phase == RED, stop_limit, limit)
    limit = np.where((phase == YELLOW) & can_stop, stop_limit, limit)
    # 前方に信号が無い（距離が有限でない）ところは制限しない
    limit = np.where(np.isfinite(distance), limit, np.inf)
    return limit


def constrain_accel(
    accel_cmd: np.ndarray,
    speed: np.ndarray,
    speed_limit: np.ndarray,
    dt: float,
    max_accel: float,
    max_decel_abs: float,
) -> np.ndarray:
    """指令加速度を、次のステップで速度上限を超えないように抑える。

    指令は [-1, 1] の正規化値で、物理量への変換は
    `a = cmd * max_accel`（cmd >= 0）/ `a = cmd * max_decel_abs`（cmd < 0）。
    この変換は cmd について単調増加なので、**物理量の上限を指令の上限に
    直してから小さいほうを採ればよい**。

    エージェントの指令を書き換えるのではなく、環境側の制約として扱う。
    こうしておけば PPO から見れば「赤信号に近づくとアクセルが効かなくなる環境」
    でしかなく、学習は成立する。
    """
    limit = np.asarray(speed_limit, dtype=np.float64)
    if not np.isfinite(limit).any():
        return accel_cmd

    dt = max(float(dt), 1e-6)
    # 次速度 speed + a*dt <= limit を満たす最大の物理加速度
    with np.errstate(invalid="ignore"):
        a_max = (limit - speed.astype(np.float64)) / dt

    max_accel = max(float(max_accel), 1e-6)
    max_decel_abs = max(float(max_decel_abs), 1e-6)
    cmd_max = np.where(a_max >= 0.0, a_max / max_accel, a_max / max_decel_abs)
    cmd_max = np.where(np.isfinite(cmd_max), cmd_max, 1.0)

    out = np.minimum(accel_cmd.astype(np.float64), cmd_max)
    return np.clip(out, -1.0, 1.0).astype(np.float32)
