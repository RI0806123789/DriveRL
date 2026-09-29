"""学習中の車に経路追従（エキスパート）を割り込ませる度合いを決める（DAgger 風のオンライン模倣）。"""

from __future__ import annotations

import numpy as np

from app import config
from app.contracts import AssistDanger

__all__ = [
    "AssistDanger",
    "OnlineAssistController",
    "ASSIST_WARMUP_STEPS",
    "ASSIST_P_MIN",
    "ASSIST_SEGMENT_STEPS",
    "ASSIST_OVERRIDE_HOLD_STEPS",
    "DANGER_LANE_OFFSET_M",
    "DANGER_TTC_SEC",
    "DANGER_RED_DISTANCE_M",
    "DANGER_RED_SPEED_MPS",
    "STALL_SPEED_MPS",
    "STALL_STEPS",
    "STALL_HOLD_STEPS",
]

#: 割り込みの確率が 1.0 から下限まで下がりきる、学習の経験のステップ数（20Hz で約 17 分）
ASSIST_WARMUP_STEPS = 20_000
#: 下がりきった後も残す割り込みの確率（方策が崩れたときの保険）
ASSIST_P_MIN = 0.05
#: 確率で決めた「エキスパートが運転する／方策が運転する」を続けるステップ数（2 秒）。
#: 毎ステップ引き直すと、50ms ごとに運転者が入れ替わって操作が細切れになる
ASSIST_SEGMENT_STEPS = 40
#: 危険で割り込んだ後、危険が去ってもエキスパートに運転させ続けるステップ数（1 秒）
ASSIST_OVERRIDE_HOLD_STEPS = 20

#: 車線中心からこれ以上ずれたら割り込む [m]
DANGER_LANE_OFFSET_M = 1.5
#: 前走車・歩行者にぶつかるまでの時間がこれ未満なら割り込む [秒]
DANGER_TTC_SEC = 1.2
#: 赤信号の停止線までこれ未満で、かつ下の速さを超えていたら割り込む
DANGER_RED_DISTANCE_M = 5.0
DANGER_RED_SPEED_MPS = 2.0

#: これより遅いまま、交通に止められてもいないのに下のステップ数が過ぎたら「止まる」に固まったとみなす
STALL_SPEED_MPS = 0.5
STALL_STEPS = 100
#: 固まりから起こしたときにエキスパートが運転し続けるステップ数（3 秒。走り出すまで任せる）
STALL_HOLD_STEPS = 60


class OnlineAssistController:
    """スロットごとに「このステップはエキスパートが運転するか」を決める。"""

    def __init__(
        self,
        num_agents: int = config.MAX_VEHICLES,
        seed: int = 0,
        warmup_steps: int = ASSIST_WARMUP_STEPS,
        p_min: float = ASSIST_P_MIN,
    ) -> None:
        self.num_agents = int(num_agents)
        self.warmup_steps = max(1, int(warmup_steps))
        self.p_min = float(np.clip(p_min, 0.0, 1.0))
        self._rng = np.random.default_rng(int(seed))
        n = self.num_agents
        self._segment_left = np.zeros(n, dtype=np.int64)
        self._segment_expert = np.zeros(n, dtype=bool)
        self._override_left = np.zeros(n, dtype=np.int64)
        self._stall_steps = np.zeros(n, dtype=np.int64)

    def reset(self) -> None:
        """スロットごとの記憶を捨てる（マップの差し替え・モードの切り替え）。"""
        self._segment_left[:] = 0
        self._segment_expert[:] = False
        self._override_left[:] = 0
        self._stall_steps[:] = 0

    def get_assist_probability(self, steps: int) -> float:
        """学習の経験のステップ数から、確率で割り込む割合を返す（線形に下限まで下げる）。"""
        progress = max(0, int(steps)) / float(self.warmup_steps)
        return float(max(self.p_min, 1.0 - progress))

    def check_critical_danger(self, danger: AssistDanger) -> np.ndarray:
        """確率に関わらず割り込むべきスロット（車線逸脱・衝突切迫・赤信号突破・固まり）。"""
        lane = np.asarray(danger.lane_offset, dtype=np.float64) > DANGER_LANE_OFFSET_M
        ttc = np.asarray(danger.ttc, dtype=np.float64) < DANGER_TTC_SEC
        speed = np.asarray(danger.speed, dtype=np.float64)
        red = (np.asarray(danger.red_distance, dtype=np.float64) < DANGER_RED_DISTANCE_M) & (
            speed > DANGER_RED_SPEED_MPS
        )
        return lane | ttc | red | self._stalled()

    def _stalled(self) -> np.ndarray:
        return self._stall_steps >= STALL_STEPS

    def decide(self, steps: int, active: np.ndarray, danger: AssistDanger) -> np.ndarray:
        """このステップでエキスパートが運転するスロットを返す。**1 ステップに 1 回だけ呼ぶこと。**"""
        n = self.num_agents
        active = np.asarray(active, dtype=bool).reshape(n)
        idle = ~active
        self._segment_left[idle] = 0
        self._override_left[idle] = 0
        self._stall_steps[idle] = 0

        stalled = active & (np.abs(np.asarray(danger.speed, dtype=np.float64)) < STALL_SPEED_MPS)
        stalled &= ~np.asarray(danger.held, dtype=bool)
        self._stall_steps = np.where(stalled, self._stall_steps + 1, 0)

        critical = self.check_critical_danger(danger) & active
        woken = self._stalled() & active
        self._override_left[critical] = np.maximum(
            self._override_left[critical], ASSIST_OVERRIDE_HOLD_STEPS
        )
        self._override_left[woken] = np.maximum(self._override_left[woken], STALL_HOLD_STEPS)
        # 固まりから起こしたら数え直す（割り込んでいる間に走り出す）
        self._stall_steps[critical] = 0

        expired = active & (self._segment_left <= 0)
        if expired.any():
            p = self.get_assist_probability(steps)
            draws = self._rng.random(n) < p
            self._segment_expert[expired] = draws[expired]
            self._segment_left[expired] = ASSIST_SEGMENT_STEPS

        expert = active & ((self._override_left > 0) | self._segment_expert)
        self._segment_left[active] -= 1
        self._override_left[active] = np.maximum(0, self._override_left[active] - 1)
        return expert
