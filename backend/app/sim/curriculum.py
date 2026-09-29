"""ヒヤリハット（歩行者の飛び出し・前走車の急制動）を成績に応じた頻度で起こすオートカリキュラム。"""

from __future__ import annotations

from collections import deque

import numpy as np

from app import config

__all__ = [
    "CurriculumManager",
    "CURRICULUM_WINDOW",
    "CURRICULUM_COOLDOWN_EPISODES",
    "LEVEL_STEP",
    "PROMOTE_REACH",
    "PROMOTE_COLLISION",
    "DEMOTE_COLLISION",
    "INCIDENT_MAX_PROB",
    "JAYWALK_AHEAD_M",
    "JAYWALK_LATERAL_M",
    "JAYWALK_MIN_SPEED_MPS",
    "JAYWALK_SPEED_MPS",
    "JAYWALK_PARALLEL_COS",
    "KIND_JAYWALK",
    "KIND_LEADER_BRAKE",
    "LEADER_GAP_M",
    "LEADER_MIN_SPEED_MPS",
    "LEADER_BRAKE_STEPS",
    "INCIDENT_COOLDOWN_STEPS",
    "INCIDENT_WINDOW_STEPS",
    "PEDESTRIAN_RECHECK_STEPS",
    "AVOIDED_WINDOW",
]

#: 難易度を決める直近のエピソード数
CURRICULUM_WINDOW = 20
#: 難易度を変えてから、次に変えてよくなるまでのエピソード数（同じ成績で続けて昇降格しない）
CURRICULUM_COOLDOWN_EPISODES = 5
LEVEL_STEP = 0.05
PROMOTE_REACH = 0.85
PROMOTE_COLLISION = 0.10
DEMOTE_COLLISION = 0.25
#: 難易度 1.0 のときに、条件のそろった場面でヒヤリハットを起こす確率
INCIDENT_MAX_PROB = 0.20

#: 飛び出し: 歩行者が自車の前方この範囲・横この範囲にいて、自車がこの速さを超えているとき
JAYWALK_AHEAD_M = (8.0, 18.0)
JAYWALK_LATERAL_M = (2.0, 4.0)
JAYWALK_MIN_SPEED_MPS = 5.0
#: 飛び出す歩行者の小走りの速さ [m/s]
JAYWALK_SPEED_MPS = (1.2, 1.8)
#: 飛び出させる歩行者の歩道が、自車の向きとこれ以上平行なこと（|cos| で約 37 度以内）
JAYWALK_PARALLEL_COS = 0.8

#: 急制動: 前走車とのバンパー間の車間がこれ未満で、自車がこの速さを超えているとき
LEADER_GAP_M = 12.0
LEADER_MIN_SPEED_MPS = 3.0
#: 前走車に最大制動を掛けるステップ数（1.5 秒）
LEADER_BRAKE_STEPS = 30

#: 1 台に続けて起こさない間隔（8 秒）
INCIDENT_COOLDOWN_STEPS = 160
#: 起こしてから回避できたかを見届けるステップ数（5 秒）
INCIDENT_WINDOW_STEPS = 100
#: 飛び出させるか 1 度判定した歩行者を、次に判定するまでのステップ数（10 秒）
PEDESTRIAN_RECHECK_STEPS = 200
#: 回避率を数える直近のヒヤリハットの件数
AVOIDED_WINDOW = 50

KIND_NONE = 0
KIND_JAYWALK = 1
KIND_LEADER_BRAKE = 2


class CurriculumManager:
    """難易度 L（0.0〜1.0）と、起こしたヒヤリハットの見届けを持つ。"""

    def __init__(
        self,
        rng: np.random.Generator,
        num_agents: int = config.MAX_VEHICLES,
        num_pedestrians: int = config.MAX_PEDESTRIANS,
        max_probability: float = INCIDENT_MAX_PROB,
    ) -> None:
        self.rng = rng
        self.num_agents = int(num_agents)
        self.max_probability = float(max_probability)
        self.level = 0.0
        self._results: deque[tuple[bool, bool]] = deque(maxlen=CURRICULUM_WINDOW)
        self._since_change = 0
        n = self.num_agents
        self._cooldown = np.zeros(n, dtype=np.int64)
        self._window = np.zeros(n, dtype=np.int64)
        self._kind = np.zeros(n, dtype=np.int8)
        self._assisted = np.zeros(n, dtype=bool)
        self._following = np.zeros(n, dtype=bool)
        self._brake_left = np.zeros(n, dtype=np.int64)
        self._pedestrian_next = np.zeros(int(num_pedestrians), dtype=np.int64)
        self._tick = 0
        self.triggered = 0
        self.triggered_by_kind = {KIND_JAYWALK: 0, KIND_LEADER_BRAKE: 0}
        self._outcomes: deque[bool] = deque(maxlen=AVOIDED_WINDOW)

    @property
    def incident_probability(self) -> float:
        return float(self.level) * self.max_probability

    @property
    def avoided_rate(self) -> float | None:
        """見届けたヒヤリハットのうち、衝突も逸脱もせず、お手本にも代わられずに済んだ割合。まだ無ければ None。"""
        if not self._outcomes:
            return None
        return sum(self._outcomes) / len(self._outcomes)

    def record_episode_end(self, reached: bool, collision: bool) -> None:
        """エピソードの結果を 1 件積み、直近 20 件の成績で難易度を昇降格する。"""
        self._results.append((bool(reached), bool(collision)))
        self._since_change += 1
        if len(self._results) < CURRICULUM_WINDOW or self._since_change < CURRICULUM_COOLDOWN_EPISODES:
            return
        count = len(self._results)
        reach = sum(1 for r, _ in self._results if r) / count
        crash = sum(1 for _, c in self._results if c) / count
        before = self.level
        if crash >= DEMOTE_COLLISION:
            self.level = max(0.0, self.level - LEVEL_STEP)
        elif reach >= PROMOTE_REACH and crash <= PROMOTE_COLLISION:
            self.level = min(1.0, self.level + LEVEL_STEP)
        # 刻みの足し引きで 0.30000000000000004 のようにならないよう丸める
        self.level = round(self.level, 6)
        if self.level != before:
            self._since_change = 0

    def roll(self) -> bool:
        """確率 L × 0.2 で当たりを引く。難易度 0 の間は乱数を引かない。"""
        p = self.incident_probability
        if p <= 0.0:
            return False
        return bool(self.rng.random() < p)

    @staticmethod
    def jaywalk_geometry(ahead: float, lateral: float, speed: float) -> bool:
        """飛び出しの幾何条件（前方 8〜18m・横 2〜4m・18km/h 超）。"""
        return (
            JAYWALK_AHEAD_M[0] <= ahead <= JAYWALK_AHEAD_M[1]
            and JAYWALK_LATERAL_M[0] <= abs(lateral) <= JAYWALK_LATERAL_M[1]
            and speed > JAYWALK_MIN_SPEED_MPS
        )

    def should_trigger_pedestrian_jaywalk(self, ahead: float, lateral: float, speed: float) -> bool:
        """条件がそろっていれば、確率 L × 0.2 で飛び出させる。"""
        return self.jaywalk_geometry(ahead, lateral, speed) and self.roll()

    @staticmethod
    def leader_geometry(gap: float, speed: float) -> bool:
        """急制動の幾何条件（車間 12m 未満・自車が 3m/s 超）。"""
        return 0.0 <= gap < LEADER_GAP_M and speed > LEADER_MIN_SPEED_MPS

    def should_trigger_leader_braking(self, gap: float, speed: float) -> bool:
        """条件がそろっていれば、確率 L × 0.2 で前走車に急制動を掛ける。"""
        return self.leader_geometry(gap, speed) and self.roll()

    def dash_speed(self) -> float:
        return float(self.rng.uniform(*JAYWALK_SPEED_MPS))

    def can_start(self, slot: int) -> bool:
        """この車に新しいヒヤリハットを起こしてよいか（見届け中・間隔を空けている間は起こさない）。"""
        return int(self._cooldown[slot]) <= 0 and int(self._window[slot]) <= 0

    def pedestrian_ready(self, person: int) -> bool:
        return int(self._pedestrian_next[person]) <= self._tick

    def mark_pedestrian(self, person: int) -> None:
        self._pedestrian_next[person] = self._tick + PEDESTRIAN_RECHECK_STEPS

    def update_following(self, slot: int, following: bool) -> bool:
        """前走車に付いたか（直前は付いていなかった）を返す。判定は付いた瞬間の 1 回だけにする。"""
        rising = bool(following) and not bool(self._following[slot])
        self._following[slot] = bool(following)
        return rising

    def start(self, slot: int, kind: int, leader: int = -1) -> None:
        self._window[slot] = INCIDENT_WINDOW_STEPS
        self._cooldown[slot] = INCIDENT_COOLDOWN_STEPS
        self._kind[slot] = kind
        self._assisted[slot] = False
        self.triggered += 1
        self.triggered_by_kind[kind] = self.triggered_by_kind.get(kind, 0) + 1
        if kind == KIND_LEADER_BRAKE and 0 <= leader < self.num_agents:
            self._brake_left[leader] = LEADER_BRAKE_STEPS

    def braking_mask(self) -> np.ndarray:
        """いま最大制動を掛けている前走車。"""
        return self._brake_left > 0

    def incident_active(self) -> np.ndarray:
        return self._window > 0

    def advance(
        self,
        failed: np.ndarray,
        ended: np.ndarray,
        assisted: np.ndarray,
        active: np.ndarray,
    ) -> None:
        """1 ステップぶん見届けを進める。`failed` は衝突・逸脱、`ended` はエピソードが終わった車。"""
        self._tick += 1
        failed = np.asarray(failed, dtype=bool)
        ended = np.asarray(ended, dtype=bool)
        watching = self._window > 0
        self._assisted |= watching & np.asarray(assisted, dtype=bool)
        for slot in np.flatnonzero(watching):
            s = int(slot)
            if failed[s]:
                self._resolve(s, False)
                continue
            self._window[s] -= 1
            if ended[s] or self._window[s] <= 0:
                self._resolve(s, not bool(self._assisted[s]))
        self._cooldown = np.maximum(0, self._cooldown - 1)
        self._brake_left = np.maximum(0, self._brake_left - 1)
        gone = ended | ~np.asarray(active, dtype=bool)
        self._brake_left[gone] = 0
        self._following[gone] = False

    def _resolve(self, slot: int, avoided: bool) -> None:
        self._outcomes.append(bool(avoided))
        self._window[slot] = 0
        self._kind[slot] = KIND_NONE
        self._assisted[slot] = False

    def reset_incidents(self) -> None:
        """見届け中のものと前走車の制動を捨てる（世界が不連続に変わったとき）。難易度と成績は残す。"""
        self._cooldown[:] = 0
        self._window[:] = 0
        self._kind[:] = KIND_NONE
        self._assisted[:] = False
        self._following[:] = False
        self._brake_left[:] = 0
        self._pedestrian_next[:] = 0
