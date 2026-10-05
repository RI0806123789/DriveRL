"""検出枠と推定距離に連動する安全ギミック（後退 AEB・切り返し・巻き込み防止・交差点の左右確認）。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

import numpy as np

from app import config
from app.contracts import BRANCH_LEFT, BRANCH_RIGHT
from app.percep.encoder import SURROUND_CLASSES, local_xy
from app.percep.types import (
    DEFAULT_CAMERA,
    LEFT_CAMERA,
    REAR_CAMERA,
    RIGHT_CAMERA,
    CameraSpec,
    DetClass,
    Detection,
    OcclusionResult,
    PerceptionResult,
)
from app.sim.signals import stop_speed_limit

if TYPE_CHECKING:
    from app.sim.world import World

__all__ = [
    "ASSIST_BLIND_SPOT",
    "ASSIST_CREEP",
    "ASSIST_DETOUR",
    "ASSIST_FRONT_HOLD",
    "ASSIST_PEEK",
    "ASSIST_REVERSE_CHECK",
    "ASSIST_REVERSE_STOP",
    "ASSIST_REVERSING",
    "ASSIST_YIELD",
    "SafetyCommand",
    "SafetySupervisor",
]

#: 画面と `frame.vehicles[].assist` に出す介入の種類
ASSIST_FRONT_HOLD = "front_hold"
ASSIST_BLIND_SPOT = "blind_spot"
ASSIST_PEEK = "peek"
ASSIST_CREEP = "creep"
ASSIST_YIELD = "yield"
ASSIST_REVERSE_CHECK = "reverse_check"
ASSIST_REVERSING = "reversing"
ASSIST_REVERSE_STOP = "reverse_stop"
ASSIST_DETOUR = "detour"

HAZARD_CAUTION = 1
HAZARD_STOP = 2

_MODE_DRIVE = "drive"
_MODE_CHECK = "check"
_MODE_REVERSE = "reverse"
_MODE_RESUME = "resume"
_MODE_DETOUR = "detour"

HALF_LENGTH = config.VEHICLE_LENGTH * 0.5
HALF_WIDTH = config.VEHICLE_WIDTH * 0.5
DECEL = abs(config.MAX_DECEL)
#: 速度の上限がこれ以下なら「止めている」とみなす [m/s]（`env.TRAFFIC_HOLD_MPS` と同じ）
HOLD_MPS = 1.0

#: 前方カメラで障害物を見る距離と、バンパーから障害物の面までに空ける距離 [m]
FRONT_RANGE_M = 30.0
FRONT_STOP_MARGIN_M = 2.0
#: 進路の幅に足す余裕 [m]（車体の半幅 + 物体の半径 + これ）
CORRIDOR_MARGIN_M = 0.5
#: 検出を経路へ写すとき、経路からこれより離れていれば進路の外とみなす [m]
ROUTE_MATCH_M = 8.0

#: 障害物の手前でこの速さ以下のまま、これだけ経ったら立ち往生とみなす
STUCK_MPS = 0.3
STUCK_SEC = 5.0
#: 同じ障害物に対して切り返しを試す回数と、あきらめた後にもう一度試すまでの時間 [秒]
MAX_ATTEMPTS = 2
RETRY_SEC = 20.0
#: 試した回数を忘れるまでに前へ進む距離 [m]
ATTEMPT_FORGET_M = 15.0

#: 切り返しで下がる距離・最低限下がりたい距離・速さ
REVERSE_DISTANCE_M = 6.0
REVERSE_MIN_M = 2.5
REVERSE_SPEED_MPS = 1.2
#: 後退 AEB: バンパーから物体の面までに残す距離 [m] と、注意を出し始める距離
REVERSE_STOP_MARGIN_M = 0.8
REVERSE_CAUTION_M = 3.0
REVERSE_MAX_SEC = 12.0
#: 後退 AEB で止まったまま、これだけ経ったら下がるのをやめる [秒]
AEB_WAIT_SEC = 2.0
#: 下がる前の安全確認で待てる時間 [秒]
CHECK_MAX_SEC = 15.0
#: 下がる前: 車体の横にこれ以内の歩行者・車がいれば待つ [m]（車体の側面から）
SIDE_CLEAR_M = 2.5
#: 後方カメラから後ろのバンパーまで [m]。走行可能距離（建物）はカメラから測るのでこの分を引く
REAR_CAMERA_TO_BUMPER_M = HALF_LENGTH + REAR_CAMERA.forward
#: 後方の走行可能距離で見る方位（±90 度を 9 本に分けたうちの中央 3 本 = ±22.5 度）
REAR_FREE_BINS = slice(3, 6)

#: 回避: 車体の側面から障害物の面まで空ける距離・ずらす上限・速さ・戻す区間 [m]
PASS_CLEARANCE_M = 0.6
DETOUR_MAX_OFFSET_M = 2.8
DETOUR_SPEED_MPS = 3.0
DETOUR_RETURN_M = 10.0
DETOUR_MAX_SEC = 40.0
#: 回避を始める前に、抜ける側のカメラが空くのを待てる時間 [秒]
DETOUR_WAIT_SEC = 10.0
#: 回避で抜ける側を見る範囲 [m]
DETOUR_SIDE_RANGE_M = 25.0

#: 巻き込み防止: 曲がり始めのこれだけ手前（バンパーから）から曲がる側のカメラを見る [m]
BLIND_CHECK_M = 10.0
BLIND_PEDESTRIAN_RANGE_M = 8.0
#: 車体の側面からこれより遠い歩行者は、曲がる先の歩道の人とみなして待たない [m]
BLIND_SIDE_M = 5.0
#: 待ち続ける上限 [秒]。ただし車体の側面からこれより近い人は、時間に関係なく待つ [m]
BLIND_MAX_HOLD_SEC = 8.0
BLIND_KEEP_M = 1.5
#: 枠が消えてからも待つ時間 [秒]。前方（±34 度）と横（56〜124 度）のカメラの間は写らない
BLIND_MEMORY_SEC = 1.0

#: 信号の無い交差点: 入口のこれだけ手前（バンパーから）から徐行する [m] と、その速さ
PEEK_ZONE_M = 10.0
PEEK_SPEED_MPS = 4.0
#: 近づいてくる車を待つ範囲 [m]。左方優先（道交法 36 条）なので左は遠くまで、右は間近だけ
PEEK_LEFT_RANGE_M = 30.0
PEEK_RIGHT_RANGE_M = 12.0
#: これより速く近づいてくる車を「接近車」とみなす [m/s]
APPROACH_MPS = 0.8
#: 待ち続ける上限 [秒]。4 方向で互いに待ち合って動けなくなるのを避ける
PEEK_MAX_WAIT_SEC = 6.0
TRACK_SMOOTH = 0.5

#: 周囲カメラの結果がこれより古ければ判断に使わない [秒]（CNN で回すときは毎ステップ撮り直さないため）
FRESH_SEC = 0.2

#: 見通しの悪い交差点の顔出し: 左右のカメラの見通し距離がこれ以上になったら開けたとみなし、これ未満に戻ったら閉じたとみなす [m]
LOS_OPEN_M = 20.0
LOS_CLOSE_M = 15.0
#: 見通しが効かない間の速さ v = √(クリープ速度² + 2 a max(入口までの距離 - 余裕, 0))
CREEP_DECEL_MPS2 = 1.5
CREEP_MARGIN_M = 1.5
CREEP_SPEED_MPS = 1.2
#: 入口を車体の中心（左右のカメラの位置）がこれだけ越えても開けなければ、顔出しをやめて通常へ戻す [m]
CREEP_PAST_M = 2.0
#: 入口のこれだけ手前（バンパーから）から見通しを見る [m]。40km/h から 1.5m/s² で落とすのに約 34m 要るので、
#: 徐行（PEEK_ZONE_M）と同じ 10m からでは間に合わず、最大減速で詰めることになる
CREEP_ZONE_M = 30.0


@dataclass
class SafetyCommand:
    """1 台ぶんの介入の指示。env.step がこれに従って操作を抑える・引き受ける。"""

    speed_cap: float = math.inf
    #: 経路からの横ずらし [m]（左が正）。回避の間だけ 0 でなくなる
    offset: float = 0.0
    reverse: bool = False
    reverse_cap: float = 0.0
    #: 操作を丸ごと引き受けている（後退・止まってからギアを戻す・回避）。PPO の車なら学習から外す
    override: bool = False
    assist: str = ""

    @property
    def holding(self) -> bool:
        """止めている（交通に止められているのと同じ扱いにする）か。"""
        return self.override or self.speed_cap <= HOLD_MPS


@dataclass
class _Track:
    """片側のカメラで一番近い車の動き（近づいてくるかを見る）。"""

    x: float = math.nan
    y: float = math.nan
    t: float = -1.0
    rate: float = 0.0


@dataclass
class _Blocker:
    arc: float
    lat: float
    radius: float


@dataclass
class _SlotSafety:
    #: 状態を作ったときの経路の通し番号（`World.route_serial`）。変わったら弧長の記憶ごと捨てる
    route_serial: int = -1
    mode: str = _MODE_DRIVE
    since: float = 0.0
    stuck_since: float = -1.0
    attempts: int = 0
    attempt_arc: float = 0.0
    retry_at: float = -math.inf
    blockers: list[_Blocker] = field(default_factory=list)
    offset: float = 0.0
    pass_arc: float = 0.0
    reverse_from: tuple[float, float] = (0.0, 0.0)
    reverse_target: float = REVERSE_DISTANCE_M
    aeb_since: float = -1.0
    blind_since: float = -1.0
    blind_seen: float = -1.0
    peek_since: float = -1.0
    #: 顔出しの間、左右の見通しが開けているか（ヒステリシスつき）。交差点の手前に入る前は None
    sight_open: bool | None = None
    creep_junction: float = math.nan
    tracks: dict[str, _Track] = field(
        default_factory=lambda: {LEFT_CAMERA.key: _Track(), RIGHT_CAMERA.key: _Track()}
    )
    demand: tuple[str, ...] = ()


def _object_radius(det: Detection, spec: CameraSpec, dist: float) -> float:
    """物体の半径の見込み [m]。歩行者と車両は既知の寸法、障害物は箱の幅から逆算する。"""
    if det.cls is DetClass.PEDESTRIAN:
        return float(config.PEDESTRIAN_RADIUS)
    if det.cls is DetClass.VEHICLE:
        return HALF_WIDTH
    width = (float(det.x1) - float(det.x0)) * spec.width / spec.focal_px * max(dist, 0.1)
    return float(np.clip(width * 0.5, 0.2, 1.5))


def _mark(det: Detection, level: int) -> None:
    det.hazard = max(int(det.hazard or 0), int(level))


class SafetySupervisor:
    """車ごとの安全ギミックの状態を持ち、毎ステップの認識結果から介入の指示を作る。"""

    def __init__(self) -> None:
        self._state = [_SlotSafety() for _ in range(config.MAX_VEHICLES)]
        self.commands = [SafetyCommand() for _ in range(config.MAX_VEHICLES)]

    def _release(self, slot: int) -> None:
        """安全ギミックを掛けなくなった車の介入を解く。経路の差し替えは `_evaluate_slot` が通し番号で見る。"""
        slot = int(slot)
        if 0 <= slot < config.MAX_VEHICLES:
            self._state[slot] = _SlotSafety()
            self.commands[slot] = SafetyCommand()

    def demand(self, slot: int) -> tuple[str, ...]:
        """次のステップで撮り直してほしい周囲カメラ（CNN の枚数予算で優先する）。"""
        return self._state[int(slot)].demand

    def assist(self, slot: int) -> str:
        return self.commands[int(slot)].assist

    def holding(self, slot: int) -> bool:
        return self.commands[int(slot)].holding

    def evaluate(
        self,
        world: "World",
        slots: Iterable[int],
        now: float,
        front: dict[int, PerceptionResult],
        surround: dict[int, dict[str, PerceptionResult]],
        ages: dict[int, dict[str, float]],
        rear_free: dict[int, np.ndarray],
        occlusion: dict[int, OcclusionResult] | None = None,
    ) -> None:
        """認識結果から介入の指示を作り直し、介入の根拠になった検出に危険度を付ける。"""
        for result in front.values():
            for det in result.detections:
                det.hazard = None
        for cams in surround.values():
            for result in cams.values():
                for det in result.detections:
                    det.hazard = None

        wanted = {int(s) for s in slots}
        for slot in range(config.MAX_VEHICLES):
            if slot not in wanted:
                if self._state[slot].mode != _MODE_DRIVE or self.commands[slot].assist:
                    self._release(slot)
                continue
            fresh = {
                key: result
                for key, result in (surround.get(slot) or {}).items()
                if (ages.get(slot) or {}).get(key, math.inf) <= FRESH_SEC
            }
            sight = _fresh_sight((occlusion or {}).get(slot), fresh)
            self.commands[slot] = self._evaluate_slot(
                world, slot, float(now), front.get(slot), fresh, rear_free.get(slot), sight
            )

    def _evaluate_slot(
        self,
        world: "World",
        slot: int,
        now: float,
        front: PerceptionResult | None,
        cams: dict[str, PerceptionResult],
        rear_free: np.ndarray | None,
        sight: tuple[float | None, float | None] = (None, None),
    ) -> SafetyCommand:
        serial = int(world.route_serial[slot])
        if self._state[slot].route_serial != serial:
            self._state[slot] = _SlotSafety(route_serial=serial)
        st = self._state[slot]
        self._update_tracks(world, slot, now, cams)
        if st.mode == _MODE_CHECK:
            return self._check(world, slot, now, cams, rear_free)
        if st.mode == _MODE_REVERSE:
            return self._reverse(world, slot, now, cams, rear_free)
        if st.mode == _MODE_RESUME:
            return self._resume(world, slot, now, cams)
        if st.mode == _MODE_DETOUR:
            return self._detour(world, slot, now, front)
        return self._drive(world, slot, now, front, cams, sight)

    def _set_mode(self, st: _SlotSafety, mode: str, now: float) -> None:
        st.mode = mode
        st.since = now
        st.aeb_since = -1.0

    def _drive(
        self,
        world: "World",
        slot: int,
        now: float,
        front: PerceptionResult | None,
        cams: dict[str, PerceptionResult],
        sight: tuple[float | None, float | None] = (None, None),
    ) -> SafetyCommand:
        st = self._state[slot]
        cmd = SafetyCommand()
        arc = float(world.arc[slot])
        if st.attempts and arc - st.attempt_arc >= ATTEMPT_FORGET_M:
            st.attempts = 0
        demand: list[str] = []

        cap, blockers = self._front_hold(world, slot, front, 0.0, (DetClass.OBSTACLE,))
        if cap < cmd.speed_cap:
            cmd.speed_cap = cap
            if cap <= HOLD_MPS:
                cmd.assist = ASSIST_FRONT_HOLD
        stopped = abs(float(world.fleet.speed[slot])) <= STUCK_MPS
        if blockers and cap <= HOLD_MPS and stopped:
            if st.stuck_since < 0.0:
                st.stuck_since = now
            demand.extend((REAR_CAMERA.key, LEFT_CAMERA.key, RIGHT_CAMERA.key))
            if (
                now - st.stuck_since >= STUCK_SEC
                and now >= st.retry_at
                and st.attempts < MAX_ATTEMPTS
            ):
                st.blockers = blockers
                st.attempts += 1
                st.attempt_arc = arc
                st.stuck_since = -1.0
                self._set_mode(st, _MODE_CHECK, now)
                st.demand = tuple(demand)
                return SafetyCommand(speed_cap=0.0, assist=ASSIST_REVERSE_CHECK)
        else:
            st.stuck_since = -1.0

        blind_cap, blind_cam, watching = self._blind_spot(world, slot, now, front, cams)
        if blind_cam:
            demand.append(blind_cam)
        if blind_cap < cmd.speed_cap:
            cmd.speed_cap = blind_cap
            if blind_cap <= HOLD_MPS:
                cmd.assist = ASSIST_BLIND_SPOT
        # 赤信号などで先に止まっていても、曲がる側に写っている間はそれを見て待っていると示す
        if watching and stopped and not cmd.assist:
            cmd.assist = ASSIST_BLIND_SPOT

        peek_cap, peek_assist, peeking = self._peek(world, slot, now, cams)
        if peeking:
            demand.extend((LEFT_CAMERA.key, RIGHT_CAMERA.key))
        if peek_cap < cmd.speed_cap:
            cmd.speed_cap = peek_cap
        if peek_assist and not cmd.assist:
            cmd.assist = peek_assist

        creep_cap, creeping = self._creep(world, slot, sight)
        if creeping:
            demand.extend((LEFT_CAMERA.key, RIGHT_CAMERA.key))
        if creep_cap < cmd.speed_cap:
            cmd.speed_cap = creep_cap
        # 見通しが悪いと見ている間は、手前の徐行（4m/s）のほうが低くても顔出しとして示す
        if math.isfinite(creep_cap) and cmd.assist in ("", ASSIST_PEEK):
            cmd.assist = ASSIST_CREEP

        st.demand = tuple(dict.fromkeys(demand))
        return cmd

    def _creep(
        self, world: "World", slot: int, sight: tuple[float | None, float | None]
    ) -> tuple[float, bool]:
        """見通しの悪い交差点の顔出し。交差道路がある側の見通しが開けるまで、入口で止まれる速さ + クリープ速度に絞る。"""
        st = self._state[slot]
        centre_gap, sides = world.junction_ahead(slot, CREEP_PAST_M)
        bumper_gap = centre_gap - HALF_LENGTH
        if not math.isfinite(centre_gap) or bumper_gap > CREEP_ZONE_M or not sides:
            st.sight_open = None
            return math.inf, False
        junction = float(world.arc[slot]) + centre_gap
        # 弧長の足し引きの丸めで毎ステップ別の交差点と取り違えないよう、幅を持たせて比べる（NaN は別物）
        if not abs(st.creep_junction - junction) <= 0.5:
            st.creep_junction = junction
            st.sight_open = None
        needed = [v for bit, v in ((BRANCH_LEFT, sight[0]), (BRANCH_RIGHT, sight[1])) if sides & bit]
        if any(v is None for v in needed):
            # 撮れていない間は判断しない（手前の徐行 `_peek` はそのまま効く）
            return math.inf, True
        reach = min(float(v) for v in needed if v is not None)
        if st.sight_open is None:
            st.sight_open = reach >= LOS_OPEN_M
        elif st.sight_open and reach < LOS_CLOSE_M:
            st.sight_open = False
        elif not st.sight_open and reach >= LOS_OPEN_M:
            st.sight_open = True
        if st.sight_open:
            return math.inf, True
        room = max(bumper_gap - CREEP_MARGIN_M, 0.0)
        # 減速度 a でちょうどクリープ速度に落ちる速さ。√(2ad) + v とすると、止まり際の減速度が際限なく大きくなる
        return math.sqrt(CREEP_SPEED_MPS * CREEP_SPEED_MPS + 2.0 * CREEP_DECEL_MPS2 * room), True

    def _front_hold(
        self,
        world: "World",
        slot: int,
        front: PerceptionResult | None,
        offset: float,
        classes: tuple[DetClass, ...],
    ) -> tuple[float, list[_Blocker]]:
        """前方カメラの検出のうち、進路（経路を `offset` だけずらした帯）に入るものの手前で止まる上限。"""
        if front is None:
            return math.inf, []
        arc = float(world.arc[slot])
        cap = math.inf
        blockers: list[_Blocker] = []
        for det in front.detections:
            if det.cls not in classes:
                continue
            fx, fy, dist = local_xy(det, DEFAULT_CAMERA)
            if fx <= 0.0 or dist > FRONT_RANGE_M:
                continue
            wx, wy = _to_world(world, slot, fx, fy)
            where = _route_frame(world, slot, wx, wy, arc, arc + FRONT_RANGE_M + 5.0)
            if where is None:
                continue
            obj_arc, obj_lat = where
            radius = _object_radius(det, DEFAULT_CAMERA, dist)
            if abs(obj_lat - offset) > HALF_WIDTH + radius + CORRIDOR_MARGIN_M:
                continue
            gap = obj_arc - arc - HALF_LENGTH - radius
            limit = float(
                stop_speed_limit(np.float64(gap), DECEL, config.DT, margin_m=FRONT_STOP_MARGIN_M)
            )
            _mark(det, HAZARD_STOP if limit <= HOLD_MPS else HAZARD_CAUTION)
            cap = min(cap, limit)
            if det.cls is DetClass.OBSTACLE:
                blockers.append(_Blocker(obj_arc, obj_lat, radius))
        return cap, blockers

    def _blind_spot(
        self,
        world: "World",
        slot: int,
        now: float,
        front: PerceptionResult | None,
        cams: dict[str, PerceptionResult],
    ) -> tuple[float, str, bool]:
        """曲がる側に歩行者・並走車が写っている間は曲がり始めない（巻き込み防止）。"""
        st = self._state[slot]
        dist, side = world.turn_ahead(slot)
        if side == 0 or int(world.turn_signal[slot]) != side:
            st.blind_since = -1.0
            return math.inf, "", False
        gap = dist - HALF_LENGTH
        cam = LEFT_CAMERA if side < 0 else RIGHT_CAMERA
        if gap > BLIND_CHECK_M + 10.0:
            st.blind_since = -1.0
            return math.inf, "", False
        if gap > BLIND_CHECK_M:
            return math.inf, cam.key, False

        # 横のカメラは真横（±34 度）しか写さないので、曲がる側の斜め前は前方カメラで見る
        views: list[tuple[CameraSpec, PerceptionResult]] = []
        if cam.key in cams:
            views.append((cam, cams[cam.key]))
        if front is not None:
            views.append((DEFAULT_CAMERA, front))
        want_left = side < 0
        danger: list[tuple[float, Detection]] = []
        for view, result in views:
            for det in result.detections:
                fx, fy, d = local_xy(det, view)
                if (fy > 0.0) != want_left:
                    continue
                side_gap = abs(fy) - HALF_WIDTH
                if det.cls is DetClass.PEDESTRIAN:
                    if (
                        d <= BLIND_PEDESTRIAN_RANGE_M
                        and -HALF_LENGTH - 2.0 <= fx <= HALF_LENGTH + 8.0
                        and side_gap <= BLIND_SIDE_M
                    ):
                        danger.append((side_gap, det))
                elif det.cls is DetClass.VEHICLE and view is not DEFAULT_CAMERA:
                    if -HALF_LENGTH - 4.0 <= fx <= HALF_LENGTH + 1.0 and abs(fy) <= 4.5:
                        danger.append((side_gap, det))
        if not danger:
            # 枠が消えても、前方と横のカメラのすき間に入っただけのことがあるので少し待つ
            if st.blind_since >= 0.0 and now - st.blind_seen <= BLIND_MEMORY_SEC:
                return (0.0 if gap <= 0.5 else float(
                    stop_speed_limit(np.float64(gap), DECEL, config.DT, margin_m=0.5)
                )), cam.key, False
            st.blind_since = -1.0
            return math.inf, cam.key, False
        st.blind_seen = now
        if st.blind_since < 0.0:
            st.blind_since = now
        nearest = min(g for g, _ in danger)
        if now - st.blind_since > BLIND_MAX_HOLD_SEC and nearest > BLIND_KEEP_M:
            for _g, det in danger:
                _mark(det, HAZARD_CAUTION)
            return math.inf, cam.key, False
        for _g, det in danger:
            _mark(det, HAZARD_STOP)
        if gap <= 0.5:
            return 0.0, cam.key, True
        return float(
            stop_speed_limit(np.float64(gap), DECEL, config.DT, margin_m=0.5)
        ), cam.key, True

    def _peek(
        self,
        world: "World",
        slot: int,
        now: float,
        cams: dict[str, PerceptionResult],
    ) -> tuple[float, str, bool]:
        """信号の無い交差点の手前で徐行し、左右のカメラに接近車が写っている間は入口で待つ。"""
        st = self._state[slot]
        gap = world.next_junction(slot) - HALF_LENGTH
        if not math.isfinite(gap) or gap > PEEK_ZONE_M + 10.0:
            st.peek_since = -1.0
            return math.inf, "", False
        if gap > PEEK_ZONE_M or gap <= 0.0:
            st.peek_since = -1.0
            return math.inf, "", gap > 0.0

        cap = PEEK_SPEED_MPS
        approaching: list[Detection] = []
        for cam, reach in ((LEFT_CAMERA, PEEK_LEFT_RANGE_M), (RIGHT_CAMERA, PEEK_RIGHT_RANGE_M)):
            result = cams.get(cam.key)
            track = st.tracks[cam.key]
            if result is None or track.rate <= APPROACH_MPS:
                continue
            nearest = _nearest(result, cam, (DetClass.VEHICLE,))
            if nearest is not None and nearest[0] <= reach:
                approaching.append(nearest[1])
        if not approaching:
            st.peek_since = -1.0
            return cap, ASSIST_PEEK, True
        if st.peek_since < 0.0:
            st.peek_since = now
        if now - st.peek_since > PEEK_MAX_WAIT_SEC:
            for det in approaching:
                _mark(det, HAZARD_CAUTION)
            return cap, ASSIST_PEEK, True
        for det in approaching:
            _mark(det, HAZARD_STOP)
        hold = float(stop_speed_limit(np.float64(gap), DECEL, config.DT, margin_m=0.5))
        return min(cap, hold), ASSIST_YIELD, True

    def _update_tracks(
        self, world: "World", slot: int, now: float, cams: dict[str, PerceptionResult]
    ) -> None:
        """左右のカメラで一番近い車が、自車のいまの位置へ近づく速さを均して持つ。"""
        st = self._state[slot]
        me_x = float(world.fleet.x[slot])
        me_y = float(world.fleet.y[slot])
        for cam in (LEFT_CAMERA, RIGHT_CAMERA):
            track = st.tracks[cam.key]
            result = cams.get(cam.key)
            if result is None:
                continue
            nearest = _nearest(result, cam, (DetClass.VEHICLE,))
            if nearest is None:
                st.tracks[cam.key] = _Track()
                continue
            fx, fy = nearest[2], nearest[3]
            wx, wy = _to_world(world, slot, fx, fy)
            if track.t >= 0.0 and now > track.t and math.isfinite(track.x):
                before = math.hypot(track.x - me_x, track.y - me_y)
                after = math.hypot(wx - me_x, wy - me_y)
                rate = (before - after) / (now - track.t)
                track.rate += (rate - track.rate) * TRACK_SMOOTH
            track.x, track.y, track.t = wx, wy, now

    def _side_clear(self, cams: dict[str, PerceptionResult], keys: Iterable[str]) -> bool:
        """車体の横すぐ（`SIDE_CLEAR_M` 以内）に歩行者・車が写っていないか。写っていればそれに印を付ける。"""
        clear = True
        for key in keys:
            cam = LEFT_CAMERA if key == LEFT_CAMERA.key else RIGHT_CAMERA
            result = cams.get(key)
            if result is None:
                return False
            for det in result.detections:
                if det.cls not in SURROUND_CLASSES:
                    continue
                fx, fy, _d = local_xy(det, cam)
                if abs(fx) <= HALF_LENGTH + 1.0 and abs(fy) - HALF_WIDTH <= SIDE_CLEAR_M:
                    _mark(det, HAZARD_STOP)
                    clear = False
        return clear

    def _rear_gap(
        self, cams: dict[str, PerceptionResult], rear_free: np.ndarray | None, reach: float
    ) -> tuple[float, list[Detection]]:
        """後ろのバンパーから、下がる帯に入る一番近い物体（か建物）までの距離 [m]。"""
        gap = math.inf
        hits: list[Detection] = []
        result = cams.get(REAR_CAMERA.key)
        if result is not None:
            for det in result.detections:
                if det.cls not in SURROUND_CLASSES:
                    continue
                fx, fy, dist = local_xy(det, REAR_CAMERA)
                radius = _object_radius(det, REAR_CAMERA, dist)
                if fx > -HALF_LENGTH + 0.3:
                    continue
                if abs(fy) > HALF_WIDTH + radius + CORRIDOR_MARGIN_M:
                    continue
                g = -fx - HALF_LENGTH - radius
                if g <= reach:
                    hits.append(det)
                gap = min(gap, g)
        if rear_free is not None and rear_free.size >= 6:
            gap = min(gap, float(np.min(rear_free[REAR_FREE_BINS])) - REAR_CAMERA_TO_BUMPER_M)
        return gap, hits

    def _check(
        self,
        world: "World",
        slot: int,
        now: float,
        cams: dict[str, PerceptionResult],
        rear_free: np.ndarray | None,
    ) -> SafetyCommand:
        """立ち往生: 後方・左右のカメラで下がれる空きを確かめてから後退に移る。"""
        st = self._state[slot]
        st.demand = (REAR_CAMERA.key, LEFT_CAMERA.key, RIGHT_CAMERA.key)
        hold = SafetyCommand(speed_cap=0.0, assist=ASSIST_REVERSE_CHECK)
        if self._plan_detour(world, slot) is None:
            self._give_up(st, now)
            return SafetyCommand(speed_cap=0.0, assist=ASSIST_FRONT_HOLD)
        if REAR_CAMERA.key not in cams:
            return hold
        gap, hits = self._rear_gap(cams, rear_free, REVERSE_DISTANCE_M + REVERSE_STOP_MARGIN_M)
        room = gap - REVERSE_STOP_MARGIN_M
        side_ok = self._side_clear(cams, (LEFT_CAMERA.key, RIGHT_CAMERA.key))
        if room >= REVERSE_MIN_M and side_ok:
            for det in hits:
                _mark(det, HAZARD_CAUTION)
            st.reverse_from = (float(world.fleet.x[slot]), float(world.fleet.y[slot]))
            st.reverse_target = float(min(REVERSE_DISTANCE_M, room))
            self._set_mode(st, _MODE_REVERSE, now)
            return SafetyCommand(
                reverse=True, reverse_cap=0.0, override=True, assist=ASSIST_REVERSING
            )
        for det in hits:
            _mark(det, HAZARD_STOP)
        if now - st.since > CHECK_MAX_SEC:
            self._give_up(st, now)
            return SafetyCommand(speed_cap=0.0, assist=ASSIST_FRONT_HOLD)
        return hold

    def _reverse(
        self,
        world: "World",
        slot: int,
        now: float,
        cams: dict[str, PerceptionResult],
        rear_free: np.ndarray | None,
    ) -> SafetyCommand:
        """切り返しの後退。後方カメラの検出枠が近づいたら止める（後退 AEB）。"""
        st = self._state[slot]
        st.demand = (REAR_CAMERA.key, LEFT_CAMERA.key, RIGHT_CAMERA.key)
        moved = math.hypot(
            float(world.fleet.x[slot]) - st.reverse_from[0],
            float(world.fleet.y[slot]) - st.reverse_from[1],
        )
        left = st.reverse_target - moved
        if REAR_CAMERA.key in cams:
            gap, hits = self._rear_gap(cams, rear_free, REVERSE_CAUTION_M)
        else:
            # 後ろが見えていないまま下がらない
            gap, hits = 0.0, []
        aeb = float(
            stop_speed_limit(np.float64(gap), DECEL, config.DT, margin_m=REVERSE_STOP_MARGIN_M)
        )
        finish = float(stop_speed_limit(np.float64(max(left, 0.0)), DECEL, config.DT, margin_m=0.0))
        cap = min(REVERSE_SPEED_MPS, aeb, finish)
        stopped_by_aeb = aeb <= 0.3
        for det in hits:
            _mark(det, HAZARD_STOP if stopped_by_aeb else HAZARD_CAUTION)
        if stopped_by_aeb:
            if st.aeb_since < 0.0:
                st.aeb_since = now
        else:
            st.aeb_since = -1.0

        done = left <= 0.05
        blocked = st.aeb_since >= 0.0 and now - st.aeb_since >= AEB_WAIT_SEC
        timeout = now - st.since >= REVERSE_MAX_SEC
        if done or blocked or timeout:
            if moved < REVERSE_MIN_M:
                self._give_up(st, now)
            else:
                self._set_mode(st, _MODE_RESUME, now)
            return SafetyCommand(speed_cap=0.0, override=True, assist=ASSIST_REVERSE_STOP if blocked else ASSIST_REVERSING)
        return SafetyCommand(
            reverse=True,
            reverse_cap=float(cap),
            override=True,
            assist=ASSIST_REVERSE_STOP if stopped_by_aeb else ASSIST_REVERSING,
        )

    def _resume(
        self,
        world: "World",
        slot: int,
        now: float,
        cams: dict[str, PerceptionResult],
    ) -> SafetyCommand:
        """下がり終えた: 止まって前進へ戻し、抜ける側のカメラが空くのを待ってから回避を始める。"""
        st = self._state[slot]
        wait = SafetyCommand(speed_cap=0.0, override=True, assist=ASSIST_DETOUR)
        if int(world.fleet.gear[slot]) < 0 or abs(float(world.fleet.speed[slot])) > 0.05:
            return wait
        plan = self._plan_detour(world, slot)
        if plan is None:
            self._give_up(st, now)
            return SafetyCommand(speed_cap=0.0, assist=ASSIST_FRONT_HOLD)
        offset, pass_arc = plan
        side = LEFT_CAMERA if offset > 0.0 else RIGHT_CAMERA
        st.demand = (side.key,)
        result = cams.get(side.key)
        clear = result is not None and self._pass_side_clear(st, result, side)
        if clear:
            st.offset = offset
            st.pass_arc = pass_arc
            self._set_mode(st, _MODE_DETOUR, now)
            return SafetyCommand(offset=offset, speed_cap=DETOUR_SPEED_MPS, override=True, assist=ASSIST_DETOUR)
        if now - st.since > DETOUR_WAIT_SEC:
            self._give_up(st, now)
            return SafetyCommand(speed_cap=0.0, assist=ASSIST_FRONT_HOLD)
        return wait

    def _pass_side_clear(self, st: _SlotSafety, result: PerceptionResult, cam: CameraSpec) -> bool:
        """回避で抜ける側に、近づいてくる車・すぐ横の歩行者がいないか。"""
        track = st.tracks.get(cam.key)
        clear = True
        for det in result.detections:
            if det.cls not in SURROUND_CLASSES:
                continue
            fx, fy, dist = local_xy(det, cam)
            near_side = abs(fy) - HALF_WIDTH <= SIDE_CLEAR_M and -HALF_LENGTH - 3.0 <= fx <= HALF_LENGTH + 6.0
            coming = (
                det.cls is DetClass.VEHICLE
                and dist <= DETOUR_SIDE_RANGE_M
                and track is not None
                and track.rate > APPROACH_MPS
            )
            if near_side or coming:
                _mark(det, HAZARD_STOP)
                clear = False
        return clear

    def _detour(
        self,
        world: "World",
        slot: int,
        now: float,
        front: PerceptionResult | None,
    ) -> SafetyCommand:
        """障害物の脇を、経路を横へずらして抜ける。抜けたら徐々に経路へ戻す。"""
        st = self._state[slot]
        st.demand = (LEFT_CAMERA.key if st.offset > 0.0 else RIGHT_CAMERA.key,)
        arc = float(world.arc[slot])
        past = arc - st.pass_arc
        if past >= DETOUR_RETURN_M or now - st.since > DETOUR_MAX_SEC:
            self._set_mode(st, _MODE_DRIVE, now)
            st.blockers = []
            return SafetyCommand()
        ramp = 1.0 if past <= 0.0 else max(0.0, 1.0 - past / DETOUR_RETURN_M)
        offset = st.offset * ramp
        # ずらした帯に入ってくる対向車・歩行者・別の障害物の手前でも止まる
        cap, _blockers = self._front_hold(world, slot, front, offset, SURROUND_CLASSES)
        return SafetyCommand(
            offset=offset,
            speed_cap=min(DETOUR_SPEED_MPS, cap),
            override=True,
            assist=ASSIST_DETOUR,
        )

    def _plan_detour(self, world: "World", slot: int) -> tuple[float, float] | None:
        """障害物を避けて抜ける横ずらし [m] と、抜け終わる弧長。建物にかかるなら反対側、どちらも駄目なら None。"""
        st = self._state[slot]
        if not st.blockers:
            return None
        right = min(b.lat - b.radius - HALF_WIDTH - PASS_CLEARANCE_M for b in st.blockers)
        left = max(b.lat + b.radius + HALF_WIDTH + PASS_CLEARANCE_M for b in st.blockers)
        pass_arc = max(b.arc + b.radius for b in st.blockers) + HALF_LENGTH + 2.0
        arc = float(world.arc[slot])
        # 左側通行なので、まず右（道路の中央側）を試す
        for offset in (right, left):
            if abs(offset) > DETOUR_MAX_OFFSET_M:
                continue
            if _path_clear(world, slot, arc, pass_arc + DETOUR_RETURN_M, offset):
                return float(offset), float(pass_arc)
        return None

    def _give_up(self, st: _SlotSafety, now: float) -> None:
        self._set_mode(st, _MODE_DRIVE, now)
        st.retry_at = now + RETRY_SEC
        st.blockers = []
        st.stuck_since = -1.0


def _fresh_sight(
    occlusion: OcclusionResult | None, fresh: dict[str, PerceptionResult]
) -> tuple[float | None, float | None]:
    """左右のカメラの見通し距離のうち、撮ってから `FRESH_SEC` 以内のものだけ（古ければ None）。"""
    if occlusion is None:
        return None, None
    left = occlusion.los_left if LEFT_CAMERA.key in fresh else None
    right = occlusion.los_right if RIGHT_CAMERA.key in fresh else None
    return left, right


def _nearest(
    result: PerceptionResult, cam: CameraSpec, classes: tuple[DetClass, ...]
) -> tuple[float, Detection, float, float] | None:
    """そのカメラで一番近い検出 (距離, 検出, 自車座標 x, y)。"""
    best: tuple[float, Detection, float, float] | None = None
    for det in result.detections:
        if det.cls not in classes:
            continue
        fx, fy, dist = local_xy(det, cam)
        if best is None or dist < best[0]:
            best = (dist, det, fx, fy)
    return best


def _to_world(world: "World", slot: int, fx: float, fy: float) -> tuple[float, float]:
    """自車座標（前方 +x / 左 +y）を ENU へ。"""
    h = float(world.fleet.heading[slot])
    c, s = math.cos(h), math.sin(h)
    return (
        float(world.fleet.x[slot]) + fx * c - fy * s,
        float(world.fleet.y[slot]) + fx * s + fy * c,
    )


def _route_frame(
    world: "World", slot: int, wx: float, wy: float, lo: float, hi: float
) -> tuple[float, float] | None:
    """点を経路の [lo, hi] の区間へ写した (弧長, 左を正とした横位置)。区間から遠ければ None。"""
    state = world.slots[slot]
    route = state.route
    cum = state.route_cum
    k = route.shape[0]
    if k < 2:
        return None
    i0 = max(0, int(np.searchsorted(cum, lo, side="left")) - 1)
    i1 = min(k - 1, int(np.searchsorted(cum, hi, side="right")) + 1)
    if i1 <= i0:
        return None
    a = route[i0:i1].astype(np.float64)
    b = route[i0 + 1 : i1 + 1].astype(np.float64)
    ab = b - a
    len2 = np.maximum((ab * ab).sum(axis=1), 1e-9)
    t = np.clip(((wx - a[:, 0]) * ab[:, 0] + (wy - a[:, 1]) * ab[:, 1]) / len2, 0.0, 1.0)
    px = a[:, 0] + ab[:, 0] * t
    py = a[:, 1] + ab[:, 1] * t
    d2 = (wx - px) ** 2 + (wy - py) ** 2
    j = int(np.argmin(d2))
    if float(d2[j]) > ROUTE_MATCH_M * ROUTE_MATCH_M:
        return None
    length = math.sqrt(float(len2[j]))
    ux, uy = ab[j, 0] / length, ab[j, 1] / length
    lat = ux * (wy - py[j]) - uy * (wx - px[j])
    return float(cum[i0 + j]) + float(t[j]) * length, float(lat)


def _path_clear(world: "World", slot: int, lo: float, hi: float, offset: float) -> bool:
    """経路を `offset` だけ横へずらした帯（車幅 + 余裕）が建物にかからないか。"""
    state = world.slots[slot]
    route = state.route
    cum = state.route_cum
    if route.shape[0] < 2:
        return False
    marks = np.arange(lo, min(hi, float(cum[-1])) + 1e-6, 1.0, dtype=np.float64)
    if marks.size < 2:
        return False
    xs = np.interp(marks, cum, route[:, 0])
    ys = np.interp(marks, cum, route[:, 1])
    tx = np.gradient(xs)
    ty = np.gradient(ys)
    norm = np.maximum(np.hypot(tx, ty), 1e-6)
    nx, ny = -ty / norm, tx / norm
    grid = world.map_index.occupancy
    for side in (-(HALF_WIDTH + 0.3), 0.0, HALF_WIDTH + 0.3):
        px = xs + nx * (offset + side)
        py = ys + ny * (offset + side)
        if bool(np.any(grid.sample_building(px, py))):
            return False
    return True
