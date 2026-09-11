/**
 * フレーム間補間（memo 5章「学習頻度と描画頻度の関係」）
 *
 * サーバーは 20Hz、描画は 60fps。そのまま描くとカクつくので、
 * 前フレーム(prev)と最新フレーム(curr)の間を線形補間して滑らかに見せる。
 *
 * 表示は意図的に 1 フレーム分（既定 50ms）遅れる。
 * 「curr に向かって進む」外挿ではなく「prev → curr を追いかける」内挿にすることで、
 * サーバーの配信ジッタで車がガタつくのを防ぐ。
 */

import type { VehicleState } from '../types/protocol'
import { frameBuffer } from '../store/frameBuffer'

/** これ以上動いたら補間せず瞬間移動（respawn / スポーン扱い）[m] */
export const TELEPORT_DISTANCE_M = 20

/** 補間結果。3D 側はこれを毎フレーム読んでメッシュに反映する */
export interface VehiclePose {
  /** ENU x [m] */
  x: number
  /** ENU y [m] */
  y: number
  /** ラジアン。+x 軸から反時計回り */
  heading: number
  /** m/s */
  speed: number
  /** 前輪舵角 rad */
  steer: number
  active: boolean
  collided: boolean
  reachedGoal: boolean
  /** 目的地 [x, y]（ENU） */
  goalX: number
  goalY: number
  /** この呼び出しで瞬間移動したか（カメラ側で減衰を切るのに使う） */
  teleported: boolean
}

/** 角度差を (-pi, pi] に畳む。heading は最短回りで補間しなければならない */
export function angleDelta(from: number, to: number): number {
  const d = to - from
  return Math.atan2(Math.sin(d), Math.cos(d))
}

/** 角度の最短回り補間 */
export function lerpAngle(from: number, to: number, t: number): number {
  return from + angleDelta(from, to) * t
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

/**
 * 現在時刻における補間係数 alpha を求める。
 * @param renderPaused 一時停止中は補間を進めない（memo 5章: 一時停止は描画のみ）
 */
export function computeAlpha(nowMs: number, renderPaused: boolean): number {
  if (renderPaused) return 1
  const dt = frameBuffer.intervalMs
  if (dt <= 0) return 1
  const t = (nowMs - frameBuffer.currTime) / dt
  // 外挿しない。遅延到着で 1 を超えたら最新値で止める
  return t <= 0 ? 0 : t >= 1 ? 1 : t
}

function findVehicle(list: VehicleState[] | undefined, id: number): VehicleState | undefined {
  if (!list) return undefined
  // vehicles は常に全スロット分・スロット番号順で来る想定なので、まず添字を試す
  const guess = list[id]
  if (guess && guess.id === id) return guess
  for (const v of list) if (v.id === id) return v
  return undefined
}

/**
 * スロット id の車両姿勢を補間して out に書き込む。
 * @returns 描画すべきなら true（非アクティブ／データ無しなら false）
 */
export function sampleVehicle(
  id: number,
  alpha: number,
  out: VehiclePose,
): boolean {
  const curr = frameBuffer.curr
  if (!curr) return false

  const cv = findVehicle(curr.vehicles, id)
  if (!cv || !cv.active) return false

  const pv = findVehicle(frameBuffer.prev?.vehicles, id)

  out.active = true
  out.collided = cv.collided
  out.reachedGoal = cv.reachedGoal
  out.goalX = cv.goal[0]
  out.goalY = cv.goal[1]
  out.teleported = false

  // 前フレームが無い / 直前まで非アクティブ（= 今スポーンした）なら補間しない
  if (!pv || !pv.active) {
    out.x = cv.x
    out.y = cv.y
    out.heading = cv.heading
    out.speed = cv.speed
    out.steer = cv.steer
    out.teleported = true
    return true
  }

  // 大きく飛んだら respawn とみなして瞬間移動させる
  const dx = cv.x - pv.x
  const dy = cv.y - pv.y
  if (dx * dx + dy * dy > TELEPORT_DISTANCE_M * TELEPORT_DISTANCE_M) {
    out.x = cv.x
    out.y = cv.y
    out.heading = cv.heading
    out.speed = cv.speed
    out.steer = cv.steer
    out.teleported = true
    return true
  }

  out.x = lerp(pv.x, cv.x, alpha)
  out.y = lerp(pv.y, cv.y, alpha)
  out.heading = lerpAngle(pv.heading, cv.heading, alpha)
  out.speed = lerp(pv.speed, cv.speed, alpha)
  out.steer = lerp(pv.steer, cv.steer, alpha)
  return true
}

/** 使い回し用の空ポーズを作る（毎フレームのアロケーションを避ける） */
export function createPose(): VehiclePose {
  return {
    x: 0,
    y: 0,
    heading: 0,
    speed: 0,
    steer: 0,
    active: false,
    collided: false,
    reachedGoal: false,
    goalX: 0,
    goalY: 0,
    teleported: false,
  }
}
