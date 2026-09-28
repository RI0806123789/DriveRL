/** 後退の予測ガイド線（純粋関数）。後ろのバンパーの左右の角が、いまの舵角のまま下がったときに通る線。 */

import { VEHICLE_LENGTH, VEHICLE_WIDTH } from './vehicleGeometry.ts'
import { WHEELBASE_M } from './vehicleMotion.ts'

/** ガイド線を引く長さ [m]（後ろのバンパーから） */
export const GUIDE_LENGTH_M = 5
export const GUIDE_STEP_M = 0.25
/** 距離の目盛り [m]。手前ほど危ないので色を変える */
export const GUIDE_MARKS_M = [1, 2, 3] as const
/** 線の太さ [m] */
export const GUIDE_WIDTH_M = 0.07

/** 車両ローカル（前方 +x / 左 +y）[m] */
export interface GuidePoint {
  x: number
  y: number
}

export interface ReverseGuide {
  left: GuidePoint[]
  right: GuidePoint[]
  /** 各点の、後ろのバンパーから下がった距離 [m] */
  distances: number[]
}

/** 後ろのバンパーから `distance` 下がった所の色。1m までは赤、2m までは黄、それより先は緑 */
export function guideColor(distance: number): string {
  if (distance <= GUIDE_MARKS_M[0]) return '#ff3b30'
  if (distance <= GUIDE_MARKS_M[1]) return '#ffc107'
  return '#34c759'
}

/** 舵角 `steer` [rad] のまま下がったときの、後ろのバンパーの左右の角の軌跡（キネマティック自転車モデル） */
export function reverseGuideLines(
  steer: number,
  length: number = GUIDE_LENGTH_M,
  step: number = GUIDE_STEP_M,
): ReverseGuide {
  const curvature = Math.tan(steer) / WHEELBASE_M
  const halfL = VEHICLE_LENGTH / 2
  const halfW = VEHICLE_WIDTH / 2
  const left: GuidePoint[] = []
  const right: GuidePoint[] = []
  const distances: number[] = []
  const count = Math.max(1, Math.round(length / step))
  for (let i = 0; i <= count; i += 1) {
    const back = (length * i) / count
    // 車体の中心が弧長 -back だけ下がった姿勢（バックエンドの車両モデルと同じく中心で積分）
    const d = -back
    const theta = curvature * d
    const cx = Math.abs(curvature) > 1e-6 ? Math.sin(theta) / curvature : d
    const cy = Math.abs(curvature) > 1e-6 ? (1 - Math.cos(theta)) / curvature : 0
    const c = Math.cos(theta)
    const s = Math.sin(theta)
    left.push({ x: cx + c * -halfL - s * halfW, y: cy + s * -halfL + c * halfW })
    right.push({ x: cx + c * -halfL + s * halfW, y: cy + s * -halfL - c * halfW })
    distances.push(back)
  }
  return { left, right, distances }
}
