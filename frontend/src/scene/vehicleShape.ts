/** 車体の外形（平面形・断面・ホイールアーチ）を式で決める。**React から切り離した純粋モジュール。** */

import type { Vec3 } from './meshBuilder.ts'
import {
  BACKLIGHT_BOTTOM_X,
  BELT_Y,
  BODY_FRONT_X,
  BODY_HALF_W,
  BODY_REAR_X,
  WHEEL_OFFSETS,
  WHEEL_RADIUS,
  WINDSHIELD_BOTTOM_X,
} from './vehicleGeometry.ts'

/** 平面形の角の丸み [m] */
export const FRONT_CORNER_R = 0.3
export const REAR_CORNER_R = 0.3

const SIDE_FRONT_X = BODY_FRONT_X - FRONT_CORNER_R
const SIDE_REAR_X = BODY_REAR_X + REAR_CORNER_R
const L_FRONT = BODY_HALF_W - FRONT_CORNER_R
const L_FRONT_ARC = (Math.PI / 2) * FRONT_CORNER_R
const L_SIDE = SIDE_FRONT_X - SIDE_REAR_X
const L_REAR_ARC = (Math.PI / 2) * REAR_CORNER_R
const L_REAR = BODY_HALF_W - REAR_CORNER_R

/** 外形の右半分を、前の中心から後ろの中心までたどった長さ [m] */
export const OUTLINE_LENGTH = L_FRONT + L_FRONT_ARC + L_SIDE + L_REAR_ARC + L_REAR

/** 外形の右半分の 1 点。`front` / `rear` は前面・後面らしさ（断面の混ぜ具合） */
export interface OutlinePoint {
  readonly x: number
  readonly z: number
  /** 外向きの法線（平面内） */
  readonly nx: number
  readonly nz: number
  readonly front: number
  readonly rear: number
}

/** 角の中で前面から側面へ滑らかに移る重み */
function blendIn(t: number): number {
  const c = Math.min(1, Math.max(0, t))
  return 0.5 + 0.5 * Math.cos(Math.PI * c)
}

/** 前の中心から周長 s [m] だけ右回りにたどった点（右半分）。 */
export function outlineAt(s: number): OutlinePoint {
  let r = Math.min(Math.max(s, 0), OUTLINE_LENGTH)
  if (r <= L_FRONT) return { x: BODY_FRONT_X, z: r, nx: 1, nz: 0, front: 1, rear: 0 }
  r -= L_FRONT
  if (r <= L_FRONT_ARC) {
    const a = r / FRONT_CORNER_R
    return {
      x: SIDE_FRONT_X + Math.cos(a) * FRONT_CORNER_R,
      z: L_FRONT + Math.sin(a) * FRONT_CORNER_R,
      nx: Math.cos(a),
      nz: Math.sin(a),
      front: blendIn(a / (Math.PI / 2)),
      rear: 0,
    }
  }
  r -= L_FRONT_ARC
  if (r <= L_SIDE) return { x: SIDE_FRONT_X - r, z: BODY_HALF_W, nx: 0, nz: 1, front: 0, rear: 0 }
  r -= L_SIDE
  if (r <= L_REAR_ARC) {
    const a = r / REAR_CORNER_R
    return {
      x: SIDE_REAR_X - Math.sin(a) * REAR_CORNER_R,
      z: L_REAR + Math.cos(a) * REAR_CORNER_R,
      nx: -Math.sin(a),
      nz: Math.cos(a),
      front: 0,
      rear: 1 - blendIn(a / (Math.PI / 2)),
    }
  }
  r -= L_REAR_ARC
  return { x: BODY_REAR_X, z: Math.max(0, L_REAR - r), nx: -1, nz: 0, front: 0, rear: 1 }
}

/** 側面の直線部で X に当たる周長 */
export function sOfSideX(x: number): number {
  return L_FRONT + L_FRONT_ARC + (SIDE_FRONT_X - x)
}

/** 前面の直線部・角で |z| に当たる周長（前面は z そのもの） */
export function sOfFrontZ(z: number): number {
  if (z <= L_FRONT) return z
  const a = Math.asin(Math.min(1, (z - L_FRONT) / FRONT_CORNER_R))
  return L_FRONT + a * FRONT_CORNER_R
}

/** 後面の直線部・角で |z| に当たる周長 */
export function sOfRearZ(z: number): number {
  const base = L_FRONT + L_FRONT_ARC + L_SIDE
  if (z <= L_REAR) return base + L_REAR_ARC + (L_REAR - z)
  const a = Math.acos(Math.min(1, (z - L_REAR) / REAR_CORNER_R))
  return base + a * REAR_CORNER_R
}

/** 周長の区切り（外形の直線と円弧の境目） */
export const OUTLINE_BREAKS = {
  frontArcStart: L_FRONT,
  sideStart: L_FRONT + L_FRONT_ARC,
  rearArcStart: L_FRONT + L_FRONT_ARC + L_SIDE,
  rearStart: L_FRONT + L_FRONT_ARC + L_SIDE + L_REAR_ARC,
} as const

/** 制御点をなめらかに結ぶ（Catmull-Rom。端では一定値） */
function curve(ys: readonly number[], vs: readonly number[], y: number): number {
  const n = ys.length
  if (y <= ys[0]) return vs[0]
  if (y >= ys[n - 1]) return vs[n - 1]
  let i = 0
  while (i < n - 2 && y > ys[i + 1]) i++
  const t = (y - ys[i]) / (ys[i + 1] - ys[i])
  const p0 = vs[Math.max(0, i - 1)]
  const p1 = vs[i]
  const p2 = vs[i + 1]
  const p3 = vs[Math.min(n - 1, i + 2)]
  const t2 = t * t
  const t3 = t2 * t
  return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
}

/** 断面。高さ y で外形からどれだけ内側へ入るか [m]（側面・前面・後面） */
const SIDE_Y = [0.2, 0.28, 0.4, 0.55, 0.7, 0.84, 0.93]
const SIDE_IN = [0.055, 0.022, 0.006, 0.0, 0.004, 0.014, 0.022]
const FRONT_Y = [0.24, 0.3, 0.42, 0.54, 0.66, 0.74]
const FRONT_IN = [0.075, 0.03, 0.006, 0.012, 0.026, 0.035]
const REAR_Y = [0.28, 0.36, 0.46, 0.6, 0.76, 0.86]
const REAR_IN = [0.07, 0.015, 0.004, 0.012, 0.024, 0.032]

export function insetAt(p: OutlinePoint, y: number): number {
  const side = 1 - p.front - p.rear
  return (
    side * curve(SIDE_Y, SIDE_IN, y) +
    p.front * curve(FRONT_Y, FRONT_IN, y) +
    p.rear * curve(REAR_Y, REAR_IN, y)
  )
}

/** 車輪の上の切り欠き（ホイールアーチ）。 */
export const ARCH_HALF_SPAN = 0.42
export const ARCH_TOP_Y = WHEEL_RADIUS + 0.43
export const ROCKER_Y = 0.2
const ARCH_POWER = 2.4
const ARCH_CENTERS = [...new Set(WHEEL_OFFSETS.map((w) => w.position[0]))]

/** X での切り欠きの上端 [m]。アーチの外なら null */
export function archY(x: number): number | null {
  for (const cx of ARCH_CENTERS) {
    const d = Math.abs(x - cx) / ARCH_HALF_SPAN
    if (d >= 1) continue
    return ROCKER_Y + (ARCH_TOP_Y - ROCKER_Y) * Math.pow(1 - Math.pow(d, ARCH_POWER), 1 / ARCH_POWER)
  }
  return null
}

/** フェンダーの張り出し [m]。アーチの周りだけ外へ膨らませる（外接寸法の内側で） */
export const FLARE = 0.013
export function flareAt(x: number, y: number): number {
  let best = 0
  for (const cx of ARCH_CENTERS) {
    const d = Math.abs(x - cx) / 0.6
    if (d >= 1 || y < 0.28 || y > 0.9) continue
    const along = Math.cos((Math.PI / 2) * d) ** 2
    const up = Math.sin((Math.PI * (y - 0.28)) / 0.62)
    best = Math.max(best, FLARE * along * up)
  }
  return best
}

/** ボンネットの見切り（前端）とトランクの後端の高さ [m] */
export const HOOD_FRONT_Y = 0.82
export const TRUNK_REAR_Y = 0.95

/** ボンネット（外周）の高さ。カウルで BELT_Y、前へ行くほど下がる */
export function hoodY(x: number): number {
  const t = Math.min(1, Math.max(0, (x - WINDSHIELD_BOTTOM_X) / (BODY_FRONT_X - WINDSHIELD_BOTTOM_X)))
  return BELT_Y - (BELT_Y - HOOD_FRONT_Y) * Math.pow(t, 1.5)
}

/** トランクの高さ。リアガラスの下端からほぼ平らに伸び、後端で丸く落ちる */
export function trunkY(x: number): number {
  const top = BELT_Y + 0.005
  const t = Math.min(1, Math.max(0, (BACKLIGHT_BOTTOM_X - x) / (BACKLIGHT_BOTTOM_X - BODY_REAR_X)))
  return top - (top - TRUNK_REAR_Y) * t * t
}

/** 外板の上端（肩の丸みの終わり）の高さ */
export function topAt(p: OutlinePoint): number {
  if (p.x >= WINDSHIELD_BOTTOM_X) return hoodY(p.x)
  if (p.x <= BACKLIGHT_BOTTOM_X) return trunkY(p.x)
  return BELT_Y
}

/** 外板の下端の高さ（アーチの中ではアーチの縁） */
export function bottomAt(p: OutlinePoint): number {
  const side = 1 - p.front - p.rear
  const base = side * ROCKER_Y + p.front * 0.24 + p.rear * 0.28
  if (p.front > 0 || p.rear > 0) return base
  return Math.max(base, archY(p.x) ?? base)
}

/** 肩（側面から上面へ回り込む丸み）の半径 [m] */
export function shoulderRadiusAt(p: OutlinePoint): number {
  const side = 1 - p.front - p.rear
  return side * 0.07 + p.front * 0.08 + p.rear * 0.09
}

/** 外形から内側へ `inset`、高さ y の点（右半分） */
export function wallPoint(p: OutlinePoint, y: number, inset: number): [number, number, number] {
  return [p.x - p.nx * inset, y, p.z - p.nz * inset]
}

/** 下端から肩の丸みの終わりまでの断面（右半分）。 */
export function sectionPoints(
  s: number,
  lower: readonly number[],
  shoulderSteps: number,
): Vec3[] {
  const p = outlineAt(s)
  const bottom = bottomAt(p)
  const top = topAt(p)
  const r = shoulderRadiusAt(p)
  const start = top - r
  const out: Vec3[] = []
  for (const f of lower) {
    const y = bottom + (start - bottom) * f
    out.push(wallPoint(p, y, insetAt(p, y) - flareAt(p.x, y) * (1 - p.front - p.rear)))
  }
  const base = insetAt(p, start) - flareAt(p.x, start) * (1 - p.front - p.rear)
  for (let k = 1; k <= shoulderSteps; k++) {
    const a = (Math.PI / 2) * (k / shoulderSteps)
    out.push(wallPoint(p, start + Math.sin(a) * r, base + (1 - Math.cos(a)) * r))
  }
  return out
}

/** 外板の面上の点と法線（灯火やハンドルを面に沿わせて置くのに使う）。肩より下だけ */
export function surfaceAt(s: number, y: number): { point: Vec3; normal: Vec3 } {
  const p = outlineAt(s)
  const k = 1 - p.front - p.rear
  const inset = (yy: number) => insetAt(p, yy) - flareAt(p.x, yy) * k
  const point = wallPoint(p, y, inset(y))
  const dy = 0.005
  const slope = (inset(y + dy) - inset(y - dy)) / (2 * dy)
  const n: Vec3 = [p.nx, slope, p.nz]
  const l = Math.hypot(n[0], n[1], n[2])
  return { point, normal: [n[0] / l, n[1] / l, n[2] / l] }
}

/** 左右反転（z の符号を変える） */
export function mirrorZ(p: Vec3): Vec3 {
  return [p[0], p[1], -p[2]]
}
