/** ワイパーの動き（止める／間欠／連続）と、フロントガラスのどこをいつ拭いたか。**React から切り離した純粋モジュール。** */

import * as THREE from 'three'
import { MeshBuilder, boxGeometry, type Vec3 } from './meshBuilder.ts'
import {
  BELT_Y,
  GLASS_BOTTOM_HALF_W,
  GLASS_TOP_HALF_W,
  GLASS_TOP_Y,
  WINDSHIELD_BOTTOM_X,
  WINDSHIELD_TOP_X,
} from './vehicleGeometry.ts'

export const WIPER_OFF = 0
export const WIPER_INT = 1
export const WIPER_LO = 2
export const WIPER_HI = 3

/** 雨の強さのしきい値（入る値・出る値）。行き来しないよう幅を持たせる。小雨（0.35）は間欠、雨（0.85）は低速 */
const MODE_ON: readonly number[] = [0, 0.06, 0.5, 0.9]
const MODE_OFF: readonly number[] = [0, 0.03, 0.42, 0.84]

/** 1 往復の時間 [s]（止める・間欠・低速・高速） */
export const WIPER_PERIOD: readonly number[] = [0, 1.4, 1.4, 0.95]
/** 間欠のときの休み [s] */
export const WIPER_INT_PAUSE = 3.0
/** 振り上げる角度 [rad] */
export const WIPER_SWEEP = THREE.MathUtils.degToRad(84)

/** 雨の強さといまの段から、次の段を決める */
export function wiperModeFor(rain: number, current: number): number {
  let mode = current
  while (mode < WIPER_HI && rain >= MODE_ON[mode + 1]) mode++
  while (mode > WIPER_OFF && rain < MODE_OFF[mode]) mode--
  return mode
}

/** 1 台ぶんの状態（時刻はクライアントの時計 [s]） */
export interface WiperState {
  mode: number
  /** いま（または直前）の往復の始まりと長さ */
  start: number
  period: number
  /** その 1 つ前の往復 */
  prevStart: number
  prevPeriod: number
  /** 最初に動き出すまでのずれ [s]。全車がそろって動くと作り物に見える */
  offset: number
}

export function createWiperState(offset: number): WiperState {
  return { mode: WIPER_OFF, start: -Infinity, period: 0, prevStart: -Infinity, prevPeriod: 0, offset }
}

/** 全車のワイパー。**進めるのは `Vehicles` の 1 か所だけ**で、水滴の描画は読むだけ */
export const vehicleWipers: { states: WiperState[] } = { states: [] }

/** 台数に合わせて入れ物を用意する。動き出しのずれは車ごとに変える */
export function ensureWiperSlots(count: number): void {
  const list = vehicleWipers.states
  while (list.length < count) list.push(createWiperState((list.length * 0.37) % 1.1))
  list.length = count
}

/** 往復の途中なら最後まで振り切ってから、次の段の動きに移る。 */
export function advanceWiper(s: WiperState, now: number, rain: number): void {
  s.mode = wiperModeFor(rain, s.mode)
  if (now < s.start + s.period) return
  if (s.mode === WIPER_OFF) return
  const period = WIPER_PERIOD[s.mode]
  let next: number
  if (!Number.isFinite(s.start)) {
    next = now + s.offset
  } else {
    const ready = s.start + s.period + (s.mode === WIPER_INT ? WIPER_INT_PAUSE : 0)
    if (now < ready) return
    next = now - ready > period ? now : ready
  }
  s.prevStart = s.start
  s.prevPeriod = s.period
  s.start = next
  s.period = period
}

/** 往復の中での角度。0 が止まっている位置（ガラスの下端に沿って寝ている） */
export function strokeAngle(elapsed: number, period: number): number {
  if (!(period > 0) || elapsed < 0 || elapsed >= period) return 0
  return (WIPER_SWEEP * (1 - Math.cos((2 * Math.PI * elapsed) / period))) / 2
}

export function wiperAngle(s: WiperState, now: number): number {
  return strokeAngle(now - s.start, s.period)
}

/** 往復の中で、角度 φ を上り・下りで通る時刻 [s]（往復の始まりから） */
function passTimes(phi: number, period: number): [number, number] {
  const f = Math.min(1, Math.max(0, phi / WIPER_SWEEP))
  const up = (period / (2 * Math.PI)) * Math.acos(1 - 2 * f)
  return [up, period - up]
}

/** 角度 φ の位置を、最後にブレードが通ってからの時間 [s]。一度も拭いていなければ Infinity */
export function lastWipeAge(s: WiperState, now: number, phi: number): number {
  if (phi < 0 || phi > WIPER_SWEEP) return Infinity
  if (Number.isFinite(s.start) && s.period > 0) {
    const e = now - s.start
    const [up, down] = passTimes(phi, s.period)
    if (e >= down) return e - down
    if (e >= up) return e - up
  }
  if (Number.isFinite(s.prevStart) && s.prevPeriod > 0) {
    const [, down] = passTimes(phi, s.prevPeriod)
    return now - s.prevStart - down
  }
  return Infinity
}

/** フロントガラスの座標。下端から斜面に沿った距離 s [m] と横の z [m] */
export const WINDSHIELD_SLOPE: readonly [number, number] = (() => {
  const dx = WINDSHIELD_TOP_X - WINDSHIELD_BOTTOM_X
  const dy = GLASS_TOP_Y - BELT_Y
  const l = Math.hypot(dx, dy)
  return [dx / l, dy / l]
})()
export const WINDSHIELD_LENGTH = Math.hypot(WINDSHIELD_TOP_X - WINDSHIELD_BOTTOM_X, GLASS_TOP_Y - BELT_Y)
/** 外向きの法線（XY 平面内） */
export const WINDSHIELD_NORMAL: readonly [number, number] = [WINDSHIELD_SLOPE[1], -WINDSHIELD_SLOPE[0]]
/** ガラスの中ほどの膨らみ [m]（`vehicleBody` のフロントガラスと同じ値） */
export const WINDSHIELD_BULGE = 0.012

/** 斜面に沿って s の位置でのガラスの半幅 */
export function windshieldHalfWidth(s: number): number {
  const t = s / WINDSHIELD_LENGTH
  return GLASS_BOTTOM_HALF_W + (GLASS_TOP_HALF_W - GLASS_BOTTOM_HALF_W) * t
}

/** ガラスの座標 (s, z) を車両ローカルの点へ。`lift` だけ外へ（負なら内へ） */
export function windshieldPoint(s: number, z: number, lift: number): Vec3 {
  const half = windshieldHalfWidth(s)
  const across = z / half
  const push = lift + WINDSHIELD_BULGE * Math.max(0, 1 - across * across)
  return [
    WINDSHIELD_BOTTOM_X + WINDSHIELD_SLOPE[0] * s + WINDSHIELD_NORMAL[0] * push,
    BELT_Y + WINDSHIELD_SLOPE[1] * s + WINDSHIELD_NORMAL[1] * push,
    z,
  ]
}

/** ワイパーの付け根（ガラスの座標）。運転席側と助手席側。どちらも右（+Z）へ寝かせて止め、上へ振り上げる */
export const WIPER_PIVOTS: ReadonlyArray<readonly [number, number]> = [
  [0.03, 0.02],
  [0.012, -0.6],
]
/** ブレードが拭く範囲（付け根からの距離 [m]） */
export const WIPER_REACH: readonly [number, number] = [0.1, 0.7]
/** ワイパーをガラスから浮かせる量 [m]（膨らみの最大より上） */
export const WIPER_LIFT = WINDSHIELD_BULGE + 0.002

/** ワイパー 1 本（アームとブレード）。付け根が原点、止めたときの向き（ブレードの長さ方向）が +X、ガラスの法線が +Y */
export function makeWiperGeometry(): THREE.BufferGeometry {
  const b = new MeshBuilder()
  const parts = [
    boxGeometry([-0.012, 0.004, -0.012], [0.012, 0.03, 0.012]),
    boxGeometry([0, 0.016, -0.007], [0.43, 0.026, 0.007]),
    boxGeometry([WIPER_REACH[0], 0.001, -0.009], [WIPER_REACH[1], 0.015, 0.009]),
    boxGeometry([0.38, 0.012, -0.011], [0.44, 0.024, 0.011]),
  ]
  for (const g of parts) {
    b.geometry(g)
    g.dispose()
  }
  return b.build()
}

/** ワイパー 1 本の姿勢（車両ローカル）。`angle` は振り上げた角度 */
export function composeWiperLocal(index: number, angle: number, out: THREE.Matrix4): THREE.Matrix4 {
  const [s, z] = WIPER_PIVOTS[index]
  const p = windshieldPoint(s, z, WIPER_LIFT - WINDSHIELD_BULGE * Math.max(0, 1 - (z / windshieldHalfWidth(s)) ** 2))
  const x = new THREE.Vector3(0, 0, 1)
  const y = new THREE.Vector3(WINDSHIELD_NORMAL[0], WINDSHIELD_NORMAL[1], 0)
  const zAxis = new THREE.Vector3().crossVectors(x, y)
  out.makeBasis(x, y, zAxis)
  out.multiply(new THREE.Matrix4().makeRotationY(-angle))
  out.setPosition(p[0], p[1], p[2])
  return out
}

/** 水滴。セル 1 つに 1 粒まで */
export const DROP_CELL_M = 0.02
/** 雨が降り続けたとき、粒の入るセルの割合 */
export const DROP_CHANCE = 0.72
/** 粒の半径（セルの大きさに対する比） */
export const DROP_RADIUS: readonly [number, number] = [0.14, 0.36]
/** 粒の不透明度 */
export const DROP_OPACITY = 0.5
/** 雨の強さ 1 で、拭いた所がいっぱいまで粒で埋まる時間 [s] */
export const DROP_FILL_SEC = 5

/** 粒が出そろうまでの時間 [s]（弱い雨ほど遅い） */
export function dropFillSec(rain: number): number {
  return DROP_FILL_SEC / Math.max(rain, 0.12)
}

/** 拭いてから `age` 秒で、視界のどれだけが粒に覆われるか（不透明度を掛けた割合） */
export function dropCoverage(age: number, rain: number): number {
  const filled = Math.min(1, Math.max(0, age / dropFillSec(rain)))
  const [r0, r1] = DROP_RADIUS
  const meanArea = (Math.PI * (r1 ** 3 - r0 ** 3)) / (3 * (r1 - r0))
  return DROP_CHANCE * filled * meanArea * DROP_OPACITY
}
