/** 車両のライト（前照灯・尾灯・制動灯・方向指示器）の配置と点灯条件。**純粋モジュール。** */

import * as THREE from 'three'
import type { Vec3 } from './meshBuilder.ts'
import {
  BACKLIGHT_BOTTOM_X,
  BACKLIGHT_TOP_X,
  BELT_Y,
  GLASS_TOP_Y,
} from './vehicleGeometry.ts'
import { sOfFrontZ, sOfRearZ, surfaceAt } from './vehicleShape.ts'

/** ライトの種類 */
export const LIGHT_HEAD = 0
export const LIGHT_TAIL = 1
export const LIGHT_TURN = 2
/** ハイマウントストップランプ（制動灯と同じ指令で点く） */
export const LIGHT_STOP = 3

/** 1 灯ぶんの取り付け。レンズは向き `facing` を +X、`up` を +Y にとった箱を `size` に伸ばしたもの */
export interface LightSlot {
  readonly position: Vec3
  readonly facing: Vec3
  readonly up: Vec3
  /** [奥行き, 高さ, 幅] [m] */
  readonly size: Vec3
  readonly kind: number
  /** 左右。左が -1・中央が 0 */
  readonly side: -1 | 0 | 1
}

/** レンズを面からどれだけ浮かせるか [m]（中心の位置。奥行きの半分より小さくして面へ食い込ませる） */
const LENS_LIFT = 0.006

/** 外板の上の 2 点を結ぶ弦にレンズを渡す。 */
function onBody(
  s0: number,
  s1: number,
  y: number,
  height: number,
  depth: number,
  kind: number,
  side: 1 | -1,
): LightSlot {
  const a = surfaceAt(s0, y)
  const b = surfaceAt(s1, y)
  const mid = surfaceAt((s0 + s1) / 2, y)
  const nx = mid.normal[0]
  const ny = mid.normal[1]
  const nz = mid.normal[2]
  const cx = (a.point[0] + b.point[0]) / 2
  const cz = (a.point[2] + b.point[2]) / 2
  const width = Math.hypot(b.point[0] - a.point[0], b.point[2] - a.point[2])
  const z = cz + nz * LENS_LIFT
  return {
    position: [cx + nx * LENS_LIFT, y, side * z],
    facing: [nx, ny, side * nz],
    up: [0, 1, 0],
    size: [depth, height, width],
    kind,
    side,
  }
}

/** 右側の灯火。左は鏡に写す */
function rightSide(): LightSlot[] {
  return [
    onBody(sOfFrontZ(0.36), sOfFrontZ(0.655), 0.675, 0.1, 0.035, LIGHT_HEAD, 1),
    onBody(sOfFrontZ(0.665), sOfFrontZ(0.795), 0.672, 0.065, 0.03, LIGHT_TURN, 1),
    onBody(sOfRearZ(0.42), sOfRearZ(0.665), 0.79, 0.1, 0.035, LIGHT_TAIL, 1),
    onBody(sOfRearZ(0.675), sOfRearZ(0.8), 0.79, 0.1, 0.03, LIGHT_TURN, 1),
  ]
}

/** ドアミラーの外側の面（`vehicleBody` のミラーと同じ寸法から出す） */
export const MIRROR = {
  /** 筐体の外接箱 [x0, x1] / [y0, y1] / [z0, z1]（右側） */
  x: [0.86, 1.06] as const,
  y: [1.0, 1.12] as const,
  z: [0.805, 0.892] as const,
} as const

/** ハイマウントストップランプ。リアガラス上端の外側、屋根の縁のすぐ下 */
function stopLamp(): LightSlot {
  const dx = BACKLIGHT_TOP_X - BACKLIGHT_BOTTOM_X
  const dy = GLASS_TOP_Y - BELT_Y
  const len = Math.hypot(dx, dy)
  const along: Vec3 = [dx / len, dy / len, 0]
  const normal: Vec3 = [-along[1], along[0], 0]
  const back = 0.045
  return {
    position: [
      BACKLIGHT_TOP_X - along[0] * back + normal[0] * 0.009,
      GLASS_TOP_Y - along[1] * back + normal[1] * 0.009,
      0,
    ],
    facing: normal,
    up: along,
    size: [0.014, 0.024, 0.32],
    kind: LIGHT_STOP,
    side: 0,
  }
}

function mirrored(slot: LightSlot): LightSlot {
  return {
    ...slot,
    position: [slot.position[0], slot.position[1], -slot.position[2]],
    facing: [slot.facing[0], slot.facing[1], -slot.facing[2]],
    side: -slot.side as -1 | 1,
  }
}

/** ドアミラーのサイドターンランプ（右） */
function mirrorRepeater(): LightSlot {
  return {
    position: [(MIRROR.x[0] + MIRROR.x[1]) / 2 + 0.03, MIRROR.y[0] + 0.02, MIRROR.z[1] - 0.016],
    facing: [0.35, -0.1, 0.93],
    up: [0, 1, 0],
    size: [0.012, 0.018, 0.08],
    kind: LIGHT_TURN,
    side: 1,
  }
}

/** 1 台ぶんのライト。**並びを変えたら verify:vehicles の灯火の検査も直すこと** */
export const LIGHT_SLOTS: ReadonlyArray<LightSlot> = (() => {
  const right = [...rightSide(), mirrorRepeater()]
  const out: LightSlot[] = []
  for (const slot of right) {
    out.push(mirrored(slot), slot)
  }
  out.push(stopLamp())
  return out
})()

/** 1 台あたりのライトの数 */
export const LIGHTS_PER_VEHICLE = LIGHT_SLOTS.length

/** 方向指示器の点滅周期 [Hz]。 */
export const BLINK_HZ = 1.5

/** 前照灯を点ける天候のしきい値。小雨（rain 0.35）から点く */
export const HEADLIGHT_RAIN = 0.25
export const HEADLIGHT_FOG = 0.1

/** 点滅の位相。true の間が点灯 */
export function blinkOn(nowMs: number): boolean {
  return ((nowMs * BLINK_HZ) / 1000) % 1 < 0.5
}

/** 前照灯を点けるか。 */
export function headlightsOn(rain: number, fog: number, night: boolean): boolean {
  return night || rain >= HEADLIGHT_RAIN || fog >= HEADLIGHT_FOG
}

/** 1 台ぶんの点灯状態 */
export interface LightState {
  /** 前照灯 */
  head: boolean
  /** 尾灯（前照灯と連動する常時灯。制動灯とは別物） */
  tail: boolean
  /** 制動灯 */
  brake: boolean
  left: boolean
  right: boolean
}

export interface LightInput {
  braking: boolean
  /** -1=左 / 0=消灯 / +1=右 */
  turnSignal: number
  /** 乗降を待っている間は左右同時に点滅させる */
  hazard: boolean
  headlights: boolean
  /** `blinkOn()` の値 */
  blink: boolean
}

/** 入力から点灯状態を決める。**ハザードは方向指示器より優先する。** */
export function lightStateFor(input: LightInput): LightState {
  const blinking = input.blink
  return {
    head: input.headlights,
    tail: input.headlights,
    brake: input.braking,
    left: blinking && (input.hazard || input.turnSignal < 0),
    right: blinking && (input.hazard || input.turnSignal > 0),
  }
}

/** 尾灯を点けたときの明るさ（制動灯の 1.0 と見分けられるように暗くする） */
export const TAIL_LEVEL = 0.28

/** ライト 1 個の明るさ 0.0〜1.0。消灯でも尾灯はうっすら点く */
export function lightIntensity(slotIndex: number, state: LightState): number {
  const slot = LIGHT_SLOTS[slotIndex]
  if (slot.kind === LIGHT_HEAD) return state.head ? 1 : 0
  if (slot.kind === LIGHT_STOP) return state.brake ? 1 : 0
  if (slot.kind === LIGHT_TAIL) {
    if (state.brake) return 1
    return state.tail ? TAIL_LEVEL : 0
  }
  return (slot.side < 0 ? state.left : state.right) ? 1 : 0
}

/** レンズ 1 枚（角を丸めた板）。**寸法 1 の箱**で、+X が前面。インスタンスの行列で各灯の大きさへ伸ばす */
export function makeLampLensGeometry(): THREE.BufferGeometry {
  const r = 0.28
  const shape = new THREE.Shape()
  shape.moveTo(-0.5 + r, -0.5)
  shape.lineTo(0.5 - r, -0.5)
  shape.quadraticCurveTo(0.5, -0.5, 0.5, -0.5 + r)
  shape.lineTo(0.5, 0.5 - r)
  shape.quadraticCurveTo(0.5, 0.5, 0.5 - r, 0.5)
  shape.lineTo(-0.5 + r, 0.5)
  shape.quadraticCurveTo(-0.5, 0.5, -0.5, 0.5 - r)
  shape.lineTo(-0.5, -0.5 + r)
  shape.quadraticCurveTo(-0.5, -0.5, -0.5 + r, -0.5)
  const g = new THREE.ExtrudeGeometry(shape, { depth: 1, bevelEnabled: false, curveSegments: 2 })
  g.translate(0, 0, -0.5)
  // 押し出しの向き（+Z）を前（+X）へ、形の横（X）を幅（Z）へ
  g.rotateY(Math.PI / 2)
  g.computeVertexNormals()
  return g
}

const lampBasis = new THREE.Matrix4()
const lampScale = new THREE.Vector3()
const lampX = new THREE.Vector3()
const lampY = new THREE.Vector3()
const lampZ = new THREE.Vector3()

/** 灯火 1 個の行列。`body` は車体（傾きを掛けたもの） */
export function composeLampMatrix(body: THREE.Matrix4, slot: LightSlot, out: THREE.Matrix4): THREE.Matrix4 {
  lampX.set(slot.facing[0], slot.facing[1], slot.facing[2]).normalize()
  lampZ.set(slot.up[0], slot.up[1], slot.up[2]).cross(lampX).negate().normalize()
  lampY.crossVectors(lampZ, lampX)
  lampBasis.makeBasis(lampX, lampY, lampZ)
  lampBasis.scale(lampScale.set(slot.size[0], slot.size[1], slot.size[2]))
  lampBasis.setPosition(slot.position[0], slot.position[1], slot.position[2])
  return out.multiplyMatrices(body, lampBasis)
}

/** 前照灯が路面を照らす範囲（車両ローカル）。板 1 枚で、形はテクスチャで出す */
export const HEADLIGHT_POOL = { near: 2.3, far: 17, halfWidth: 3.9, y: 0.05 } as const

/** 路面に貼る板。u が前後（手前 0 → 奥 1）、v が左右 */
export function makeHeadlightPoolGeometry(): THREE.BufferGeometry {
  const P = HEADLIGHT_POOL
  const g = new THREE.PlaneGeometry(P.far - P.near, 2 * P.halfWidth)
  g.rotateX(-Math.PI / 2)
  g.translate((P.near + P.far) / 2, P.y, 0)
  return g
}
