/** 車輪（タイヤ・ホイール・ブレーキ）の形。**React から切り離した純粋モジュール。** */

import * as THREE from 'three'
import { MeshBuilder, type PartStyle, type Vec3 } from './meshBuilder.ts'
import { WHEEL_RADIUS } from './vehicleGeometry.ts'

/** 車輪の軸は +Z、外側の面が +Z。左の車輪は `composeWheelMatrix` が Y 軸まわりに半回転させる */

/** タイヤの幅 [m]。リムとスポークはこの中へ収める */
export const TYRE_WIDTH = 0.24
const HALF = TYRE_WIDTH / 2
/** リムの半径 [m]。タイヤの内側に見える金属部分 */
export const RIM_RADIUS = WHEEL_RADIUS * 0.62

/** タイヤ 1 周の分割数 */
export const TYRE_SEGMENTS = 22

/** タイヤの断面（半径, 軸方向）。ビード → サイドウォールの膨らみ → 肩 → 溝のあるトレッド */
const TYRE_PROFILE: ReadonlyArray<readonly [number, number]> = [
  [RIM_RADIUS - 0.003, -0.1],
  [0.262, -0.117],
  [0.3, -HALF],
  [0.329, -0.109],
  [WHEEL_RADIUS - 0.001, -0.088],
  [WHEEL_RADIUS, -0.045],
  [WHEEL_RADIUS - 0.007, -0.038],
  [WHEEL_RADIUS - 0.007, -0.024],
  [WHEEL_RADIUS, -0.017],
  [WHEEL_RADIUS, 0.017],
  [WHEEL_RADIUS - 0.007, 0.024],
  [WHEEL_RADIUS - 0.007, 0.038],
  [WHEEL_RADIUS, 0.045],
  [WHEEL_RADIUS - 0.001, 0.088],
  [0.329, 0.109],
  [0.3, HALF],
  [0.262, 0.117],
  [RIM_RADIUS - 0.003, 0.1],
]

/** 軸を Y にとった回転体を、軸が Z になるよう寝かせる。**断面は外側を右手に見て進む向きに並べる**（法線と巻き順がそれで決まる） */
function lathe(profile: ReadonlyArray<readonly [number, number]>, segments: number): THREE.BufferGeometry {
  const g = new THREE.LatheGeometry(
    profile.map(([r, y]) => new THREE.Vector2(r, y)),
    segments,
  )
  g.rotateX(Math.PI / 2)
  return g
}

/** タイヤ（ゴム）。 */
export function makeTyreGeometry(segments = TYRE_SEGMENTS): THREE.BufferGeometry {
  const b = new MeshBuilder()
  const g = lathe(TYRE_PROFILE, segments)
  b.geometry(g)
  g.dispose()
  return b.build()
}

/** ホイールの頂点色（材質は 1 つで、明るさだけ変える） */
const SILVER: Vec3 = [1, 1, 1]
const NUT: Vec3 = [0.78, 0.8, 0.83]
const CAP: Vec3 = [0.62, 0.64, 0.68]
const DISC: Vec3 = [0.42, 0.42, 0.43]
const BARREL: Vec3 = [0.55, 0.56, 0.58]

/** 円盤の面（XY 平面）に置く部品を、軸まわりの角度 `a` だけ回して足す */
function addRotated(b: MeshBuilder, g: THREE.BufferGeometry, a: number, style: PartStyle): void {
  const m = new THREE.Matrix4().makeRotationZ(a)
  b.geometry(g.clone(), style, m)
}

/** スポーク 1 本（根元と先で幅の違う板。外側の面は少しだけ皿状に） */
function spokeGeometry(r0: number, r1: number, w0: number, w1: number, z0: number, z1: number, depth: number): THREE.BufferGeometry {
  const b = new MeshBuilder()
  const p = (r: number, w: number, z: number): Vec3 => [r, w, z]
  const front = [p(r0, -w0 / 2, z0), p(r1, -w1 / 2, z1), p(r1, w1 / 2, z1), p(r0, w0 / 2, z0)]
  const back = front.map(([x, y, z]): Vec3 => [x, y, z - depth])
  b.quad(front[0], front[1], front[2], front[3], {}, [0, 0, 1])
  b.quad(back[3], back[2], back[1], back[0], {}, [0, 0, -1])
  b.quad(back[0], back[1], front[1], front[0], {}, [0, -1, 0])
  b.quad(front[3], front[2], back[2], back[3], {}, [0, 1, 0])
  return b.build()
}

/** ホイール（リム・スポーク・ハブ・ナット・センターキャップ）とブレーキディスク。回転する部分をまとめる */
export function makeRimGeometry(): THREE.BufferGeometry {
  const b = new MeshBuilder()
  const barrel = lathe(
    [
      [RIM_RADIUS - 0.008, 0.07],
      [RIM_RADIUS - 0.008, -0.095],
    ],
    TYRE_SEGMENTS,
  )
  b.geometry(barrel, { color: BARREL })
  barrel.dispose()
  const lip = lathe(
    [
      [RIM_RADIUS - 0.008, 0.07],
      [RIM_RADIUS + 0.006, 0.082],
      [RIM_RADIUS + 0.004, 0.1],
      [RIM_RADIUS - 0.012, 0.106],
      [RIM_RADIUS - 0.03, 0.098],
    ],
    TYRE_SEGMENTS,
  )
  b.geometry(lip, { color: SILVER })
  lip.dispose()

  const hubR = 0.058
  const hub = lathe(
    [
      [hubR, 0.07],
      [hubR, 0.09],
      [hubR - 0.012, 0.098],
      [0.0001, 0.098],
    ],
    14,
  )
  b.geometry(hub, { color: SILVER })
  hub.dispose()
  const cap = lathe(
    [
      [0.028, 0.096],
      [0.028, 0.1],
      [0.02, 0.105],
      [0.0001, 0.106],
    ],
    12,
  )
  b.geometry(cap, { color: CAP })
  cap.dispose()

  // 5 本の 2 股スポーク（特定の車のホイールに似せない、ごく一般的な形）
  const spoke = spokeGeometry(hubR - 0.01, RIM_RADIUS - 0.01, 0.026, 0.02, 0.094, 0.086, 0.018)
  for (let k = 0; k < 5; k++) {
    const a = (k * Math.PI * 2) / 5
    addRotated(b, spoke, a - 0.11, { color: SILVER })
    addRotated(b, spoke, a + 0.11, { color: SILVER })
  }
  spoke.dispose()

  const nut = new THREE.CylinderGeometry(0.0085, 0.0085, 0.012, 6)
  nut.rotateX(Math.PI / 2)
  nut.translate(0.036, 0, 0.102)
  for (let k = 0; k < 5; k++) addRotated(b, nut, (k * Math.PI * 2) / 5 + Math.PI / 5, { color: NUT })
  nut.dispose()

  const disc = lathe(
    [
      [0.075, -0.018],
      [0.075, -0.042],
      [0.158, -0.042],
      [0.158, -0.018],
      [0.075, -0.018],
    ],
    TYRE_SEGMENTS,
  )
  b.geometry(disc, { color: DISC })
  disc.dispose()
  return b.build({ color: true })
}

/** ブレーキキャリパーを置く角度 [rad]（XY 平面、+X が前）。後ろ寄りの上 */
export const CALIPER_ANGLE = Math.PI * 0.78

/** ブレーキキャリパー。回転しない（舵角だけで向きが変わる） */
export function makeCaliperGeometry(): THREE.BufferGeometry {
  const b = new MeshBuilder()
  const r0 = 0.118
  const r1 = 0.172
  const z0 = -0.058
  const z1 = -0.002
  const span = 0.62
  const steps = 5
  const section: Array<readonly [number, number]> = [
    [r0, z0],
    [r1, z0],
    [r1, z1],
    [r0, z1],
  ]
  const at = (a: number, r: number, z: number): Vec3 => [Math.cos(a) * r, Math.sin(a) * r, z]
  for (let k = 0; k < section.length; k++) {
    const [ra, za] = section[k]
    const [rb, zb] = section[(k + 1) % section.length]
    const rows: Vec3[][] = []
    for (let i = 0; i <= steps; i++) {
      const a = CALIPER_ANGLE - span / 2 + (span * i) / steps
      rows.push([at(a, ra, za), at(a, rb, zb)])
    }
    const mid = [(ra + rb) / 2, (za + zb) / 2] as const
    b.grid(rows, (p) => {
      const a = Math.atan2(p[1], p[0])
      const dr = mid[0] - (r0 + r1) / 2
      const dz = mid[1] - (z0 + z1) / 2
      return [Math.cos(a) * dr, Math.sin(a) * dr, dz]
    })
  }
  for (const a of [CALIPER_ANGLE - span / 2, CALIPER_ANGLE + span / 2]) {
    const pts = section.map(([r, z]) => at(a, r, z))
    const out: Vec3 = a < CALIPER_ANGLE ? [Math.sin(a), -Math.cos(a), 0] : [-Math.sin(a), Math.cos(a), 0]
    b.quad(pts[0], pts[1], pts[2], pts[3], {}, out)
  }
  return b.build()
}

/** 遠景の車輪（タイヤとホイールの面を頂点色で分けた 1 つの円柱） */
export function makeFarWheelGeometry(): THREE.BufferGeometry {
  const b = new MeshBuilder()
  const tyre = new THREE.CylinderGeometry(WHEEL_RADIUS, WHEEL_RADIUS, TYRE_WIDTH, 12, 1, true)
  tyre.rotateX(Math.PI / 2)
  b.geometry(tyre, { color: [0.16, 0.16, 0.17] })
  tyre.dispose()
  for (const side of [1, -1]) {
    const face = new THREE.CircleGeometry(WHEEL_RADIUS, 12)
    if (side < 0) face.rotateY(Math.PI)
    face.translate(0, 0, side * (HALF - 0.001))
    b.geometry(face, { color: side > 0 ? [0.62, 0.64, 0.68] : [0.16, 0.16, 0.17] })
    face.dispose()
  }
  return b.build({ color: true })
}
