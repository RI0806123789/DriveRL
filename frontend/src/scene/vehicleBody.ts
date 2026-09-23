/** 車体の外側（塗装・黒い樹脂・メッキ・窓ガラス）を組み立てる。**React から切り離した純粋モジュール。** */

import * as THREE from 'three'
import { MeshBuilder, orientedBox, vec, type PartStyle, type Vec3 } from './meshBuilder.ts'
import {
  BACKLIGHT_BOTTOM_X,
  BACKLIGHT_TOP_X,
  BELT_Y,
  BODY_HALF_W,
  B_PILLAR_X,
  DOOR_BOTTOM_Y,
  FRONT_DOOR_X,
  GLASS_BOTTOM_HALF_W,
  GLASS_TOP_HALF_W,
  GLASS_TOP_Y,
  REAR_DOOR_X,
  SIDE_GLASS_TOP_Y,
  WHEEL_OFFSETS,
  WHEEL_RADIUS,
  WINDSHIELD_BOTTOM_X,
  WINDSHIELD_TOP_X,
  sideGlassZ,
} from './vehicleGeometry.ts'
import { LIGHT_SLOTS, MIRROR } from './vehicleLights.ts'
import { WINDSHIELD_BULGE } from './wiper.ts'
import {
  ARCH_HALF_SPAN,
  OUTLINE_BREAKS,
  OUTLINE_LENGTH,
  ROCKER_Y,
  archY,
  mirrorZ,
  outlineAt,
  sOfFrontZ,
  sOfRearZ,
  sOfSideX,
  sectionPoints,
  surfaceAt,
} from './vehicleShape.ts'

export type BodyLod = 'near' | 'far'

/** 黒い樹脂・暗い部分の頂点色（遠景の車体で塗装に掛けて黒くする） */
export const DARK: Vec3 = [0.06, 0.065, 0.07]

interface LodSpec {
  /** 外板の下半分を刻む割合（0..1）。ドアの下端の割合を必ず含める */
  readonly lower: readonly number[]
  readonly shoulder: number
  readonly frontStraight: number
  readonly frontArc: number
  readonly rearArc: number
  readonly rearStraight: number
  /** 側面で点を置く最大の間隔 [m] */
  readonly sideStep: number
  readonly archSteps: number
  /** ボンネット・トランクを横に何分割するか */
  readonly rung: number
  readonly roofRows: number
  readonly pillarRows: number
  readonly pillarCols: number
}

/** 側面の断面で、ドアの下端（DOOR_BOTTOM_Y）に当たる割合 */
const DOOR_FRACTION = (DOOR_BOTTOM_Y - ROCKER_Y) / (BELT_Y - 0.07 - ROCKER_Y)

const LOD: Record<BodyLod, LodSpec> = {
  near: {
    lower: [0, 0.06, DOOR_FRACTION, 0.3, 0.5, 0.7, 0.87, 1],
    shoulder: 4,
    frontStraight: 3,
    frontArc: 6,
    rearArc: 6,
    rearStraight: 3,
    sideStep: 0.26,
    archSteps: 12,
    rung: 6,
    roofRows: 10,
    pillarRows: 5,
    pillarCols: 5,
  },
  far: {
    lower: [0, 0.35, 1],
    shoulder: 2,
    frontStraight: 1,
    frontArc: 2,
    rearArc: 2,
    rearStraight: 1,
    sideStep: 0.9,
    archSteps: 4,
    rung: 2,
    roofRows: 4,
    pillarRows: 2,
    pillarCols: 3,
  },
}

/** 右半分の外形に置く点（周長 s）。ドアの境目・アーチ・ガラスの付け根を必ず含む */
function outlineSamples(spec: LodSpec): number[] {
  const out: number[] = []
  const push = (s: number) => out.push(Math.min(OUTLINE_LENGTH, Math.max(0, s)))
  const { frontArcStart, sideStart, rearArcStart, rearStart } = OUTLINE_BREAKS
  for (let k = 0; k <= spec.frontStraight; k++) push((frontArcStart * k) / spec.frontStraight)
  for (let k = 1; k <= spec.frontArc; k++) push(frontArcStart + ((sideStart - frontArcStart) * k) / spec.frontArc)

  const forced = new Set<number>([
    FRONT_DOOR_X[1],
    REAR_DOOR_X[0],
    REAR_DOOR_X[1],
    FRONT_DOOR_X[0],
    WINDSHIELD_BOTTOM_X,
    BACKLIGHT_BOTTOM_X,
  ])
  const arches = [...new Set(WHEEL_OFFSETS.map((w) => w.position[0]))]
  for (const cx of arches) {
    for (let k = 0; k <= spec.archSteps; k++) {
      forced.add(cx - ARCH_HALF_SPAN + (2 * ARCH_HALF_SPAN * k) / spec.archSteps)
    }
  }
  const sideFront = outlineAt(sideStart).x
  const sideRear = outlineAt(rearArcStart).x
  const xs = [...forced].filter((x) => x < sideFront && x > sideRear).sort((a, b) => b - a)
  let prev = sideFront
  for (const x of [...xs, sideRear]) {
    const gap = prev - x
    const n = Math.ceil(gap / spec.sideStep - 1e-9)
    for (let k = 1; k < n; k++) push(sOfSideX(prev - (gap * k) / n))
    push(sOfSideX(x))
    prev = x
  }
  for (let k = 1; k <= spec.rearArc; k++) push(rearArcStart + ((rearStart - rearArcStart) * k) / spec.rearArc)
  for (let k = 1; k <= spec.rearStraight; k++) {
    push(rearStart + ((OUTLINE_LENGTH - rearStart) * k) / spec.rearStraight)
  }
  const sorted = out.sort((a, b) => a - b)
  return sorted.filter((s, i) => i === 0 || s - sorted[i - 1] > 1e-6)
}

/** 平面形の外向き（上面と見分けがつくよう少し上へ寄せる） */
function wallOutward(p: Vec3): Vec3 {
  const x = p[0]
  const z = p[2]
  const cx = Math.min(Math.abs(x), 1.895) * Math.sign(x)
  const cz = Math.min(Math.abs(z), 0.585) * Math.sign(z)
  let nx = x - cx
  let nz = z - cz
  if (Math.abs(x) < 1.895) nx = 0
  if (Math.abs(z) < 0.585) nz = 0
  const l = Math.hypot(nx, nz) || 1
  return [nx / l, 0.35, nz / l]
}

interface Wall {
  /** 右半分の各点の断面（下から上） */
  readonly right: Vec3[][]
  readonly samples: number[]
}

function wallSections(spec: LodSpec): Wall {
  const samples = outlineSamples(spec)
  const right = samples.map((s) => sectionPoints(s, spec.lower, spec.shoulder))
  return { right, samples }
}

/** 外板（前・側面・後ろを 1 周する面）。左後席ドアの部分に印を付ける */
function addWall(b: MeshBuilder, wall: Wall, spec: LodSpec, style: PartStyle): void {
  const n = wall.right.length
  const rows: Vec3[][] = [...wall.right]
  const rowX: number[] = wall.samples.map((s) => outlineAt(s).x)
  const rowLeft: boolean[] = wall.samples.map(() => false)
  for (let i = n - 2; i >= 0; i--) {
    rows.push(wall.right[i].map(mirrorZ))
    rowX.push(outlineAt(wall.samples[i]).x)
    rowLeft.push(true)
  }
  const doorJ = spec.lower.indexOf(DOOR_FRACTION)
  const inDoor = (i: number) =>
    rowLeft[i] && rowX[i] >= REAR_DOOR_X[0] - 1e-6 && rowX[i] <= REAR_DOOR_X[1] + 1e-6
  b.grid(rows, wallOutward, (i, j) => ({
    ...style,
    door: doorJ >= 0 && j >= doorJ && inDoor(i) && inDoor(i + 1),
  }))
}

/** ボンネットとトランク（左右の肩の終わりを横木のように結ぶ）。 */
function addDecks(b: MeshBuilder, wall: Wall, spec: LodSpec, style: PartStyle): void {
  const end = (i: number) => wall.right[i][wall.right[i].length - 1]
  const hoodIdx: number[] = []
  const trunkIdx: number[] = []
  for (let i = 0; i < wall.samples.length; i++) {
    const x = outlineAt(wall.samples[i]).x
    if (x >= WINDSHIELD_BOTTOM_X - 1e-9) hoodIdx.push(i)
    if (x <= BACKLIGHT_BOTTOM_X + 1e-9) trunkIdx.push(i)
  }
  const deck = (idx: number[], crown: number, edge: (x: number) => number) => {
    const rows: Vec3[][] = []
    for (const i of idx) {
      const p = end(i)
      const row: Vec3[] = []
      for (let k = 0; k <= spec.rung; k++) {
        const t = k / spec.rung
        const z = p[2] * (2 * t - 1)
        const across = p[2] > 1e-6 ? 1 - (z / p[2]) ** 2 : 0
        row.push([p[0], p[1] + crown * across * edge(p[0]), z])
      }
      rows.push(row)
    }
    b.grid(rows, [0, 1, 0], style)
  }
  deck(hoodIdx, HOOD_CROWN, hoodEdge)
  deck(trunkIdx, TRUNK_CROWN, trunkEdge)
}

const HOOD_CROWN = 0.02
const TRUNK_CROWN = 0.014

function smoothEdge(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)))
  return t * t * (3 - 2 * t)
}

/** 肩の丸みの終わり（ボンネット・トランクの縁）の点 */
function shoulderEnd(s: number): Vec3 {
  const sec = sectionPoints(s, [1], 1)
  return sec[sec.length - 1]
}

const DECK_FRONT_X = shoulderEnd(0)[0]
const DECK_REAR_X = shoulderEnd(OUTLINE_LENGTH)[0]

/** ボンネット・トランクの盛り上がり（縁で 0、中ほどで最大） */
function hoodEdge(x: number): number {
  return smoothEdge(DECK_FRONT_X, DECK_FRONT_X - 0.3, x) * smoothEdge(WINDSHIELD_BOTTOM_X, WINDSHIELD_BOTTOM_X + 0.25, x)
}
function trunkEdge(x: number): number {
  return smoothEdge(DECK_REAR_X, DECK_REAR_X + 0.3, x) * smoothEdge(BACKLIGHT_BOTTOM_X, BACKLIGHT_BOTTOM_X - 0.2, x)
}

/** ボンネット・トランクの上面の高さ（側面の直線部の範囲で。見切り線を沿わせるのに使う） */
export function deckY(x: number, z: number): number {
  const sideFront = outlineAt(OUTLINE_BREAKS.sideStart).x
  const sideRear = outlineAt(OUTLINE_BREAKS.rearArcStart).x
  const p = shoulderEnd(sOfSideX(Math.min(sideFront, Math.max(sideRear, x))))
  const across = Math.max(0, 1 - (z / p[2]) ** 2)
  if (x >= WINDSHIELD_BOTTOM_X) return p[1] + HOOD_CROWN * across * hoodEdge(x)
  if (x <= BACKLIGHT_BOTTOM_X) return p[1] + TRUNK_CROWN * across * trunkEdge(x)
  return p[1]
}

/** フロントガラス・リアガラスの面。`t` は下端 0 → 上端 1、`u` は左端 -1 → 右端 1 */
interface GlassPlane {
  readonly bottom: readonly [number, number]
  readonly top: readonly [number, number]
  /** 外向き（XY 平面内） */
  readonly normal: readonly [number, number]
}

const WINDSHIELD: GlassPlane = (() => {
  const dx = WINDSHIELD_TOP_X - WINDSHIELD_BOTTOM_X
  const dy = GLASS_TOP_Y - BELT_Y
  const l = Math.hypot(dx, dy)
  return { bottom: [WINDSHIELD_BOTTOM_X, BELT_Y], top: [WINDSHIELD_TOP_X, GLASS_TOP_Y], normal: [dy / l, -dx / l] }
})()

const BACKLIGHT: GlassPlane = (() => {
  const dx = BACKLIGHT_TOP_X - BACKLIGHT_BOTTOM_X
  const dy = GLASS_TOP_Y - BELT_Y
  const l = Math.hypot(dx, dy)
  return { bottom: [BACKLIGHT_BOTTOM_X, BELT_Y], top: [BACKLIGHT_TOP_X, GLASS_TOP_Y], normal: [-dy / l, dx / l] }
})()

/** 前後のガラスの面上の点。`lift` だけ外へ浮かせる */
function onGlass(g: GlassPlane, t: number, zAbs: number, sign: number, lift: number, bulge = 0): Vec3 {
  const x = g.bottom[0] + (g.top[0] - g.bottom[0]) * t
  const y = g.bottom[1] + (g.top[1] - g.bottom[1]) * t
  const halfW = GLASS_BOTTOM_HALF_W + (GLASS_TOP_HALF_W - GLASS_BOTTOM_HALF_W) * t
  const across = zAbs / halfW
  const push = lift + bulge * Math.max(0, 1 - across * across)
  return [x + g.normal[0] * push, y + g.normal[1] * push, sign * zAbs]
}

/** ガラスの縁の長さ方向の割合を、下端より少し下（外板へ潜らせる）から上端まで伸ばす */
const GLASS_T0 = -0.03
const GLASS_T1 = 1.0

/** 前後のガラス（少しだけ前へ膨らませる）。 */
function addScreen(b: MeshBuilder, g: GlassPlane, bulge: number, rows: number, cols: number): void {
  const grid: Vec3[][] = []
  for (let r = 0; r <= rows; r++) {
    const t = GLASS_T0 + ((GLASS_T1 + 0.01 - GLASS_T0) * r) / rows
    const halfW = GLASS_BOTTOM_HALF_W + (GLASS_TOP_HALF_W - GLASS_BOTTOM_HALF_W) * t + 0.005
    const row: Vec3[] = []
    for (let c = 0; c <= cols; c++) {
      const u = -1 + (2 * c) / cols
      row.push(onGlass(g, t, Math.abs(u) * halfW, Math.sign(u), 0, bulge))
    }
    grid.push(row)
  }
  b.grid(grid, [g.normal[0], g.normal[1], 0])
}

/** 横窓の輪郭（右側。X と Y の組）。前の窓と後ろの窓 */
export const FRONT_WINDOW: ReadonlyArray<readonly [number, number]> = [
  [1.03, BELT_Y - 0.008],
  [0.06, BELT_Y - 0.008],
  [0.06, SIDE_GLASS_TOP_Y + 0.004],
  [0.59, SIDE_GLASS_TOP_Y + 0.004],
]
export const REAR_WINDOW: ReadonlyArray<readonly [number, number]> = [
  [-0.06, BELT_Y - 0.008],
  [-0.93, BELT_Y - 0.008],
  [-0.48, SIDE_GLASS_TOP_Y + 0.004],
  [-0.06, SIDE_GLASS_TOP_Y + 0.004],
]

function sidePoint(x: number, y: number, sign: number, lift = 0): Vec3 {
  return [x, y, sign * (sideGlassZ(y) + lift)]
}

function addSideWindow(
  b: MeshBuilder,
  outline: ReadonlyArray<readonly [number, number]>,
  sign: number,
  style: PartStyle,
): void {
  const pts = outline.map(([x, y]) => sidePoint(x, y, sign))
  const out: Vec3 = [0, 0.2, sign]
  for (let k = 1; k + 1 < pts.length; k++) b.triangle(pts[0], pts[k], pts[k + 1], style, undefined, out)
}

/** 窓ガラス（全部）。**遠景・近景で共通**なので、車体の両方の段がこれに合わせる */
export function makeVehicleGlass(): THREE.BufferGeometry {
  const b = new MeshBuilder()
  addScreen(b, WINDSHIELD, WINDSHIELD_BULGE, 2, 6)
  addScreen(b, BACKLIGHT, 0.01, 2, 6)
  for (const sign of [1, -1]) {
    addSideWindow(b, FRONT_WINDOW, sign, {})
    addSideWindow(b, REAR_WINDOW, sign, { door: sign < 0 })
  }
  return b.build({ door: true })
}

/** 2 本の縁の間に、外へ膨らんだ帯を張る（A ピラー・C ピラー）。 */
function pillarRows(
  inner: (t: number) => Vec3,
  outer: (t: number) => Vec3,
  bulgeDir: Vec3,
  bulge: number,
  rows: number,
  cols: number,
): Vec3[][] {
  const grid: Vec3[][] = []
  for (let r = 0; r <= rows; r++) {
    const t = r / rows
    const a = inner(t)
    const c = outer(t)
    const row: Vec3[] = []
    for (let k = 0; k <= cols; k++) {
      const u = k / cols
      const push = bulge * Math.sin(Math.PI * u)
      row.push([
        a[0] + (c[0] - a[0]) * u + bulgeDir[0] * push,
        a[1] + (c[1] - a[1]) * u + bulgeDir[1] * push,
        a[2] + (c[2] - a[2]) * u + bulgeDir[2] * push,
      ])
    }
    grid.push(row)
  }
  return grid
}

/** A ピラー（フロントガラスの縁から横窓の前縁まで）。根元は外板へ潜らせる */
function aPillar(sign: number, spec: LodSpec): Vec3[][] {
  const [f0, , , f3] = FRONT_WINDOW
  const inner = (t: number): Vec3 => {
    const tt = GLASS_T0 + (1.0 - GLASS_T0) * t
    const halfW = GLASS_BOTTOM_HALF_W + (GLASS_TOP_HALF_W - GLASS_BOTTOM_HALF_W) * tt
    return onGlass(WINDSHIELD, tt, halfW - 0.026, sign, 0.007)
  }
  const outer = (t: number): Vec3 => {
    const y = f0[1] - 0.02 + (f3[1] - f0[1] + 0.02) * t
    const x = f0[0] + (f3[0] - f0[0]) * t - 0.022
    return sidePoint(x, y, sign, 0.007)
  }
  const dir = vec.normalize([WINDSHIELD.normal[0], WINDSHIELD.normal[1], sign], [0, 1, 0])
  return pillarRows(inner, outer, dir, 0.016, spec.pillarRows, spec.pillarCols)
}

/** C ピラー（リアガラスの縁から後ろの横窓の後縁まで） */
function cPillar(sign: number, spec: LodSpec): Vec3[][] {
  const [, r1, r2] = REAR_WINDOW
  const inner = (t: number): Vec3 => {
    const tt = GLASS_T0 + (1.0 - GLASS_T0) * t
    const halfW = GLASS_BOTTOM_HALF_W + (GLASS_TOP_HALF_W - GLASS_BOTTOM_HALF_W) * tt
    return onGlass(BACKLIGHT, tt, halfW - 0.026, sign, 0.007)
  }
  const outer = (t: number): Vec3 => {
    const y = r1[1] - 0.02 + (r2[1] - r1[1] + 0.02) * t
    const x = r1[0] + (r2[0] - r1[0]) * t + 0.022
    return sidePoint(x, y, sign, 0.007)
  }
  const dir = vec.normalize([BACKLIGHT.normal[0], BACKLIGHT.normal[1], sign], [0, 1, 0])
  return pillarRows(inner, outer, dir, 0.02, spec.pillarRows, spec.pillarCols)
}

/** 屋根。前後の縁はガラスの上端より外、横の縁は横窓の上端より外へ下ろす */
const ROOF_RAIL_Y = 1.358
const ROOF_RAIL_Z = 0.724
const ROOF_FLAT_Z = 0.66

function roofRows(spec: LodSpec): Vec3[][] {
  const front = onGlass(WINDSHIELD, 1, 0, 1, 0.013)
  const rear = onGlass(BACKLIGHT, 1, 0, 1, 0.013)
  const lengthwise = (t: number): [number, number] => {
    const x = front[0] + (rear[0] - front[0]) * t
    const lip = Math.min(t, 1 - t)
    const rise = Math.sin(Math.min(1, lip / 0.16) * (Math.PI / 2))
    const yLip = front[1] + (rear[1] - front[1]) * t - 0.003
    return [x, yLip + (1.45 - yLip) * rise]
  }
  const zs = [0, 0.32, 0.56, ROOF_FLAT_Z, 0.695, 0.714, ROOF_RAIL_Z]
  const rows: Vec3[][] = []
  const n = spec.roofRows
  for (let r = 0; r <= n; r++) {
    const t = r / n
    const tt = 0.5 - 0.5 * Math.cos(Math.PI * t)
    const [x, yc] = lengthwise(tt)
    const lip = Math.min(tt, 1 - tt)
    const crown = 0.012 * Math.min(1, lip / 0.12)
    const row: Vec3[] = []
    const across = spec.roofRows > 4 ? zs : [0, 0.5, ROOF_FLAT_Z, 0.705, ROOF_RAIL_Z]
    const full = [...across.slice(1).reverse().map((z) => -z), ...across]
    for (const z of full) {
      const a = Math.abs(z)
      let y: number
      if (a <= ROOF_FLAT_Z) {
        y = yc - crown * (a / ROOF_FLAT_Z) ** 2
      } else {
        const edge = yc - crown
        const q = (a - ROOF_FLAT_Z) / (ROOF_RAIL_Z - ROOF_FLAT_Z)
        y = ROOF_RAIL_Y + (edge - ROOF_RAIL_Y) * Math.sqrt(Math.max(0, 1 - q * q))
      }
      row.push([x, y, z])
    }
    rows.push(row)
  }
  return rows
}

/** 外板と屋根・ピラー（塗装される部分）。 */
function addPaintShell(b: MeshBuilder, spec: LodSpec, wall: Wall, style: PartStyle): void {
  addWall(b, wall, spec, style)
  addDecks(b, wall, spec, style)
  b.grid(roofRows(spec), (p) => [0, 1, p[2] * 0.6], style)
  for (const sign of [1, -1]) {
    b.grid(aPillar(sign, spec), [WINDSHIELD.normal[0], WINDSHIELD.normal[1], sign], style)
    b.grid(cPillar(sign, spec), [BACKLIGHT.normal[0], BACKLIGHT.normal[1], sign], style)
  }
}

/** 面に沿った帯（見切り線・サイドシル）。`at(t)` が面上の点と法線を返す */
function addStrip(
  b: MeshBuilder,
  at: (t: number) => { point: Vec3; normal: Vec3 },
  width: number,
  lift: number,
  steps: number,
  style: PartStyle,
): void {
  const rows: Vec3[][] = []
  const normals: Vec3[] = []
  for (let k = 0; k <= steps; k++) {
    const t = k / steps
    const here = at(t)
    normals.push(here.normal)
    const ahead = at(Math.min(1, t + 1 / steps))
    const behind = at(Math.max(0, t - 1 / steps))
    const tangent = vec.sub(ahead.point, behind.point)
    const side = vec.normalize(vec.cross(here.normal, tangent), [0, 1, 0])
    const n = here.normal
    const base: Vec3 = [
      here.point[0] + n[0] * lift,
      here.point[1] + n[1] * lift,
      here.point[2] + n[2] * lift,
    ]
    rows.push([
      [base[0] - side[0] * width * 0.5, base[1] - side[1] * width * 0.5, base[2] - side[2] * width * 0.5],
      [base[0] + side[0] * width * 0.5, base[1] + side[1] * width * 0.5, base[2] + side[2] * width * 0.5],
    ])
  }
  b.grid(rows, (_p, i) => normals[Math.max(0, i)], style)
}

/** 側面の点（左右は sign） */
function sideSurface(x: number, y: number, sign: number): { point: Vec3; normal: Vec3 } {
  const r = surfaceAt(sOfSideX(x), y)
  return sign > 0 ? r : { point: mirrorZ(r.point), normal: mirrorZ(r.normal) }
}

/** 面の上に張る四角い板（格子）。`at(u, v)` が面上の点と法線を返す。**縦は外板の断面より細かく刻む**（粗いと外板の膨らみに埋もれる） */
function addPatch(
  b: MeshBuilder,
  at: (u: number, v: number) => { point: Vec3; normal: Vec3 },
  lift: number,
  cols: number,
  minRows: number,
  style: PartStyle,
): void {
  const span = Math.abs(at(0.5, 1).point[1] - at(0.5, 0).point[1])
  const rows = Math.max(minRows, Math.ceil(span / 0.025))
  const grid: Vec3[][] = []
  for (let r = 0; r <= rows; r++) {
    const row: Vec3[] = []
    for (let c = 0; c <= cols; c++) {
      const h = at(c / cols, r / rows)
      row.push([
        h.point[0] + h.normal[0] * lift,
        h.point[1] + h.normal[1] * lift,
        h.point[2] + h.normal[2] * lift,
      ])
    }
    grid.push(row)
  }
  const centre = at(0.5, 0.5).normal
  b.grid(grid, centre, style)
}

/** 前面・後面の点。z は符号つき */
function frontSurface(z: number, y: number): { point: Vec3; normal: Vec3 } {
  const r = surfaceAt(sOfFrontZ(Math.abs(z)), y)
  return z >= 0 ? r : { point: mirrorZ(r.point), normal: mirrorZ(r.normal) }
}
function rearSurface(z: number, y: number): { point: Vec3; normal: Vec3 } {
  const r = surfaceAt(sOfRearZ(Math.abs(z)), y)
  return z >= 0 ? r : { point: mirrorZ(r.point), normal: mirrorZ(r.normal) }
}

/** ホイールハウスの内張り（天井と奥の壁）。車体の中空がアーチから透けないようにする */
function addWheelHouses(b: MeshBuilder, steps: number, ceiling: boolean, style: PartStyle): void {
  for (const w of WHEEL_OFFSETS) {
    const cx = w.position[0]
    const sign = Math.sign(w.position[2])
    const inner = w.steered ? 0.52 : 0.6
    const edge: Vec3[] = []
    for (let k = 0; k <= steps; k++) {
      const x = cx - ARCH_HALF_SPAN + (2 * ARCH_HALF_SPAN * k) / steps
      const y = archY(x) ?? ROCKER_Y
      edge.push([x, y, 0])
    }
    const toCentre = (p: Vec3): Vec3 => [cx - p[0], WHEEL_RADIUS - p[1], 0]
    if (ceiling) {
      const rows = edge.map((p): Vec3[] => {
        const outerZ = BODY_HALF_W - 0.03
        return [
          [p[0], p[1] - 0.004, sign * inner],
          [p[0], p[1] - 0.004, sign * outerZ],
        ]
      })
      b.grid(rows, toCentre, style)
    }
    const hub: Vec3 = [cx, ROCKER_Y + 0.02, sign * inner]
    for (let k = 0; k + 1 < edge.length; k++) {
      b.triangle(
        hub,
        [edge[k][0], edge[k][1] - 0.004, sign * inner],
        [edge[k + 1][0], edge[k + 1][1] - 0.004, sign * inner],
        style,
        undefined,
        [0, 0, sign],
      )
    }
  }
}

/** 灯火の縁取り（レンズの後ろに一回り大きい黒い板）。車体の面に付いている灯火だけ */
function addLampBezels(b: MeshBuilder, style: PartStyle): void {
  for (const slot of LIGHT_SLOTS) {
    if (slot.position[1] > 0.95) continue
    const f = vec.normalize(slot.facing, [1, 0, 0])
    const centre: Vec3 = [
      slot.position[0] - f[0] * 0.006,
      slot.position[1] - f[1] * 0.006,
      slot.position[2] - f[2] * 0.006,
    ]
    const g = orientedBox(centre, f, slot.up, [slot.size[0] - 0.002, slot.size[1] + 0.018, slot.size[2] + 0.018])
    b.geometry(g, style)
    g.dispose()
  }
}

/** 見切り線（ドア・ボンネット・トランクの隙間）。細い黒い帯を面から少し浮かせる */
function addPanelGaps(b: MeshBuilder, style: PartStyle): void {
  const W = 0.005
  const LIFT = 0.0025
  const topY = BELT_Y - 0.075
  for (const sign of [1, -1]) {
    for (const x of [FRONT_DOOR_X[1], 0, REAR_DOOR_X[0]]) {
      const door = sign < 0 && x === REAR_DOOR_X[0]
      addStrip(b, (t) => sideSurface(x, DOOR_BOTTOM_Y + (topY - DOOR_BOTTOM_Y) * t, sign), W, LIFT, 6, { ...style, door })
    }
    addStrip(
      b,
      (t) => sideSurface(REAR_DOOR_X[0] + (REAR_DOOR_X[1] - REAR_DOOR_X[0]) * t, DOOR_BOTTOM_Y, sign),
      W,
      LIFT,
      4,
      { ...style, door: sign < 0 },
    )
    addStrip(
      b,
      (t) => sideSurface(FRONT_DOOR_X[0] + (FRONT_DOOR_X[1] - FRONT_DOOR_X[0]) * t, DOOR_BOTTOM_Y, sign),
      W,
      LIFT,
      4,
      style,
    )
    const hoodZ = sign * 0.64
    addStrip(
      b,
      (t) => {
        const x = WINDSHIELD_BOTTOM_X + 0.03 + (1.93 - WINDSHIELD_BOTTOM_X - 0.03) * t
        return { point: [x, deckY(x, hoodZ), hoodZ], normal: [0, 1, 0] }
      },
      W,
      0.003,
      6,
      style,
    )
    const trunkZ = sign * 0.62
    addStrip(
      b,
      (t) => {
        const x = BACKLIGHT_BOTTOM_X - 0.03 + (-1.93 - BACKLIGHT_BOTTOM_X + 0.03) * t
        return { point: [x, deckY(x, trunkZ), trunkZ], normal: [0, 1, 0] }
      },
      W,
      0.003,
      6,
      style,
    )
  }
  addStrip(b, (t) => frontSurface(-0.62 + 1.24 * t, 0.738), W, LIFT, 8, style)
  addStrip(b, (t) => rearSurface(-0.6 + 1.2 * t, 0.868), W, LIFT, 8, style)
}

/** ドアミラーの筐体（角を丸めた箱）。右側 */
function mirrorHousing(sign: number): THREE.BufferGeometry {
  const w = MIRROR.x[1] - MIRROR.x[0]
  const h = MIRROR.y[1] - MIRROR.y[0]
  const bevel = 0.014
  const shape = new THREE.Shape()
  const r = 0.035
  const x0 = -w / 2 + bevel
  const y0 = -h / 2 + bevel
  const x1 = w / 2 - bevel
  const y1 = h / 2 - bevel
  shape.moveTo(x0 + r, y0)
  shape.lineTo(x1 - r, y0)
  shape.quadraticCurveTo(x1, y0, x1, y0 + r)
  shape.lineTo(x1, y1 - r)
  shape.quadraticCurveTo(x1, y1, x1 - r, y1)
  shape.lineTo(x0 + r, y1)
  shape.quadraticCurveTo(x0, y1, x0, y1 - r)
  shape.lineTo(x0, y0 + r)
  shape.quadraticCurveTo(x0, y0, x0 + r, y0)
  const depth = MIRROR.z[1] - MIRROR.z[0] - 2 * bevel
  const g = new THREE.ExtrudeGeometry(shape, {
    depth,
    bevelEnabled: true,
    bevelThickness: bevel,
    bevelSize: bevel,
    bevelSegments: 2,
    curveSegments: 3,
  })
  g.translate(0, 0, bevel)
  // 形は上下対称なので、X 軸まわりに半回転すれば巻き順を保ったまま左右が入れ替わる
  if (sign < 0) g.rotateX(Math.PI)
  g.translate((MIRROR.x[0] + MIRROR.x[1]) / 2, (MIRROR.y[0] + MIRROR.y[1]) / 2, sign * MIRROR.z[0])
  return g
}

/** 塗装される部分。 */
export function makeBodyPaintGeometry(): THREE.BufferGeometry {
  const spec = LOD.near
  const b = new MeshBuilder()
  const wall = wallSections(spec)
  addPaintShell(b, spec, wall, {})
  for (const sign of [1, -1]) {
    const g = mirrorHousing(sign)
    b.geometry(g)
    g.dispose()
  }
  const lid = sideSurface(-1.55, 0.855, -1)
  b.geometry(orientedBox(offset(lid, 0.0035), [1, 0, 0], [0, 1, 0], [0.15, 0.1, 0.004]))
  return b.build({ door: true })
}

function offset(h: { point: Vec3; normal: Vec3 }, d: number): Vec3 {
  return [h.point[0] + h.normal[0] * d, h.point[1] + h.normal[1] * d, h.point[2] + h.normal[2] * d]
}

/** 黒い樹脂の部分（グリル・バンパー下部・B ピラー・内張り・見切り線など）。 */
export function makeBodyTrimGeometry(): THREE.BufferGeometry {
  const b = new MeshBuilder()
  const s: PartStyle = {}
  addPatch(b, (u, v) => frontSurface(-0.35 + 0.7 * u, 0.53 + 0.16 * v), 0.005, 4, 2, s)
  addPatch(b, (u, v) => frontSurface(-0.52 + 1.04 * u, 0.268 + 0.092 * v), 0.004, 4, 1, s)
  addPatch(b, (u, v) => rearSurface(-0.58 + 1.16 * u, 0.29 + 0.07 * v), 0.004, 4, 1, s)
  for (const sign of [1, -1]) {
    addPatch(b, (u, v) => sideSurface(-1.0 + 2.0 * u, 0.205 + 0.075 * v, sign), 0.003, 6, 2, s)
    const x0 = B_PILLAR_X[0]
    const x1 = B_PILLAR_X[1]
    const y0 = BELT_Y - 0.012
    const y1 = SIDE_GLASS_TOP_Y + 0.002
    b.quad(
      sidePoint(x0, y0, sign, 0.008),
      sidePoint(x1, y0, sign, 0.008),
      sidePoint(x1, y1, sign, 0.008),
      sidePoint(x0, y1, sign, 0.008),
      s,
      [0, 0, sign],
    )
    const topFront = FRONT_WINDOW[3]
    const topRear = REAR_WINDOW[2]
    for (const [xa, xb, door] of [
      [topFront[0], x1, false],
      [x0, topRear[0], sign < 0],
    ] as const) {
      const ya = SIDE_GLASS_TOP_Y - 0.012
      const yb = SIDE_GLASS_TOP_Y + 0.004
      b.quad(
        sidePoint(xa, ya, sign, 0.006),
        sidePoint(xb, ya, sign, 0.006),
        sidePoint(xb, yb, sign, 0.006),
        sidePoint(xa, yb, sign, 0.006),
        { ...s, door },
        [0, 0, sign],
      )
    }
    const stay = orientedBox([0.965, 0.995, sign * 0.815], [1, 0, 0], [0, 1, 0], [0.07, 0.05, 0.03])
    b.geometry(stay, s)
    stay.dispose()
  }
  addPatch(
    b,
    (u, v) => {
      const x = WINDSHIELD_BOTTOM_X + 0.01 + 0.11 * v
      const z = -0.76 + 1.52 * u
      return { point: [x, deckY(x, z), z], normal: [0, 1, 0] }
    },
    0.004,
    6,
    1,
    s,
  )
  const under = orientedBox([0, 0.212, 0], [1, 0, 0], [0, 1, 0], [4.1, 0.045, 1.64])
  b.geometry(under, s)
  under.dispose()
  addWheelHouses(b, 10, true, s)
  addLampBezels(b, s)
  addPanelGaps(b, s)
  const lidRing = sideSurface(-1.55, 0.855, -1)
  b.geometry(orientedBox(offset(lidRing, 0.0015), [1, 0, 0], [0, 1, 0], [0.16, 0.11, 0.003]), s)
  return b.build({ door: true })
}

/** メッキ（窓の下のモール・ドアハンドル・グリルの桟・ミラーの鏡）。 */
export function makeBodyChromeGeometry(): THREE.BufferGeometry {
  const b = new MeshBuilder()
  for (const y of [0.567, 0.607, 0.647]) {
    addPatch(b, (u, v) => frontSurface(-0.33 + 0.66 * u, y - 0.006 + 0.012 * v), 0.01, 4, 1, {})
  }
  for (const sign of [1, -1]) {
    for (const [x0, x1, door] of [
      [FRONT_WINDOW[0][0], B_PILLAR_X[1], false],
      [B_PILLAR_X[0], REAR_WINDOW[1][0], sign < 0],
    ] as const) {
      b.quad(
        sidePoint(x0, BELT_Y - 0.004, sign, 0.009),
        sidePoint(x1, BELT_Y - 0.004, sign, 0.009),
        sidePoint(x1, BELT_Y + 0.008, sign, 0.009),
        sidePoint(x0, BELT_Y + 0.008, sign, 0.009),
        { door },
        [0, 0.3, sign],
      )
    }
    for (const [x, door] of [
      [0.4, false],
      [-0.58, sign < 0],
    ] as const) {
      const h = sideSurface(x, 0.855, sign)
      b.geometry(orientedBox(offset(h, 0.008), [1, 0, 0], [0, 1, 0], [0.15, 0.026, 0.018]), { door })
    }
    const glass = orientedBox(
      [MIRROR.x[0] - 0.004, (MIRROR.y[0] + MIRROR.y[1]) / 2, sign * ((MIRROR.z[0] + MIRROR.z[1]) / 2 + 0.004)],
      [0, 0, 1],
      [0, 1, 0],
      [MIRROR.z[1] - MIRROR.z[0] - 0.026, MIRROR.y[1] - MIRROR.y[0] - 0.03, 0.006],
    )
    b.geometry(glass)
    glass.dispose()
  }
  return b.build({ door: true })
}

/** 遠景の車体。塗装と黒い部分を頂点色で分け、1 回の描画で済ませる */
export function makeFarBodyGeometry(): THREE.BufferGeometry {
  const spec = LOD.far
  const b = new MeshBuilder()
  const wall = wallSections(spec)
  const paint: PartStyle = { color: [1, 1, 1] }
  const dark: PartStyle = { color: DARK }
  addPaintShell(b, spec, wall, paint)
  addPatch(b, (u, v) => frontSurface(-0.35 + 0.7 * u, 0.53 + 0.16 * v), 0.005, 1, 1, dark)
  addPatch(b, (u, v) => frontSurface(-0.52 + 1.04 * u, 0.268 + 0.092 * v), 0.004, 1, 1, dark)
  addPatch(b, (u, v) => rearSurface(-0.58 + 1.16 * u, 0.29 + 0.07 * v), 0.004, 1, 1, dark)
  for (const sign of [1, -1]) {
    const y0 = BELT_Y - 0.012
    const y1 = SIDE_GLASS_TOP_Y + 0.002
    b.quad(
      sidePoint(B_PILLAR_X[0], y0, sign, 0.008),
      sidePoint(B_PILLAR_X[1], y0, sign, 0.008),
      sidePoint(B_PILLAR_X[1], y1, sign, 0.008),
      sidePoint(B_PILLAR_X[0], y1, sign, 0.008),
      dark,
      [0, 0, sign],
    )
    const m = orientedBox(
      [(MIRROR.x[0] + MIRROR.x[1]) / 2, (MIRROR.y[0] + MIRROR.y[1]) / 2, sign * ((MIRROR.z[0] + MIRROR.z[1]) / 2)],
      [1, 0, 0],
      [0, 1, 0],
      [MIRROR.x[1] - MIRROR.x[0], MIRROR.y[1] - MIRROR.y[0], MIRROR.z[1] - MIRROR.z[0]],
    )
    b.geometry(m, paint)
    m.dispose()
  }
  addWheelHouses(b, 4, false, dark)
  addCabinCore(b, dark)
  return b.build({ color: true })
}

/** 遠景の車内の代わり（窓越しに向こうが透けないよう、暗い塊を置く） */
function addCabinCore(b: MeshBuilder, style: PartStyle): void {
  const shape = new THREE.Shape()
  shape.moveTo(1.1, BELT_Y - 0.005)
  shape.lineTo(0.54, 1.38)
  shape.lineTo(-0.57, 1.38)
  shape.lineTo(-1.06, BELT_Y - 0.005)
  shape.closePath()
  const g = new THREE.ExtrudeGeometry(shape, { depth: 1.32, bevelEnabled: false, curveSegments: 1 })
  g.translate(0, 0, -0.66)
  b.geometry(g, style)
  g.dispose()
}
