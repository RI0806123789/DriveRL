/** 車両の形と姿勢の計算。**React から切り離した純粋モジュール。** */

import * as THREE from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'

/** 車両ローカル座標は **+X が前方・+Y が上・+Z が右**。 */

/** 外接寸法 [m]。**backend の `config.VEHICLE_*` と揃える。ここを越える部品を作らない** */
export const VEHICLE_LENGTH = 4.4
export const VEHICLE_WIDTH = 1.8
export const VEHICLE_HEIGHT = 1.45

const HALF_L = VEHICLE_LENGTH / 2
const HALF_W = VEHICLE_WIDTH / 2

/** 車輪の半径 [m]。転がり角の計算にも使う */
export const WHEEL_RADIUS = 0.34

/** 各ジオメトリの外接箱の中心（車両ローカル）。 */
export const BODY_OFFSET: readonly [number, number, number] = [0, 0.815, 0]
/** ボンネット先端の飾りの中心 */
export const NOSE_OFFSET: readonly [number, number, number] = [1.94, 0.975, 0]
/** キャビン（グラスハウス）の中心 */
export const CABIN_OFFSET: readonly [number, number, number] = [-0.025, 1.215, 0]

/** 車輪の取り付け位置と、舵角を効かせるかどうか。 */
export const WHEEL_OFFSETS: ReadonlyArray<{
  readonly position: readonly [number, number, number]
  readonly steered: boolean
}> = [
  { position: [1.45, WHEEL_RADIUS, 0.86], steered: true },
  { position: [1.45, WHEEL_RADIUS, -0.86], steered: true },
  { position: [-1.45, WHEEL_RADIUS, 0.86], steered: false },
  { position: [-1.45, WHEEL_RADIUS, -0.86], steered: false },
]

/** 運転席の中心（車両ローカル）。`cameraMath` のアイポイントと同じ側に置くこと */
export const DRIVER_SEAT_Z = 0.36
/** 助手席 */
export const PASSENGER_SEAT_Z = -0.36

/** キャビンの床・天井の高さ [m] */
const FLOOR_Y = 0.34
const ROOF_Y = VEHICLE_HEIGHT
/** ベルトライン（窓の下端）[m] */
const BELT_Y = 0.98
/** 外板の厚み（片側）[m]。車体を中空にするときの壁の厚さ */
const BODY_SKIN = 0.06

/** 部品どうしを重ねるときの食い込み量 [m]。 */
const OVERLAP = 0.04

/** フロントガラスの下端・上端の X。寝かせるほど差が開く */
const WINDSHIELD_BOTTOM_X = 1.15
const WINDSHIELD_TOP_X = 0.55
/** リアガラス */
const BACKLIGHT_BOTTOM_X = -1.1
const BACKLIGHT_TOP_X = -0.6

/** 箱を 1 つ作る。`min`〜`max` で与えるほうが、外接寸法を守れているか読めば分かる */
function box(
  min: readonly [number, number, number],
  max: readonly [number, number, number],
): THREE.BufferGeometry {
  const g = new THREE.BoxGeometry(max[0] - min[0], max[1] - min[1], max[2] - min[2])
  g.translate((min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2)
  return g
}

/** 左右対称に 2 つ置く。 */
function mirrored(
  make: (z: number) => THREE.BufferGeometry,
  z: number,
): THREE.BufferGeometry[] {
  return [make(z), make(-z)]
}

/** 2 点を結ぶ板（ガラス・ピラーに使う）。 */
function slab(
  from: readonly [number, number],
  to: readonly [number, number],
  zMin: number,
  zMax: number,
  thickness: number,
): THREE.BufferGeometry {
  const dx = to[0] - from[0]
  const dy = to[1] - from[1]
  const length = Math.hypot(dx, dy)
  const g = new THREE.BoxGeometry(length, thickness, zMax - zMin)
  g.rotateZ(Math.atan2(dy, dx))
  g.translate((from[0] + to[0]) / 2, (from[1] + to[1]) / 2, (zMin + zMax) / 2)
  return g
}

/** 車体（塗装される外板）。 */
export function makeBodyGeometry(): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = [
    // 左右の側面（サイドシル〜ベルトライン）
    ...mirrored(
      (z) => box([-HALF_L + 0.06, 0.2, z - BODY_SKIN], [HALF_L - 0.06, BELT_Y, z + BODY_SKIN]),
      HALF_W - BODY_SKIN,
    ),
    // 床下（アンダーボディ）。内装のフロアへ食い込ませ、面を合わせない
    box([-HALF_L + 0.1, 0.18, -HALF_W + 0.02], [HALF_L - 0.1, 0.33, HALF_W - 0.02]),
    // 前端・後端（バンパーとその上のパネル）。エンジンルームとトランクの前後を塞ぐ。
    // 上下端は内装の側壁（0.26〜0.96）とずらしてある
    box([HALF_L - 0.24, 0.24, -HALF_W + 0.02], [HALF_L, BELT_Y - 0.04, HALF_W - 0.02]),
    box([-HALF_L, 0.24, -HALF_W + 0.02], [-HALF_L + 0.24, BELT_Y - 0.04, HALF_W - 0.02]),
    // ボンネット（前下がり）
    slab([WINDSHIELD_BOTTOM_X, BELT_Y], [HALF_L - 0.12, BELT_Y - 0.06], -0.86, 0.86, 0.09),
    // トランクリッド
    slab([BACKLIGHT_BOTTOM_X, BELT_Y], [-HALF_L + 0.12, BELT_Y - 0.02], -0.86, 0.86, 0.09),
    // ルーフ
    box([BACKLIGHT_TOP_X, ROOF_Y - 0.06, -0.8], [WINDSHIELD_TOP_X, ROOF_Y, 0.8]),
    // A ピラー（フロントガラスの左右端）。
    // 傾いた板は回転のぶん角が伸びるので、上端をルーフより下げておく
    ...mirrored(
      (z) =>
        slab(
          [WINDSHIELD_BOTTOM_X, BELT_Y],
          [WINDSHIELD_TOP_X + 0.03, ROOF_Y - 0.09],
          z - 0.06,
          z + 0.06,
          0.09,
        ),
      0.78,
    ),
    // B ピラー（前後ドアの境目）
    ...mirrored((z) => box([-0.08, BELT_Y, z - 0.05], [0.08, ROOF_Y, z + 0.05]), 0.82),
    // C ピラー（リアガラスの左右端）
    ...mirrored(
      (z) =>
        slab(
          [BACKLIGHT_BOTTOM_X, BELT_Y],
          [BACKLIGHT_TOP_X - 0.03, ROOF_Y - 0.09],
          z - 0.06,
          z + 0.06,
          0.09,
        ),
      0.78,
    ),
    // ルーフサイドレール（左右の窓の上端）
    ...mirrored(
      (z) => box([BACKLIGHT_TOP_X, ROOF_Y - 0.05, z - 0.05], [WINDSHIELD_TOP_X, ROOF_Y, z + 0.05]),
      0.8,
    ),
    // ドアミラー。**車幅の中に収める**（外接寸法を越えると検出枠とずれる）
    ...mirrored((z) => box([0.72, 0.99, z - 0.075], [0.88, 1.12, z + 0.075]), 0.82),
  ]
  const merged = mergeGeometries(parts, false)
  for (const p of parts) p.dispose()
  if (!merged) throw new Error('車体ジオメトリをまとめられませんでした')
  return merged
}

/** ボンネット先端の飾り（前方が一目で分かるようにする）。車体色とは別の明度で塗る */
export function makeNoseGeometry(): THREE.BufferGeometry {
  return box([HALF_L - 0.34, BELT_Y - 0.04, -0.52], [HALF_L - 0.18, BELT_Y + 0.03, 0.52])
}

/** キャビン（いまはルーフごと車体へ含めたので、当たり判定の代表箱としてだけ残す） */
export function makeCabinGeometry(): THREE.BufferGeometry {
  return box([BACKLIGHT_TOP_X, BELT_Y, -0.8], [WINDSHIELD_TOP_X, ROOF_Y, 0.8])
}

/** 窓ガラス。 */
export function makeGlassGeometry(): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = [
    // フロントガラス
    slab(
      [WINDSHIELD_BOTTOM_X, BELT_Y],
      [WINDSHIELD_TOP_X, ROOF_Y - 0.04],
      -0.76,
      0.76,
      0.04,
    ),
    // リアガラス
    slab([BACKLIGHT_BOTTOM_X, BELT_Y], [BACKLIGHT_TOP_X, ROOF_Y - 0.04], -0.76, 0.76, 0.04),
    // サイドガラス（前後ドア × 左右）
    ...mirrored((z) => box([0.12, BELT_Y, z - 0.02], [0.78, ROOF_Y - 0.06, z + 0.02]), 0.85),
    ...mirrored((z) => box([-0.84, BELT_Y, z - 0.02], [-0.16, ROOF_Y - 0.06, z + 0.02]), 0.85),
  ]
  const merged = mergeGeometries(parts, false)
  for (const p of parts) p.dispose()
  if (!merged) throw new Error('窓ガラスをまとめられませんでした')
  return merged
}

/** ダッシュボード上面の高さ [m]。メーターとハンドルの位置はここから決める */
const DASH_TOP_Y = 1.02
/** ダッシュボードの前端・後端 [m]。 */
const DASH_FRONT_X = 1.16
export const DASH_REAR_X = 0.86

/** 前席を車両中心からどれだけ前に置くか [m]。 */
const FRONT_SEAT_X = 0.25

/** 内装の動かない部分（フロア・ダッシュボード・センターコンソール・シート・ ドアトリム・メーターの文字盤）。 */
export function makeInteriorGeometry(): THREE.BufferGeometry {
  const seat = (z: number, at: number): THREE.BufferGeometry[] => [
    // 座面（フロアへ食い込ませる。浮かせると面が一致して点滅する）
    box([at - 0.28, FLOOR_Y - 0.02, z - 0.24], [at + 0.28, FLOOR_Y + 0.16, z + 0.24]),
    // 背もたれ（わずかに後ろへ倒す）
    slab([at - 0.3, FLOOR_Y + 0.1], [at - 0.42, 1.06], z - 0.24, z + 0.24, 0.14),
    // ヘッドレスト
    box([at - 0.46, 1.06, z - 0.14], [at - 0.34, 1.24, z + 0.14]),
  ]

  const parts: THREE.BufferGeometry[] = [
    // フロア。側壁と後ろの壁へ食い込ませる。
    // Z を外板の内面（0.78）に合わせないこと
    box([-1.74, FLOOR_Y - 0.04, -0.79], [DASH_FRONT_X + 0.02, FLOOR_Y, 0.79]),
    // ダッシュボード
    box([DASH_REAR_X, 0.7, -0.74], [DASH_FRONT_X, DASH_TOP_Y, 0.74]),
    box([DASH_FRONT_X - OVERLAP, FLOOR_Y - 0.08, -0.74], [DASH_FRONT_X + 0.06, DASH_TOP_Y - 0.05, 0.74]),
    // メーターの庇（フード）は**置かない**。目からメーターへの視線は
    //   ダッシュボード上面のすぐ下を通るので、庇を立てるとそれ自体が視線を塞ぐ
    // センターコンソール（フロアへ食い込ませる）
    box([-0.35, FLOOR_Y - 0.02, -0.14], [0.5, 0.62, 0.14]),
    // 車内の側壁（ドア内張り）。**外板の内側へ食い込ませること。**
    //   外板と同じ Z に面を置くと、ドア一面が点滅する（実際にそうなった）
    ...mirrored(
      (z) =>
        box(
          [-1.74, FLOOR_Y - 0.08, z - 0.038],
          [DASH_FRONT_X + 0.02, BELT_Y - 0.02, z + 0.038],
        ),
      HALF_W - BODY_SKIN - OVERLAP + 0.005,
    ),
    // 後ろの壁（トランクとの隔壁）。側壁の中で終わらせる
    box([-1.78, FLOOR_Y - 0.06, -0.74], [-1.7, BELT_Y - 0.06, 0.74]),
    // 前席（アイポイントに合わせて前へ出す）
    ...seat(DRIVER_SEAT_Z, FRONT_SEAT_X),
    ...seat(PASSENGER_SEAT_Z, FRONT_SEAT_X),
    // 後席（ベンチ）
    box([-1.24, FLOOR_Y - 0.02, -0.7], [-0.7, FLOOR_Y + 0.18, 0.7]),
    slab([-1.2, FLOOR_Y + 0.12], [-1.34, 1.08], -0.7, 0.7, 0.16),
    // リアパーセルシェルフ。側壁の上面（BELT_Y）と高さを合わせない
    box([-1.68, BELT_Y - 0.08, -0.74], [-1.3, BELT_Y - 0.03, 0.74]),
  ]
  const merged = mergeGeometries(parts, false)
  for (const p of parts) p.dispose()
  if (!merged) throw new Error('内装をまとめられませんでした')
  return merged
}

/** メーターの種類。 */
export const GAUGE_KINDS = ['speed', 'power'] as const
export type GaugeKind = (typeof GAUGE_KINDS)[number]

/** 速度計の目盛りの上限 [km/h]。`maxSpeed` の設定上限（40 m/s = 144 km/h）を覆う */
export const GAUGE_SPEED_MAX_KMH = 160

/** 文字盤アトラスの何段目を貼るかを、テクスチャ座標 v の段へ写す。 */
export function gaugeUvRow(kindIndex: number, rows: number): number {
  return Math.max(0, rows - 1 - kindIndex)
}

/** メーターの文字盤。針と同じ面に置く */
export const GAUGE_SLOTS: ReadonlyArray<{
  readonly center: readonly [number, number, number]
  readonly radius: number
  readonly kind: GaugeKind
}> = [
  {
    center: [DASH_REAR_X - 0.01, DASH_TOP_Y - 0.08, DRIVER_SEAT_Z + 0.09],
    radius: 0.075,
    kind: 'speed',
  },
  {
    center: [DASH_REAR_X - 0.01, DASH_TOP_Y - 0.08, DRIVER_SEAT_Z - 0.09],
    radius: 0.075,
    kind: 'power',
  },
]

/** 速度 [m/s] を速度計の割合 0..1 へ。**目盛りは固定なので `maxSpeed` に依らない** */
export function speedRatio(speedMps: number): number {
  return Math.max(0, Math.min(1, (speedMps * 3.6) / GAUGE_SPEED_MAX_KMH))
}

/** 加速指令 -1..1 をパワーメーターの割合 0..1 へ。 */
export function powerRatio(throttle: number): number {
  return Math.max(0, Math.min(1, (throttle + 1) / 2))
}

/** メーターの文字盤。 */
export function makeGaugeFaceGeometry(radius: number): THREE.BufferGeometry {
  const g = new THREE.CircleGeometry(radius, 48)
  g.rotateY(-Math.PI / 2)
  return g
}

/** ウインカーのインジケーター。 */
export const TURN_INDICATOR_SLOTS: ReadonlyArray<{
  readonly center: readonly [number, number, number]
  readonly side: -1 | 1
}> = [
  { center: [DASH_REAR_X - 0.01, DASH_TOP_Y - 0.02, DRIVER_SEAT_Z - 0.028], side: -1 },
  { center: [DASH_REAR_X - 0.01, DASH_TOP_Y - 0.02, DRIVER_SEAT_Z + 0.028], side: 1 },
]

/** インジケーターの大きさ [m]（三角形の高さ） */
export const TURN_INDICATOR_SIZE = 0.017

/** ウインカーの矢印 1 個。 */
export function makeTurnIndicatorGeometry(size: number): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry()
  g.setAttribute(
    'position',
    new THREE.BufferAttribute(
      new Float32Array([
        0, 0, size,
        0, size * 0.62, -size * 0.45,
        0, -size * 0.62, -size * 0.45,
      ]),
      3,
    ),
  )
  g.setIndex([0, 2, 1])
  g.computeVertexNormals()
  return g
}

/** カーナビの画面（インパネ中央）。 */
export const NAV_SCREEN = {
  center: [DASH_REAR_X - 0.01, 0.86, 0] as readonly [number, number, number],
  width: 0.19,
  height: 0.12,
} as const

/** ナビが見せる範囲の幅 [m]。 */
export const NAV_SPAN_MIN_M = 80
export const NAV_SPAN_MAX_M = 320

/** 速度 [m/s] から、ナビが見せる範囲の幅 [m] を出す。 */
export function navSpanFor(speedMps: number, maxSpeedMps: number): number {
  const t = maxSpeedMps > 0 ? Math.min(1, Math.max(0, speedMps / maxSpeedMps)) : 0
  return NAV_SPAN_MIN_M + (NAV_SPAN_MAX_M - NAV_SPAN_MIN_M) * t
}

/** ナビの画面。法線を運転者の側（-X）へ向ける */
export function makeNavScreenGeometry(width: number, height: number): THREE.BufferGeometry {
  const g = new THREE.PlaneGeometry(width, height)
  g.rotateY(-Math.PI / 2)
  return g
}

/** ステアリングコラムの傾き [rad]（前下がり）。ハンドル面はこれに垂直 */
export const COLUMN_TILT = 0.42
/** ハンドルの中心。 */
export const STEERING_CENTER: readonly [number, number, number] = [
  DASH_REAR_X - 0.06,
  DASH_TOP_Y - 0.14,
  DRIVER_SEAT_Z,
]
/** ハンドルの外径 [m]（実車の 370mm 級） */
export const STEERING_RADIUS = 0.185
/** 舵角 [rad] からハンドルの回転角 [rad] を出す比。 */
export const STEERING_RATIO = 17.0

/** ハンドル（リム・スポーク 3 本・ハブ）。 */
export function makeSteeringGeometry(): THREE.BufferGeometry {
  const rim = new THREE.TorusGeometry(STEERING_RADIUS, 0.018, 8, 28)
  rim.rotateY(Math.PI / 2)

  const parts: THREE.BufferGeometry[] = [rim]
  // スポークは**下・左・右の 3 本**にして、上を空けること。
  //   上に 1 本かかると、リングの中から覗くメーターをちょうど隠す
  for (const angle of [0, Math.PI / 2, -Math.PI / 2]) {
    const spoke = new THREE.BoxGeometry(0.02, STEERING_RADIUS * 0.82, 0.035)
    spoke.translate(0, -STEERING_RADIUS * 0.41, 0)
    spoke.rotateX(angle)
    parts.push(spoke)
  }
  const hub = new THREE.CylinderGeometry(0.052, 0.052, 0.05, 14)
  hub.rotateZ(Math.PI / 2)
  parts.push(hub)

  const merged = mergeGeometries(parts, false)
  for (const p of parts) p.dispose()
  if (!merged) throw new Error('ハンドルをまとめられませんでした')
  return merged
}

/** ペダル。0=アクセル / 1=ブレーキ。**吊り下げ式なので支点は上端** */
export const PEDAL_SLOTS: ReadonlyArray<{
  readonly pivot: readonly [number, number, number]
  readonly kind: 'throttle' | 'brake'
}> = [
  { pivot: [DASH_FRONT_X - 0.06, 0.6, DRIVER_SEAT_Z + 0.11], kind: 'throttle' },
  { pivot: [DASH_FRONT_X - 0.06, 0.62, DRIVER_SEAT_Z - 0.09], kind: 'brake' },
]

/** ペダルの長さ [m]。支点から踏面の下端まで */
const PEDAL_LENGTH = 0.22
/** 目いっぱい踏んだときの回り角 [rad] */
export const PEDAL_TRAVEL = 0.38

/** ペダル 1 枚。**支点が原点**に来るよう下へ伸ばす */
export function makePedalGeometry(): THREE.BufferGeometry {
  const g = new THREE.BoxGeometry(0.03, PEDAL_LENGTH, 0.07)
  g.translate(0, -PEDAL_LENGTH / 2, 0)
  return g
}

/** メーターの針。 */
export function makeNeedleGeometry(radius: number): THREE.BufferGeometry {
  const g = new THREE.BoxGeometry(0.0017, radius * 0.86, 0.0023)
  g.translate(0, (radius * 0.86) / 2, 0)
  return g
}

/** 針が振れる範囲 [rad]。 */
export const NEEDLE_START = -2.2
export const NEEDLE_SWEEP = 4.4

/** タイヤの幅 [m]。リムとスポークはこの中へ収める */
const WHEEL_WIDTH = 0.24
/** ホイール（リム）の半径 [m]。タイヤの内側に見える金属部分 */
const RIM_RADIUS = WHEEL_RADIUS * 0.62

/** 車輪。 */
export function makeWheelGeometry(): THREE.BufferGeometry {
  const tyre = new THREE.CylinderGeometry(WHEEL_RADIUS, WHEEL_RADIUS, WHEEL_WIDTH, 18)
  const parts: THREE.BufferGeometry[] = [tyre]

  // リムとスポークは**タイヤの幅の中へ収める**。はみ出すと車輪が太くなり、
  //   外接寸法の検証が落ちる（見た目にもタイヤから金属が飛び出す）
  const rim = new THREE.CylinderGeometry(RIM_RADIUS, RIM_RADIUS, WHEEL_WIDTH * 0.92, 16)
  parts.push(rim)

  // スポークは**円盤の面（XZ 平面）**に並べること。CylinderGeometry の軸は Y なので、
  //   XY 平面で放射状にすると、軸を寝かせたときに車輪の幅方向へ広がる
  for (let i = 0; i < 5; i++) {
    const spoke = new THREE.BoxGeometry(RIM_RADIUS * 0.9, WHEEL_WIDTH * 0.6, 0.035)
    spoke.translate(RIM_RADIUS * 0.45, 0, 0)
    spoke.rotateY((i * Math.PI * 2) / 5)
    parts.push(spoke)
  }

  const merged = mergeGeometries(parts, false)
  for (const p of parts) p.dispose()
  if (!merged) throw new Error('車輪をまとめられませんでした')
  merged.rotateX(Math.PI / 2)
  return merged
}

/** ナンバープレート 1 枚。法線を +X に向ける（車両ローカルの前方） */
export function makePlateGeometry(width: number, height: number): THREE.BufferGeometry {
  const g = new THREE.PlaneGeometry(width, height)
  g.rotateY(Math.PI / 2)
  return g
}

/** プレート 1 枚のワールド行列。前は前方、後ろは真後ろを向く。 */
export function composePlateMatrix(
  scratch: TransformScratch,
  base: THREE.Matrix4,
  position: readonly [number, number, number],
  yaw: number,
  out: THREE.Matrix4,
): THREE.Matrix4 {
  const p = scratch.plate
  p.position.set(position[0], position[1], position[2])
  p.rotation.set(0, yaw, 0)
  p.scale.setScalar(1)
  p.updateMatrix()
  return out.multiplyMatrices(base, p.matrix)
}

/** 灯体 1 個。位置は `scene/vehicleLights.ts` の `LIGHT_SLOTS` が決める */
export function makeLightGeometry(size: readonly [number, number, number]): THREE.BufferGeometry {
  return new THREE.BoxGeometry(size[0], size[1], size[2])
}

/** 行列を組み立てるときの作業用オブジェクト。毎フレームの new を避ける */
export interface TransformScratch {
  node: THREE.Object3D
  wheel: THREE.Object3D
  light: THREE.Object3D
  plate: THREE.Object3D
  part: THREE.Object3D
  axis: THREE.Vector3
  spin: THREE.Quaternion
  tilt: THREE.Quaternion
}

export function createTransformScratch(): TransformScratch {
  return {
    node: new THREE.Object3D(),
    wheel: new THREE.Object3D(),
    light: new THREE.Object3D(),
    plate: new THREE.Object3D(),
    part: new THREE.Object3D(),
    axis: new THREE.Vector3(),
    spin: new THREE.Quaternion(),
    tilt: new THREE.Quaternion(),
  }
}

/** 車両そのものの姿勢（ENU の位置と方位 → three のワールド行列）。 */
export function composeVehicleMatrix(
  scratch: TransformScratch,
  enuX: number,
  enuY: number,
  heading: number,
  out: THREE.Matrix4,
): THREE.Matrix4 {
  const n = scratch.node
  n.position.set(enuX, 0, -enuY)
  n.rotation.set(0, heading, 0)
  n.scale.setScalar(1)
  n.updateMatrix()
  return out.copy(n.matrix)
}

/** 灯体 1 個のワールド行列。車体に固定なので回転は持たない。 */
export function composeLightMatrix(
  scratch: TransformScratch,
  base: THREE.Matrix4,
  position: readonly [number, number, number],
  out: THREE.Matrix4,
): THREE.Matrix4 {
  const l = scratch.light
  l.position.set(position[0], position[1], position[2])
  l.rotation.set(0, 0, 0)
  l.scale.setScalar(1)
  l.updateMatrix()
  return out.multiplyMatrices(base, l.matrix)
}

/** 車輪 1 本のワールド行列。 */
export function composeWheelMatrix(
  scratch: TransformScratch,
  base: THREE.Matrix4,
  index: number,
  steer: number,
  roll: number,
  out: THREE.Matrix4,
): THREE.Matrix4 {
  const spec = WHEEL_OFFSETS[index]
  const w = scratch.wheel
  w.position.set(spec.position[0], spec.position[1], spec.position[2])
  w.rotation.set(0, spec.steered ? steer : 0, roll)
  w.scale.setScalar(1)
  w.updateMatrix()
  return out.multiplyMatrices(base, w.matrix)
}

/** 車体に固定された部品（内装・文字盤）のワールド行列。 */
export function composeFixedMatrix(
  scratch: TransformScratch,
  base: THREE.Matrix4,
  position: readonly [number, number, number],
  out: THREE.Matrix4,
): THREE.Matrix4 {
  const p = scratch.part
  p.position.set(position[0], position[1], position[2])
  p.rotation.set(0, 0, 0)
  p.scale.setScalar(1)
  p.updateMatrix()
  return out.multiplyMatrices(base, p.matrix)
}

/** ウインカーの矢印 1 個のワールド行列。右側は左右を反転させる。 */
export function composeTurnIndicatorMatrix(
  scratch: TransformScratch,
  base: THREE.Matrix4,
  index: number,
  out: THREE.Matrix4,
): THREE.Matrix4 {
  const spec = TURN_INDICATOR_SLOTS[index]
  const n = scratch.part
  n.position.set(spec.center[0], spec.center[1], spec.center[2])
  // X 軸まわりに 180 度回すと、法線を保ったまま指す向きだけが裏返る。
  // 素の矢印は +Z（運転者から見て右）を指すので、**裏返すのは左側**
  n.rotation.set(spec.side < 0 ? Math.PI : 0, 0, 0)
  n.scale.setScalar(1)
  n.updateMatrix()
  return out.multiplyMatrices(base, n.matrix)
}

/** 舵角 [rad] に対するハンドルの回転角 [rad]。左へ切ると反時計回りに回る。 */
export function steeringAngle(steer: number): number {
  return steer * STEERING_RATIO
}

/** ハンドルのワールド行列。 */
export function composeSteeringMatrix(
  scratch: TransformScratch,
  base: THREE.Matrix4,
  steer: number,
  out: THREE.Matrix4,
): THREE.Matrix4 {
  const s = scratch.part
  scratch.spin.setFromAxisAngle(scratch.axis.set(1, 0, 0), -steeringAngle(steer))
  scratch.tilt.setFromAxisAngle(scratch.axis.set(0, 0, 1), -COLUMN_TILT)
  s.position.set(STEERING_CENTER[0], STEERING_CENTER[1], STEERING_CENTER[2])
  s.quaternion.copy(scratch.tilt).multiply(scratch.spin)
  s.scale.setScalar(1)
  s.updateMatrix()
  return out.multiplyMatrices(base, s.matrix)
}

/** ペダルの踏み込み 0..1。アクセルは加速指令の正、ブレーキは負を見る。 */
export function pedalPress(kind: 'throttle' | 'brake', throttle: number): number {
  const v = kind === 'throttle' ? throttle : -throttle
  return Math.max(0, Math.min(1, v))
}

/** ペダル 1 枚のワールド行列。踏むと踏面が前方へ回る。 */
export function composePedalMatrix(
  scratch: TransformScratch,
  base: THREE.Matrix4,
  index: number,
  throttle: number,
  out: THREE.Matrix4,
): THREE.Matrix4 {
  const spec = PEDAL_SLOTS[index]
  const p = scratch.part
  p.position.set(spec.pivot[0], spec.pivot[1], spec.pivot[2])
  // Z 軸まわりに正で回すと、支点から下（-Y）へ伸びた踏面が前方（+X）へ出る
  p.rotation.set(0, 0, pedalPress(spec.kind, throttle) * PEDAL_TRAVEL)
  p.scale.setScalar(1)
  p.updateMatrix()
  return out.multiplyMatrices(base, p.matrix)
}

/** 針の振れ角 [rad]。0..1 の割合を文字盤の範囲へ写す。 */
export function needleAngle(ratio: number): number {
  return NEEDLE_START + Math.max(0, Math.min(1, ratio)) * NEEDLE_SWEEP
}

/** メーターの針のワールド行列。文字盤の面（法線 +X）に沿って回る。 */
export function composeNeedleMatrix(
  scratch: TransformScratch,
  base: THREE.Matrix4,
  index: number,
  ratio: number,
  out: THREE.Matrix4,
): THREE.Matrix4 {
  const spec = GAUGE_SLOTS[index]
  const n = scratch.part
  // 針は文字盤の**手前**（目に近い側 = X が小さいほう）へ置くこと。
  //   奥へ置くと文字盤に隠れる（+0.006 にしていて見えにくかった）
  n.position.set(spec.center[0] - 0.008, spec.center[1], spec.center[2])
  n.rotation.set(needleAngle(ratio), 0, 0)
  n.scale.setScalar(1)
  n.updateMatrix()
  return out.multiplyMatrices(base, n.matrix)
}
