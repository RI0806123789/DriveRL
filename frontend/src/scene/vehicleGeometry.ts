/** 車両の形と姿勢の計算。**React から切り離した純粋モジュール。** */

import * as THREE from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'

/** 車両ローカル座標は **+X が前方・+Y が上・+Z が右**。 */

/** 外接寸法 [m]。**backend の `config.VEHICLE_*` と揃える。ここを越える部品を作らない** */
export const VEHICLE_LENGTH = 4.4
export const VEHICLE_WIDTH = 1.8
export const VEHICLE_HEIGHT = 1.45

const HALF_L = VEHICLE_LENGTH / 2

/** 車輪の半径 [m]。転がり角の計算にも使う */
export const WHEEL_RADIUS = 0.34

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
export const FLOOR_Y = 0.34
export const ROOF_Y = VEHICLE_HEIGHT
/** ベルトライン（窓の下端）[m] */
export const BELT_Y = 0.98
/** 部品どうしを重ねるときの食い込み量 [m]。 */
export const OVERLAP = 0.04

/** 平面形。前後端と、ドアの面での半幅 [m]。フェンダーはここから少しだけ張り出す */
export const BODY_FRONT_X = HALF_L - 0.005
export const BODY_REAR_X = -HALF_L + 0.005
export const BODY_HALF_W = 0.885

/** フロントガラスの下端・上端の X。寝かせるほど差が開く */
export const WINDSHIELD_BOTTOM_X = 1.15
export const WINDSHIELD_TOP_X = 0.55
/** リアガラス */
export const BACKLIGHT_BOTTOM_X = -1.1
export const BACKLIGHT_TOP_X = -0.6
/** 前後のガラスの上端の高さ [m] */
export const GLASS_TOP_Y = ROOF_Y - 0.04
/** 前後のガラスの下端・上端での半幅 [m]（上ほど細い） */
export const GLASS_BOTTOM_HALF_W = 0.785
export const GLASS_TOP_HALF_W = 0.685

/** 横窓の面。ベルトラインで z = SIDE_GLASS_Z、上へ行くほど内側へ倒れる */
export const SIDE_GLASS_Z = 0.8
export const SIDE_GLASS_LEAN = 0.25
/** 横窓の上端の高さ [m] */
export const SIDE_GLASS_TOP_Y = 1.372

/** 高さ y での横窓の面の |z| */
export function sideGlassZ(y: number): number {
  return SIDE_GLASS_Z - (y - BELT_Y) * SIDE_GLASS_LEAN
}

/** 前後のドアの X の範囲 [後端, 前端]。B ピラーの X の範囲 */
export const FRONT_DOOR_X: readonly [number, number] = [0.02, 1.0]
export const REAR_DOOR_X: readonly [number, number] = [-0.98, -0.02]
export const B_PILLAR_X: readonly [number, number] = [-0.07, 0.07]
/** ドアの下端の高さ [m]（その下はサイドシル） */
export const DOOR_BOTTOM_Y = 0.3

/** 開け閉めする左後席ドアの蝶番（鉛直軸の X と Z） */
export const TAXI_DOOR_HINGE: readonly [number, number] = [REAR_DOOR_X[1] - 0.01, -(BODY_HALF_W - 0.02)]
/** ドアを開き切ったときの角度 [rad] */
export const TAXI_DOOR_OPEN = 1.15

/** ダッシュボード上面の高さ [m]。メーターとハンドルの位置はここから決める */
export const DASH_TOP_Y = 1.02
/** ダッシュボードの前端・後端 [m]。 */
export const DASH_FRONT_X = 1.16
export const DASH_REAR_X = 0.86

/** 前席を車両中心からどれだけ前に置くか [m]。 */
export const FRONT_SEAT_X = 0.25

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

/** POWER の針の時定数 [s]。指令は 20Hz の段差で届き、方策の探索ノイズも乗るので慣性を付ける */
export const POWER_NEEDLE_TAU_S = 0.15

/** 針を目標の割合へ指数的に寄せる。間隔が長くても行き過ぎない */
export function dampNeedle(current: number, target: number, dtSec: number, tauSec: number): number {
  if (!(tauSec > 0)) return target
  const k = 1 - Math.exp(-Math.max(0, dtSec) / tauSec)
  return current + (target - current) * k
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
  { pivot: [DASH_FRONT_X - 0.06, 0.6, DRIVER_SEAT_Z + 0.1], kind: 'throttle' },
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

/** 車輪 1 本のワールド行列。**左の車輪は Y 軸まわりに半回転**させて、ホイールの面を外へ向ける */
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
  const left = spec.position[2] < 0
  w.position.set(spec.position[0], spec.position[1], spec.position[2])
  // 半回転させると回転軸の向きも裏返るので、転がる角度の符号を戻す
  w.rotation.set(0, (spec.steered ? steer : 0) + (left ? Math.PI : 0), left ? -roll : roll)
  w.scale.setScalar(1)
  w.updateMatrix()
  return out.multiplyMatrices(base, w.matrix)
}

/** ブレーキキャリパーのワールド行列。転がらず、舵角だけで向きが変わる */
export function composeCaliperMatrix(
  scratch: TransformScratch,
  base: THREE.Matrix4,
  index: number,
  steer: number,
  angle: number,
  out: THREE.Matrix4,
): THREE.Matrix4 {
  const spec = WHEEL_OFFSETS[index]
  const w = scratch.wheel
  const left = spec.position[2] < 0
  w.position.set(spec.position[0], spec.position[1], spec.position[2])
  // 左は半回転で前後も入れ替わるので、置く角度を鏡に写して同じ「後ろ寄りの上」に揃える
  w.rotation.set(0, (spec.steered ? steer : 0) + (left ? Math.PI : 0), left ? Math.PI - 2 * angle : 0)
  w.scale.setScalar(1)
  w.updateMatrix()
  return out.multiplyMatrices(base, w.matrix)
}

/** 車体の傾きを掛けた行列（`tilt` は `vehicleMotion.composeTiltMatrix`）。車輪以外はこれを親にする */
export function composeBodyMatrix(base: THREE.Matrix4, tilt: THREE.Matrix4, out: THREE.Matrix4): THREE.Matrix4 {
  return out.multiplyMatrices(base, tilt)
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
