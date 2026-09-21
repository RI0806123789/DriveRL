/** 車両の形と姿勢の計算。**React から切り離した純粋モジュール。** */

import * as THREE from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'

/**
 * 車両ローカル座標は **+X が前方・+Y が上・+Z が右**。
 * 右が +Z なのは `composeVehicleMatrix` が ENU の方位をそのまま three の Y 回転に
 * 入れており、three の z が ENU の -y だから（`cameraMath.DRIVER_RIGHT` も +Z 側）。
 * 日本車なので運転席は +Z 側に置く。
 *
 * ★ **運転席から前を見たときも、画面の右は +Z のまま。**
 * ここを「運転者から見ると +Z は左」と取り違えて、メーターの針を逆回りにし、
 * ウインカーの矢印を左右あべこべに置いた。
 * `verify:vehicles` は実際の運転席カメラへ投影して左右を決めており、
 * **思い込みを書かないこと**（`screenX()`）。
 */

/** 外接寸法 [m]。**backend の `config.VEHICLE_*` と揃える。ここを越える部品を作らない** */
export const VEHICLE_LENGTH = 4.4
export const VEHICLE_WIDTH = 1.8
export const VEHICLE_HEIGHT = 1.45

const HALF_L = VEHICLE_LENGTH / 2
const HALF_W = VEHICLE_WIDTH / 2

/** 車輪の半径 [m]。転がり角の計算にも使う */
export const WHEEL_RADIUS = 0.34

/**
 * 各ジオメトリの外接箱の中心（車両ローカル）。**手で決めた値ではなく実測値**で、
 * `npm run verify:vehicles` が形を変えたときに気づくための控えとして置いてある。
 * ★ `Z` が 0 でなくなったら、左右非対称な部品が混ざった合図
 * （`mirrored()` に片側へ寄った形を渡すと起きる）。
 */
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

/**
 * 部品どうしを重ねるときの食い込み量 [m]。
 * ★ **面をぴったり合わせないこと。** 同じ深度に 2 つの面が来ると、どちらが手前か
 * 決まらず**フレームごとに色が入れ替わって点滅する**（Z ファイティング）。
 * 内装と外板、ダッシュボードとバルクヘッドなど、隣り合う部品はこの量だけ
 * 食い込ませて、境目の面が重ならないようにする。
 */
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

/**
 * 左右対称に 2 つ置く。`z` は右側（+Z）の値。
 * ★ **`make` は `z` を中心とした左右対称な形を返すこと。** 片側へ寄った形
 * （例: `z - 0.03` 〜 `z + 0.12`）を渡すと、符号を変えただけでは鏡像にならず、
 * 車体が左右非対称になる（重心の Z がずれるので verify が捕まえる）。
 */
function mirrored(
  make: (z: number) => THREE.BufferGeometry,
  z: number,
): THREE.BufferGeometry[] {
  return [make(z), make(-z)]
}

/**
 * 2 点を結ぶ板（ガラス・ピラーに使う）。XY 平面での傾きを持ち、Z 方向へ幅を持つ。
 * `thickness` は板の厚み。
 */
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

/**
 * 車体（塗装される外板）。ロッカー・ドア・ボンネット・トランク・ルーフ・ピラー・
 * ドアミラーを 1 つへまとめる。**instancedMesh を部品ごとに増やさないため。**
 */
export function makeBodyGeometry(): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = [
    // ★ 車体は**中空の殻**にすること。中身の詰まった箱で作ると、その上面が
    //   運転席の目の 0.24m 下に広がり、**室内の下半分を覆って何も見えなくなる**
    //   （ハンドルの上端しか映らなかった原因がこれ）。外から見た形は変わらない。
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
    // ★ 傾いた板は回転のぶん角が伸びるので、上端をルーフより下げておく
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

/**
 * 窓ガラス。フロント・リア・サイド 4 枚をまとめる。
 * **半透明で描くので、車体とは別の instancedMesh にすること。**
 */
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
/**
 * ダッシュボードの前端・後端 [m]。
 * ★ **後端をアイポイント（`cameraMath.DRIVER_FORWARD` = 0.35m）へ近づけすぎないこと。**
 * 近いほど運転席視点の下半分が壁で埋まる。実車は目からダッシュボードまで 0.5m 以上
 * 空いている（最初 0.52m に置いたら、目の 0.17m 先に壁が立って前が見えなくなった）。
 */
const DASH_FRONT_X = 1.16
export const DASH_REAR_X = 0.86

/**
 * 前席を車両中心からどれだけ前に置くか [m]。
 * アイポイントは座面の少し前・上にあるので、**座席もそこへ合わせる**
 * （合わせないと運転者が後席に座っていることになる）。
 */
const FRONT_SEAT_X = 0.25

/**
 * 内装の動かない部分（フロア・ダッシュボード・センターコンソール・シート・
 * ドアトリム・メーターの文字盤）。
 */
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
    // ★ ここから下は**どの 2 つも面が一致しないように**置いてある。
    //   合わせると境目が点滅する（Z ファイティング）。隣り合うものは OVERLAP ぶん
    //   食い込ませ、内部に隠れた面が深度を奪い合わないようにする。
    // フロア。側壁と後ろの壁へ食い込ませる。
    // ★ Z を外板の内面（0.78）に合わせないこと
    box([-1.74, FLOOR_Y - 0.04, -0.79], [DASH_FRONT_X + 0.02, FLOOR_Y, 0.79]),
    // ダッシュボード
    box([DASH_REAR_X, 0.7, -0.74], [DASH_FRONT_X, DASH_TOP_Y, 0.74]),
    // ★ 前方の隔壁（バルクヘッド）。**これが無いと運転席から背景が透ける。**
    //   外板は裏面が描かれないので、車内は内装だけで閉じておく必要がある。
    //   ダッシュボードの中で終わらせて、上面と前面を一致させない
    box([DASH_FRONT_X - OVERLAP, FLOOR_Y - 0.08, -0.74], [DASH_FRONT_X + 0.06, DASH_TOP_Y - 0.05, 0.74]),
    // ★ メーターの庇（フード）は**置かない**。目からメーターへの視線は
    //   ダッシュボード上面のすぐ下を通るので、庇を立てるとそれ自体が視線を塞ぐ
    // センターコンソール（フロアへ食い込ませる）
    box([-0.35, FLOOR_Y - 0.02, -0.14], [0.5, 0.62, 0.14]),
    // ★ 車内の側壁（ドア内張り）。**外板の内側へ食い込ませること。**
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

/**
 * メーターの種類。**並びは文字盤アトラスの段と同じ**（`gaugeTexture.ts`）。
 * EV なので回転計は持たず、出力と回生を 1 本の針で示すパワーメーターにする。
 */
export const GAUGE_KINDS = ['speed', 'power'] as const
export type GaugeKind = (typeof GAUGE_KINDS)[number]

/** 速度計の目盛りの上限 [km/h]。`maxSpeed` の設定上限（40 m/s = 144 km/h）を覆う */
export const GAUGE_SPEED_MAX_KMH = 160

/**
 * 文字盤アトラスの何段目を貼るかを、テクスチャ座標 v の段へ写す。
 * ★ **Canvas は上から順に描くが、v は下から数える**（`plateUvRow` と同じ罠）。
 * 素通しにすると速度計の枠に**パワーメーターの絵**が貼られ、針だけが速度で動く。
 * 停車中でもパワーメーターの針は中央（真上）を指すので、
 * **速度計が常に 80km/h を指しているように見える**という形で表に出た。
 */
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

/**
 * 加速指令 -1..1 をパワーメーターの割合 0..1 へ。**0.5 が中央（出力も回生もゼロ）**。
 * 負が回生（CHARGE）、正が出力（POWER）で、`frame.vehicles[].throttle` をそのまま使う。
 */
export function powerRatio(throttle: number): number {
  return Math.max(0, Math.min(1, (throttle + 1) / 2))
}

/**
 * メーターの文字盤。法線を車両後方（運転者の側）へ向ける。
 * ★ **`rotateY` の前に UV を決めておくこと。** 回してから貼ると、文字盤の
 * 上下左右が入れ替わって目盛りと針がずれる。
 */
export function makeGaugeFaceGeometry(radius: number): THREE.BufferGeometry {
  const g = new THREE.CircleGeometry(radius, 48)
  g.rotateY(-Math.PI / 2)
  return g
}

/**
 * ウインカーのインジケーター。**2 つのメーターの間の上部**に、左右へ開いて並べる。
 * `side` は `frame.vehicles[].turnSignal` の符号に対応する（-1 が左）。
 * ★ **文字盤の円と重ならない高さに置くこと。** メーターは半径 75mm で上端が
 * ダッシュボード上面のすぐ下まで来るので、間の上部は隙間が狭い。
 * ここはハンドルのリングの内側だが、**スポークを下・左・右にしてあるので上は空いている**。
 */
export const TURN_INDICATOR_SLOTS: ReadonlyArray<{
  readonly center: readonly [number, number, number]
  readonly side: -1 | 1
}> = [
  { center: [DASH_REAR_X - 0.01, DASH_TOP_Y - 0.02, DRIVER_SEAT_Z - 0.028], side: -1 },
  { center: [DASH_REAR_X - 0.01, DASH_TOP_Y - 0.02, DRIVER_SEAT_Z + 0.028], side: 1 },
]

/** インジケーターの大きさ [m]（三角形の高さ） */
export const TURN_INDICATOR_SIZE = 0.017

/**
 * ウインカーの矢印 1 個。平面 X=0 の上に、**+Z（運転者から見て右）を指す**三角形を作る。
 * 左側は行列側で X 軸まわりに反転させる。
 */
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

/**
 * カーナビの画面（インパネ中央）。**運転席から見てハンドルのリムの外**に来る位置。
 * 中央（Z=0）なので運転席（Z=+0.36）からは右寄りに見える。
 */
export const NAV_SCREEN = {
  center: [DASH_REAR_X - 0.01, 0.86, 0] as readonly [number, number, number],
  width: 0.19,
  height: 0.12,
} as const

/**
 * ナビが見せる範囲の幅 [m]。**停車で寄り、速度が上がるほど引く。**
 * 速い車ほど先を見せたいので、実車のナビと同じ振る舞いにしてある。
 */
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
/**
 * ハンドルの中心。
 * ★ **低く置きすぎると運転席視点の画角から外れて見えなくなる。**
 * 運転席カメラの垂直画角は約 41 度（`DRIVER_FOV_DEG` 68 度を横長の画面で割ったもの）で、
 * その半分より下に外れると映らない。最初 `DASH_TOP_Y - 0.18` に置いたら
 * 目から 43 度下になり、ハンドルが一度も映らなかった。
 */
export const STEERING_CENTER: readonly [number, number, number] = [
  DASH_REAR_X - 0.06,
  DASH_TOP_Y - 0.14,
  DRIVER_SEAT_Z,
]
/** ハンドルの外径 [m]（実車の 370mm 級） */
export const STEERING_RADIUS = 0.185
/**
 * 舵角 [rad] からハンドルの回転角 [rad] を出す比。
 * 実車のステアリングギア比（最大舵角 0.55rad でおよそ 1.5 回転）に合わせる。
 */
export const STEERING_RATIO = 17.0

/**
 * ハンドル（リム・スポーク 3 本・ハブ）。
 * **軸を +X（車両前方）に向けて作る。** 傾きは行列側で掛けるので、
 * ここで傾けると二重に傾く。
 */
export function makeSteeringGeometry(): THREE.BufferGeometry {
  const rim = new THREE.TorusGeometry(STEERING_RADIUS, 0.018, 8, 28)
  rim.rotateY(Math.PI / 2)

  const parts: THREE.BufferGeometry[] = [rim]
  // ★ スポークは**下・左・右の 3 本**にして、上を空けること。
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

/**
 * メーターの針。**根元が原点**で、文字盤の面に沿って伸びる。
 * 文字盤に目盛りを焼く前は、幅 6mm だと暗い車内で判別できず 12mm まで太らせていた。
 * 目盛り入りの明るい文字盤になってコントラストが付いたので、実車に近い 1.7mm まで
 * 細くしてある。**これ以上細くすると、0.57m 先では 1 画素を割って消える。**
 */
export function makeNeedleGeometry(radius: number): THREE.BufferGeometry {
  const g = new THREE.BoxGeometry(0.0017, radius * 0.86, 0.0023)
  g.translate(0, (radius * 0.86) / 2, 0)
  return g
}

/**
 * 針が振れる範囲 [rad]。**運転者から見て左下から右下へ（時計回りに）**回る。
 * ★ 針は X 軸まわりに角度 θ で回り、針先は `(0, L·cosθ, L·sinθ)`。
 * **運転者から見て +Z は右**なので θ が増えるほど右へ回る。つまり
 * **開始が負・振れ幅が正**で時計回り。逆にすると、速度が上がるほど針が左へ動く。
 */
export const NEEDLE_START = -2.2
export const NEEDLE_SWEEP = 4.4

/** タイヤの幅 [m]。リムとスポークはこの中へ収める */
const WHEEL_WIDTH = 0.24
/** ホイール（リム）の半径 [m]。タイヤの内側に見える金属部分 */
const RIM_RADIUS = WHEEL_RADIUS * 0.62

/**
 * 車輪。シリンダーの軸（+Y）を車体左右方向（+Z）へ向ける。
 * タイヤ・リム・スポークを 1 つへまとめる（部品ごとに instancedMesh を増やさない）。
 */
export function makeWheelGeometry(): THREE.BufferGeometry {
  const tyre = new THREE.CylinderGeometry(WHEEL_RADIUS, WHEEL_RADIUS, WHEEL_WIDTH, 18)
  const parts: THREE.BufferGeometry[] = [tyre]

  // ★ リムとスポークは**タイヤの幅の中へ収める**。はみ出すと車輪が太くなり、
  //   外接寸法の検証が落ちる（見た目にもタイヤから金属が飛び出す）
  const rim = new THREE.CylinderGeometry(RIM_RADIUS, RIM_RADIUS, WHEEL_WIDTH * 0.92, 16)
  parts.push(rim)

  // ★ スポークは**円盤の面（XZ 平面）**に並べること。CylinderGeometry の軸は Y なので、
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

/**
 * ハンドルのワールド行列。
 * **コラムの傾きと、軸まわりの回転を分けて掛けること。** ジオメトリ側を傾けると、
 * 回転軸まで一緒に傾いて「斜めに首を振る」動きになる。
 */
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
  // ★ 針は文字盤の**手前**（目に近い側 = X が小さいほう）へ置くこと。
  //   奥へ置くと文字盤に隠れる（+0.006 にしていて見えにくかった）
  n.position.set(spec.center[0] - 0.008, spec.center[1], spec.center[2])
  n.rotation.set(needleAngle(ratio), 0, 0)
  n.scale.setScalar(1)
  n.updateMatrix()
  return out.multiplyMatrices(base, n.matrix)
}
