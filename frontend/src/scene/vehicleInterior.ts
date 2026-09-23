/** 車内の動かない部分（床・ダッシュボード・座席・内張り・天井）。**React から切り離した純粋モジュール。** */

import * as THREE from 'three'
import { MeshBuilder, boxGeometry, orientedBox, rodGeometry, slabGeometry, type PartStyle, type Vec3 } from './meshBuilder.ts'
import {
  BACKLIGHT_BOTTOM_X,
  BACKLIGHT_TOP_X,
  BELT_Y,
  B_PILLAR_X,
  COLUMN_TILT,
  DASH_FRONT_X,
  DASH_REAR_X,
  DASH_TOP_Y,
  DRIVER_SEAT_Z,
  FLOOR_Y,
  FRONT_DOOR_X,
  FRONT_SEAT_X,
  GLASS_BOTTOM_HALF_W,
  GLASS_TOP_HALF_W,
  GLASS_TOP_Y,
  NAV_SCREEN,
  OVERLAP,
  PASSENGER_SEAT_Z,
  REAR_DOOR_X,
  SIDE_GLASS_TOP_Y,
  STEERING_CENTER,
  WINDSHIELD_BOTTOM_X,
  WINDSHIELD_TOP_X,
  sideGlassZ,
} from './vehicleGeometry.ts'

/** 頂点色（内装のマテリアルの色に掛ける明るさ） */
const FABRIC: Vec3 = [1, 1, 1]
const DARK: Vec3 = [0.5, 0.5, 0.52]
const CARPET: Vec3 = [0.72, 0.72, 0.74]
const LINER: Vec3 = [1.75, 1.72, 1.66]
const ACCENT: Vec3 = [0.32, 0.32, 0.34]
const BUTTON: Vec3 = [1.3, 1.3, 1.32]

/** 内張りの面（外板の内側）の |z| と厚み。**外板と同じ Z に面を置かないこと** */
const TRIM_Z = 0.805
const TRIM_HALF = 0.038

/** ルームミラー（小さく保つ。運転席の視界を塞がない）。鏡面を運転席へ向けて置く */
export const ROOM_MIRROR = {
  center: [WINDSHIELD_TOP_X + 0.07, 1.33, 0.02] as const,
  width: 0.15,
  height: 0.042,
  /** 運転席へ向ける角度 [rad]（Y 軸まわり） */
  yaw: 0.45,
} as const

/** 車室の後ろの端（後ろの壁）[m]。リアガラスの付け根より少し後ろ */
const CABIN_REAR_X = BACKLIGHT_BOTTOM_X - 0.06

/** ホイールハウスを避ける範囲。**アーチの切り欠きより低い所には、外側（|z| が大きい側）へ内装を置かない**（タイヤの上から見える） */
const WELL_FRONT_X = 1.02
const WELL_REAR_X = -1.02
/** ホイールハウスの内張りの奥の壁の |z|（`vehicleBody` と同じ） */
const WELL_INNER_Z = { front: 0.515, rear: 0.59 } as const
/** これより上ならアーチより高い [m] */
const ABOVE_WELL_Y = 0.71

/** エアコンの吹き出し口（ダッシュボードの後ろの面）。中央の 2 つはナビとメーターの間 */
export const AC_VENTS: ReadonlyArray<{ readonly z: number; readonly y: number; readonly w: number; readonly h: number }> = [
  { z: 0.135, y: 0.96, w: 0.066, h: 0.05 },
  { z: -0.135, y: 0.96, w: 0.066, h: 0.05 },
  { z: 0.665, y: 0.95, w: 0.09, h: 0.055 },
  { z: -0.665, y: 0.95, w: 0.09, h: 0.055 },
]

/** シフトレバーの頭 */
export const SHIFT_KNOB: readonly [number, number, number] = [0.33, 0.705, 0]

function add(b: MeshBuilder, g: THREE.BufferGeometry, style: PartStyle): void {
  b.geometry(g, style)
  g.dispose()
}

/** 前席 1 脚（座面・背もたれ・脇の張り出し・ヘッドレストとステー） */
function frontSeat(b: MeshBuilder, z: number, at: number): void {
  add(b, boxGeometry([at - 0.28, FLOOR_Y - 0.02, z - 0.24], [at + 0.28, FLOOR_Y + 0.16, z + 0.24]), { color: FABRIC })
  for (const side of [-1, 1]) {
    add(b, boxGeometry([at - 0.26, FLOOR_Y + 0.12, z + side * 0.24 - 0.035], [at + 0.24, FLOOR_Y + 0.205, z + side * 0.24 + 0.035]), { color: FABRIC })
    add(b, slabGeometry([at - 0.28, FLOOR_Y + 0.16], [at - 0.39, 0.97], z + side * 0.21 - 0.035, z + side * 0.21 + 0.035, 0.19), { color: FABRIC })
  }
  add(b, slabGeometry([at - 0.31, FLOOR_Y + 0.1], [at - 0.42, 1.04], z - 0.2, z + 0.2, 0.13), { color: FABRIC })
  add(b, boxGeometry([at - 0.475, 1.085, z - 0.13], [at - 0.375, 1.255, z + 0.13]), { color: FABRIC })
  for (const side of [-1, 1]) add(b, rodGeometry([at - 0.425, 1.02, z + side * 0.07], [at - 0.425, 1.1, z + side * 0.07], 0.007), { color: ACCENT })
}

/** ドアの内張り 1 枚（肘掛け・内側のドアハンドル付き）。`x0 < x1` */
function doorTrim(b: MeshBuilder, sign: number, x0: number, x1: number, door: boolean): void {
  const zc = sign * TRIM_Z
  add(b, boxGeometry([x0, FLOOR_Y - 0.08, zc - TRIM_HALF], [x1, BELT_Y - 0.02, zc + TRIM_HALF]), { color: FABRIC, door })
  // 窓の下の縁。ガラスの面より内側へ収める（外へ出すと窓越しに内張りが外へ飛び出して見える）
  const capOuter = sideGlassZ(BELT_Y + 0.01) - 0.012
  add(
    b,
    boxGeometry([x0 + 0.003, BELT_Y - 0.03, sign > 0 ? capOuter - 0.055 : -capOuter], [x1 - 0.003, BELT_Y + 0.006, sign > 0 ? capOuter : -capOuter + 0.055]),
    { color: DARK, door },
  )
  const inner = zc - sign * TRIM_HALF
  add(
    b,
    boxGeometry(
      [x0 + 0.08, 0.62, sign > 0 ? inner - 0.055 : inner - 0.004],
      [x1 - 0.1, 0.665, sign > 0 ? inner + 0.004 : inner + 0.055],
    ),
    { color: FABRIC, door },
  )
  add(
    b,
    boxGeometry(
      [x1 - 0.2, 0.79, sign > 0 ? inner - 0.012 : inner - 0.002],
      [x1 - 0.12, 0.815, sign > 0 ? inner + 0.002 : inner + 0.012],
    ),
    { color: ACCENT, door },
  )
}

/** 内側から見たピラーと天井。外板は裏面が描かれないので、内側にも面を置く */
function addLinings(b: MeshBuilder): void {
  const ws = (t: number): [number, number] => [
    WINDSHIELD_BOTTOM_X + (WINDSHIELD_TOP_X - WINDSHIELD_BOTTOM_X) * t,
    BELT_Y + (GLASS_TOP_Y - BELT_Y) * t,
  ]
  const bl = (t: number): [number, number] => [
    BACKLIGHT_BOTTOM_X + (BACKLIGHT_TOP_X - BACKLIGHT_BOTTOM_X) * t,
    BELT_Y + (GLASS_TOP_Y - BELT_Y) * t,
  ]
  const halfW = (t: number) => GLASS_BOTTOM_HALF_W + (GLASS_TOP_HALF_W - GLASS_BOTTOM_HALF_W) * t
  // 天井。前後はガラスの上端、横は横窓の上端まで
  const rows: Vec3[][] = []
  const n = 6
  for (let r = 0; r <= n; r++) {
    const t = r / n
    const [xf] = ws(1)
    const [xr] = bl(1)
    const x = xf - 0.03 + (xr + 0.03 - (xf - 0.03)) * t
    const yc = 1.405 - 0.012 * (1 - Math.sin(Math.PI * t))
    const row: Vec3[] = []
    for (const z of [-0.69, -0.6, -0.3, 0, 0.3, 0.6, 0.69]) {
      const a = Math.abs(z)
      const y = a <= 0.6 ? yc - 0.006 * (a / 0.6) ** 2 : SIDE_GLASS_TOP_Y - 0.004 + (yc - 0.006 - SIDE_GLASS_TOP_Y + 0.004) * (1 - (a - 0.6) / 0.09)
      row.push([x, y, z])
    }
    rows.push(row)
  }
  b.grid(rows, [0, -1, 0], { color: LINER })

  // A ピラー・C ピラーの内側（ガラスの縁に沿った帯）
  for (const sign of [1, -1]) {
    for (const [edge, backward] of [
      [ws, false],
      [bl, true],
    ] as const) {
      const strip: Vec3[][] = []
      for (let k = 0; k <= 4; k++) {
        const t = k / 4
        const [x, y] = edge(t)
        const w = halfW(t)
        const sideZ = sideGlassZ(y) - 0.022
        const shift = backward ? 0.02 : -0.02
        strip.push([
          [x + shift, y - 0.012, sign * (w - 0.035)],
          [x + shift * 2.5, y - 0.02, sign * Math.min(sideZ, w - 0.02)],
        ])
      }
      b.grid(strip, [backward ? 0.5 : -0.5, -0.4, -sign], { color: LINER })
    }
    // B ピラーの内側
    const bx0 = B_PILLAR_X[0] - 0.01
    const bx1 = B_PILLAR_X[1] + 0.01
    const z0 = sign * (sideGlassZ(BELT_Y) - 0.03)
    const z1 = sign * (sideGlassZ(SIDE_GLASS_TOP_Y) - 0.025)
    b.quad(
      [bx0, BELT_Y - 0.01, z0],
      [bx1, BELT_Y - 0.01, z0],
      [bx1, SIDE_GLASS_TOP_Y, z1],
      [bx0, SIDE_GLASS_TOP_Y, z1],
      { color: LINER },
      [0, 0, -sign],
    )
    // シートベルト（B ピラーの内側に沿って垂れている）
    const belt: Vec3[][] = []
    for (const y of [SIDE_GLASS_TOP_Y - 0.08, 0.95, 0.72, 0.52]) {
      const z = sign * (Math.min(sideGlassZ(Math.max(y, BELT_Y)), TRIM_Z - TRIM_HALF) - 0.045)
      belt.push([
        [-0.022, y, z],
        [0.022, y, z],
      ])
    }
    b.grid(belt, [0, 0, -sign], { color: ACCENT })
    add(b, boxGeometry([-0.03, SIDE_GLASS_TOP_Y - 0.1, sign * 0.715 - 0.02], [0.03, SIDE_GLASS_TOP_Y - 0.06, sign * 0.715 + 0.02]), { color: DARK })
    // 日よけ（天井に沿わせてたたんだ状態）
    add(b, boxGeometry([WINDSHIELD_TOP_X - 0.16, 1.378, sign > 0 ? 0.08 : -0.6], [WINDSHIELD_TOP_X - 0.02, 1.392, sign > 0 ? 0.6 : -0.08]), { color: LINER })
  }
}

/** 内装の動かない部分。 */
export function makeInteriorGeometry(): THREE.BufferGeometry {
  const b = new MeshBuilder()

  // 床。側壁と後ろの壁へ食い込ませる。Z を外板の内面に合わせないこと。
  //   外寄りはホイールハウスの手前で止める（中央だけ前後の壁まで伸ばす）
  add(b, boxGeometry([CABIN_REAR_X - 0.02, FLOOR_Y - 0.04, -0.49], [DASH_FRONT_X + 0.02, FLOOR_Y, 0.49]), { color: CARPET })
  for (const sign of [1, -1]) {
    const z0 = sign > 0 ? 0.48 : -0.79
    const z1 = sign > 0 ? 0.79 : -0.48
    add(b, boxGeometry([WELL_REAR_X + 0.006, FLOOR_Y - 0.035, z0], [WELL_FRONT_X - 0.006, FLOOR_Y - 0.004, z1]), { color: CARPET })
  }
  // ダッシュボードと、足もとの隔壁（前から背景が透けないように）。隔壁の下半分はホイールハウスの内側に収める
  add(b, boxGeometry([DASH_REAR_X, 0.7, -0.74], [DASH_FRONT_X, DASH_TOP_Y, 0.74]), { color: FABRIC })
  add(
    b,
    boxGeometry(
      [DASH_FRONT_X - OVERLAP + 0.005, FLOOR_Y - 0.08, -WELL_INNER_Z.front + 0.003],
      [DASH_FRONT_X + 0.055, ABOVE_WELL_Y + 0.01, WELL_INNER_Z.front - 0.003],
    ),
    { color: DARK },
  )
  add(b, boxGeometry([DASH_FRONT_X - OVERLAP, ABOVE_WELL_Y, -0.735], [DASH_FRONT_X + 0.06, DASH_TOP_Y - 0.05, 0.735]), { color: DARK })
  // 足もとの横の壁（キックパネル）。ホイールハウスとの間を内側から塞ぐ
  for (const sign of [1, -1]) {
    const zc = sign * (WELL_INNER_Z.front - 0.005)
    add(b, boxGeometry([WELL_FRONT_X - 0.05, FLOOR_Y - 0.03, zc - 0.004], [DASH_FRONT_X - OVERLAP + 0.02, ABOVE_WELL_Y + 0.012, zc + 0.004]), { color: DARK })
    const zr = sign * (WELL_INNER_Z.rear - 0.008)
    add(b, boxGeometry([CABIN_REAR_X + 0.01, FLOOR_Y - 0.03, zr - 0.006], [WELL_REAR_X + 0.035, ABOVE_WELL_Y + 0.013, zr + 0.006]), { color: DARK })
  }
  // センタースタック（ナビの下からセンターコンソールまで）
  add(b, boxGeometry([0.48, FLOOR_Y - 0.026, -0.115], [DASH_REAR_X + 0.03, 0.72, 0.115]), { color: DARK })
  // 吹き出し口（枠と羽根）
  for (const v of AC_VENTS) {
    add(b, boxGeometry([DASH_REAR_X - 0.006, v.y - v.h / 2, v.z - v.w / 2], [DASH_REAR_X + 0.01, v.y + v.h / 2, v.z + v.w / 2]), { color: ACCENT })
    for (let k = 1; k <= 3; k++) {
      const y = v.y - v.h / 2 + (v.h * k) / 4
      add(b, boxGeometry([DASH_REAR_X - 0.011, y - 0.003, v.z - v.w / 2 + 0.004], [DASH_REAR_X - 0.005, y + 0.003, v.z + v.w / 2 - 0.004]), { color: DARK })
    }
  }
  // ナビの下のスイッチ列
  const navBottom = NAV_SCREEN.center[1] - NAV_SCREEN.height / 2
  add(b, boxGeometry([DASH_REAR_X - 0.004, navBottom - 0.042, -0.095], [DASH_REAR_X + 0.01, navBottom - 0.008, 0.095]), { color: ACCENT })
  for (let k = 0; k < 5; k++) {
    const z = -0.072 + k * 0.036
    add(b, boxGeometry([DASH_REAR_X - 0.01, navBottom - 0.034, z - 0.012], [DASH_REAR_X - 0.003, navBottom - 0.016, z + 0.012]), { color: BUTTON })
  }
  // ステアリングコラムのカバー（ハブの奥。メーターへの視線より下）
  const axis: Vec3 = [Math.cos(COLUMN_TILT), -Math.sin(COLUMN_TILT), 0]
  add(
    b,
    orientedBox(
      [STEERING_CENTER[0] + axis[0] * 0.09, STEERING_CENTER[1] + axis[1] * 0.09 - 0.05, STEERING_CENTER[2]],
      axis,
      [0, 1, 0],
      [0.12, 0.07, 0.085],
    ),
    { color: DARK },
  )
  // センターコンソール・肘掛け・シフトレバー・カップホルダー
  add(b, boxGeometry([-0.35, FLOOR_Y - 0.03, -0.14], [0.5, 0.62, 0.14]), { color: FABRIC })
  add(b, boxGeometry([-0.36, 0.615, -0.13], [-0.02, 0.665, 0.13]), { color: FABRIC })
  add(b, boxGeometry([0.24, 0.617, -0.06], [0.42, 0.628, 0.06]), { color: ACCENT })
  add(b, rodGeometry([SHIFT_KNOB[0], 0.612, 0], [SHIFT_KNOB[0], SHIFT_KNOB[1] - 0.018, 0], 0.009), { color: ACCENT })
  add(b, boxGeometry([SHIFT_KNOB[0] - 0.028, SHIFT_KNOB[1] - 0.025, -0.022], [SHIFT_KNOB[0] + 0.028, SHIFT_KNOB[1] + 0.02, 0.022]), { color: ACCENT })
  for (const x of [0.04, 0.13]) {
    const cup = new THREE.CylinderGeometry(0.036, 0.036, 0.006, 10)
    cup.translate(x, 0.621, 0)
    add(b, cup, { color: ACCENT })
  }

  // ドアの内張り。**左後ろのドアは開け閉めするので、ドアの範囲で分けておく**
  for (const sign of [1, -1]) {
    doorTrim(b, sign, FRONT_DOOR_X[0], WELL_FRONT_X, false)
    doorTrim(b, sign, REAR_DOOR_X[0], REAR_DOOR_X[1], sign < 0)
    const zc = sign * TRIM_Z
    add(b, boxGeometry([REAR_DOOR_X[1] - 0.004, FLOOR_Y - 0.078, zc - TRIM_HALF + 0.002], [FRONT_DOOR_X[0] + 0.004, BELT_Y - 0.022, zc + TRIM_HALF - 0.002]), { color: DARK })
    // ホイールハウスの上だけに伸ばす部分（前はダッシュボードの横、後ろは後席の横）
    add(b, boxGeometry([WELL_FRONT_X - 0.004, ABOVE_WELL_Y - 0.06, zc - TRIM_HALF + 0.002], [DASH_FRONT_X + 0.02, BELT_Y - 0.022, zc + TRIM_HALF - 0.002]), { color: FABRIC })
    add(b, boxGeometry([CABIN_REAR_X - 0.02, ABOVE_WELL_Y - 0.06, zc - TRIM_HALF + 0.002], [REAR_DOOR_X[0] + 0.004, BELT_Y - 0.022, zc + TRIM_HALF - 0.002]), { color: FABRIC })
  }
  // 後ろの壁（トランクとの隔壁）。側壁の中で終わらせる。下半分はホイールハウスの内側に収める
  add(
    b,
    boxGeometry([CABIN_REAR_X - 0.057, FLOOR_Y - 0.06, -WELL_INNER_Z.rear], [CABIN_REAR_X + 0.017, ABOVE_WELL_Y + 0.01, WELL_INNER_Z.rear]),
    { color: DARK },
  )
  add(b, boxGeometry([CABIN_REAR_X - 0.06, ABOVE_WELL_Y, -0.74], [CABIN_REAR_X + 0.02, BELT_Y - 0.06, 0.74]), { color: DARK })
  frontSeat(b, DRIVER_SEAT_Z, FRONT_SEAT_X)
  frontSeat(b, PASSENGER_SEAT_Z, FRONT_SEAT_X)
  // 後席（ベンチ）とヘッドレスト。**リアガラスより前に収める**（ガラスは上へ行くほど前へ寝ているので、
  //   背もたれやヘッドレストを後ろへ置くとガラスの外＝トランクの上へ飛び出す）
  add(b, boxGeometry([-0.99, FLOOR_Y - 0.02, -0.7], [-0.5, FLOOR_Y + 0.18, 0.7]), { color: FABRIC })
  add(b, slabGeometry([-0.94, FLOOR_Y + 0.12], [-1.02, BELT_Y - 0.03], -0.694, 0.694, 0.14), { color: FABRIC })
  for (const z of [-0.46, 0, 0.46]) {
    add(b, boxGeometry([-0.985, BELT_Y - 0.07, z - 0.11], [-0.9, BELT_Y + 0.07, z + 0.11]), { color: FABRIC })
  }
  // リアパーセルシェルフ（リアガラスの付け根の下）。側壁の上面（BELT_Y）と高さを合わせない
  add(b, boxGeometry([CABIN_REAR_X - 0.04, BELT_Y - 0.05, -0.74], [-0.97, BELT_Y - 0.018, 0.74]), { color: DARK })
  // ルームミラー（ステーと本体と鏡面）
  const rm = ROOM_MIRROR.center
  add(b, rodGeometry([rm[0] - 0.015, 1.4, rm[2]], [rm[0], rm[1] + 0.015, rm[2]], 0.005), { color: ACCENT })
  const facing: Vec3 = [-Math.cos(ROOM_MIRROR.yaw), 0, Math.sin(ROOM_MIRROR.yaw)]
  const across: Vec3 = [Math.sin(ROOM_MIRROR.yaw), 0, Math.cos(ROOM_MIRROR.yaw)]
  add(b, orientedBox([rm[0], rm[1], rm[2]], across, [0, 1, 0], [ROOM_MIRROR.width, ROOM_MIRROR.height, 0.02]), { color: ACCENT })
  add(
    b,
    orientedBox(
      [rm[0] + facing[0] * 0.011, rm[1], rm[2] + facing[2] * 0.011],
      across,
      [0, 1, 0],
      [ROOM_MIRROR.width - 0.012, ROOM_MIRROR.height - 0.01, 0.003],
    ),
    { color: BUTTON },
  )

  addLinings(b)
  return b.build({ color: true, door: true })
}
