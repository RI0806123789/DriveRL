/**
 * カメラ位置の計算（純粋関数）。
 *
 * React に依存させず、`frontend/scripts/verify-camera-route.ts` から Node で
 * 直接検証できるようにしてある。3D の向きは目で見ないと分からないうえ、
 * 間違えても型チェックもビルドも通ってしまうため。
 *
 * 座標変換の規約は docs/protocol.md 1.3（three.z = -enu.y）。
 * 戻り値はすべて three 空間。
 */

/** 追従カメラの相対位置（車両座標系: 後方 x、上方 y）[m] */
export const FOLLOW_BACK = 15
export const FOLLOW_UP = 7.5
/** 追従カメラが見る、車両前方の距離 [m] */
export const FOLLOW_LOOK_AHEAD = 6

/**
 * 運転席の視点位置（車両座標系）。
 *
 * **日本車は右ハンドル**なので、進行方向に対して右側にずらす。
 * 左側通行の日本では、これが道路中央側＝対向車を見やすい位置になる。
 */
export const DRIVER_FORWARD = 0.35
export const DRIVER_RIGHT = 0.36
export const DRIVER_EYE_HEIGHT = 1.22
/** 運転席から見る先の距離と、その高さの下げ幅 [m] */
export const DRIVER_LOOK_AHEAD = 30
export const DRIVER_LOOK_DROP = 1.1

/**
 * 運転席カメラの視野角 [度]。
 * **three の PerspectiveCamera.fov は垂直画角**（横は aspect から決まる）。
 */
export const DRIVER_FOV_DEG = 68

export interface Vec3 {
  x: number
  y: number
  z: number
}

/** 追従できるかどうかを判断するのに必要な最小限の情報 */
export interface FollowCandidate {
  id: number
  active: boolean
}

/**
 * 追従できる最小のスロット番号を返す。1 台もいなければ -1。
 *
 * 車両数を減らすとバックエンドは**上位スロットから**非アクティブにするので
 * （world.set_active_count）、追従中の車両が黙って消えることがある。
 * そのときの乗り換え先をここで決める。番号の小さい車ほど長く走っている
 * （スポーン順）ので、最小のアクティブスロットを選ぶ。
 */
export function firstActiveSlot(
  vehicles: ReadonlyArray<FollowCandidate> | null | undefined,
): number {
  if (!vehicles) return -1
  let best = -1
  for (const v of vehicles) {
    if (!v.active) continue
    if (best < 0 || v.id < best) best = v.id
  }
  return best
}

/** ENU の点を three 空間へ写す */
function toThree(enuX: number, enuY: number, height: number): Vec3 {
  return { x: enuX, y: height, z: -enuY }
}

/** 運転席のアイポイント（three 空間） */
export function driverEye(x: number, y: number, heading: number): Vec3 {
  const cos = Math.cos(heading)
  const sin = Math.sin(heading)
  // 進行方向の右手。ENU では heading - 90 度 = (sin, -cos)
  const rightX = sin
  const rightY = -cos
  return toThree(
    x + cos * DRIVER_FORWARD + rightX * DRIVER_RIGHT,
    y + sin * DRIVER_FORWARD + rightY * DRIVER_RIGHT,
    DRIVER_EYE_HEIGHT,
  )
}

/** 運転席から見る先（three 空間） */
export function driverLookAt(x: number, y: number, heading: number): Vec3 {
  const cos = Math.cos(heading)
  const sin = Math.sin(heading)
  return toThree(
    x + cos * DRIVER_LOOK_AHEAD,
    y + sin * DRIVER_LOOK_AHEAD,
    DRIVER_EYE_HEIGHT - DRIVER_LOOK_DROP,
  )
}

/** 追従カメラの位置（three 空間） */
export function followEye(x: number, y: number, heading: number): Vec3 {
  const cos = Math.cos(heading)
  const sin = Math.sin(heading)
  return toThree(x - cos * FOLLOW_BACK, y - sin * FOLLOW_BACK, FOLLOW_UP)
}

/** 追従カメラが見る先（three 空間） */
export function followLookAt(x: number, y: number, heading: number): Vec3 {
  const cos = Math.cos(heading)
  const sin = Math.sin(heading)
  return toThree(x + cos * FOLLOW_LOOK_AHEAD, y + sin * FOLLOW_LOOK_AHEAD, 1.2)
}
