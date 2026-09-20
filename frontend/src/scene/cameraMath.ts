/** カメラ位置の計算（純粋関数）。 */

/** 追従カメラの相対位置（車両座標系: 後方 x、上方 y）[m] */
export const FOLLOW_BACK = 15
export const FOLLOW_UP = 7.5
/** 追従カメラが見る、車両前方の距離 [m] */
export const FOLLOW_LOOK_AHEAD = 6

/** 運転席の視点位置（車両座標系）。 */
export const DRIVER_FORWARD = 0.35
export const DRIVER_RIGHT = 0.36
export const DRIVER_EYE_HEIGHT = 1.22
/** 運転席から見る先の距離と、その高さの下げ幅 [m] */
export const DRIVER_LOOK_AHEAD = 30
export const DRIVER_LOOK_DROP = 1.1

/**
 * 運転席カメラの**垂直**視野角 [度]。
 * ★ **ここが唯一の出典。** `CameraRig` のカメラと `detectionProjection` の
 * 検出枠の両方がこれを読む。片方だけ変えると、認識結果の枠が実物からずれる。
 * 68 度だとアクセル・ブレーキ（目から俯角 48 度）が画角の下へ外れて映らないので、
 * 足元まで入る値にしてある（下端は視線の 2.1 度下 + 47.5 度 = 49.6 度）。
 */
export const DRIVER_FOV_DEG = 95

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

/** 追従できる最小のスロット番号を返す。1 台もいなければ -1。 */
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
