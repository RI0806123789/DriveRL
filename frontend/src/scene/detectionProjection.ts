/** バックエンドの擬似カメラ座標を、three のカメラの画面座標へ写し直す。 */

import { DRIVER_FOV_DEG, PSEUDO_CAMERA } from './cameraMath.ts'

/** バックエンドの擬似カメラ諸元（4 台とも同じ）。 */
export const BACKEND_CAMERA = PSEUDO_CAMERA

const DEG = Math.PI / 180

/** 擬似カメラの焦点距離 [px]（水平基準） */
export const BACKEND_FOCAL_PX =
  (BACKEND_CAMERA.width * 0.5) / Math.tan(BACKEND_CAMERA.fovDeg * DEG * 0.5)

export interface ViewportPoint {
  /** 0 = キャンバス左端 / 1 = 右端 */
  x: number
  /** 0 = 上端 / 1 = 下端 */
  y: number
}

/** 擬似カメラの正規化座標を、**垂直**画角 `fovDeg` のカメラの正規化画面座標へ写す。 */
export function projectToViewport(
  nx: number,
  ny: number,
  aspect: number,
  fovDeg: number = DRIVER_FOV_DEG,
): ViewportPoint {
  const tanX = (nx - 0.5) * (BACKEND_CAMERA.width / BACKEND_FOCAL_PX)
  const tanY = (ny - 0.5) * (BACKEND_CAMERA.height / BACKEND_FOCAL_PX)

  const tanHalfV = Math.tan(fovDeg * DEG * 0.5)
  const safeAspect = aspect > 1e-6 ? aspect : 1e-6
  return {
    x: 0.5 + (tanX / (tanHalfV * safeAspect)) * 0.5,
    y: 0.5 + (tanY / tanHalfV) * 0.5,
  }
}

export interface ViewportRect {
  left: number
  top: number
  width: number
  height: number
}

/** 検出のバウンディングボックス（擬似カメラの正規化座標）を、画面の正規化矩形へ写す。 */
export function projectBox(
  box: readonly [number, number, number, number],
  aspect: number,
  fovDeg: number = DRIVER_FOV_DEG,
): ViewportRect {
  const a = projectToViewport(box[0], box[1], aspect, fovDeg)
  const b = projectToViewport(box[2], box[3], aspect, fovDeg)
  const left = clamp01(Math.min(a.x, b.x))
  const top = clamp01(Math.min(a.y, b.y))
  const right = clamp01(Math.max(a.x, b.x))
  const bottom = clamp01(Math.max(a.y, b.y))
  return {
    left,
    top,
    width: Math.max(0, right - left),
    height: Math.max(0, bottom - top),
  }
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v
}
