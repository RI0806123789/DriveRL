/**
 * バックエンドの擬似カメラ座標を、three のカメラの画面座標へ写し直す。
 *
 * ★ **両者は画角の定義が違うので、正規化座標をそのまま重ねてはいけない。**
 *
 *   - バックエンド（`backend/app/percep/types.py` の `CameraSpec`）の `fov_deg` は
 *     **水平**画角。`focal_px = (width / 2) / tan(fov / 2)` で使っている。
 *     画像は 192x144（4:3）固定で、CNN の入力なので可変にできない。
 *   - three の `PerspectiveCamera.fov` は**垂直**画角で、横は aspect から決まる
 *     （`cameraMath.ts` の `DRIVER_FOV_DEG`）。キャンバスの縦横比は画面しだい。
 *
 *   同じ 68 という数字でも、バックエンドの垂直画角は約 53.7 度にしかならない
 *   （`2 * atan(72 / 142.3)`）。そのまま重ねると**縦に 1.27 倍ずれる**うえ、
 *   横はキャンバスの縦横比しだいでさらに動く。ボックスが対象物から外れると
 *   「何を認識しているか見せる」という目的そのものが果たせない。
 *
 *   そこで一度「光軸からの方向（tan）」へ戻し、three 側の投影で描き直す。
 *   方向は画角の定義に依存しない共通の表現なので、これで両者が噛み合う。
 */

// ★ 拡張子を明示しているのは、`scripts/verify-detections.ts` が **Node で直接**
//   このモジュールを読むため（Vite を通らないので拡張子を省略できない）。
//   tsconfig の `allowImportingTsExtensions` を有効にしてあるのはこのため。
import { DRIVER_FOV_DEG } from './cameraMath.ts'

/**
 * バックエンドの擬似カメラ諸元。
 * **`backend/app/percep/types.py` の `CameraSpec` と一致させること。**
 * 片方だけ変えるとボックスが静かにずれる（型でもビルドでも検出できない）。
 */
export const BACKEND_CAMERA = {
  width: 192,
  height: 144,
  /** ★ 水平画角。three の垂直画角とは別物 */
  fovDeg: 68,
} as const

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

/**
 * 擬似カメラの正規化座標を、キャンバスの正規化座標へ写す。
 *
 * @param nx 擬似カメラ画像の x（0 = 左端 / 1 = 右端）
 * @param ny 擬似カメラ画像の y（0 = 上端 / 1 = 下端）
 * @param aspect キャンバスの 幅 / 高さ
 * @returns キャンバス上の位置。視野外なら 0〜1 の外側を返す（呼び出し側で丸める）
 */
export function projectToViewport(nx: number, ny: number, aspect: number): ViewportPoint {
  // 1. 画素位置 -> 光軸からの方向。焦点距離は水平・垂直で共通（正方画素）
  const tanX = (nx - 0.5) * (BACKEND_CAMERA.width / BACKEND_FOCAL_PX)
  const tanY = (ny - 0.5) * (BACKEND_CAMERA.height / BACKEND_FOCAL_PX)

  // 2. three の投影に載せ直す。fov は垂直なので、横だけ aspect で割る
  const tanHalfV = Math.tan(DRIVER_FOV_DEG * DEG * 0.5)
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

/**
 * 検出のバウンディングボックス（擬似カメラの正規化座標）を
 * キャンバスの矩形へ写す。0〜1 に丸めるので、はみ出した分は切り詰められる。
 */
export function projectBox(
  box: readonly [number, number, number, number],
  aspect: number,
): ViewportRect {
  const a = projectToViewport(box[0], box[1], aspect)
  const b = projectToViewport(box[2], box[3], aspect)
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
