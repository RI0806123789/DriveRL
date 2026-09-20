/** 車載カメラの転送先。**15fps で書き換わるので zustand には入れない**（frameBuffer と同じ）。 */

/** 描く解像度の下限・上限 [px]。上限は GPU からの読み戻し量を抑えるため */
export const FEED_MIN_PX = 64
export const FEED_MAX_W = 1280
export const FEED_MAX_H = 720

export interface TaxiCameraLink {
  /** スマホ画面側の 2D キャンバス。出していない間は null */
  canvas: HTMLCanvasElement | null
  /** 映す車両スロット。-1 なら映さない */
  vehicleId: number
  /** 直近で 1 枚でも転送できたか（「接続中」の表示に使う） */
  live: boolean
  /** 描いてほしい解像度。**表示側が実寸×dpr で入れる**（荒さの出どころはここ） */
  width: number
  height: number
}

export const taxiCamera: TaxiCameraLink = {
  canvas: null,
  vehicleId: -1,
  live: false,
  width: 640,
  height: 360,
}

/** 表示の実寸から、描く解像度を決める。 */
export function feedSizeFor(cssWidth: number, cssHeight: number, dpr: number): [number, number] {
  const scale = Math.min(2, Math.max(1, dpr))
  const w = Math.round(cssWidth * scale)
  const h = Math.round(cssHeight * scale)
  return [
    Math.min(FEED_MAX_W, Math.max(FEED_MIN_PX, w)),
    Math.min(FEED_MAX_H, Math.max(FEED_MIN_PX, h)),
  ]
}

export function resetTaxiCamera(): void {
  taxiCamera.canvas = null
  taxiCamera.vehicleId = -1
  taxiCamera.live = false
}
