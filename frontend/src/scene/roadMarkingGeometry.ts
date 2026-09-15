/** 道路標示の寸法と、破線の割り付け計算（純粋関数）。 */

/** 区画線の幅 */
export const LINE_WIDTH = 0.15
/** 破線の線部 */
export const DASH_ON = 5.0
/** 破線の空白部 */
export const DASH_OFF = 5.0
/** 車道外側線を路端からどれだけ内側に引くか */
export const EDGE_LINE_INSET = 0.35
/** 停止線の太さ */
export const STOP_LINE_WIDTH = 0.45
/** 横断歩道の横断方向の長さ */
export const CROSSWALK_LENGTH = 4.0
/** ゼブラの帯幅 */
export const CROSSWALK_STRIPE = 0.45
/** ゼブラの間隔 */
export const CROSSWALK_GAP = 0.45
/** 停止線から横断歩道までの距離 */
export const CROSSWALK_OFFSET = 1.0

/** 破線 1 本ぶんの、ポリライン始点からの弧長 [m] */
export interface DashSpan {
  start: number
  end: number
}

/** 全長 `total` [m] の線に破線を割り付ける。 */
export function dashSpans(total: number, on: number = DASH_ON, off: number = DASH_OFF): DashSpan[] {
  if (!(total > 0)) return []
  const period = on + off
  const count = Math.max(1, Math.floor(total / period))
  const margin = Math.max(0, (total - count * period + off) / 2)

  const spans: DashSpan[] = []
  for (let k = 0; k < count; k++) {
    const start = margin + k * period
    if (start >= total) break
    const end = Math.min(start + on, total)
    if (end - start <= 1e-6) continue
    spans.push({ start, end })
  }
  return spans
}
