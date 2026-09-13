/**
 * 道路標示の寸法と、破線の割り付け計算（純粋関数）。
 *
 * React にも three にも依存させていないのは、`frontend/scripts/verify-road-markings.ts`
 * から Node で検証するため。**破線が 1 本も引かれない**ような割り付けの誤りは
 * 型チェックでもビルドでも捕まらず、画面を見ても「その道路には中央線が無い」と
 * しか見えないので、数値で不変条件を確かめる（code_review S-03）。
 *
 * 寸法は「道路標識、区画線及び道路標示に関する命令」に合わせてある。
 */

// --- 寸法（メートル） ---
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

/**
 * 全長 `total` [m] の線に破線を割り付ける。
 *
 * 両端が中途半端な線で終わらないよう、余った長さを前後へ均等に振る。
 *
 * ★ **`total` が線部（5m）より短くても必ず 1 本引く。**
 *   以前は余白が負になったぶんを範囲チェックで弾いており、
 *   5m 未満の対面通行路には中央線が 1 本も引かれていなかった
 *   （金沢で 472 本。code_review S-03）。短い道路は線を道路の長さに収める。
 */
export function dashSpans(total: number, on: number = DASH_ON, off: number = DASH_OFF): DashSpan[] {
  if (!(total > 0)) return []
  const period = on + off
  const count = Math.max(1, Math.floor(total / period))
  // total < on のときだけ負になる。0 で止めれば線は始点から引かれる
  const margin = Math.max(0, (total - count * period + off) / 2)

  const spans: DashSpan[] = []
  for (let k = 0; k < count; k++) {
    const start = margin + k * period
    if (start >= total) break
    // 道路の端をはみ出さないところで切る（はみ出すのは total < on のときだけ）
    const end = Math.min(start + on, total)
    if (end - start <= 1e-6) continue
    spans.push({ start, end })
  }
  return spans
}
