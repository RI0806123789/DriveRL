/** 学習指標グラフの計算。React から切り離した純粋モジュール。 */

/** SVG の viewBox の横幅 */
export const VIEW_W = 300

/** 横方向に何本まで打つか。viewBox の幅と同じにしてある（1 本 = 1px 相当）。 */
export const MAX_BUCKETS = VIEW_W

/** 1 本ぶんの縦の範囲。min/max を出た順に持つ（描く順を時系列に合わせるため） */
interface Bucket {
  min: number
  max: number
  /** min と max のどちらが先に来たか。true なら min が先 */
  minFirst: boolean
}

export interface ChartGeometry {
  /** 折れ線の d 属性 */
  line: string
  /** 塗り用に下端まで閉じた d 属性 */
  area: string
  /** 全データの最小・最大・平均（間引く前から出す） */
  min: number
  max: number
  mean: number
  /** 最後の有限値 */
  latest: number
  /** 有限値の個数 */
  count: number
  /** 0 の基準線の y。範囲に 0 が入らなければ null */
  zeroY: number | null
}

export interface ChartOptions {
  height: number
  /** 0 を必ず範囲に入れる（報酬など正負をまたぐ指標で有効） */
  showZero?: boolean
}

/** 値の列から SVG のパスと目盛りを作る。 */
/**
 * 添字 `i` のバケツ中心の x。マップ切替の目印を線と同じ位置へ置くのに使う。
 */
export function xForIndex(i: number, n: number): number {
  if (n <= 1) return 0
  const buckets = Math.min(MAX_BUCKETS, n)
  const b = Math.min(buckets - 1, Math.floor((i * buckets) / n))
  return b * (VIEW_W / (buckets - 1))
}

/**
 * 値の列から SVG のパスと目盛りを作る。
 * 有限値が 2 点に満たなければ null（グラフとして意味が無い）。
 */
export function buildChart(
  values: readonly number[],
  { height, showZero = false }: ChartOptions,
): ChartGeometry | null {
  const n = values.length

  let min = Infinity
  let max = -Infinity
  let sum = 0
  let count = 0
  let latest = NaN
  for (let i = 0; i < n; i++) {
    const v = values[i]
    if (!Number.isFinite(v)) continue
    if (v < min) min = v
    if (v > max) max = v
    sum += v
    count += 1
    latest = v
  }
  if (count < 2) return null

  const mean = sum / count

  let lo = min
  let hi = max
  if (showZero) {
    lo = Math.min(lo, 0)
    hi = Math.max(hi, 0)
  }
  if (hi - lo < 1e-9) {
    const pad = Math.max(1e-6, Math.abs(hi) * 0.1)
    lo -= pad
    hi += pad
  }
  const span = hi - lo
  const toY = (v: number) => height - ((v - lo) / span) * height

  const buckets = Math.min(MAX_BUCKETS, n)
  const slots: (Bucket | null)[] = new Array(buckets).fill(null)
  for (let i = 0; i < n; i++) {
    const v = values[i]
    if (!Number.isFinite(v)) continue
    const b = Math.min(buckets - 1, Math.floor((i * buckets) / n))
    const cur = slots[b]
    if (cur === null) {
      slots[b] = { min: v, max: v, minFirst: true }
    } else if (v < cur.min) {
      cur.min = v
      cur.minFirst = false
    } else if (v > cur.max) {
      cur.max = v
      cur.minFirst = true
    }
  }

  const stepX = buckets > 1 ? VIEW_W / (buckets - 1) : 0
  let line = ''
  for (let b = 0; b < buckets; b++) {
    const s = slots[b]
    if (s === null) continue
    const x = (b * stepX).toFixed(2)
    const first = s.minFirst ? s.min : s.max
    const second = s.minFirst ? s.max : s.min
    line += `${line === '' ? 'M' : 'L'}${x},${toY(first).toFixed(2)}`
    if (second !== first) line += `L${x},${toY(second).toFixed(2)}`
  }

  return {
    line,
    area: `${line}L${VIEW_W},${height}L0,${height}Z`,
    min,
    max,
    mean,
    latest,
    count,
    zeroY: lo <= 0 && hi >= 0 ? toY(0) : null,
  }
}
