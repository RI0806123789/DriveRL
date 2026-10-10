/** 学習指標グラフの計算。React から切り離した純粋モジュール。 */

/** SVG の viewBox の横幅 */
export const VIEW_W = 300

/** 横方向に何本まで打つか。viewBox の幅と同じにしてある（1 本 = 1px 相当）。 */
export const MAX_BUCKETS = VIEW_W

/** 区間木の葉 1 枚に入れる点の数。葉にまとめきれない端は元の配列を直接なめる */
export const LEAF_SIZE = 32

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

/** 区間の最小・最大と、それぞれが最初に現れた添字（有限値が無ければ添字は -1）。 */
interface Extremes {
  min: number
  minAt: number
  max: number
  maxAt: number
}

/** 区間木の 1 段。k 段目の j 番目は元の配列の [j·LEAF·2^k, (j+1)·LEAF·2^k) をまとめる */
interface Level {
  min: number[]
  minAt: number[]
  max: number[]
  maxAt: number[]
}

/** 追記だけされる系列 1 本ぶんの集計。 */
interface SeriesIndex {
  length: number
  last: number
  min: number
  max: number
  sum: number
  count: number
  latest: number
  leaf: Extremes
  levels: Level[]
}

const indexes = new WeakMap<readonly number[], SeriesIndex>()

function emptyExtremes(): Extremes {
  return { min: Infinity, minAt: -1, max: -Infinity, maxAt: -1 }
}

function emptyIndex(): SeriesIndex {
  return {
    length: 0,
    last: NaN,
    min: Infinity,
    max: -Infinity,
    sum: 0,
    count: 0,
    latest: NaN,
    leaf: emptyExtremes(),
    levels: [],
  }
}

/** 後ろに来る 1 点を集約へ足す。同じ値なら前の添字を残す */
function pushPoint(acc: Extremes, v: number, i: number): void {
  if (v < acc.min) {
    acc.min = v
    acc.minAt = i
  }
  if (v > acc.max) {
    acc.max = v
    acc.maxAt = i
  }
}

function pushLevel(levels: Level[], k: number, e: Extremes): void {
  let level = levels[k]
  if (level === undefined) {
    level = { min: [], minAt: [], max: [], maxAt: [] }
    levels[k] = level
  }
  level.min.push(e.min)
  level.minAt.push(e.minAt)
  level.max.push(e.max)
  level.maxAt.push(e.maxAt)
  const n = level.min.length
  if (n % 2 !== 0) return
  const a = n - 2
  const b = n - 1
  const takeMin = level.min[b] < level.min[a] ? b : a
  const takeMax = level.max[b] > level.max[a] ? b : a
  pushLevel(levels, k + 1, {
    min: level.min[takeMin],
    minAt: level.minAt[takeMin],
    max: level.max[takeMax],
    maxAt: level.maxAt[takeMax],
  })
}

function appendValue(index: SeriesIndex, v: number, i: number): void {
  if (Number.isFinite(v)) {
    if (v < index.min) index.min = v
    if (v > index.max) index.max = v
    index.sum += v
    index.count += 1
    index.latest = v
    pushPoint(index.leaf, v, i)
  }
  if ((i + 1) % LEAF_SIZE === 0) {
    pushLevel(index.levels, 0, index.leaf)
    index.leaf = emptyExtremes()
  }
}

/** 系列の集計を最新にする。追記された分だけ足し、縮んだ・書き換わった配列は作り直す */
function syncIndex(values: readonly number[]): SeriesIndex {
  const n = values.length
  let index = indexes.get(values)
  if (
    index === undefined ||
    index.length > n ||
    (index.length > 0 && !Object.is(values[index.length - 1], index.last))
  ) {
    index = emptyIndex()
    indexes.set(values, index)
  }
  for (let i = index.length; i < n; i++) appendValue(index, values[i], i)
  index.length = n
  index.last = n > 0 ? values[n - 1] : NaN
  return index
}

function scanRaw(acc: Extremes, values: readonly number[], from: number, to: number): void {
  for (let i = from; i < to; i++) {
    const v = values[i]
    if (Number.isFinite(v)) pushPoint(acc, v, i)
  }
}

/** 元の配列の [from, to) の最小・最大と、それぞれが最初に現れた添字。 */
function queryRange(index: SeriesIndex, values: readonly number[], from: number, to: number): Extremes {
  const acc = emptyExtremes()
  const leaves = index.levels[0]?.min.length ?? 0
  const lo = Math.ceil(from / LEAF_SIZE)
  const hi = Math.min(Math.floor(to / LEAF_SIZE), leaves)
  if (lo >= hi) {
    scanRaw(acc, values, from, to)
    return acc
  }
  scanRaw(acc, values, from, lo * LEAF_SIZE)

  // 左から順に足す側と、右端から前へ足す側に分けて木を登る（同じ値なら前の添字を残すため）
  const right = emptyExtremes()
  let l = lo
  let r = hi
  for (let k = 0; l < r; k++) {
    const level = index.levels[k]
    if (l & 1) {
      if (level.min[l] < acc.min) {
        acc.min = level.min[l]
        acc.minAt = level.minAt[l]
      }
      if (level.max[l] > acc.max) {
        acc.max = level.max[l]
        acc.maxAt = level.maxAt[l]
      }
      l += 1
    }
    if (r & 1) {
      r -= 1
      if (level.min[r] <= right.min) {
        right.min = level.min[r]
        right.minAt = level.minAt[r]
      }
      if (level.max[r] >= right.max) {
        right.max = level.max[r]
        right.maxAt = level.maxAt[r]
      }
    }
    l >>= 1
    r >>= 1
  }
  if (right.min < acc.min) {
    acc.min = right.min
    acc.minAt = right.minAt
  }
  if (right.max > acc.max) {
    acc.max = right.max
    acc.maxAt = right.maxAt
  }

  scanRaw(acc, values, hi * LEAF_SIZE, to)
  return acc
}

/** 添字 `i` のバケツ中心の x。 */
export function xForIndex(i: number, n: number): number {
  if (n <= 1) return 0
  const buckets = Math.min(MAX_BUCKETS, n)
  const b = Math.min(buckets - 1, Math.floor((i * buckets) / n))
  return b * (VIEW_W / (buckets - 1))
}

/** 値の列から SVG のパスと目盛りを作る。 */
export function buildChart(
  values: readonly number[],
  { height, showZero = false }: ChartOptions,
): ChartGeometry | null {
  const n = values.length
  const index = syncIndex(values)
  const { min, max, count, latest } = index
  if (count < 2) return null

  const mean = index.sum / count

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

  // バケツ b は floor(i·buckets / n) = b となる添字 i、つまり [ceil(b·n / buckets), ceil((b+1)·n / buckets))
  const buckets = Math.min(MAX_BUCKETS, n)
  const stepX = buckets > 1 ? VIEW_W / (buckets - 1) : 0
  let line = ''
  let start = 0
  for (let b = 0; b < buckets; b++) {
    const end = Math.floor(((b + 1) * n + buckets - 1) / buckets)
    const s = queryRange(index, values, start, end)
    start = end
    if (s.minAt < 0) continue
    const minFirst = s.minAt <= s.maxAt
    const x = (b * stepX).toFixed(2)
    const first = minFirst ? s.min : s.max
    const second = minFirst ? s.max : s.min
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
