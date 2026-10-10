/** 学習グラフの計算（panel/metricsChartMath.ts）が、全履歴を毎回なめていた以前の計算と同じ結果を出すかの単体テスト。 */

import assert from 'node:assert/strict'
import { describe, test } from 'node:test'

import { buildChart, LEAF_SIZE, MAX_BUCKETS, VIEW_W, type ChartGeometry, type ChartOptions } from '../panel/metricsChartMath.ts'

/** #120 より前の buildChart をそのまま写したもの（比べる相手）。 */
function referenceBuildChart(values: readonly number[], { height, showZero = false }: ChartOptions): ChartGeometry | null {
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
  const slots: ({ min: number; max: number; minFirst: boolean } | null)[] = new Array(buckets).fill(null)
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

function makeRng(seed: number): () => number {
  let s = seed >>> 0
  return () => {
    s = (s + 0x6d2b79f5) >>> 0
    let t = s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

type Gen = (i: number, rng: () => number) => number

const GENERATORS: Record<string, Gen> = {
  雑音: (_, rng) => rng() * 2 - 1,
  // 同じ値が何度も出る（最初の添字を残すかが効く）
  段: (_, rng) => Math.round(rng() * 4),
  // 極値がまれに何度も出る（1 本のバケツの中で、木の別々の節に同じ極値が入る）
  まれな極値: (_, rng) => (rng() < 0.005 ? 0 : rng() < 0.005 ? 2 : 1),
  NaN混じり: (_, rng) => (rng() < 0.3 ? Number.NaN : rng() * 10 - 5),
  無限大混じり: (_, rng) => (rng() < 0.05 ? Infinity : rng() < 0.05 ? -Infinity : rng()),
  一定: () => 0.25,
  ゼロ: () => 0,
  正の報酬: (i, rng) => 50 + Math.sin(i / 37) * 20 + rng(),
  単調減少: (i) => 1000 - i * 0.5,
}

function check(values: readonly number[], label: string): void {
  for (const opts of [{ height: 64 }, { height: 48, showZero: true }]) {
    assert.deepEqual(buildChart(values, opts), referenceBuildChart(values, opts), `${label} ${JSON.stringify(opts)}`)
  }
}

describe('学習グラフの集計', () => {
  test('どの長さ・どの値の並びでも以前の計算と一致する（新しい配列で 1 回だけ作る）', () => {
    // 1 本のバケツが葉を何枚もまたぐのは数万点から（それより短いと木の節をほとんど通らない）
    const lengths = [
      0, 1, 2, 3, LEAF_SIZE - 1, LEAF_SIZE, LEAF_SIZE + 1, 299, 300, 301, 599, 600, 1023, 1024, 1025, 4097, 9001,
      38_401, 131_071, 262_147,
    ]
    for (const [name, gen] of Object.entries(GENERATORS)) {
      const rng = makeRng(name.length * 7919)
      for (const n of lengths) {
        const values = Array.from({ length: n }, (_, i) => gen(i, rng))
        check(values, `${name} n=${n}`)
      }
    }
  })

  test('同じ配列へ追記し続けても毎回以前の計算と一致する（差分だけ足す経路）', () => {
    for (const [name, gen] of Object.entries(GENERATORS)) {
      const rng = makeRng(name.length * 104729)
      const values: number[] = []
      for (let i = 0; i < 2500; i++) {
        values.push(gen(i, rng))
        if (i < 400 || i % 53 === 0) check(values, `${name} n=${values.length}`)
      }
    }
  })

  test('1 回に何点も追記してから描いても一致する（パネルを畳んでいた間の追いつき）', () => {
    for (const [name, gen] of Object.entries(GENERATORS)) {
      const rng = makeRng(name.length * 20261010)
      const values: number[] = []
      for (let step = 0; step < 24; step++) {
        const burst = Math.floor(rng() * 12_000)
        for (let k = 0; k < burst; k++) values.push(gen(values.length, rng))
        check(values, `${name} n=${values.length}`)
      }
    }
  })

  test('配列が縮んだ・末尾が書き換わったら作り直す', () => {
    const rng = makeRng(42)
    const values = Array.from({ length: 5000 }, () => rng())
    check(values, '初回')
    values.length = 1200
    check(values, '縮めた後')
    values[values.length - 1] = 99
    check(values, '末尾を書き換えた後')
    for (let i = 0; i < 300; i++) values.push(rng() * -5)
    check(values, '追記した後')
  })

  test('有限値が 2 つ未満なら null（データを収集しています）', () => {
    assert.equal(buildChart([Number.NaN, 1, Number.NaN], { height: 48 }), null)
    assert.equal(buildChart([], { height: 48 }), null)
  })
})
