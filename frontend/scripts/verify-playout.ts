/** frame の simTime を時計にした再生を、揺れた到着列で検証する（ブラウザ不要）。 */

import {
  PLAYOUT_DELAY_MS,
  PLAYOUT_JITTER_MS,
  PLAYOUT_MAX_ADJUST,
  createPlayoutClock,
  notePlayoutArrival,
  playoutTime,
} from '../src/store/playout.ts'
import type { PlayoutEvent } from '../src/store/playout.ts'

let failures = 0

function check(label: string, ok: boolean, detail = ''): void {
  console.log(`  [${ok ? 'OK  ' : 'NG  '}] ${label}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures++
}

/** 最高速 [m/s]。sim 時刻のずれを、この速さで走る車の前後のずれ [m] に直す */
const TOP_SPEED = 13.9
const RENDER_MS = 1000 / 60
const STEP_SIM = 0.05

interface Arrival {
  wall: number
  sim: number
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

interface StreamOptions {
  seconds: number
  /** 1 ステップの計算時間 [ms] */
  computeMs: (rnd: () => number) => number
  simSpeed?: number
  /** 配信ループの起床周期 [ms]。Windows の既定タイマーでは asyncio.sleep(1/60) が約 31ms になる */
  pollMs?: number
  /** ブラウザ側で処理が遅れる幅 [ms] */
  browserJitterMs?: number
  stallAtMs?: number
  stallMs?: number
  seed?: number
}

/** engine._run_loop が作った frame を broadcast_loop が最新 1 件だけ拾って送る、を再現する */
function stream(o: StreamOptions): Arrival[] {
  const rnd = mulberry32(o.seed ?? 1)
  const stepMs = 50 / (o.simSpeed ?? 1)
  const produced: Arrival[] = []
  let deadline = 0
  let t = 0
  let sim = 0
  while (t < o.seconds * 1000) {
    let done = t + o.computeMs(rnd)
    if (o.stallAtMs !== undefined && t <= o.stallAtMs && o.stallAtMs < t + stepMs) {
      done += o.stallMs ?? 0
    }
    sim += STEP_SIM
    produced.push({ wall: done, sim })
    deadline += stepMs
    if (deadline < done - 1000) deadline = done
    t = Math.max(deadline, done)
  }
  const poll = o.pollMs ?? 31.25
  const out: Arrival[] = []
  let lastSim = -1
  let k = -1
  for (let now = rnd() * poll; now < o.seconds * 1000; now += poll * (0.97 + rnd() * 0.06)) {
    while (k + 1 < produced.length && produced[k + 1].wall <= now) k++
    if (k < 0 || produced[k].sim === lastSim) continue
    lastSim = produced[k].sim
    out.push({ wall: now + rnd() * (o.browserJitterMs ?? 8), sim: lastSim })
  }
  out.sort((a, b) => a.wall - b.wall)
  return out
}

type Player = (wallMs: number) => number | null

/** いまの実装。frameBuffer と同じく、張り直した直後は届いた frame で止めて待つ */
function playoutPlayer(arrivals: readonly Arrival[], events: PlayoutEvent[]): Player {
  const clock = createPlayoutClock()
  let i = 0
  let floor = -Infinity
  return (wallMs) => {
    while (i < arrivals.length && arrivals[i].wall <= wallMs) {
      const e = notePlayoutArrival(clock, arrivals[i].sim, arrivals[i].wall)
      events.push(e)
      if (e === 'reset') floor = arrivals[i].sim
      i++
    }
    if (i === 0) return null
    return Math.max(playoutTime(clock, wallMs), floor)
  }
}

/** 旧実装。直前の受信間隔で次の区間を再生する（受信するたびに時刻を張り直す） */
function legacyPlayer(arrivals: readonly Arrival[]): Player {
  let i = 0
  let interval = 50
  return (wallMs) => {
    while (i < arrivals.length && arrivals[i].wall <= wallMs) {
      if (i > 0) {
        const dt = arrivals[i].wall - arrivals[i - 1].wall
        if (dt > 1) interval = Math.min(500, Math.max(16, dt))
      }
      i++
    }
    if (i === 0) return null
    const curr = arrivals[i - 1]
    if (i === 1) return curr.sim
    const prev = arrivals[i - 2]
    const t = (wallMs - curr.wall) / interval
    const a = t <= 0 ? 0 : t >= 1 ? 1 : t
    return prev.sim + (curr.sim - prev.sim) * a
  }
}

interface Report {
  /** 等速の軌跡（前後 0.5 秒に当てた直線）からの前後のずれ [m] */
  wobble95: number
  wobbleMax: number
  /** 1 描画フレームで進んだ量 ÷ 理想の量 */
  advanceMax: number
  /** 進まなかった描画フレームの割合 */
  stalled: number
  /** 届いている最新より何 ms 遅れて表示しているか（実時間換算の中央値） */
  lagMs: number
  /** 届いた最新を追い越して表示したフレーム数 */
  ahead: number
  /** 表示が戻ったフレーム数 */
  backward: number
}

function quantile(xs: number[], q: number): number {
  const s = [...xs].sort((a, b) => a - b)
  return s[Math.min(s.length - 1, Math.floor(q * (s.length - 1)))]
}

function evaluate(
  arrivals: readonly Arrival[],
  player: Player,
  from: number,
  to: number,
  nominalRate?: number,
): Report {
  const first = arrivals[0]
  const last = arrivals[arrivals.length - 1]
  const rate = nominalRate ?? (last.sim - first.sim) / ((last.wall - first.wall) / 1000)
  const ts: number[] = []
  const shown: number[] = []
  const latest: number[] = []
  let j = -1
  for (let t = first.wall; t < last.wall; t += RENDER_MS) {
    const s = player(t)
    while (j + 1 < arrivals.length && arrivals[j + 1].wall <= t) j++
    if (s === null || t < from || t > to) continue
    ts.push(t)
    shown.push(s)
    latest.push(arrivals[j].sim)
  }
  const ideal = rate * (RENDER_MS / 1000)
  const advance: number[] = []
  let backward = 0
  for (let k = 1; k < shown.length; k++) {
    const d = shown[k] - shown[k - 1]
    if (d < -1e-9) backward++
    advance.push(d / ideal)
  }
  const wobble: number[] = []
  const W = 30
  for (let k = W; k < shown.length - W; k++) {
    let mx = 0
    let my = 0
    for (let m = k - W; m <= k + W; m++) {
      mx += ts[m]
      my += shown[m]
    }
    mx /= 2 * W + 1
    my /= 2 * W + 1
    let sxy = 0
    let sxx = 0
    for (let m = k - W; m <= k + W; m++) {
      sxy += (ts[m] - mx) * (shown[m] - my)
      sxx += (ts[m] - mx) ** 2
    }
    const fit = my + (sxy / sxx) * (ts[k] - mx)
    wobble.push(Math.abs(shown[k] - fit) * TOP_SPEED)
  }
  const lag = latest.map((l, k) => ((l - shown[k]) / rate) * 1000)
  return {
    wobble95: quantile(wobble, 0.95),
    wobbleMax: Math.max(...wobble),
    advanceMax: Math.max(...advance),
    stalled: advance.filter((a) => a < 0.05).length / advance.length,
    lagMs: quantile(lag, 0.5),
    ahead: latest.filter((l, k) => shown[k] > l + 1e-9).length,
    backward,
  }
}

function describe(r: Report): string {
  return (
    `ずれ 95% ${(r.wobble95 * 100).toFixed(1)}cm / 最大 ${(r.wobbleMax * 100).toFixed(1)}cm` +
    ` / 跳び ${r.advanceMax.toFixed(2)} 倍 / 停止 ${(r.stalled * 100).toFixed(1)}%` +
    ` / 遅れ ${r.lagMs.toFixed(0)}ms`
  )
}

function notable(events: readonly PlayoutEvent[]): string {
  return events.filter((e) => e !== 'ok').join(', ') || 'なし'
}

const WARMUP_MS = 2000
const NOMINAL = (rnd: () => number) => 12 + rnd() * 3
const CNN = (rnd: () => number) => 52 + rnd() * 28

console.log('='.repeat(70))
console.log('1. 受信が揺れても表示は等速に近いまま進むか（13.9m/s の車の前後のずれ）')
console.log('='.repeat(70))

interface Case {
  name: string
  arrivals: Arrival[]
  wobble95: number
  wobbleMax: number
}

const cases: Case[] = [
  {
    name: '1 倍・配信 31ms 周期（Windows の既定タイマー）',
    arrivals: stream({ seconds: 20, computeMs: NOMINAL }),
    wobble95: 0.03,
    wobbleMax: 0.05,
  },
  {
    name: '1 倍・配信 16.7ms 周期',
    arrivals: stream({ seconds: 20, computeMs: NOMINAL, pollMs: 1000 / 60 }),
    wobble95: 0.03,
    wobbleMax: 0.05,
  },
  {
    name: 'CNN で 1 ステップ 52〜80ms（実時間の約 0.8 倍）',
    arrivals: stream({ seconds: 20, computeMs: CNN }),
    wobble95: 0.08,
    wobbleMax: 0.12,
  },
  {
    name: '4 倍速',
    arrivals: stream({ seconds: 20, computeMs: (rnd) => 6 + rnd() * 3, simSpeed: 4 }),
    wobble95: 0.1,
    wobbleMax: 0.15,
  },
  {
    name: '0.25 倍速（受信が 200ms おき）',
    arrivals: stream({ seconds: 30, computeMs: NOMINAL, simSpeed: 0.25 }),
    wobble95: 0.03,
    wobbleMax: 0.05,
  },
]

for (const c of cases) {
  const events: PlayoutEvent[] = []
  const end = c.arrivals[c.arrivals.length - 1].wall
  const meanInterval = (end - c.arrivals[0].wall) / (c.arrivals.length - 1)
  const delay = Math.max(PLAYOUT_DELAY_MS, meanInterval + PLAYOUT_JITTER_MS)
  const now = evaluate(c.arrivals, playoutPlayer(c.arrivals, events), WARMUP_MS, end)
  const old = evaluate(c.arrivals, legacyPlayer(c.arrivals), WARMUP_MS, end)
  console.log(`  ${c.name}（${c.arrivals.length} 通）`)
  console.log(`      いま: ${describe(now)}`)
  console.log(`      旧  : ${describe(old)}`)
  check(
    `前後のずれ（95%）が ${c.wobble95 * 100}cm 未満`,
    now.wobble95 < c.wobble95,
    `${(now.wobble95 * 100).toFixed(1)}cm（旧 ${(old.wobble95 * 100).toFixed(1)}cm）`,
  )
  check(
    `前後のずれ（最大）が ${c.wobbleMax * 100}cm 未満`,
    now.wobbleMax < c.wobbleMax,
    `${(now.wobbleMax * 100).toFixed(1)}cm（旧 ${(old.wobbleMax * 100).toFixed(1)}cm）`,
  )
  check(
    `1 フレームで補正の上限（${(1 + PLAYOUT_MAX_ADJUST).toFixed(2)} 倍）を超えて進まない`,
    now.advanceMax <= (1 + PLAYOUT_MAX_ADJUST) * 1.05,
    `${now.advanceMax.toFixed(2)} 倍（旧 ${old.advanceMax.toFixed(2)} 倍）`,
  )
  check('止まるフレームが無い', now.stalled === 0, `${(now.stalled * 100).toFixed(1)}%`)
  check('届いた最新より先を出さない（外挿しない）', now.ahead === 0, `${now.ahead} フレーム`)
  check('表示が戻らない', now.backward === 0, `${now.backward} フレーム`)
  check(
    `遅れは再生の遅れ（${delay.toFixed(0)}ms）程度に収まる`,
    now.lagMs < delay * 1.2,
    `中央値 ${now.lagMs.toFixed(0)}ms（旧 ${old.lagMs.toFixed(0)}ms）`,
  )
  check(
    '張り直しは最初の 1 回だけで、追いつくための飛ばしも無い',
    events.filter((e) => e === 'reset').length === 1 && !events.includes('snap'),
    notable(events),
  )
}

{
  const c = cases[0]
  const old = evaluate(
    c.arrivals,
    legacyPlayer(c.arrivals),
    WARMUP_MS,
    c.arrivals[c.arrivals.length - 1].wall,
  )
  check(
    '旧実装はこの到着列で揺れる（この検証が今回の揺れを捕まえられる）',
    old.wobble95 > 0.1 && old.stalled > 0.05,
    `旧 ${describe(old)}`,
  )
}

console.log('')
console.log('='.repeat(70))
console.log('2. サーバーが一時的に止まったとき')
console.log('='.repeat(70))

for (const stallMs of [300, 1500]) {
  const arrivals = stream({ seconds: 16, computeMs: NOMINAL, stallAtMs: 8000, stallMs })
  const end = arrivals[arrivals.length - 1].wall
  const events: PlayoutEvent[] = []
  // 止まっていた時間ぶん平均が下がるので、公称の速さ（1 倍）を物差しにする
  const r = evaluate(arrivals, playoutPlayer(arrivals, events), WARMUP_MS, end, 1)
  check(
    `${stallMs}ms 止まった後: 待っていた位置から進み、前へ跳ばない`,
    r.advanceMax <= (1 + PLAYOUT_MAX_ADJUST) * 1.05,
    `跳び ${r.advanceMax.toFixed(2)} 倍`,
  )
  check(`${stallMs}ms 止まった後: 表示が戻らない`, r.backward === 0, `${r.backward} フレーム`)
  check(
    `${stallMs}ms 止まった後: 張り直しも追いつくための飛ばしも無い`,
    !events.includes('snap') && events.filter((e) => e === 'reset').length === 1,
    notable(events),
  )
  const tail = evaluate(arrivals, playoutPlayer(arrivals, []), 8000 + stallMs + 3000, end, 1)
  check(
    `${stallMs}ms 止まった 3 秒後: 元の滑らかさに戻っている`,
    tail.wobble95 < 0.03 && tail.stalled === 0,
    describe(tail),
  )
}

console.log('')
console.log('='.repeat(70))
console.log('3. 流れが切れたとき（マップの読み直し・描画の一時停止）')
console.log('='.repeat(70))

{
  const clock = createPlayoutClock()
  let wall = 0
  for (let k = 1; k <= 40; k++) {
    wall += 50
    notePlayoutArrival(clock, 100 + k * STEP_SIM, wall)
  }
  const before = playoutTime(clock, wall)
  wall += 50
  const e = notePlayoutArrival(clock, STEP_SIM, wall)
  check('simTime が戻ったら張り直す（マップの読み直し）', e === 'reset', e)
  check(
    '張り直した直後は、新しい frame より先を出さない',
    playoutTime(clock, wall) <= STEP_SIM,
    `${playoutTime(clock, wall).toFixed(3)}s（直前 ${before.toFixed(3)}s）`,
  )

  wall += 50
  const e2 = notePlayoutArrival(clock, STEP_SIM + 5, wall)
  check('simTime が大きく飛んだら張り直す（描画の一時停止から戻った）', e2 === 'reset', e2)

  wall += 50
  const e3 = notePlayoutArrival(clock, STEP_SIM + 5, wall)
  check('同じ simTime の frame は読み捨てる', e3 === 'stale', e3)
}

console.log('')
console.log('='.repeat(70))
if (failures > 0) {
  console.log(`結果: ${failures} 件の不合格`)
  process.exit(1)
}
console.log('結果: すべて合格')
