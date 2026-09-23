/** frame の simTime を時計にした再生（純粋関数。`npm run verify:playout` から読む）。 */

/** 再生を最新の frame からどれだけ遅らせるか（下限）[ms・実時間]。受信の揺れはここで吸収する */
export const PLAYOUT_DELAY_MS = 120
/** 平均の受信間隔に上乗せする揺れの見込み [ms]。遅れは「受信間隔 + これ」と下限の大きいほう */
export const PLAYOUT_JITTER_MS = 60
/** 目標とのずれを詰めきるまでの時間 [s] */
export const PLAYOUT_HORIZON_S = 1.0
/** 再生の速さを補正してよい幅（±割合） */
export const PLAYOUT_MAX_ADJUST = 0.12
/** sim の進む速さと受信間隔を測る窓（直近の受信間隔の数） */
export const PLAYOUT_RATE_INTERVALS = 30
/** 受信の間隔がこれ（と平均の 3 倍）より長ければ、サーバーが止まっていたとみなす [ms] */
export const PLAYOUT_RATE_GAP_MS = 250
/** 止まった後、サーバーが遅れを取り戻している間も速さを測らない [ms] */
export const PLAYOUT_RATE_SETTLE_MS = 300
/** 表示が目標からこれ以上遅れたら、滑らかさを捨てて追いつく [s・実時間] */
export const PLAYOUT_SNAP_S = 0.6
/** simTime がこれ以上進んだら（または戻ったら）別の流れとみなして張り直す [s] */
export const PLAYOUT_RESET_GAP_S = 1.0

const RATE_MIN_SPAN_MS = 100
const RATE_MIN = 0.05
const RATE_MAX = 20

/** 受信 1 通の扱い。`reset` のときは表示中の区間を捨てること */
export type PlayoutEvent = 'reset' | 'snap' | 'ok' | 'stale'

export interface PlayoutClock {
  ready: boolean
  /** 時計を張り直した実時間 [ms] */
  anchorWall: number
  /** 張り直した時点で表示していた sim 時刻 [s] */
  anchorSim: number
  /** 表示の進む速さ（実時間 1 秒あたりの sim 秒） */
  slope: number
  /** 受信から測った sim の進む速さ（実時間 1 秒あたりの sim 秒） */
  rate: number
  /** 受信から測った平均の受信間隔 [ms] */
  intervalMs: number
  /** 受信した frame のうち最も新しい simTime [s] と、それを受信した実時間 [ms] */
  latestSim: number
  latestWall: number
  /** この実時間 [ms] までは速さを測らない（止まった直後の追いつきを除く） */
  settleUntil: number
  /** 速さを測る窓。受信の間隔ごとの実時間 [ms] と sim の進み [s] を古い順に持つ */
  intervalWall: number[]
  intervalSim: number[]
}

export function createPlayoutClock(): PlayoutClock {
  return {
    ready: false,
    anchorWall: 0,
    anchorSim: 0,
    slope: 1,
    rate: 1,
    intervalMs: 50,
    latestSim: 0,
    latestWall: 0,
    settleUntil: 0,
    intervalWall: [],
    intervalSim: [],
  }
}

/** 次に届く frame から時計を張り直させる。 */
export function resetPlayoutClock(clock: PlayoutClock): void {
  clock.ready = false
  clearIntervals(clock)
}

/** いまの再生の遅れ [ms・実時間]。 */
export function playoutDelayMs(clock: PlayoutClock): number {
  return Math.max(PLAYOUT_DELAY_MS, clock.intervalMs + PLAYOUT_JITTER_MS)
}

/** 実時間 `wallMs` に表示する sim 時刻 [s]。届いた frame より先へは進めない */
export function playoutTime(clock: PlayoutClock, wallMs: number): number {
  if (!clock.ready) return clock.latestSim
  const t = clock.anchorSim + ((wallMs - clock.anchorWall) / 1000) * clock.slope
  return Math.min(t, clock.latestSim)
}

/** frame が 1 通届いたときに呼ぶ。 */
export function notePlayoutArrival(
  clock: PlayoutClock,
  simTime: number,
  wallMs: number,
): PlayoutEvent {
  if (
    !clock.ready ||
    simTime < clock.latestSim ||
    simTime - clock.latestSim > PLAYOUT_RESET_GAP_S
  ) {
    startClock(clock, simTime, wallMs)
    return 'reset'
  }
  if (simTime === clock.latestSim) return 'stale'

  // 追い越して待たされていたなら、止まっていた位置から張り直す（前へ跳ばない）
  const shown = playoutTime(clock, wallMs)
  measureRate(clock, simTime, wallMs)
  clock.latestSim = simTime
  clock.latestWall = wallMs

  const target = simTime - delaySim(clock)
  const lag = target - shown
  clock.anchorWall = wallMs
  if (lag > clock.rate * PLAYOUT_SNAP_S) {
    clock.anchorSim = target
    clock.slope = clock.rate
    return 'snap'
  }
  const adjust = lag / (clock.rate * PLAYOUT_HORIZON_S)
  clock.anchorSim = shown
  clock.slope = clock.rate * (1 + clampAbs(adjust, PLAYOUT_MAX_ADJUST))
  return 'ok'
}

function delaySim(clock: PlayoutClock): number {
  return (clock.rate * playoutDelayMs(clock)) / 1000
}

function startClock(clock: PlayoutClock, simTime: number, wallMs: number): void {
  clock.ready = true
  clearIntervals(clock)
  clock.latestSim = simTime
  clock.latestWall = wallMs
  clock.anchorWall = wallMs
  clock.anchorSim = simTime - delaySim(clock)
  clock.slope = clock.rate
}

function clearIntervals(clock: PlayoutClock): void {
  clock.settleUntil = 0
  clock.intervalWall.length = 0
  clock.intervalSim.length = 0
}

function measureRate(clock: PlayoutClock, simTime: number, wallMs: number): void {
  const walls = clock.intervalWall
  const sims = clock.intervalSim
  const gap = wallMs - clock.latestWall
  // 止まっていた間隔を入れると再開後の再生が遅すぎ、直後の追いつき（1 通で数ステップ進む）を
  //   入れると速すぎる。どちらも外し、止まったぶんの遅れは補正だけでゆっくり取り戻す
  if (gap > Math.max(PLAYOUT_RATE_GAP_MS, 3 * clock.intervalMs)) {
    clock.settleUntil = wallMs + PLAYOUT_RATE_SETTLE_MS
    return
  }
  if (wallMs < clock.settleUntil) return
  walls.push(gap)
  sims.push(simTime - clock.latestSim)
  if (walls.length > PLAYOUT_RATE_INTERVALS) {
    walls.shift()
    sims.shift()
  }
  let wall = 0
  let sim = 0
  for (let i = 0; i < walls.length; i++) {
    wall += walls[i]
    sim += sims[i]
  }
  if (wall < RATE_MIN_SPAN_MS) return
  clock.rate = Math.min(RATE_MAX, Math.max(RATE_MIN, sim / (wall / 1000)))
  clock.intervalMs = wall / walls.length
}

function clampAbs(x: number, limit: number): number {
  return x > limit ? limit : x < -limit ? -limit : x
}
