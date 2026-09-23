/** frame（20Hz）専用の可変バッファ。 */

import type {
  FrameMessage,
  ObstacleState,
  Vec2,
  VehicleState,
  WeatherState,
} from '../types/protocol'
import {
  createPlayoutClock,
  notePlayoutArrival,
  playoutTime,
  resetPlayoutClock,
} from './playout'
import type { PlayoutClock } from './playout'

/** 補間に使う 2 フレームと、再生を待っているフレームを保持する箱 */
export interface FrameBuffer {
  /** 表示している区間の始点。**受信順ではなく再生の時計（`clock`）で進む** */
  prev: FrameMessage | null
  /** 表示している区間の終点（表示に使っている最新のフレーム） */
  curr: FrameMessage | null
  /** 届いたがまだ表示に使っていないフレーム（simTime の昇順） */
  pending: FrameMessage[]
  /** simTime を時計にした再生の状態 */
  clock: PlayoutClock
  /** これまでに受信したフレーム数 */
  received: number
  /** prev / curr が入れ替わるたびに増える。表示に使うフレームが変わった合図 */
  displayed: number
  /** 車両ごとの最新経路。frame.route は変化時のみ届くので、ここで保持する */
  routes: Map<number, Vec2[]>
  /** 車両ごとの経路の版。その車両の経路が届くたびに +1 する。 */
  routeRevisions: Map<number, number>
  /** routes が変わるたびに増える。3D 側はこれを見て作り直しの要否を判断する */
  routeVersion: number
  /** 最新の障害物一覧 */
  obstacles: ObstacleState[]
  /** 障害物の集合が変わるたびに増える */
  obstacleVersion: number
  /** 信号の現示（0=青 / 1=黄 / 2=赤）。map.signals と同じ並び */
  signals: number[]
  /** 現示が変わるたびに増える。3D 側はこれを見て灯色を塗り替える */
  signalVersion: number
  /** いま効いている天候。weatherAuto の間は params ではなくこちらが正 */
  weather: WeatherState
}

const CLEAR_WEATHER: WeatherState = { rain: 0, fog: 0, visibility: 120 }

const EMPTY_ROUTES = new Map<number, Vec2[]>()
const EMPTY_REVISIONS = new Map<number, number>()

export const frameBuffer: FrameBuffer = {
  prev: null,
  curr: null,
  pending: [],
  clock: createPlayoutClock(),
  received: 0,
  displayed: 0,
  routes: EMPTY_ROUTES,
  routeRevisions: EMPTY_REVISIONS,
  routeVersion: 0,
  obstacles: [],
  obstacleVersion: 0,
  signals: [],
  signalVersion: 0,
  weather: CLEAR_WEATHER,
}

/** 障害物集合の同一性を安く判定するための署名 */
function obstacleSignature(obstacles: ObstacleState[]): string {
  let sig = ''
  for (const o of obstacles) sig += o.id + ':' + o.x.toFixed(1) + ',' + o.y.toFixed(1) + ';'
  return sig
}

let lastObstacleSig = ''

/** 新しい frame を受け取ってバッファを進める */
export function pushFrame(frame: FrameMessage): void {
  const now = performance.now()
  frameBuffer.received += 1

  const event = notePlayoutArrival(frameBuffer.clock, frame.simTime, now)
  if (event === 'reset') {
    frameBuffer.prev = null
    frameBuffer.curr = frame
    frameBuffer.pending.length = 0
    frameBuffer.displayed += 1
  } else if (event !== 'stale') {
    frameBuffer.pending.push(frame)
  }
  // 描画が止まっているタブでも待ち行列を溜め込まない
  advanceDisplay(now, false)

  let routeChanged = false
  for (const v of frame.vehicles) {
    if (v.route !== undefined) {
      if (frameBuffer.routes === EMPTY_ROUTES) frameBuffer.routes = new Map()
      if (frameBuffer.routeRevisions === EMPTY_REVISIONS) frameBuffer.routeRevisions = new Map()
      frameBuffer.routes.set(v.id, v.route)
      frameBuffer.routeRevisions.set(v.id, (frameBuffer.routeRevisions.get(v.id) ?? 0) + 1)
      routeChanged = true
    }
    if (!v.active && frameBuffer.routes.has(v.id)) {
      frameBuffer.routes.delete(v.id)
      frameBuffer.routeRevisions.delete(v.id)
      routeChanged = true
    }
  }
  if (routeChanged) frameBuffer.routeVersion += 1

  const nextSignals = frame.signals
  if (nextSignals) {
    const prevSignals = frameBuffer.signals
    let changed = prevSignals.length !== nextSignals.length
    if (!changed) {
      for (let i = 0; i < nextSignals.length; i++) {
        if (prevSignals[i] !== nextSignals[i]) {
          changed = true
          break
        }
      }
    }
    if (changed) {
      frameBuffer.signals = nextSignals
      frameBuffer.signalVersion += 1
    }
  }

  if (frame.weather) frameBuffer.weather = frame.weather

  frameBuffer.obstacles = frame.obstacles
  const sig = obstacleSignature(frame.obstacles)
  if (sig !== lastObstacleSig) {
    lastObstacleSig = sig
    frameBuffer.obstacleVersion += 1
  }
}

/** 再生の時計まで表示の区間（prev / curr）を進め、区間内の補間係数 0..1 を返す。 */
export function advanceDisplay(nowMs: number, paused: boolean): number {
  if (!frameBuffer.curr) return 1
  if (paused) {
    // 止めている間は届いている最新を出す。再開後の simTime は飛ぶので時計を張り直す
    resetPlayoutClock(frameBuffer.clock)
    promoteUntil(Infinity)
    return 1
  }
  const t = playoutTime(frameBuffer.clock, nowMs)
  promoteUntil(t)
  const prev = frameBuffer.prev
  if (!prev) return 1
  const span = frameBuffer.curr.simTime - prev.simTime
  if (!(span > 0)) return 1
  const a = (t - prev.simTime) / span
  return a <= 0 ? 0 : a >= 1 ? 1 : a
}

function promoteUntil(simTime: number): void {
  let next = frameBuffer.pending[0]
  while (next !== undefined && frameBuffer.curr !== null && frameBuffer.curr.simTime <= simTime) {
    frameBuffer.prev = frameBuffer.curr
    frameBuffer.curr = next
    frameBuffer.pending.shift()
    frameBuffer.displayed += 1
    next = frameBuffer.pending[0]
  }
}

/** 切断・マップ切替時にバッファを空にする */
export function resetFrameBuffer(): void {
  frameBuffer.prev = null
  frameBuffer.curr = null
  frameBuffer.pending.length = 0
  resetPlayoutClock(frameBuffer.clock)
  frameBuffer.received = 0
  frameBuffer.displayed += 1
  frameBuffer.routes = new Map()
  frameBuffer.routeRevisions = new Map()
  frameBuffer.routeVersion += 1
  frameBuffer.obstacles = []
  frameBuffer.obstacleVersion += 1
  frameBuffer.signals = []
  frameBuffer.signalVersion += 1
  frameBuffer.weather = CLEAR_WEATHER
  lastObstacleSig = ''
}

/** スロット番号から表示中の車両状態を引く（補間なしの生値） */
export function getLatestVehicle(id: number): VehicleState | null {
  const curr = frameBuffer.curr
  if (!curr) return null
  for (const v of curr.vehicles) if (v.id === id) return v
  return null
}
