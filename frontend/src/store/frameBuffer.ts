/** frame（20Hz）専用の可変バッファ。 */

import type { FrameMessage, ObstacleState, Vec2, VehicleState } from '../types/protocol'

/** 補間に必要な 2 フレーム分 + 受信時刻を保持する箱 */
export interface FrameBuffer {
  /** 1 つ前のフレーム（補間の始点） */
  prev: FrameMessage | null
  /** 最新フレーム（補間の終点） */
  curr: FrameMessage | null
  /** prev を受信した時刻（performance.now()） */
  prevTime: number
  /** curr を受信した時刻（performance.now()） */
  currTime: number
  /** 直近の受信間隔 [ms]。既定は 20Hz = 50ms */
  intervalMs: number
  /** これまでに受信したフレーム数 */
  received: number
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
}

const EMPTY_ROUTES = new Map<number, Vec2[]>()
const EMPTY_REVISIONS = new Map<number, number>()

export const frameBuffer: FrameBuffer = {
  prev: null,
  curr: null,
  prevTime: 0,
  currTime: 0,
  intervalMs: 50,
  received: 0,
  routes: EMPTY_ROUTES,
  routeRevisions: EMPTY_REVISIONS,
  routeVersion: 0,
  obstacles: [],
  obstacleVersion: 0,
  signals: [],
  signalVersion: 0,
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

  frameBuffer.prev = frameBuffer.curr
  frameBuffer.prevTime = frameBuffer.currTime
  frameBuffer.curr = frame
  frameBuffer.currTime = now
  frameBuffer.received += 1

  if (frameBuffer.prev) {
    const dt = frameBuffer.currTime - frameBuffer.prevTime
    if (dt > 1) frameBuffer.intervalMs = Math.min(500, Math.max(16, dt))
  }

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

  frameBuffer.obstacles = frame.obstacles
  const sig = obstacleSignature(frame.obstacles)
  if (sig !== lastObstacleSig) {
    lastObstacleSig = sig
    frameBuffer.obstacleVersion += 1
  }
}

/** 切断・マップ切替時にバッファを空にする */
export function resetFrameBuffer(): void {
  frameBuffer.prev = null
  frameBuffer.curr = null
  frameBuffer.prevTime = 0
  frameBuffer.currTime = 0
  frameBuffer.intervalMs = 50
  frameBuffer.received = 0
  frameBuffer.routes = new Map()
  frameBuffer.routeRevisions = new Map()
  frameBuffer.routeVersion += 1
  frameBuffer.obstacles = []
  frameBuffer.obstacleVersion += 1
  frameBuffer.signals = []
  frameBuffer.signalVersion += 1
  lastObstacleSig = ''
}

/** スロット番号から最新の車両状態を引く（補間なしの生値） */
export function getLatestVehicle(id: number): VehicleState | null {
  const curr = frameBuffer.curr
  if (!curr) return null
  for (const v of curr.vehicles) if (v.id === id) return v
  return null
}
