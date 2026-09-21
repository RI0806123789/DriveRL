/** 配車を最初から最後まで自分で回す（`TaxiScreen` の隠し操作から入る）。 */

import type { MapNode, TaxiPhase, Vec2 } from '../types/protocol'

/** 降車してから次を呼ぶまで [秒] */
export const IDLE_WAIT_SEC = 1.5
/** 到着してから降りるまで [秒] */
export const ARRIVED_WAIT_SEC = 1.0
/** 同じ指令を送り直すまで [秒]。断られたときはこの間隔で再試行する */
export const RETRY_SEC = 1.5

/** 行き先に選ぶ距離の範囲 [m]。金沢でも経路が作れる長さに収める */
export const TRIP_MIN_M = 150
export const TRIP_MAX_M = 1200

/** 行き先の候補を何回引くか */
const PICK_TRIES = 24

export type AutoAction = 'none' | 'request' | 'board' | 'alight' | 'approach'

export interface AutoInput {
  phase: TaxiPhase
  /** 徴用している車両スロット */
  vehicleId: number
  /** 照準に入っている車両スロット。-1 なら無し */
  aimed: number
  /** 徒歩キャラが街に立っているか */
  placed: boolean
  /** エリアが読み込まれているか */
  mapReady: boolean
  /** いまの時刻 [秒] */
  now: number
  /** 直前に指令を送った時刻 [秒] */
  lastSentAt: number
}

/**
 * いま何をすべきかを決める。**送信も歩行もしない**（呼んだ側が実行する）。
 */
export function decideAutoAction(input: AutoInput): AutoAction {
  if (!input.mapReady || !input.placed) return 'none'
  const since = input.now - input.lastSentAt

  switch (input.phase) {
    case 'idle':
      return since >= IDLE_WAIT_SEC ? 'request' : 'none'
    case 'approaching':
      return 'none'
    case 'waiting':
      if (input.aimed === input.vehicleId) return since >= RETRY_SEC ? 'board' : 'none'
      return 'approach'
    case 'riding':
      return 'none'
    case 'arrived':
      return since >= ARRIVED_WAIT_SEC ? 'alight' : 'none'
    default:
      return 'none'
  }
}

/**
 * 行き先にする道路ノードを選ぶ。近すぎ・遠すぎを避け、見つからなければ最も近い候補を返す。
 */
export function pickDropoff(
  nodes: readonly MapNode[],
  fromX: number,
  fromY: number,
  random: () => number,
  tries: number = PICK_TRIES,
): Vec2 | null {
  if (nodes.length === 0) return null
  const middle = (TRIP_MIN_M + TRIP_MAX_M) / 2
  let fallback: Vec2 | null = null
  let fallbackGap = Infinity

  for (let i = 0; i < tries; i++) {
    const node = nodes[Math.min(nodes.length - 1, Math.floor(random() * nodes.length))]
    const dist = Math.hypot(node.x - fromX, node.y - fromY)
    if (dist >= TRIP_MIN_M && dist <= TRIP_MAX_M) return [node.x, node.y]
    const gap = Math.abs(dist - middle)
    if (gap < fallbackGap) {
      fallbackGap = gap
      fallback = [node.x, node.y]
    }
  }
  return fallback
}

/** 自動操作のうち、毎フレーム触るぶん。**zustand に入れない**（`pedestrian` と同じ作法） */
export const taxiAutopilot = {
  /** 直前に指令を送った（か段階が変わった）時刻 [秒] */
  lastSentAt: 0,
  /** 直前に見た段階。変わったら時計を入れ直す */
  lastPhase: 'idle' as TaxiPhase,
  /** いま自動操作が前進キーを押しているか */
  walking: false,
}

/** 入り直したときに前回の続きから動き出さないよう、時計と足を戻す。 */
export function resetTaxiAutopilot(now: number): void {
  taxiAutopilot.lastSentAt = now
  taxiAutopilot.lastPhase = 'idle'
  taxiAutopilot.walking = false
}
