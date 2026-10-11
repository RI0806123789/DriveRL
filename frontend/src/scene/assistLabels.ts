/** 安全ギミックの介入（`frame.vehicles[].assist`）を、4 分割のペインごとの表示へ写す純粋関数。 */

import type { AssistKind, IntersectionDecision } from '../types/protocol.ts'
import type { QuadPane } from './quadLayout.ts'

export type AssistLevel = 'info' | 'warn' | 'stop'

export interface AssistBadge {
  text: string
  level: AssistLevel
}

/** 画面の上に 1 行で出す、いま何をしているか。介入していなければ null */
export function assistSummary(kind: AssistKind | undefined): AssistBadge | null {
  switch (kind) {
    case 'front_hold':
      return { text: '前方に障害物: 停止中', level: 'stop' }
    case 'reverse_check':
      return { text: '立ち往生: 後方と左右を確認中', level: 'warn' }
    case 'reversing':
      return { text: '切り返し: 後退中', level: 'warn' }
    case 'reverse_stop':
      return { text: '後方に障害物: 自動停止', level: 'stop' }
    case 'detour':
      return { text: '障害物の脇を通過中', level: 'warn' }
    case 'blind_spot':
      return { text: '巻き込み確認: 一時停止', level: 'stop' }
    case 'peek':
      return { text: '交差点: 徐行して左右確認', level: 'info' }
    case 'creep':
      return { text: '見通しの悪い交差点: 車頭をゆっくり出して確認', level: 'warn' }
    case 'yield':
      return { text: '接近車あり: 交差点の手前で待機', level: 'stop' }
    case 'intersection_yield':
      return { text: '交差点: 優先される交通に譲る', level: 'stop' }
    case 'intersection_stop':
      return { text: '交差点: 一時停止を確認', level: 'stop' }
    case 'intersection_exit':
      return { text: '交差点: 出口の空きを待つ', level: 'stop' }
    case 'intersection_visibility':
      return { text: '交差点: 見通しとカメラ情報を確認', level: 'warn' }
    default:
      return null
  }
}

/** そのペインのカメラが介入の根拠を写しているときの表示。`turnSignal` は曲がる側（-1=左 / +1=右） */
export function paneAssist(
  kind: AssistKind | undefined,
  pane: QuadPane,
  turnSignal: number,
): AssistBadge | null {
  if (!kind) return null
  if (pane === 'front') {
    if (kind === 'front_hold') return { text: '障害物で停止', level: 'stop' }
    if (kind === 'detour') return { text: '回避中', level: 'warn' }
    if (kind === 'intersection_yield') return { text: '対向車・横断者の確認', level: 'stop' }
    if (kind === 'intersection_stop') return { text: '一時停止', level: 'stop' }
    if (kind === 'intersection_exit') return { text: '出口の詰まりを確認', level: 'stop' }
    if (kind === 'intersection_visibility') return { text: 'カメラ情報の確認', level: 'warn' }
    return null
  }
  if (pane === 'rear') {
    if (kind === 'reverse_check') return { text: '後方確認', level: 'warn' }
    if (kind === 'reversing') return { text: '後退中', level: 'warn' }
    if (kind === 'reverse_stop') return { text: '自動停止', level: 'stop' }
    return null
  }
  const side = pane === 'left' ? -1 : 1
  if (kind === 'reverse_check') return { text: '側方確認', level: 'warn' }
  if (kind === 'blind_spot' && turnSignal === side) return { text: '巻き込み確認', level: 'stop' }
  if (kind === 'yield') return { text: '接近車を確認', level: 'stop' }
  if (kind === 'peek') return { text: '左右確認', level: 'info' }
  if (kind === 'creep') return { text: '見通し確認', level: 'warn' }
  if (kind === 'intersection_yield') return { text: '交錯する交通を確認', level: 'stop' }
  if (kind === 'intersection_visibility') return { text: '見通し確認', level: 'warn' }
  return null
}

export interface IntersectionBadge extends AssistBadge {
  detail: string
}

const INTERSECTION_STATES: Record<IntersectionDecision['state'], string> = {
  approaching: '接近中',
  waiting: '進入待ち',
  clear: '進入可能',
  committed: '通過中',
}

const INTERSECTION_REASONS: Record<IntersectionDecision['reason'], string> = {
  '': '',
  oncoming: '対向車に譲る',
  pedestrian: '横断者に譲る',
  exit_blocked: '出口の空きを待つ',
  stop: '一時停止',
  priority: '優先される交通に譲る',
  visibility: '見通し・カメラ情報を確認',
}

/** 交差点判断の状態・根拠と、操作へ実際に反映したかを表示用にまとめる。 */
export function intersectionSummary(decision: IntersectionDecision | undefined): IntersectionBadge | null {
  if (!decision) return null
  const state = INTERSECTION_STATES[decision.state]
  const reason = INTERSECTION_REASONS[decision.reason]
  if (!state || reason === undefined) return null
  const waiting = Number.isFinite(decision.waitingSeconds) ? Math.max(0, decision.waitingSeconds) : 0
  const source = decision.source === 'background' ? '背景交通の判断' : 'カメラ判断'
  const intervention = decision.imposed ? '操作介入あり' : '判定のみ'
  const details = [`待機 ${waiting.toFixed(1)}秒`, source, intervention]
  if (decision.gapSeconds !== null && Number.isFinite(decision.gapSeconds)) {
    details.push(`間隔 ${decision.gapSeconds.toFixed(1)}秒`)
  }
  return {
    text: `交差点: ${state}${reason ? `（${reason}）` : ''}`,
    detail: details.join(' · '),
    level: decision.state === 'waiting' ? (decision.reason === 'visibility' ? 'warn' : 'stop') : 'info',
  }
}
