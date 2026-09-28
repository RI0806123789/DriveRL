/** 安全ギミックの介入（`frame.vehicles[].assist`）を、4 分割のペインごとの表示へ写す純粋関数。 */

import type { AssistKind } from '../types/protocol.ts'
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
    case 'yield':
      return { text: '接近車あり: 交差点の手前で待機', level: 'stop' }
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
  return null
}
