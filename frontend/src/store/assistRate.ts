/** オンライン模倣のアシスト率（`metrics.assistRate`）の正規化と表示の形。 */

import type { ChipTone } from '../ui/Chip'

/** 割り込みの確率の下限（`backend/app/rl/online_assist.py` の `ASSIST_P_MIN`） */
export const ASSIST_P_MIN = 0.05

/** 画面の表示の形 */
export interface AssistRateView {
  /** 0〜100 の整数。値が届いていなければ null */
  percent: number | null
  text: string
  tone: ChipTone
}

/** 受け取った値を 0〜1 に収める。数でない・届いていないときは null。 */
export function normalizeAssistRate(rate: unknown): number | null {
  if (typeof rate !== 'number' || !Number.isFinite(rate)) return null
  return Math.min(1, Math.max(0, rate))
}

/** チップに出す文言と色。高いほど方策がまだ自分で走れていない。 */
export function assistRateView(rate: unknown): AssistRateView {
  const value = normalizeAssistRate(rate)
  if (value === null) return { percent: null, text: 'Assist Rate —', tone: 'neutral' }
  const percent = Math.round(value * 100)
  const tone: ChipTone = value >= 0.5 ? 'warning' : value > ASSIST_P_MIN * 2 ? 'neutral' : 'ok'
  return { percent, text: `Assist Rate ${percent}%`, tone }
}
