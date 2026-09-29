/** 階層型の方策の意図（`frame.vehicles[].currentOption` / `metrics.optionShares`）の正規化と、バッジの配色・アイコン。 */

import { DRIVE_OPTIONS, type DriveOption } from '../types/protocol.ts'

/** バッジ 1 つぶんの見た目 */
export interface DriveOptionStyle {
  option: DriveOption
  /** 日本語の呼び名 */
  label: string
  /** バッジの地の色（青 / 緑 / 黄 / 赤） */
  hue: 'blue' | 'green' | 'yellow' | 'red'
  /** `styles/global.css` のクラス */
  className: string
  /** 地の色のトークン（`styles/tokens.css`） */
  token: string
  /** 名前の前に置く記号 */
  icon: string
  /** 下位方策に期待する動き */
  description: string
}

export const DRIVE_OPTION_STYLES: Record<DriveOption, DriveOptionStyle> = {
  CRUISE: {
    option: 'CRUISE',
    label: '巡航',
    hue: 'blue',
    className: 'option-badge--cruise',
    token: '--m3-option-cruise-container',
    icon: '▶',
    description: '規制速度へ向けて滑らかに加速し、車線の中央をなぞる',
  },
  FOLLOW: {
    option: 'FOLLOW',
    label: '追従',
    hue: 'green',
    className: 'option-badge--follow',
    token: '--m3-option-follow-container',
    icon: '⇉',
    description: '前走車の速さに合わせ、車間を保って走る',
  },
  YIELD: {
    option: 'YIELD',
    label: '徐行',
    hue: 'yellow',
    className: 'option-badge--yield',
    token: '--m3-option-yield-container',
    icon: '△',
    description: '交差点や歩行者の手前で 10km/h 以下まで落とし、左右を確かめる',
  },
  STOP: {
    option: 'STOP',
    label: '停止',
    hue: 'red',
    className: 'option-badge--stop',
    token: '--m3-option-stop-container',
    icon: '■',
    description: '赤信号の停止線や障害物の手前で止まって待つ',
  },
}

/** 受け取った値が意図の名前ならそれを、違えば null を返す。 */
export function normalizeDriveOption(value: unknown): DriveOption | null {
  return typeof value === 'string' && (DRIVE_OPTIONS as readonly string[]).includes(value)
    ? (value as DriveOption)
    : null
}

/** 意図の割合の 1 行 */
export interface OptionShareRow {
  style: DriveOptionStyle
  /** 0〜1 */
  share: number
  /** 0〜100 の整数 */
  percent: number
}

/** `metrics.optionShares` を `DRIVE_OPTIONS` の順の行にする。届いていない・形が違うときは null。 */
export function optionShareRows(shares: unknown): OptionShareRow[] | null {
  if (!Array.isArray(shares) || shares.length !== DRIVE_OPTIONS.length) return null
  const values = shares.map((v) => (typeof v === 'number' && Number.isFinite(v) ? Math.max(0, v) : NaN))
  if (values.some((v) => Number.isNaN(v))) return null
  const total = values.reduce((a, b) => a + b, 0)
  if (total <= 0) return null
  return DRIVE_OPTIONS.map((option, i) => {
    const share = values[i] / total
    return { style: DRIVE_OPTION_STYLES[option], share, percent: Math.round(share * 100) }
  })
}

/** 加加速度の表示。届いていなければ「—」。 */
export function jerkText(jerk: unknown): string {
  if (typeof jerk !== 'number' || !Number.isFinite(jerk)) return '—'
  return `${jerk.toFixed(1)} m/s³`
}
