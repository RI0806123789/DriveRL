/** 学習の自動化（`autotune` メッセージ）の表示の形と、トグルの状態の控え（localStorage）。 */

import type { AutotuneMessage, AutotuneOutcome, SimParams } from '../types/protocol'
import type { ChipTone } from '../ui/Chip'

/** トグルの状態を控えるキー。**正はサーバーの `autotune.running`** で、これは前回開いていたときの様子を覚えておくだけ */
export const AUTOTUNE_STORAGE_KEY = 'driverl_auto_tune_active'

/** 画面に出す直近の試行の数 */
export const AUTOTUNE_HISTORY_SHOWN = 8

/** 画面の状況（`useSimStore` から集める） */
export interface AutotuneContext {
  connected: boolean
  practical: boolean
  suspended: boolean
  mapLoaded: boolean
}

export interface AutotuneHistoryItem {
  trial: number
  text: string
  tone: ChipTone
  title: string
}

export interface AutotuneView {
  checked: boolean
  canToggle: boolean
  /** トグルを押せない理由（押せるときは null） */
  blockedReason: string | null
  /** 「自動調整中」のバッジ */
  badge: boolean
  statusText: string
  /** いまの試行の進み具合 0〜1（`--m3-bar-value` に渡す） */
  progress: number
  trialsText: string
  bestText: string
  history: AutotuneHistoryItem[]
}

const OUTCOME_TEXT: Record<AutotuneOutcome, string> = {
  complete: '完了',
  pruned: '打ち切り',
  diverged: '発散',
}

function scoreText(score: number | null | undefined): string {
  if (typeof score !== 'number' || !Number.isFinite(score)) return '—'
  return `${score >= 0 ? '+' : ''}${score.toFixed(3)}`
}

function clamp01(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 0
  return Math.min(1, Math.max(0, value))
}

/** トグルを押せない理由。押せるなら null（止めるほうはいつでも押せる） */
function blocked(msg: AutotuneMessage | null, ctx: AutotuneContext): string | null {
  if (msg?.running) return ctx.connected ? null : 'サーバーに接続していません'
  if (!ctx.connected) return 'サーバーに接続していません'
  if (msg === null) return 'サーバーの状態を待っています'
  if (!msg.available) return 'Optuna が入っていません。backend/.venv の Python で pip install -r requirements.txt を実行し、サーバーを起動し直してください'
  if (!ctx.mapLoaded) return 'マップを読み込んでから始めてください'
  if (ctx.practical) return '実用モードの間は学習しないため始められません'
  if (ctx.suspended) return '認識器の学習中は始められません'
  return null
}

function statusText(msg: AutotuneMessage | null): string {
  if (msg === null) return '—'
  if (!msg.running) return msg.message || '止まっています'
  if (msg.phase === 'preparing') return '探索の履歴を開いています'
  if (msg.phase === 'waiting' || msg.trial === null) return '次の試行のパラメータを選んでいます'
  return `試行 #${msg.trial} を走らせています（${Math.round(clamp01(msg.trialProgress) * 100)}%）`
}

/** `autotune` メッセージと画面の状況から、カードの表示を作る。 */
export function autotuneView(msg: AutotuneMessage | null, ctx: AutotuneContext): AutotuneView {
  const reason = blocked(msg, ctx)
  const running = msg?.running ?? false
  const finished = Math.max(0, Math.floor(msg?.finishedTrials ?? 0))
  const prior = Math.max(0, Math.floor(msg?.priorTrials ?? 0))
  const history = (msg?.history ?? []).slice(-AUTOTUNE_HISTORY_SHOWN).map((t) => ({
    trial: t.trial,
    text: `#${t.trial} ${scoreText(t.score)}`,
    tone: (t.outcome === 'complete' ? 'ok' : t.outcome === 'pruned' ? 'neutral' : 'error') as ChipTone,
    title: `試行 #${t.trial}: ${OUTCOME_TEXT[t.outcome] ?? t.outcome}（スコア ${scoreText(t.score)}）`,
  }))
  return {
    checked: running,
    canToggle: reason === null,
    blockedReason: reason,
    badge: running,
    statusText: statusText(msg),
    progress: running && msg?.phase === 'running' ? clamp01(msg.trialProgress) : 0,
    trialsText: `この探索で ${finished.toLocaleString()} 試行` + (prior > 0 ? `（前回までの完了 ${prior.toLocaleString()} 件）` : ''),
    bestText: msg?.best ? `試行 #${msg.best.trial}（スコア ${scoreText(msg.best.score)}）` : 'まだありません',
    history,
  }
}

/** その params のキーをいま学習の自動化が決めているか（スライダーを動かせなくする）。 */
export function isAutotuned(msg: AutotuneMessage | null, key: keyof SimParams): boolean {
  return Boolean(msg?.running && msg.tunedKeys?.includes(key))
}

/** 接続し直した最初の `autotune` が止まっているのに、前回は動いていたなら知らせる文。それ以外は null */
export function lostSessionNotice(previous: boolean | null, msg: AutotuneMessage): string | null {
  if (previous !== true || msg.running) return null
  return (
    '前回開いていたときは学習の自動化が動いていましたが、いまは止まっています（サーバーの再起動など）。' +
    '最良の試行の重みは backend/data/checkpoints/best_tuned_policy.pt、パラメータは backend/data/tuning/best_params.json に残っています'
  )
}

function storage(): Storage | null {
  try {
    return typeof localStorage === 'undefined' ? null : localStorage
  } catch {
    return null
  }
}

/** 前回控えたトグルの状態。控えが無い・読めないときは null。 */
export function readAutotuneFlag(store: Storage | null = storage()): boolean | null {
  try {
    const raw = store?.getItem(AUTOTUNE_STORAGE_KEY)
    return raw === 'true' ? true : raw === 'false' ? false : null
  } catch {
    return null
  }
}

/** トグルの状態を控える。書けなくても画面は動く（プライベートウィンドウなど）。 */
export function writeAutotuneFlag(running: boolean, store: Storage | null = storage()): void {
  try {
    store?.setItem(AUTOTUNE_STORAGE_KEY, running ? 'true' : 'false')
  } catch {
    // 控えは便宜のためだけなので、書けなくても困らない
  }
}
