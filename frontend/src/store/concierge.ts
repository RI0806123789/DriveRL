/** 実用モードの車載 AI コンシェルジュ（`POST /api/taxi/concierge`）とのやり取り。 */

import { create } from 'zustand'

import type { TaxiDriveMode } from '../types/protocol'

export type ConciergeAction = 'hurry' | 'comfort' | 'normal' | 'explain'

export interface ConciergeEntry {
  id: number
  role: 'user' | 'ai' | 'error'
  text: string
}

/** 画面に残す発言の数。古いものから捨てる */
export const CONCIERGE_KEEP = 12

export const CONCIERGE_MAX_CHARS = 200

export const DRIVE_MODE_LABELS: Record<TaxiDriveMode, string> = {
  normal: '標準',
  hurry: '少し急いで',
  comfort: '快適重視',
}

export const QUICK_ACTION_LABELS: Record<Exclude<ConciergeAction, 'normal'>, string> = {
  hurry: '⚡ 少し急いで',
  comfort: '🛋️ 快適重視',
  explain: '❓ 停車理由',
}

export const CONCIERGE_GREETING = 'ご用件をどうぞ。走り方の調整や、停まっている理由をお答えします。'

interface ConciergeReplyBody {
  ok?: boolean
  reply?: string
  error?: string
}

/** 応答の本文から画面に出す 1 行を作る。 */
export function entryFromResponse(status: number, body: unknown): Omit<ConciergeEntry, 'id'> {
  const data = (body && typeof body === 'object' ? body : {}) as ConciergeReplyBody
  if (status >= 200 && status < 300 && data.ok && typeof data.reply === 'string' && data.reply) {
    return { role: 'ai', text: data.reply }
  }
  if (typeof data.error === 'string' && data.error) return { role: 'error', text: data.error }
  return { role: 'error', text: `AI コンシェルジュがエラーを返しました（HTTP ${status}）` }
}

interface ConciergeState {
  /** null はまだ確かめていない */
  available: boolean | null
  pending: boolean
  entries: ConciergeEntry[]
  checkAvailability(): Promise<void>
  ask(request: { message: string } | { action: ConciergeAction }): Promise<void>
  clear(): void
}

let entrySeq = 0

function push(entries: ConciergeEntry[], entry: Omit<ConciergeEntry, 'id'>): ConciergeEntry[] {
  entrySeq += 1
  return [...entries, { ...entry, id: entrySeq }].slice(-CONCIERGE_KEEP)
}

export const useConcierge = create<ConciergeState>((set, get) => ({
  available: null,
  pending: false,
  entries: [],

  async checkAvailability() {
    try {
      const response = await fetch('/api/taxi/concierge')
      const body = (await response.json()) as { available?: boolean }
      set({ available: response.ok && body.available === true })
    } catch {
      set({ available: false })
    }
  },

  async ask(request) {
    if (get().pending) return
    let said: string
    if ('message' in request) {
      said = request.message.trim().slice(0, CONCIERGE_MAX_CHARS)
      if (!said) return
    } else {
      said = request.action === 'normal' ? '標準の走り方に戻して' : QUICK_ACTION_LABELS[request.action]
    }
    const payload = 'message' in request ? { message: said } : { action: request.action }
    set((s) => ({ pending: true, entries: push(s.entries, { role: 'user', text: said }) }))

    let entry: Omit<ConciergeEntry, 'id'>
    try {
      const response = await fetch('/api/taxi/concierge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      let body: unknown = null
      try {
        body = await response.json()
      } catch {
        body = null
      }
      entry = entryFromResponse(response.status, body)
    } catch {
      entry = { role: 'error', text: 'サーバーに接続できませんでした' }
    }
    set((s) => ({ pending: false, entries: push(s.entries, entry) }))
  },

  clear() {
    set({ entries: [], pending: false })
  },
}))
