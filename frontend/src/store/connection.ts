/**
 * WebSocket クライアント（自動再接続つき）。
 *
 * 接続先は `ws://${location.host}/ws`。
 * Vite の dev server が /ws を 8000 番（FastAPI）へプロキシするので、
 * フロントは同一オリジンの相対パスだけを見ればよい。
 *
 * 開発用モック:
 *   `?mock=1` を付けて DEV ビルドを開くと、実サーバーの代わりに
 *   store/mockServer.ts の内蔵フェイクサーバーに接続する。
 *   ★ これはあくまで開発補助。本番では必ず実サーバーに繋ぐこと。
 */

import type { ClientMessage, ServerMessage } from '../types/protocol'
import { useSimStore } from './simStore'

/** 実 WebSocket とモックを同じ形で扱うための最小インターフェース */
export interface Transport {
  send(json: string): void
  close(): void
}

const RECONNECT_MIN_MS = 300
const RECONNECT_MAX_MS = 5000

let transport: Transport | null = null
let reconnectTimer: ReturnType<typeof setTimeout> | null = null
let reconnectDelay = RECONNECT_MIN_MS
let started = false
let disposed = false
let mockMode = false

/** URL に ?mock=1 が付いた DEV ビルドか */
export function isMockRequested(): boolean {
  if (!import.meta.env.DEV) return false
  try {
    return new URLSearchParams(window.location.search).get('mock') === '1'
  } catch {
    return false
  }
}

function handleRaw(data: string): void {
  let msg: ServerMessage
  try {
    msg = JSON.parse(data) as ServerMessage
  } catch (e) {
    console.warn('[connection] JSON の解析に失敗しました', e)
    return
  }
  if (!msg || typeof (msg as { type?: unknown }).type !== 'string') {
    console.warn('[connection] type を持たないメッセージを無視しました', msg)
    return
  }
  useSimStore.getState().applyServerMessage(msg)
}

function scheduleReconnect(): void {
  if (disposed || mockMode) return
  if (reconnectTimer !== null) return
  const delay = reconnectDelay
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null
    openSocket()
  }, delay)
  // 指数バックオフ（上限 5 秒）
  reconnectDelay = Math.min(RECONNECT_MAX_MS, Math.round(reconnectDelay * 1.8))
}

function openSocket(): void {
  if (disposed) return

  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const url = `${proto}//${window.location.host}/ws`

  useSimStore.getState().setConnection('connecting')

  let ws: WebSocket
  try {
    ws = new WebSocket(url)
  } catch (e) {
    // コンストラクタ自体が投げるケース（不正な URL など）
    console.warn('[connection] WebSocket を作成できませんでした', e)
    useSimStore.getState().setConnection('closed')
    scheduleReconnect()
    return
  }

  transport = {
    send: (json) => {
      if (ws.readyState === WebSocket.OPEN) ws.send(json)
    },
    close: () => ws.close(),
  }

  ws.onopen = () => {
    reconnectDelay = RECONNECT_MIN_MS
    useSimStore.getState().setConnection('open')
  }

  ws.onmessage = (ev) => {
    if (typeof ev.data === 'string') handleRaw(ev.data)
  }

  ws.onerror = () => {
    // onerror の直後には必ず onclose が来るので、ここでは何もしない。
    // （バックエンド未起動時にコンソールを汚さないため console.error を出さない）
  }

  ws.onclose = () => {
    if (transport && ws.readyState === WebSocket.CLOSED) transport = null
    useSimStore.getState().setConnection('closed')
    scheduleReconnect()
  }
}

/** アプリ起動時に 1 回だけ呼ぶ */
export function startConnection(): void {
  if (started) return
  started = true
  disposed = false

  if (isMockRequested()) {
    mockMode = true
    useSimStore.getState().setUsingMock(true)
    useSimStore.getState().setConnection('connecting')
    // 動的 import にしておくと、本番ビルドでは別チャンクに切り出され
    // （かつ isMockRequested() が常に false なので）読み込まれない。
    void import('./mockServer')
      .then((m) => {
        if (disposed) return
        transport = m.createMockTransport(handleRaw)
        useSimStore.getState().setConnection('open')
      })
      .catch((e) => {
        console.error('[connection] モックサーバーの読込に失敗しました', e)
        useSimStore.getState().setConnection('closed')
      })
    return
  }

  openSocket()
}

/** HMR / アンマウント時の後片付け */
export function stopConnection(): void {
  disposed = true
  started = false
  if (reconnectTimer !== null) {
    clearTimeout(reconnectTimer)
    reconnectTimer = null
  }
  transport?.close()
  transport = null
  useSimStore.getState().setConnection('closed')
}

/**
 * サーバーへ送信する。未接続なら黙って捨てる（例外にしない）。
 * @returns 実際に送れたら true
 */
export function send(msg: ClientMessage): boolean {
  if (!transport) return false
  if (useSimStore.getState().connection !== 'open') return false
  try {
    transport.send(JSON.stringify(msg))
    return true
  } catch (e) {
    console.warn('[connection] 送信に失敗しました', e)
    return false
  }
}

// Vite の HMR で二重接続にならないようにする
if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    stopConnection()
  })
}
