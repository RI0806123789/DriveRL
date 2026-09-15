/** WebSocket クライアント（自動再接続つき）。 */

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
    console.warn('[connection] WebSocket を作成できませんでした', e)
    useSimStore.getState().setConnection('closed')
    scheduleReconnect()
    return
  }

  const socketTransport: Transport = {
    send: (json) => {
      if (ws.readyState === WebSocket.OPEN) ws.send(json)
    },
    close: () => ws.close(),
  }
  transport = socketTransport

  ws.onopen = () => {
    reconnectDelay = RECONNECT_MIN_MS
    useSimStore.getState().setConnection('open')
  }

  ws.onmessage = (ev) => {
    if (typeof ev.data === 'string') handleRaw(ev.data)
  }

  ws.onclose = () => {
    if (transport !== socketTransport) return
    transport = null
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

/** サーバーへ送信する。未接続なら黙って捨てる（例外にしない）。 */
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

if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    stopConnection()
  })
}
