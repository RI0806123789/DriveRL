/**
 * エントリポイント。
 *
 * WebSocket は React のライフサイクルの外（モジュールスコープ）で 1 本だけ張る。
 * StrictMode の二重マウントで接続が 2 本にならないよう、startConnection() 自身が
 * 多重呼び出しを弾く実装になっている。
 */

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './App'
import { startConnection } from './store/connection'
import { setupServiceWorker } from './store/pwa'

import './styles/tokens.css'
import './styles/global.css'

const container = document.getElementById('root')
if (!container) {
  throw new Error('#root が見つかりません。index.html を確認してください。')
}

startConnection()
// PWA（インストールと全画面起動）。本番ビルドのときだけ登録される
setupServiceWorker()

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
