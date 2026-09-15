/** エントリポイント。 */

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
setupServiceWorker()

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
