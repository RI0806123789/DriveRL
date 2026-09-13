/**
 * 全体レイアウト（memo 4章）。
 *
 * 左（シミュレーター）: 右（操作パネル） = 2 : 1。
 * 右パネルはハンバーガーボタンで完全に消え、左が画面幅一杯に広がる
 * （memo 5章の決定事項）。比率は tokens.css の --m3-panel-ratio: 33.3333% で持つ。
 */

import { useEffect } from 'react'
import { ControlPanel } from './panel/ControlPanel'
import { SimulatorView } from './scene/SimulatorView'
import { useAutoTheme } from './store/autoTheme'
import { useSimStore } from './store/simStore'
import { IconButton } from './ui/IconButton'
import { MenuIcon } from './ui/Icons'

export function App() {
  const panelOpen = useSimStore((s) => s.panelOpen)
  const togglePanel = useSimStore((s) => s.togglePanel)

  // 配色を日の出・日の入りに合わせ続ける。設定項目は無い
  useAutoTheme()

  // キーボードでもパネルを畳めるようにする（Vim ライクな操作を好むユーザー向け）
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      // 入力中は拾わない
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return
      if (e.key === 'm' || e.key === 'M') togglePanel()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [togglePanel])

  return (
    <div className="app-root">
      <main className="app-stage">
        <div className="app-brand">
          <span aria-hidden>🚗</span>
          <span>
            <span className="app-brand-title">DriveRL</span>
            <br />
            <span className="app-brand-sub">マルチエージェント強化学習 自動運転</span>
          </span>
        </div>

        {/* 条件レンダリングにせず、data-visible で出し入れする。
            外すと消えるときのアニメーションが一切かからず、
            パネルが 400ms かけて畳まれている途中でボタンだけ先に現れる */}
        <div
          className="app-hamburger"
          data-visible={panelOpen ? 'false' : 'true'}
          inert={panelOpen}
        >
          <IconButton
            variant="surface"
            large
            label="操作パネルを開く（M キー）"
            onClick={togglePanel}
          >
            <MenuIcon size={22} />
          </IconButton>
        </div>

        <SimulatorView />
      </main>

      <aside className="app-panel-slot" data-collapsed={panelOpen ? 'false' : 'true'}>
        <ControlPanel />
      </aside>
    </div>
  )
}
