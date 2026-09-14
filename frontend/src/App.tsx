/**
 * 全体レイアウト（memo 4章）。
 *
 * 左（シミュレーター）: 右（操作パネル） = 2 : 1。
 * 右パネルはハンバーガーボタンで完全に消える（memo 5章の決定事項）。
 * 幅は tokens.css の --m3-panel-ratio: 33.3333% で持つ。
 *
 * ★ 並べているのは**見た目だけ**。ステージは常に画面全面で、パネルは
 *   その上に重なっている（パネル周囲の余白に 3D を透かせるため）。
 *   だからパネルを開いても 3D キャンバスは狭くならず、
 *   俯瞰時のマップは**パネルの裏を含めた画面全体の中心**に入る。
 *
 * パネルを閉じたときは **3D だけにする**。ブランド名（DriveRL）は
 * ControlPanel のヘッダへ、走行状況の HUD は `panelOpen` 連動に移してあり、
 * 画面に残るのはハンバーガーボタンだけになる。
 */

import { useEffect } from 'react'
import { ControlPanel } from './panel/ControlPanel'
import { SimulatorView } from './scene/SimulatorView'
import { StageHud } from './scene/StageHud'
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

      {/* ★ 走行状況は 3D の上に浮かせず、**パネルと同じ列の一番下**に積む。
          カードと幅を揃えるためにチップは折り返すので、行数は画面幅と
          チップの枚数で変わる。flex の並びに置いておけばカード側が縮むので、
          「何行分場所を空けておくか」を CSS で見積もる必要がない（重なりようがない）。
          パネルを畳むときはスロットごと右へ抜けるので、動きも自動的に揃う */}
      <aside className="app-panel-slot" data-collapsed={panelOpen ? 'false' : 'true'}>
        <ControlPanel />
        <StageHud />
      </aside>
    </div>
  )
}
