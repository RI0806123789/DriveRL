/** 全体レイアウト（memo 4章）。 */

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

  useAutoTheme()

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey || e.isComposing) return
      const target = e.target as HTMLElement | null
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return
      if (e.key === 'm' || e.key === 'M') togglePanel()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [togglePanel])

  return (
    <div className="app-root">
      <main className="app-stage">
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
        <StageHud />
      </aside>
    </div>
  )
}
