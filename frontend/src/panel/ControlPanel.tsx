/**
 * 右側の操作パネル（memo 4章）。
 *
 * ・設定セクションはタブ方式で切り替える。
 * ・バックエンド未接続のときは操作できないようにし、その理由を明示する
 *   （押しても何も起きないのに操作できてしまう、という状態を避ける）。
 */

import { useSimStore } from '../store/simStore'
import type { PanelTab } from '../store/simStore'
import { LearningTab } from './LearningTab'
import { MapTab } from './MapTab'
import { SimulationTab } from './SimulationTab'
import { ViewTab } from './ViewTab'
import { Tabs } from '../ui/Tabs'
import type { TabItem } from '../ui/Tabs'
import { IconButton } from '../ui/IconButton'
import {
  BrainIcon,
  CloseIcon,
  EyeIcon,
  MapIcon,
  PlayIcon,
  PlugIcon,
  WarningIcon,
} from '../ui/Icons'

const TABS: TabItem<PanelTab>[] = [
  { id: 'simulation', label: 'シミュレーション', icon: <PlayIcon size={16} /> },
  { id: 'map', label: 'マップ', icon: <MapIcon size={16} /> },
  { id: 'learning', label: '学習', icon: <BrainIcon size={16} /> },
  { id: 'view', label: '表示', icon: <EyeIcon size={16} /> },
]

const CONNECTION_LABEL: Record<string, string> = {
  connecting: '接続中…',
  open: '接続済み',
  closed: '未接続',
}

export function ControlPanel() {
  const tab = useSimStore((s) => s.tab)
  const setTab = useSimStore((s) => s.setTab)
  const setPanelOpen = useSimStore((s) => s.setPanelOpen)
  const connection = useSimStore((s) => s.connection)
  const usingMock = useSimStore((s) => s.usingMock)
  const protocolMismatch = useSimStore((s) => s.protocolMismatch)
  const errors = useSimStore((s) => s.errors)
  const dismissError = useSimStore((s) => s.dismissError)

  const disabled = connection !== 'open'

  return (
    <div className="panel-wrap">
      <div className="panel">
        <header className="panel-header">
          <span className="panel-title">操作パネル</span>
          <span className="m3-conn" data-state={connection}>
            <span className="m3-conn-dot" />
            {usingMock ? 'モック接続' : CONNECTION_LABEL[connection]}
          </span>
          <IconButton label="パネルを閉じる" onClick={() => setPanelOpen(false)}>
            <CloseIcon size={20} />
          </IconButton>
        </header>

        <div className="panel-tabs">
          <Tabs items={TABS} value={tab} onChange={setTab} label="設定セクション" />
        </div>

        <div className="panel-body m3-scroll">
          {usingMock && (
            <div className="m3-banner m3-banner--warning">
              <span className="m3-banner-icon">
                <WarningIcon size={16} />
              </span>
              <span>
                <span className="m3-banner-title">開発用モックに接続しています</span>
                実際の強化学習は動いていません。URL から <code className="m3-mono">?mock=1</code> を
                外すと本物のバックエンドに繋ぎます。
              </span>
            </div>
          )}

          {protocolMismatch && (
            <div className="m3-banner m3-banner--warning">
              <span className="m3-banner-icon">
                <WarningIcon size={16} />
              </span>
              <span>
                <span className="m3-banner-title">プロトコルのバージョンが一致しません</span>
                バックエンドとフロントエンドのどちらかが古い可能性があります。
              </span>
            </div>
          )}

          {errors.map((e) => (
            <div key={e.id} className="m3-banner m3-banner--error">
              <span className="m3-banner-icon">
                <WarningIcon size={16} />
              </span>
              <span className="m3-grow">
                <span className="m3-banner-title">{e.code}</span>
                {e.message}
              </span>
              <IconButton label="閉じる" onClick={() => dismissError(e.id)}>
                <CloseIcon size={16} />
              </IconButton>
            </div>
          ))}

          {tab === 'simulation' && <SimulationTab />}
          {tab === 'map' && <MapTab />}
          {tab === 'learning' && <LearningTab />}
          {tab === 'view' && <ViewTab />}
        </div>
      </div>

      {disabled && (
        <div className="panel-disabled-veil">
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 12,
              textAlign: 'center',
              padding: 24,
            }}
          >
            <PlugIcon size={28} />
            <div style={{ fontSize: 15, fontWeight: 600 }}>バックエンドに接続していません</div>
            <div className="m3-note" style={{ maxWidth: 300 }}>
              <code className="m3-mono">backend/run.py</code> を起動してください。
              <br />
              接続できるようになると自動で復帰します。
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
