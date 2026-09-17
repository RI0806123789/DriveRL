/** 右側の操作パネル（memo 4章）。 */

import { useEffect, useRef, useState } from 'react'
import { useSimStore } from '../store/simStore'
import type { PanelTab } from '../store/simStore'
import { prefersReducedMotion } from '../ui/motion'
import { LearningTab } from './LearningTab'
import { MapTab } from './MapTab'
import { ModelTab } from './ModelTab'
import { SimulationTab } from './SimulationTab'
import { ViewTab } from './ViewTab'
import { Tabs } from '../ui/Tabs'
import type { TabItem } from '../ui/Tabs'
import { IconButton } from '../ui/IconButton'
import {
  BrainIcon,
  CameraIcon,
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
  { id: 'model', label: 'モデル作成', icon: <CameraIcon size={16} /> },
  { id: 'view', label: '表示', icon: <EyeIcon size={16} /> },
]

/** 滑り込む向きを決めるための並び順。TABS と同じ順にしておくこと */
const TAB_ORDER: PanelTab[] = TABS.map((t) => t.id)

const CONNECTION_LABEL: Record<string, string> = {
  connecting: '接続中…',
  open: '接続済み',
  closed: '未接続',
}

export function ControlPanel() {
  const tab = useSimStore((s) => s.tab)
  const setTab = useSimStore((s) => s.setTab)
  const connection = useSimStore((s) => s.connection)
  const usingMock = useSimStore((s) => s.usingMock)
  const protocolMismatch = useSimStore((s) => s.protocolMismatch)
  const errors = useSimStore((s) => s.errors)
  const dismissError = useSimStore((s) => s.dismissError)

  const disabled = connection !== 'open'

  const prevTab = useRef<PanelTab>(tab)
  const paneDir = useRef<'next' | 'prev'>('next')
  if (prevTab.current !== tab) {
    paneDir.current =
      TAB_ORDER.indexOf(tab) > TAB_ORDER.indexOf(prevTab.current) ? 'next' : 'prev'
    prevTab.current = tab
  }

  // ★ 退場中のバナーは**同時に複数ありうる**ので、id ごとに持つこと。
  //    1 つの ref で使い回すと、180ms 以内に 2 つ閉じたときに先の削除が
  //    キャンセルされ、1 つ目が is-leaving を外されたまま画面に残る。
  const [leaving, setLeaving] = useState<ReadonlySet<number>>(() => new Set())
  const leaveTimers = useRef(new Map<number, number>())
  useEffect(
    () => () => {
      for (const timer of leaveTimers.current.values()) window.clearTimeout(timer)
      leaveTimers.current.clear()
    },
    [],
  )
  const dismissWithExit = (id: number) => {
    if (prefersReducedMotion()) {
      dismissError(id)
      return
    }
    if (leaveTimers.current.has(id)) return
    setLeaving((curr) => new Set(curr).add(id))
    leaveTimers.current.set(
      id,
      window.setTimeout(() => {
        leaveTimers.current.delete(id)
        setLeaving((curr) => {
          if (!curr.has(id)) return curr
          const next = new Set(curr)
          next.delete(id)
          return next
        })
        dismissError(id)
      }, 180),
    )
  }

  return (
    <div className="panel-wrap">
      <div className="panel">
        <header className="panel-header">
          <div className="panel-brand">
            <span className="panel-brand-mark" aria-hidden>
              🚗
            </span>
            <span className="panel-brand-text">
              <span className="panel-brand-title">DriveRL</span>
              <span className="panel-brand-sub">マルチエージェント強化学習 自動運転</span>
            </span>
          </div>

          <div className="panel-header-row">
            <span className="panel-title">操作パネル</span>
            <span className="m3-conn" data-state={connection}>
              <span key={usingMock ? 'mock' : connection} className="m3-conn-dot" />
              {usingMock ? 'モック接続' : CONNECTION_LABEL[connection]}
            </span>
          </div>
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
            <div
              key={e.id}
              className={`m3-banner m3-banner--error${leaving.has(e.id) ? ' is-leaving' : ''}`}
            >
              <span className="m3-banner-icon">
                <WarningIcon size={16} />
              </span>
              <span className="m3-grow">
                <span className="m3-banner-title">{e.code}</span>
                {e.message}
              </span>
              <IconButton label="閉じる" onClick={() => dismissWithExit(e.id)}>
                <CloseIcon size={16} />
              </IconButton>
            </div>
          ))}

          <div key={tab} className="panel-pane" data-dir={paneDir.current}>
            {tab === 'simulation' && <SimulationTab />}
            {tab === 'map' && <MapTab />}
            {tab === 'learning' && <LearningTab />}
            {tab === 'model' && <ModelTab />}
            {tab === 'view' && <ViewTab />}
          </div>
        </div>
      </div>

      <div
        className="panel-disabled-veil"
        data-visible={disabled ? 'true' : 'false'}
        inert={!disabled}
      >
        <PlugIcon size={28} />
        <div className="panel-veil-title">バックエンドに接続していません</div>
        <div className="m3-note panel-veil-note">
          <code className="m3-mono">backend/run.py</code> を起動してください。
          <br />
          接続できるようになると自動で復帰します。
        </div>
      </div>
    </div>
  )
}
