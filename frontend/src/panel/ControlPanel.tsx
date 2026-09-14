/**
 * 右側の操作パネル（memo 4章）。
 *
 * ・設定セクションはタブ方式で切り替える。
 * ・バックエンド未接続のときは操作できないようにし、その理由を明示する
 *   （押しても何も起きないのに操作できてしまう、という状態を避ける）。
 *
 * タブの中身はタブの並び順に合わせた向きから滑り込ませる（TABS の左隣なら左から）。
 * 4 つのタブはどれも縦に長いカードの列で見分けが付きにくく、切り替えが一瞬だと
 * 「切り替わったのか、同じものが再描画されただけなのか」が読めないため。
 */

import { useRef, useState } from 'react'
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

  // 直前のタブと比べて、右へ移ったのか左へ移ったのかを覚えておく。
  // 描画中に ref を書き換えているが、同じ tab で 2 度描画されても
  // prevTab.current === tab になって向きは変わらない（StrictMode でも安定）
  const prevTab = useRef<PanelTab>(tab)
  const paneDir = useRef<'next' | 'prev'>('next')
  if (prevTab.current !== tab) {
    paneDir.current =
      TAB_ORDER.indexOf(tab) > TAB_ORDER.indexOf(prevTab.current) ? 'next' : 'prev'
    prevTab.current = tab
  }

  // 閉じるアニメーションを見せてから消すエラー（消えた実感を残す）。
  // ここで消し損ねても simStore 側の一覧が正なので、表示が壊れることはない
  const [leaving, setLeaving] = useState<number | null>(null)
  const dismissWithExit = (id: number) => {
    if (prefersReducedMotion()) {
      dismissError(id)
      return
    }
    setLeaving(id)
    window.setTimeout(() => {
      setLeaving((curr) => (curr === id ? null : curr))
      dismissError(id)
    }, 180)
  }

  return (
    <div className="panel-wrap">
      <div className="panel">
        <header className="panel-header">
          {/* 元はステージ左上に浮かせていた（.app-brand）。
              3D を全画面で見るときに残っていても使い道が無いので、
              パネルと一緒に出入りする位置（「操作パネル」の真上）へ移した */}
          <div className="panel-brand">
            <span className="panel-brand-mark" aria-hidden>
              🚗
            </span>
            <span className="panel-brand-text">
              <span className="panel-brand-title">DriveRL</span>
              <span className="panel-brand-sub">マルチエージェント強化学習 自動運転</span>
            </span>
          </div>

          {/* ★ ここにあった「×（パネルを閉じる）」は外してある。
              そのため**畳む手段は M キーだけ**。ハンバーガーは畳んだ後にしか出ないので、
              閉じるボタンを戻すかハンバーガーを常時表示にしない限り、
              マウスだけではパネルを閉じられない */}
          <div className="panel-header-row">
            <span className="panel-title">操作パネル</span>
            <span className="m3-conn" data-state={connection}>
              {/* key で付け替えて、状態が変わった瞬間だけ入場アニメーションを流す。
                  外すと「接続済み」への復帰が文字の差し替えだけになって気づけない */}
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
              className={`m3-banner m3-banner--error${leaving === e.id ? ' is-leaving' : ''}`}
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

          {/* key を tab にして、切り替えのたびにカードの列を入場させ直す。
              元々タブごとにアンマウントしている（中身の状態は持ち越さない）ので
              挙動は変わらない */}
          <div key={tab} className="panel-pane" data-dir={paneDir.current}>
            {tab === 'simulation' && <SimulationTab />}
            {tab === 'map' && <MapTab />}
            {tab === 'learning' && <LearningTab />}
            {tab === 'model' && <ModelTab />}
            {tab === 'view' && <ViewTab />}
          </div>
        </div>
      </div>

      {/* 条件レンダリングにしない。接続できた瞬間にベールが消えるところを
          見せたいので（「復帰した」という手応えがここにしかない） */}
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
