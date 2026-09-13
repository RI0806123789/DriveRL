/**
 * 「マップ」タブ。
 *
 * memo 5章の「複数の固定エリアを事前プリセット」に対応する選択 UI。
 * OSM の取得は Overpass API 経由で 10〜60 秒かかるため、
 * 「なぜ待たされているのか」を必ずユーザーに伝えること。
 */

import { send } from '../store/connection'
import { markMapLoading, useSimStore } from '../store/simStore'
import { Card } from '../ui/Card'
import { Collapse } from '../ui/Collapse'
import { startRipple } from '../ui/motion'
import { ValueFlash } from '../ui/ValueFlash'
import { InfoIcon, MapIcon, WarningIcon } from '../ui/Icons'

export function MapTab() {
  const presets = useSimStore((s) => s.presets)
  const map = useSimStore((s) => s.map)
  const status = useSimStore((s) => s.status)
  const pendingPresetId = useSimStore((s) => s.pendingPresetId)
  const connection = useSimStore((s) => s.connection)

  const loading = status.state === 'loading_map'
  const activePresetId = pendingPresetId ?? status.presetId ?? map?.presetId ?? null

  const handleSelect = (presetId: string) => {
    if (loading) return
    if (send({ type: 'load_map', presetId })) markMapLoading(presetId)
  }

  return (
    <>
      <Card title="エリアを選ぶ" icon={<MapIcon size={16} />}>
        <div className="m3-note">
          事前に用意された大都市中心部から選びます。選んだ瞬間に読み込みが始まり、
          完了すると強化学習が走り出します。
        </div>

        <div className="m3-col">
          {presets.length === 0 ? (
            <div className="m3-note">
              プリセットを取得できていません。バックエンドの起動状態を確認してください。
            </div>
          ) : (
            presets.map((p) => (
              <button
                key={p.id}
                type="button"
                className="m3-preset m3-ripple"
                data-selected={activePresetId === p.id ? 'true' : 'false'}
                data-loading={loading && activePresetId === p.id ? 'true' : 'false'}
                disabled={loading || connection !== 'open'}
                onPointerDown={startRipple}
                onClick={() => handleSelect(p.id)}
              >
                <span className="m3-preset-name">{p.name}</span>
                <span className="m3-preset-desc">{p.description}</span>
                <span className="m3-preset-meta">
                  {p.centerLat.toFixed(4)}, {p.centerLon.toFixed(4)} / 半径 {p.radiusM} m
                </span>
              </button>
            ))
          )}
        </div>

        <Collapse open={loading}>
          <div className="m3-col">
            <div className="m3-progress" />
            <div className="m3-banner m3-banner--info">
              <span className="m3-banner-icon">
                <InfoIcon size={16} />
              </span>
              <span>
                <span className="m3-banner-title">地図データを取得しています</span>
                OpenStreetMap の Overpass API から道路網と建物を取り寄せています。
                初回は 10〜60 秒かかります。二度目以降はディスクキャッシュから即座に読み込まれます。
              </span>
            </div>
          </div>
        </Collapse>

        <Collapse open={status.state === 'error'}>
          <div className="m3-banner m3-banner--error">
            <span className="m3-banner-icon">
              <WarningIcon size={16} />
            </span>
            <span>
              <span className="m3-banner-title">読み込みに失敗しました</span>
              {status.message ?? 'Overpass API が混雑している可能性があります。少し待って再試行してください。'}
            </span>
          </div>
        </Collapse>
      </Card>

      {map && (
        <Card title="読み込み済みマップ" icon={<MapIcon size={16} />} variant="outlined">
          <div className="m3-statgrid">
            <div className="m3-stat">
              <span className="m3-stat-label">エリア</span>
              <ValueFlash style={{ fontSize: 13 }} className="m3-stat-value">
                {map.name}
              </ValueFlash>
            </div>
            <div className="m3-stat">
              <span className="m3-stat-label">交差点ノード</span>
              <ValueFlash className="m3-stat-value">{map.nodes.length.toLocaleString()}</ValueFlash>
            </div>
            <div className="m3-stat">
              <span className="m3-stat-label">道路セグメント</span>
              <ValueFlash className="m3-stat-value">{map.edges.length.toLocaleString()}</ValueFlash>
            </div>
            <div className="m3-stat">
              <span className="m3-stat-label">建物</span>
              <ValueFlash className="m3-stat-value">{map.buildings.length.toLocaleString()}</ValueFlash>
            </div>
            <div className="m3-stat">
              <span className="m3-stat-label">東西の広さ</span>
              <ValueFlash className="m3-stat-value">{Math.round(map.bounds.maxX - map.bounds.minX)} m</ValueFlash>
            </div>
            <div className="m3-stat">
              <span className="m3-stat-label">南北の広さ</span>
              <ValueFlash className="m3-stat-value">{Math.round(map.bounds.maxY - map.bounds.minY)} m</ValueFlash>
            </div>
          </div>
          <div className="m3-note">
            建物の高さは OSM の height / building:levels から取り、情報が無いものは
            3 階相当（約 9 m）を当てています。実際には多くの建物が既定値です。
          </div>
        </Card>
      )}
    </>
  )
}
