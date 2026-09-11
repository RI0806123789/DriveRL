/**
 * 「表示」タブ。カメラワークと描画の重さの調整。
 *
 * 建物は数千件あるので、フレームレートが出ないときはまず影を切るのが効く。
 * その旨をユーザーに伝えるため、トグルに説明文を添えている。
 */

import { useEffect, useState } from 'react'
import { frameBuffer } from '../store/frameBuffer'
import { useSimStore } from '../store/simStore'
import { vehicleColor } from '../scene/vehicleColors'
import { Card } from '../ui/Card'
import { Chip } from '../ui/Chip'
import { Select } from '../ui/Select'
import { Switch } from '../ui/Switch'
import { CameraIcon, EyeIcon } from '../ui/Icons'

export function ViewTab() {
  const cameraMode = useSimStore((s) => s.cameraMode)
  const setCameraMode = useSimStore((s) => s.setCameraMode)
  const followTarget = useSimStore((s) => s.followTarget)
  const setFollowTarget = useSimStore((s) => s.setFollowTarget)
  const maxVehicles = useSimStore((s) => s.config.maxVehicles)
  const view = useSimStore((s) => s.view)
  const toggleView = useSimStore((s) => s.toggleView)
  const map = useSimStore((s) => s.map)

  // 追従対象の候補は「いまアクティブなスロット」だけにする。
  // パネルが畳まれている間は止め（F-10）、集合が変わったときだけ state を
  // 差し替える（F-11）。毎回新しい配列を作ると 64 個の Chip が 2Hz で
  // 無条件に再レンダリングされる。
  const panelOpen = useSimStore((s) => s.panelOpen)
  const [activeIds, setActiveIds] = useState<number[]>([])
  useEffect(() => {
    if (!panelOpen) return
    const timer = window.setInterval(() => {
      const curr = frameBuffer.curr
      const next = curr ? curr.vehicles.filter((v) => v.active).map((v) => v.id) : []
      setActiveIds((prev) =>
        prev.length === next.length && prev.every((v, i) => v === next[i]) ? prev : next,
      )
    }, 500)
    return () => window.clearInterval(timer)
  }, [panelOpen])

  const followOptions =
    activeIds.length > 0
      ? activeIds.map((id) => ({ value: String(id), label: `車両 #${id}` }))
      : Array.from({ length: maxVehicles }, (_, i) => ({
          value: String(i),
          label: `車両 #${i}（停止中）`,
        }))

  return (
    <>
      <Card title="カメラ" icon={<CameraIcon size={16} />}>
        <div className="m3-row m3-row--wrap">
          <Chip selected={cameraMode === 'orbit'} onClick={() => setCameraMode('orbit')}>
            俯瞰（自由視点）
          </Chip>
          <Chip selected={cameraMode === 'follow'} onClick={() => setCameraMode('follow')}>
            追従（後方上空）
          </Chip>
          <Chip selected={cameraMode === 'driver'} onClick={() => setCameraMode('driver')}>
            運転席
          </Chip>
        </div>

        {cameraMode === 'orbit' ? (
          <div className="m3-note">
            ドラッグで回転、ホイールでズーム、右ドラッグで平行移動できます。
          </div>
        ) : (
          <>
            <Select
              label="追従する車両"
              value={String(followTarget)}
              options={followOptions}
              onChange={(v) => setFollowTarget(Number(v))}
            />
            <div className="m3-row m3-row--wrap">
              {Array.from({ length: maxVehicles }, (_, i) => i).map((id) => (
                <Chip
                  key={id}
                  small
                  selected={followTarget === id}
                  disabled={activeIds.length > 0 && !activeIds.includes(id)}
                  onClick={() => setFollowTarget(id)}
                  icon={
                    <span
                      className="m3-vehicle-swatch"
                      style={{ background: vehicleColor(id) }}
                    />
                  }
                >
                  #{id}
                </Chip>
              ))}
            </div>
            <div className="m3-note">
              {cameraMode === 'driver'
                ? '右ハンドル（日本仕様）の運転席から見た視点です。進路は太い矢印で示されます。自車の車体は視界を塞ぐため非表示になります。'
                : '車両の後方上空から追いかけます。'}
              <br />
              追従中は手動のカメラ操作を受け付けません。「シミュレーション」タブの
              車両一覧をクリックしても対象を切り替えられます。
            </div>
          </>
        )}
      </Card>

      <Card title="表示するもの" icon={<EyeIcon size={16} />}>
        <Switch
          label="建物"
          description={
            map ? `${map.buildings.length.toLocaleString()} 件を 1 メッシュに統合して描画` : undefined
          }
          checked={view.buildings}
          onChange={() => toggleView('buildings')}
        />
        <Switch
          label="道路"
          description={map ? `${map.edges.length.toLocaleString()} セグメント` : undefined}
          checked={view.roads}
          onChange={() => toggleView('roads')}
        />
        <Switch
          label="道路標示"
          description="中央線・車線境界線・車道外側線・停止線・横断歩道（日本の規格寸法）"
          checked={view.markings}
          onChange={() => toggleView('markings')}
        />
        <Switch
          label="信号機"
          description={
            map?.signals?.length
              ? `${map.signals.length} 基。青25秒→黄3秒→全赤2秒で交互に変わります`
              : '車両用の横型3灯式（運転者から見て左から青・黄・赤）'
          }
          checked={view.signals}
          onChange={() => toggleView('signals')}
        />
        <Switch
          label="最高速度標識"
          description={
            map?.signs?.length
              ? `${map.signs.length} 基。規制速度が変わる進入口に立っています`
              : '規制標識「最高速度」（白地の円に赤縁・直径 60cm、下端は路面から 1.8m）'
          }
          checked={view.showSigns}
          onChange={() => toggleView('showSigns')}
        />
        <Switch
          label="経路線"
          description="各車両が目的地まで辿る予定の経路"
          checked={view.routes}
          onChange={() => toggleView('routes')}
        />
        <Switch
          label="目的地マーカー"
          checked={view.goals}
          onChange={() => toggleView('goals')}
        />
        <Switch
          label="影"
          description="重いと感じたら最初にこれを切ってください"
          checked={view.shadows}
          onChange={() => toggleView('shadows')}
        />
        <Switch label="グリッド" checked={view.grid} onChange={() => toggleView('grid')} />
        <Switch
          label="認識結果（バウンディングボックス）"
          description="運転席・追従カメラでのみ表示。PPO が観測として受け取っている検出結果と同じものです"
          checked={view.detections}
          onChange={() => toggleView('detections')}
        />
      </Card>

    </>
  )
}
