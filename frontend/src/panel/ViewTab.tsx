/** 「表示」タブ。カメラワークと描画の重さの調整。 */

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
  const appMode = useSimStore((s) => s.mode)

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
            {cameraMode === 'driver' && appMode === 'dev' && (
              <Switch
                label="前後左右の 4 分割で表示"
                description="左上が前方（運転席）、右上が後方（リアガラス上端）、左下が左側方・右下が右側方（B ピラー上端）のカメラです。各カメラの検出枠を重ね、安全ギミックが止めている根拠の枠は赤、注意の枠は黄で示します"
                checked={view.quad}
                onChange={() => toggleView('quad')}
              />
            )}
            <div className="m3-note">
              {cameraMode === 'driver'
                ? '右ハンドル（日本仕様）の運転席から見た視点です。進路は太い矢印で示されます。ダッシュボード・メーター・ハンドルなどの内装も描きます。後退しているときは後方に予測ガイド線（赤 1m・黄 2m・緑 3m の目盛り）が出ます。'
                : '車両の後方上空から追いかけます。後退しているときは後方に予測ガイド線が出ます。'}
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
              ? `${map.signals.length} 基。60 秒を現示の数で分け合い、青→黄3秒→全赤2秒で順に変わります（2 現示なら青25秒）`
              : '車両用の横型3灯式（運転者から見て左から青・黄・赤）'
          }
          checked={view.signals}
          onChange={() => toggleView('signals')}
        />
        <Switch
          label="交通標識"
          description={
            map?.signs?.length
              ? `${map.signs.length} 基。速度・一時停止・横断歩道・通行方向・駐車や停車の禁止を表示します`
              : '最高速度・一時停止・横断歩道・一方通行・指定方向・駐車禁止・駐停車禁止'
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
          description="運転席カメラでのみ表示（4 分割のときは前後左右の各カメラ）。擬似カメラは運転席の位置・向きで描いているため、追従カメラ（車体後方15m）に重ねると対象物の位置が合いません。PPO が観測として受け取っている検出結果と同じものです"
          checked={view.detections}
          onChange={() => toggleView('detections')}
        />
        <Switch
          label="カメラの視野コーン"
          description="追従中の車の前後左右 4 台のカメラが見えている範囲（半透明の緑・半径 30m）。走行可能距離と検出した車両の陰で切り取った、PPO の観測（末尾 8 次元）と同じものです。開発モードの追従・運転席カメラで表示します"
          checked={view.cameraFrustums}
          onChange={() => toggleView('cameraFrustums')}
        />
        <Switch
          label="死角シャドウ"
          description="カメラだけで推定した死角。赤は検出した車両の陰、暗い紫は建物の陰や霧で視程の外になった所です。見通しの悪い交差点では、交差道路の側の見通しが開けるまで車頭をゆっくり出します（介入「顔出し」）"
          checked={view.occlusionShadows}
          onChange={() => toggleView('occlusionShadows')}
        />
      </Card>

    </>
  )
}
