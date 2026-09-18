/** 実用モードのスマホ画面（決定 9・10）。 */

import { useEffect, useState } from 'react'

import { send } from '../store/connection'
import { useSimStore } from '../store/simStore'
import type { Vec2 } from '../types/protocol'
import { formatEta } from '../scene/TaxiHud'
import { TaxiMap } from './TaxiMap'
import type { PickTarget } from './TaxiMap'
import { Button } from '../ui/Button'
import { CarIcon, MapIcon, TargetIcon, WarningIcon } from '../ui/Icons'

/** 時計の更新間隔 [ms]。分表示なので 15 秒で足りる */
const CLOCK_MS = 15000

function nowLabel(): string {
  const d = new Date()
  return `${d.getHours()}:${String(d.getMinutes()).padStart(2, '0')}`
}

export function TaxiScreen() {
  const taxi = useSimStore((s) => s.taxi)
  const status = useSimStore((s) => s.status)
  const map = useSimStore((s) => s.map)
  const presets = useSimStore((s) => s.presets)

  const [clock, setClock] = useState(nowLabel)
  const [picking, setPicking] = useState<PickTarget>(null)
  const [draftPickup, setDraftPickup] = useState<Vec2 | null>(null)
  const [draftDropoff, setDraftDropoff] = useState<Vec2 | null>(null)

  useEffect(() => {
    const timer = window.setInterval(() => setClock(nowLabel()), CLOCK_MS)
    return () => window.clearInterval(timer)
  }, [])

  // 配車が動き出したら、選びかけの地点は捨てる（サーバーが返す地点が正）
  useEffect(() => {
    if (taxi.phase === 'idle') return
    setPicking(null)
    setDraftPickup(null)
    setDraftDropoff(null)
  }, [taxi.phase])

  const areaName = presets.find((p) => p.id === map?.presetId)?.name ?? map?.name ?? '—'
  const riding = taxi.phase === 'riding' || taxi.phase === 'arrived'
  const waiting = taxi.phase === 'approaching' || taxi.phase === 'waiting'
  const ready = draftPickup !== null && draftDropoff !== null

  const handlePick = (point: Vec2) => {
    if (picking === 'pickup') {
      setDraftPickup(point)
      setPicking('dropoff')
    } else if (picking === 'dropoff') {
      setDraftDropoff(point)
      setPicking(null)
    }
  }

  const startPicking = () => {
    setDraftPickup(null)
    setDraftDropoff(null)
    setPicking('pickup')
  }

  const callTaxi = () => {
    if (!draftPickup || !draftDropoff) return
    send({ type: 'request_taxi', pickup: draftPickup, dropoff: draftDropoff })
  }

  let headline: string
  if (picking === 'pickup') headline = '地図をタップして乗車地点を選ぶ'
  else if (picking === 'dropoff') headline = '次に降車地点を選ぶ'
  else if (taxi.phase === 'approaching') headline = `到着まで ${formatEta(taxi.etaSeconds)}`
  else if (taxi.phase === 'waiting') headline = '乗車地点で待っています'
  else if (taxi.phase === 'riding') headline = `目的地まで ${formatEta(taxi.etaSeconds)}`
  else if (taxi.phase === 'arrived') headline = '目的地に到着しました'
  else if (ready) headline = 'この経路で呼べます'
  else headline = 'タクシーを呼ぶ'

  return (
    <div className="taxi-phone">
      <div className="taxi-phone-bezel">
        <div className="taxi-status-bar">
          <span className="taxi-clock">{clock}</span>
          <span className="taxi-status-icons" aria-hidden>
            <SignalIcon />
            <BatteryIcon />
          </span>
        </div>

        <div className="taxi-app-head">
          <span className="taxi-app-title">DriveRL Taxi</span>
          <span className="taxi-app-area">
            <MapIcon size={12} />
            {areaName}
          </span>
        </div>

        <TaxiMap
          picking={picking}
          onPick={handlePick}
          draftPickup={draftPickup}
          draftDropoff={draftDropoff}
        />

        <div className="taxi-phone-body">
          <div className="taxi-headline">{headline}</div>

          <div className="taxi-meta">
            {taxi.phase !== 'idle' && (
              <>
                <span>
                  <CarIcon size={12} /> 車両 #{taxi.vehicleId}
                </span>
                <span>残り {Math.round(taxi.remainingDistanceM)} m</span>
              </>
            )}
            {taxi.phase === 'idle' && picking === null && !ready && (
              <span>乗車地点と降車地点を地図から選びます</span>
            )}
            {picking !== null && (
              <span>
                <TargetIcon size={12} /> 地図はドラッグで移動、ホイールで拡大できます
              </span>
            )}
          </div>

          {taxi.message && <div className="taxi-message">{taxi.message}</div>}

          {!status.mapLoaded && (
            <div className="taxi-warning">
              <WarningIcon size={14} />
              エリアが読み込まれていません。開発モードの「マップ」タブから選んでください。
            </div>
          )}

          <div className="taxi-actions">
            {riding ? (
              <Button variant="filled" block onClick={() => send({ type: 'alight_taxi' })}>
                降車する
              </Button>
            ) : waiting ? (
              <Button
                variant="outlined"
                block
                onClick={() => send({ type: 'cancel_taxi' })}
              >
                配車を取り消す
              </Button>
            ) : ready ? (
              <>
                <Button variant="filled" block onClick={callTaxi}>
                  この経路で呼ぶ
                </Button>
                <Button variant="text" block onClick={startPicking}>
                  選び直す
                </Button>
              </>
            ) : picking !== null ? (
              <Button variant="outlined" block onClick={() => setPicking(null)}>
                選択をやめる
              </Button>
            ) : (
              <Button
                variant="filled"
                block
                disabled={!status.mapLoaded}
                onClick={startPicking}
              >
                乗車する
              </Button>
            )}
          </div>

          {riding && (
            <div className="taxi-note">
              3D 画面で [Space] を押すと、その場で緊急停止して自動運転を終了します。
            </div>
          )}
        </div>

        <div className="taxi-home-bar" aria-hidden />
      </div>
    </div>
  )
}

function SignalIcon() {
  return (
    <svg width="14" height="10" viewBox="0 0 14 10" fill="currentColor">
      <rect x="0" y="7" width="2.4" height="3" rx="0.6" />
      <rect x="3.7" y="5" width="2.4" height="5" rx="0.6" />
      <rect x="7.4" y="2.6" width="2.4" height="7.4" rx="0.6" />
      <rect x="11.1" y="0" width="2.4" height="10" rx="0.6" />
    </svg>
  )
}

function BatteryIcon() {
  return (
    <svg width="20" height="10" viewBox="0 0 20 10" fill="none" stroke="currentColor">
      <rect x="0.6" y="0.6" width="15.8" height="8.8" rx="2.2" strokeWidth="1.2" />
      <rect x="2.2" y="2.2" width="10.6" height="5.6" rx="1.2" fill="currentColor" stroke="none" />
      <path d="M18 3.4v3.2" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  )
}
