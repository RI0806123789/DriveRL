/** 実用モードのスマホ画面（決定 9・10）。 */

import { useEffect, useRef, useState } from 'react'

import { send } from '../store/connection'
import { pedestrian } from '../store/pedestrian'
import { useSimStore } from '../store/simStore'
import { resetTaxiAutopilot } from '../store/taxiAutopilot'
import { isBoardablePhase, isRidingPhase } from '../types/protocol'
import type { Vec2 } from '../types/protocol'
import { PLATE_BG, PLATE_INK, plateLabel, withPlateNames } from '../scene/licensePlate'
import { formatEta } from '../scene/TaxiHud'
import { TaxiCameraView } from './TaxiCameraView'
import { TaxiMap } from './TaxiMap'
import type { PickTarget } from './TaxiMap'
import { Button } from '../ui/Button'
import { CameraIcon, CarIcon, MapIcon, TargetIcon, WarningIcon } from '../ui/Icons'

/** 時計の更新間隔 [ms]。分表示なので 15 秒で足りる */
const CLOCK_MS = 15000

/** 時計を続けて叩いたと見なす間隔 [ms] と、その回数 */
const TAP_GAP_MS = 600
const TAP_COUNT = 5

function nowLabel(): { hh: string; mm: string } {
  const d = new Date()
  return { hh: String(d.getHours()), mm: String(d.getMinutes()).padStart(2, '0') }
}

export function TaxiScreen() {
  const taxi = useSimStore((s) => s.taxi)
  const status = useSimStore((s) => s.status)
  const map = useSimStore((s) => s.map)
  const presets = useSimStore((s) => s.presets)
  const cameraOn = useSimStore((s) => s.taxiCameraOn)
  const setCameraOn = useSimStore((s) => s.setTaxiCameraOn)
  const autoOn = useSimStore((s) => s.taxiAutoOn)
  const setAutoOn = useSimStore((s) => s.setTaxiAutoOn)

  // 車はナンバープレートで指す（実車の配車アプリと同じ）
  const plate = plateLabel(taxi.vehicleId, status.presetId)

  const [clock, setClock] = useState(nowLabel)
  const [picking, setPicking] = useState<PickTarget>(null)
  const [draftDropoff, setDraftDropoff] = useState<Vec2 | null>(null)

  const taps = useRef({ count: 0, at: 0 })

  useEffect(() => {
    const timer = window.setInterval(() => setClock(nowLabel()), CLOCK_MS)
    return () => window.clearInterval(timer)
  }, [])

  const tapClock = () => {
    const now = performance.now()
    const t = taps.current
    t.count = now - t.at <= TAP_GAP_MS ? t.count + 1 : 1
    t.at = now
    if (t.count < TAP_COUNT) return
    t.count = 0
    resetTaxiAutopilot(now / 1000)
    setAutoOn(!autoOn)
  }

  // 配車が動き出したら、選びかけの地点は捨てる（サーバーが返す地点が正）
  useEffect(() => {
    if (taxi.phase === 'idle') {
      // 映す相手がいなくなるので畳む。アニメーションは閉じる向きで流れる
      setCameraOn(false)
      return
    }
    setPicking(null)
    setDraftDropoff(null)
  }, [taxi.phase, setCameraOn])

  const areaName = presets.find((p) => p.id === map?.presetId)?.name ?? map?.name ?? '—'
  const riding = isRidingPhase(taxi.phase)
  const waiting = isBoardablePhase(taxi.phase)
  const ready = draftDropoff !== null

  const handlePick = (point: Vec2) => {
    if (picking !== 'dropoff') return
    setDraftDropoff(point)
    setPicking(null)
  }

  const startPicking = () => {
    setDraftDropoff(null)
    setPicking('dropoff')
  }

  // ★ 乗車地点は**呼ぶ瞬間の現在地**。60fps で動くので state には置かず、
  //   ここで `pedestrian` から直に読む（`frameBuffer` と同じ作法）
  const callTaxi = () => {
    if (!draftDropoff) return
    if (!pedestrian.placed) return
    send({
      type: 'request_taxi',
      pickup: [pedestrian.x, pedestrian.y],
      dropoff: draftDropoff,
    })
  }

  let headline: string
  if (picking === 'dropoff') headline = '地図をタップして行き先を選ぶ'
  else if (taxi.phase === 'approaching') headline = `到着まで ${formatEta(taxi.etaSeconds)}`
  else if (taxi.phase === 'waiting') headline = '乗車地点で待っています'
  else if (taxi.phase === 'riding') headline = `目的地まで ${formatEta(taxi.etaSeconds)}`
  else if (taxi.phase === 'arrived') headline = '目的地に到着しました'
  else if (ready) headline = 'この行き先で呼べます'
  else headline = 'タクシーを呼ぶ'

  return (
    <div className="taxi-phone">
      <div className="taxi-phone-bezel">
        <div className="taxi-status-bar">
          <span className="taxi-clock" data-auto={autoOn} onPointerDown={tapClock}>
            {clock.hh}
            <span className="taxi-clock-colon">:</span>
            {clock.mm}
          </span>
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

        <TaxiMap picking={picking} onPick={handlePick} draftDropoff={draftDropoff}>
          <TaxiCameraView on={cameraOn} vehicleId={taxi.vehicleId} />
        </TaxiMap>

        <div className="taxi-phone-body">
          {/* ★ key は「段階」で切る。ETA を含めると毎秒アニメーションして落ち着かない */}
          <div
            className="taxi-headline"
            key={`${taxi.phase}:${picking ?? '-'}:${ready}`}
            data-alert={taxi.phase === 'waiting' || taxi.phase === 'arrived' ? 'true' : 'false'}
          >
            {headline}
          </div>

          <div className="taxi-meta">
            {taxi.phase !== 'idle' && (
              <>
                <span
                  className="taxi-plate"
                  style={{ background: PLATE_BG, color: PLATE_INK }}
                  title="この車のナンバープレート"
                >
                  <CarIcon size={12} /> {plate}
                </span>
                <span>残り {Math.round(taxi.remainingDistanceM)} m</span>
              </>
            )}
            {taxi.phase === 'idle' && picking === null && !ready && (
              <span>いまいる場所まで迎えに来ます。行き先だけ地図から選びます</span>
            )}
            {picking !== null && (
              <span>
                <TargetIcon size={12} /> 地図はドラッグで移動、ホイールで拡大できます
              </span>
            )}
          </div>

          {taxi.message && (
            <div className="taxi-message" key={taxi.message}>
              {withPlateNames(taxi.message, status.presetId)}
            </div>
          )}

          {!status.mapLoaded && (
            <div className="taxi-warning">
              <WarningIcon size={14} />
              エリアが読み込まれていません。開発モードの「マップ」タブから選んでください。
            </div>
          )}

          <div
            className="taxi-actions"
            key={`${riding}:${waiting}:${ready}:${picking ?? '-'}`}
          >
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
                  ここまで呼ぶ
                </Button>
                <Button variant="text" block onClick={startPicking}>
                  行き先を選び直す
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
                タクシーを呼ぶ
              </Button>
            )}
          </div>

          <button
            type="button"
            className="taxi-camera-toggle"
            data-on={cameraOn ? 'true' : 'false'}
            aria-pressed={cameraOn}
            disabled={taxi.phase === 'idle'}
            onClick={() => setCameraOn(!cameraOn)}
          >
            <span className="taxi-camera-toggle-icon">
              <CameraIcon size={15} />
            </span>
            <span className="taxi-camera-toggle-text">
              {cameraOn ? 'カメラを閉じる' : 'カメラ'}
            </span>
            <span className="taxi-camera-toggle-dot" data-on={cameraOn ? 'true' : 'false'} />
          </button>

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
