/** 実用モードの画面表示（照準と操作ヒント）。**HTML 側のオーバーレイ。** */

import { useEffect, useState } from 'react'

import { frameBuffer } from '../store/frameBuffer'
import { pedestrian } from '../store/pedestrian'
import { useSimStore } from '../store/simStore'

/** ポーリング間隔 [ms]。pedestrian は 60fps で動くので、表示だけ間引いて読む */
const POLL_MS = 150

interface HudSample {
  locked: boolean
  aimed: number
  speedKph: number
}

function readSample(vehicleId: number): HudSample {
  let speedKph = 0
  const curr = frameBuffer.curr
  if (curr && vehicleId >= 0) {
    const v = curr.vehicles.find((item) => item.id === vehicleId)
    if (v) speedKph = Math.round(v.speed * 3.6)
  }
  return { locked: pedestrian.locked, aimed: pedestrian.aimed, speedKph }
}

function same(a: HudSample, b: HudSample): boolean {
  return a.locked === b.locked && a.aimed === b.aimed && a.speedKph === b.speedKph
}

/** 残り時間を「1 分 20 秒」の形にする */
export function formatEta(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return 'まもなく'
  const total = Math.round(seconds)
  if (total < 60) return `${total} 秒`
  return `${Math.floor(total / 60)} 分 ${total % 60} 秒`
}

export function TaxiHud() {
  const taxi = useSimStore((s) => s.taxi)
  const [sample, setSample] = useState<HudSample>({ locked: false, aimed: -1, speedKph: 0 })

  useEffect(() => {
    const timer = window.setInterval(() => {
      const next = readSample(taxi.vehicleId)
      setSample((prev) => (same(prev, next) ? prev : next))
    }, POLL_MS)
    return () => window.clearInterval(timer)
  }, [taxi.vehicleId])

  const riding = taxi.phase === 'riding' || taxi.phase === 'arrived'
  const canBoard = taxi.phase === 'waiting' && sample.aimed === taxi.vehicleId

  // ★ 乗れる／降りられることを最優先で出す。
  //   ポインタロックの案内を先に置くと、肝心の「いま押せるキー」が隠れる
  let hint: string
  if (canBoard) {
    hint = '[Enter] で乗車'
  } else if (taxi.phase === 'arrived') {
    hint = '目的地に到着しました。[Enter] で降車'
  } else if (!sample.locked && !riding) {
    hint = '3D 画面をクリックすると歩けます（W / A / S / D、Esc で解除）'
  } else if (taxi.phase === 'waiting') {
    hint = 'タクシーが待っています。近づいて照準を合わせ [Enter]'
  } else if (taxi.phase === 'approaching') {
    hint = `タクシーが迎えに来ています（あと ${formatEta(taxi.etaSeconds)}）`
  } else if (taxi.phase === 'riding') {
    hint = `目的地まで ${formatEta(taxi.etaSeconds)}　[Space] で緊急停止`
  } else {
    hint = 'スマホの「乗車する」から、地図で乗車地点と降車地点を選んでください'
  }

  return (
    <div className="taxi-hud">
      {!riding && (
        <div className="taxi-crosshair" data-aimed={canBoard ? 'true' : 'false'} aria-hidden>
          <span />
          <span />
        </div>
      )}

      {/* 速度計と操作の案内は同じ場所（中央下部）に積む */}
      <div className="taxi-bottom">
        {riding && (
          <div className="taxi-speed">
            <span className="taxi-speed-value">{sample.speedKph}</span>
            <span className="taxi-speed-unit">km/h</span>
          </div>
        )}

        <div
          className="taxi-hint"
          data-strong={canBoard || taxi.phase === 'arrived' ? 'true' : 'false'}
        >
          {hint}
        </div>
      </div>
    </div>
  )
}
