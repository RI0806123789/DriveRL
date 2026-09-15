/** 画面右下の走行状況（HUD）。 */

import { useEffect, useRef, useState } from 'react'
import { currentCameraNotice, sceneStats } from './sceneStats'
import { frameBuffer } from '../store/frameBuffer'
import { useSimStore } from '../store/simStore'

interface HudState {
  tick: number
  simTime: number
  active: number
  obstacles: number
  hz: number
  drawCalls: number
  notice: string
}

const EMPTY_HUD: HudState = {
  tick: 0,
  simTime: 0,
  active: 0,
  obstacles: 0,
  hz: 0,
  drawCalls: 0,
  notice: '',
}

/** 要求倍速に対してこの割合を下回ったら「追いつけていない」と見なす */
const SPEED_SHORTFALL_RATIO = 0.9

/** 表示に使う桁だけを並べた署名。これが同じなら再レンダリングしても見た目は変わらない */
function hudSignature(h: HudState): string {
  return [
    h.tick,
    h.simTime.toFixed(1),
    h.active,
    h.obstacles,
    h.hz.toFixed(1),
    h.drawCalls,
    h.notice,
  ].join('|')
}

export function StageHud() {
  const mapLoaded = useSimStore((s) => s.map !== null)
  const panelOpen = useSimStore((s) => s.panelOpen)
  const renderPaused = useSimStore((s) => s.status.renderPaused)
  const stepsPerSec = useSimStore((s) => s.latestMetrics?.stepsPerSec ?? 0)
  const requestedSpeed = useSimStore((s) => s.params.simSpeed)
  const simHz = useSimStore((s) => s.config.simHz)
  const actualSpeed = simHz > 0 ? stepsPerSec / simHz : 0
  const keepingUp =
    requestedSpeed <= 0 || actualSpeed >= requestedSpeed * SPEED_SHORTFALL_RATIO
  const [hud, setHud] = useState<HudState>(EMPTY_HUD)
  const lastSample = useRef({ received: 0, at: performance.now() })
  const lastSignature = useRef(hudSignature(EMPTY_HUD))

  const visible = mapLoaded && panelOpen

  useEffect(() => {
    if (!visible) return
    lastSample.current = { received: frameBuffer.received, at: performance.now() }

    const timer = window.setInterval(() => {
      const curr = frameBuffer.curr
      const now = performance.now()
      const dt = (now - lastSample.current.at) / 1000
      const hz = dt > 0 ? (frameBuffer.received - lastSample.current.received) / dt : 0
      lastSample.current = { received: frameBuffer.received, at: now }

      const next: HudState = {
        tick: curr?.tick ?? 0,
        simTime: curr?.simTime ?? 0,
        active: curr ? curr.vehicles.reduce((n, v) => n + (v.active ? 1 : 0), 0) : 0,
        obstacles: frameBuffer.obstacles.length,
        hz,
        drawCalls: sceneStats.drawCalls,
        notice: currentCameraNotice(),
      }
      const sig = hudSignature(next)
      if (sig === lastSignature.current) return
      lastSignature.current = sig
      setHud(next)
    }, 250)
    return () => window.clearInterval(timer)
  }, [visible])

  if (!mapLoaded) return null

  return (
    <div className="app-hud" data-visible={visible ? 'true' : 'false'} inert={!visible}>
      <div className="app-hud-item">
        走行中<span className="app-hud-value">{hud.active}</span>台
      </div>
      <div className="app-hud-item">
        経過<span className="app-hud-value">{hud.simTime.toFixed(1)}</span>秒
      </div>
      <div className="app-hud-item">
        ステップ<span className="app-hud-value">{hud.tick.toLocaleString()}</span>
      </div>
      <div className="app-hud-item">
        障害物<span className="app-hud-value">{hud.obstacles}</span>
      </div>
      <div className="app-hud-item">
        受信<span className="app-hud-value">{hud.hz.toFixed(1)}</span>Hz
      </div>
      <div
        className="app-hud-item"
        title="1 フレームあたりのドローコール数（three の gl.info.render.calls）。台数を増やしたときの描画負荷の目安"
      >
        描画<span className="app-hud-value">{hud.drawCalls}</span>call
      </div>
      <div
        className="app-hud-item"
        style={keepingUp ? undefined : { color: 'var(--m3-warning)' }}
        title={
          keepingUp
            ? '要求した倍速で進んでいます'
            : `要求 ${requestedSpeed.toFixed(2)} 倍に対して計算が間に合っていません。`
              + '車両数か倍速を下げると追いつきます（学習は止まりません）'
        }
      >
        実効<span className="app-hud-value">{actualSpeed.toFixed(2)}</span>倍
        {!keepingUp && ` / 要求 ${requestedSpeed.toFixed(2)} 倍`}
      </div>
      {hud.notice && (
        <div className="app-hud-item" style={{ color: 'var(--m3-warning)' }}>
          {hud.notice}
        </div>
      )}
      {renderPaused && (
        <div className="app-hud-item" style={{ color: 'var(--m3-warning)' }}>
          描画停止中（学習は継続）
        </div>
      )}
    </div>
  )
}
