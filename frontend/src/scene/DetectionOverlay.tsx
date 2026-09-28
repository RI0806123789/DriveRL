/** 認識結果（バウンディングボックス）のオーバーレイ。運転席の 4 分割表示ではペインごとに重ねる。 */

import { useEffect, useRef, useState, type RefObject } from 'react'
import { send } from '../store/connection'
import { frameBuffer } from '../store/frameBuffer'
import { isQuadView, useSimStore } from '../store/simStore'
import {
  DET_LANE,
  type Detection,
  type FrameMessage,
  type SurroundDetections,
  type VehicleState,
} from '../types/protocol'
import { assistSummary, paneAssist } from './assistLabels'
import { detectionFrameColor, detectionLabel } from './detectionLabels'
import { projectBox } from './detectionProjection'
import { PANE_LABELS, QUAD_PANES, paneFov, quadPaneRect, type QuadPane } from './quadLayout'

const EMPTY_DETECTIONS: Detection[] = []
const EMPTY_SURROUND: SurroundDetections = {}

/** ラベルを枠の上ではなく内側へ回り込ませるしきい値（正規化 y0）。 */
const LABEL_FLIP_THRESHOLD = 0.08
/** 周囲カメラの検出を載せてもらう頼みを送り直す間隔 [ms]（サーバーは 2.5 秒で切る） */
const WATCH_INTERVAL_MS = 1000

interface Tracked {
  front: Detection[]
  surround: SurroundDetections
  vehicle: VehicleState | undefined
}

const EMPTY_TRACKED: Tracked = { front: EMPTY_DETECTIONS, surround: EMPTY_SURROUND, vehicle: undefined }

/** 追従対象の車両について、いま表示している frame の検出結果と車両の状態を返すフック */
function useTracked(followTarget: number, active: boolean): Tracked {
  const [tracked, setTracked] = useState<Tracked>(EMPTY_TRACKED)
  const lastDisplayed = useRef(-1)

  useEffect(() => {
    if (!active) {
      setTracked(EMPTY_TRACKED)
      return
    }
    lastDisplayed.current = -1
    let raf = 0
    const tick = () => {
      if (frameBuffer.displayed !== lastDisplayed.current) {
        lastDisplayed.current = frameBuffer.displayed
        const curr: FrameMessage | null = frameBuffer.curr
        const key = String(followTarget)
        setTracked({
          front: curr?.detections?.[key] ?? EMPTY_DETECTIONS,
          surround: curr?.surround?.[key] ?? EMPTY_SURROUND,
          vehicle: curr?.vehicles.find((v) => v.id === followTarget),
        })
      }
      raf = requestAnimationFrame(tick)
    }
    tick()
    return () => cancelAnimationFrame(raf)
  }, [active, followTarget])

  return tracked
}

/** 4 分割表示の間だけ、その車の周囲カメラの検出を frame に載せてもらう */
function useWatchSurround(active: boolean, vehicleId: number): void {
  useEffect(() => {
    if (!active || vehicleId < 0) return
    const ping = () => send({ type: 'watch_surround', vehicleId })
    ping()
    const timer = window.setInterval(ping, WATCH_INTERVAL_MS)
    return () => window.clearInterval(timer)
  }, [active, vehicleId])
}

/** キャンバスの縦横比を測る。再投影に要る（three の fov は垂直画角のため） */
function useAspect(ref: RefObject<HTMLDivElement | null>): number {
  const [aspect, setAspect] = useState(16 / 9)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const update = () => {
      const rect = el.getBoundingClientRect()
      if (rect.height > 0) setAspect(rect.width / rect.height)
    }
    update()
    const observer = new ResizeObserver(update)
    observer.observe(el)
    return () => observer.disconnect()
  }, [ref])
  return aspect
}

export function DetectionOverlay() {
  const cameraMode = useSimStore((s) => s.cameraMode)
  const followTarget = useSimStore((s) => s.followTarget)
  const showDetections = useSimStore((s) => s.view.detections)
  const quad = useSimStore(isQuadView)

  const single = showDetections && cameraMode === 'driver' && !quad
  const tracked = useTracked(followTarget, single || quad)
  useWatchSurround(quad, followTarget)
  const ref = useRef<HTMLDivElement>(null)
  // 4 分割は縦横とも半分に割るので、ペインの縦横比は画面と同じ
  const aspect = useAspect(ref)

  return (
    <div ref={ref} className="detection-overlay">
      {single &&
        tracked.front
          .filter((det) => det.cls !== DET_LANE)
          .map((det, i) => (
            <DetectionBox key={`${det.cls}-${i}`} det={det} aspect={aspect} fovDeg={paneFov('front')} />
          ))}
      {quad && (
        <QuadPanes tracked={tracked} aspect={aspect} showDetections={showDetections} />
      )}
    </div>
  )
}

function QuadPanes({
  tracked,
  aspect,
  showDetections,
}: {
  tracked: Tracked
  aspect: number
  showDetections: boolean
}) {
  const vehicle = tracked.vehicle
  const summary = assistSummary(vehicle?.assist)
  return (
    <>
      {QUAD_PANES.map((pane, index) => {
        const rect = quadPaneRect(index)
        const dets = pane === 'front' ? tracked.front : (tracked.surround[pane] ?? EMPTY_DETECTIONS)
        const badge = paneAssist(vehicle?.assist, pane, vehicle?.turnSignal ?? 0)
        return (
          <div
            key={pane}
            className="quad-pane"
            style={{
              left: `${rect.left * 100}%`,
              top: `${rect.top * 100}%`,
              width: `${rect.width * 100}%`,
              height: `${rect.height * 100}%`,
            }}
          >
            {showDetections &&
              dets
                .filter((det) => det.cls !== DET_LANE)
                .map((det, i) => (
                  <DetectionBox key={`${det.cls}-${i}`} det={det} aspect={aspect} fovDeg={paneFov(pane)} />
                ))}
            <div className="quad-pane-head">
              <PaneLabel pane={pane} reverse={pane === 'rear' && !!vehicle?.reverse} />
              {badge && <span className={`quad-badge quad-badge--${badge.level}`}>{badge.text}</span>}
            </div>
          </div>
        )
      })}
      {summary && (
        <div className={`quad-summary quad-badge--${summary.level}`}>{summary.text}</div>
      )}
    </>
  )
}

function PaneLabel({ pane, reverse }: { pane: QuadPane; reverse: boolean }) {
  return (
    <span className="quad-pane-label">
      {PANE_LABELS[pane]}
      {reverse && <span className="quad-gear">R</span>}
    </span>
  )
}

function DetectionBox({ det, aspect, fovDeg }: { det: Detection; aspect: number; fovDeg: number }) {
  const { left, top, width, height } = projectBox(det.box, aspect, fovDeg)
  if (width <= 0 || height <= 0) return null
  const color = detectionFrameColor(det)
  const label = detectionLabel(det)
  const flip = top < LABEL_FLIP_THRESHOLD

  return (
    <div
      className={det.hazard ? 'detection-box detection-box--hazard' : 'detection-box'}
      style={{
        left: `${left * 100}%`,
        top: `${top * 100}%`,
        width: `${width * 100}%`,
        height: `${height * 100}%`,
        borderColor: color,
      }}
    >
      <span
        className="detection-label"
        style={flip ? { top: '2px' } : { bottom: 'calc(100% + 3px)' }}
      >
        <span className="detection-label-dot" style={{ background: color }} />
        {label}
        {det.distance !== undefined && (
          <span className="detection-label-distance">{det.distance.toFixed(1)}m</span>
        )}
      </span>
    </div>
  )
}
