/** 認識結果（バウンディングボックス）のオーバーレイ。 */

import { useEffect, useRef, useState, type RefObject } from 'react'
import { frameBuffer } from '../store/frameBuffer'
import { useSimStore } from '../store/simStore'
import { DET_LANE, type Detection, type FrameMessage } from '../types/protocol'
import { detectionColor, detectionLabel } from './detectionLabels'
import { projectBox } from './detectionProjection'

const EMPTY_DETECTIONS: Detection[] = []

/** ラベルを枠の上ではなく内側へ回り込ませるしきい値（正規化 y0）。 */
const LABEL_FLIP_THRESHOLD = 0.08

/** 追従対象の車両について、いま表示すべき検出結果一覧を返すフック */
function useTrackedDetections(followTarget: number, active: boolean): Detection[] {
  const [dets, setDets] = useState<Detection[]>(EMPTY_DETECTIONS)
  const lastDisplayed = useRef(-1)

  useEffect(() => {
    if (!active) {
      setDets(EMPTY_DETECTIONS)
      return
    }
    lastDisplayed.current = -1
    let raf = 0
    const tick = () => {
      if (frameBuffer.displayed !== lastDisplayed.current) {
        lastDisplayed.current = frameBuffer.displayed
        const curr: FrameMessage | null = frameBuffer.curr
        setDets(curr?.detections?.[String(followTarget)] ?? EMPTY_DETECTIONS)
      }
      raf = requestAnimationFrame(tick)
    }
    tick()
    return () => cancelAnimationFrame(raf)
  }, [active, followTarget])

  return dets
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

  const active = showDetections && cameraMode === 'driver'
  const dets = useTrackedDetections(followTarget, active)
  const ref = useRef<HTMLDivElement>(null)
  const aspect = useAspect(ref)

  return (
    <div ref={ref} className="detection-overlay">
      {active &&
        dets
          .filter((det) => det.cls !== DET_LANE)
          .map((det, i) => (
            <DetectionBox key={`${det.cls}-${i}`} det={det} aspect={aspect} />
          ))}
    </div>
  )
}

function DetectionBox({ det, aspect }: { det: Detection; aspect: number }) {
  const { left, top, width, height } = projectBox(det.box, aspect)
  if (width <= 0 || height <= 0) return null
  const color = detectionColor(det)
  const label = detectionLabel(det)
  const flip = top < LABEL_FLIP_THRESHOLD

  return (
    <div
      className="detection-box"
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
