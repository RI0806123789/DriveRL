/** スマホ画面の車載カメラ。**映像は `scene/TaxiCameraFeed` が 15fps で書き込む。** */

import { useEffect, useRef, useState } from 'react'

import { feedSizeFor, taxiCamera } from '../store/taxiCamera'

/** 「映像が来ているか」を見に行く間隔 [ms]。表示は 1 行なので粗くてよい */
const LIVE_POLL_MS = 250

export interface TaxiCameraViewProps {
  on: boolean
  vehicleId: number
  /** 画面に出す車の呼び名。**番号ではなくナンバープレート**（同じ画面で揃える） */
  plate: string
}

export function TaxiCameraView({ on, vehicleId, plate }: TaxiCameraViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [live, setLive] = useState(false)

  // 転送先を貸し出す。閉じたら必ず返す（返さないと畳んだ裏で描き続ける）
  useEffect(() => {
    if (!on || vehicleId < 0) return
    const el = canvasRef.current
    if (!el) return

    taxiCamera.canvas = el
    taxiCamera.vehicleId = vehicleId

    // 描く解像度は**表示の実寸 × dpr**。固定にすると拡大されて荒く見える
    const measure = () => {
      const rect = el.getBoundingClientRect()
      if (rect.width < 1 || rect.height < 1) return
      const [w, h] = feedSizeFor(rect.width, rect.height, window.devicePixelRatio || 1)
      taxiCamera.width = w
      taxiCamera.height = h
    }
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(el)

    return () => {
      observer.disconnect()
      taxiCamera.canvas = null
      taxiCamera.vehicleId = -1
      taxiCamera.live = false
    }
  }, [on, vehicleId])

  // 中身が変わったときだけ state を触る（code_review F-11）
  useEffect(() => {
    if (!on) {
      setLive(false)
      return
    }
    const timer = window.setInterval(() => {
      setLive((prev) => (prev === taxiCamera.live ? prev : taxiCamera.live))
    }, LIVE_POLL_MS)
    return () => window.clearInterval(timer)
  }, [on])

  return (
    <div className="taxi-camera" data-on={on ? 'true' : 'false'} aria-hidden={!on}>
      <div className="taxi-camera-inner">
        <canvas
          ref={canvasRef}
          className="taxi-camera-canvas"
          data-live={live ? 'true' : 'false'}
        />
        <div className="taxi-camera-scan" />
        <div className="taxi-camera-bar">
          <span className="taxi-camera-rec" data-live={live ? 'true' : 'false'} />
          <span>車載カメラ</span>
          {vehicleId >= 0 && <span className="taxi-camera-id">{plate}</span>}
        </div>
        {!live && <div className="taxi-camera-wait">映像を待っています…</div>}
      </div>
    </div>
  )
}
