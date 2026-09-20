/** スマホ画面の 2D 地図（Canvas）。**3D シーンは使わず自前で描く**（決定 10）。 */

import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'

import { frameBuffer, getLatestVehicle } from '../store/frameBuffer'
import { pedestrian } from '../store/pedestrian'
import { useSimStore } from '../store/simStore'
import { usePalette } from '../scene/usePalette'
import { vehicleColor } from '../scene/vehicleColors'
import { TargetIcon } from '../ui/Icons'
import { isBoardablePhase } from '../types/protocol'
import type { Vec2 } from '../types/protocol'
import {
  ZOOM_MIN,
  baseScale,
  boundsCenter,
  clampCenter,
  createProjection,
  drawBuildings,
  drawCompass,
  drawHeadingMark,
  drawPin,
  drawRoads,
  drawRoute,
  fitView,
  layerSizeFor,
  layerTransform,
  lerpView,
  normalizeAngle,
  rotationForHeading,
  toEnu,
  viewsClose,
  zoomAround,
} from './taxiMapMath'
import type { MapProjection, MapView } from './taxiMapMath'

/** 止まっているときに描き直す間隔 [ms]。動いている間は毎フレーム描く */
const IDLE_REDRAW_MS = 120

/** 目標の見え方へ寄る速さ（1 秒あたりの寄り具合）。大きいほど機敏 */
const FOLLOW_RATE = 3.4

/** ドラッグとみなす移動量 [px] */
const DRAG_THRESHOLD_PX = 4

/** 道路と建物のレイヤを作り直す基準（倍率の比の対数と、画面幅に対するずれ） */
const LAYER_ZOOM_TOLERANCE = 0.16
const LAYER_SHIFT_RATIO = 0.22

/** 向きを追う速さ。寄り引き（FOLLOW_RATE）より遅い — 交差点で画面が振られないように */
const TURN_RATE = 1.6

/** 北から何ラジアン回れば方角の目印を出しきるか */
const COMPASS_FADE_RAD = 0.18
const COMPASS_RADIUS = 11
const COMPASS_MARGIN = 12

export type PickTarget = 'pickup' | 'dropoff' | null

export interface TaxiMapProps {
  /** いま地点を選んでいる最中か */
  picking: PickTarget
  onPick: (point: Vec2) => void
  draftPickup: Vec2 | null
  draftDropoff: Vec2 | null
  /** 地図の上へ重ねるもの（車載カメラ）。位置は重ねる側が決める */
  children?: ReactNode
}

export function TaxiMap({
  picking,
  onPick,
  draftPickup,
  draftDropoff,
  children,
}: TaxiMapProps) {
  const map = useSimStore((s) => s.map)
  const phase = useSimStore((s) => s.taxi.phase)
  const palette = usePalette()
  const panelOpen = useSimStore((s) => s.panelOpen)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [size, setSize] = useState({ width: 300, height: 300 })

  // ★ 見え方は state に置かない。毎フレーム寄せるので、state にすると
  //   60Hz でパネルごと再レンダリングされる（`frameBuffer` と同じ理由）
  const viewRef = useRef<MapView>({ zoom: ZOOM_MIN, centerX: 0, centerY: 0, rotation: 0 })
  /** 自動で追いかけるのをやめているか（利用者が自分で動かしたあと） */
  const manual = useRef(false)
  const [manualShown, setManualShown] = useState(false)

  const setManual = useCallback((on: boolean) => {
    manual.current = on
    setManualShown((prev) => (prev === on ? prev : on))
  }, [])

  const staticLayer = useRef<HTMLCanvasElement | null>(null)
  const staticAt = useRef<MapProjection | null>(null)
  const staticKey = useRef('')
  const drag = useRef<{ x: number; y: number; moved: number; view: MapView } | null>(null)

  // マップが変わったら全体が入る見え方へ戻す
  useEffect(() => {
    if (!map) return
    const [cx, cy] = boundsCenter(map.bounds)
    viewRef.current = { zoom: ZOOM_MIN, centerX: cx, centerY: cy, rotation: 0 }
    setManual(false)
    staticKey.current = ''
    staticAt.current = null
  }, [map, setManual])

  // 段階が変わったら自動追従へ戻す（手で動かしたまま置き去りにしない）
  useEffect(() => {
    setManual(false)
  }, [phase, picking, setManual])

  useEffect(() => {
    const el = canvasRef.current?.parentElement
    if (!el) return
    const observer = new ResizeObserver(() => {
      const rect = el.getBoundingClientRect()
      setSize((prev) => {
        const width = Math.max(120, Math.round(rect.width))
        const height = Math.max(120, Math.round(rect.height))
        return prev.width === width && prev.height === height ? prev : { width, height }
      })
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  const projectionFor = useCallback(
    (view: MapView): MapProjection | null =>
      map ? createProjection(map.bounds, size.width, size.height, view) : null,
    [map, size.width, size.height],
  )

  /**
   * 配車の段階に合わせて「収めたいもの」と「向き」を決める（自動ズームの目標）。
   *
   * ★ 進行方向へ向けるのは**乗車中（`riding`）だけ**。迎車中は乗る側が街の中で
   * 自分の居場所を掴めるよう北で固定し、到着（`arrived`）したら北へ戻し始める。
   */
  const targetView = useCallback((): MapView | null => {
    if (!map) return null
    const taxi = useSimStore.getState().taxi
    const car = taxi.vehicleId >= 0 ? getLatestVehicle(taxi.vehicleId) : null
    const points: Vec2[] = []
    const rotation = taxi.phase === 'riding' && car ? rotationForHeading(car.heading) : 0

    if (isBoardablePhase(taxi.phase)) {
      if (car) points.push([car.x, car.y])
      if (taxi.pickup) points.push(taxi.pickup)
    } else if (taxi.phase === 'riding') {
      if (car) points.push([car.x, car.y])
      if (taxi.dropoff) points.push(taxi.dropoff)
    } else if (taxi.phase === 'arrived') {
      if (taxi.dropoff) points.push(taxi.dropoff)
      if (car) points.push([car.x, car.y])
    } else if (draftPickup && draftDropoff && picking === null) {
      // 2 点を選び終えたら、その区間が見えるところまで寄せる
      points.push(draftPickup, draftDropoff)
    }
    return fitView(map.bounds, points, size.width, size.height, rotation)
  }, [map, size.width, size.height, draftPickup, draftDropoff, picking])

  /** 道路と建物は動かないので、見え方が大きく変わったときだけ描き直す */
  const ensureStaticLayer = useCallback(
    (p: MapProjection, settled: boolean): HTMLCanvasElement | null => {
      if (!map) return null
      // ★ レイヤは**画面の対角を一辺とする正方形**で描く。回した角度に関わらず
      //   画面を覆えるので、**向きが変わっても引き直さずに貼り替えだけで済む**
      //   （金沢の 58,120 本を回転のたびに引き直すと 1 フレームを使い切る）。
      const side = layerSizeFor(p.width, p.height, true)
      const key = [map.presetId, side, palette.roadSurface, palette.buildingLow].join('|')
      const at = staticAt.current
      const fresh = staticLayer.current !== null && staticKey.current === key && at !== null

      let stale = !fresh
      if (!stale && at !== null) {
        const zoomGap = Math.abs(Math.log(p.scale / at.scale))
        const shift =
          Math.hypot(p.centerX - at.centerX, p.centerY - at.centerY) * p.scale
        const turn = Math.abs(normalizeAngle(p.rotation - at.rotation))
        stale =
          zoomGap > LAYER_ZOOM_TOLERANCE ||
          shift > p.width * LAYER_SHIFT_RATIO ||
          // 動きが落ち着いたら、ぼけたまま残さずピントを合わせ直す
          (settled && (zoomGap > 1e-4 || shift > 0.5 || turn > 1e-4))
      }
      if (!stale) return staticLayer.current

      // 描くのは画面ではなく正方形の投影。貼るときに layerTransform が向きを合わせる
      const layerProj: MapProjection = { ...p, width: side, height: side }
      const layer = staticLayer.current ?? document.createElement('canvas')
      staticLayer.current = layer
      const dpr = Math.min(2, window.devicePixelRatio || 1)
      layer.width = Math.round(side * dpr)
      layer.height = Math.round(side * dpr)
      const ctx = layer.getContext('2d')
      if (!ctx) return null
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, side, side)
      const zoom = p.scale / baseScale(map.bounds, p.width, p.height)
      drawBuildings(ctx, layerProj, map.buildings, palette.buildingLow)
      drawRoads(
        ctx,
        layerProj,
        map.edges,
        palette.roadSurface,
        Math.max(1, 1.2 * Math.sqrt(zoom)),
      )
      staticKey.current = key
      staticAt.current = layerProj
      return layer
    },
    [map, palette.buildingLow, palette.roadSurface],
  )

  const render = useCallback(
    (settled: boolean) => {
      const canvas = canvasRef.current
      const p = projectionFor(viewRef.current)
      if (!canvas || !p) return
      const dpr = Math.min(2, window.devicePixelRatio || 1)
      const wantW = Math.round(p.width * dpr)
      const wantH = Math.round(p.height * dpr)
      if (canvas.width !== wantW || canvas.height !== wantH) {
        canvas.width = wantW
        canvas.height = wantH
      }
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, p.width, p.height)

      const layer = ensureStaticLayer(p, settled)
      const at = staticAt.current
      if (layer && at) {
        // ★ 寄っている途中も回っている途中も、レイヤは貼り替えるだけにする。
        //   金沢の 58,120 本を毎フレーム引き直すと、パネルだけで 1 フレームを使い切る
        const t = layerTransform(at, p)
        ctx.save()
        ctx.translate(t.x, t.y)
        ctx.rotate(t.angle)
        ctx.scale(t.scale, t.scale)
        ctx.drawImage(layer, -at.width / 2, -at.height / 2, at.width, at.height)
        ctx.restore()
      }

      const taxi = useSimStore.getState().taxi
      if (taxi.route && taxi.route.length >= 2) {
        drawRoute(ctx, p, taxi.route, palette.taxiMarker, 2.5)
      }

      const pickup = draftPickup ?? taxi.pickup
      const dropoff = draftDropoff ?? taxi.dropoff
      if (pickup) drawPin(ctx, p, pickup[0], pickup[1], palette.vehicleReached, 4, true)
      if (dropoff) drawPin(ctx, p, dropoff[0], dropoff[1], palette.vehicleCollided, 4, true)

      const curr = frameBuffer.curr
      if (curr) {
        for (const v of curr.vehicles) {
          if (!v.active) continue
          const isTaxi = v.id === taxi.vehicleId && taxi.phase !== 'idle'
          drawHeadingMark(
            ctx,
            p,
            v.x,
            v.y,
            v.heading,
            isTaxi ? palette.taxiMarker : vehicleColor(v.id),
            isTaxi ? 7 : 5,
            isTaxi ? palette.signText : undefined,
          )
        }
      }

      if (pedestrian.placed && !pedestrian.riding) {
        drawHeadingMark(
          ctx,
          p,
          pedestrian.x,
          pedestrian.y,
          pedestrian.heading,
          palette.pedestrianTop,
          6,
          palette.marking,
        )
      }

      // 回していない間は出さない（北が上なら目印は要らない）
      drawCompass(
        ctx,
        p,
        p.width - COMPASS_MARGIN - COMPASS_RADIUS,
        COMPASS_MARGIN + COMPASS_RADIUS,
        COMPASS_RADIUS,
        palette.marking,
        palette.buildingLow,
        Math.abs(normalizeAngle(p.rotation)) / COMPASS_FADE_RAD,
      )
    },
    [projectionFor, ensureStaticLayer, palette, draftPickup, draftDropoff],
  )

  // ★ パネルが閉じている間はループごと止める（code_review F-10）
  useEffect(() => {
    if (!panelOpen || !map) return
    let raf = 0
    let lastDraw = 0
    let lastTime = performance.now()

    const loop = (now: number) => {
      raf = requestAnimationFrame(loop)
      const dt = Math.min(0.1, (now - lastTime) / 1000)
      lastTime = now

      let moving = false
      if (!manual.current && drag.current === null) {
        const target = targetView()
        const p = projectionFor(viewRef.current)
        if (target && p) {
          if (!viewsClose(viewRef.current, target, p.scale)) {
            viewRef.current = lerpView(
              viewRef.current,
              target,
              1 - Math.exp(-FOLLOW_RATE * dt),
              1 - Math.exp(-TURN_RATE * dt),
            )
            moving = true
          }
        }
      }

      if (moving || now - lastDraw >= IDLE_REDRAW_MS) {
        lastDraw = now
        render(!moving)
      }
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [panelOpen, map, targetView, projectionFor, render])

  const pointFromEvent = useCallback(
    (e: { clientX: number; clientY: number }): Vec2 | null => {
      const canvas = canvasRef.current
      const p = projectionFor(viewRef.current)
      if (!canvas || !p) return null
      const rect = canvas.getBoundingClientRect()
      return toEnu(p, e.clientX - rect.left, e.clientY - rect.top)
    },
    [projectionFor],
  )

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    try {
      e.currentTarget.setPointerCapture(e.pointerId)
    } catch {
      /* 掴めなくてもドラッグ自体は成立する */
    }
    drag.current = { x: e.clientX, y: e.clientY, moved: 0, view: viewRef.current }
  }

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const start = drag.current
    const p = projectionFor(viewRef.current)
    if (!start || !p || !map) return
    const dx = e.clientX - start.x
    const dy = e.clientY - start.y
    start.moved = Math.max(start.moved, Math.hypot(dx, dy))
    if (start.moved <= DRAG_THRESHOLD_PX) return
    setManual(true)
    // 回していると画面の右は東とは限らない。掴んだ点が指の下から動かないよう戻す
    const rx = dx / p.scale
    const ry = -dy / p.scale
    viewRef.current = clampCenter(
      {
        zoom: start.view.zoom,
        centerX: start.view.centerX - (rx * p.cos - ry * p.sin),
        centerY: start.view.centerY - (rx * p.sin + ry * p.cos),
        rotation: start.view.rotation,
      },
      map.bounds,
    )
    render(false)
  }

  const onPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const start = drag.current
    drag.current = null
    if (!start || start.moved > DRAG_THRESHOLD_PX) return
    if (!picking) return
    const point = pointFromEvent(e)
    if (point) onPick(point)
  }

  /**
   * ホイールの拡大縮小。**`passive: false` で自前に登録して `preventDefault()` する**
   * （React の `onWheel` は passive なので止められず、Ctrl+ホイールでページごと
   * 拡大されて地図が読めなくなる）。
   */
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !map) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const p = projectionFor(viewRef.current)
      if (!p) return
      const rect = canvas.getBoundingClientRect()
      const anchor = toEnu(p, e.clientX - rect.left, e.clientY - rect.top)
      setManual(true)
      viewRef.current = zoomAround(
        viewRef.current,
        map.bounds,
        anchor,
        e.deltaY < 0 ? 1.2 : 1 / 1.2,
      )
      render(false)
    }
    canvas.addEventListener('wheel', onWheel, { passive: false })
    return () => canvas.removeEventListener('wheel', onWheel)
  }, [map, projectionFor, render])

  return (
    <div className="taxi-map" data-picking={picking ? 'true' : 'false'}>
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: '100%', display: 'block', touchAction: 'none' }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={() => {
          drag.current = null
        }}
      />

      <button
        type="button"
        className="taxi-map-recenter"
        data-visible={manualShown && map !== null ? 'true' : 'false'}
        inert={!manualShown}
        onClick={() => setManual(false)}
      >
        <TargetIcon size={13} />
        自動に戻す
      </button>

      {!map && <div className="taxi-map-empty">地図が読み込まれていません</div>}

      {children}
    </div>
  )
}
