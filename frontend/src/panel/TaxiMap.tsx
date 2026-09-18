/** スマホ画面の 2D 地図（Canvas）。**3D シーンは使わず自前で描く**（決定 10）。 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { frameBuffer } from '../store/frameBuffer'
import { pedestrian } from '../store/pedestrian'
import { useSimStore } from '../store/simStore'
import { usePalette } from '../scene/usePalette'
import { vehicleColor } from '../scene/vehicleColors'
import type { Vec2 } from '../types/protocol'
import {
  boundsCenter,
  clampCenter,
  createProjection,
  drawBuildings,
  drawHeadingMark,
  drawPin,
  drawRoads,
  drawRoute,
  toEnu,
  zoomAround,
} from './taxiMapMath'
import type { MapProjection, MapView } from './taxiMapMath'

/** 動くもの（車両・自分）を描き直す間隔 [ms] */
const REDRAW_MS = 200

/** ドラッグとみなす移動量 [px] */
const DRAG_THRESHOLD_PX = 4

export type PickTarget = 'pickup' | 'dropoff' | null

export interface TaxiMapProps {
  /** いま地点を選んでいる最中か */
  picking: PickTarget
  onPick: (point: Vec2) => void
  draftPickup: Vec2 | null
  draftDropoff: Vec2 | null
}

export function TaxiMap({ picking, onPick, draftPickup, draftDropoff }: TaxiMapProps) {
  const map = useSimStore((s) => s.map)
  const taxi = useSimStore((s) => s.taxi)
  const palette = usePalette()
  const panelOpen = useSimStore((s) => s.panelOpen)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [view, setView] = useState<MapView>({ zoom: 1, centerX: 0, centerY: 0 })
  const [size, setSize] = useState({ width: 300, height: 300 })

  const staticLayer = useRef<HTMLCanvasElement | null>(null)
  const staticKey = useRef('')
  const drag = useRef<{ x: number; y: number; moved: number; view: MapView } | null>(null)

  // マップが変わったら全体が入る見え方へ戻す
  useEffect(() => {
    if (!map) return
    const [cx, cy] = boundsCenter(map.bounds)
    setView({ zoom: 1, centerX: cx, centerY: cy })
    staticKey.current = ''
  }, [map])

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

  const projection = useMemo<MapProjection | null>(
    () => (map ? createProjection(map.bounds, size.width, size.height, view) : null),
    [map, size.width, size.height, view],
  )

  /** 道路と建物は動かないので、見え方が変わったときだけ描き直してキャッシュする */
  const ensureStaticLayer = useCallback(
    (p: MapProjection): HTMLCanvasElement | null => {
      if (!map) return null
      const key = [
        map.presetId,
        p.scale.toFixed(5),
        p.centerX.toFixed(2),
        p.centerY.toFixed(2),
        p.width,
        p.height,
        palette.roadSurface,
      ].join('|')
      if (staticLayer.current && staticKey.current === key) return staticLayer.current

      const layer = staticLayer.current ?? document.createElement('canvas')
      staticLayer.current = layer
      const dpr = Math.min(2, window.devicePixelRatio || 1)
      layer.width = Math.round(p.width * dpr)
      layer.height = Math.round(p.height * dpr)
      const ctx = layer.getContext('2d')
      if (!ctx) return null
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, p.width, p.height)
      drawBuildings(ctx, p, map.buildings, palette.buildingLow)
      drawRoads(ctx, p, map.edges, palette.roadSurface, Math.max(1, 1.2 * Math.sqrt(view.zoom)))
      staticKey.current = key
      return layer
    },
    [map, palette.buildingLow, palette.roadSurface, view.zoom],
  )

  const render = useCallback(() => {
    const canvas = canvasRef.current
    const p = projection
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

    const layer = ensureStaticLayer(p)
    if (layer) ctx.drawImage(layer, 0, 0, p.width, p.height)

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
  }, [projection, ensureStaticLayer, taxi, palette, draftPickup, draftDropoff])

  // ★ パネルが閉じている間はポーリングごと止める（code_review F-10）
  useEffect(() => {
    if (!panelOpen) return
    render()
    const timer = window.setInterval(render, REDRAW_MS)
    return () => window.clearInterval(timer)
  }, [render, panelOpen])

  const pointFromEvent = useCallback(
    (e: { clientX: number; clientY: number }): Vec2 | null => {
      const canvas = canvasRef.current
      const p = projection
      if (!canvas || !p) return null
      const rect = canvas.getBoundingClientRect()
      return toEnu(p, e.clientX - rect.left, e.clientY - rect.top)
    },
    [projection],
  )

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    try {
      e.currentTarget.setPointerCapture(e.pointerId)
    } catch {
      /* 掴めなくてもドラッグ自体は成立する */
    }
    drag.current = { x: e.clientX, y: e.clientY, moved: 0, view }
  }

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const start = drag.current
    const p = projection
    if (!start || !p || !map) return
    const dx = e.clientX - start.x
    const dy = e.clientY - start.y
    start.moved = Math.max(start.moved, Math.hypot(dx, dy))
    if (start.moved <= DRAG_THRESHOLD_PX) return
    setView(
      clampCenter(
        {
          zoom: start.view.zoom,
          centerX: start.view.centerX - dx / p.scale,
          centerY: start.view.centerY + dy / p.scale,
        },
        map.bounds,
      ),
    )
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
    const p = projection
    if (!canvas || !p || !map) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = canvas.getBoundingClientRect()
      const anchor = toEnu(p, e.clientX - rect.left, e.clientY - rect.top)
      setView((prev) => zoomAround(prev, map.bounds, anchor, e.deltaY < 0 ? 1.2 : 1 / 1.2))
    }
    canvas.addEventListener('wheel', onWheel, { passive: false })
    return () => canvas.removeEventListener('wheel', onWheel)
  }, [projection, map])

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
      {!map && <div className="taxi-map-empty">地図が読み込まれていません</div>}
    </div>
  )
}
