/** スマホ画面の 2D 地図の投影と描画。**React から切り離した純粋モジュール。** */

import type { MapBuilding, MapBounds, MapEdge, Vec2 } from '../types/protocol'

/** 地図の見え方（倍率 1 でマップ全体がちょうど収まる） */
export interface MapView {
  zoom: number
  /** 画面中心の ENU 座標 */
  centerX: number
  centerY: number
  /** 地図を回す角度 [rad]。0 で北が上、`heading - π/2` で進行方向が上 */
  rotation: number
}

/** ENU ↔ キャンバス座標の変換。**y は上下が反転する**（北が上） */
export interface MapProjection {
  width: number
  height: number
  /** ENU 1m あたりのピクセル数 */
  scale: number
  centerX: number
  centerY: number
  /** 地図を回す角度 [rad] */
  rotation: number
  /** `rotation` の余弦・正弦。毎フレーム数千回引くのでここに持つ */
  cos: number
  sin: number
}

export const ZOOM_MIN = 1
/**
 * 倍率の上限。**マップの一辺に対する比なので、広いエリアほど大きな値が要る**
 * （金沢は 12.3km 四方あり、40 倍では 300m 幅までしか寄れなかった）。
 * 120 なら銀座で 7m 幅、金沢で 102m 幅まで寄れる。
 */
export const ZOOM_MAX = 120

/** 倍率 1（マップ全体が収まる状態）の 1m あたりピクセル数。 */
export function baseScale(
  bounds: MapBounds,
  width: number,
  height: number,
  padding = 6,
): number {
  const spanX = Math.max(1e-6, bounds.maxX - bounds.minX)
  const spanY = Math.max(1e-6, bounds.maxY - bounds.minY)
  return Math.min((width - padding * 2) / spanX, (height - padding * 2) / spanY)
}

/** マップ全体が収まる倍率を 1 として投影を作る。 */
export function createProjection(
  bounds: MapBounds,
  width: number,
  height: number,
  view: MapView,
  padding = 6,
): MapProjection {
  const rotation = Number.isFinite(view.rotation) ? view.rotation : 0
  return {
    width,
    height,
    scale: baseScale(bounds, width, height, padding) * clampZoom(view.zoom),
    centerX: view.centerX,
    centerY: view.centerY,
    rotation,
    cos: Math.cos(rotation),
    sin: Math.sin(rotation),
  }
}

export function clampZoom(zoom: number): number {
  if (!Number.isFinite(zoom)) return ZOOM_MIN
  return Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, zoom))
}

/** 角度を -π..π に畳む。 */
export function normalizeAngle(angle: number): number {
  return Math.atan2(Math.sin(angle), Math.cos(angle))
}

/** 角度を**近いほうへ回して**補間する（359° → 1° は +2°）。 */
export function lerpAngle(from: number, to: number, t: number): number {
  const k = Math.max(0, Math.min(1, t))
  return from + normalizeAngle(to - from) * k
}

/** 進行方向を画面の上に向けるための回転角。 */
export function rotationForHeading(heading: number): number {
  return heading - Math.PI / 2
}

/** マップの中心（倍率 1 のときの見え方） */
export function boundsCenter(bounds: MapBounds): Vec2 {
  return [(bounds.minX + bounds.maxX) / 2, (bounds.minY + bounds.maxY) / 2]
}

/**
 * ENU の点をキャンバス座標へ。
 *
 * ★ **回転が入ると x と y は独立に変換できない**ので、軸ごとの
 * `toCanvasX()` / `toCanvasY()` は廃止した（片方だけ呼ぶと北向きの値が返り、
 * 回っていることに気づけないまま座標がずれる）。
 */
export function toCanvas(p: MapProjection, x: number, y: number): Vec2 {
  const dx = x - p.centerX
  const dy = y - p.centerY
  const rx = dx * p.cos + dy * p.sin
  const ry = -dx * p.sin + dy * p.cos
  return [p.width / 2 + rx * p.scale, p.height / 2 - ry * p.scale]
}

/** キャンバス座標を ENU へ戻す（地図をクリックして地点を選ぶのに使う）。 */
export function toEnu(p: MapProjection, cx: number, cy: number): Vec2 {
  const rx = (cx - p.width / 2) / p.scale
  const ry = -(cy - p.height / 2) / p.scale
  return [
    p.centerX + rx * p.cos - ry * p.sin,
    p.centerY + rx * p.sin + ry * p.cos,
  ]
}

/** 見えている範囲の ENU 矩形（描画を間引くのに使う）。**回転すると広がる。** */
export function visibleBounds(p: MapProjection, margin = 0): MapBounds {
  let minX = Infinity
  let maxX = -Infinity
  let minY = Infinity
  let maxY = -Infinity
  for (const [cx, cy] of [
    [0, 0],
    [p.width, 0],
    [0, p.height],
    [p.width, p.height],
  ]) {
    const [x, y] = toEnu(p, cx, cy)
    if (x < minX) minX = x
    if (x > maxX) maxX = x
    if (y < minY) minY = y
    if (y > maxY) maxY = y
  }
  return {
    minX: minX - margin,
    maxX: maxX + margin,
    minY: minY - margin,
    maxY: maxY + margin,
  }
}

/** 中心を動かしても地図が視界から外れないように留める。 */
export function clampCenter(view: MapView, bounds: MapBounds): MapView {
  return {
    zoom: clampZoom(view.zoom),
    centerX: Math.min(bounds.maxX, Math.max(bounds.minX, view.centerX)),
    centerY: Math.min(bounds.maxY, Math.max(bounds.minY, view.centerY)),
    rotation: view.rotation,
  }
}

/** 自動で寄るときに、最低これだけの範囲は見せる [m]（1 点だけのとき近づきすぎない） */
export const FIT_MIN_SPAN_M = 150

/**
 * 指定した点が全部入る見え方を作る（配車の段階に合わせた自動ズーム）。
 * 点が無ければマップ全体へ戻す。**枠は回転後の向きで測る。**
 */
export function fitView(
  bounds: MapBounds,
  points: readonly Vec2[],
  width: number,
  height: number,
  rotation = 0,
  marginRatio = 0.26,
): MapView {
  const [cx, cy] = boundsCenter(bounds)
  if (points.length === 0) {
    return { zoom: ZOOM_MIN, centerX: cx, centerY: cy, rotation }
  }

  const cos = Math.cos(rotation)
  const sin = Math.sin(rotation)
  let x0 = Infinity
  let y0 = Infinity
  let x1 = -Infinity
  let y1 = -Infinity
  for (const [px, py] of points) {
    const rx = px * cos + py * sin
    const ry = -px * sin + py * cos
    if (rx < x0) x0 = rx
    if (ry < y0) y0 = ry
    if (rx > x1) x1 = rx
    if (ry > y1) y1 = ry
  }

  const spanX = Math.max(FIT_MIN_SPAN_M, x1 - x0)
  const spanY = Math.max(FIT_MIN_SPAN_M, y1 - y0)
  const room = Math.max(0.05, 1 - marginRatio)
  const want = Math.min((width * room) / spanX, (height * room) / spanY)

  // 回転した座標での中心を ENU へ戻す
  const mx = (x0 + x1) / 2
  const my = (y0 + y1) / 2

  return clampCenter(
    {
      zoom: clampZoom(want / baseScale(bounds, width, height)),
      centerX: mx * cos - my * sin,
      centerY: mx * sin + my * cos,
      rotation,
    },
    bounds,
  )
}

/**
 * 見え方を補間する。**倍率は対数で混ぜる**（線形だと桁の違いで寄り方が跳ねる。
 * 天候のフォグ距離と同じ理由）。**向きは近いほうへ回す。**
 *
 * `tRotation` を分けられるのは、向きだけゆっくり追わせるため（車の方位は
 * 交差点で一気に変わるので、寄り引きと同じ速さで回すと画面が振られる）。
 */
export function lerpView(from: MapView, to: MapView, t: number, tRotation = t): MapView {
  const k = Math.max(0, Math.min(1, t))
  const logFrom = Math.log(clampZoom(from.zoom))
  const logTo = Math.log(clampZoom(to.zoom))
  return {
    zoom: Math.exp(logFrom + (logTo - logFrom) * k),
    centerX: from.centerX + (to.centerX - from.centerX) * k,
    centerY: from.centerY + (to.centerY - from.centerY) * k,
    rotation: lerpAngle(from.rotation, to.rotation, tRotation),
  }
}

/** 2 つの見え方が実質同じか（画面上のずれと倍率比・向きで見る）。 */
export function viewsClose(
  a: MapView,
  b: MapView,
  scale: number,
  pixelEpsilon = 0.4,
  zoomEpsilon = 0.004,
  rotationEpsilon = 0.0015,
): boolean {
  const moved = Math.hypot(a.centerX - b.centerX, a.centerY - b.centerY) * scale
  const ratio = Math.abs(Math.log(clampZoom(a.zoom) / clampZoom(b.zoom)))
  const turned = Math.abs(normalizeAngle(a.rotation - b.rotation))
  return moved <= pixelEpsilon && ratio <= zoomEpsilon && turned <= rotationEpsilon
}

/** 別の見え方で描いたレイヤを、いまの見え方へ貼るときの変換。 */
export interface LayerTransform {
  /** レイヤの中心を置くキャンバス座標 */
  x: number
  y: number
  /** `ctx.rotate()` に渡す角度 [rad] */
  angle: number
  /** `ctx.scale()` に渡す倍率 */
  scale: number
}

/**
 * 別の見え方で描いたレイヤを、いまの見え方へ貼るときの変換。
 * **道路と建物を毎フレーム描き直さないため**にある（金沢は 58,120 本ある）。
 *
 * 回転が入ると平行移動と拡大だけでは貼れないので、`drawImage` の引数ではなく
 * 変換（移動・回転・拡大）を返す。貼る側は次の順で使う:
 *
 * ```
 * ctx.translate(t.x, t.y); ctx.rotate(t.angle); ctx.scale(t.scale, t.scale)
 * ctx.drawImage(layer, -w / 2, -h / 2, w, h)
 * ```
 */
export function layerTransform(
  layer: MapProjection,
  current: MapProjection,
): LayerTransform {
  const [x, y] = toCanvas(current, layer.centerX, layer.centerY)
  return {
    x,
    y,
    angle: current.rotation - layer.rotation,
    scale: current.scale / layer.scale,
  }
}

/**
 * レイヤ画像上の画素が、`layerTransform` で貼ったあとキャンバスのどこに来るか。
 * **検証用**（貼り替えても地物が同じ画素に乗ることを数値で確かめる）。
 */
export function applyLayerTransform(
  t: LayerTransform,
  layerWidth: number,
  layerHeight: number,
  lx: number,
  ly: number,
): Vec2 {
  const ox = (lx - layerWidth / 2) * t.scale
  const oy = (ly - layerHeight / 2) * t.scale
  const cos = Math.cos(t.angle)
  const sin = Math.sin(t.angle)
  return [t.x + ox * cos - oy * sin, t.y + ox * sin + oy * cos]
}

/** 回転して貼っても隅が欠けないレイヤの一辺 [px]（画面の対角）。 */
export function layerSizeFor(width: number, height: number, rotated: boolean): number {
  return rotated ? Math.ceil(Math.hypot(width, height)) : Math.ceil(Math.max(width, height))
}

/** ある点を固定したままズームする（ホイール操作）。 */
export function zoomAround(
  view: MapView,
  bounds: MapBounds,
  anchor: Vec2,
  factor: number,
): MapView {
  const next = clampZoom(view.zoom * factor)
  const ratio = next === view.zoom ? 1 : view.zoom / next
  return clampCenter(
    {
      zoom: next,
      centerX: anchor[0] + (view.centerX - anchor[0]) * ratio,
      centerY: anchor[1] + (view.centerY - anchor[1]) * ratio,
      rotation: view.rotation,
    },
    bounds,
  )
}

/**
 * 道路。**見えない区間と、縮尺に対して短すぎる区間は引かない**。
 * 金沢は 58,120 本あり、全体表示で全部なぞると 1 フレームを使い切る。
 * 引いた本数を返す（測るときの手がかり）。
 */
export function drawRoads(
  ctx: CanvasRenderingContext2D,
  p: MapProjection,
  edges: readonly MapEdge[],
  color: string,
  lineWidth: number,
  minPixels = 2.5,
): number {
  const box = visibleBounds(p, 50)
  const minSpanM = minPixels / p.scale
  let drawn = 0
  ctx.save()
  ctx.strokeStyle = color
  ctx.lineWidth = lineWidth
  ctx.lineCap = 'round'
  ctx.beginPath()
  for (const e of edges) {
    const line = e.polyline
    if (!line || line.length < 2) continue
    const a = line[0]
    const b = line[line.length - 1]
    if (
      (a[0] < box.minX && b[0] < box.minX) ||
      (a[0] > box.maxX && b[0] > box.maxX) ||
      (a[1] < box.minY && b[1] < box.minY) ||
      (a[1] > box.maxY && b[1] > box.maxY)
    ) {
      continue
    }
    if (Math.hypot(b[0] - a[0], b[1] - a[1]) < minSpanM) continue
    const [sx, sy] = toCanvas(p, a[0], a[1])
    ctx.moveTo(sx, sy)
    for (let i = 1; i < line.length; i++) {
      const [px, py] = toCanvas(p, line[i][0], line[i][1])
      ctx.lineTo(px, py)
    }
    drawn++
  }
  ctx.stroke()
  ctx.restore()
  return drawn
}

/** 建物。**投影して 2px 未満になるものは描かない**（見えないのに件数だけかさむ） */
export function drawBuildings(
  ctx: CanvasRenderingContext2D,
  p: MapProjection,
  buildings: readonly MapBuilding[],
  color: string,
  minPixels = 2,
): number {
  const box = visibleBounds(p, 30)
  let drawn = 0
  ctx.save()
  ctx.fillStyle = color
  ctx.beginPath()
  for (const b of buildings) {
    const pts = b.outline
    if (!pts || pts.length < 3) continue
    let x0 = Infinity
    let y0 = Infinity
    let x1 = -Infinity
    let y1 = -Infinity
    for (const [px, py] of pts) {
      if (px < x0) x0 = px
      if (py < y0) y0 = py
      if (px > x1) x1 = px
      if (py > y1) y1 = py
    }
    if (x1 < box.minX || x0 > box.maxX || y1 < box.minY || y0 > box.maxY) continue
    if ((x1 - x0) * p.scale < minPixels && (y1 - y0) * p.scale < minPixels) continue
    const [sx, sy] = toCanvas(p, pts[0][0], pts[0][1])
    ctx.moveTo(sx, sy)
    for (let i = 1; i < pts.length; i++) {
      const [px, py] = toCanvas(p, pts[i][0], pts[i][1])
      ctx.lineTo(px, py)
    }
    ctx.closePath()
    drawn++
  }
  ctx.fill()
  ctx.restore()
  return drawn
}

/** 経路 */
export function drawRoute(
  ctx: CanvasRenderingContext2D,
  p: MapProjection,
  route: readonly Vec2[],
  color: string,
  lineWidth: number,
): void {
  if (route.length < 2) return
  ctx.save()
  ctx.strokeStyle = color
  ctx.lineWidth = lineWidth
  ctx.lineJoin = 'round'
  ctx.lineCap = 'round'
  ctx.beginPath()
  const [sx, sy] = toCanvas(p, route[0][0], route[0][1])
  ctx.moveTo(sx, sy)
  for (let i = 1; i < route.length; i++) {
    const [px, py] = toCanvas(p, route[i][0], route[i][1])
    ctx.lineTo(px, py)
  }
  ctx.stroke()
  ctx.restore()
}

/** 車両（進行方向を向いた三角）。 */
export function drawHeadingMark(
  ctx: CanvasRenderingContext2D,
  p: MapProjection,
  x: number,
  y: number,
  heading: number,
  color: string,
  size: number,
  outline?: string,
): void {
  const [cx, cy] = toCanvas(p, x, y)
  // ENU の反時計回りは、y を反転したキャンバスでは時計回りになる。
  // 地図を回していれば、そのぶんだけ戻す
  const a = p.rotation - heading
  ctx.save()
  ctx.translate(cx, cy)
  ctx.rotate(a)
  ctx.beginPath()
  ctx.moveTo(size, 0)
  ctx.lineTo(-size * 0.72, size * 0.66)
  ctx.lineTo(-size * 0.72, -size * 0.66)
  ctx.closePath()
  ctx.fillStyle = color
  ctx.fill()
  if (outline) {
    ctx.strokeStyle = outline
    ctx.lineWidth = 1
    ctx.stroke()
  }
  ctx.restore()
}

/** 乗降地点の目印。 */
export function drawPin(
  ctx: CanvasRenderingContext2D,
  p: MapProjection,
  x: number,
  y: number,
  color: string,
  radius: number,
  ring: boolean,
): void {
  const [cx, cy] = toCanvas(p, x, y)
  ctx.save()
  ctx.beginPath()
  ctx.arc(cx, cy, radius, 0, Math.PI * 2)
  ctx.fillStyle = color
  ctx.fill()
  if (ring) {
    ctx.beginPath()
    ctx.arc(cx, cy, radius * 2.1, 0, Math.PI * 2)
    ctx.strokeStyle = color
    ctx.lineWidth = 1.5
    ctx.globalAlpha = 0.55
    ctx.stroke()
  }
  ctx.restore()
}

/** 北を指す針が、キャンバス上で向く角度 [rad]（`ctx.rotate()` に渡す値）。 */
export function northAngle(p: MapProjection): number {
  return p.rotation
}

/**
 * 北を示す針。**地図を回している間だけ出す**（`opacity` が 0 なら描かない）。
 * 北向きに戻りきると消えるので、回していないときに画面の要素が増えない。
 */
export function drawCompass(
  ctx: CanvasRenderingContext2D,
  p: MapProjection,
  x: number,
  y: number,
  radius: number,
  needle: string,
  dial: string,
  opacity: number,
): void {
  if (opacity <= 0.01) return
  ctx.save()
  ctx.globalAlpha = Math.min(1, opacity)
  ctx.translate(x, y)

  ctx.beginPath()
  ctx.arc(0, 0, radius, 0, Math.PI * 2)
  ctx.fillStyle = dial
  ctx.globalAlpha = Math.min(1, opacity) * 0.55
  ctx.fill()
  ctx.globalAlpha = Math.min(1, opacity)

  ctx.rotate(northAngle(p))
  ctx.beginPath()
  ctx.moveTo(0, -radius * 0.78)
  ctx.lineTo(radius * 0.42, radius * 0.5)
  ctx.lineTo(0, radius * 0.24)
  ctx.lineTo(-radius * 0.42, radius * 0.5)
  ctx.closePath()
  ctx.fillStyle = needle
  ctx.fill()
  ctx.restore()
}
