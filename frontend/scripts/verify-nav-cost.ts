/** カーナビの描き直し 1 回にかかる時間を、金沢の規模で測る（ブラウザ不要）。 */

import {
  createProjection,
  drawBuildings,
  drawRoads,
  drawRoute,
  rotationForHeading,
} from '../src/panel/taxiMapMath.ts'
import type { MapBounds, MapEdge, MapBuilding, Vec2 } from '../src/types/protocol.ts'
import { NAV_SPAN_MAX_M, NAV_SPAN_MIN_M } from '../src/scene/vehicleGeometry.ts'

/** NavScreen.tsx と同じ値。増やすとそのまま負荷になる */
const TEX_W = 256
const TEX_H = 160
const NAV_FPS = 12
const MIN_ROAD_PX = 2.0

/** 金沢の実測値（CLAUDE.md より）。**手で書いた数字なので、出典はそちら** */
const KANAZAWA = {
  edges: 58120,
  buildings: 35607,
  spanM: 12300,
}

let failures = 0

function check(label: string, ok: boolean, detail = ''): void {
  console.log(`  [${ok ? 'OK  ' : 'NG  '}] ${label}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures++
}

/** Canvas が無い環境でも測れるよう、呼び出し回数だけ数える最小の ctx */
function stubContext(): CanvasRenderingContext2D {
  const noop = () => {}
  const ctx = {
    save: noop,
    restore: noop,
    beginPath: noop,
    moveTo: noop,
    lineTo: noop,
    closePath: noop,
    stroke: noop,
    fill: noop,
    fillRect: noop,
    arc: noop,
    translate: noop,
    rotate: noop,
    fillText: noop,
    measureText: () => ({ width: 0 }),
    strokeStyle: '',
    fillStyle: '',
    lineWidth: 0,
    lineCap: '',
    lineJoin: '',
    font: '',
    textAlign: '',
    textBaseline: '',
  }
  return ctx as unknown as CanvasRenderingContext2D
}

/** 金沢と同じ規模の地図を作る（形は問わない。本数だけ合わせる） */
function makeMap(): { bounds: MapBounds; edges: MapEdge[]; buildings: MapBuilding[] } {
  const half = KANAZAWA.spanM / 2
  const bounds: MapBounds = { minX: -half, minY: -half, maxX: half, maxY: half }
  let seed = 1
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff
    return seed / 0x7fffffff
  }

  const edges: MapEdge[] = []
  for (let i = 0; i < KANAZAWA.edges; i++) {
    const x = (rand() * 2 - 1) * half
    const y = (rand() * 2 - 1) * half
    const dx = (rand() * 2 - 1) * 60
    const dy = (rand() * 2 - 1) * 60
    edges.push({
      id: i,
      u: 0,
      v: 1,
      lanes: 2,
      width: 6.5,
      oneway: false,
      speedLimit: 13.9,
      length: Math.hypot(dx, dy),
      polyline: [
        [x, y],
        [x + dx, y + dy],
      ],
    } as unknown as MapEdge)
  }

  const buildings: MapBuilding[] = []
  for (let i = 0; i < KANAZAWA.buildings; i++) {
    const x = (rand() * 2 - 1) * half
    const y = (rand() * 2 - 1) * half
    buildings.push({
      id: i,
      height: 10,
      outline: [
        [x, y],
        [x + 12, y],
        [x + 12, y + 12],
        [x, y + 12],
      ],
    } as unknown as MapBuilding)
  }
  return { bounds, edges, buildings }
}

console.log('='.repeat(70))
console.log('カーナビの描き直し 1 回（金沢の規模）')
console.log('='.repeat(70))

const map = makeMap()
const ctx = stubContext()
const route: Vec2[] = Array.from({ length: 400 }, (_, i) => [i * 2, i * 1.5] as Vec2)

for (const [name, spanM] of [
  ['停車（いちばん寄る）', NAV_SPAN_MIN_M],
  ['最高速（いちばん引く）', NAV_SPAN_MAX_M],
] as Array<[string, number]>) {
  const spanX = map.bounds.maxX - map.bounds.minX
  const base = Math.min((TEX_W - 12) / spanX, (TEX_H - 12) / spanX)
  const zoom = TEX_W / spanM / base

  const marks: number[] = []
  for (let i = 0; i < 40; i++) {
    const view = {
      zoom,
      centerX: 0,
      centerY: 0,
      rotation: rotationForHeading(i * 0.1),
    }
    const p = createProjection(map.bounds, TEX_W, TEX_H, view, 6)
    const t0 = performance.now()
    drawBuildings(ctx, p, map.buildings, '#333')
    drawRoads(ctx, p, map.edges, '#888', 2.2, MIN_ROAD_PX)
    drawRoute(ctx, p, route, '#fc0', 2.4)
    marks.push(performance.now() - t0)
  }
  marks.sort((a, b) => a - b)
  const median = marks[Math.floor(marks.length / 2)]
  const budget = 1000 / NAV_FPS
  console.log(
    `  ${name.padEnd(22, '　')} ${median.toFixed(2)} ms/回` +
      `（${NAV_FPS}fps で ${(median * NAV_FPS).toFixed(0)} ms/秒）`,
  )
  check(
    `  ${name}: 1 回が ${budget.toFixed(0)}ms の予算に収まる`,
    median < budget,
    median.toFixed(2) + 'ms < ' + budget.toFixed(0) + 'ms',
  )
  check(
    `  ${name}: 60fps の 1 フレーム（16.7ms）を超えない`,
    median < 16.7,
    median.toFixed(2) + 'ms',
  )
}

console.log()
console.log('='.repeat(70))
if (failures > 0) {
  console.log(`結果: ${failures} 件の不合格`)
  process.exit(1)
}
console.log('結果: すべて合格')
