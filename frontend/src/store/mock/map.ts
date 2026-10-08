/** モックの地図（碁盤の目の合成マップ）と、信号・標識の引き当て */

import type {
  MapBuilding,
  MapEdge,
  MapMessage,
  MapNode,
  MapPreset,
  MapSign,
  MapSignal,
  SignKind,
  Vec2,
} from '../../types/protocol.ts'
import { SIGNAL_GREEN, SIGNAL_RED, SIGNAL_YELLOW } from '../../types/protocol.ts'
import { GRID_HALF, GRID_N, GRID_SPACING, ROAD_WIDTH, hopKey, makeRng, nodeId, nodeX, nodeY } from './grid.ts'

const MASK64 = (1n << 64n) - 1n

/** 交差点のキーを 0〜1 の実数へ散らす（backend/app/sim/signals.py の `_scatter01`。splitmix64 の finalizer） */
function scatter01(key: number): number {
  let z = (BigInt.asUintN(64, BigInt(key)) + 0x9e3779b97f4a7c15n) & MASK64
  z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64
  z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64
  z = z ^ (z >> 31n)
  return Number(z >> 11n) / 2 ** 53
}

/** 交差点とみなす最小の接続本数（loader.SIGNAL_MIN_STREETS） */
const SIGNAL_MIN_STREETS = 3
/** 停止線を交差点端からどれだけ手前に引くか（横断歩道 4m + 余裕 1m） */
const SIGNAL_SETBACK_EXTRA_M = 4.0 + 1.0
/** 青 25 / 黄 3 / 全赤 2 秒。現示の数だけ枠を分け合う（config.SIGNAL_*） */
const SIGNAL_GREEN_SEC = 25
const SIGNAL_YELLOW_SEC = 3
const SIGNAL_ALL_RED_SEC = 2
const SIGNAL_GREEN_MIN_SEC = 6
const SIGNAL_CYCLE = (SIGNAL_GREEN_SEC + SIGNAL_YELLOW_SEC + SIGNAL_ALL_RED_SEC) * 2
/** 同じ現示にまとめてよい進入路の軸差の上限（loader.SIGNAL_CONFLICT_ANGLE） */
const SIGNAL_CONFLICT_ANGLE = (30 * Math.PI) / 180

/** 交差点から標識までの距離 [m]（config.SPEED_SIGN_SETBACK_M） */
const SPEED_SIGN_SETBACK_M = 12.0
/** 路端から支柱の中心までの距離 [m]（config.SPEED_SIGN_SIDE_MARGIN） */
const SPEED_SIGN_SIDE_MARGIN = 0.8
/** 規制速度が「変わった」とみなす差 [m/s]（loader.SIGN_LIMIT_EPSILON_MPS） */
const SIGN_LIMIT_EPSILON_MPS = 1.0 / 3.6

export const MOCK_PRESETS: MapPreset[] = [
  {
    id: 'ginza',
    name: '東京・銀座',
    description: '中央通り〜晴海通り周辺の高密度市街地',
    centerLat: 35.6717,
    centerLon: 139.765,
    radiusM: 400,
  },
  {
    id: 'umeda',
    name: '大阪・梅田',
    description: '阪急とJRの大阪駅に隣接する、大通りと商業ビルが集中する梅田',
    centerLat: 34.7025,
    centerLon: 135.4959,
    radiusM: 400,
  },
  {
    id: 'sakae',
    name: '名古屋・栄',
    description: '久屋大通と広小路通が交わる、碁盤の目状の街路が広がる名古屋・栄',
    centerLat: 35.1681,
    centerLon: 136.9083,
    radiusM: 400,
  },
  {
    id: 'kanazawa',
    name: '石川・金沢（モックは縮小版）',
    description: '香林坊付近を模した合成マップ。実サーバーは市街地全域（12km 四方）',
    centerLat: 36.5606,
    centerLon: 136.6555,
    radiusM: 400,
  },
]

export function buildMockMap(presetId: string, name: string): MapMessage {
  const rng = makeRng(presetId.length * 7919 + 12345)

  const nodes: MapNode[] = []
  for (let gy = 0; gy < GRID_N; gy++) {
    for (let gx = 0; gx < GRID_N; gx++) {
      nodes.push({ id: nodeId(gx, gy), x: nodeX(gx), y: nodeY(gy) })
    }
  }

  const edges: MapEdge[] = []
  let edgeId = 0
  const pushEdge = (a: number, b: number, lanes: number) => {
    const na = nodes[a]
    const nb = nodes[b]
    const polyline: Vec2[] = [
      [na.x, na.y],
      [(na.x + nb.x) / 2, (na.y + nb.y) / 2],
      [nb.x, nb.y],
    ]
    edges.push({
      id: edgeId++,
      u: a,
      v: b,
      lanes,
      width: lanes >= 4 ? ROAD_WIDTH * 1.7 : ROAD_WIDTH,
      oneway: false,
      speedLimit: lanes >= 4 ? 16.7 : 11.1,
      polyline,
    })
  }
  for (let gy = 0; gy < GRID_N; gy++) {
    for (let gx = 0; gx < GRID_N; gx++) {
      const majorX = gy % 3 === 0
      const majorY = gx % 3 === 0
      if (gx + 1 < GRID_N) pushEdge(nodeId(gx, gy), nodeId(gx + 1, gy), majorX ? 4 : 2)
      if (gy + 1 < GRID_N) pushEdge(nodeId(gx, gy), nodeId(gx, gy + 1), majorY ? 4 : 2)
    }
  }

  const buildings: MapBuilding[] = []
  let buildingId = 0
  const margin = 10
  for (let gy = 0; gy + 1 < GRID_N; gy++) {
    for (let gx = 0; gx + 1 < GRID_N; gx++) {
      const x0 = nodeX(gx) + margin
      const y0 = nodeY(gy) + margin
      const inner = GRID_SPACING - margin * 2
      const sub = 3
      const cell = inner / sub
      for (let sy = 0; sy < sub; sy++) {
        for (let sx = 0; sx < sub; sx++) {
          if (rng() < 0.18) continue
          const cx = x0 + cell * (sx + 0.5) + (rng() - 0.5) * 4
          const cy = y0 + cell * (sy + 0.5) + (rng() - 0.5) * 4
          const w = cell * (0.55 + rng() * 0.3)
          const d = cell * (0.55 + rng() * 0.3)
          const rot = (rng() - 0.5) * 0.12
          const cos = Math.cos(rot)
          const sin = Math.sin(rot)
          const corners: Vec2[] = [
            [-w / 2, -d / 2],
            [w / 2, -d / 2],
            [w / 2, d / 2],
            [-w / 2, d / 2],
          ]
          const outline: Vec2[] = corners.map(([ox, oy]) => [
            cx + ox * cos - oy * sin,
            cy + ox * sin + oy * cos,
          ])
          const distToCenter = Math.hypot(cx, cy) / GRID_HALF
          const base = 9 + (1 - distToCenter) * 34
          buildings.push({
            id: buildingId++,
            height: Math.max(6, base * (0.5 + rng())),
            outline,
          })
        }
      }
    }
  }

  const signs = buildMockSpeedSigns(nodes, edges)
  const signKinds: SignKind[] = ['stop', 'crosswalk', 'one_way', 'mandatory_direction', 'no_parking', 'no_stopping']
  for (const [i, kind] of signKinds.entries()) {
    const source = signs[i * Math.max(1, Math.floor(signs.length / signKinds.length))]
    if (source) signs.push({ ...source, id: signs.length, x: source.x + 1, kind, speedLimit: 0, direction: kind === 'mandatory_direction' ? 'left_or_straight' : 'straight' })
  }
  return {
    type: 'map',
    presetId,
    name,
    bounds: {
      minX: -GRID_HALF - 20,
      maxX: GRID_HALF + 20,
      minY: -GRID_HALF - 20,
      maxY: GRID_HALF + 20,
    },
    nodes,
    edges,
    buildings,
    signals: buildMockSignals(nodes, edges),
    signs,
  }
}

/** 進入路を、軸のそろった組へ分けた群番号を返す（loader._phase_groups と同じ） */
function phaseGroups(headings: number[]): number[] {
  const n = headings.length
  if (n <= 1) return new Array<number>(n).fill(0)

  const modPi = (v: number) => ((v % Math.PI) + Math.PI) % Math.PI
  const axes = headings.map((h, i) => ({ axis: modPi(h), i })).sort((a, b) => a.axis - b.axis)
  let start = 0
  let widest = -1
  for (let k = 0; k < n; k++) {
    const gap = modPi(axes[(k + 1) % n].axis - axes[k].axis)
    if (gap > widest) {
      widest = gap
      start = (k + 1) % n
    }
  }

  const out = new Array<number>(n).fill(0)
  let group = 0
  let base = axes[start].axis
  for (let step = 0; step < n; step++) {
    const { axis, i } = axes[(start + step) % n]
    if (step && modPi(axis - base) >= SIGNAL_CONFLICT_ANGLE) {
      group += 1
      base = axis
    }
    out[i] = group
  }
  return out
}

/** 交差点ごとに、進入路 1 本につき 1 基の信号機を作る。 */
function buildMockSignals(nodes: MapNode[], edges: MapEdge[]): MapSignal[] {
  const incident = new Map<number, MapEdge[]>()
  for (const e of edges) {
    for (const end of [e.u, e.v]) {
      const list = incident.get(end)
      if (list) list.push(e)
      else incident.set(end, [e])
    }
  }

  const signals: MapSignal[] = []
  for (const node of nodes) {
    const around = incident.get(node.id) ?? []
    const neighbours = new Set(around.map((e) => (e.u === node.id ? e.v : e.u)))
    if (neighbours.size < SIGNAL_MIN_STREETS) continue

    const halfWidth = Math.max(...around.map((e) => e.width)) / 2
    const setback = halfWidth + SIGNAL_SETBACK_EXTRA_M

    const approaches = new Map<number, { heading: number; x: number; y: number; width: number }>()
    for (const e of around) {
      const neighbourId = e.u === node.id ? e.v : e.u
      if (approaches.has(neighbourId)) continue
      const from = nodes[neighbourId]
      const heading = Math.atan2(node.y - from.y, node.x - from.x)
      approaches.set(neighbourId, {
        heading,
        x: node.x - Math.cos(heading) * setback,
        y: node.y - Math.sin(heading) * setback,
        width: e.width,
      })
    }

    const ordered = [...approaches.keys()].sort((a, b) => a - b).map((k) => approaches.get(k)!)
    const groups = phaseGroups(ordered.map((a) => a.heading))
    for (let k = 0; k < ordered.length; k++) {
      const a = ordered[k]
      signals.push({
        id: signals.length,
        nodeId: node.id,
        x: a.x,
        y: a.y,
        heading: a.heading,
        group: groups[k],
        roadWidth: a.width,
      })
    }
  }
  return signals
}

/** 規制速度が変わる進入口に、最高速度標識を 1 基ずつ立てる。 */
function buildMockSpeedSigns(nodes: MapNode[], edges: MapEdge[]): MapSign[] {
  const arriving = new Map<number, Array<[number, number]>>()
  const leaving = new Map<number, Array<[MapEdge, number]>>()
  const pushTo = <T>(m: Map<number, T[]>, key: number, value: T) => {
    const list = m.get(key)
    if (list) list.push(value)
    else m.set(key, [value])
  }
  for (const e of edges) {
    pushTo(arriving, e.v, [e.id, e.speedLimit])
    pushTo(arriving, e.u, [e.id, e.speedLimit])
    pushTo(leaving, e.u, [e, e.v])
    pushTo(leaving, e.v, [e, e.u])
  }

  const signs: MapSign[] = []
  for (const node of nodes) {
    const out = leaving.get(node.id)
    if (!out) continue
    for (const [edge, toId] of [...out].sort((a, b) => a[0].id - b[0].id)) {
      const incoming = (arriving.get(node.id) ?? [])
        .filter(([otherId]) => otherId !== edge.id)
        .map(([, limit]) => limit)
      if (
        incoming.length > 0 &&
        incoming.every((limit) => Math.abs(limit - edge.speedLimit) < SIGN_LIMIT_EPSILON_MPS)
      ) {
        continue
      }

      const to = nodes[toId]
      const heading = Math.atan2(to.y - node.y, to.x - node.x)
      const offset = edge.width / 2 + SPEED_SIGN_SIDE_MARGIN
      signs.push({
        id: signs.length,
        nodeId: node.id,
        edgeId: edge.id,
        x: node.x + Math.cos(heading) * SPEED_SIGN_SETBACK_M - Math.sin(heading) * offset,
        y: node.y + Math.sin(heading) * SPEED_SIGN_SETBACK_M + Math.cos(heading) * offset,
        heading,
        speedLimit: edge.speedLimit,
      })
    }
  }
  return signs
}

interface SignalTiming {
  green: number
  cycle: number
  offset: number
  shift: number
}

/** 読み込んだ地図の信号・標識の引き当て表。**灯色は simTime だけの関数**（backend/app/sim/signals.py と同じ式） */
export class MockSignals {
  /** 「どのノードへどちらから進入するか」→ signals の添字 */
  private readonly index = new Map<string, number>()
  /** 各信号の停止線が交差点からどれだけ手前か [m]（signals と同じ並び） */
  private setbacks: number[] = []
  private timing: SignalTiming[] = []
  /** 「どの区間に入るか」→ その入口に立つ標識の規制速度 [m/s] */
  private readonly limits = new Map<string, number>()

  /** マップを読むたびに作り直す。 */
  build(map: MapMessage): void {
    this.index.clear()
    this.setbacks = []
    this.timing = []
    this.limits.clear()
    for (const sg of map.signs ?? []) {
      if (sg.kind && sg.kind !== 'speed_limit') continue
      const gx = sg.nodeId % GRID_N
      const gy = Math.floor(sg.nodeId / GRID_N)
      const tx = gx + Math.round(Math.cos(sg.heading))
      const ty = gy + Math.round(Math.sin(sg.heading))
      this.limits.set(hopKey(gx, gy, tx, ty), sg.speedLimit)
    }
    const signals = map.signals ?? []
    const phaseCount = new Map<number, number>()
    for (const sg of signals) {
      phaseCount.set(sg.nodeId, Math.max(phaseCount.get(sg.nodeId) ?? 2, sg.group + 1))
    }
    for (let i = 0; i < signals.length; i++) {
      const sg = signals[i]
      const gx = sg.nodeId % GRID_N
      const gy = Math.floor(sg.nodeId / GRID_N)
      const fx = gx - Math.round(Math.cos(sg.heading))
      const fy = gy - Math.round(Math.sin(sg.heading))
      this.index.set(hopKey(fx, fy, gx, gy), i)
      this.setbacks.push(Math.hypot(nodeX(gx) - sg.x, nodeY(gy) - sg.y))

      const count = phaseCount.get(sg.nodeId) ?? 2
      const green = Math.max(
        SIGNAL_GREEN_MIN_SEC,
        SIGNAL_CYCLE / count - SIGNAL_YELLOW_SEC - SIGNAL_ALL_RED_SEC,
      )
      const slot = green + SIGNAL_YELLOW_SEC + SIGNAL_ALL_RED_SEC
      const cycle = slot * count
      this.timing.push({
        green,
        cycle,
        // 整数の剰余で散らすと、番号が隣り合う交差点が 1 秒ずつずれた「波」になる（CLAUDE.md）
        offset: scatter01(sg.nodeId) * cycle,
        shift: slot * sg.group,
      })
    }
  }

  /** いまの灯色（map.signals と同じ並び） */
  phases(simTime: number): number[] {
    const out: number[] = new Array(this.timing.length)
    for (let i = 0; i < this.timing.length; i++) {
      const timing = this.timing[i]
      const local = (simTime + timing.offset + timing.shift) % timing.cycle
      out[i] =
        local < timing.green
          ? SIGNAL_GREEN
          : local < timing.green + SIGNAL_YELLOW_SEC
            ? SIGNAL_YELLOW
            : SIGNAL_RED
    }
    return out
  }

  /** その区間（`hopKey`）の先に立つ信号の添字。無ければ undefined */
  signalOn(key: string): number | undefined {
    return this.index.get(key)
  }

  /** 停止線が交差点からどれだけ手前か [m] */
  setbackOf(signal: number): number {
    return this.setbacks[signal]
  }

  /** その区間（`hopKey`）の入口に立つ標識の規制速度 [m/s]。無ければ undefined */
  limitOn(key: string): number | undefined {
    return this.limits.get(key)
  }
}
