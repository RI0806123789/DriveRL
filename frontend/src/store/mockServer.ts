/** 開発用のモックサーバー（ブラウザ内で動く偽 WebSocket サーバー）。 */

import type {
  ClientMessage,
  Detection,
  DetectorEvaluation,
  DetectorMessage,
  DetectorRequest,
  FrameMessage,
  MapBuilding,
  MapEdge,
  MapMessage,
  MapNode,
  MapPreset,
  MapSign,
  MapSignal,
  MetricsMessage,
  NpcPedestrianState,
  ObstacleState,
  ServerMessage,
  SimConfig,
  SimParams,
  StatusPayload,
  TaxiMessage,
  Vec2,
  VehicleState,
  WeatherPreset,
  WeatherState,
} from '../types/protocol'
import {
  DET_LANE,
  DET_PEDESTRIAN,
  DET_TRAFFIC_LIGHT,
  PROTOCOL_VERSION,
  SIGNAL_GREEN,
  SIGNAL_RED,
  SIGNAL_YELLOW,
} from '../types/protocol'
import type { Transport } from './connection'

function makeRng(seed: number) {
  let s = seed >>> 0
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 4294967296
  }
}

const GRID_N = 9
const GRID_SPACING = 100
const GRID_HALF = ((GRID_N - 1) * GRID_SPACING) / 2
const ROAD_WIDTH = 7.0
const SIM_HZ = 20
const FRAME_MS = 1000 / SIM_HZ

/** 交差点とみなす最小の接続本数（loader.SIGNAL_MIN_STREETS） */
const SIGNAL_MIN_STREETS = 3
/** 停止線を交差点端からどれだけ手前に引くか（横断歩道 4m + 余裕 1m） */
const SIGNAL_SETBACK_EXTRA_M = 4.0 + 1.0
/** 青 25 / 黄 3 / 全赤 2 秒。半サイクルで交差方向と入れ替わる */
const SIGNAL_GREEN_SEC = 25
const SIGNAL_YELLOW_SEC = 3
const SIGNAL_ALL_RED_SEC = 2
const SIGNAL_HALF_CYCLE = SIGNAL_GREEN_SEC + SIGNAL_YELLOW_SEC + SIGNAL_ALL_RED_SEC
const SIGNAL_CYCLE = SIGNAL_HALF_CYCLE * 2

/** 交差点から標識までの距離 [m]（config.SPEED_SIGN_SETBACK_M） */
const SPEED_SIGN_SETBACK_M = 12.0
/** 路端から支柱の中心までの距離 [m]（config.SPEED_SIGN_SIDE_MARGIN） */
const SPEED_SIGN_SIDE_MARGIN = 0.8
/** 規制速度が「変わった」とみなす差 [m/s]（loader.SIGN_LIMIT_EPSILON_MPS） */
const SIGN_LIMIT_EPSILON_MPS = 1.0 / 3.6

/** 停止に使う減速度 [m/s^2]（signals.BRAKE_USE_RATIO 相当の値） */
const BRAKE_ACCEL = 4.8
/** 前走車との車間 [m]。車長 4.4m + 余裕 */
const CAR_GAP_M = 6.0

/** 徒歩キャラの手前で空ける距離 [m]・見る範囲 [m]・車体中心からの横幅 [m]（env.py と同じ値） */
const PLAYER_MARGIN_M = 4.5
const PLAYER_RANGE_M = 30.0
const PLAYER_HALF_WIDTH_M = 2.0
/** 位置が届かなくなってから街から消すまで [ms]（engine.PLAYER_POSE_TTL_SEC と同じ） */
const PLAYER_POSE_TTL_MS = 1000

const MOCK_PRESETS: MapPreset[] = [
  {
    id: 'ginza',
    name: '東京・銀座',
    description: '中央通り〜晴海通り周辺の高密度市街地',
    centerLat: 35.6717,
    centerLon: 139.765,
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
  {
    id: 'nonoichi',
    name: '野々市・扇が丘',
    description: '住宅と大学が混在する低層エリア',
    centerLat: 36.5301,
    centerLon: 136.6266,
    radiusM: 400,
  },
]

const MOCK_CONFIG: SimConfig = {
  maxVehicles: 8,
  maxPedestrians: 64,
  simHz: SIM_HZ,
  obsDim: 66,
  actionDim: 2,
}

/** 本物は `percep/weather.py` の PRESETS が出典。ここはモック用の写し。 */
const MOCK_WEATHER_PRESETS: WeatherPreset[] = [
  { id: 'clear', rain: 0, fog: 0 },
  { id: 'drizzle', rain: 0.35, fog: 0 },
  { id: 'rain', rain: 0.85, fog: 0 },
  { id: 'fog', rain: 0, fog: 0.75 },
  { id: 'heavy_fog', rain: 0.15, fog: 0.95 },
]

const DEFAULT_PARAMS: SimParams = {
  vehicleCount: 4,
  pedestrianCount: 16,
  simSpeed: 1,
  learningRate: 3e-4,
  gamma: 0.99,
  clipRange: 0.2,
  entropyCoef: 0.001,
  rolloutLength: 256,
  maxSpeed: 13.9,
  rewardGoal: 100,
  rewardCollision: -100,
  rewardProgress: 1,
  rewardOffroad: -1,
  rewardTime: -0.05,
  rewardSignal: -60,
  rewardOverspeed: -5,
  obeySignals: true,
  obeySpeedSigns: true,
  weatherRain: 0,
  weatherFog: 0,
  weatherAuto: false,
}

function nodeId(gx: number, gy: number): number {
  return gy * GRID_N + gx
}
function nodeX(gx: number): number {
  return gx * GRID_SPACING - GRID_HALF
}
function nodeY(gy: number): number {
  return gy * GRID_SPACING - GRID_HALF
}

function buildMockMap(presetId: string, name: string): MapMessage {
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
    signs: buildMockSpeedSigns(nodes, edges),
  }
}

/** 2 つの方位を「軸」として比べた角度差 [0, pi/2]（loader._axis_angle と同じ） */
function axisAngle(a: number, b: number): number {
  const d = Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b)))
  return Math.min(d, Math.PI - d)
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
    const ref = ordered[0].heading
    for (const a of ordered) {
      signals.push({
        id: signals.length,
        nodeId: node.id,
        x: a.x,
        y: a.y,
        heading: a.heading,
        group: axisAngle(a.heading, ref) < Math.PI / 4 ? 0 : 1,
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

/** モックの歩行者。格子の道に沿って歩き、交差点でたまに車道を横断する */
interface MockPedestrian {
  id: number
  gx: number
  gy: number
  /** 次に向かうノード */
  tx: number
  ty: number
  /** 区間の進み具合 0.0〜1.0 */
  t: number
  /** 歩道の左右。+1 が進行方向の左 */
  side: number
  /** 横断の進み具合 0.0〜1.0。0 なら歩道 */
  cross: number
  crossing: boolean
  stride: number
  speed: number
}

/** 歩道が車道中心から離れている距離 [m]（バックエンドの PEDESTRIAN_SIDEWALK_MARGIN 相当） */
const WALK_OFFSET_M = ROAD_WIDTH / 2 + 1.7

/** これより遠い歩行者は車の近くへ回す [m]（バックエンドの RECYCLE_FAR_M 相当） */
const PEDESTRIAN_RECYCLE_M = 220

/** 方向指示器を出し始める距離 [m]（バックエンドの TURN_LOOKAHEAD_M 相当） */
const TURN_LOOKAHEAD_M = 30

interface MockVehicle {
  id: number
  active: boolean
  gx: number
  gy: number
  /** 次に向かうノード */
  tx: number
  ty: number
  /** 現在エッジ上の進捗 0..1 */
  t: number
  x: number
  y: number
  heading: number
  speed: number
  steer: number
  collided: boolean
  reachedGoal: boolean
  /** いま適用されている規制速度 [m/s]。まだ標識を 1 基も通っていなければ 0 */
  speedLimit: number
  /** このエピソード中に規制速度を超えた回数 */
  speedViolations: number
  /** 前ステップで超過していたか。超え「始めた」瞬間だけを数えるため */
  overspeeding: boolean
  goalGx: number
  goalGy: number
  /** 目的地を決めた時点での残り距離 [m]。progress の分母 */
  routeTotal: number
  /** 経路の達成度 0..1（protocol.md 2.3 の progress） */
  progress: number
  /** route を送るべきか */
  routeDirty: boolean
  /** 制動指令が出ているか（ブレーキランプ） */
  braking: boolean
  /** 方向指示器。-1=左 / 0=消灯 / +1=右 */
  turnSignal: number
}

function clampGrid(v: number): number {
  return Math.max(0, Math.min(GRID_N - 1, v))
}

/** 目的地へ向かう次のノードを貪欲に選ぶ（x 方向を先に詰める） */
function chooseNext(v: MockVehicle, rng: () => number): void {
  const dx = v.goalGx - v.gx
  const dy = v.goalGy - v.gy
  if (dx === 0 && dy === 0) return
  const preferX = dy === 0 ? true : dx === 0 ? false : rng() < 0.5
  if (dx !== 0 && preferX) {
    v.tx = clampGrid(v.gx + Math.sign(dx))
    v.ty = v.gy
  } else if (dy !== 0) {
    v.tx = v.gx
    v.ty = clampGrid(v.gy + Math.sign(dy))
  } else {
    v.tx = clampGrid(v.gx + Math.sign(dx))
    v.ty = v.gy
  }
}

/** 目的地までの残り距離 [m]。 */
function remainingDistance(v: MockVehicle): number {
  const segRest = Math.hypot(nodeX(v.tx) - v.x, nodeY(v.ty) - v.y)
  const grid = (Math.abs(v.goalGx - v.tx) + Math.abs(v.goalGy - v.ty)) * GRID_SPACING
  return segRest + grid
}

/** 現在地から目的地までの経路をノード列で作る（x → y の順に詰める） */
function buildRoute(v: MockVehicle): Vec2[] {
  const pts: Vec2[] = [[v.x, v.y]]
  let gx = v.tx
  let gy = v.ty
  pts.push([nodeX(gx), nodeY(gy)])
  let guard = 0
  while ((gx !== v.goalGx || gy !== v.goalGy) && guard++ < 64) {
    if (gx !== v.goalGx) gx += Math.sign(v.goalGx - gx)
    else gy += Math.sign(v.goalGy - gy)
    pts.push([nodeX(gx), nodeY(gy)])
  }
  return pts
}

const MOCK_IDLE_TAXI: TaxiMessage = {
  type: 'taxi',
  phase: 'idle',
  vehicleId: -1,
  pickup: null,
  dropoff: null,
  routeRevision: 0,
  route: [],
  etaSeconds: 0,
  remainingDistanceM: 0,
  message: '',
}

/** 乗降地点に「着いた」とみなす距離 [m] と、配車を受け付ける最短距離 [m] */
const MOCK_TAXI_ARRIVE_M = 6
const MOCK_TAXI_MIN_TRIP_M = 50
/** ETA を出すときの速度の下限 [m/s] */
const MOCK_TAXI_MIN_SPEED = 3.5

/** ENU 座標を、いちばん近いグリッドノードへ寄せる（サーバーの道路スナップの代わり） */
function snapToGrid(x: number, y: number): { gx: number; gy: number; point: Vec2 } {
  const gx = clampGrid(Math.round((x + GRID_HALF) / GRID_SPACING))
  const gy = clampGrid(Math.round((y + GRID_HALF) / GRID_SPACING))
  return { gx, gy, point: [nodeX(gx), nodeY(gy)] }
}

const MOCK_EVAL_SAMPLES = 400

const MOCK_WEATHER_PERIOD_SEC = 480

const MOCK_MIN_VISIBILITY_M = 15

/** 採点結果の見本。数字は銀座での実測に寄せてある（霧で崩れる形）。 */
function makeMockEvaluation(): DetectorEvaluation {
  const classes = [
    { cls: 0, name: 'TRAFFIC_LIGHT', truth: 231, recall: 0.619, attr: 0.594 },
    { cls: 1, name: 'SPEED_SIGN', truth: 42, recall: 0.524, attr: 0.955 },
    { cls: 2, name: 'VEHICLE', truth: 171, recall: 0.456, attr: 1 },
    { cls: 3, name: 'OBSTACLE', truth: 389, recall: 0.586, attr: 1 },
    { cls: 4, name: 'LANE', truth: 236, recall: 0.665, attr: 1 },
  ]
  const weathers = [
    { name: 'clear', truth: 380, recall: 0.808 },
    { name: 'rain', truth: 371, recall: 0.814 },
    { name: 'fog', truth: 318, recall: 0.16 },
  ]
  return {
    samples: MOCK_EVAL_SAMPLES,
    elapsedSec: 4.3,
    overallRecall: 0.584,
    classes: classes.map((c) => ({
      cls: c.cls,
      name: c.name,
      truth: c.truth,
      matched: Math.round(c.truth * c.recall),
      recall: c.recall,
      attributeTotal: c.attr < 1 ? Math.round(c.truth * c.recall) : 0,
      attributeOk: c.attr < 1 ? Math.round(c.truth * c.recall * c.attr) : 0,
      attributeAccuracy: c.attr,
    })),
    weathers: weathers.map((w) => ({
      name: w.name,
      truth: w.truth,
      matched: Math.round(w.truth * w.recall),
      recall: w.recall,
    })),
    weakest: '（モック）信号機 の成績が最も低い（37%）。霧でも落ちています',
  }
}

/** 何も実行していないときの `detector` メッセージ。 */
function makeIdleDetector(): DetectorMessage {
  return {
    type: 'detector',
    state: 'idle',
    running: false,
    message: '（モック）認識器の学習はまだ実行していません',
    progress: 0,
    collected: 0,
    samples: 0,
    epoch: 0,
    epochs: 0,
    batch: 0,
    batches: 0,
    history: [],
    elapsedSec: 0,
    warning: '',
    paramCount: 0,
    presetName: null,
    request: null,
    evaluation: null,
    dataset: null,
    model: { exists: false, filename: 'detector.keras', sizeBytes: 0, modifiedAt: null, inUse: false },
    datasetFile: { exists: false, sizeBytes: 0, modifiedAt: null },
    limits: {
      samplesMin: 200,
      samplesMax: 4800,
      epochsMin: 1,
      epochsMax: 60,
      batchMin: 8,
      batchMax: 128,
      widthMin: 0.25,
      widthMax: 2.0,
      seedMin: 0,
      seedMax: 999999,
    },
  }
}

class MockServer {
  private emit: (json: string) => void
  private rng = makeRng(20260905)
  private params: SimParams = { ...DEFAULT_PARAMS }
  private status: StatusPayload = {
    state: 'idle',
    mapLoaded: false,
    presetId: null,
    renderPaused: false,
    learning: true,
    practicalMode: false,
    taxiVehicleId: -1,
    message: '（開発用モック）マップを選択してください',
  }
  private map: MapMessage | null = null
  private vehicles: MockVehicle[] = []
  /** 現在の灯色（map.signals と同じ並び）。step() の先頭で作り直す */
  private phases: number[] = []
  /** 「どのノードへどちらから進入するか」→ signals の添字 */
  private signalIndex = new Map<string, number>()
  /** 各信号の停止線が交差点からどれだけ手前か [m]（signals と同じ並び） */
  private signalSetback: number[] = []
  /** 「どの区間に入るか」→ その入口に立つ標識の規制速度 [m/s] */
  private signLimits = new Map<string, number>()
  private obstacles: ObstacleState[] = []
  private pedestrians: MockPedestrian[] = []
  private obstacleSeq = 1
  private tick = 0
  private simTime = 0
  private startedAt = performance.now()
  private frameTimer: ReturnType<typeof setInterval> | null = null
  private metricsTimer: ReturnType<typeof setInterval> | null = null
  private loadTimer: ReturnType<typeof setTimeout> | null = null
  private closed = false
  private updates = 0
  private episodes = 0
  private progress = 0
  private detector: DetectorMessage = makeIdleDetector()
  private detectorTimer: ReturnType<typeof setInterval> | null = null
  /** 実用モードの配車。**擬似的な再現で、経路はグリッドに沿った折れ線** */
  private taxi: TaxiMessage = { ...MOCK_IDLE_TAXI }
  private taxiSentAt = 0
  /** 徒歩キャラ。届かなくなったら消す（実機の TTL と同じ） */
  private player: Vec2 | null = null
  private playerAt = 0

  constructor(emit: (json: string) => void) {
    this.emit = emit
    for (let i = 0; i < MOCK_CONFIG.maxVehicles; i++) {
      this.vehicles.push(this.makeVehicle(i))
    }
    this.applyVehicleCount(this.params.vehicleCount)
    this.applyPedestrianCount(this.params.pedestrianCount)

    setTimeout(() => {
      this.sendInit()
      this.sendDetector()
    }, 60)

    this.frameTimer = setInterval(() => this.step(), FRAME_MS)
    this.metricsTimer = setInterval(() => this.sendMetrics(), 1000)
  }

  private send(msg: ServerMessage): void {
    if (this.closed) return
    this.emit(JSON.stringify(msg))
  }

  private sendInit(): void {
    this.send({
      type: 'init',
      protocolVersion: PROTOCOL_VERSION,
      presets: MOCK_PRESETS,
      config: MOCK_CONFIG,
      weatherPresets: MOCK_WEATHER_PRESETS,
      params: this.params,
      status: this.status,
    })
  }

  private sendStatus(patch?: Partial<StatusPayload>): void {
    if (patch) this.status = { ...this.status, ...patch }
    this.send({ type: 'status', ...this.status })
  }

  private sendParams(): void {
    this.send({ type: 'params', params: this.params })
  }

  private sendDetector(): void {
    this.send({ ...this.detector })
  }

  /** 認識器の学習を模した進行。**実機の段階（収集 -> 保存 -> 学習 -> 完了）を */
  private startDetectorJob(request: DetectorRequest): void {
    const preset = MOCK_PRESETS.find((p) => p.id === request.presetId)
    const startedAt = performance.now()
    const collects = request.mode !== 'train'
    const trains = request.mode !== 'collect'

    const evaluates = collects && request.focusWeak
    this.detector = {
      ...makeIdleDetector(),
      state: evaluates ? 'evaluating' : collects ? 'collecting' : 'preparing',
      running: true,
      message: evaluates
        ? '（モック）いまの認識器の弱点を測っています'
        : collects
          ? `（モック）${preset?.name ?? 'マップ'} を走らせて教師データを集めています`
          : '（モック）保存済みの教師データを読み込んでいます',
      samples: request.samples,
      epochs: request.epochs,
      batches: 8,
      presetName: preset?.name ?? null,
      request,
    }
    this.sendStatus({
      simSuspended: true,
      learning: false,
      suspendReason: '（モック）認識器の学習中はシミュレーションを止めています',
      message: '（モック）認識器の学習を始めました',
    })
    this.sendDetector()

    let evaluated = evaluates ? 0 : MOCK_EVAL_SAMPLES
    let collected = collects ? 0 : request.samples
    let epoch = 0
    let batch = 0

    if (this.detectorTimer) clearInterval(this.detectorTimer)
    this.detectorTimer = setInterval(() => {
      const d = this.detector
      d.elapsedSec = (performance.now() - startedAt) / 1000

      if (evaluated < MOCK_EVAL_SAMPLES) {
        evaluated = Math.min(MOCK_EVAL_SAMPLES, evaluated + MOCK_EVAL_SAMPLES / 4)
        d.state = 'evaluating'
        d.collected = evaluated
        d.samples = MOCK_EVAL_SAMPLES
        d.progress = evaluated / MOCK_EVAL_SAMPLES
        d.message = `（モック）いまの認識器の弱点を測っています（${evaluated} / ${MOCK_EVAL_SAMPLES} 枚）`
        if (evaluated >= MOCK_EVAL_SAMPLES) {
          d.evaluation = makeMockEvaluation()
          d.samples = request.samples
        }
        this.sendDetector()
        return
      }

      if (collected < request.samples) {
        collected = Math.min(request.samples, collected + Math.ceil(request.samples / 24))
        d.state = 'collecting'
        d.collected = collected
        d.progress = collected / request.samples
        d.message = `（モック）教師データを集めています（${collected} / ${request.samples} 枚）`
        if (collected >= request.samples) {
          d.dataset = {
            samples: request.samples,
            objectCellRatio: 0.107,
            objectsPerImage: 5.1,
            classCounts: {
              TRAFFIC_LIGHT: Math.round(request.samples * 1.1),
              SPEED_SIGN: Math.round(request.samples * 0.25),
              VEHICLE: Math.round(request.samples * 0.75),
              OBSTACLE: Math.round(request.samples * 2.0),
              LANE: request.samples,
            },
            weatherCounts: request.weatherMix
              ? {
                  clear: Math.round(request.samples * 0.3),
                  drizzle: Math.round(request.samples * 0.12),
                  rain: Math.round(request.samples * 0.13),
                  fog: Math.round(request.samples * 0.33),
                  heavy_fog: Math.round(request.samples * 0.12),
                }
              : { clear: request.samples },
          }
          d.datasetFile = {
            exists: true,
            sizeBytes: request.samples * 1050,
            modifiedAt: new Date().toLocaleString('sv-SE'),
          }
          if (!trains) return this.finishDetectorJob('done', '（モック）教師データの収集が終わりました')
          d.state = 'training'
          d.progress = 0
          d.message = '（モック）認識器を学習しています'
        }
        return this.sendDetector()
      }

      if (!trains) return this.finishDetectorJob('done', '（モック）教師データの収集が終わりました')

      batch += 1
      if (batch > d.batches) {
        batch = 1
        epoch += 1
      }
      if (epoch === 0) epoch = 1
      d.state = 'training'
      d.epoch = epoch
      d.batch = batch
      d.progress = (epoch - 1 + batch / d.batches) / request.epochs
      d.message = `（モック）学習中（${epoch} / ${request.epochs} エポック）`
      if (batch === d.batches) {
        const t = epoch / Math.max(1, request.epochs)
        d.history = [
          ...d.history,
          { epoch, loss: 2.9 * Math.exp(-1.8 * t) + 0.15, valLoss: 3.4 * Math.exp(-1.5 * t) + 0.3 },
        ]
      }
      if (epoch >= request.epochs && batch >= d.batches) {
        d.paramCount = Math.round(152992 * request.width * request.width)
        d.model = {
          exists: true,
          filename: 'detector.keras',
          sizeBytes: Math.round(1965573 * request.width * request.width),
          modifiedAt: new Date().toLocaleString('sv-SE'),
          inUse: true,
        }
        return this.finishDetectorJob(
          'done',
          `（モック）学習が完了しました（${request.epochs} エポック）`,
        )
      }
      this.sendDetector()
    }, 250)
  }

  private finishDetectorJob(state: DetectorMessage['state'], message: string): void {
    if (this.detectorTimer) clearInterval(this.detectorTimer)
    this.detectorTimer = null
    this.detector = { ...this.detector, state, running: false, message, progress: 1 }
    this.sendDetector()
    this.sendStatus({
      simSuspended: false,
      learning: true,
      suspendReason: '',
      message,
    })
  }

  private sendMetrics(): void {
    if (!this.map) return
    this.updates += 1
    const p = this.progress
    const noise = () => (this.rng() - 0.5) * 2
    const metrics: MetricsMessage = {
      type: 'metrics',
      tick: this.tick,
      wallTime: (performance.now() - this.startedAt) / 1000,
      updates: this.updates,
      episodes: this.episodes,
      meanEpisodeReward: -80 + 120 * p + noise() * 9,
      meanEpisodeLength: 90 + 120 * p + noise() * 12,
      policyLoss: Math.max(0.001, 0.09 * (1 - p) + Math.abs(noise()) * 0.006),
      valueLoss: Math.max(0.01, 1.4 * (1 - p * 0.9) + Math.abs(noise()) * 0.08),
      entropy: Math.max(0.05, 1.35 - 0.85 * p + noise() * 0.03),
      approxKl: Math.max(0.0002, 0.014 * (1 - p * 0.6) + Math.abs(noise()) * 0.002),
      collisionRate: Math.max(0, 0.55 * (1 - p) + noise() * 0.03),
      // 学習が進むほど歩行者を轢かなくなる。衝突率の一部なので必ずそれ以下にする
      pedestrianCollisionRate: Math.max(0, 0.18 * (1 - p) + noise() * 0.01),
      goalRate: Math.max(0, Math.min(1, 0.05 + 0.85 * p + noise() * 0.04)),
      stepsPerSec: SIM_HZ * this.params.simSpeed + noise() * 0.4,
      signalViolations: 0,
      speedViolations: 0,
      laneDeviation: 0,
    }
    this.send(metrics)
  }

  /** マップ読込時に、信号の引き当て表を作る。 */
  private buildSignalIndex(): void {
    this.signalIndex.clear()
    this.signalSetback = []
    this.signLimits.clear()
    for (const sg of this.map?.signs ?? []) {
      const gx = sg.nodeId % GRID_N
      const gy = Math.floor(sg.nodeId / GRID_N)
      const tx = gx + Math.round(Math.cos(sg.heading))
      const ty = gy + Math.round(Math.sin(sg.heading))
      this.signLimits.set(`${gx},${gy}>${tx},${ty}`, sg.speedLimit)
    }
    const signals = this.map?.signals ?? []
    for (let i = 0; i < signals.length; i++) {
      const sg = signals[i]
      const gx = sg.nodeId % GRID_N
      const gy = Math.floor(sg.nodeId / GRID_N)
      const fx = gx - Math.round(Math.cos(sg.heading))
      const fy = gy - Math.round(Math.sin(sg.heading))
      this.signalIndex.set(`${fx},${fy}>${gx},${gy}`, i)
      this.signalSetback.push(Math.hypot(nodeX(gx) - sg.x, nodeY(gy) - sg.y))
    }
  }

  /** 灯色を simTime だけの関数として決める（backend/app/sim/signals.py と同じ式）。 */
  private computePhases(): number[] {
    const signals = this.map?.signals ?? []
    const out: number[] = new Array(signals.length)
    for (let i = 0; i < signals.length; i++) {
      const sg = signals[i]
      const offset = (sg.nodeId * 7919) % SIGNAL_CYCLE
      const t = (this.simTime + offset) % SIGNAL_CYCLE
      const local = sg.group === 0 ? t : (t + SIGNAL_HALF_CYCLE) % SIGNAL_CYCLE
      out[i] =
        local < SIGNAL_GREEN_SEC
          ? SIGNAL_GREEN
          : local < SIGNAL_GREEN_SEC + SIGNAL_YELLOW_SEC
            ? SIGNAL_YELLOW
            : SIGNAL_RED
    }
    return out
  }

  private makeVehicle(id: number): MockVehicle {
    const gx = Math.floor(this.rng() * GRID_N)
    const gy = Math.floor(this.rng() * GRID_N)
    const v: MockVehicle = {
      id,
      active: false,
      gx,
      gy,
      tx: gx,
      ty: gy,
      t: 0,
      x: nodeX(gx),
      y: nodeY(gy),
      heading: 0,
      speed: 0,
      steer: 0,
      collided: false,
      reachedGoal: false,
      speedLimit: 0,
      speedViolations: 0,
      overspeeding: false,
      goalGx: Math.floor(this.rng() * GRID_N),
      goalGy: Math.floor(this.rng() * GRID_N),
      routeTotal: 1,
      progress: 0,
      routeDirty: true,
      braking: false,
      turnSignal: 0,
    }
    chooseNext(v, this.rng)
    this.beginRoute(v)
    return v
  }

  /** 新しい目的地を決めた直後に呼ぶ。progress の分母を取り直す */
  private beginRoute(v: MockVehicle): void {
    v.routeTotal = Math.max(1, remainingDistance(v))
    v.progress = 0
    v.routeDirty = true
    v.speedViolations = 0
    v.overspeeding = false
  }

  private makePedestrian(id: number): MockPedestrian {
    const gx = Math.floor(this.rng() * GRID_N)
    const gy = Math.floor(this.rng() * GRID_N)
    const along = this.rng() < 0.5
    return {
      id,
      gx,
      gy,
      tx: along ? Math.min(GRID_N - 1, gx + 1) : gx,
      ty: along ? gy : Math.min(GRID_N - 1, gy + 1),
      t: this.rng(),
      side: this.rng() < 0.5 ? 1 : -1,
      cross: 0,
      crossing: false,
      stride: this.rng() * Math.PI * 2,
      speed: 1.1 + this.rng() * 0.5,
    }
  }

  private applyPedestrianCount(n: number): void {
    const want = Math.max(0, Math.min(MOCK_CONFIG.maxPedestrians ?? 64, Math.round(n)))
    while (this.pedestrians.length < want) {
      this.pedestrians.push(this.makePedestrian(this.pedestrians.length))
    }
    if (this.pedestrians.length > want) this.pedestrians.length = want
  }

  private stepPedestrian(p: MockPedestrian, dt: number): void {
    const ax = nodeX(p.gx)
    const ay = nodeY(p.gy)
    const bx = nodeX(p.tx)
    const by = nodeY(p.ty)
    const length = Math.max(1e-3, Math.hypot(bx - ax, by - ay))
    p.stride += p.speed * dt * 2

    if (p.crossing) {
      p.cross += (p.speed / (WALK_OFFSET_M * 2)) * dt
      if (p.cross >= 1) {
        p.cross = 0
        p.crossing = false
        p.side = -p.side
      }
      return
    }

    p.t += (p.speed * dt) / length
    if (p.t < 1) return

    p.t = 0
    p.gx = p.tx
    p.gy = p.ty
    if (this.rng() < 0.3) {
      p.crossing = true
      p.cross = 0
    }
    // 次の区間を選ぶ（格子なので上下左右のいずれか）
    const options: Array<[number, number]> = []
    if (p.gx > 0) options.push([p.gx - 1, p.gy])
    if (p.gx < GRID_N - 1) options.push([p.gx + 1, p.gy])
    if (p.gy > 0) options.push([p.gx, p.gy - 1])
    if (p.gy < GRID_N - 1) options.push([p.gx, p.gy + 1])
    const [nx, ny] = options[Math.floor(this.rng() * options.length)]
    p.tx = nx
    p.ty = ny
  }

  /**
   * どの車からも遠い歩行者を車の近くへ回す（実サーバーの `PedestrianCrowd.recycle`）。
   * これが無いと、格子全体に散った人がいつまでも画に入らない。
   */
  private recyclePedestrians(): void {
    const cars = this.vehicles.filter((v) => v.active)
    if (!cars.length) return
    for (const p of this.pedestrians) {
      const at = this.pedestrianAt(p)
      let nearest = Infinity
      for (const v of cars) nearest = Math.min(nearest, Math.hypot(v.x - at.x, v.y - at.y))
      if (nearest <= PEDESTRIAN_RECYCLE_M) continue
      const v = cars[Math.floor(this.rng() * cars.length)]
      p.gx = Math.max(0, Math.min(GRID_N - 1, Math.round((v.x + GRID_HALF) / GRID_SPACING)))
      p.gy = Math.max(0, Math.min(GRID_N - 1, Math.round((v.y + GRID_HALF) / GRID_SPACING)))
      p.tx = Math.max(0, Math.min(GRID_N - 1, p.gx + (this.rng() < 0.5 ? 1 : -1)))
      p.ty = p.gy
      if (p.tx === p.gx) {
        p.tx = p.gx
        p.ty = Math.max(0, Math.min(GRID_N - 1, p.gy + 1))
      }
      p.t = this.rng()
      p.crossing = false
      p.cross = 0
    }
  }

  private pedestrianAt(p: MockPedestrian): NpcPedestrianState {
    const ax = nodeX(p.gx)
    const ay = nodeY(p.gy)
    const bx = nodeX(p.tx)
    const by = nodeY(p.ty)
    const length = Math.max(1e-3, Math.hypot(bx - ax, by - ay))
    const dx = (bx - ax) / length
    const dy = (by - ay) / length
    const side = p.crossing ? p.side * (1 - 2 * p.cross) : p.side
    const offset = side * WALK_OFFSET_M
    return {
      id: p.id,
      x: ax + dx * (p.t * length) - dy * offset,
      y: ay + dy * (p.t * length) + dx * offset,
      heading: p.crossing ? Math.atan2(dx * -p.side, dy * p.side) : Math.atan2(dy, dx),
      stride: p.stride % (Math.PI * 2),
      crossing: p.crossing,
    }
  }

  private pedestrianStates(): NpcPedestrianState[] {
    return this.pedestrians.map((p) => this.pedestrianAt(p))
  }

  private applyVehicleCount(n: number): void {
    for (const v of this.vehicles) {
      const shouldBeActive = v.id < n
      if (shouldBeActive && !v.active) {
        v.active = true
        v.routeDirty = true
      } else if (!shouldBeActive && v.active) {
        v.active = false
      }
    }
  }

  /**
   * 次の交差点で曲がるなら方向指示器を出す。-1=左 / 0=消灯 / +1=右。
   * **モックの簡易判定**で、格子の目的地が横にずれていれば曲がるものとして扱う。
   */
  private turnSignalFor(v: MockVehicle): number {
    const rest = Math.hypot(nodeX(v.tx) - v.x, nodeY(v.ty) - v.y)
    if (rest > TURN_LOOKAHEAD_M) return 0
    const dirX = Math.sign(v.tx - v.gx)
    const dirY = Math.sign(v.ty - v.gy)
    const turnY = dirX !== 0 ? Math.sign(v.goalGy - v.ty) : 0
    const turnX = dirY !== 0 ? Math.sign(v.goalGx - v.tx) : 0
    // ENU は反時計回りが正なので、外積が正なら左へ曲がる
    const cross = dirX * turnY - dirY * turnX
    return cross > 0 ? -1 : cross < 0 ? 1 : 0
  }

  private stepVehicle(v: MockVehicle, dt: number): void {
    if (!v.active) return

    const ax = nodeX(v.gx)
    const ay = nodeY(v.gy)
    const bx = nodeX(v.tx)
    const by = nodeY(v.ty)
    const segLen = Math.hypot(bx - ax, by - ay)

    if (segLen < 1e-6) {
      v.goalGx = Math.floor(this.rng() * GRID_N)
      v.goalGy = Math.floor(this.rng() * GRID_N)
      chooseNext(v, this.rng)
      this.beginRoute(v)
      v.reachedGoal = true
      this.episodes += 1
      return
    }

    let targetSpeed = this.params.maxSpeed * (0.45 + 0.55 * Math.sin(Math.PI * v.t))

    const limit = this.signLimits.get(`${v.gx},${v.gy}>${v.tx},${v.ty}`)
    if (limit !== undefined) v.speedLimit = limit
    if (this.params.obeySpeedSigns && v.speedLimit > 0) {
      targetSpeed = Math.min(targetSpeed, v.speedLimit)
    }

    let stopT = 1
    if (this.params.obeySignals) {
      const idx = this.signalIndex.get(`${v.gx},${v.gy}>${v.tx},${v.ty}`)
      const phase = idx === undefined ? SIGNAL_GREEN : this.phases[idx]
      if (idx !== undefined && phase !== SIGNAL_GREEN) {
        const setback = this.signalSetback[idx]
        const stopDist = (1 - v.t) * segLen - setback
        const canStop = (v.speed * v.speed) / (2 * BRAKE_ACCEL) <= Math.max(0, stopDist)
        if (phase === SIGNAL_RED || canStop) {
          stopT = Math.max(0, 1 - setback / segLen)
          targetSpeed = Math.min(targetSpeed, Math.sqrt(2 * BRAKE_ACCEL * Math.max(0, stopDist)))
        }
      }
    }

    const playerGap = this.playerGap(v)
    if (playerGap < Infinity) {
      const room = Math.max(0, playerGap - PLAYER_MARGIN_M)
      targetSpeed = Math.min(targetSpeed, Math.sqrt(2 * BRAKE_ACCEL * room))
    }

    // 実機と同じく「指令が減速側か」で決める（実測の加速度では見ない）
    v.braking = targetSpeed < v.speed - 0.2
    v.turnSignal = this.turnSignalFor(v)

    v.speed += (targetSpeed - v.speed) * Math.min(1, dt * 1.6)

    const over = v.speedLimit > 0 && v.speed > v.speedLimit
    if (over && !v.overspeeding) v.speedViolations += 1
    v.overspeeding = over

    const advance = (v.speed * dt) / segLen
    v.t += advance
    if (stopT < 1 && v.t > stopT) {
      v.t = stopT
      v.speed = 0
    }

    const prevHeading = v.heading
    v.heading = Math.atan2(by - ay, bx - ax)
    const dh = Math.atan2(Math.sin(v.heading - prevHeading), Math.cos(v.heading - prevHeading))
    v.steer += (Math.max(-0.5, Math.min(0.5, dh * 4)) - v.steer) * 0.25

    if (v.t >= 1) {
      v.t = 0
      v.gx = v.tx
      v.gy = v.ty
      v.x = nodeX(v.gx)
      v.y = nodeY(v.gy)
      if (v.gx === v.goalGx && v.gy === v.goalGy) {
        // 徴用中の 1 台は乗降地点で止める（実機と同じく respawn しない）
        if (v.id === this.taxi.vehicleId && this.taxi.phase !== 'idle') {
          v.speed = 0
          v.reachedGoal = true
          return
        }
        v.reachedGoal = true
        this.episodes += 1
        v.goalGx = Math.floor(this.rng() * GRID_N)
        v.goalGy = Math.floor(this.rng() * GRID_N)
        chooseNext(v, this.rng)
        this.beginRoute(v)
      } else {
        v.reachedGoal = false
        chooseNext(v, this.rng)
        v.routeDirty = true
      }
      return
    }

    v.x = ax + (bx - ax) * v.t
    v.y = ay + (by - ay) * v.t
    v.reachedGoal = false
  }

  /** 前方の進路上にいる徒歩キャラまでの距離 [m]。いなければ Infinity。 */
  private playerGap(v: MockVehicle): number {
    const at = this.player
    if (!at) return Infinity
    const cos = Math.cos(v.heading)
    const sin = Math.sin(v.heading)
    const dx = at[0] - v.x
    const dy = at[1] - v.y
    const lon = dx * cos + dy * sin
    const lat = -dx * sin + dy * cos
    if (lon <= 0 || lon > PLAYER_RANGE_M) return Infinity
    if (Math.abs(lat) > PLAYER_HALF_WIDTH_M) return Infinity
    return lon
  }

  /** 同じ区間を走る前走車に追突しないよう、後続の位置を後ろへ詰める。 */
  private applyQueueing(): void {
    const lanes = new Map<string, MockVehicle[]>()
    for (const v of this.vehicles) {
      if (!v.active) continue
      const key = `${v.gx},${v.gy}>${v.tx},${v.ty}`
      const list = lanes.get(key)
      if (list) list.push(v)
      else lanes.set(key, [v])
    }

    for (const list of lanes.values()) {
      if (list.length < 2) continue
      const head = list[0]
      const ax = nodeX(head.gx)
      const ay = nodeY(head.gy)
      const bx = nodeX(head.tx)
      const by = nodeY(head.ty)
      const segLen = Math.hypot(bx - ax, by - ay)
      if (segLen < 1e-6) continue

      const gap = CAR_GAP_M / segLen
      list.sort((a, b) => b.t - a.t)
      for (let i = 1; i < list.length; i++) {
        const ahead = list[i - 1]
        const v = list[i]
        const limit = ahead.t - gap
        if (v.t > limit) {
          v.t = Math.max(0, limit)
          v.speed = Math.min(v.speed, ahead.speed)
          v.x = ax + (bx - ax) * v.t
          v.y = ay + (by - ay) * v.t
        }
      }
    }
  }

  private detectCollisions(): void {
    for (const v of this.vehicles) v.collided = false
    const active = this.vehicles.filter((v) => v.active)
    for (let i = 0; i < active.length; i++) {
      for (let j = i + 1; j < active.length; j++) {
        const a = active[i]
        const b = active[j]
        if (Math.hypot(a.x - b.x, a.y - b.y) < 4.5) {
          a.collided = true
          b.collided = true
        }
      }
      for (const o of this.obstacles) {
        const a = active[i]
        if (Math.hypot(a.x - o.x, a.y - o.y) < o.radius + 2.2) a.collided = true
      }
    }
  }

  private step(): void {
    if (this.closed || !this.map) return

    const dt = (FRAME_MS / 1000) * this.params.simSpeed
    this.tick += 1
    this.simTime += dt
    this.progress = Math.min(1, this.progress + dt * 0.004)

    // 届かなくなった徒歩キャラは街から消す（見えない人の前で止まり続ける）
    if (this.player && Date.now() - this.playerAt >= PLAYER_POSE_TTL_MS) {
      this.player = null
      this.playerAt = 0
    }

    for (const v of this.vehicles) this.stepVehicle(v, dt)
    for (const p of this.pedestrians) this.stepPedestrian(p, dt)
    if (this.tick % Math.round(SIM_HZ * 2) === 0) this.recyclePedestrians()
    this.detectCollisions()
    this.stepTaxi()

    if (this.status.renderPaused) return

    const vehicles: VehicleState[] = this.vehicles.map((v) => {
      const state: VehicleState = {
        id: v.id,
        active: v.active,
        x: v.x,
        y: v.y,
        heading: v.heading,
        speed: v.active ? v.speed : 0,
        steer: v.steer,
        collided: v.collided,
        reachedGoal: v.reachedGoal,
        goal: [nodeX(v.goalGx), nodeY(v.goalGy)],
        progress: v.progress,
        signalViolations: 0,
        laneDepartures: 0,
        speedLimit: v.speedLimit,
        speedViolations: v.speedViolations,
        braking: v.braking,
        turnSignal: v.turnSignal,
      }
      if (v.active && v.routeDirty) {
        state.route = buildRoute(v)
        v.routeDirty = false
      }
      return state
    })

    const frame: FrameMessage = {
      type: 'frame',
      tick: this.tick,
      simTime: this.simTime,
      vehicles,
      obstacles: this.obstacles,
    }
    if (this.pedestrians.length) frame.pedestrians = this.pedestrianStates()
    if (this.map?.signals?.length) frame.signals = this.computePhases()
    frame.weather = this.currentWeather()
    const detections = this.buildDetections()
    if (detections) frame.detections = detections
    this.send(frame)
  }

  private sendTaxi(force = false): void {
    const now = performance.now()
    if (!force && now - this.taxiSentAt < 1000) return
    this.taxiSentAt = now
    this.send({ ...this.taxi, route: [...(this.taxi.route ?? [])] })
  }

  /** 配車の段階を進める。**実機と同じく、到着は「残り距離」で判定する。** */
  private stepTaxi(): void {
    if (this.taxi.phase === 'idle') return
    const v = this.vehicles[this.taxi.vehicleId]
    if (!v || !v.active) {
      this.cancelTaxi('（モック）配車していた車両がいなくなりました')
      return
    }

    const target = this.taxi.phase === 'riding' ? this.taxi.dropoff : this.taxi.pickup
    if (!target) return
    const remaining = remainingDistance(v)
    this.taxi.remainingDistanceM = remaining
    this.taxi.etaSeconds =
      this.taxi.phase === 'waiting' || this.taxi.phase === 'arrived'
        ? 0
        : remaining / Math.max(MOCK_TAXI_MIN_SPEED, v.speed)

    const near = Math.hypot(v.x - target[0], v.y - target[1]) <= MOCK_TAXI_ARRIVE_M
    const before = this.taxi.phase
    if (this.taxi.phase === 'approaching' && near) {
      v.speed = 0
      this.taxi.phase = 'waiting'
      this.taxi.message = '（モック）乗車地点に到着しました。[Enter] で乗車できます'
    } else if (this.taxi.phase === 'riding' && near) {
      v.speed = 0
      this.taxi.phase = 'arrived'
      this.taxi.message = '（モック）目的地に到着しました。[Enter] で降車できます'
    }
    if (this.taxi.phase === 'waiting' || this.taxi.phase === 'arrived') v.speed = 0
    this.sendTaxi(before !== this.taxi.phase)
  }

  private cancelTaxi(message: string): void {
    const v = this.vehicles[this.taxi.vehicleId]
    if (v) {
      v.goalGx = Math.floor(this.rng() * GRID_N)
      v.goalGy = Math.floor(this.rng() * GRID_N)
      chooseNext(v, this.rng)
      v.routeDirty = true
    }
    this.taxi = {
      ...MOCK_IDLE_TAXI,
      routeRevision: this.taxi.routeRevision + 1,
      message,
    }
    this.sendStatus({ taxiVehicleId: -1 })
    this.sendTaxi(true)
  }

  /** いま効いている天候。視程の式は `percep/weather.py` の `visibility_m` に合わせる。 */
  private currentWeather(): WeatherState {
    let rain = this.params.weatherRain
    let fog = this.params.weatherFog
    if (this.params.weatherAuto) {
      const phase = (this.simTime / MOCK_WEATHER_PERIOD_SEC) % 1
      rain = Math.max(0, Math.sin(2 * Math.PI * phase)) ** 2
      fog = Math.max(0, Math.sin(2 * Math.PI * (phase - 0.5))) ** 2
    }
    const far = 120
    const visibility = fog <= 1e-3 ? far : far * (MOCK_MIN_VISIBILITY_M / far) ** (fog * fog)
    return { rain, fog, visibility }
  }

  /** 認識結果のダミー。**運転席カメラのボックスと路面の車線オーバーレイを */
  private buildDetections(): Record<string, Detection[]> | null {
    const phases = this.map?.signals?.length ? this.computePhases() : null
    const out: Record<string, Detection[]> = {}
    for (const v of this.vehicles) {
      if (!v.active) continue
      const lateral = Math.sin(this.simTime * 0.4 + v.id) * 0.6
      const lane: Detection = {
        cls: DET_LANE,
        box: [0.2, 0.55, 0.8, 0.95],
        conf: 0.9,
        distance: 25,
        lateral,
        lanePoints: [0, 5, 10, 15, 20, 25].map(
          (fx) => [fx, -lateral] as [number, number],
        ),
      }
      const light: Detection = {
        cls: DET_TRAFFIC_LIGHT,
        box: [0.44, 0.3, 0.56, 0.4],
        conf: 0.8,
        phase: phases ? phases[v.id % phases.length] : SIGNAL_GREEN,
        distance: 30,
      }
      const dets: Detection[] = [lane, light]
      // 近くを歩いている人がいれば、擬似的な検出枠を 1 つ出す
      const near = this.pedestrians.find(
        (p) => {
          const state = this.pedestrianAt(p)
          return Math.hypot(state.x - v.x, state.y - v.y) < 24
        },
      )
      if (near) {
        dets.push({
          cls: DET_PEDESTRIAN,
          box: [0.56, 0.42, 0.65, 0.72],
          conf: 0.72,
          distance: 18,
        })
      }
      out[String(v.id)] = dets
    }
    return Object.keys(out).length ? out : null
  }

  handle(json: string): void {
    if (this.closed) return
    let msg: ClientMessage
    try {
      msg = JSON.parse(json) as ClientMessage
    } catch {
      this.send({ type: 'error', code: 'INVALID_MESSAGE', message: 'JSON を解析できません' })
      return
    }

    switch (msg.type) {
      case 'load_map': {
        if (this.detector.running) {
          this.send({
            type: 'error',
            code: 'DETECTOR_TRAINING',
            message: '（モック）認識器の学習中はエリアを変えられません',
          })
          return
        }
        const preset = MOCK_PRESETS.find((p) => p.id === msg.presetId)
        if (!preset) {
          this.send({
            type: 'error',
            code: 'MAP_LOAD_FAILED',
            message: `未知のプリセット: ${msg.presetId}`,
          })
          return
        }
        this.sendStatus({
          state: 'loading_map',
          mapLoaded: false,
          presetId: preset.id,
          message: '（モック）地図データを取得しています…',
        })
        if (this.loadTimer) clearTimeout(this.loadTimer)
        this.loadTimer = setTimeout(() => {
          if (this.closed) return
          this.map = buildMockMap(preset.id, preset.name)
          this.buildSignalIndex()
          this.tick = 0
          this.simTime = 0
          for (const v of this.vehicles) v.routeDirty = true
          this.send(this.map)
          this.sendStatus({
            state: 'running',
            mapLoaded: true,
            presetId: preset.id,
            message: `（モック）${preset.name} を読み込みました`,
          })
        }, 1400)
        break
      }

      case 'set_params': {
        this.params = { ...this.params, ...msg.params }
        this.params.vehicleCount = Math.max(
          1,
          Math.min(MOCK_CONFIG.maxVehicles, Math.round(this.params.vehicleCount)),
        )
        this.params.simSpeed = Math.max(0.25, Math.min(4, this.params.simSpeed))
        this.params.rewardOverspeed = Math.max(
          -1000,
          Math.min(0, this.params.rewardOverspeed),
        )
        this.params.pedestrianCount = Math.max(
          0,
          Math.min(
            MOCK_CONFIG.maxPedestrians ?? 64,
            Math.round(this.params.pedestrianCount),
          ),
        )
        this.applyVehicleCount(this.params.vehicleCount)
        this.applyPedestrianCount(this.params.pedestrianCount)
        this.sendParams()
        break
      }

      case 'spawn_vehicle': {
        if (!this.map) {
          this.sendStatus({ message: 'マップが読み込まれていません' })
          return
        }
        const slot = this.vehicles.find((v) => !v.active)
        if (!slot) {
          this.sendStatus({ message: '空きスロットがありません' })
          return
        }
        const gx = clampGrid(Math.round((msg.x + GRID_HALF) / GRID_SPACING))
        const gy = clampGrid(Math.round((msg.y + GRID_HALF) / GRID_SPACING))
        slot.gx = gx
        slot.gy = gy
        slot.tx = gx
        slot.ty = gy
        slot.t = 0
        slot.x = nodeX(gx)
        slot.y = nodeY(gy)
        slot.speed = 0
        slot.active = true
        slot.routeDirty = true
        slot.goalGx = Math.floor(this.rng() * GRID_N)
        slot.goalGy = Math.floor(this.rng() * GRID_N)
        chooseNext(slot, this.rng)
        this.params.vehicleCount = this.vehicles.filter((v) => v.active).length
        this.sendParams()
        this.sendStatus({ message: `（モック）車両 #${slot.id} をスポーンしました` })
        break
      }

      case 'despawn_vehicle': {
        const v = this.vehicles[msg.id]
        if (v) v.active = false
        this.params.vehicleCount = Math.max(1, this.vehicles.filter((x) => x.active).length)
        this.sendParams()
        break
      }

      case 'add_obstacle': {
        this.obstacles.push({
          id: this.obstacleSeq++,
          x: msg.x,
          y: msg.y,
          radius: msg.radius,
        })
        this.sendStatus({ message: `（モック）障害物を設置しました（${this.obstacles.length}）` })
        break
      }

      case 'remove_obstacle': {
        this.obstacles = this.obstacles.filter((o) => o.id !== msg.id)
        break
      }

      case 'clear_obstacles': {
        this.obstacles = []
        this.sendStatus({ message: '（モック）障害物をすべて削除しました' })
        break
      }

      case 'set_render_paused': {
        this.sendStatus({
          renderPaused: msg.paused,
          message: msg.paused
            ? '（モック）描画を一時停止しました。学習は継続中です'
            : '（モック）描画を再開しました',
        })
        break
      }

      case 'reset_episode': {
        for (const v of this.vehicles) {
          const gx = Math.floor(this.rng() * GRID_N)
          const gy = Math.floor(this.rng() * GRID_N)
          v.gx = gx
          v.gy = gy
          v.tx = gx
          v.ty = gy
          v.t = 0
          v.x = nodeX(gx)
          v.y = nodeY(gy)
          v.speed = 0
          v.goalGx = Math.floor(this.rng() * GRID_N)
          v.goalGy = Math.floor(this.rng() * GRID_N)
          v.speedLimit = 0
          chooseNext(v, this.rng)
          this.beginRoute(v)
        }
        this.episodes += 1
        this.sendStatus({ message: '（モック）エピソードをリセットしました' })
        break
      }

      case 'save_checkpoint':
        this.sendStatus({ message: '（モック）チェックポイントを保存しました' })
        break

      case 'load_checkpoint':
        this.sendStatus({ message: '（モック）チェックポイントを読み込みました' })
        break

      case 'reset_policy':
        this.progress = 0
        this.updates = 0
        this.episodes = 0
        this.sendStatus({ message: '（モック）ポリシーを初期化しました' })
        break

      case 'start_detector_training': {
        if (this.detector.running) {
          this.sendStatus({ message: '（モック）すでに学習を実行中です' })
          return
        }
        this.startDetectorJob(msg.request)
        break
      }

      case 'cancel_detector_training': {
        if (!this.detector.running) {
          this.sendStatus({ message: '（モック）実行中の学習はありません' })
          return
        }
        this.finishDetectorJob('cancelled', '（モック）学習を中断しました')
        break
      }

      case 'set_app_mode': {
        const practical = msg.mode === 'taxi'
        if (practical === (this.status.practicalMode ?? false)) return
        if (!practical) this.cancelTaxi('（モック）開発モードに戻したため配車を終了しました')
        this.sendStatus({
          practicalMode: practical,
          learning: !practical,
          message: practical
            ? '（モック）実用モードに入りました。学習は止まります'
            : '（モック）開発モードに戻りました',
        })
        this.sendTaxi(true)
        break
      }

      case 'request_taxi': {
        if (!this.map) {
          this.sendStatus({ message: '（モック）先にエリアを読み込んでください' })
          return
        }
        if (this.taxi.phase !== 'idle') {
          this.sendStatus({ message: '（モック）すでに配車中です' })
          return
        }
        const pick = snapToGrid(msg.pickup[0], msg.pickup[1])
        const drop = snapToGrid(msg.dropoff[0], msg.dropoff[1])
        if (
          Math.hypot(drop.point[0] - pick.point[0], drop.point[1] - pick.point[1]) <
          MOCK_TAXI_MIN_TRIP_M
        ) {
          this.sendStatus({ message: '（モック）乗車地点と降車地点が近すぎます' })
          return
        }
        let slot = -1
        let best = Infinity
        for (const v of this.vehicles) {
          if (!v.active) continue
          const d = Math.hypot(v.x - pick.point[0], v.y - pick.point[1])
          if (d < best) {
            best = d
            slot = v.id
          }
        }
        if (slot < 0) {
          this.sendStatus({ message: '（モック）配車できる車両がいません' })
          return
        }
        const v = this.vehicles[slot]
        v.goalGx = pick.gx
        v.goalGy = pick.gy
        chooseNext(v, this.rng)
        v.routeDirty = true
        this.taxi = {
          ...MOCK_IDLE_TAXI,
          phase: 'approaching',
          vehicleId: slot,
          pickup: pick.point,
          dropoff: drop.point,
          route: buildRoute(v),
          routeRevision: this.taxi.routeRevision + 1,
          remainingDistanceM: remainingDistance(v),
          // 最初の 1 通から出しておく（次の step まで「まもなく」と出てしまう）
          etaSeconds: remainingDistance(v) / MOCK_TAXI_MIN_SPEED,
          message: `（モック）車両 #${slot} が迎えに向かっています`,
        }
        this.sendStatus({ taxiVehicleId: slot })
        this.sendTaxi(true)
        break
      }

      case 'board_taxi': {
        if (this.taxi.phase !== 'waiting' && this.taxi.phase !== 'approaching') return
        const v = this.vehicles[this.taxi.vehicleId]
        const drop = this.taxi.dropoff
        if (!v || !drop) return
        const snapped = snapToGrid(drop[0], drop[1])
        v.goalGx = snapped.gx
        v.goalGy = snapped.gy
        chooseNext(v, this.rng)
        v.routeDirty = true
        this.taxi = {
          ...this.taxi,
          phase: 'riding',
          route: buildRoute(v),
          routeRevision: this.taxi.routeRevision + 1,
          message: '（モック）目的地へ向かっています',
        }
        this.sendTaxi(true)
        break
      }

      case 'alight_taxi': {
        if (this.taxi.phase !== 'riding' && this.taxi.phase !== 'arrived') return
        this.cancelTaxi('（モック）降車しました')
        break
      }

      case 'cancel_taxi': {
        if (this.taxi.phase === 'idle') return
        const v = this.vehicles[this.taxi.vehicleId]
        if (msg.halt && v) v.speed = 0
        this.cancelTaxi(
          msg.halt
            ? '（モック）緊急停止しました。自動運転を終了します'
            : '（モック）配車を取り消しました',
        )
        break
      }

      case 'player_pose': {
        // 乗車中は車内にいるので歩行者として扱わない（実機と同じ）
        const onboard = this.taxi.phase === 'riding' || this.taxi.phase === 'arrived'
        this.player = msg.at && !onboard ? [msg.at[0], msg.at[1]] : null
        this.playerAt = this.player ? Date.now() : 0
        break
      }

      case 'ping':
        this.send({ type: 'pong', t: Date.now() })
        break

      default:
        this.send({ type: 'error', code: 'INVALID_MESSAGE', message: '未知のメッセージ種別です' })
        break
    }
  }

  close(): void {
    this.closed = true
    if (this.frameTimer) clearInterval(this.frameTimer)
    if (this.metricsTimer) clearInterval(this.metricsTimer)
    if (this.loadTimer) clearTimeout(this.loadTimer)
    if (this.detectorTimer) clearInterval(this.detectorTimer)
    this.frameTimer = null
    this.metricsTimer = null
    this.loadTimer = null
    this.detectorTimer = null
  }
}

/** connection.ts から呼ばれる。実 WebSocket と同じ Transport の形を返す */
export function createMockTransport(onMessage: (json: string) => void): Transport {
  console.info(
    '[mockServer] 開発用モックに接続しました（?mock=1）。本番では実サーバーに接続してください。',
  )
  const server = new MockServer(onMessage)
  return {
    send: (json) => server.handle(json),
    close: () => server.close(),
  }
}
