/**
 * 開発用のモックサーバー（ブラウザ内で動く偽 WebSocket サーバー）。
 *
 * ★ これは開発補助です。本番では必ず実サーバー（FastAPI / ws://host/ws）に繋いでください。
 *   有効になるのは `import.meta.env.DEV` かつ URL に `?mock=1` が付いているときだけで、
 *   connection.ts から動的 import されるため本番バンドルには載りません。
 *
 * 目的:
 *   バックエンドが未完成でも、3D 描画・フレーム補間・パネル操作を目視確認できるようにする。
 *   docs/protocol.md v1 に準拠したメッセージだけを流す。
 *
 * 中身:
 *   - 100m 間隔の格子状道路網（9x9 ノード）と、街区に並べた建物
 *   - すべての交差点に信号機（backend の SIGNALS_AT_ALL_INTERSECTIONS = True と同じ扱い）
 *   - 車両は格子上を貪欲探索で目的地へ向かい、赤信号では停止線の手前で止まる
 *   - frame は 20Hz、metrics は 1Hz
 *
 * ★ 実バックエンドと**同じ形のメッセージ**しか作らないこと。
 *   台数・観測次元・信号のサイクルなどの数値も backend/app/config.py と
 *   docs/protocol.md に合わせる。ここがずれるとモックで確認した意味が無くなる。
 */

import type {
  ClientMessage,
  FrameMessage,
  MapBuilding,
  MapEdge,
  MapMessage,
  MapNode,
  MapPreset,
  MapSign,
  MapSignal,
  MetricsMessage,
  ObstacleState,
  ServerMessage,
  SimConfig,
  SimParams,
  StatusPayload,
  Vec2,
  VehicleState,
} from '../types/protocol'
import {
  PROTOCOL_VERSION,
  SIGNAL_GREEN,
  SIGNAL_RED,
  SIGNAL_YELLOW,
} from '../types/protocol'
import type { Transport } from './connection'

// ---------------------------------------------------------------------------
// 決定的な擬似乱数（毎回同じ街ができるように）
// ---------------------------------------------------------------------------

function makeRng(seed: number) {
  let s = seed >>> 0
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 4294967296
  }
}

// ---------------------------------------------------------------------------
// 定数
// ---------------------------------------------------------------------------

const GRID_N = 9 // ノード数（片側）
const GRID_SPACING = 100 // m
const GRID_HALF = ((GRID_N - 1) * GRID_SPACING) / 2 // 400m
const ROAD_WIDTH = 7.0
const SIM_HZ = 20
const FRAME_MS = 1000 / SIM_HZ

// ---- 信号機（backend/app/map/loader.py・app/sim/signals.py と同じ値） ----

/** 交差点とみなす最小の接続本数（loader.SIGNAL_MIN_STREETS） */
const SIGNAL_MIN_STREETS = 3
/** 停止線を交差点端からどれだけ手前に引くか（横断歩道 4m + 余裕 1m） */
const SIGNAL_SETBACK_EXTRA_M = 4.0 + 1.0
/** 青 25 / 黄 3 / 全赤 2 秒。半サイクルで交差方向と入れ替わる */
const SIGNAL_GREEN_SEC = 25
const SIGNAL_YELLOW_SEC = 3
const SIGNAL_ALL_RED_SEC = 2
const SIGNAL_HALF_CYCLE = SIGNAL_GREEN_SEC + SIGNAL_YELLOW_SEC + SIGNAL_ALL_RED_SEC // 30
const SIGNAL_CYCLE = SIGNAL_HALF_CYCLE * 2 // 60

// ---- 最高速度標識（backend/app/config.py・app/map/loader.py と同じ値） ----

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
    // ★ 中心座標だけは実バックエンドの presets.py と一致させること。
    //   食い違うと、配色の切り替え時刻（autoTheme）の検証がモックと実機でずれる。
    //   一方 radiusM は 400 のまま：実プリセットは 6,000m（12km 四方）だが、
    //   モックは**合成した小さな格子を返すだけ**で radiusM をジオメトリに使わない。
    //   ここだけ 6000 にすると、返す地図の広さと申告値が食い違う嘘になる。
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

// 実バックエンドの config.py と揃える（MAX_VEHICLES = 64 / OBS_DIM = 56）。
// ここを小さいままにすると、64 台での描画負荷・色の見分け・Chip 列といった
// 「いま一番確認したいもの」がモックで再現できなくなる。
const MOCK_CONFIG: SimConfig = {
  maxVehicles: 64,
  simHz: SIM_HZ,
  obsDim: 56,
  actionDim: 2,
}

const DEFAULT_PARAMS: SimParams = {
  vehicleCount: 4,
  simSpeed: 1,
  learningRate: 3e-4,
  gamma: 0.99,
  clipRange: 0.2,
  entropyCoef: 0.01,
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
}

// ---------------------------------------------------------------------------
// マップ生成
// ---------------------------------------------------------------------------

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
    // 中間点を入れて、リボン生成側の法線平均処理が動くことを確認できるようにする
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
      // 3 本おきに幹線道路（車線数を増やす）
      const majorX = gy % 3 === 0
      const majorY = gx % 3 === 0
      if (gx + 1 < GRID_N) pushEdge(nodeId(gx, gy), nodeId(gx + 1, gy), majorX ? 4 : 2)
      if (gy + 1 < GRID_N) pushEdge(nodeId(gx, gy), nodeId(gx, gy + 1), majorY ? 4 : 2)
    }
  }

  // 街区の中に建物を並べる
  const buildings: MapBuilding[] = []
  let buildingId = 0
  const margin = 10 // 道路から離す距離
  for (let gy = 0; gy + 1 < GRID_N; gy++) {
    for (let gx = 0; gx + 1 < GRID_N; gx++) {
      const x0 = nodeX(gx) + margin
      const y0 = nodeY(gy) + margin
      const inner = GRID_SPACING - margin * 2
      const sub = 3
      const cell = inner / sub
      for (let sy = 0; sy < sub; sy++) {
        for (let sx = 0; sx < sub; sx++) {
          if (rng() < 0.18) continue // 空き地
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
          // 中心に近いほど高い（都心っぽさを出す）
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

/**
 * 交差点ごとに、進入路 1 本につき 1 基の信号機を作る。
 *
 * backend/app/map/loader.py の `_build_signals` と同じ考え方:
 *   - 接続する道路が 3 本以上のノードだけを交差点とみなす（格子の角は 2 本なので除外）
 *   - 日本の信号機は進入車両に正対するので、灯器の姿勢は進入方向で決まる
 *   - 停止線は交差点端から「横断歩道 + 余裕」だけ手前に引く
 *   - group は進入方向の**軸**で 0 / 1 に分け、直交する流れが同時に青にならないようにする
 *
 * 9x9 の格子では角 4 か所を除く 77 交差点に、合計 280 基できる（銀座の 261 基と同程度）。
 */
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

    // 停止線の位置は交差する道路の広さで決まる
    const halfWidth = Math.max(...around.map((e) => e.width)) / 2
    const setback = halfWidth + SIGNAL_SETBACK_EXTRA_M

    // 隣ノードごとに 1 本の進入路（モックの辺はすべて双方向）
    const approaches = new Map<number, { heading: number; x: number; y: number; width: number }>()
    for (const e of around) {
      const neighbourId = e.u === node.id ? e.v : e.u
      if (approaches.has(neighbourId)) continue
      // nodes は id 昇順に詰めてあるので添字がそのまま id になる
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

/**
 * 規制速度が変わる進入口に、最高速度標識を 1 基ずつ立てる。
 *
 * backend/app/map/loader.py の `_build_speed_signs` と同じ考え方:
 *   - OSM の traffic_sign タグは当てにならないので、道路（エッジ）の規制速度から作る
 *   - そのノードへ入ってくる**別の**道路と規制速度が同じなら置かない
 *     （同じ数字の標識を並べても情報が増えないため。規制が変わる地点に置く運用と同じ）
 *   - 支柱は進行方向の**左側**の路端に立てる（左側通行）
 *   - `heading` は「その標識が規制する側の進行方向」。標示板は heading + PI を向く
 *
 * モックの格子は幹線（16.7 m/s）と生活道路（11.1 m/s）が 3 本おきに交わるので、
 * その交点だけに標識が立つ。
 */
function buildMockSpeedSigns(nodes: MapNode[], edges: MapEdge[]): MapSign[] {
  // ノードへ入ってくる有向エッジ（到着ノード -> [エッジ id, 規制速度]）
  const arriving = new Map<number, Array<[number, number]>>()
  // ノードから出ていく有向エッジ（出発ノード -> [エッジ, 到着ノード]）
  const leaving = new Map<number, Array<[MapEdge, number]>>()
  const pushTo = <T>(m: Map<number, T[]>, key: number, value: T) => {
    const list = m.get(key)
    if (list) list.push(value)
    else m.set(key, [value])
  }
  for (const e of edges) {
    // モックの辺はすべて双方向なので、両向きを進入路・退出路として登録する
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
      // 同じエッジの逆走（U ターン）は比較対象にしない
      const incoming = (arriving.get(node.id) ?? [])
        .filter(([otherId]) => otherId !== edge.id)
        .map(([, limit]) => limit)
      if (
        incoming.length > 0 &&
        incoming.every((limit) => Math.abs(limit - edge.speedLimit) < SIGN_LIMIT_EPSILON_MPS)
      ) {
        continue // 手前と同じ規制なので標識は要らない
      }

      const to = nodes[toId]
      const heading = Math.atan2(to.y - node.y, to.x - node.x)
      // 交差点から少し進んだ位置の、進行方向左側（= heading + pi/2）へ寄せる
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

// ---------------------------------------------------------------------------
// 車両シミュレーション（格子上の貪欲ナビゲーション）
// ---------------------------------------------------------------------------

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

/**
 * 目的地までの残り距離 [m]。
 *
 * 「いま向かっているノードまでの距離 + そこから目的地までの格子距離」。
 * 貪欲ナビは x か y のどちらかを必ず 1 マス詰めるので、経路の取り方が変わっても
 * この値は単調に減る。したがって progress の分母として使える。
 */
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

// ---------------------------------------------------------------------------
// モック本体
// ---------------------------------------------------------------------------

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
  private obstacleSeq = 1
  private tick = 0
  private simTime = 0
  private startedAt = performance.now()
  private frameTimer: ReturnType<typeof setInterval> | null = null
  private metricsTimer: ReturnType<typeof setInterval> | null = null
  private loadTimer: ReturnType<typeof setTimeout> | null = null
  private closed = false
  // 学習カーブの疑似状態
  private updates = 0
  private episodes = 0
  private progress = 0

  constructor(emit: (json: string) => void) {
    this.emit = emit
    for (let i = 0; i < MOCK_CONFIG.maxVehicles; i++) {
      this.vehicles.push(this.makeVehicle(i))
    }
    this.applyVehicleCount(this.params.vehicleCount)

    // init は接続直後に 1 回だけ
    setTimeout(() => this.sendInit(), 60)

    this.frameTimer = setInterval(() => this.step(), FRAME_MS)
    this.metricsTimer = setInterval(() => this.sendMetrics(), 1000)
  }

  // ---- 送信 ----

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
      goalRate: Math.max(0, Math.min(1, 0.05 + 0.85 * p + noise() * 0.04)),
      stepsPerSec: SIM_HZ * this.params.simSpeed + noise() * 0.4,
      signalViolations: 0,
      speedViolations: 0,
      laneDeviation: 0,
    }
    this.send(metrics)
  }

  // ---- 信号 ----

  /**
   * マップ読込時に、信号の引き当て表を作る。
   *
   * 灯器は「進入車両に正対」しているので、heading の逆側にある隣ノードが
   * 「どこから来る車のための信号か」を表す。格子の heading は 0 / ±pi/2 / pi の
   * いずれかなので、cos・sin を丸めればそのまま隣マスの差分になる。
   */
  private buildSignalIndex(): void {
    this.signalIndex.clear()
    this.signalSetback = []
    this.signLimits.clear()
    // 標識は「出発ノードから heading の向きへ出る区間」の入口に立っている
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

  /**
   * 灯色を simTime だけの関数として決める（backend/app/sim/signals.py と同じ式）。
   * 内部状態を持たないので、一時停止やリセットと自然に整合する。
   */
  private computePhases(): number[] {
    const signals = this.map?.signals ?? []
    const out: number[] = new Array(signals.length)
    for (let i = 0; i < signals.length; i++) {
      const sg = signals[i]
      // 街じゅうの信号が一斉に変わらないよう、ノードごとに位相をずらす
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

  // ---- 車両 ----

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
    // エピソードの切れ目なので速度超過の統計は持ち越さない
    // （backend の env._reset_slot_stats() と同じ扱い）
    v.speedViolations = 0
    v.overspeeding = false
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

  private stepVehicle(v: MockVehicle, dt: number): void {
    if (!v.active) return

    const ax = nodeX(v.gx)
    const ay = nodeY(v.gy)
    const bx = nodeX(v.tx)
    const by = nodeY(v.ty)
    const segLen = Math.hypot(bx - ax, by - ay)

    // 目的地に着いていたら次の目的地を決める
    if (segLen < 1e-6) {
      v.goalGx = Math.floor(this.rng() * GRID_N)
      v.goalGy = Math.floor(this.rng() * GRID_N)
      chooseNext(v, this.rng)
      this.beginRoute(v)
      v.reachedGoal = true
      this.episodes += 1
      return
    }

    // 交差点手前で少し減速し、直線で加速する（見た目の変化をつける）
    let targetSpeed = this.params.maxSpeed * (0.45 + 0.55 * Math.sin(Math.PI * v.t))

    // 区間の入口に標識があれば、そこから規制速度が切り替わる。
    // 無い区間では手前の標識の値がそのまま効き続ける（実際の規制と同じ）。
    const limit = this.signLimits.get(`${v.gx},${v.gy}>${v.tx},${v.ty}`)
    if (limit !== undefined) v.speedLimit = limit
    // 「標識に従う」が入っていれば規制速度で頭打ちにする（backend の加速指令の抑制と同じ）
    if (this.params.obeySpeedSigns && v.speedLimit > 0) {
      targetSpeed = Math.min(targetSpeed, v.speedLimit)
    }

    // 前方の信号（protocol.md 5章）。赤、および安全に止まれる黄では
    // 停止線の手前で止まる。t が停止線を越えないようにクランプするので、
    // 実バックエンドと違って信号無視は起きない。
    let stopT = 1
    if (this.params.obeySignals) {
      const idx = this.signalIndex.get(`${v.gx},${v.gy}>${v.tx},${v.ty}`)
      const phase = idx === undefined ? SIGNAL_GREEN : this.phases[idx]
      if (idx !== undefined && phase !== SIGNAL_GREEN) {
        const setback = this.signalSetback[idx]
        const stopDist = (1 - v.t) * segLen - setback
        const canStop = (v.speed * v.speed) / (2 * BRAKE_ACCEL) <= Math.max(0, stopDist)
        // 黄で止まりきれないときはそのまま進んでよい（道交法施行令 2 条ただし書き）
        if (phase === SIGNAL_RED || canStop) {
          stopT = Math.max(0, 1 - setback / segLen)
          targetSpeed = Math.min(targetSpeed, Math.sqrt(2 * BRAKE_ACCEL * Math.max(0, stopDist)))
        }
      }
    }

    v.speed += (targetSpeed - v.speed) * Math.min(1, dt * 1.6)

    // 超え「始めた」ステップだけを 1 回と数える（protocol.md の speedViolations）
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
    // 舵角は heading の変化から作る（最短回りで差を取る）
    const dh = Math.atan2(Math.sin(v.heading - prevHeading), Math.cos(v.heading - prevHeading))
    v.steer += (Math.max(-0.5, Math.min(0.5, dh * 4)) - v.steer) * 0.25

    if (v.t >= 1) {
      v.t = 0
      v.gx = v.tx
      v.gy = v.ty
      v.x = nodeX(v.gx)
      v.y = nodeY(v.gy)
      if (v.gx === v.goalGx && v.gy === v.goalGy) {
        v.reachedGoal = true
        this.episodes += 1
        v.goalGx = Math.floor(this.rng() * GRID_N)
        v.goalGy = Math.floor(this.rng() * GRID_N)
        chooseNext(v, this.rng)
        // 目的地が変わったので達成度の分母を取り直す
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

  /**
   * 同じ区間を走る前走車に追突しないよう、後続の位置を後ろへ詰める。
   *
   * モックには車間制御が無いので、これが無いと赤信号の停止線で
   * 全車が同じ点に重なり、衝突フラグが立ちっぱなしになる。
   */
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
      if (segLen < 1e-6) continue // 目的地で停止中のスロット同士

      const gap = CAR_GAP_M / segLen
      list.sort((a, b) => b.t - a.t) // 前を走っている車から順に見る
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
      // 障害物との接触
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

    for (const v of this.vehicles) this.stepVehicle(v, dt)
    this.detectCollisions()

    // 「一時停止」は frame 配信だけを止める。裏の学習・物理は回り続ける（protocol.md 4章）
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
    // 信号のあるマップなら現示も載せる。これが無いと TrafficSignals も
    // RoadMarkings の停止線・横断歩道もモックでは一切動かない（code_review F-13）
    if (this.map?.signals?.length) frame.signals = this.computePhases()
    this.send(frame)
  }

  // ---- 受信 ----

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
        // 実機では Overpass API から取るので 10〜60 秒かかる。
        // モックでは進捗 UI を確認できる程度の 1.4 秒にしている。
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
        this.applyVehicleCount(this.params.vehicleCount)
        this.sendParams()
        break
      }

      case 'spawn_vehicle': {
        if (!this.map) {
          this.send({ type: 'error', code: 'NO_MAP_LOADED', message: 'マップが未読込です' })
          return
        }
        const slot = this.vehicles.find((v) => !v.active)
        if (!slot) {
          this.send({ type: 'error', code: 'SPAWN_FAILED', message: '空きスロットがありません' })
          return
        }
        // 最寄りの格子ノードへスナップ
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
    this.frameTimer = null
    this.metricsTimer = null
    this.loadTimer = null
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
