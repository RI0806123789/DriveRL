/** 開発用のモックサーバー（ブラウザ内で動く偽 WebSocket サーバー）。中身は `mock/` に分けてある */

import type {
  ClientMessage,
  Detection,
  FrameMessage,
  MapMessage,
  MetricsMessage,
  ServerMessage,
  SimConfig,
  SimParams,
  StatusPayload,
  SurroundDetections,
  VehicleState,
  WeatherPreset,
  WeatherState,
} from '../types/protocol.ts'
import {
  DET_LANE,
  DET_PEDESTRIAN,
  DET_TRAFFIC_LIGHT,
  PROTOCOL_VERSION,
  SIGNAL_GREEN,
} from '../types/protocol.ts'
import type { Transport } from './connection.ts'
import { MockDetectorJob } from './mock/detectorJob.ts'
import { FRAME_MS, SIM_HZ, makeRng } from './mock/grid.ts'
import { MOCK_PRESETS, MockSignals, buildMockMap } from './mock/map.ts'
import { mockSurround } from './mock/surround.ts'
import { MockTaxi } from './mock/taxi.ts'
import { MockTraffic } from './mock/traffic.ts'
import { ASSIST_P_MIN } from './assistRate.ts'
import { mockCurriculum } from './curriculum.ts'
import { nearestPeers } from '../scene/v2xLinks.ts'

const MOCK_CONFIG: SimConfig = {
  maxVehicles: 8,
  maxPedestrians: 64,
  simHz: SIM_HZ,
  obsDim: 79,
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
  safetyAssist: false,
  onlineAssist: true,
  incidentCurriculum: true,
  v2xComm: true,
}

const MOCK_MAX_PEDESTRIANS = MOCK_CONFIG.maxPedestrians ?? 64

/** 実機と同じ規則（届く距離の中で近い 2 台）で V2X の相手を載せる。乱数は引かない */
function withV2xLinks(vehicles: VehicleState[], enabled: boolean): VehicleState[] {
  if (!enabled) return vehicles
  const peers = nearestPeers(vehicles)
  for (const v of vehicles) {
    const links = peers.get(v.id)
    if (links) v.v2xConnectedIds = links
  }
  return vehicles
}

const MOCK_WEATHER_PERIOD_SEC = 480
/** 周囲カメラの検出を載せ続ける時間 [ms]。本物の `engine.SURROUND_WATCH_TTL_SEC` と同じ */
const MOCK_SURROUND_TTL_MS = 2500

const MOCK_MIN_VISIBILITY_M = 15

class MockServer {
  private readonly emit: (json: string) => void
  private readonly rng = makeRng(20260905)
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
  private readonly signals = new MockSignals()
  private readonly traffic: MockTraffic
  private readonly taxi: MockTaxi
  private readonly detector: MockDetectorJob
  private tick = 0
  private simTime = 0
  private readonly startedAt = performance.now()
  private frameTimer: ReturnType<typeof setInterval> | null = null
  private metricsTimer: ReturnType<typeof setInterval> | null = null
  private loadTimer: ReturnType<typeof setTimeout> | null = null
  private closed = false
  private updates = 0
  private progress = 0
  /** 周囲カメラの検出を頼まれた車と、頼まれた時刻 [ms] */
  private surroundWatch = new Map<number, number>()

  constructor(emit: (json: string) => void) {
    this.emit = emit
    this.traffic = new MockTraffic(this.rng, MOCK_CONFIG.maxVehicles)
    this.traffic.applyVehicleCount(this.params.vehicleCount)
    this.traffic.applyPedestrianCount(this.params.pedestrianCount, MOCK_MAX_PEDESTRIANS)
    this.taxi = new MockTaxi(this.traffic.vehicles, this.rng, {
      sendTaxi: (msg) => this.send(msg),
      sendStatus: (patch) => this.sendStatus(patch),
    })
    this.detector = new MockDetectorJob({
      sendDetector: (msg) => this.send(msg),
      sendStatus: (patch) => this.sendStatus(patch),
    })

    setTimeout(() => {
      this.sendInit()
      this.detector.send()
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
    // message を渡したときは 1 回きりの知らせ（本物のサーバーの notice と同じ扱い）
    this.send({ type: 'status', ...this.status, notice: patch?.message !== undefined })
  }

  private sendParams(): void {
    this.send({ type: 'params', params: this.params })
  }

  private sendMetrics(): void {
    if (!this.map) return
    this.updates += 1
    const p = this.progress
    const noise = () => (this.rng() - 0.5) * 2
    const collisions = Math.max(0, 0.55 * (1 - p) + noise() * 0.03)
    // 歩行者との衝突は衝突率の一部。**別の式で出さないこと** — 独立に揺らすと
    // 「衝突 0% / うち歩行者 1%」という、実サーバーでは起こりえない表示になる
    const pedestrianShare = Math.min(1, Math.max(0, 0.33 + noise() * 0.05))
    const metrics: MetricsMessage = {
      type: 'metrics',
      tick: this.tick,
      wallTime: (performance.now() - this.startedAt) / 1000,
      updates: this.updates,
      episodes: this.traffic.episodes,
      meanEpisodeReward: -80 + 120 * p + noise() * 9,
      meanEpisodeLength: 90 + 120 * p + noise() * 12,
      policyLoss: Math.max(0.001, 0.09 * (1 - p) + Math.abs(noise()) * 0.006),
      valueLoss: Math.max(0.01, 1.4 * (1 - p * 0.9) + Math.abs(noise()) * 0.08),
      entropy: Math.max(0.05, 1.35 - 0.85 * p + noise() * 0.03),
      approxKl: Math.max(0.0002, 0.014 * (1 - p * 0.6) + Math.abs(noise()) * 0.002),
      collisionRate: collisions,
      pedestrianCollisionRate: collisions * pedestrianShare,
      goalRate: Math.max(0, Math.min(1, 0.05 + 0.85 * p + noise() * 0.04)),
      stepsPerSec: SIM_HZ * this.params.simSpeed + noise() * 0.4,
      signalViolations: 0,
      speedViolations: 0,
      laneDeviation: 0,
      // 実機と同じ下がり方（`rl/online_assist.py`）。乱数は引かない（引く順番を変えない）
      assistRate: this.params.onlineAssist ? Math.max(ASSIST_P_MIN, 1 - p) : 0,
      bcLoss: this.params.onlineAssist ? 0.02 * Math.max(ASSIST_P_MIN, 1 - p) : 0,
      ...mockCurriculum(this.params.incidentCurriculum, p, this.updates),
    }
    this.send(metrics)
  }

  private step(): void {
    if (this.closed || !this.map) return

    const dt = (FRAME_MS / 1000) * this.params.simSpeed
    this.tick += 1
    this.simTime += dt
    this.progress = Math.min(1, this.progress + dt * 0.004)

    // 灯色はこのステップの simTime だけで決まる。車・frame・認識結果のダミーが同じものを見る
    const phases = this.signals.phases(this.simTime)
    this.traffic.step({
      dt,
      tick: this.tick,
      params: this.params,
      phases,
      signals: this.signals,
      heldVehicle: this.taxi.heldVehicle,
    })
    this.taxi.step()

    if (this.status.renderPaused) return

    const frame: FrameMessage = {
      type: 'frame',
      tick: this.tick,
      simTime: this.simTime,
      vehicles: withV2xLinks(this.traffic.vehicleStates(), this.params.v2xComm),
      obstacles: this.traffic.obstacles,
    }
    const pedestrians = this.traffic.pedestrianStates()
    if (pedestrians.length) frame.pedestrians = pedestrians
    if (phases.length) frame.signals = phases
    frame.weather = this.currentWeather()
    const detections = this.buildDetections(phases)
    if (detections) frame.detections = detections
    const surround = this.buildSurround(frame)
    if (surround) frame.surround = surround
    this.send(frame)
  }

  /** 頼まれている車の周囲カメラの検出（本物は `env._surround_wire`） */
  private buildSurround(frame: FrameMessage): Record<string, SurroundDetections> | null {
    const now = Date.now()
    const out: Record<string, SurroundDetections> = {}
    for (const [id, at] of this.surroundWatch) {
      if (now - at > MOCK_SURROUND_TTL_MS) {
        this.surroundWatch.delete(id)
        continue
      }
      const car = frame.vehicles.find((v) => v.id === id && v.active)
      if (car) out[String(id)] = mockSurround(car, frame.vehicles, frame.pedestrians ?? [])
    }
    return Object.keys(out).length ? out : null
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

  /** 認識結果のダミー。**運転席カメラのボックスと路面の車線オーバーレイをモックでも確かめるため**の固定の 2 件 */
  private buildDetections(phases: readonly number[]): Record<string, Detection[]> | null {
    const out: Record<string, Detection[]> = {}
    for (const v of this.traffic.vehicles) {
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
        phase: phases.length ? phases[v.id % phases.length] : SIGNAL_GREEN,
        distance: 30,
      }
      const dets: Detection[] = [lane, light]
      // 近くを歩いている人がいれば、擬似的な検出枠を 1 つ出す
      if (this.traffic.pedestrianNear(v.x, v.y, 24)) {
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
          const map = buildMockMap(preset.id, preset.name)
          this.map = map
          this.signals.build(map)
          this.tick = 0
          this.simTime = 0
          this.traffic.markRoutesDirty()
          this.send(map)
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
        // 値域は実機（contracts.SimParams の simSpeed）と同じ 0.25〜8
        this.params.simSpeed = Math.max(0.25, Math.min(8, this.params.simSpeed))
        this.params.rewardOverspeed = Math.max(
          -1000,
          Math.min(0, this.params.rewardOverspeed),
        )
        this.params.pedestrianCount = Math.max(
          0,
          Math.min(MOCK_MAX_PEDESTRIANS, Math.round(this.params.pedestrianCount)),
        )
        this.traffic.applyVehicleCount(this.params.vehicleCount)
        this.traffic.applyPedestrianCount(this.params.pedestrianCount, MOCK_MAX_PEDESTRIANS)
        this.sendParams()
        break
      }

      case 'spawn_vehicle': {
        if (!this.map) {
          this.sendStatus({ message: 'マップが読み込まれていません' })
          return
        }
        const slot = this.traffic.spawnAt(msg.x, msg.y)
        if (!slot) {
          this.sendStatus({ message: '空きスロットがありません' })
          return
        }
        this.params.vehicleCount = this.traffic.activeCount()
        this.sendParams()
        this.sendStatus({ message: `（モック）車両 #${slot.id} をスポーンしました` })
        break
      }

      case 'despawn_vehicle': {
        this.traffic.despawn(msg.id)
        this.params.vehicleCount = Math.max(1, this.traffic.activeCount())
        this.sendParams()
        break
      }

      case 'add_obstacle': {
        const count = this.traffic.addObstacle(msg.x, msg.y, msg.radius)
        this.sendStatus({ message: `（モック）障害物を設置しました（${count}）` })
        break
      }

      case 'remove_obstacle': {
        this.traffic.removeObstacle(msg.id)
        break
      }

      case 'clear_obstacles': {
        this.traffic.clearObstacles()
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
        this.traffic.resetAll()
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
        this.traffic.episodes = 0
        this.sendStatus({ message: '（モック）ポリシーを初期化しました' })
        break

      case 'start_detector_training': {
        if (this.detector.running) {
          this.sendStatus({ message: '（モック）すでに学習を実行中です' })
          return
        }
        if (this.status.state === 'loading_map') {
          this.sendStatus({
            message: '（モック）エリアを読み込んでいる間は認識器の学習を始められません',
          })
          return
        }
        this.detector.start(msg.request)
        break
      }

      case 'cancel_detector_training': {
        if (!this.detector.running) {
          this.sendStatus({ message: '（モック）実行中の学習はありません' })
          return
        }
        this.detector.finish('cancelled', '（モック）学習を中断しました')
        break
      }

      case 'set_app_mode': {
        const practical = msg.mode === 'taxi'
        if (practical === (this.status.practicalMode ?? false)) return
        if (!practical) this.taxi.cancel('（モック）開発モードに戻したため配車を終了しました')
        this.sendStatus({
          practicalMode: practical,
          learning: !practical,
          message: practical
            ? '（モック）実用モードに入りました。学習は止まります'
            : '（モック）開発モードに戻りました',
        })
        this.taxi.send(true)
        break
      }

      case 'request_taxi': {
        if (!this.map) {
          this.sendStatus({ message: '（モック）先にエリアを読み込んでください' })
          return
        }
        this.taxi.request(msg.pickup, msg.dropoff)
        break
      }

      case 'board_taxi':
        this.taxi.board()
        break

      case 'alight_taxi':
        this.taxi.alight()
        break

      case 'cancel_taxi':
        this.taxi.abort(Boolean(msg.halt))
        break

      case 'player_pose': {
        // 乗車中は車内にいるので歩行者として扱わない（実機と同じ）
        this.traffic.setPlayer(msg.at && !this.taxi.onboard ? [msg.at[0], msg.at[1]] : null)
        break
      }

      case 'ping':
        this.send({ type: 'pong', t: Date.now() })
        break

      case 'watch_surround': {
        // 本物と同じく、頼まれてから 2.5 秒だけ載せる
        this.surroundWatch.set(msg.vehicleId, Date.now())
        break
      }

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
    this.detector.close()
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
