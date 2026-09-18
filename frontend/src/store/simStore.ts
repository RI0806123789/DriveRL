/** アプリ全体の状態（zustand）。 */

import type { ThemeName } from '../scene/palette'
import { bootstrapTheme } from './themeClock'
import { create } from 'zustand'
import type {
  ConnectionState,
  DetectorMessage,
  ErrorMessage,
  InitMessage,
  MapMessage,
  MapPreset,
  MetricsMessage,
  NetworkMessage,
  ServerMessage,
  SimConfig,
  SimParams,
  StatusPayload,
  TaxiMessage,
  WeatherPreset,
} from '../types/protocol'
import { PROTOCOL_VERSION } from '../types/protocol'
import { pushFrame, resetFrameBuffer } from './frameBuffer'

/** メトリクス履歴に**上限は設けない**。 */

/** グラフ 1 本ぶんの系列。キーは LearningTab の並びと揃える */
export interface MetricsSeries {
  reward: number[]
  policyLoss: number[]
  valueLoss: number[]
  entropy: number[]
  goalRate: number[]
  violations: number[]
  laneDeviation: number[]
}

function emptyMetricsSeries(): MetricsSeries {
  return {
    reward: [],
    policyLoss: [],
    valueLoss: [],
    entropy: [],
    goalRate: [],
    violations: [],
    laneDeviation: [],
  }
}

export type PanelTab = 'simulation' | 'map' | 'learning' | 'model' | 'view'
/** 開発モード（学習を回す） / 実用モード（学習済みモデルでタクシーを呼ぶ） */
export type AppMode = 'dev' | 'taxi'
/** 俯瞰（自由視点） / 追従（後方上空） / 運転席（一人称） */
export type CameraMode = 'orbit' | 'follow' | 'driver'
/** 3D 画面クリック時のふるまい（memo F-05） */
export type InteractionMode = 'none' | 'obstacle' | 'vehicle'

/** ViewTab の表示トグル */
export interface ViewToggles {
  buildings: boolean
  roads: boolean
  /** 道路標示（中央線・車線境界線・停止線・横断歩道） */
  markings: boolean
  /** 交通信号機 */
  signals: boolean
  /** 最高速度標識。3D 側はこの名前で購読する */
  showSigns: boolean
  routes: boolean
  goals: boolean
  shadows: boolean
  grid: boolean
  /** 認識結果のバウンディングボックス（運転席／追従カメラのみ）。scene/DetectionOverlay.tsx */
  detections: boolean
}

/** サーバーから来たエラーを画面に出すためのログ 1 行 */
export interface ErrorEntry {
  id: number
  code: ErrorMessage['code']
  message: string
  at: number
}

const DEFAULT_PARAMS: SimParams = {
  vehicleCount: 3,
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
  weatherRain: 0,
  weatherFog: 0,
  weatherAuto: false,
}

const DEFAULT_CONFIG: SimConfig = {
  maxVehicles: 8,
  simHz: 20,
  obsDim: 57,
  actionDim: 2,
}

const DEFAULT_STATUS: StatusPayload = {
  state: 'idle',
  mapLoaded: false,
  presetId: null,
  renderPaused: false,
  learning: false,
  simSuspended: false,
  practicalMode: false,
  taxiVehicleId: -1,
}

const IDLE_TAXI: TaxiMessage = {
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

export interface SimStore {
  connection: ConnectionState
  /** init を受け取ったか（= プロトコル交渉が済んだか） */
  handshaked: boolean
  protocolMismatch: boolean
  /** モックサーバーに接続しているか（開発用） */
  usingMock: boolean

  presets: MapPreset[]
  /** 天候の選択肢。数値はサーバーが配る（表示名だけ UI 側が持つ） */
  weatherPresets: WeatherPreset[]
  config: SimConfig
  params: SimParams
  status: StatusPayload
  map: MapMessage | null
  /** 「読込中」を押した直後の楽観的表示に使う。status が来たら解除される */
  pendingPresetId: string | null
  /**
   * グラフ用の系列。**その場で追記する**（毎秒 O(n) のコピーを避けるため。
   * `frameBuffer` と同じ作法。code_review E-03）。読む側は `metricsRevision` を
   * 再計算の契機にすること。
   */
  metricsSeries: MetricsSeries
  /** `metricsSeries` に追記するたびに増える */
  metricsRevision: number
  /** マップを切り替えた位置（系列の添字）。グラフの縦線。code_review E-04 */
  metricsMarks: number[]
  latestMetrics: MetricsMessage | null
  /** ネットワークの状態（層ごとの重み・勾配・変化量）。1Hz で更新される */
  network: NetworkMessage | null
  /** 認識器（CNN）の学習状況。接続直後に 1 通届き、以後は進捗が動いたときだけ。 */
  detector: DetectorMessage | null
  /** 実用モードの配車状態。`route` は届いた最新のものを保持し続ける */
  taxi: TaxiMessage
  errors: ErrorEntry[]

  panelOpen: boolean
  /** 配色。**日の出・日の入りで自動的に切り替わる**（store/autoTheme.ts）。 */
  theme: ThemeName
  /** 開発 / 実用。**切り替えるとタブ構成ごと入れ替わる**（決定 9） */
  mode: AppMode
  tab: PanelTab
  cameraMode: CameraMode
  /** 追従対象のスロット番号 */
  followTarget: number
  interaction: InteractionMode
  /** 設置する障害物の半径 [m] */
  obstacleRadius: number
  view: ViewToggles

  setPanelOpen(open: boolean): void
  togglePanel(): void
  setMode(mode: AppMode): void
  setTab(tab: PanelTab): void
  setCameraMode(mode: CameraMode): void
  setFollowTarget(id: number): void
  setInteraction(mode: InteractionMode): void
  setObstacleRadius(r: number): void
  toggleView(key: keyof ViewToggles): void
  /** サーバー応答を待たずにスライダーを滑らかに動かすための楽観的更新 */
  patchParamsLocal(patch: Partial<SimParams>): void
  dismissError(id: number): void

  setConnection(state: ConnectionState): void
  setUsingMock(v: boolean): void
  applyServerMessage(msg: ServerMessage): void
}

let errorSeq = 0

export type { ThemeName }

/** <html data-theme="..."> を書き換える。tokens.css の変数がこれで切り替わる */
export function applyTheme(theme: ThemeName): void {
  if (typeof document === 'undefined') return
  document.documentElement.dataset.theme = theme
}

/** 初期値。**ストアを作るより先に DOM へ反映する。** */
const INITIAL_THEME = bootstrapTheme()
applyTheme(INITIAL_THEME)

export const useSimStore = create<SimStore>((set, get) => ({
  connection: 'connecting',
  handshaked: false,
  protocolMismatch: false,
  usingMock: false,

  presets: [],
  weatherPresets: [],
  config: DEFAULT_CONFIG,
  params: DEFAULT_PARAMS,
  status: DEFAULT_STATUS,
  map: null,
  pendingPresetId: null,
  metricsSeries: emptyMetricsSeries(),
  metricsRevision: 0,
  metricsMarks: [],
  latestMetrics: null,
  network: null,
  detector: null,
  taxi: IDLE_TAXI,
  errors: [],

  panelOpen: true,
  theme: INITIAL_THEME,
  mode: 'dev',
  tab: 'simulation',
  cameraMode: 'orbit',
  followTarget: 0,
  interaction: 'none',
  obstacleRadius: 0.5,
  view: {
    buildings: true,
    roads: true,
    markings: true,
    signals: true,
    showSigns: true,
    routes: true,
    goals: true,
    shadows: true,
    grid: false,
    detections: true,
  },

  setPanelOpen: (open) => set({ panelOpen: open }),
  togglePanel: () => set((s) => ({ panelOpen: !s.panelOpen })),
  setMode: (mode) => set({ mode }),
  setTab: (tab) => set({ tab }),
  setCameraMode: (cameraMode) => set({ cameraMode }),
  setFollowTarget: (followTarget) => set({ followTarget }),
  setInteraction: (interaction) => set({ interaction }),
  setObstacleRadius: (obstacleRadius) => set({ obstacleRadius }),
  toggleView: (key) => set((s) => ({ view: { ...s.view, [key]: !s.view[key] } })),
  patchParamsLocal: (patch) => set((s) => ({ params: { ...s.params, ...patch } })),
  dismissError: (id) => set((s) => ({ errors: s.errors.filter((e) => e.id !== id) })),

  setConnection: (connection) => {
    if (connection !== 'open') {
      resetFrameBuffer()
      set({ connection, handshaked: false })
    } else {
      set({ connection })
    }
  },

  setUsingMock: (usingMock) => set({ usingMock }),

  applyServerMessage: (msg) => {
    switch (msg.type) {
      case 'init': {
        const init = msg as InitMessage
        resetFrameBuffer()
        const status = init.status ?? DEFAULT_STATUS
        set({
          handshaked: true,
          protocolMismatch: init.protocolVersion !== PROTOCOL_VERSION,
          presets: init.presets ?? [],
          weatherPresets: init.weatherPresets ?? [],
          config: init.config ?? DEFAULT_CONFIG,
          params: init.params ?? DEFAULT_PARAMS,
          status,
          // ★ リロードしてもサーバーのモードへ戻す（配車の途中で開発モードに落とさない）
          mode: status.practicalMode ? 'taxi' : 'dev',
          pendingPresetId: null,
        })
        const cfg = init.config ?? DEFAULT_CONFIG
        if (get().followTarget >= cfg.maxVehicles) set({ followTarget: 0 })
        break
      }

      case 'map': {
        resetFrameBuffer()
        set((s) => ({
          map: msg as MapMessage,
          pendingPresetId: null,
          // 重みはマップをまたいで引き継ぐので系列は消さず、位置だけ覚える
          metricsMarks:
            s.metricsRevision > 0 && s.metricsMarks.at(-1) !== s.metricsSeries.reward.length
              ? [...s.metricsMarks, s.metricsSeries.reward.length]
              : s.metricsMarks,
        }))
        break
      }

      case 'frame': {
        pushFrame(msg)
        break
      }

      case 'status': {
        const { type: _t, ...payload } = msg
        set((s) => ({
          status: payload as StatusPayload,
          pendingPresetId: payload.state === 'loading_map' ? s.pendingPresetId : null,
          // モードはサーバーが正。トグルの楽観的更新はここで確定する
          mode:
            payload.practicalMode === undefined
              ? s.mode
              : payload.practicalMode
                ? 'taxi'
                : 'dev',
        }))
        break
      }

      case 'params': {
        set({ params: msg.params })
        break
      }

      case 'metrics': {
        set((s) => {
          const series = s.metricsSeries
          series.reward.push(msg.meanEpisodeReward)
          series.policyLoss.push(msg.policyLoss)
          series.valueLoss.push(msg.valueLoss)
          series.entropy.push(msg.entropy)
          series.goalRate.push(msg.goalRate)
          series.violations.push(msg.signalViolations)
          series.laneDeviation.push(msg.laneDeviation)
          return { metricsRevision: s.metricsRevision + 1, latestMetrics: msg }
        })
        break
      }

      case 'network': {
        set({ network: msg })
        break
      }

      case 'detector': {
        set({ detector: msg as DetectorMessage })
        break
      }

      case 'taxi': {
        // 経路は版が変わった通にしか入らないので、無い通では前回のものを引き継ぐ
        set((s) => ({
          taxi: {
            ...msg,
            route: msg.route ?? (msg.routeRevision === s.taxi.routeRevision ? s.taxi.route : []),
          },
        }))
        break
      }

      case 'error': {
        errorSeq += 1
        const entry: ErrorEntry = {
          id: errorSeq,
          code: msg.code,
          message: msg.message,
          at: Date.now(),
        }
        set((s) => ({
          errors: [...s.errors, entry].slice(-5),
          pendingPresetId: msg.code === 'MAP_LOAD_FAILED' ? null : s.pendingPresetId,
        }))
        break
      }

      case 'pong':
        break

      default: {
        break
      }
    }
  },
}))

/** load_map を押した直後の楽観的表示に使う */
export function markMapLoading(presetId: string): void {
  useSimStore.setState((s) => ({
    pendingPresetId: presetId,
    status: { ...s.status, state: 'loading_map' },
  }))
}
