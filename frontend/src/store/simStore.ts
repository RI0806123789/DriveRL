/**
 * アプリ全体の状態（zustand）。
 *
 * ここに入れてよいのは「低頻度で変わるもの」だけ。
 * 20Hz の frame は store/frameBuffer.ts の可変オブジェクトに置き、
 * Three.js の useFrame から直接読む（React を再レンダリングしない）。
 */

import type { ThemeName } from '../scene/palette'
import { bootstrapTheme } from './themeClock'
import { create } from 'zustand'
import type {
  ConnectionState,
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
} from '../types/protocol'
import { PROTOCOL_VERSION } from '../types/protocol'
import { pushFrame, resetFrameBuffer } from './frameBuffer'

/** メトリクス履歴のリングバッファ上限（要件: 300 点） */
export const METRICS_CAPACITY = 300

export type PanelTab = 'simulation' | 'map' | 'learning' | 'view'
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
}

// init メッセージが来るまでの暫定値。実装（backend/app/config.py）と揃える。
// ★ ずれていると、init 到着の 1 回で `Vehicles` の count が変わり、
//   InstancedMesh が起動のたびに作り直される（code_review Q-04）。
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
}

export interface SimStore {
  // ---- 通信 ----
  connection: ConnectionState
  /** init を受け取ったか（= プロトコル交渉が済んだか） */
  handshaked: boolean
  protocolMismatch: boolean
  /** モックサーバーに接続しているか（開発用） */
  usingMock: boolean

  // ---- サーバーから来る状態 ----
  presets: MapPreset[]
  config: SimConfig
  params: SimParams
  status: StatusPayload
  map: MapMessage | null
  /** 「読込中」を押した直後の楽観的表示に使う。status が来たら解除される */
  pendingPresetId: string | null
  metrics: MetricsMessage[]
  latestMetrics: MetricsMessage | null
  /** ネットワークの状態（層ごとの重み・勾配・変化量）。1Hz で更新される */
  network: NetworkMessage | null
  errors: ErrorEntry[]

  // ---- UI 状態 ----
  panelOpen: boolean
  /**
   * 配色。**日の出・日の入りで自動的に切り替わる**（store/autoTheme.ts）。
   * 利用者が変える手段は無いので、対応する setter も置いていない。
   * パネル UI は tokens.css の `:root[data-theme]`、
   * 3D シーンは scene/palette.ts が、この同じ値を見る。
   */
  theme: ThemeName
  tab: PanelTab
  cameraMode: CameraMode
  /** 追従対象のスロット番号 */
  followTarget: number
  interaction: InteractionMode
  /** 設置する障害物の半径 [m] */
  obstacleRadius: number
  view: ViewToggles

  // ---- アクション（UI 側） ----
  setPanelOpen(open: boolean): void
  togglePanel(): void
  setTab(tab: PanelTab): void
  setCameraMode(mode: CameraMode): void
  setFollowTarget(id: number): void
  setInteraction(mode: InteractionMode): void
  setObstacleRadius(r: number): void
  toggleView(key: keyof ViewToggles): void
  /** サーバー応答を待たずにスライダーを滑らかに動かすための楽観的更新 */
  patchParamsLocal(patch: Partial<SimParams>): void
  dismissError(id: number): void

  // ---- 通信側から呼ばれる ----
  setConnection(state: ConnectionState): void
  setUsingMock(v: boolean): void
  applyServerMessage(msg: ServerMessage): void
}

let errorSeq = 0

// ---------------------------------------------------------------------------
// テーマ
//
// 利用者は選べない。**日の出・日の入りに連動して自動で切り替わる**。
// 判定は store/themeClock.ts（純粋）、タイマーは store/autoTheme.ts（React）。
//
// 保存もしない。時刻から決まる値なので、保存した値と現在時刻が食い違うと
// 「夜なのに前回の昼の配色で開く」ことになり、かえって邪魔になるため。
// ---------------------------------------------------------------------------

export type { ThemeName }

/** <html data-theme="..."> を書き換える。tokens.css の変数がこれで切り替わる */
export function applyTheme(theme: ThemeName): void {
  if (typeof document === 'undefined') return
  document.documentElement.dataset.theme = theme
}

/**
 * 初期値。**ストアを作るより先に DOM へ反映する。**
 *
 * これを React の副作用に任せると、最初の 1 フレームだけ `data-theme` が
 * 未設定になり、CSS が既定（暗い方）に落ちてパネルが一瞬黒く光る。
 * `tokens.css` はライトを `:root[data-theme='light']` の上書きとして持つので、
 * **属性が無い＝ダーク**という非対称があることに注意。
 */
const INITIAL_THEME = bootstrapTheme()
applyTheme(INITIAL_THEME)

export const useSimStore = create<SimStore>((set, get) => ({
  connection: 'connecting',
  handshaked: false,
  protocolMismatch: false,
  usingMock: false,

  presets: [],
  config: DEFAULT_CONFIG,
  params: DEFAULT_PARAMS,
  status: DEFAULT_STATUS,
  map: null,
  pendingPresetId: null,
  metrics: [],
  latestMetrics: null,
  network: null,
  errors: [],

  panelOpen: true,
  theme: INITIAL_THEME,
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
      // 切断されたら補間バッファを捨てる（古い位置を引きずらせない）
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
        set({
          handshaked: true,
          protocolMismatch: init.protocolVersion !== PROTOCOL_VERSION,
          presets: init.presets ?? [],
          config: init.config ?? DEFAULT_CONFIG,
          params: init.params ?? DEFAULT_PARAMS,
          status: init.status ?? DEFAULT_STATUS,
          pendingPresetId: null,
        })
        // 追従対象が maxVehicles を超えていたら丸める
        const cfg = init.config ?? DEFAULT_CONFIG
        if (get().followTarget >= cfg.maxVehicles) set({ followTarget: 0 })
        break
      }

      case 'map': {
        resetFrameBuffer()
        set({ map: msg as MapMessage, pendingPresetId: null })
        break
      }

      case 'frame': {
        // ★ ここだけは zustand に入れない（20Hz で React を回さないため）
        pushFrame(msg)
        break
      }

      case 'status': {
        const { type: _t, ...payload } = msg
        set((s) => ({
          status: payload as StatusPayload,
          pendingPresetId: payload.state === 'loading_map' ? s.pendingPresetId : null,
        }))
        break
      }

      case 'params': {
        set({ params: msg.params })
        break
      }

      case 'metrics': {
        set((s) => {
          const next = s.metrics.length >= METRICS_CAPACITY
            ? s.metrics.slice(s.metrics.length - METRICS_CAPACITY + 1)
            : s.metrics.slice()
          next.push(msg)
          return { metrics: next, latestMetrics: msg }
        })
        break
      }

      case 'network': {
        set({ network: msg })
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
        // 未知の type は無視する（前方互換）
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
