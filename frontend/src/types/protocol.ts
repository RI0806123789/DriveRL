/** docs/protocol.md v3 の TypeScript 表現。 */

/** プロトコルバージョン。init.protocolVersion がこれと違えば警告する */
export const PROTOCOL_VERSION = 3

/** モデル書き出し API の形式。 */
export type ExportKind = 'checkpoint' | 'torchscript' | 'pt2' | 'keras'

/** ENU 平面上の座標 [x（東）, y（北）]（メートル） */
export type Vec2 = [number, number]

/** マップのプリセット（memo 5章: 固定エリアを事前プリセットで用意する） */
export interface MapPreset {
  id: string
  name: string
  description: string
  centerLat: number
  centerLon: number
  radiusM: number
}

/** サーバー側の固定設定（起動後に変わらない） */
export interface SimConfig {
  /** 事前確保するエージェントスロット数（擬似固定エージェント数方式） */
  maxVehicles: number
  /** 街を歩く NPC 歩行者の上限 */
  maxPedestrians?: number
  /** 物理・学習ステップの周波数 [Hz] */
  simHz: number
  obsDim: number
  actionDim: number
}

/** 実行時に変更できるパラメータ一式（2.5 params） */
export interface SimParams {
  /** 標準シナリオの最大車両数ぶんの予約枠。通常運転では変更できず、実際の走行台数は frame で確認する。 */
  vehicleCount: number
  /** 街を歩く NPC 歩行者の数 (0..maxPedestrians) */
  pedestrianCount: number
  /** 実時間に対する倍率 (0.25..8.0) */
  simSpeed: number
  learningRate: number
  gamma: number
  clipRange: number
  entropyCoef: number
  rolloutLength: number
  /** m/s */
  maxSpeed: number
  rewardGoal: number
  rewardCollision: number
  rewardProgress: number
  rewardOffroad: number
  rewardTime: number
  /** 赤信号で停止線を越えたときの罰（道交法施行令 2 条） */
  rewardSignal: number
  /** 最高速度標識の規制速度を超え始めたときの罰（道交法 22 条）。 */
  rewardOverspeed: number
  /** 信号に従わせるか。 */
  obeySignals: boolean
  /** 最高速度標識に従わせるか。 */
  obeySpeedSigns: boolean
  /** 雨の強さ 0.0〜1.0。画を濁らせるだけで視程は縮めない。 */
  weatherRain: number
  /** 霧の濃さ 0.0〜1.0。視程を縮める。 */
  weatherFog: number
  /** true の間はサーバーが simTime から天候を決める（上の 2 つは据え置かれる）。 */
  weatherAuto: boolean
  /** 学習中の車にも安全ギミックを掛けるか（実用モードの経路追従の車には常に掛かる）。 */
  safetyAssist: boolean
  /** 学習中の車に経路追従（エキスパート）を割り込ませ、模倣の教師にするか。 */
  onlineAssist: boolean
  /** 成績に応じて学習中の車にヒヤリハット（歩行者の飛び出し・前走車の急制動）を起こすか。 */
  incidentCurriculum: boolean
  /** 車車間通信（V2X）で近くの車のメッセージを観測に足すか。 */
  v2xComm: boolean
}

/** いま効いている天候（2.3 frame.weather） */
export interface WeatherState {
  rain: number
  fog: number
  /** 有効視程 [m]。擬似カメラの描画と正解ラベルが共有する唯一の値。 */
  visibility: number
}

/** サーバーの実行状態 */
export type SimState = 'idle' | 'loading_map' | 'running' | 'error'

/** status メッセージのペイロード部（init にも同じ形が入る） */
export interface StatusPayload {
  state: SimState
  mapLoaded: boolean
  presetId: string | null
  /** 「一時停止」は描画のみ。学習は継続している */
  renderPaused: boolean
  learning: boolean
  /** **物理と PPO ごと止まっているか。** `renderPaused` とは別物で、認識器の学習中だけ true（古いサーバーには無い） */
  simSuspended?: boolean
  /** `simSuspended` が true のときの理由 */
  suspendReason?: string
  /** 実用モード。**重みの更新だけが止まり、物理と推論は動き続ける**（2.10） */
  practicalMode?: boolean
  /** 配車で徴用中の車両スロット。-1 なら無し */
  taxiVehicleId?: number
  message?: string
  /** `message` が 1 回きりの知らせ（介入や配車を断った理由など）か。真なら画面にバナーで出す */
  notice?: boolean
}

/** 2.1 init — 接続確立直後に 1 回だけ */
export interface InitMessage {
  type: 'init'
  protocolVersion: number
  presets: MapPreset[]
  config: SimConfig
  /** 天候の選択肢。表示名だけクライアントが持ち、数値はサーバーが配る。 */
  weatherPresets?: WeatherPreset[]
  params: SimParams
  status: StatusPayload
}

/** 天候の選択肢 1 つ。`rain` / `fog` の出典はバックエンドの `PRESETS`。 */
export interface WeatherPreset {
  id: string
  rain: number
  fog: number
}

/** 天候プリセットの表示名。**キーはバックエンドの `percep.weather.PRESETS` と同じ。** */
export const WEATHER_LABELS: Record<string, string> = {
  clear: '晴れ',
  drizzle: '小雨',
  rain: '雨',
  fog: '霧',
  heavy_fog: '濃霧',
}

export interface MapBounds {
  minX: number
  maxX: number
  minY: number
  maxY: number
}

export interface MapNode {
  id: number
  x: number
  y: number
}

export interface MapEdge {
  id: number
  /** nodes[].id への参照（始点） */
  u: number
  /** nodes[].id への参照（終点） */
  v: number
  lanes: number
  /** メートル。道路メッシュの全幅 */
  width: number
  oneway: boolean
  /** m/s */
  speedLimit: number
  /** [x, y] の列。始点 = u、終点 = v */
  polyline: Vec2[]
}

export interface MapBuilding {
  id: number
  /** メートル。OSM に情報がなければ既定値（memo 5章: 約 9m） */
  height: number
  /** 外周（閉じない。最後の点と最初の点は別）。CCW 保証なし */
  outline: Vec2[]
}

/** 交通信号機（車両用）。OSM の highway=traffic_signals から作られる。 */
export interface MapSignal {
  id: number
  /** 交差点ノード（MapNode.id）への参照 */
  nodeId: number
  /** 停止線の位置（ENU） */
  x: number
  y: number
  /** 進入車両の進行方向 [rad]。灯器は heading + PI を向く */
  heading: number
  /** 交差点ごとの現示の番号（0..N-1）。同じ交差点で同じ値どうしが同時に青になる */
  group: number
  /** 進入路の幅 [m]。停止線・横断歩道の長さに使う */
  roadWidth: number
  /** OSM タグ由来・シミュレーターの補完・出典不明。 */
  source?: 'osm' | 'synthetic' | 'unknown'
}

export type SignKind = 'speed_limit' | 'stop' | 'crosswalk' | 'one_way' | 'mandatory_direction' | 'no_parking' | 'no_stopping'
export type SignDirection = 'straight' | 'left' | 'right' | 'left_or_straight' | 'right_or_straight' | 'left_or_right'

/** 種類と規制方向を持つ道路標識。 */
export interface MapSign {
  id: number
  /** 標識が立つ交差点ノード（MapNode.id）への参照 */
  nodeId: number
  /** この標識が規制する道路（MapEdge.id）への参照 */
  edgeId: number
  /** 支柱の位置（ENU）。左側通行なので進行方向左側の路端に立つ */
  x: number
  y: number
  /** *その標識が規制する側の進行方向** [rad]。標示板は heading + PI を向く */
  heading: number
  /** 最高速度標識の規制速度 [m/s]。ほかの種類は 0 */
  speedLimit: number
  /** 省略時は speed_limit */
  kind?: SignKind
  /** 標識の矢印が示す許可方向。省略時は straight */
  direction?: SignDirection
}

/** 2.2 map — マップ読込完了時（サイズが大きい・低頻度） */
export interface MapMessage {
  type: 'map'
  presetId: string
  name: string
  bounds: MapBounds
  nodes: MapNode[]
  edges: MapEdge[]
  buildings: MapBuilding[]
  /** 信号機。古いサーバーだと入っていないことがある */
  signals?: MapSignal[]
  /** 道路標識。古いサーバーだと入っていないことがある */
  signs?: MapSign[]
}

/** 信号の灯色。日本の車両用信号機は青→黄→赤の順に変わる */
export const SIGNAL_GREEN = 0
export const SIGNAL_YELLOW = 1
export const SIGNAL_RED = 2

/** frame に含まれる車両スロット 1 台分 */
export interface VehicleState {
  /** 0..maxVehicles-1 のスロット番号で固定 */
  id: number
  /** false のスロットは描画しない */
  active: boolean
  x: number
  y: number
  /** ラジアン。+x 軸（東）から反時計回り */
  heading: number
  /** m/s */
  speed: number
  /** rad（前輪舵角） */
  steer: number
  /** このステップで衝突したか（判定はバックエンド） */
  collided: boolean
  reachedGoal: boolean
  /** 目的地 [x, y] */
  goal: Vec2
  /** 経路の達成度 0.0〜1.0（経路始点からの進行距離 / 経路全長） */
  progress: number
  /** このエピソード中に赤信号で停止線を越えた回数 */
  signalViolations: number
  /** このエピソード中に車線を外れた回数（外れ始めた瞬間を 1 回と数える） */
  laneDepartures: number
  /** いま適用されている最高速度標識の規制速度 [m/s]。**標識が無い区間は 0.0** */
  speedLimit: number
  /** このエピソード中に規制速度を超えた回数（超え始めた瞬間を 1 回と数える） */
  speedViolations: number
  /** 制動指令が出ているか（ブレーキランプ）。**加速度の実測ではなく指令** */
  braking?: boolean
  /** 加速指令 -1..1（ペダルの踏み込み）。`braking` と同じ指令から出る */
  throttle?: number
  /** 方向指示器。-1=左 / 0=消灯 / +1=右 */
  turnSignal?: number
  /** 後退ギア（R）か。後退中は `speed` が負になる */
  reverse?: boolean
  /** 安全ギミックの介入（protocol.md 2.3）。介入していなければ空文字 */
  assist?: AssistKind
  /** V2X でこの車がメッセージを受け取った相手（近い順）。いなければ省略 */
  v2xConnectedIds?: number[]
  /** 階層型の方策がいま選んでいる意図（protocol.md 2.3）。方策が運転していない車は省略 */
  currentOption?: DriveOption
  /** 目的地までの経路。変化があったフレームのみ含まれる。省略時は前回値を保持 */
  route?: Vec2[]
}

/** 階層型の方策の意図。並びはサーバーの `config.HRL_OPTIONS`（上位方策の出力の添字）と同じ */
export const DRIVE_OPTIONS = ['CRUISE', 'FOLLOW', 'YIELD', 'STOP'] as const
export type DriveOption = (typeof DRIVE_OPTIONS)[number]

/** 安全ギミックの介入の種類。**バックエンドの `sim/safety.py` の ASSIST_* と同じ値** */
export type AssistKind =
  | ''
  | 'front_hold'
  | 'reverse_check'
  | 'reversing'
  | 'reverse_stop'
  | 'detour'
  | 'blind_spot'
  | 'peek'
  | 'creep'
  | 'yield'

/** 周囲カメラのキー。**バックエンドの `percep/types.py` の CAMERA_RIG と同じ値** */
export type SurroundKey = 'rear' | 'left' | 'right'

/** 1 台ぶんの周囲カメラの検出。まだ撮っていないカメラのキーは無い */
export type SurroundDetections = Partial<Record<SurroundKey, Detection[]>>

/** 見通しと死角の 1 台のカメラ（protocol.md 2.3）。方位はカメラの向きから測った rad（左が正） */
export interface OcclusionCamera {
  key: 'front' | SurroundKey
  /** カメラの位置。**自車座標系**（前方 +x / 左 +y、単位 m） */
  at: Vec2
  /** カメラの向き。車両の進行方向から測った rad（左が正） */
  yaw: number
  /** 見えている扇の列。[始まりの方位, 終わりの方位, 奥行き m] */
  seen: [number, number, number][]
}

/** 死角 1 つ。カメラから見た方位 [from, to] と、奥行き near から range までの扇 */
export interface OcclusionShadow {
  camera: 'front' | SurroundKey
  /** dynamic = 検出した車両の陰 / static = 走行可能距離の先（建物の陰・視程の外） */
  kind: 'dynamic' | 'static'
  from: number
  to: number
  near: number
}

/** 1 台ぶんの見通しと死角。4 台のカメラの検出と走行可能距離だけから作る（`percep/occlusion.py`） */
export interface OcclusionView {
  /** 見えている所・死角を考える半径 [m] */
  range: number
  /** 左・右のカメラの見通し距離 [m]。そのカメラの結果が無ければ null */
  los: [number | null, number | null]
  /** 前・後・左・右の 90 度の扇のうち見えている面積の割合 0..1 */
  sectors: [number, number, number, number]
  /** 前方カメラの画角のうち、検出した車両の陰の割合 0..1 */
  frontOccluded: number
  /** 結果のあるカメラだけ */
  cameras: OcclusionCamera[]
  shadows: OcclusionShadow[]
  /** いちばん近い遮蔽の角までの距離 [m]。無ければ省略 */
  corner?: number
}

export interface ObstacleState {
  id: number
  x: number
  y: number
  radius: number
}

/** 街を歩く NPC 歩行者 1 人（protocol.md 2.3）。 */
export interface NpcPedestrianState {
  /** スロット番号。消えるまで変わらない */
  id: number
  /** ENU x [m] */
  x: number
  /** ENU y [m] */
  y: number
  /** 体の向き [rad] */
  heading: number
  /** 手足の振りの位相 [rad]。**描画のためだけに送られる** */
  stride: number
  /** 車道を横断中か */
  crossing: boolean
}

/** 画像認識の検出クラス。**バックエンドの `percep.DetClass` と同じ並び**。 */
export const DET_TRAFFIC_LIGHT = 0
export const DET_SPEED_SIGN = 1
export const DET_VEHICLE = 2
export const DET_OBSTACLE = 3
export const DET_LANE = 4
export const DET_PEDESTRIAN = 5
export const DET_STOP_SIGN = 6
export const DET_CROSSWALK_SIGN = 7
export const DET_ONE_WAY_SIGN = 8
export const DET_MANDATORY_DIRECTION_SIGN = 9
export const DET_NO_PARKING_SIGN = 10
export const DET_NO_STOPPING_SIGN = 11

/** 擬似カメラ画像から認識器が見つけた物体 1 個（protocol.md 2.3）。 */
export interface Detection {
  /** DET_* のいずれか */
  cls: number
  /** 正規化画像座標 [x0, y0, x1, y1]（左上 0,0 / 右下 1,1） */
  box: [number, number, number, number]
  /** 信頼度 0.0〜1.0 */
  conf: number
  /** 信号のみ: 0=青 / 1=黄 / 2=赤 */
  phase?: number
  /** 最高速度標識のみ: 規制速度 [m/s]。表示は km/h に直す */
  speedLimit?: number
  /** 指定方向外進行禁止の標識のみ: 認識した許可方向 */
  direction?: SignDirection
  /** 推定距離 [m] */
  distance?: number
  /** 車線のみ: 車線中心からの横方向偏差 [m] */
  lateral?: number
  /** 車線のみ: 認識した車線中心線の点列。**自車座標系**（前方 +x / 左 +y、単位 m）。 */
  lanePoints?: [number, number][]
  /** 安全ギミックがこの検出を根拠に介入しているとき。1 = 注意 / 2 = これで止めている */
  hazard?: 1 | 2
}

/** 2.3 frame — 毎シミュレーションステップ（既定 20Hz） */
export interface FrameMessage {
  type: 'frame'
  /** マップ読込からの通算ステップ数（切り替えると 0 に戻る） */
  tick: number
  /** シミュレーション内経過秒 */
  simTime: number
  /** 常に全スロット分（maxVehicles 個） */
  vehicles: VehicleState[]
  obstacles: ObstacleState[]
  /** 街を歩く NPC 歩行者。0 人のときは省略される。 */
  pedestrians?: NpcPedestrianState[]
  /** 信号の現示。map.signals と同じ並びで 0=青 / 1=黄 / 2=赤。 */
  signals?: number[]
  /** 車両ごとの認識結果。**キーはスロット番号の文字列**（JSON のキーは文字列のため）。 */
  detections?: Record<string, Detection[]>
  /** 周囲カメラの検出。`watch_surround` で頼んだ車の分だけ載る（キーはスロット番号の文字列） */
  surround?: Record<string, SurroundDetections>
  /** 見通しと死角。`watch_occlusion` で頼んだ車の分だけ載る（キーはスロット番号の文字列） */
  occlusion?: Record<string, OcclusionView>
  /** いま効いている天候。weatherAuto の間は params ではなくこちらが正。 */
  weather?: WeatherState
}

/** 2.4 status */
export interface StatusMessage extends StatusPayload {
  type: 'status'
}

/** 2.5 params */
export interface ParamsMessage {
  type: 'params'
  params: SimParams
}

/** 2.6 metrics — 学習指標（既定 1Hz） */
export interface MetricsMessage {
  type: 'metrics'
  tick: number
  /** サーバー起動からの実時間秒 */
  wallTime: number
  /** PPO 更新回数 */
  updates: number
  episodes: number
  meanEpisodeReward: number
  meanEpisodeLength: number
  policyLoss: number
  valueLoss: number
  entropy: number
  approxKl: number
  /** 直近エピソードのうち衝突終了の割合 */
  collisionRate: number
  /** そのうち歩行者に当たった割合（`collisionRate` にも含まれる） */
  pedestrianCollisionRate?: number
  goalRate: number
  stepsPerSec: number
  /** 1 エピソードあたりの信号無視回数 */
  signalViolations: number
  /** 1 エピソードあたりの速度超過回数 */
  speedViolations: number
  /** 車線中心からの横方向のずれの平均 [m]。 */
  laneDeviation: number
  /** 直近 10 秒の学習中の車のステップのうち、エキスパートが運転した割合 0.0〜1.0（古いサーバーは送らない）。 */
  assistRate?: number
  /** 直前の PPO 更新での模倣の損失。 */
  bcLoss?: number
  /** ヒヤリハットの難易度 0.0〜1.0（古いサーバーは送らない）。 */
  curriculumLevel?: number
  /** いまのマップで起こしたヒヤリハットの件数。 */
  incidentsTriggered?: number
  /** 直近 50 件のヒヤリハットを自力で回避できた割合。まだ 1 件も見届けていなければ null。 */
  incidentsAvoidedRate?: number | null
  /** 直近 10 秒の方策が運転したステップで選ばれていた意図の割合（`DRIVE_OPTIONS` の順）。無ければ空配列。 */
  optionShares?: number[]
  /** 同じステップの加加速度の二乗平均平方根 [m/s^3]。方策が運転したステップが無ければ null。 */
  jerkRms?: number | null
}

/** 学習ジョブの段階 */
export type DetectorState =
  | 'idle'
  | 'preparing'
  | 'evaluating'
  | 'collecting'
  | 'training'
  | 'saving'
  | 'done'
  | 'error'
  | 'cancelled'

/** 何をするか。CLI の `--collect-only` / `--train-only` と同じ 3 通り */
export type DetectorMode = 'full' | 'collect' | 'train'

/** エポック 1 回ぶんの損失 */
export interface DetectorHistoryPoint {
  epoch: number
  loss: number
  valLoss: number
}

/** 学習の依頼内容（`start_detector_training` で送ったもののエコー） */
export interface DetectorRequest {
  mode: DetectorMode
  presetId: string | null
  samples: number
  epochs: number
  batchSize: number
  /** モデルのチャンネル倍率 */
  width: number
  /** 収集の乱数種。`SimulationEnv` と行動のランダム化の両方に渡るので、同じ種なら同じ画が集まる */
  seed: number
  /** 晴れ以外の天候も混ぜて集めるか。走行中の天候とは無関係。 */
  weatherMix: boolean
  /** 収集前に現行の認識器を採点し、弱点を狙って集めるか。 */
  focusWeak: boolean
}

/** 集めた教師データに何が写っているか。 */
export interface DetectorDataset {
  samples: number
  /** 物体があるセルの割合 0.0〜1.0 */
  objectCellRatio: number
  objectsPerImage: number
  /** クラス名（バックエンドの DetClass）-> 件数 */
  classCounts: Record<string, number>
  /** 天候プリセット名 -> 枚数 */
  weatherCounts: Record<string, number>
}

/** 収集前に測った、クラス 1 つぶんの成績。 */
export interface DetectorClassScore {
  cls: number
  name: string
  /** 真値にあった件数 */
  truth: number
  matched: number
  /** matched / truth */
  recall: number
  /** 灯色・規制速度を持つクラスのみ 0 より大きい */
  attributeTotal: number
  attributeOk: number
  attributeAccuracy: number
}

/** 収集前に測った、天候 1 つぶんの成績。 */
export interface DetectorWeatherScore {
  name: string
  truth: number
  matched: number
  recall: number
}

/** 収集前の採点。**これから学習するモデルではなく、いまの認識器の成績**。 */
export interface DetectorEvaluation {
  samples: number
  elapsedSec: number
  overallRecall: number
  classes: DetectorClassScore[]
  weathers: DetectorWeatherScore[]
  /** 弱点を 1 行で言ったもの */
  weakest: string
}

/** ディスク上のファイルの情報 */
export interface DetectorFileInfo {
  exists: boolean
  filename?: string
  sizeBytes: number
  modifiedAt: string | null
  /** 認識器のみ: いま観測を作るのに実際に使われているか */
  inUse?: boolean
}

/** サーバーが受け付ける値域。UI のスライダーはこれに合わせる */
export interface DetectorLimits {
  samplesMin: number
  samplesMax: number
  epochsMin: number
  epochsMax: number
  batchMin: number
  batchMax: number
  widthMin: number
  widthMax: number
  seedMin: number
  seedMax: number
}

/** 2.9 detector — 認識器の学習状況（進捗が動いたときだけ、最大 1Hz）。 */
export interface DetectorMessage {
  type: 'detector'
  state: DetectorState
  /** ジョブスレッドが走っているか */
  running: boolean
  message: string
  /** *いまの段階の**進捗 0.0〜1.0（段階をまたいで通算しない） */
  progress: number
  collected: number
  samples: number
  epoch: number
  epochs: number
  batch: number
  batches: number
  history: DetectorHistoryPoint[]
  elapsedSec: number
  /** 気づいたこと（写っていないクラスがある、何も検出しない等）。空なら問題なし */
  warning: string
  paramCount: number
  presetName: string | null
  request: DetectorRequest | null
  /** 収集前の採点。focusWeak が false か、認識器がまだ無いときは null。 */
  evaluation: DetectorEvaluation | null
  dataset: DetectorDataset | null
  model: DetectorFileInfo
  datasetFile: DetectorFileInfo
  limits: DetectorLimits
}

/** 実用モードの配車の段階（2.10 taxi）。 */
export type TaxiPhase = 'idle' | 'approaching' | 'waiting' | 'riding' | 'arrived'

/** 配車中の車の走り方（`contracts.TAXI_DRIVE_MODES`）。AI コンシェルジュが切り替える */
export type TaxiDriveMode = 'normal' | 'hurry' | 'comfort'

/** その段階で車に乗っているか。**`arrived` も乗ったまま**（降車は alight_taxi で確定する） */
export function isRidingPhase(phase: TaxiPhase): boolean {
  return phase === 'riding' || phase === 'arrived'
}

/** その段階で乗り込めるか（迎車の途中でも乗れる）。 */
export function isBoardablePhase(phase: TaxiPhase): boolean {
  return phase === 'approaching' || phase === 'waiting'
}

/** その段階で利用者を待っているか（ハザードを出す段階）。 */
export function isWaitingPhase(phase: TaxiPhase): boolean {
  return phase === 'waiting' || phase === 'arrived'
}

/** 2.10 taxi — 実用モードの配車状態（段階が変わったときと 1Hz）。 */
export interface TaxiMessage {
  type: 'taxi'
  phase: TaxiPhase
  /** 配車固有の識別子。迎車の引き継ぎでも維持し、idle は null */
  rideId: string | null
  /** 徴用している車両スロット。-1 なら無し */
  vehicleId: number
  /** 道路へスナップ済みの乗車地点（クライアントが送った座標ではない） */
  pickup: Vec2 | null
  dropoff: Vec2 | null
  /** 経路が変わるたびに増える。**変わらない限り前回の経路を保持すること** */
  routeRevision: number
  /** いま向かっている経路。routeRevision が変わった通だけ入る */
  route?: Vec2[]
  /** 到着まで [秒]。毎ステップ引き直される */
  etaSeconds: number
  remainingDistanceM: number
  message: string
  /** いまの走り方。配車が始まるたびに normal へ戻る */
  driveMode: TaxiDriveMode
}

/** 学習の自動化の段階。preparing は履歴の DB を開いている間、waiting は次の試行のパラメータを選んでいる間 */
export type AutotunePhase = 'idle' | 'preparing' | 'running' | 'waiting'

/** 試行の終わり方。pruned は見込みが薄く途中で打ち切った試行、diverged は重みが NaN / Inf になった試行 */
export type AutotuneOutcome = 'complete' | 'pruned' | 'diverged'

/** 自動探索の試行 1 つの結果 */
export interface AutotuneTrial {
  trial: number
  /** pruned は途中のスコア、diverged は最低点 */
  score: number | null
  outcome: AutotuneOutcome
}

/** この探索（ON にしてから）の最良の試行 */
export interface AutotuneBest {
  trial: number
  score: number
  params: Partial<SimParams>
}

/** 2.11 autotune — 学習の自動化（Optuna）の状態（接続直後と、変わったとき・1Hz）。 */
export interface AutotuneMessage {
  type: 'autotune'
  /** Optuna が入っているか。false ならトグルを押しても始まらない */
  available: boolean
  running: boolean
  phase: AutotunePhase
  studyName: string | null
  /** いま走らせている試行の番号（study の通し番号）。試行の合間は null */
  trial: number | null
  /** いまの試行の進み具合 0〜1 */
  trialProgress: number
  /** 1 試行のステップ数（20Hz） */
  trialSteps: number
  /** この探索で終えた試行の数（打ち切りも含む） */
  finishedTrials: number
  /** 同じ study に前から入っていた完了済みの試行の数（再開したとき） */
  priorTrials: number
  /** 探索している params のキー。**この間はスライダーを動かせない** */
  tunedKeys: Array<keyof SimParams>
  /** いまの試行のパラメータ。試行の合間は null */
  current: Partial<SimParams> | null
  best: AutotuneBest | null
  /** 直近の試行（古い順・最大 20 件） */
  history: AutotuneTrial[]
  message: string
}

/** `docs/protocol.md` のエラーコード表が唯一の出典。 */
export type ErrorCode =
  | 'MAP_LOAD_FAILED'
  | 'INVALID_MESSAGE'
  /** 認識器の学習中に `load_map`・`set_app_mode`（taxi へ）・`request_taxi`・`board_taxi` が来た */
  | 'DETECTOR_TRAINING'

/** 2.8 error */
export interface ErrorMessage {
  type: 'error'
  code: ErrorCode
  message: string
}

/** ping への応答（3章に記載） */
export interface PongMessage {
  type: 'pong'
  /** server epoch ms */
  t: number
}

/** ネットワークの 1 層ぶんの要約 */
export interface NetworkLayer {
  /** パラメータ名（`.weight` を除いたもの）。例: `policy_trunk.0` */
  name: string
  /** 方策側か価値側か */
  role: 'policy' | 'value'
  inDim: number
  outDim: number
  /** 重みの絶対値の平均。学習が進むと動く */
  weightAbsMean: number
  weightStd: number
  /** 直近の更新で流れた勾配の大きさ（**クリップ前**）。0 なら学習していない */
  gradNorm: number
  /** **直前の 1 更新で重みが動いた量。** 更新ごとに確定する。 */
  deltaNorm: number
}

/** 2.7 network — ネットワークの状態（既定 1Hz） */
export interface NetworkMessage {
  type: 'network'
  updates: number
  obsDim: number
  actionDim: number
  hiddenSizes: number[]
  layers: NetworkLayer[]
  /** 探索ノイズの対数標準偏差。行動の次元ぶんある */
  logStd: number[]
  /** logStd を可動域でクランプして exp したもの（実際に使われる std） */
  actionStd: number[]
  logStdMin: number
  logStdMax: number
  /** クリップ前の勾配の全体ノルム（ミニバッチ平均） */
  gradTotalNorm?: number
  /** クリップが効いたミニバッチの割合（0〜1）。1 に張り付いていれば暴れている */
  gradClipRate?: number
  /** クリップの上限（`config.PPO_MAX_GRAD_NORM`） */
  gradMaxNorm?: number
}

/** サーバー → クライアントの全メッセージ（type による判別可能ユニオン） */
export type ServerMessage =
  | InitMessage
  | MapMessage
  | FrameMessage
  | StatusMessage
  | ParamsMessage
  | MetricsMessage
  | NetworkMessage
  | DetectorMessage
  | TaxiMessage
  | AutotuneMessage
  | ErrorMessage
  | PongMessage

export interface LoadMapMessage {
  type: 'load_map'
  presetId: string
}

/** 部分更新。渡したキーのみ反映される */
export interface SetParamsMessage {
  type: 'set_params'
  params: Partial<SimParams>
}

/** 最寄りの道路上にスナップされる */
export interface SpawnVehicleMessage {
  type: 'spawn_vehicle'
  x: number
  y: number
}

export interface DespawnVehicleMessage {
  type: 'despawn_vehicle'
  id: number
}

export interface AddObstacleMessage {
  type: 'add_obstacle'
  x: number
  y: number
  radius: number
}

export interface RemoveObstacleMessage {
  type: 'remove_obstacle'
  id: number
}

export interface ClearObstaclesMessage {
  type: 'clear_obstacles'
}

/** 描画のみ停止。学習は継続する */
export interface SetRenderPausedMessage {
  type: 'set_render_paused'
  paused: boolean
}

export interface ResetEpisodeMessage {
  type: 'reset_episode'
}

export interface SaveCheckpointMessage {
  type: 'save_checkpoint'
}

export interface LoadCheckpointMessage {
  type: 'load_checkpoint'
}

/** 重みを初期化して学習をやり直す（破壊的） */
export interface ResetPolicyMessage {
  type: 'reset_policy'
}

export interface PingMessage {
  type: 'ping'
}

/** 認識器（CNN）の学習を始める（操作パネルの「モデル作成」タブ）。 */
export interface StartDetectorTrainingMessage {
  type: 'start_detector_training'
  request: DetectorRequest
}

/** 学習の中断を要求する。**すぐには止まらない**（バッチ／ステップの境界で抜ける） */
export interface CancelDetectorTrainingMessage {
  type: 'cancel_detector_training'
}

/** 隠れ層の構成を変える。 */
export interface SetNetworkMessage {
  type: 'set_network'
  hiddenSizes: number[]
}

/** 学習の自動化（Optuna の探索）を始める。断るときは理由が status で返る */
export interface StartAutoTuneMessage {
  type: 'start_auto_tune'
}

/** 学習の自動化をやめ、最良の試行のパラメータと重みを適用して保存する（元の重みは .before-autotune へ退避） */
export interface StopAutoTuneAndSaveMessage {
  type: 'stop_auto_tune_and_save'
}

/** 開発モードと実用モードを切り替える。**学習の可否だけが変わる**（物理は止まらない） */
export interface SetAppModeMessage {
  type: 'set_app_mode'
  mode: 'dev' | 'taxi'
}

/** 配車を呼ぶ。乗降地点はサーバー側で最寄りの道路へスナップされる */
export interface RequestTaxiMessage {
  type: 'request_taxi'
  pickup: Vec2
  dropoff: Vec2
}

/** 乗車する（phase=waiting / approaching で通る） */
export interface BoardTaxiMessage {
  type: 'board_taxi'
}

/** 降車する（phase=riding / arrived で通る） */
export interface AlightTaxiMessage {
  type: 'alight_taxi'
}

/** 配車を取り消す。`halt` は [space] の緊急停止（その場で速度 0 にしてから解除） */
export interface CancelTaxiMessage {
  type: 'cancel_taxi'
  halt?: boolean
}

/** 徒歩キャラの位置を知らせる。`at` が null なら街から消える（乗車中・実用モードを抜けたとき） */
export interface PlayerPoseMessage {
  type: 'player_pose'
  at: Vec2 | null
}

/** その車の周囲カメラの検出を frame に載せてもらう。**4 分割表示の間は 1 秒ごとに送り直す**（2.5 秒で切れる） */
export interface WatchSurroundMessage {
  type: 'watch_surround'
  vehicleId: number
}

/** その車の見通しと死角を frame に載せてもらう。**視野コーン・死角を出している間は 1 秒ごとに送り直す**（2.5 秒で切れる） */
export interface WatchOcclusionMessage {
  type: 'watch_occlusion'
  vehicleId: number
}

/** クライアント → サーバーの全メッセージ */
export type ClientMessage =
  | LoadMapMessage
  | SetParamsMessage
  | SpawnVehicleMessage
  | DespawnVehicleMessage
  | AddObstacleMessage
  | RemoveObstacleMessage
  | ClearObstaclesMessage
  | SetRenderPausedMessage
  | ResetEpisodeMessage
  | SaveCheckpointMessage
  | LoadCheckpointMessage
  | ResetPolicyMessage
  | SetNetworkMessage
  | StartDetectorTrainingMessage
  | CancelDetectorTrainingMessage
  | StartAutoTuneMessage
  | StopAutoTuneAndSaveMessage
  | SetAppModeMessage
  | RequestTaxiMessage
  | BoardTaxiMessage
  | AlightTaxiMessage
  | CancelTaxiMessage
  | PlayerPoseMessage
  | WatchSurroundMessage
  | WatchOcclusionMessage
  | PingMessage

/** WebSocket の接続状態 */
export type ConnectionState = 'connecting' | 'open' | 'closed'
