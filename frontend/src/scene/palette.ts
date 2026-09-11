/**
 * 3D シーンの色。**3D 側の色はここだけに書く。**
 *
 * もともと空・フォグ・地面の色は各コンポーネントに直接書かれていて、
 * `styles/tokens.css` の `--m3-scene-*` とも食い違っていた
 * （トークンは `#0d1218` なのに実際の空は `#0e1216`）。
 * CSS 変数は 3D からは読まれないので、片方を直しても永久に気づけない。
 * そこで色は TypeScript 側の 1 か所に寄せ、CSS からは `--m3-scene-*` を削除した。
 *
 * **昼と夜の 2 組を持つ。** どちらを使うかは `store/simStore` の `theme` が決め、
 * それは `sunTimes.ts` が計算する日の出・日の入りに連動する（利用者の設定項目は無い）。
 * パネル UI 側は `styles/tokens.css` の `:root[data-theme]` が担当していて、
 * **同じ `theme` を見ている**が、色の実体は別々に持っている
 * （CSS 変数は three から読めないため）。
 *
 * three / React に依存させないでおくと Node からも読める。
 */

export type ThemeName = 'dark' | 'light'

export interface ScenePalette {
  // --- 空と地面 ---
  /** 背景色。フォグも同じ色にして遠景を溶かす */
  readonly sky: string
  /** フォグの開始・終了距離 [m] */
  readonly fogNear: number
  readonly fogFar: number
  /** 地面の色 */
  readonly ground: string
  /** グリッドの色（中心線・目盛り） */
  readonly gridMajor: string
  readonly gridMinor: string

  // --- 光 ---
  /** 半球光の空側・地面側と強さ */
  readonly hemiSky: string
  readonly hemiGround: string
  readonly hemiIntensity: number
  /** 環境光の強さ */
  readonly ambientIntensity: number
  /** 太陽光（影を落とす平行光）の色と強さ */
  readonly sun: string
  readonly sunIntensity: number

  // --- 建物 ---
  /** 低層 → 高層のグラデーション */
  readonly buildingLow: string
  readonly buildingHigh: string

  // --- 道路 ---
  /** 舗装面。昼でもアスファルトは暗い */
  readonly roadSurface: string
  /** 車道中央線（黄色）*/
  readonly roadCenterline: string
  /** 白い道路標示（車線境界線・停止線・横断歩道）*/
  readonly marking: string

  // --- 信号機 ---
  /** 灯器の筐体と支柱 */
  readonly signalHousing: string
  /** 消灯している灯火 */
  readonly signalLampOff: string

  // --- 最高速度標識 ---
  // ★ 標示板の 3 色（白地・赤縁・黒数字）は昼夜で変えない。
  //   信号の灯火と同じで、現実の標識がそうであるように、
  //   環境によって色が変わっては規制の意味を成さない。
  //   そのため DARK_SCENE / LIGHT_SCENE の両方で同じ値を入れてある。
  /** 標示板の白地 */
  readonly signBoard: string
  /** 標示板の赤縁 */
  readonly signRing: string
  /** 標示板の黒数字 */
  readonly signText: string
  /** 支柱。こちらは背景に馴染ませたいので昼夜で変える */
  readonly signPole: string

  // --- 車両 ---
  /** タイヤ */
  readonly vehicleWheel: string
  /** ガラス（キャビン）*/
  readonly vehicleGlass: string
  /** 衝突中に寄せる色 */
  readonly vehicleCollided: string
  /** 目的地に着いたときに寄せる色 */
  readonly vehicleReached: string
  /** 追従対象の足元リング */
  readonly vehicleHighlight: string

  // --- 障害物（パイロン）---
  readonly obstacleCone: string
  readonly obstacleBase: string

  // --- 認識オーバーレイ ---
  /**
   * CNN が認識した車線（LaneDetectionOverlay）の帯・中心線の色。
   * ★ 実際の白線標示（marking）の上に重ねるので、昼夜どちらでも白系統とは
   *   はっきり見分けが付く色にすること。
   */
  readonly laneOverlay: string
}

/** 夜。もとからあった配色をそのまま残してある */
export const DARK_SCENE: ScenePalette = {
  sky: '#0e1216',
  fogNear: 600,
  fogFar: 2600,
  ground: '#1a1f24',
  gridMajor: '#4a545e',
  gridMinor: '#2b333b',

  hemiSky: '#9fc4e8',
  hemiGround: '#1a1f24',
  hemiIntensity: 1.15,
  ambientIntensity: 0.35,
  sun: '#fff3e0',
  sunIntensity: 2.1,

  buildingLow: '#39424c',
  buildingHigh: '#5d6b7c',

  roadSurface: '#3a4149',
  roadCenterline: '#d9c98a',
  marking: '#e8ebee',

  signalHousing: '#4a5560',
  signalLampOff: '#14181c',

  // 標示板の 3 色は LIGHT_SCENE と同じ値（現実の標識と同じ色を昼夜で保つ）
  signBoard: '#f2f4f5',
  signRing: '#d0121b',
  signText: '#16191c',
  signPole: '#6b7885',

  vehicleWheel: '#15181c',
  vehicleGlass: '#0f1b26',
  vehicleCollided: '#ff5a4d',
  vehicleReached: '#3ad2a0',
  vehicleHighlight: '#7adcc8',

  obstacleCone: '#ff7a3d',
  obstacleBase: '#20262c',

  // 白線(marking #e8ebee)・路面(#3a4149)のどちらとも被らない鮮やかなマゼンタ
  laneOverlay: '#ff4fd1',
}

/**
 * 昼。
 *
 * 単純に反転させると読めなくなるものがあるので、実物に寄せてある。
 * - **アスファルトは昼でも暗い。** 明るくすると道路と歩道の区別が付かなくなる
 * - **白線は白のまま。** 夜は少し落としてあるが、昼は塗料どおりの白でよい
 * - **建物は高いほど明るい。** 遠景の霞（空気遠近法）と向きが揃う
 * - フォグは夜より遠くまで効かせる。晴れた昼のほうが見通しが利くため
 */
export const LIGHT_SCENE: ScenePalette = {
  sky: '#a9c4dd',
  fogNear: 700,
  fogFar: 2800,
  ground: '#9fa8a6',
  gridMajor: '#767f86',
  gridMinor: '#8f979d',

  // ★ 光の総量は夜より **少なく** する。
  //   R3F の既定のトーンマッピング（ACESFilmic）はハイライトを圧縮するので、
  //   明るい基準色に夜と同じ光量を当てると建物が一様な白に潰れて
  //   低層と高層の差も影も見えなくなる（実測で確認した）。
  //   合計 2.65（夜は 3.60）。
  hemiSky: '#cfe0f2',
  hemiGround: '#9fa8a6',
  hemiIntensity: 0.62,
  ambientIntensity: 0.28,
  sun: '#fff4df',
  sunIntensity: 1.75,

  buildingLow: '#7e8993',
  buildingHigh: '#b4bec8',

  roadSurface: '#4e555c',
  roadCenterline: '#d9bb3f',
  marking: '#f5f7f9',

  signalHousing: '#39434a',
  signalLampOff: '#1b2126',

  // 標示板の 3 色は DARK_SCENE と同じ値（現実の標識と同じ色を昼夜で保つ）
  signBoard: '#f2f4f5',
  signRing: '#d0121b',
  signText: '#16191c',
  signPole: '#525c66',

  vehicleWheel: '#1c2024',
  vehicleGlass: '#4a6a8a',
  vehicleCollided: '#d43526',
  vehicleReached: '#0f9a6d',
  vehicleHighlight: '#0b8f79',

  obstacleCone: '#ef5416',
  obstacleBase: '#2f353a',

  // 白線(marking #f5f7f9)・路面(#4e555c)のどちらとも被らない濃いマゼンタ
  laneOverlay: '#c2158f',
}

/** テーマ名から配色を引く */
export function scenePalette(theme: ThemeName): ScenePalette {
  return theme === 'light' ? LIGHT_SCENE : DARK_SCENE
}

/**
 * 信号の灯火（0=青 / 1=黄 / 2=赤）。日本の LED 信号機の見え方に寄せている
 * （「青」は実際には緑）。
 *
 * ★ **昼夜で変えないので `ScenePalette` には入れない。** 現実の信号がそうであるように、
 *   環境によって色が変わっては意味を成さない。
 *   灯器（`TrafficSignals`）と認識結果の枠（`detectionLabels`）が同じ値を使うため、
 *   ここを唯一の出どころにしてある。
 */
export const SIGNAL_LAMP_COLORS = ['#00b06e', '#f2b700', '#e8302a'] as const
