/** 3D シーンの色。**3D 側の色はここだけに書く。** */

export type ThemeName = 'dark' | 'light'

export interface ScenePalette {
  /** 背景色。フォグも同じ色にして遠景を溶かす */
  readonly sky: string
  /** フォグの開始・終了距離 [m] */
  readonly fogNear: number
  readonly fogFar: number
  /** 雨のときに空とフォグを寄せる色。暗く濁らせる */
  readonly rainSky: string
  /** 霧のときに空とフォグを寄せる色。擬似カメラ側の霧色と揃える */
  readonly fogVeil: string
  /** 雨粒の色 */
  readonly rainDrop: string
  /** 地面の色 */
  readonly ground: string
  /** グリッドの色（中心線・目盛り） */
  readonly gridMajor: string
  readonly gridMinor: string

  /** 半球光の空側・地面側と強さ */
  readonly hemiSky: string
  readonly hemiGround: string
  readonly hemiIntensity: number
  /** 環境光の強さ */
  readonly ambientIntensity: number
  /** 太陽光（影を落とす平行光）の色と強さ */
  readonly sun: string
  readonly sunIntensity: number

  /** 低層 → 高層のグラデーション */
  readonly buildingLow: string
  readonly buildingHigh: string

  /** 舗装面。昼でもアスファルトは暗い */
  readonly roadSurface: string
  /** 車道中央線（黄色） */
  readonly roadCenterline: string
  /** 白い道路標示（車線境界線・停止線・横断歩道） */
  readonly marking: string

  /** 灯器の筐体と支柱 */
  readonly signalHousing: string
  /** 消灯している灯火 */
  readonly signalLampOff: string

  /** 標示板の白地 */
  readonly signBoard: string
  /** 標示板の赤縁 */
  readonly signRing: string
  /** 標示板の黒数字 */
  readonly signText: string
  /** 支柱。こちらは背景に馴染ませたいので昼夜で変える */
  readonly signPole: string

  /** タイヤ */
  readonly vehicleWheel: string
  /** ガラス（キャビン） */
  readonly vehicleGlass: string
  /** 衝突中に寄せる色 */
  readonly vehicleCollided: string
  /** 目的地に着いたときに寄せる色 */
  readonly vehicleReached: string
  /** 追従対象の足元リング */
  readonly vehicleHighlight: string

  readonly obstacleCone: string
  readonly obstacleBase: string

  /** CNN が認識した車線（LaneDetectionOverlay）の帯・中心線の色。 */
  readonly laneOverlay: string
}

/** 夜。もとからあった配色をそのまま残してある */
export const DARK_SCENE: ScenePalette = {
  sky: '#0e1216',
  fogNear: 600,
  fogFar: 2600,
  rainSky: '#070a0d',
  fogVeil: '#39424b',
  rainDrop: '#9fb6c9',
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

  laneOverlay: '#ff4fd1',
}

/** 昼。 */
export const LIGHT_SCENE: ScenePalette = {
  sky: '#a9c4dd',
  fogNear: 700,
  fogFar: 2800,
  rainSky: '#6d7a88',
  fogVeil: '#ced2d6',
  rainDrop: '#e8eef4',
  ground: '#9fa8a6',
  gridMajor: '#767f86',
  gridMinor: '#8f979d',

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

  laneOverlay: '#c2158f',
}

/** テーマ名から配色を引く */
export function scenePalette(theme: ThemeName): ScenePalette {
  return theme === 'light' ? LIGHT_SCENE : DARK_SCENE
}

/** 信号の灯火（0=青 / 1=黄 / 2=赤）。日本の LED 信号機の見え方に寄せている */
export const SIGNAL_LAMP_COLORS = ['#00b06e', '#f2b700', '#e8302a'] as const
