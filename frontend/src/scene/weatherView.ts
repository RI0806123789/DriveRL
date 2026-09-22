/** 天候を 3D の見た目（フォグ・明るさ・雨粒）へ落とす計算。React から切り離してある。 */

import type { WeatherState } from '../types/protocol'

/** 霧が効き始める強さ。これ未満は晴れとして扱う */
export const FOG_EPS = 0.004

/** 雨が効き始める強さ */
export const RAIN_EPS = 0.004

/** 視程の何倍でフォグが完全に閉じるか（擬似カメラは視程で透過率 13.5%）*/
const FOG_FAR_RATIO = 1.25

/** 視程の何倍からフォグが乗り始めるか */
const FOG_NEAR_RATIO = 0.06

/** 濃霧でも最低これだけは見える [m]。真っ白で何も分からない画を避ける */
const FOG_FAR_MIN_M = 9

/** 視点がこの高さ [m] を超えたら、超えたぶんだけ霧を薄く見せる。 */
const FOG_EYE_REF_M = 12

/** 雨・霧でどこまで暗くするか（1.0 = 変化なし）*/
const RAIN_DIM = 0.42
const FOG_DIM = 0.3

/** 3D シーンへ適用する値。 */
export interface WeatherLook {
  /** フォグと背景の色を、この割合だけ雨の色へ寄せる 0..1 */
  rainTint: number
  /** 同じく霧の色へ寄せる割合 0..1 */
  fogTint: number
  /** フォグの開始距離 [m] */
  fogNear: number
  /** フォグの終了距離 [m] */
  fogFar: number
  /** ライトの強さに掛ける係数 0..1 */
  dim: number
  /** 雨粒の不透明度 0..1。0 なら描かない */
  dropOpacity: number
  /** 路面の濡れ具合 0..1。上げるほど暗く、つやが出る */
  wetness: number
}

/** 濡れた路面の粗さ・金属感・暗さ（乾いた状態からの行き先）。 */
export const WET_ROUGHNESS = 0.26
export const WET_METALNESS = 0.1
export const WET_DARKEN = 0.66

/** 濡れた路面へ映り込ませる空の割合。上げすぎると路面が空と同じ色になる */
export const WET_SKY_MIX = 0.14

/** 雨がやんでも路面はすぐには乾かない [秒] */
const DRY_TAU_SEC = 9

/** 天候の切り替えをこの時定数で追わせる [秒] */
const SHIFT_TAU_SEC = 1.6

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * Math.min(Math.max(t, 0), 1)
}

/** 指数で目標へ近づく 1 ステップぶん（フレームレートに依らない）。 */
export function approach(current: number, target: number, dt: number, tau: number): number {
  if (tau <= 0) return target
  return current + (target - current) * (1 - Math.exp(-Math.max(dt, 0) / tau))
}

/** 画面に出ている天候。 */
export const displayedWeather = { rain: 0, fog: 0, visibility: 120, wet: 0 }

/** 表示用の天候を 1 フレーム進める。戻り値は `displayedWeather` そのもの。 */
export function advanceWeather(
  target: { rain: number; fog: number; visibility: number },
  dt: number,
): typeof displayedWeather {
  const d = displayedWeather
  d.rain = approach(d.rain, target.rain, dt, SHIFT_TAU_SEC)
  d.fog = approach(d.fog, target.fog, dt, SHIFT_TAU_SEC)
  d.visibility = approach(d.visibility, target.visibility, dt, SHIFT_TAU_SEC)
  // 濡れるのは雨に追随し、乾くのはゆっくり
  d.wet = approach(d.wet, target.rain, dt, target.rain > d.wet ? SHIFT_TAU_SEC : DRY_TAU_SEC)
  return d
}

/** 0 と 1 の間を滑らかにつなぐ（両端で傾きが 0）。 */
export function smoothstep(t: number): number {
  const x = Math.min(Math.max(t, 0), 1)
  return x * x * (3 - 2 * x)
}

/** 晴れの値と霧の値を**幾何補間**する。 */
export function blendDistance(clear: number, foggy: number, t: number): number {
  if (t <= 0) return clear
  if (t >= 1) return foggy
  return Math.exp(Math.log(clear) * (1 - t) + Math.log(Math.max(foggy, 1e-3)) * t)
}

/** 視点の高さから、霧の距離に掛ける倍率を出す。 */
export function fogLift(eyeHeight: number): number {
  return Math.max(1, Math.abs(eyeHeight) / FOG_EYE_REF_M)
}

/** 天候と配色から、シーンへ入れる値を作る。`eyeHeight` はカメラの地上高 [m]。 */
export function weatherLook(
  weather: WeatherState & { wet?: number },
  clearNear: number,
  clearFar: number,
  eyeHeight = 0,
): WeatherLook {
  const rain = Math.min(Math.max(weather.rain, 0), 1)
  const fog = Math.min(Math.max(weather.fog, 0), 1)
  const visibility = Math.max(weather.visibility, 1)

  const t = smoothstep(fog)
  const lift = fogLift(eyeHeight)
  const foggyFar = Math.max(visibility * FOG_FAR_RATIO * lift, FOG_FAR_MIN_M)
  const foggyNear = Math.max(visibility * FOG_NEAR_RATIO * lift, 0.5)

  return {
    rainTint: rain * 0.55,
    fogTint: t,
    fogNear: fog < FOG_EPS ? clearNear : blendDistance(clearNear, foggyNear, t),
    fogFar: fog < FOG_EPS ? clearFar : blendDistance(clearFar, foggyFar, t),
    dim: Math.max(1 - RAIN_DIM * rain - FOG_DIM * t, 0.2),
    dropOpacity: rain < RAIN_EPS ? 0 : 0.25 + 0.45 * rain,
    wetness: Math.min(Math.max(weather.wet ?? rain, 0), 1),
  }
}
