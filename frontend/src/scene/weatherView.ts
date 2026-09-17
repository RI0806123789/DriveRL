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

/**
 * 視点がこの高さ [m] を超えたら、超えたぶんだけ霧を薄く見せる。
 *
 * 視程は**運転席から前を見たときの距離**なので、300m 上空の俯瞰へそのまま
 * 適用すると地面が完全に霧へ沈み、街が 1 つも見えなくなる（実際にそうなった）。
 * 運転席・追従（数 m）では視程どおり、俯瞰では「遠くが霞む」程度に留める。
 */
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
}

/** 0 と 1 の間を滑らかにつなぐ（両端で傾きが 0）。 */
export function smoothstep(t: number): number {
  const x = Math.min(Math.max(t, 0), 1)
  return x * x * (3 - 2 * x)
}

/**
 * 晴れの値と霧の値を**幾何補間**する。
 *
 * 晴れのフォグは遠景を溶かすための数 km、霧のフォグは視程の数十 m と桁が
 * 違うので、線形に混ぜると霧が濃くなるまでほとんど効かない（fog=0.5 でも
 * まだ数百 m 先まで見える）。対数の上で混ぜると、薄い霧から素直に効く。
 */
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
  weather: WeatherState,
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
  }
}
