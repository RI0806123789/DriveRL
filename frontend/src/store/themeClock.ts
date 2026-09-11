/**
 * 「いま何時だから、どちらの配色か」を決める純粋なロジック。
 *
 * React にも zustand にも依存させない。`store/autoTheme.ts`（React のフック）と
 * `store/simStore.ts`（初期値）の両方から読むため、ここを独立させておかないと
 * import が循環する。
 *
 * 場所の決め方は `autoTheme.ts` の冒頭を参照。要点は
 * **走らせている街の座標を使い、無ければタイムゾーンから経度を逆算する**こと。
 */

import { isDaylight, nextSunEvent } from '../scene/sunTimes'
import type { ThemeName } from '../scene/palette'

export interface SunLocation {
  readonly lat: number
  readonly lon: number
  /** どこから得た座標か。ログや説明に出す */
  readonly source: string
}

/**
 * タイムゾーンから逆算した、おおよその現在地。
 *
 * 標準時は経度 15 度ごとに 1 時間ずれるので、UTC からの差を 4 で割れば経度が出る
 * （日本標準時 UTC+9 → 東経 135 度）。
 * **緯度は分からないので 0 度（赤道）として扱う。** 実質「6 時〜18 時が昼」になり、
 * 季節による前後は表現できないが、座標が届くまでの数百ミリ秒しか使わない。
 */
export function timezoneLocation(): SunLocation {
  let offsetMin = 0
  try {
    // getTimezoneOffset() は「UTC からの遅れ（分）」。JST は -540
    offsetMin = new Date().getTimezoneOffset()
  } catch {
    // 取れなければ本初子午線とみなす
  }
  const lon = Math.max(-180, Math.min(180, -offsetMin / 4))
  return { lat: 0, lon, source: 'タイムゾーンからの推定' }
}

/** その場所・その時刻に対する配色 */
export function themeFor(at: Date, where: SunLocation): ThemeName {
  try {
    return isDaylight(at, where.lat, where.lon) ? 'light' : 'dark'
  } catch {
    return 'dark'
  }
}

/**
 * 次に配色が変わる瞬間。白夜・極夜のように切り替わりが無ければ null。
 */
export function nextThemeChange(at: Date, where: SunLocation): Date | null {
  try {
    return nextSunEvent(at, where.lat, where.lon)
  } catch {
    return null
  }
}

/**
 * 画面を出す前に決めておく初期値。
 *
 * プリセットの座標が届く前の一瞬に「夜なのに白い画面が光る」のを避けるため、
 * タイムゾーンだけで先に見当を付ける。届いた時点で本来の座標へ切り替わる。
 */
export function bootstrapTheme(): ThemeName {
  return themeFor(new Date(), timezoneLocation())
}
