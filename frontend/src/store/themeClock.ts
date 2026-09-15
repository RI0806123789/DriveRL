/** 「いま何時だから、どちらの配色か」を決める純粋なロジック。 */

import { isDaylight, nextSunEvent } from '../scene/sunTimes'
import type { ThemeName } from '../scene/palette'

export interface SunLocation {
  readonly lat: number
  readonly lon: number
  /** どこから得た座標か。ログや説明に出す */
  readonly source: string
}

/** タイムゾーンから逆算した、おおよその現在地。 */
export function timezoneLocation(): SunLocation {
  let offsetMin = 0
  try {
    offsetMin = new Date().getTimezoneOffset()
  } catch {
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

/** 次に配色が変わる瞬間。白夜・極夜のように切り替わりが無ければ null。 */
export function nextThemeChange(at: Date, where: SunLocation): Date | null {
  try {
    return nextSunEvent(at, where.lat, where.lon)
  } catch {
    return null
  }
}

/** 画面を出す前に決めておく初期値。 */
export function bootstrapTheme(): ThemeName {
  return themeFor(new Date(), timezoneLocation())
}
