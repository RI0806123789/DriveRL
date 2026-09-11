/**
 * 日の出・日の入りの計算。
 *
 * 配色を時刻に連動させるために使う。**外部 API は叩かない。**
 * 天文計算は緯度・経度・日時だけで閉じるので、オフラインでも同じ答えが出る
 * （このアプリ全体がローカル完結という前提のため、ここで通信を入れない）。
 *
 * 方式は NOAA の日の出方程式。`Date` 以外に依存しないので Node からも読め、
 * `npm run verify:sun` で不変条件を数値で確かめている。
 *
 * 精度は日本の緯度で実測 1 分以内。高緯度では大気差の扱いで数分ずれるが、
 * 配色の切り替えに使うだけなので問題にならない。
 */

const DAY_MS = 86_400_000
/** Unix エポック（1970-01-01T00:00:00Z）のユリウス日 */
const JULIAN_EPOCH = 2_440_587.5
/** J2000.0（2000-01-01T12:00:00 TT）のユリウス日 */
const J2000 = 2_451_545.0

const RAD = Math.PI / 180
/** 地球の赤道傾斜角 [rad] */
const OBLIQUITY = 23.4397 * RAD

/**
 * 日の出・日の入りとみなす太陽高度 [rad]。
 *
 * -0.833 度は「太陽の上端が地平線に接する瞬間」で、
 * 太陽の視半径 16 分と標準大気差 34 分の合計。各国の暦がこの定義を使う。
 * このぶん昼が長くなるので、**春分・秋分の昼は 12 時間ちょうどより 7 分ほど長い。**
 */
const HORIZON = -0.833 * RAD

/** ユリウス日への変換 */
function toJulian(date: Date): number {
  return date.getTime() / DAY_MS + JULIAN_EPOCH
}

function fromJulian(julian: number): Date {
  return new Date((julian - JULIAN_EPOCH) * DAY_MS)
}

/** J2000 からの経過日数 */
function toDays(date: Date): number {
  return toJulian(date) - J2000
}

/** 太陽の平均近点角 [rad] */
function solarMeanAnomaly(days: number): number {
  return RAD * (357.5291 + 0.985_600_28 * days)
}

/** 黄道座標での太陽の黄経 [rad] */
function eclipticLongitude(meanAnomaly: number): number {
  // 中心差（軌道が円でないぶんのずれ）
  const center =
    RAD *
    (1.9148 * Math.sin(meanAnomaly) +
      0.02 * Math.sin(2 * meanAnomaly) +
      0.0003 * Math.sin(3 * meanAnomaly))
  // 近日点の黄経
  const perihelion = 102.9372 * RAD
  return meanAnomaly + center + perihelion + Math.PI
}

/** 太陽の赤緯 [rad] */
function declination(eclipticLon: number): number {
  return Math.asin(Math.sin(OBLIQUITY) * Math.sin(eclipticLon))
}

/** 平均太陽時での「その日」の通し番号 */
function julianCycle(days: number, lonWestRad: number): number {
  return Math.round(days - 0.0009 - lonWestRad / (2 * Math.PI))
}

function approxTransit(hourAngle: number, lonWestRad: number, cycle: number): number {
  return 0.0009 + (hourAngle + lonWestRad) / (2 * Math.PI) + cycle
}

function solarTransitJulian(approx: number, meanAnomaly: number, eclipticLon: number): number {
  return (
    J2000 + approx + 0.0053 * Math.sin(meanAnomaly) - 0.0069 * Math.sin(2 * eclipticLon)
  )
}

export interface SunTimes {
  /** 日の出。白夜・極夜では null */
  readonly sunrise: Date | null
  /** 日の入り。白夜・極夜では null */
  readonly sunset: Date | null
  /** 南中。緯度によらず必ず求まる */
  readonly solarNoon: Date
  /** 一日中沈まない（白夜） */
  readonly alwaysUp: boolean
  /** 一日中昇らない（極夜） */
  readonly alwaysDown: boolean
}

/**
 * 指定した日時を含む太陽日の、日の出・日の入り・南中を求める。
 *
 * @param date  基準の日時
 * @param latitude   緯度 [度]。北が正
 * @param longitude  経度 [度]。**東が正**（日本は +135 前後）
 */
export function sunTimes(date: Date, latitude: number, longitude: number): SunTimes {
  // 日の出方程式は西経を正で扱うので符号を反転する
  const lonWest = -longitude * RAD
  const phi = latitude * RAD

  const days = toDays(date)
  const cycle = julianCycle(days, lonWest)
  const approxNoon = approxTransit(0, lonWest, cycle)
  const meanAnomaly = solarMeanAnomaly(approxNoon)
  const eclipticLon = eclipticLongitude(meanAnomaly)
  const dec = declination(eclipticLon)

  const noonJulian = solarTransitJulian(approxNoon, meanAnomaly, eclipticLon)
  const solarNoon = fromJulian(noonJulian)

  // 地平線に達するときの時角。|cos| が 1 を超えると太陽は地平線を跨がない
  const cosHourAngle =
    (Math.sin(HORIZON) - Math.sin(phi) * Math.sin(dec)) / (Math.cos(phi) * Math.cos(dec))

  if (cosHourAngle >= 1) {
    // 一日中、太陽が地平線より下（極夜）
    return { sunrise: null, sunset: null, solarNoon, alwaysUp: false, alwaysDown: true }
  }
  if (cosHourAngle <= -1) {
    // 一日中、太陽が地平線より上（白夜）
    return { sunrise: null, sunset: null, solarNoon, alwaysUp: true, alwaysDown: false }
  }

  const hourAngle = Math.acos(cosHourAngle)
  const setJulian = solarTransitJulian(
    approxTransit(hourAngle, lonWest, cycle),
    meanAnomaly,
    eclipticLon,
  )
  // 日の出は南中を挟んで日の入りと対称
  const riseJulian = noonJulian - (setJulian - noonJulian)

  return {
    sunrise: fromJulian(riseJulian),
    sunset: fromJulian(setJulian),
    solarNoon,
    alwaysUp: false,
    alwaysDown: false,
  }
}

/** その瞬間、太陽が出ているか */
export function isDaylight(date: Date, latitude: number, longitude: number): boolean {
  const t = sunTimes(date, latitude, longitude)
  if (t.alwaysUp) return true
  if (t.alwaysDown) return false
  if (!t.sunrise || !t.sunset) return false
  const now = date.getTime()
  return now >= t.sunrise.getTime() && now < t.sunset.getTime()
}

/**
 * 次に昼夜が入れ替わる瞬間。
 *
 * 配色を切り替えるタイマーを仕掛けるために使う。ポーリングせずに済む。
 * 白夜・極夜が続く間は切り替わりが無いので、見つからなければ null を返す
 * （呼び出し側で適当な間隔の再確認に落とすこと）。
 *
 * @param searchDays 何日先まで探すか。白夜・極夜は半年続きうるので既定は 370 日
 */
export function nextSunEvent(
  date: Date,
  latitude: number,
  longitude: number,
  searchDays = 370,
): Date | null {
  const from = date.getTime()
  for (let i = 0; i <= searchDays; i += 1) {
    const probe = new Date(from + i * DAY_MS)
    const t = sunTimes(probe, latitude, longitude)
    // その日の 2 つの境目のうち、いま以降で最も早いもの
    let best = Number.POSITIVE_INFINITY
    for (const edge of [t.sunrise, t.sunset]) {
      if (!edge) continue
      const at = edge.getTime()
      if (at > from && at < best) best = at
    }
    if (best !== Number.POSITIVE_INFINITY) return new Date(best)
  }
  return null
}

/** その日の昼の長さ [ms]。白夜は 24 時間、極夜は 0 */
export function daylightMs(date: Date, latitude: number, longitude: number): number {
  const t = sunTimes(date, latitude, longitude)
  if (t.alwaysUp) return DAY_MS
  if (t.alwaysDown) return 0
  if (!t.sunrise || !t.sunset) return 0
  return t.sunset.getTime() - t.sunrise.getTime()
}
