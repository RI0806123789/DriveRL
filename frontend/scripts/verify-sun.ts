/**
 * 日の出・日の入りの計算を検証する（ブラウザ不要）。
 *
 *     cd frontend
 *     node scripts/verify-sun.ts
 *
 * 配色がこの計算に連動するので、ここが狂うと「昼なのに夜の画面」になる。
 * しかも**間違っていても型チェックもビルドも通る**ため、数値で確かめる。
 *
 * 外部の暦データは持ち込まない（ローカル完結が前提のため）。
 * 代わりに、暦の知識から確実に言える不変条件と、広く知られた値を使う。
 */

import { daylightMs, isDaylight, nextSunEvent, sunTimes } from '../src/scene/sunTimes.ts'

let failures = 0

function check(label: string, ok: boolean, detail = ''): void {
  console.log(`  [${ok ? 'OK  ' : 'NG  '}] ${label}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures += 1
}

/** JST（UTC+9）での時刻表示。Node のローカル時刻に左右されないようにする */
function jst(d: Date | null): string {
  if (!d) return '--:--'
  const t = new Date(d.getTime() + 9 * 3600_000)
  return `${String(t.getUTCHours()).padStart(2, '0')}:${String(t.getUTCMinutes()).padStart(2, '0')}`
}

function hours(ms: number): string {
  const h = Math.floor(ms / 3600_000)
  const m = Math.round((ms - h * 3600_000) / 60_000)
  return `${h}時間${String(m).padStart(2, '0')}分`
}

/** UTC で日時を作る（実行環境のタイムゾーンに影響されないため） */
function utc(y: number, mo: number, d: number, h = 0, mi = 0): Date {
  return new Date(Date.UTC(y, mo - 1, d, h, mi))
}

// プリセットの座標（backend/app/map/presets.py と同じ値）
const GINZA = { lat: 35.6717, lon: 139.765, name: '銀座' }
const UMEDA = { lat: 34.7025, lon: 135.4959, name: '梅田' }
const SAKAE = { lat: 35.1681, lon: 136.9083, name: '栄' }

console.log('='.repeat(70))
console.log('1. 順序の不変条件（日の出 < 南中 < 日の入り）')
console.log('='.repeat(70))

let orderOk = true
let noonOk = true
for (let month = 1; month <= 12; month += 1) {
  for (const place of [GINZA, UMEDA, SAKAE]) {
    const t = sunTimes(utc(2026, month, 15), place.lat, place.lon)
    if (!t.sunrise || !t.sunset) {
      orderOk = false
      continue
    }
    if (!(t.sunrise.getTime() < t.solarNoon.getTime())) orderOk = false
    if (!(t.solarNoon.getTime() < t.sunset.getTime())) orderOk = false
    // 南中は日の出と日の入りのちょうど中間（対称性）
    const mid = (t.sunrise.getTime() + t.sunset.getTime()) / 2
    if (Math.abs(mid - t.solarNoon.getTime()) > 1000) noonOk = false
  }
}
check('全 12 か月・3 プリセットで 日の出 < 南中 < 日の入り', orderOk)
check('南中が日の出と日の入りの中点（誤差 1 秒以内）', noonOk)

console.log()
console.log('='.repeat(70))
console.log('2. 銀座の一年（日の出 / 日の入り / 昼の長さ）')
console.log('='.repeat(70))
for (const [label, date] of [
  ['春分   3/20', utc(2026, 3, 20)],
  ['夏至   6/21', utc(2026, 6, 21)],
  ['秋分   9/23', utc(2026, 9, 23)],
  ['冬至  12/22', utc(2026, 12, 22)],
] as const) {
  const t = sunTimes(date, GINZA.lat, GINZA.lon)
  console.log(
    `    ${label}  日の出 ${jst(t.sunrise)}  南中 ${jst(t.solarNoon)}  ` +
      `日の入り ${jst(t.sunset)}  昼 ${hours(daylightMs(date, GINZA.lat, GINZA.lon))}`,
  )
}

// 東京の夏至の昼は約 14 時間 35 分、冬至は約 9 時間 45 分（理科年表などで広く知られた値）
const summer = daylightMs(utc(2026, 6, 21), GINZA.lat, GINZA.lon)
const winter = daylightMs(utc(2026, 12, 22), GINZA.lat, GINZA.lon)
check('夏至の昼が 14 時間 35 分 ±10 分', Math.abs(summer - (14 * 60 + 35) * 60_000) < 10 * 60_000,
  hours(summer))
check('冬至の昼が 9 時間 45 分 ±10 分', Math.abs(winter - (9 * 60 + 45) * 60_000) < 10 * 60_000,
  hours(winter))

// 春分・秋分の昼はちょうど 12 時間ではなく、太陽の視半径と大気差のぶん 7 分ほど長い
const equinox = daylightMs(utc(2026, 3, 20), GINZA.lat, GINZA.lon)
check('春分の昼が 12 時間より長く、12 時間 15 分未満',
  equinox > 12 * 3600_000 && equinox < (12 * 60 + 15) * 60_000, hours(equinox))

console.log()
console.log('='.repeat(70))
console.log('3. 経度の効き（東ほど日の出が早い）')
console.log('='.repeat(70))
const day = utc(2026, 9, 6)
const rises = [GINZA, SAKAE, UMEDA].map((p) => ({
  name: p.name,
  lon: p.lon,
  at: sunTimes(day, p.lat, p.lon).sunrise,
}))
for (const r of rises) console.log(`    ${r.name}（東経 ${r.lon.toFixed(2)}）  日の出 ${jst(r.at)}`)
check('銀座 → 栄 → 梅田 の順に日の出が遅い',
  rises[0].at !== null && rises[1].at !== null && rises[2].at !== null &&
  rises[0].at.getTime() < rises[1].at.getTime() &&
  rises[1].at.getTime() < rises[2].at.getTime())

// 経度 1 度で 4 分。銀座と梅田は 4.27 度離れているので約 17 分の差になるはず
const lonGapMin = (GINZA.lon - UMEDA.lon) * 4
const riseGapMin = (rises[2].at!.getTime() - rises[0].at!.getTime()) / 60_000
check('銀座と梅田の日の出差が経度差から出る値（±3 分）',
  Math.abs(riseGapMin - lonGapMin) < 3,
  `実測 ${riseGapMin.toFixed(1)} 分 / 経度から ${lonGapMin.toFixed(1)} 分`)

console.log()
console.log('='.repeat(70))
console.log('4. 昼夜の判定と、次に切り替わる時刻')
console.log('='.repeat(70))
{
  const t = sunTimes(day, GINZA.lat, GINZA.lon)
  const rise = t.sunrise!.getTime()
  const set = t.sunset!.getTime()
  check('日の出の 1 分前は夜', !isDaylight(new Date(rise - 60_000), GINZA.lat, GINZA.lon))
  check('日の出の 1 分後は昼', isDaylight(new Date(rise + 60_000), GINZA.lat, GINZA.lon))
  check('日の入りの 1 分前は昼', isDaylight(new Date(set - 60_000), GINZA.lat, GINZA.lon))
  check('日の入りの 1 分後は夜', !isDaylight(new Date(set + 60_000), GINZA.lat, GINZA.lon))
  check('正午は昼', isDaylight(t.solarNoon, GINZA.lat, GINZA.lon))
  check('真夜中は夜', !isDaylight(utc(2026, 9, 5, 15, 0), GINZA.lat, GINZA.lon)) // JST 9/6 00:00

  // 次の切り替わりは必ず未来で、かつそこで昼夜が反転する
  let eventOk = true
  let flipOk = true
  for (let h = 0; h < 24; h += 1) {
    const at = utc(2026, 9, 5, 15 + h, 0) // JST 9/6 の各正時
    const next = nextSunEvent(at, GINZA.lat, GINZA.lon)
    if (!next || next.getTime() <= at.getTime()) {
      eventOk = false
      continue
    }
    if (next.getTime() - at.getTime() > 24 * 3600_000) eventOk = false
    const before = isDaylight(new Date(next.getTime() - 60_000), GINZA.lat, GINZA.lon)
    const after = isDaylight(new Date(next.getTime() + 60_000), GINZA.lat, GINZA.lon)
    if (before === after) flipOk = false
  }
  check('24 通りの時刻すべてで、次の切り替わりが 24 時間以内の未来', eventOk)
  check('その前後で昼夜が必ず反転する', flipOk)
}

console.log()
console.log('='.repeat(70))
console.log('5. 高緯度（白夜・極夜）で壊れないこと')
console.log('='.repeat(70))
{
  // トロムソ（ノルウェー、北緯 69.65）は 5〜7 月が白夜、11〜1 月が極夜
  const TROMSO = { lat: 69.6496, lon: 18.9553 }
  const midsummer = sunTimes(utc(2026, 6, 21), TROMSO.lat, TROMSO.lon)
  const midwinter = sunTimes(utc(2026, 12, 22), TROMSO.lat, TROMSO.lon)
  check('トロムソの夏至は白夜（沈まない）', midsummer.alwaysUp && !midsummer.alwaysDown)
  check('トロムソの冬至は極夜（昇らない）', midwinter.alwaysDown && !midwinter.alwaysUp)
  check('白夜は昼と判定される', isDaylight(utc(2026, 6, 21, 1, 0), TROMSO.lat, TROMSO.lon))
  check('極夜は夜と判定される', !isDaylight(utc(2026, 12, 22, 12, 0), TROMSO.lat, TROMSO.lon))

  // 極夜のさなかでも、いずれ日は昇る（無限ループや null 固定にならないこと）
  const escape = nextSunEvent(utc(2026, 12, 22), TROMSO.lat, TROMSO.lon)
  check('極夜からでも次の日の出が見つかる', escape !== null,
    escape ? escape.toISOString().slice(0, 10) : 'null')

  // 赤道は年間を通して昼が約 12 時間
  let equatorOk = true
  for (let month = 1; month <= 12; month += 1) {
    const len = daylightMs(utc(2026, month, 15), 0, 0)
    if (Math.abs(len - 12 * 3600_000) > 12 * 60_000) equatorOk = false
  }
  check('赤道の昼は年間を通して 12 時間 ±12 分', equatorOk)
}

console.log()
console.log('='.repeat(70))
console.log('6. 素性の悪い入力で落ちないこと')
console.log('='.repeat(70))
{
  let robust = true
  const cases: Array<[string, number, number]> = [
    ['北極点', 90, 0],
    ['南極点', -90, 0],
    ['日付変更線', 0, 180],
    ['日付変更線（西）', 0, -180],
  ]
  for (const [, lat, lon] of cases) {
    try {
      const t = sunTimes(utc(2026, 6, 21), lat, lon)
      if (Number.isNaN(t.solarNoon.getTime())) robust = false
      if (t.sunrise && Number.isNaN(t.sunrise.getTime())) robust = false
    } catch {
      robust = false
    }
  }
  check('極点・日付変更線でも NaN や例外にならない', robust,
    cases.map(([n]) => n).join(' / '))
}

console.log()
console.log('='.repeat(70))
if (failures > 0) {
  console.log(`結果: ${failures} 件の不合格`)
  process.exit(1)
}
console.log('結果: すべて合格')
