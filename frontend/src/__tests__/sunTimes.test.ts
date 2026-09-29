/** 日の出・日の入り（scene/sunTimes.ts）の境界値の単体テスト。 */

import assert from 'node:assert/strict'
import { describe, test } from 'node:test'

import { daylightMs, isDaylight, nextSunEvent, sunTimes } from '../scene/sunTimes.ts'

const HOUR_MS = 3_600_000
const DAY_MS = 24 * HOUR_MS
const GINZA = { lat: 35.6717, lon: 139.765 }

function utc(y: number, mo: number, d: number, h = 0, mi = 0): Date {
  return new Date(Date.UTC(y, mo - 1, d, h, mi))
}

describe('昼の長さ', () => {
  test('赤道の春分はほぼ 12 時間（大気差と太陽の半径のぶんだけ長い）', () => {
    const ms = daylightMs(utc(2026, 3, 20, 12), 0, 0)
    assert.ok(ms > 12 * HOUR_MS && ms < 12 * HOUR_MS + 15 * 60_000, `${ms / HOUR_MS} 時間`)
  })

  test('北緯 80 度の夏至は白夜・冬至は極夜', () => {
    const summer = sunTimes(utc(2026, 6, 21, 12), 80, 15)
    assert.equal(summer.alwaysUp, true)
    assert.equal(summer.sunrise, null)
    assert.equal(daylightMs(utc(2026, 6, 21, 12), 80, 15), DAY_MS)
    assert.equal(isDaylight(utc(2026, 6, 21, 0), 80, 15), true)

    const winter = sunTimes(utc(2026, 12, 21, 12), 80, 15)
    assert.equal(winter.alwaysDown, true)
    assert.equal(daylightMs(utc(2026, 12, 21, 12), 80, 15), 0)
    assert.equal(isDaylight(utc(2026, 12, 21, 12), 80, 15), false)
  })

  test('南半球では白夜と極夜の季節が逆になる', () => {
    assert.equal(sunTimes(utc(2026, 6, 21, 12), -80, 15).alwaysDown, true)
    assert.equal(sunTimes(utc(2026, 12, 21, 12), -80, 15).alwaysUp, true)
  })

  test('緯度・経度・季節のどこでも 0〜24 時間に収まり NaN にならない', () => {
    for (const lat of [-89.9, -66, -30, 0, 35.6717, 66, 89.9]) {
      for (const lon of [-179.9, -60, 0, 139.765, 179.9]) {
        for (const month of [1, 4, 7, 10]) {
          const ms = daylightMs(utc(2026, month, 15, 3), lat, lon)
          assert.ok(Number.isFinite(ms) && ms >= 0 && ms <= DAY_MS, `${lat},${lon} ${month}月: ${ms}`)
        }
      }
    }
  })
})

describe('日の出・南中・日の入りの並び', () => {
  test('銀座では毎月 日の出 < 南中 < 日の入り', () => {
    for (let month = 1; month <= 12; month += 1) {
      const t = sunTimes(utc(2026, month, 10, 3), GINZA.lat, GINZA.lon)
      assert.ok(t.sunrise && t.sunset, `${month}月`)
      assert.ok(t.sunrise.getTime() < t.solarNoon.getTime(), `${month}月`)
      assert.ok(t.solarNoon.getTime() < t.sunset.getTime(), `${month}月`)
    }
  })

  test('経度が 15 度東なら日の出はほぼ 1 時間早い', () => {
    const date = utc(2026, 9, 29, 3)
    const here = sunTimes(date, GINZA.lat, GINZA.lon).sunrise
    const east = sunTimes(date, GINZA.lat, GINZA.lon + 15).sunrise
    assert.ok(here && east)
    const diff = here.getTime() - east.getTime()
    assert.ok(Math.abs(diff - HOUR_MS) < 5 * 60_000, `${diff / 60_000} 分`)
  })
})

describe('次に昼夜が入れ替わる瞬間', () => {
  test('必ず今より後で、その前後で昼夜が入れ替わる', () => {
    for (let day = 0; day < 365; day += 17) {
      for (const hour of [0, 7, 13, 21]) {
        const now = new Date(utc(2026, 1, 1, hour).getTime() + day * DAY_MS)
        const next = nextSunEvent(now, GINZA.lat, GINZA.lon)
        assert.ok(next, now.toISOString())
        assert.ok(next.getTime() > now.getTime(), now.toISOString())
        assert.ok(next.getTime() - now.getTime() <= DAY_MS, now.toISOString())
        const before = isDaylight(new Date(next.getTime() - 60_000), GINZA.lat, GINZA.lon)
        const after = isDaylight(new Date(next.getTime() + 60_000), GINZA.lat, GINZA.lon)
        assert.notEqual(before, after, `${now.toISOString()} → ${next.toISOString()}`)
      }
    }
  })

  test('白夜の間は探す日数が足りなければ null、足りれば白夜が明ける日を返す', () => {
    const now = utc(2026, 6, 21, 12)
    assert.equal(nextSunEvent(now, 80, 15, 1), null)
    const next = nextSunEvent(now, 80, 15)
    assert.ok(next && next.getTime() > now.getTime())
    assert.ok(next.getTime() - now.getTime() < 120 * DAY_MS, next?.toISOString())
  })
})
