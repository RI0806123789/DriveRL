/** 配色を日の出・日の入りに連動させる。 */

import { useEffect } from 'react'
import { applyTheme, useSimStore } from './simStore'
import { nextThemeChange, themeFor, timezoneLocation, type SunLocation } from './themeClock'

/** setTimeout が扱える上限（約 24.8 日）。超えると即座に発火してしまう */
const MAX_TIMEOUT_MS = 2_147_483_647

/** 次の切り替わりが無いとき（白夜・極夜）に様子を見る間隔 */
const FALLBACK_RECHECK_MS = 6 * 3600_000

/** いま基準にすべき座標。 */
function currentLocation(): SunLocation {
  const { presets, status, pendingPresetId } = useSimStore.getState()
  const id = status.presetId ?? pendingPresetId
  if (presets.length > 0) {
    const hit = (id ? presets.find((p) => p.id === id) : undefined) ?? presets[0]
    if (hit && Number.isFinite(hit.centerLat) && Number.isFinite(hit.centerLon)) {
      return { lat: hit.centerLat, lon: hit.centerLon, source: hit.name }
    }
  }
  return timezoneLocation()
}

/** 日の出・日の入りに合わせて配色を切り替え続ける。 */
export function useAutoTheme(): void {
  const presetCount = useSimStore((s) => s.presets.length)
  const presetId = useSimStore((s) => s.status.presetId)

  useEffect(() => {
    let timer: number | undefined
    let cancelled = false

    const tick = (): void => {
      if (cancelled) return
      const where = currentLocation()
      const now = new Date()

      const theme = themeFor(now, where)
      applyTheme(theme)
      if (useSimStore.getState().theme !== theme) {
        useSimStore.setState({ theme })
      }

      const next = nextThemeChange(now, where)
      const wait = next
        ? Math.min(MAX_TIMEOUT_MS, Math.max(1000, next.getTime() - now.getTime() + 1000))
        : FALLBACK_RECHECK_MS
      timer = window.setTimeout(tick, wait)
    }

    tick()

    const onVisible = (): void => {
      if (document.visibilityState !== 'visible') return
      if (timer !== undefined) window.clearTimeout(timer)
      tick()
    }
    document.addEventListener('visibilitychange', onVisible)

    return () => {
      cancelled = true
      if (timer !== undefined) window.clearTimeout(timer)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [presetCount, presetId])
}
