/** 視野コーン・死角を出している間だけ、追従中の車の見通しと死角を frame に載せてもらう（`watch_occlusion`）。 */

import { useEffect } from 'react'
import { send } from '../store/connection'
import { useSimStore } from '../store/simStore'

/** 頼みを送り直す間隔 [ms]（サーバーは 2.5 秒で切る） */
const WATCH_INTERVAL_MS = 1000

export function useWatchOcclusion(): void {
  const active = useSimStore(
    (s) =>
      s.mode === 'dev' &&
      s.cameraMode !== 'orbit' &&
      (s.view.cameraFrustums || s.view.occlusionShadows),
  )
  const vehicleId = useSimStore((s) => s.followTarget)
  useEffect(() => {
    if (!active || vehicleId < 0) return
    const ping = () => send({ type: 'watch_occlusion', vehicleId })
    ping()
    const timer = window.setInterval(ping, WATCH_INTERVAL_MS)
    return () => window.clearInterval(timer)
  }, [active, vehicleId])
}
