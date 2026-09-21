/** カメラワーク（memo F-03「俯瞰や追従などのカメラワーク」）。 */

import { memo, useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { frameBuffer } from '../store/frameBuffer'
import { pedestrian } from '../store/pedestrian'
import { useSimStore } from '../store/simStore'
import type { MapBounds } from '../types/protocol'
import { computeAlpha, createPose, sampleVehicle } from './interpolation'
import {
  DRIVER_FOLLOW_RATE,
  DRIVER_FOV_DEG,
  FOLLOW_FOLLOW_RATE,
  driverEye,
  driverLookAt,
  firstActiveSlot,
  followEye,
  followLerpFactor,
  followLookAt,
} from './cameraMath'
import { eyeLookAt, eyePosition } from './pedestrianGeometry'
import { setCameraNotice } from './sceneStats'

/** 追従・運転席それぞれの追従の速さ（1 秒あたりの寄り具合） */

/** 視野角 [度]。運転席は `cameraMath.DRIVER_FOV_DEG` が唯一の出典（検出枠と共有） */
const FOV_DEFAULT = 50
/** ニアクリップ [m]。運転席は自車のすぐ前まで見せたいので小さくする */
const NEAR_DEFAULT = 0.5
const NEAR_DRIVER = 0.15

/** 追従対象が取れなくなってから乗り換えるまでの猶予 [ms]。 */
const LOST_GRACE_MS = 600

export interface CameraRigProps {
  bounds: MapBounds | null
}

/** マップ全体が入る俯瞰カメラの初期位置と注視点（Three 座標） */
function overviewFor(bounds: MapBounds | null) {
  if (!bounds) {
    return { position: new THREE.Vector3(300, 320, 300), target: new THREE.Vector3(0, 0, 0) }
  }
  const cx = (bounds.minX + bounds.maxX) / 2
  const cz = -((bounds.minY + bounds.maxY) / 2)
  const span = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY)
  const d = Math.max(240, span * 0.85)
  return {
    position: new THREE.Vector3(cx + d * 0.55, d * 0.72, cz + d * 0.55),
    target: new THREE.Vector3(cx, 0, cz),
  }
}

export const CameraRig = memo(function CameraRig({ bounds }: CameraRigProps) {
  const camera = useThree((s) => s.camera)
  const controls = useRef<OrbitControlsImpl>(null)

  const mode = useSimStore((s) => s.cameraMode)
  const followTarget = useSimStore((s) => s.followTarget)
  const appMode = useSimStore((s) => s.mode)
  /** 実用モードで歩いている間は一人称に固定する（決定 11） */
  const taxiWalk = appMode === 'taxi' && mode !== 'driver'

  const pose = useMemo(createPose, [])
  const desiredPos = useMemo(() => new THREE.Vector3(), [])
  const desiredLook = useMemo(() => new THREE.Vector3(), [])
  const currentLook = useMemo(() => new THREE.Vector3(), [])
  const initialised = useRef(false)
  /** 追従対象を見失った時刻。0 なら見えている */
  const lostSince = useRef(0)

  const overview = useMemo(() => overviewFor(bounds), [bounds])

  useEffect(() => {
    const cam = camera as THREE.PerspectiveCamera
    if (!cam.isPerspectiveCamera) return
    const firstPerson = mode === 'driver' || taxiWalk
    cam.fov = firstPerson ? DRIVER_FOV_DEG : FOV_DEFAULT
    cam.near = firstPerson ? NEAR_DRIVER : NEAR_DEFAULT
    cam.updateProjectionMatrix()
  }, [camera, mode, taxiWalk])

  // 実用モードでも、立つ場所が決まるまでは俯瞰へ置いておく
  // （置かないとカメラが初期位置のまま街を向かず、真っ白な画面になる）
  useEffect(() => {
    if (mode !== 'orbit') return
    camera.position.copy(overview.position)
    const c = controls.current
    if (c) {
      c.target.copy(overview.target)
      c.update()
    } else {
      camera.lookAt(overview.target)
    }
  }, [overview, camera, mode])

  useEffect(() => {
    initialised.current = false
    lostSince.current = 0
  }, [mode, followTarget])

  useFrame((_state, delta) => {
    // 歩いている間は補間せず、そのまま目線へ置く（動かしているのがフロント自身のため）。
    // **立つ場所が決まるまでは俯瞰のまま**にする（原点は建物の中のことがある）
    if (taxiWalk && pedestrian.placed) {
      const eye = eyePosition(pedestrian.x, pedestrian.y, pedestrian.heading, pedestrian.bob)
      const look = eyeLookAt(
        pedestrian.x,
        pedestrian.y,
        pedestrian.heading,
        pedestrian.pitch,
        12,
        pedestrian.bob,
      )
      camera.position.set(eye.x, eye.y, eye.z)
      camera.lookAt(look.x, look.y, look.z)
      return
    }

    if (mode === 'orbit') return

    const paused = useSimStore.getState().status.renderPaused
    const alpha = computeAlpha(performance.now(), paused)
    const ok = sampleVehicle(followTarget, alpha, pose)
    if (!ok) {
      recoverLostTarget(followTarget, lostSince)
      return
    }
    lostSince.current = 0

    const eye =
      mode === 'driver'
        ? driverEye(pose.x, pose.y, pose.heading)
        : followEye(pose.x, pose.y, pose.heading)
    const look =
      mode === 'driver'
        ? driverLookAt(pose.x, pose.y, pose.heading)
        : followLookAt(pose.x, pose.y, pose.heading)
    desiredPos.set(eye.x, eye.y, eye.z)
    desiredLook.set(look.x, look.y, look.z)

    if (!initialised.current || pose.teleported) {
      camera.position.copy(desiredPos)
      currentLook.copy(desiredLook)
      initialised.current = true
    } else {
      // ★ 運転席は rate 0（車体に固定）。遅れて追わせると加減速で車内が前後に滑る
      const rate = mode === 'driver' ? DRIVER_FOLLOW_RATE : FOLLOW_FOLLOW_RATE
      const t = followLerpFactor(rate, delta)
      camera.position.lerp(desiredPos, t)
      currentLook.lerp(desiredLook, t)
    }
    camera.lookAt(currentLook)
  })

  return (
    <OrbitControls
      ref={controls}
      enabled={mode === 'orbit' && !taxiWalk}
      enableDamping
      dampingFactor={0.08}
      maxPolarAngle={Math.PI * 0.49}
      minDistance={12}
      maxDistance={2200}
      makeDefault
    />
  )
})

/** 追従対象を見失ったときの復帰。 */
function recoverLostTarget(followTarget: number, lostSince: { current: number }): void {
  const curr = frameBuffer.curr
  if (!curr) return

  const now = performance.now()
  if (lostSince.current === 0) {
    lostSince.current = now
    return
  }
  if (now - lostSince.current < LOST_GRACE_MS) return
  lostSince.current = now

  const next = firstActiveSlot(curr.vehicles)
  const store = useSimStore.getState()
  if (next >= 0 && next !== followTarget) {
    setCameraNotice(`車両 #${followTarget} が走行を終えたので #${next} に切り替えました`)
    store.setFollowTarget(next)
  } else if (next < 0) {
    setCameraNotice('追従できる車両がいないため俯瞰に戻しました')
    store.setCameraMode('orbit')
  }
}
