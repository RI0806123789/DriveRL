/**
 * カメラワーク（memo F-03「俯瞰や追従などのカメラワーク」）。
 *
 * - **俯瞰**: drei の OrbitControls。マップ全体が入る初期位置に置く。
 * - **追従**: 対象車両の斜め後方上空から追う。カメラを減衰させて滑らかにする。
 * - **運転席**: 対象車両の運転席から見た一人称視点。
 *   日本車は右ハンドルなので、視点は車体中心より**右側**に置く。
 *
 * 姿勢は補間済みの値（scene/interpolation.ts）を使うので、サーバーが 20Hz でも
 * 60fps で滑らかに動く。respawn などで大きく飛んだときだけ補間を切って瞬間移動させる。
 *
 * カメラモードと追従対象は**ここで購読する**。SimulatorView 側で購読すると、
 * 車両一覧をクリックするたびにシーン全体（車両・経路・建物）の差分計算が走る。
 */

import { memo, useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { frameBuffer } from '../store/frameBuffer'
import { useSimStore } from '../store/simStore'
import type { MapBounds } from '../types/protocol'
import { computeAlpha, createPose, sampleVehicle } from './interpolation'
import { driverEye, driverLookAt, firstActiveSlot, followEye, followLookAt } from './cameraMath'
import { setCameraNotice } from './sceneStats'

/** 追従・運転席それぞれの追従の速さ（1 秒あたりの寄り具合） */
const FOLLOW_LERP = 4.5
const DRIVER_LERP = 22.0

/** 視野角 [度]。運転席は広めに取ると自然に見える */
const FOV_DEFAULT = 50
const FOV_DRIVER = 68
/** ニアクリップ [m]。運転席は自車のすぐ前まで見せたいので小さくする */
const NEAR_DEFAULT = 0.5
const NEAR_DRIVER = 0.15

/**
 * 追従対象が取れなくなってから乗り換えるまでの猶予 [ms]。
 * respawn の谷間や 1 フレームの取りこぼしで切り替わらないよう、少し待つ。
 */
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

  const pose = useMemo(createPose, [])
  const desiredPos = useMemo(() => new THREE.Vector3(), [])
  const desiredLook = useMemo(() => new THREE.Vector3(), [])
  const currentLook = useMemo(() => new THREE.Vector3(), [])
  const initialised = useRef(false)
  /** 追従対象を見失った時刻。0 なら見えている */
  const lostSince = useRef(0)

  const overview = useMemo(() => overviewFor(bounds), [bounds])

  // 視野角とニアクリップはモードで切り替える
  useEffect(() => {
    const cam = camera as THREE.PerspectiveCamera
    if (!cam.isPerspectiveCamera) return
    cam.fov = mode === 'driver' ? FOV_DRIVER : FOV_DEFAULT
    cam.near = mode === 'driver' ? NEAR_DRIVER : NEAR_DEFAULT
    cam.updateProjectionMatrix()
  }, [camera, mode])

  // マップが変わったら俯瞰位置を取り直す
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

  // モードや対象を変えたら、次のフレームで一度だけ瞬間移動させる
  useEffect(() => {
    initialised.current = false
    lostSince.current = 0
  }, [mode, followTarget])

  useFrame((_state, delta) => {
    if (mode === 'orbit') return

    const paused = useSimStore.getState().status.renderPaused
    const alpha = computeAlpha(performance.now(), paused)
    const ok = sampleVehicle(followTarget, alpha, pose)
    if (!ok) {
      // 追従対象が非アクティブになった（車両数を減らした・エピソードが終わった）。
      // 何もしないとカメラが最後の位置で固まり、OrbitControls も無効なので
      // マウス操作も効かなくなる。別の車へ乗り換えるか、俯瞰へ戻す。
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
      // モード切替直後と respawn 直後は即座に合わせる（画面が滑って酔うのを防ぐ）
      camera.position.copy(desiredPos)
      currentLook.copy(desiredLook)
      initialised.current = true
    } else {
      // フレームレートに依存しない減衰
      const rate = mode === 'driver' ? DRIVER_LERP : FOLLOW_LERP
      const t = 1 - Math.exp(-rate * delta)
      camera.position.lerp(desiredPos, t)
      currentLook.lerp(desiredLook, t)
    }
    camera.lookAt(currentLook)
  })

  return (
    <OrbitControls
      ref={controls}
      // 追従・運転席では手動操作を無効化する（カメラを毎フレーム上書きするため）
      enabled={mode === 'orbit'}
      enableDamping
      dampingFactor={0.08}
      maxPolarAngle={Math.PI * 0.49}
      minDistance={12}
      maxDistance={2200}
      makeDefault
    />
  )
})

/**
 * 追従対象を見失ったときの復帰。
 *
 * 猶予を置いてから、最小のアクティブスロットへ乗り換える。
 * 1 台も走っていなければ俯瞰へ落として理由を HUD に出す
 * （黙って固まると「操作が効かない」としか見えないため）。
 */
function recoverLostTarget(followTarget: number, lostSince: { current: number }): void {
  const curr = frameBuffer.curr
  if (!curr) return // まだフレームが 1 枚も来ていない。切り替える根拠が無い

  const now = performance.now()
  if (lostSince.current === 0) {
    lostSince.current = now
    return
  }
  if (now - lostSince.current < LOST_GRACE_MS) return
  lostSince.current = now // 復帰できなかったときのために、次の試行まで再び待つ

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
