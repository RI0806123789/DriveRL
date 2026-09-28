/** 開発モードの運転席視点を、前方・後方・左側方・右側方の 4 分割で描く。**Canvas の中に置くこと。** */

import { useEffect, useMemo } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'

import { useSimStore } from '../store/simStore'
import { SURROUND_CAMERAS, SURROUND_FOV_DEG, surroundEye, surroundLookAt } from './cameraMath'
import { computeAlpha, createPose, sampleVehicle } from './interpolation'
import { hiddenFromMirrors } from './mirrorHidden'
import { QUAD_PANES, frontPanePointer, quadViewport, surroundCameraFor } from './quadLayout'

/** 自動描画より後に走らせる（正の優先度なので R3F は自分では描かなくなる） */
const QUAD_PRIORITY = 1
const NEAR = 0.15

export function QuadViewRenderer() {
  const cameras = useMemo(
    () => SURROUND_CAMERAS.map(() => new THREE.PerspectiveCamera(SURROUND_FOV_DEG, 1, NEAR, 8000)),
    [],
  )
  const pose = useMemo(createPose, [])
  const hidden = useMemo<THREE.Object3D[]>(() => [], [])
  const setEvents = useThree((s) => s.setEvents)

  // クリック（障害物を置く・消す）は前方のペインの中だけを前方カメラの画として当てる。
  //   既定の計算は画面全体を前方カメラの画とみなすので、4 分割のままでは狙った所に置けない
  useEffect(() => {
    setEvents({
      compute: (event, state) => {
        const pane = frontPanePointer(event.offsetX, event.offsetY, state.size.width, state.size.height)
        if (pane) {
          state.pointer.set(pane.x, pane.y)
          state.raycaster.far = Infinity
        } else {
          state.raycaster.far = 0
        }
        state.raycaster.setFromCamera(state.pointer, state.camera)
      },
    })
    return () => {
      setEvents({
        compute: (event, state) => {
          state.pointer.set(
            (event.offsetX / state.size.width) * 2 - 1,
            -(event.offsetY / state.size.height) * 2 + 1,
          )
          state.raycaster.far = Infinity
          state.raycaster.setFromCamera(state.pointer, state.camera)
        },
      })
    }
  }, [setEvents])

  useFrame((state) => {
    const { gl, scene, camera, size } = state
    const store = useSimStore.getState()
    const ok = sampleVehicle(
      store.followTarget,
      computeAlpha(state.clock.oldTime, store.status.renderPaused),
      pose,
    )
    if (!ok) {
      gl.setViewport(0, 0, size.width, size.height)
      gl.render(scene, camera)
      return
    }

    const far = camera instanceof THREE.PerspectiveCamera ? camera.far : 8000
    const shadowAuto = gl.shadowMap.autoUpdate
    const sceneAuto = scene.matrixWorldAutoUpdate
    gl.setScissorTest(true)
    QUAD_PANES.forEach((pane, index) => {
      const vp = quadViewport(index, size.width, size.height)
      gl.setViewport(vp.x, vp.y, vp.w, vp.h)
      gl.setScissor(vp.x, vp.y, vp.w, vp.h)
      const spec = surroundCameraFor(pane)
      if (!spec) {
        // 前方は画面のカメラ（CameraRig が運転席に置いている）をそのまま使う。縦横比は画面と同じ
        gl.render(scene, camera)
        // 影と行列は 1 枚目で作ったものを使い回す（ペインごとに作り直すと、その分だけ重くなる）
        gl.shadowMap.autoUpdate = false
        scene.matrixWorldAutoUpdate = false
        hidden.length = 0
        for (const o of hiddenFromMirrors) {
          if (!o.visible) continue
          o.visible = false
          hidden.push(o)
        }
        return
      }
      const cam = cameras[SURROUND_CAMERAS.indexOf(spec)]
      const eye = surroundEye(spec, pose.x, pose.y, pose.heading)
      const look = surroundLookAt(spec, pose.x, pose.y, pose.heading)
      cam.aspect = vp.h > 0 ? vp.w / vp.h : 1
      if (cam.far !== far) cam.far = far
      cam.updateProjectionMatrix()
      cam.position.set(eye.x, eye.y, eye.z)
      cam.lookAt(look.x, look.y, look.z)
      cam.updateMatrixWorld()
      gl.render(scene, cam)
    })
    for (const o of hidden) o.visible = true
    hidden.length = 0
    gl.setScissorTest(false)
    gl.setViewport(0, 0, size.width, size.height)
    scene.matrixWorldAutoUpdate = sceneAuto
    gl.shadowMap.autoUpdate = shadowAuto
  }, QUAD_PRIORITY)

  return null
}
