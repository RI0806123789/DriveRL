/** 運転席から見ている 1 台のルームミラーとドアミラーに、後ろの景色を映す。**Canvas の中に置くこと。** */

import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

import { useSimStore } from '../store/simStore'
import { computeAlpha, createPose, sampleVehicle } from './interpolation'
import { hiddenFromMirrors } from './mirrorHidden'
import { MIRROR_FACES, makeMirrorQuadGeometry, mirrorView, mirrorsDue, type MirrorFace } from './mirrorView'
import { composeBodyMatrix, composeVehicleMatrix, createTransformScratch } from './vehicleGeometry'
import { composeTiltMatrix, tiltOf } from './vehicleMotion'

/** 映像の板を鏡面から運転席側へ浮かせる量 [m] */
const QUAD_LIFT = 0.001
/** 鏡の反射率（映像をわずかに暗くする） */
const REFLECTANCE = 0.9
/** アンチエイリアスの標本数（メインの `antialias: true` と見え方を揃える） */
const SAMPLES = 4

interface Rig {
  readonly face: MirrorFace
  readonly target: THREE.WebGLRenderTarget
  readonly camera: THREE.PerspectiveCamera
  /** カメラの姿勢（車両ローカル） */
  readonly local: THREE.Matrix4
  readonly mesh: THREE.Mesh<THREE.BufferGeometry, THREE.MeshBasicMaterial>
  readonly bounds: THREE.Sphere
  age: number
  ready: boolean
}

function makeRig(face: MirrorFace): Rig {
  const target = new THREE.WebGLRenderTarget(face.pixels[0], face.pixels[1], { samples: SAMPLES })
  // sRGB で持つと 8bit でも暗部が潰れない（書き込みと読み出しで GPU が変換する）
  target.texture.colorSpace = THREE.SRGBColorSpace
  const view = mirrorView(face)
  const camera = new THREE.PerspectiveCamera()
  camera.near = view.near
  camera.far = view.far
  camera.projectionMatrix.makePerspective(view.left, view.right, view.top, view.bottom, view.near, view.far)
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert()
  const geometry = makeMirrorQuadGeometry(face, QUAD_LIFT)
  const material = new THREE.MeshBasicMaterial({ map: target.texture, fog: false })
  material.color.setScalar(REFLECTANCE)
  const mesh = new THREE.Mesh(geometry, material)
  mesh.visible = false
  mesh.frustumCulled = false
  const bounds = geometry.boundingSphere!.clone()
  return { face, target, camera, local: view.matrix, mesh, bounds, age: Infinity, ready: false }
}

function disposeRig(r: Rig): void {
  r.target.dispose()
  r.mesh.geometry.dispose()
  r.mesh.material.dispose()
}

export function RearViewMirrors() {
  const group = useRef<THREE.Group>(null)
  const rigs = useMemo(() => MIRROR_FACES.map(makeRig), [])
  useEffect(() => () => rigs.forEach(disposeRig), [rigs])

  const scratch = useMemo(
    () => ({
      transform: createTransformScratch(),
      base: new THREE.Matrix4(),
      tilt: new THREE.Matrix4(),
      world: new THREE.Matrix4(),
      projView: new THREE.Matrix4(),
      frustum: new THREE.Frustum(),
      sphere: new THREE.Sphere(),
      ages: rigs.map(() => Infinity),
      visible: rigs.map(() => false),
      hidden: [] as THREE.Object3D[],
      pose: createPose(),
      lastTarget: -1,
    }),
    [rigs],
  )

  useFrame((state, delta) => {
    const g = group.current
    if (!g) return
    const store = useSimStore.getState()
    const paused = store.status.renderPaused
    const target = store.cameraMode === 'driver' ? store.followTarget : -1
    const ok = target >= 0 && sampleVehicle(target, computeAlpha(state.clock.oldTime, paused), scratch.pose)
    g.visible = ok
    if (!ok) {
      scratch.lastTarget = -1
      return
    }
    // 乗り換えたら、前の車の映像を出さない
    if (target !== scratch.lastTarget) {
      scratch.lastTarget = target
      for (const r of rigs) {
        r.ready = false
        r.age = Infinity
        r.mesh.visible = false
      }
    }

    const pose = scratch.pose
    composeVehicleMatrix(scratch.transform, pose.x, pose.y, pose.heading, scratch.base)
    const tilt = tiltOf(target)
    composeTiltMatrix(tilt.pitch, tilt.roll, scratch.tilt)
    composeBodyMatrix(scratch.base, scratch.tilt, g.matrix)
    g.matrixWorldNeedsUpdate = true

    // 画面に映っている鏡だけ描き直す（運転席から遠い助手席側は、画面の外にあることが多い）
    const cam = state.camera
    scratch.projView.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse)
    scratch.frustum.setFromProjectionMatrix(scratch.projView)
    for (let i = 0; i < rigs.length; i++) {
      const r = rigs[i]
      if (!paused) r.age += delta
      scratch.sphere.copy(r.bounds).applyMatrix4(g.matrix)
      scratch.ages[i] = r.age
      scratch.visible[i] = scratch.frustum.intersectsSphere(scratch.sphere)
    }
    const due = mirrorsDue(scratch.ages, scratch.visible)
    if (due.length === 0) return

    const gl = state.gl
    const scene = state.scene
    const previous = gl.getRenderTarget()
    const shadowAuto = gl.shadowMap.autoUpdate
    const sceneAuto = scene.matrixWorldAutoUpdate
    // 影と行列は画面の描画で作ったものを使い回す（鏡ごとに作り直すと、その分だけ重くなる）
    gl.shadowMap.autoUpdate = false
    scene.matrixWorldAutoUpdate = false
    const hidden = scratch.hidden
    hidden.length = 0
    g.visible = false
    for (const o of hiddenFromMirrors) {
      if (!o.visible) continue
      o.visible = false
      hidden.push(o)
    }
    for (const i of due) {
      const r = rigs[i]
      scratch.world.multiplyMatrices(g.matrix, r.local)
      scratch.world.decompose(r.camera.position, r.camera.quaternion, r.camera.scale)
      r.camera.updateMatrixWorld()
      gl.setRenderTarget(r.target)
      gl.render(scene, r.camera)
      r.age = 0
      r.ready = true
      r.mesh.visible = true
    }
    gl.setRenderTarget(previous)
    for (const o of hidden) o.visible = true
    g.visible = true
    scene.matrixWorldAutoUpdate = sceneAuto
    gl.shadowMap.autoUpdate = shadowAuto
  })

  return (
    <group ref={group} matrixAutoUpdate={false} visible={false}>
      {rigs.map((r) => (
        <primitive key={r.face.key} object={r.mesh} />
      ))}
    </group>
  )
}
