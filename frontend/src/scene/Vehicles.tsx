/** 車両。 */

import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

import { useSimStore } from '../store/simStore'
import { computeAlpha, createPose, sampleVehicle } from './interpolation'
import {
} from './palette'
import { usePalette } from './usePalette'
import { vehicleColor } from './vehicleColors'
import {
  WHEEL_OFFSETS,
  WHEEL_RADIUS,
  composeVehicleMatrix,
  composeWheelMatrix,
  createTransformScratch,
  makeBodyGeometry,
  makeCabinGeometry,
  makeNoseGeometry,
  makeWheelGeometry,
} from './vehicleGeometry'

const ringGeom = new THREE.TorusGeometry(2.9, 0.08, 8, 44)
ringGeom.rotateX(-Math.PI / 2)

/** ピンの先端が指す高さ [m]。車の屋根（約 1.6m）より上 */
const PIN_TIP_Y = 2.6
const PIN_CONE_H = 1.1
const PIN_BALL_R = 0.46

const pinConeGeom = new THREE.ConeGeometry(PIN_BALL_R * 0.95, PIN_CONE_H, 16)
pinConeGeom.rotateX(Math.PI)
pinConeGeom.translate(0, PIN_CONE_H / 2, 0)

const pinBallGeom = new THREE.SphereGeometry(PIN_BALL_R, 20, 14)
pinBallGeom.translate(0, PIN_CONE_H + PIN_BALL_R * 0.55, 0)

/** 車両の見た目の状態。色と自己発光を決める */
const STATE_NORMAL = 0
const STATE_GOAL = 1
const STATE_COLLIDED = 2

/** MeshStandardMaterial に「インスタンスごとの自己発光」を足す。 */
function attachInstanceEmissive(material: THREE.MeshStandardMaterial): void {
  material.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        '#include <common>\nattribute vec3 instanceEmissive;\nvarying vec3 vInstanceEmissive;',
      )
      .replace(
        '#include <begin_vertex>',
        '#include <begin_vertex>\nvInstanceEmissive = instanceEmissive;',
      )
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vInstanceEmissive;')
      .replace(
        'vec3 totalEmissiveRadiance = emissive;',
        'vec3 totalEmissiveRadiance = emissive + vInstanceEmissive;',
      )
  }
  material.customProgramCacheKey = () => 'instanceEmissive'
}

export interface VehiclesProps {
  maxVehicles: number
  castShadow: boolean
}

export function Vehicles({ maxVehicles, castShadow }: VehiclesProps) {
  const palette = usePalette()

  const stateColors = useMemo(
    () => ({
      collided: new THREE.Color(palette.vehicleCollided),
      /** 目的地に着いたときの自己発光。palette の色を薄めて使う */
      reachedEmissive: new THREE.Color(palette.vehicleReached).multiplyScalar(0.35),
    }),
    [palette.vehicleCollided, palette.vehicleReached],
  )

  const count = Math.max(1, maxVehicles)

  const bodyRef = useRef<THREE.InstancedMesh>(null)
  const noseRef = useRef<THREE.InstancedMesh>(null)
  const cabinRef = useRef<THREE.InstancedMesh>(null)
  const wheelRef = useRef<THREE.InstancedMesh>(null)
  const ringRef = useRef<THREE.Mesh>(null)
  const pinRef = useRef<THREE.Group>(null)

  const pose = useMemo(createPose, [])
  const spin = useRef(0)
  /** 直前に色を書いたときの状態。変わったときだけ GPU へ送る */
  const lastState = useRef<Int8Array>(new Int8Array(0))

  const resources = useMemo(() => {
    const bodyGeometry = makeBodyGeometry()
    const noseGeometry = makeNoseGeometry()
    const cabinGeometry = makeCabinGeometry()
    const wheelGeometry = makeWheelGeometry()

    const bodyEmissive = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3)
    const noseEmissive = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3)
    bodyGeometry.setAttribute('instanceEmissive', bodyEmissive)
    noseGeometry.setAttribute('instanceEmissive', noseEmissive)

    const bodyMaterial = new THREE.MeshStandardMaterial({
      color: '#ffffff',
      roughness: 0.4,
      metalness: 0.45,
    })
    attachInstanceEmissive(bodyMaterial)

    const glassMaterial = new THREE.MeshStandardMaterial({
      color: palette.vehicleGlass,
      roughness: 0.25,
      metalness: 0.6,
    })
    const wheelMaterial = new THREE.MeshStandardMaterial({
      color: palette.vehicleWheel,
      roughness: 0.9,
      metalness: 0.1,
    })
    const ringMaterial = new THREE.MeshBasicMaterial({
      color: palette.vehicleHighlight,
      transparent: true,
      opacity: 0.85,
    })
    const pinMaterial = new THREE.MeshBasicMaterial({
      color: '#ffffff',
      toneMapped: false,
      depthTest: false,
      depthWrite: false,
      transparent: true,
      opacity: 1,
    })

    /** 車体色。スロット番号ごとに固定（パネル側の一覧と揃うこと） */
    const baseColors = Array.from({ length: count }, (_, i) => new THREE.Color(vehicleColor(i)))

    return {
      bodyGeometry,
      noseGeometry,
      cabinGeometry,
      wheelGeometry,
      bodyEmissive,
      noseEmissive,
      bodyMaterial,
      glassMaterial,
      wheelMaterial,
      ringMaterial,
      pinMaterial,
      baseColors,
    }
  }, [count])

  useEffect(() => {
    lastState.current = new Int8Array(count).fill(-1)
  }, [count, resources])

  useEffect(() => {
    resources.glassMaterial.color.set(palette.vehicleGlass)
    resources.wheelMaterial.color.set(palette.vehicleWheel)
    resources.ringMaterial.color.set(palette.vehicleHighlight)
    lastState.current.fill(-1)
  }, [resources, palette])

  useEffect(() => {
    const r = resources
    return () => {
      r.bodyGeometry.dispose()
      r.noseGeometry.dispose()
      r.cabinGeometry.dispose()
      r.wheelGeometry.dispose()
      r.bodyMaterial.dispose()
      r.glassMaterial.dispose()
      r.wheelMaterial.dispose()
      r.ringMaterial.dispose()
      r.pinMaterial.dispose()
    }
  }, [resources])

  const scratch = useMemo(
    () => ({
      transform: createTransformScratch(),
      base: new THREE.Matrix4(),
      out: new THREE.Matrix4(),
      hidden: new THREE.Matrix4().makeScale(0, 0, 0),
      color: new THREE.Color(),
      emissive: new THREE.Color(),
    }),
    [],
  )

  useFrame((_state, delta) => {
    const body = bodyRef.current
    const nose = noseRef.current
    const cabin = cabinRef.current
    const wheel = wheelRef.current
    if (!body || !nose || !cabin || !wheel) return
    if (lastState.current.length !== count) return

    const store = useSimStore.getState()
    const paused = store.status.renderPaused
    const alpha = computeAlpha(performance.now(), paused)
    const followTarget = store.followTarget
    const driverView = store.cameraMode === 'driver'

    const now = performance.now()
    const flash = 0.5 + 0.5 * Math.sin(now * 0.018)
    let colorDirty = false
    let highlightX = 0
    let highlightY = 0
    let hasHighlight = false
    let rollSpeed = 0

    for (let id = 0; id < count; id++) {
      const ok = sampleVehicle(id, alpha, pose)
      const hiddenAsEgo = driverView && followTarget === id
      if (!ok || hiddenAsEgo) {
        body.setMatrixAt(id, scratch.hidden)
        nose.setMatrixAt(id, scratch.hidden)
        cabin.setMatrixAt(id, scratch.hidden)
        for (let k = 0; k < 4; k++) wheel.setMatrixAt(id * 4 + k, scratch.hidden)
        continue
      }

      if (rollSpeed === 0 || id === followTarget) rollSpeed = pose.speed

      composeVehicleMatrix(scratch.transform, pose.x, pose.y, pose.heading, scratch.base)

      body.setMatrixAt(id, scratch.base)
      nose.setMatrixAt(id, scratch.base)
      cabin.setMatrixAt(id, scratch.base)

      for (let k = 0; k < WHEEL_OFFSETS.length; k++) {
        composeWheelMatrix(
          scratch.transform,
          scratch.base,
          k,
          pose.steer,
          spin.current,
          scratch.out,
        )
        wheel.setMatrixAt(id * 4 + k, scratch.out)
      }

      const state = pose.collided ? STATE_COLLIDED : pose.reachedGoal ? STATE_GOAL : STATE_NORMAL
      if (state === STATE_COLLIDED || lastState.current[id] !== state) {
        lastState.current[id] = state
        colorDirty = true
        const base = resources.baseColors[id]
        if (state === STATE_COLLIDED) {
          scratch.color.copy(base).lerp(stateColors.collided, 0.75)
          scratch.emissive.copy(stateColors.collided).multiplyScalar(0.35 + 0.45 * flash)
        } else if (state === STATE_GOAL) {
          scratch.color.copy(base)
          scratch.emissive.copy(stateColors.reachedEmissive)
        } else {
          scratch.color.copy(base)
          scratch.emissive.setRGB(0, 0, 0)
        }
        body.setColorAt(id, scratch.color)
        nose.setColorAt(id, scratch.color)
        const e = id * 3
        resources.bodyEmissive.array[e] = scratch.emissive.r
        resources.bodyEmissive.array[e + 1] = scratch.emissive.g
        resources.bodyEmissive.array[e + 2] = scratch.emissive.b
        resources.noseEmissive.array[e] = scratch.emissive.r
        resources.noseEmissive.array[e + 1] = scratch.emissive.g
        resources.noseEmissive.array[e + 2] = scratch.emissive.b
      }

      if (id === followTarget) {
        hasHighlight = true
        highlightX = pose.x
        highlightY = pose.y
      }
    }

    if (!paused) spin.current -= (rollSpeed * delta) / WHEEL_RADIUS

    body.instanceMatrix.needsUpdate = true
    nose.instanceMatrix.needsUpdate = true
    cabin.instanceMatrix.needsUpdate = true
    wheel.instanceMatrix.needsUpdate = true
    if (colorDirty) {
      if (body.instanceColor) body.instanceColor.needsUpdate = true
      if (nose.instanceColor) nose.instanceColor.needsUpdate = true
      resources.bodyEmissive.needsUpdate = true
      resources.noseEmissive.needsUpdate = true
    }

    const ring = ringRef.current
    if (ring) {
      ring.visible = hasHighlight
      if (hasHighlight) {
        ring.position.set(highlightX, 0.06, -highlightY)
        ring.scale.setScalar(1 + 0.05 * Math.sin(now * 0.004))
      }
    }
    const pin = pinRef.current
    if (pin) {
      pin.visible = hasHighlight
      if (hasHighlight) {
        const bob = 0.18 * Math.sin(now * 0.0026)
        pin.position.set(highlightX, PIN_TIP_Y + bob, -highlightY)
        const target = resources.baseColors[followTarget]
        if (target) resources.pinMaterial.color.copy(target)
      }
    }
  })

  return (
    <group>
      <instancedMesh
        key={`body-${count}`}
        ref={bodyRef}
        args={[resources.bodyGeometry, resources.bodyMaterial, count]}
        castShadow={castShadow}
        frustumCulled={false}
      />
      <instancedMesh
        key={`nose-${count}`}
        ref={noseRef}
        args={[resources.noseGeometry, resources.bodyMaterial, count]}
        frustumCulled={false}
      />
      <instancedMesh
        key={`cabin-${count}`}
        ref={cabinRef}
        args={[resources.cabinGeometry, resources.glassMaterial, count]}
        castShadow={castShadow}
        frustumCulled={false}
      />
      <instancedMesh
        key={`wheel-${count}`}
        ref={wheelRef}
        args={[resources.wheelGeometry, resources.wheelMaterial, count * 4]}
        castShadow={castShadow}
        frustumCulled={false}
      />

      <mesh ref={ringRef} geometry={ringGeom} material={resources.ringMaterial} visible={false} />

      <group ref={pinRef} visible={false}>
        <mesh geometry={pinConeGeom} material={resources.pinMaterial} renderOrder={999} />
        <mesh geometry={pinBallGeom} material={resources.pinMaterial} renderOrder={999} />
      </group>
    </group>
  )
}
