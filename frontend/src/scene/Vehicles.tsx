/** 車両。 */

import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

import { frameBuffer } from '../store/frameBuffer'
import { useSimStore } from '../store/simStore'
import { isWaitingPhase } from '../types/protocol'
import { computeAlpha, createPose, sampleVehicle } from './interpolation'
import {
  PLATES_PER_VEHICLE,
  PLATE_H,
  PLATE_SLOTS,
  PLATE_W,
  plateUvRow,
} from './licensePlate'
import { createPlateAtlas } from './licensePlateTexture'
import { VEHICLE_LIGHT_COLORS, VEHICLE_LIGHT_OFF } from './palette'
import { usePalette } from './usePalette'
import { vehicleColor } from './vehicleColors'
import {
  LIGHTS_PER_VEHICLE,
  LIGHT_SIZE,
  LIGHT_SLOTS,
  blinkOn,
  headlightsOn,
  lightIntensity,
  lightStateFor,
} from './vehicleLights'
import {
  WHEEL_OFFSETS,
  WHEEL_RADIUS,
  composeLightMatrix,
  composePlateMatrix,
  composeVehicleMatrix,
  composeWheelMatrix,
  createTransformScratch,
  makeBodyGeometry,
  makeCabinGeometry,
  makeLightGeometry,
  makeNoseGeometry,
  makePlateGeometry,
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

/**
 * 1 枚のアトラスから「そのインスタンスの段」だけを貼る。
 * **instancedMesh は 1 つのテクスチャしか持てない**ので、8 台ぶんを縦に並べて UV をずらす。
 */
function attachPlateAtlas(material: THREE.MeshStandardMaterial, count: number): void {
  const rows = Math.max(1, count).toFixed(1)
  material.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        '#include <common>\nattribute float instancePlateRow;',
      )
      .replace(
        '#include <uv_vertex>',
        `#include <uv_vertex>
#ifdef USE_MAP
	vMapUv = vec2( vMapUv.x, ( instancePlateRow + vMapUv.y ) / ${rows} );
#endif`,
      )
  }
  material.customProgramCacheKey = () => `plateAtlas:${rows}`
}

export interface VehiclesProps {
  maxVehicles: number
  castShadow: boolean
}

export function Vehicles({ maxVehicles, castShadow }: VehiclesProps) {
  const palette = usePalette()
  // プレートの地名は走っている街に合わせる（licensePlate.ts の対応表）
  const presetId = useSimStore((s) => s.status.presetId)

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
  const lightRef = useRef<THREE.InstancedMesh>(null)
  const plateRef = useRef<THREE.InstancedMesh>(null)
  const ringRef = useRef<THREE.Mesh>(null)
  const pinRef = useRef<THREE.Group>(null)

  const pose = useMemo(createPose, [])
  const spin = useRef(0)
  /** 直前に色を書いたときの状態。変わったときだけ GPU へ送る */
  const lastState = useRef<Int8Array>(new Int8Array(0))
  /** 直前に書いたライトの明るさ。1.5Hz の点滅で毎フレーム送らないため */
  const lastLights = useRef<Float32Array>(new Float32Array(0))

  const resources = useMemo(() => {
    const bodyGeometry = makeBodyGeometry()
    const noseGeometry = makeNoseGeometry()
    const cabinGeometry = makeCabinGeometry()
    const wheelGeometry = makeWheelGeometry()
    const lightGeometry = makeLightGeometry(LIGHT_SIZE)
    const plateGeometry = makePlateGeometry(PLATE_W, PLATE_H)
    const plateRows = new THREE.InstancedBufferAttribute(
      new Float32Array(count * PLATES_PER_VEHICLE),
      1,
    )
    plateGeometry.setAttribute('instancePlateRow', plateRows)

    const lightCount = count * LIGHTS_PER_VEHICLE
    const lightEmissive = new THREE.InstancedBufferAttribute(
      new Float32Array(lightCount * 3),
      3,
    )
    lightGeometry.setAttribute('instanceEmissive', lightEmissive)

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
    // 灯体そのものは暗く、点灯は instanceEmissive で出す（消灯時に黒い穴にしない）
    const lightMaterial = new THREE.MeshStandardMaterial({
      color: VEHICLE_LIGHT_OFF,
      roughness: 0.35,
      metalness: 0.1,
      toneMapped: false,
    })
    attachInstanceEmissive(lightMaterial)

    const plateMaterial = new THREE.MeshStandardMaterial({
      roughness: 0.55,
      metalness: 0.05,
      // 夜でも読めるよう、わずかに自己発光させる（反射板の代わり）
      emissive: new THREE.Color('#2a2a26'),
    })
    attachPlateAtlas(plateMaterial, count)

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

    /** 灯体の色。点いていなくても赤・橙の見分けはつくよう、薄く混ぜておく */
    const lightColors = LIGHT_SLOTS.map((slot) =>
      new THREE.Color(VEHICLE_LIGHT_OFF).lerp(
        new THREE.Color(VEHICLE_LIGHT_COLORS[slot.kind]),
        0.45,
      ),
    )
    /** 点灯したときの発光色 */
    const lightEmissiveColors = LIGHT_SLOTS.map(
      (slot) => new THREE.Color(VEHICLE_LIGHT_COLORS[slot.kind]),
    )

    return {
      bodyGeometry,
      noseGeometry,
      cabinGeometry,
      wheelGeometry,
      lightGeometry,
      plateGeometry,
      plateRows,
      plateMaterial,
      bodyEmissive,
      noseEmissive,
      lightEmissive,
      bodyMaterial,
      glassMaterial,
      wheelMaterial,
      lightMaterial,
      ringMaterial,
      pinMaterial,
      baseColors,
      lightColors,
      lightEmissiveColors,
    }
  }, [count])

  useEffect(() => {
    lastState.current = new Int8Array(count).fill(-1)
    lastLights.current = new Float32Array(count * LIGHTS_PER_VEHICLE).fill(-1)
    for (let id = 0; id < count; id++) {
      const row = plateUvRow(id, count)
      for (let k = 0; k < PLATES_PER_VEHICLE; k++) {
        resources.plateRows.array[id * PLATES_PER_VEHICLE + k] = row
      }
    }
    resources.plateRows.needsUpdate = true
    const lights = lightRef.current
    if (lights) {
      for (let id = 0; id < count; id++) {
        for (let k = 0; k < LIGHTS_PER_VEHICLE; k++) {
          lights.setColorAt(id * LIGHTS_PER_VEHICLE + k, resources.lightColors[k])
        }
      }
      if (lights.instanceColor) lights.instanceColor.needsUpdate = true
    }
  }, [count, resources])

  // ★ テクスチャは resources とは別に持つ。プリセットだけが変わったときに
    //   ジオメトリごと作り直すと、instancedMesh が count=0 に戻って車が消える
  useEffect(() => {
    const atlas = createPlateAtlas(count, presetId)
    resources.plateMaterial.map = atlas
    resources.plateMaterial.needsUpdate = true
    return () => {
      resources.plateMaterial.map = null
      atlas.dispose()
    }
  }, [resources, count, presetId])

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
      r.lightGeometry.dispose()
      r.plateGeometry.dispose()
      r.plateMaterial.dispose()
      r.bodyMaterial.dispose()
      r.glassMaterial.dispose()
      r.wheelMaterial.dispose()
      r.lightMaterial.dispose()
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
      light: new THREE.Color(),
    }),
    [],
  )

  useFrame((_state, delta) => {
    const body = bodyRef.current
    const nose = noseRef.current
    const cabin = cabinRef.current
    const wheel = wheelRef.current
    const lights = lightRef.current
    const plates = plateRef.current
    if (!body || !nose || !cabin || !wheel || !lights || !plates) return
    if (lastState.current.length !== count) return

    const store = useSimStore.getState()
    const paused = store.status.renderPaused
    const alpha = computeAlpha(performance.now(), paused)
    const followTarget = store.followTarget
    const driverView = store.cameraMode === 'driver'

    const now = performance.now()
    const flash = 0.5 + 0.5 * Math.sin(now * 0.018)
    // ライトは全車で共通の条件（天候・昼夜・点滅の位相）を先に 1 回だけ出す
    const weather = frameBuffer.weather
    const headOn = headlightsOn(weather.rain, weather.fog, store.theme === 'dark')
    const blink = blinkOn(now)
    const taxi = store.taxi
    const hazardId = isWaitingPhase(taxi.phase) ? taxi.vehicleId : -1
    let colorDirty = false
    let lightDirty = false
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
        for (let k = 0; k < LIGHTS_PER_VEHICLE; k++) {
          lights.setMatrixAt(id * LIGHTS_PER_VEHICLE + k, scratch.hidden)
        }
        for (let k = 0; k < PLATES_PER_VEHICLE; k++) {
          plates.setMatrixAt(id * PLATES_PER_VEHICLE + k, scratch.hidden)
        }
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

      for (let k = 0; k < PLATES_PER_VEHICLE; k++) {
        composePlateMatrix(
          scratch.transform,
          scratch.base,
          PLATE_SLOTS[k].position,
          PLATE_SLOTS[k].yaw,
          scratch.out,
        )
        plates.setMatrixAt(id * PLATES_PER_VEHICLE + k, scratch.out)
      }

      const lightState = lightStateFor({
        braking: pose.braking,
        turnSignal: pose.turnSignal,
        hazard: id === hazardId,
        headlights: headOn,
        blink,
      })
      for (let k = 0; k < LIGHTS_PER_VEHICLE; k++) {
        const index = id * LIGHTS_PER_VEHICLE + k
        composeLightMatrix(
          scratch.transform,
          scratch.base,
          LIGHT_SLOTS[k].position,
          scratch.out,
        )
        lights.setMatrixAt(index, scratch.out)

        const level = lightIntensity(k, lightState)
        if (lastLights.current[index] === level) continue
        lastLights.current[index] = level
        lightDirty = true
        scratch.light.copy(resources.lightEmissiveColors[k]).multiplyScalar(level)
        const e = index * 3
        resources.lightEmissive.array[e] = scratch.light.r
        resources.lightEmissive.array[e + 1] = scratch.light.g
        resources.lightEmissive.array[e + 2] = scratch.light.b
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
    lights.instanceMatrix.needsUpdate = true
    plates.instanceMatrix.needsUpdate = true
    if (lightDirty) resources.lightEmissive.needsUpdate = true
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

      <instancedMesh
        key={`light-${count}`}
        ref={lightRef}
        args={[resources.lightGeometry, resources.lightMaterial, count * LIGHTS_PER_VEHICLE]}
        frustumCulled={false}
      />

      <instancedMesh
        key={`plate-${count}`}
        ref={plateRef}
        args={[resources.plateGeometry, resources.plateMaterial, count * PLATES_PER_VEHICLE]}
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
