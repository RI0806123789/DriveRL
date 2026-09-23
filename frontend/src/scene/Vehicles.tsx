/** 車両。近景（細かい形と内装）と遠景（粗い形）の 2 段を、車ごとに毎フレーム選んで描く。 */

import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'

import { frameBuffer } from '../store/frameBuffer'
import { useSimStore } from '../store/simStore'
import { isWaitingPhase } from '../types/protocol'
import { computeAlpha, createPose, sampleVehicle } from './interpolation'
import { PLATES_PER_VEHICLE, PLATE_H, PLATE_SLOTS, PLATE_W, plateUvRow } from './licensePlate'
import { createPlateAtlas } from './licensePlateTexture'
import { VEHICLE_LIGHT_COLORS, VEHICLE_LIGHT_OFF } from './palette'
import { usePalette } from './usePalette'
import { vehicleColor } from './vehicleColors'
import {
  LIGHTS_PER_VEHICLE,
  LIGHT_HEAD,
  LIGHT_SLOTS,
  blinkOn,
  composeLampMatrix,
  headlightsOn,
  lightIntensity,
  lightStateFor,
  makeHeadlightPoolGeometry,
  makeLampLensGeometry,
} from './vehicleLights'
import {
  GAUGE_KINDS,
  GAUGE_SLOTS,
  POWER_NEEDLE_TAU_S,
  TAXI_DOOR_OPEN,
  TURN_INDICATOR_SIZE,
  TURN_INDICATOR_SLOTS,
  WHEEL_RADIUS,
  composeBodyMatrix,
  composeCaliperMatrix,
  composeFixedMatrix,
  composeNeedleMatrix,
  composePedalMatrix,
  composePlateMatrix,
  composeSteeringMatrix,
  composeTurnIndicatorMatrix,
  composeVehicleMatrix,
  composeWheelMatrix,
  createTransformScratch,
  dampNeedle,
  gaugeUvRow,
  makeGaugeFaceGeometry,
  makeNeedleGeometry,
  makePedalGeometry,
  makePlateGeometry,
  makeSteeringGeometry,
  makeTurnIndicatorGeometry,
  powerRatio,
  speedRatio,
} from './vehicleGeometry'
import {
  makeBodyChromeGeometry,
  makeBodyPaintGeometry,
  makeBodyTrimGeometry,
  makeFarBodyGeometry,
  makeVehicleGlass,
} from './vehicleBody'
import { makeInteriorGeometry } from './vehicleInterior'
import {
  CALIPER_ANGLE,
  makeCaliperGeometry,
  makeFarWheelGeometry,
  makeRimGeometry,
  makeTyreGeometry,
} from './vehicleWheels'
import { createGaugeAtlas } from './gaugeTexture'
import { createTaxiSignAtlas } from './taxiSignTexture'
import { lampGlow, makeTaxiSignGeometry, statusRow, taxiSignStatus } from './taxiSign'
import {
  composeTiltMatrix,
  createTiltState,
  ensureTiltSlots,
  resetTilt,
  stepTilt,
  vehicleTilt,
  type TiltState,
} from './vehicleMotion'
import { LOD_FAR, LOD_NEAR, nearestViewDistance, nextLod } from './vehicleLod'
import {
  advanceWiper,
  composeWiperLocal,
  ensureWiperSlots,
  makeWiperGeometry,
  vehicleWipers,
  wiperAngle,
} from './wiper'
import {
  applyVehicleEnvironment,
  applyVehiclePalette,
  applyVehicleWeather,
  createHeadlightPoolTexture,
  createVehicleMaterials,
  disposeVehicleMaterials,
  tyreWetDarken,
} from './vehicleMaterials'
import { createVehicleEnvironment } from './vehicleEnvironment'
import {
  FAR_PARTS,
  GAUGES_PER_VEHICLE,
  INDICATORS_PER_VEHICLE,
  NEAR_PARTS,
  PEDALS_PER_VEHICLE,
  VEHICLE_PARTS,
  WHEELS_PER_VEHICLE,
  WIPERS_PER_VEHICLE,
  type PartKey,
} from './vehicleParts'
import { driverEye } from './cameraMath'
import { displayedWeather, weatherLook } from './weatherView'


/** 車両の動き（傾き・ワイパー・ドア）を進める優先度。**カメラ・ナビ・水滴より先に走らせる**（負なら自動描画は止まらない） */
const MOTION_PRIORITY = -1

/** ドアが開き切る・閉まり切るまでの時間 [s] */
const DOOR_OPEN_SEC = 1.3
const DOOR_CLOSE_SEC = 1.0
/** これより遅ければ止まっているとみなしてドアを開ける [m/s] */
const DOOR_STOPPED_MPS = 0.3

/** メーターの針が指す割合 0..1。POWER は慣性を付けた割合（`dampNeedle`）を渡す */
function gaugeRatio(kind: (typeof GAUGE_KINDS)[number], speed: number, power: number): number {
  return kind === 'speed' ? speedRatio(speed) : power
}

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


function makeGeometries(count: number) {
  const g: Record<PartKey, THREE.BufferGeometry> = {
    bodyNear: makeBodyPaintGeometry(),
    trim: makeBodyTrimGeometry(),
    chrome: makeBodyChromeGeometry(),
    interior: makeInteriorGeometry(),
    steering: makeSteeringGeometry(),
    pedal: makePedalGeometry(),
    gauge: makeGaugeFaceGeometry(GAUGE_SLOTS[0].radius),
    needle: makeNeedleGeometry(GAUGE_SLOTS[0].radius),
    indicator: makeTurnIndicatorGeometry(TURN_INDICATOR_SIZE),
    tyre: makeTyreGeometry(),
    rim: makeRimGeometry(),
    caliper: makeCaliperGeometry(),
    wiper: makeWiperGeometry(),
    bodyFar: makeFarBodyGeometry(),
    wheelFar: makeFarWheelGeometry(),
    glass: makeVehicleGlass(),
    lamp: makeLampLensGeometry(),
    plate: makePlateGeometry(PLATE_W, PLATE_H),
    sign: makeTaxiSignGeometry(),
    pool: makeHeadlightPoolGeometry(),
  }
  const attr = (key: PartKey, name: string, size: number, per: number) => {
    const a = new THREE.InstancedBufferAttribute(new Float32Array(count * per * size), size)
    g[key].setAttribute(name, a)
    return a
  }
  return Object.assign(g, {
    attrs: {
      bodyEmissive: attr('bodyNear', 'instanceEmissive', 3, 1),
      farEmissive: attr('bodyFar', 'instanceEmissive', 3, 1),
      doorBody: attr('bodyNear', 'instanceDoor', 1, 1),
      doorTrim: attr('trim', 'instanceDoor', 1, 1),
      doorChrome: attr('chrome', 'instanceDoor', 1, 1),
      doorGlass: attr('glass', 'instanceDoor', 1, 1),
      doorInterior: attr('interior', 'instanceDoor', 1, 1),
      lampEmissive: attr('lamp', 'instanceEmissive', 3, LIGHTS_PER_VEHICLE),
      indicatorEmissive: attr('indicator', 'instanceEmissive', 3, INDICATORS_PER_VEHICLE),
      gaugeRows: attr('gauge', 'instanceGaugeRow', 1, GAUGES_PER_VEHICLE),
      plateRows: attr('plate', 'instancePlateRow', 1, PLATES_PER_VEHICLE),
      signRows: attr('sign', 'instanceSignRow', 1, 1),
      signGlow: attr('sign', 'instanceSignGlow', 2, 1),
    },
  })
}

type Geometries = ReturnType<typeof makeGeometries>

export interface VehiclesProps {
  maxVehicles: number
  castShadow: boolean
}

export function Vehicles({ maxVehicles, castShadow }: VehiclesProps) {
  const palette = usePalette()
  const gl = useThree((s) => s.gl)
  // プレートの地名は走っている街に合わせる（licensePlate.ts の対応表）
  const presetId = useSimStore((s) => s.status.presetId)
  const theme = useSimStore((s) => s.theme)

  const stateColors = useMemo(
    () => ({
      collided: new THREE.Color(palette.vehicleCollided),
      /** 目的地に着いたときの自己発光。palette の色を薄めて使う */
      reachedEmissive: new THREE.Color(palette.vehicleReached).multiplyScalar(0.35),
    }),
    [palette.vehicleCollided, palette.vehicleReached],
  )

  const count = Math.max(1, maxVehicles)
  const meshes = useRef<Partial<Record<PartKey, THREE.InstancedMesh>>>({})
  const ringRef = useRef<THREE.Mesh>(null)
  const pinRef = useRef<THREE.Group>(null)

  const pose = useMemo(createPose, [])
  const spin = useRef(0)

  /** 車ごとの状態。台数が変わったら作り直す */
  const perCar = useMemo(
    () => ({
      tilt: Array.from({ length: count }, (): TiltState => createTiltState()),
      lod: new Uint8Array(count).fill(LOD_FAR),
      door: new Float32Array(count),
      /** 直前に書いた値。変わったときだけ GPU へ送る（-1 は未送信） */
      lastState: new Int8Array(count).fill(-1),
      lastLights: new Float32Array(count * LIGHTS_PER_VEHICLE).fill(-1),
      lastIndicators: new Float32Array(count * INDICATORS_PER_VEHICLE).fill(-1),
      lastDoor: new Float32Array(count).fill(-1),
      lastSign: new Float32Array(count * 2).fill(-1),
      /** POWER の針がいま指している割合。負なら未設定（次のフレームで指令へ合わせる） */
      powerNeedle: new Float32Array(count).fill(-1),
    }),
    [count],
  )

  const resources = useMemo(() => {
    const geometries = makeGeometries(count)
    const materials = createVehicleMaterials(palette, count, GAUGE_KINDS.length)
    /** 車体色。スロット番号ごとに固定（パネル側の一覧と揃うこと） */
    const baseColors = Array.from({ length: count }, (_, i) => new THREE.Color(vehicleColor(i)))
    /** 灯体の色。点いていなくても赤・橙の見分けがつくよう薄く混ぜる（前照灯は透明なレンズらしく明るく） */
    const lightColors = LIGHT_SLOTS.map((slot) =>
      slot.kind === LIGHT_HEAD
        ? new THREE.Color('#c8d0d8')
        : new THREE.Color(VEHICLE_LIGHT_OFF).lerp(new THREE.Color(VEHICLE_LIGHT_COLORS[slot.kind]), 0.5),
    )
    /** 点灯したときの発光色 */
    const lightEmissiveColors = LIGHT_SLOTS.map((slot) => new THREE.Color(VEHICLE_LIGHT_COLORS[slot.kind]))
    return { geometries, materials, baseColors, lightColors, lightEmissiveColors }
    // 配色は下の effect で差し替えるので、ここでは作り直さない
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [count])

  useEffect(() => {
    perCar.lastState.fill(-1)
    const { attrs } = resources.geometries
    for (let id = 0; id < count; id++) {
      const row = plateUvRow(id, count)
      for (let k = 0; k < PLATES_PER_VEHICLE; k++) attrs.plateRows.array[id * PLATES_PER_VEHICLE + k] = row
      // 文字盤の段はメーターの種類で決まる。ナンバープレートと同じく `gaugeUvRow()` で v の向きへ直す
      for (let k = 0; k < GAUGES_PER_VEHICLE; k++) {
        attrs.gaugeRows.array[id * GAUGES_PER_VEHICLE + k] = gaugeUvRow(
          GAUGE_KINDS.indexOf(GAUGE_SLOTS[k].kind),
          GAUGE_KINDS.length,
        )
      }
    }
    attrs.plateRows.needsUpdate = true
    attrs.gaugeRows.needsUpdate = true
    const lamps = meshes.current.lamp
    if (lamps) {
      for (let id = 0; id < count; id++) {
        for (let k = 0; k < LIGHTS_PER_VEHICLE; k++) lamps.setColorAt(id * LIGHTS_PER_VEHICLE + k, resources.lightColors[k])
      }
      if (lamps.instanceColor) lamps.instanceColor.needsUpdate = true
    }
  }, [count, resources, perCar])

  // テクスチャは resources とは別に持つ。プリセットだけが変わったときにジオメトリごと作り直すと、
  //   instancedMesh が count=0 に戻って車が消える
  useEffect(() => {
    const atlas = createGaugeAtlas()
    const m = resources.materials.gauge
    m.map = atlas
    m.emissiveMap = atlas
    m.needsUpdate = true
    const sign = createTaxiSignAtlas()
    const s = resources.materials.sign
    s.map = sign
    s.emissiveMap = sign
    s.needsUpdate = true
    const pool = createHeadlightPoolTexture()
    resources.materials.pool.map = pool
    resources.materials.pool.needsUpdate = true
    return () => {
      m.map = null
      m.emissiveMap = null
      s.map = null
      s.emissiveMap = null
      resources.materials.pool.map = null
      atlas.dispose()
      sign.dispose()
      pool.dispose()
    }
  }, [resources])

  useEffect(() => {
    const atlas = createPlateAtlas(count, presetId)
    resources.materials.plate.map = atlas
    resources.materials.plate.needsUpdate = true
    return () => {
      resources.materials.plate.map = null
      atlas.dispose()
    }
  }, [resources, count, presetId])

  // 環境マップは昼夜で作り直す（天候は強さだけ変える）
  useEffect(() => {
    const env = createVehicleEnvironment(gl, palette, theme === 'dark')
    applyVehicleEnvironment(resources.materials, env.texture)
    return () => {
      applyVehicleEnvironment(resources.materials, null)
      env.dispose()
    }
  }, [gl, resources, palette, theme])

  // 色だけ差し替える。マテリアルを作り直すと `args` が変わり、instancedMesh が count=0 に戻って車が消える
  useEffect(() => {
    applyVehiclePalette(resources.materials, palette)
    // 覚えているのは明るさだけなので、色が変わったら**全部**塗り直させる
    perCar.lastState.fill(-1)
    perCar.lastLights.fill(-1)
    perCar.lastIndicators.fill(-1)
  }, [resources, palette, perCar])

  useEffect(() => {
    const r = resources
    return () => {
      for (const g of Object.values(r.geometries)) {
        if (g instanceof THREE.BufferGeometry) g.dispose()
      }
      disposeVehicleMaterials(r.materials)
    }
  }, [resources])

  const scratch = useMemo(
    () => ({
      transform: createTransformScratch(),
      base: new THREE.Matrix4(),
      tilt: new THREE.Matrix4(),
      body: new THREE.Matrix4(),
      local: new THREE.Matrix4(),
      out: new THREE.Matrix4(),
      hidden: new THREE.Matrix4().makeScale(0, 0, 0),
      color: new THREE.Color(),
      emissive: new THREE.Color(),
      light: new THREE.Color(),
      views: [] as Array<[number, number, number]>,
    }),
    [],
  )

  useFrame((state, delta) => {
    const m = meshes.current
    const get = (key: PartKey) => m[key]
    for (const p of VEHICLE_PARTS) if (!m[p.key]) return
    if (perCar.lastState.length !== count) return
    ensureTiltSlots(count)
    ensureWiperSlots(count)
    const geo: Geometries = resources.geometries
    const mats = resources.materials

    const store = useSimStore.getState()
    const paused = store.status.renderPaused
    // 同じフレームの中ではカメラ・ナビ・水滴と同じ時刻で補間する（ずれると車内が揺れる）
    const nowMs = state.clock.oldTime
    const alpha = computeAlpha(nowMs, paused)
    const nowSec = nowMs / 1000
    const followTarget = store.followTarget
    const taxi = store.taxi
    const practical = store.mode === 'taxi'
    const hazardId = isWaitingPhase(taxi.phase) ? taxi.vehicleId : -1
    const flash = 0.5 + 0.5 * Math.sin(nowMs * 0.018)
    // ライトは全車で共通の条件（天候・昼夜・点滅の位相）を先に 1 回だけ出す
    const weather = frameBuffer.weather
    const night = store.theme === 'dark'
    const headOn = headlightsOn(weather.rain, weather.fog, night)
    const blink = blinkOn(nowMs)
    const look = weatherLook(displayedWeather, palette.fogNear, palette.fogFar)
    applyVehicleWeather(mats, displayedWeather.wet, look.dim)
    mats.tyre.color.set(palette.vehicleWheel).multiplyScalar(tyreWetDarken(displayedWeather.wet))
    mats.pool.color.set('#fff0d2').multiplyScalar(night ? 0.5 : 0.16)

    // 近景に切り替える距離は、画面のカメラと（出していれば）車載カメラの近いほうで測る
    const views = scratch.views
    views.length = 0
    views.push([state.camera.position.x, state.camera.position.y, state.camera.position.z])
    const feedId = store.taxiCameraOn && practical ? taxi.vehicleId : -1
    const driverId = store.cameraMode === 'driver' ? followTarget : -1

    let maxNear = -1
    let maxFar = -1
    let maxAny = -1
    let colorDirty = false
    let lightDirty = false
    let indicatorDirty = false
    let doorDirty = false
    let signDirty = false
    let highlightX = 0
    let highlightY = 0
    let hasHighlight = false
    let rollSpeed = 0

    const hide = (key: PartKey, id: number, per: number) => {
      const mesh = get(key)!
      for (let k = 0; k < per; k++) mesh.setMatrixAt(id * per + k, scratch.hidden)
    }

    if (feedId >= 0 && sampleVehicle(feedId, alpha, pose)) {
      const eye = driverEye(pose.x, pose.y, pose.heading)
      views.push([eye.x, eye.y, eye.z])
    }

    for (let id = 0; id < count; id++) {
      const ok = sampleVehicle(id, alpha, pose)
      const tilt = perCar.tilt[id]
      if (!ok) {
        for (const p of VEHICLE_PARTS) hide(p.key, id, p.per)
        tilt.primed = false
        perCar.powerNeedle[id] = -1
        perCar.door[id] = 0
        vehicleTilt.pitch[id] = 0
        vehicleTilt.roll[id] = 0
        continue
      }
      maxAny = id
      if (rollSpeed === 0 || id === followTarget) rollSpeed = pose.speed

      // 傾き。**補間済みの速度**から加速度を出して均す（指令の throttle は 20Hz の段差なので使わない）
      if (pose.teleported) resetTilt(tilt, pose.speed)
      else if (!paused) stepTilt(tilt, pose.speed, pose.steer, delta)
      vehicleTilt.pitch[id] = tilt.pitch
      vehicleTilt.roll[id] = tilt.roll

      composeVehicleMatrix(scratch.transform, pose.x, pose.y, pose.heading, scratch.base)
      composeTiltMatrix(tilt.pitch, tilt.roll, scratch.tilt)
      composeBodyMatrix(scratch.base, scratch.tilt, scratch.body)

      const forced = id === driverId || id === feedId
      const lod = nextLod(perCar.lod[id], nearestViewDistance(views, pose.x, pose.y), forced)
      perCar.lod[id] = lod

      // ワイパー（雨の強さで止める／間欠／連続。時刻はクライアントの時計）
      const wipers = vehicleWipers.states[id]
      if (wipers && !paused) advanceWiper(wipers, nowSec, displayedWeather.rain)

      // 乗降のときだけ左後席のドアを開ける（見た目だけ）。止まってから開け、乗り降りが済んだら閉める
      const wantOpen =
        practical && id === taxi.vehicleId && isWaitingPhase(taxi.phase) && Math.abs(pose.speed) < DOOR_STOPPED_MPS
      const doorStep = (paused ? 0 : delta) * TAXI_DOOR_OPEN / (wantOpen ? DOOR_OPEN_SEC : DOOR_CLOSE_SEC)
      const door = wantOpen
        ? Math.min(TAXI_DOOR_OPEN, perCar.door[id] + doorStep)
        : Math.max(0, perCar.door[id] - doorStep)
      perCar.door[id] = door
      if (perCar.lastDoor[id] !== door) {
        perCar.lastDoor[id] = door
        doorDirty = true
        const eased = TAXI_DOOR_OPEN * (0.5 - 0.5 * Math.cos((Math.PI * door) / TAXI_DOOR_OPEN))
        const a = geo.attrs
        a.doorBody.array[id] = eased
        a.doorTrim.array[id] = eased
        a.doorChrome.array[id] = eased
        a.doorGlass.array[id] = eased
        a.doorInterior.array[id] = eased
      }

      if (lod === LOD_NEAR) {
        maxNear = id
        for (const p of FAR_PARTS) hide(p.key, id, p.per)
        get('bodyNear')!.setMatrixAt(id, scratch.body)
        get('trim')!.setMatrixAt(id, scratch.body)
        get('chrome')!.setMatrixAt(id, scratch.body)
        get('interior')!.setMatrixAt(id, scratch.body)

        // 可動部。**サーバーが送ってきた指令（steer / throttle）だけで決める**
        composeSteeringMatrix(scratch.transform, scratch.body, pose.steer, scratch.out)
        get('steering')!.setMatrixAt(id, scratch.out)
        for (let k = 0; k < PEDALS_PER_VEHICLE; k++) {
          composePedalMatrix(scratch.transform, scratch.body, k, pose.throttle, scratch.out)
          get('pedal')!.setMatrixAt(id * PEDALS_PER_VEHICLE + k, scratch.out)
        }

        // ペダルとブレーキランプは指令をそのまま使い、POWER の針だけ慣性を付ける
        const powerTarget = powerRatio(pose.throttle)
        const held = perCar.powerNeedle[id]
        const power =
          held < 0 || pose.teleported ? powerTarget : dampNeedle(held, powerTarget, delta, POWER_NEEDLE_TAU_S)
        perCar.powerNeedle[id] = power
        for (let k = 0; k < GAUGES_PER_VEHICLE; k++) {
          composeFixedMatrix(scratch.transform, scratch.body, GAUGE_SLOTS[k].center, scratch.out)
          get('gauge')!.setMatrixAt(id * GAUGES_PER_VEHICLE + k, scratch.out)
          const ratio = gaugeRatio(GAUGE_SLOTS[k].kind, pose.speed, power)
          composeNeedleMatrix(scratch.transform, scratch.body, k, ratio, scratch.out)
          get('needle')!.setMatrixAt(id * GAUGES_PER_VEHICLE + k, scratch.out)
        }
        for (let k = 0; k < INDICATORS_PER_VEHICLE; k++) {
          composeTurnIndicatorMatrix(scratch.transform, scratch.body, k, scratch.out)
          get('indicator')!.setMatrixAt(id * INDICATORS_PER_VEHICLE + k, scratch.out)
        }

        // 車輪は路面に残す（傾きを掛けない）
        for (let k = 0; k < WHEELS_PER_VEHICLE; k++) {
          composeWheelMatrix(scratch.transform, scratch.base, k, pose.steer, spin.current, scratch.out)
          get('tyre')!.setMatrixAt(id * WHEELS_PER_VEHICLE + k, scratch.out)
          get('rim')!.setMatrixAt(id * WHEELS_PER_VEHICLE + k, scratch.out)
          composeCaliperMatrix(scratch.transform, scratch.base, k, pose.steer, CALIPER_ANGLE, scratch.out)
          get('caliper')!.setMatrixAt(id * WHEELS_PER_VEHICLE + k, scratch.out)
        }
        const wipeAngle = wipers ? wiperAngle(wipers, nowSec) : 0
        for (let k = 0; k < WIPERS_PER_VEHICLE; k++) {
          composeWiperLocal(k, wipeAngle, scratch.local)
          scratch.out.multiplyMatrices(scratch.body, scratch.local)
          get('wiper')!.setMatrixAt(id * WIPERS_PER_VEHICLE + k, scratch.out)
        }
      } else {
        maxFar = id
        for (const p of NEAR_PARTS) hide(p.key, id, p.per)
        perCar.powerNeedle[id] = -1
        get('bodyFar')!.setMatrixAt(id, scratch.body)
        for (let k = 0; k < WHEELS_PER_VEHICLE; k++) {
          composeWheelMatrix(scratch.transform, scratch.base, k, pose.steer, spin.current, scratch.out)
          get('wheelFar')!.setMatrixAt(id * WHEELS_PER_VEHICLE + k, scratch.out)
        }
      }

      get('glass')!.setMatrixAt(id, scratch.body)
      for (let k = 0; k < PLATES_PER_VEHICLE; k++) {
        composePlateMatrix(scratch.transform, scratch.body, PLATE_SLOTS[k].position, PLATE_SLOTS[k].yaw, scratch.out)
        get('plate')!.setMatrixAt(id * PLATES_PER_VEHICLE + k, scratch.out)
      }
      get('pool')!.setMatrixAt(id, scratch.base)

      // タクシーの表示（実用モードだけ）。空車／迎車／賃走／支払は配車の段階から出す
      if (practical) {
        get('sign')!.setMatrixAt(id, scratch.body)
        const status = taxiSignStatus(id, taxi)
        const row = statusRow(status)
        const glow = lampGlow(status)
        if (perCar.lastSign[id * 2] !== row || perCar.lastSign[id * 2 + 1] !== glow) {
          perCar.lastSign[id * 2] = row
          perCar.lastSign[id * 2 + 1] = glow
          geo.attrs.signRows.array[id] = row
          geo.attrs.signGlow.array[id * 2] = glow
          geo.attrs.signGlow.array[id * 2 + 1] = 1
          signDirty = true
        }
      } else {
        hide('sign', id, 1)
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
        composeLampMatrix(scratch.body, LIGHT_SLOTS[k], scratch.out)
        get('lamp')!.setMatrixAt(index, scratch.out)
        const level = lightIntensity(k, lightState)
        if (perCar.lastLights[index] === level) continue
        perCar.lastLights[index] = level
        lightDirty = true
        scratch.light.copy(resources.lightEmissiveColors[k]).multiplyScalar(level)
        const e = index * 3
        geo.attrs.lampEmissive.array[e] = scratch.light.r
        geo.attrs.lampEmissive.array[e + 1] = scratch.light.g
        geo.attrs.lampEmissive.array[e + 2] = scratch.light.b
      }

      // 車外の方向指示器と**同じ `lightState`** から明るさを決める
      for (let k = 0; k < INDICATORS_PER_VEHICLE; k++) {
        const index = id * INDICATORS_PER_VEHICLE + k
        const on = TURN_INDICATOR_SLOTS[k].side < 0 ? lightState.left : lightState.right
        const level = on ? 1 : 0
        if (perCar.lastIndicators[index] === level) continue
        perCar.lastIndicators[index] = level
        indicatorDirty = true
        scratch.light.set(palette.vehicleIndicator).multiplyScalar(level * 1.6)
        const e = index * 3
        geo.attrs.indicatorEmissive.array[e] = scratch.light.r
        geo.attrs.indicatorEmissive.array[e + 1] = scratch.light.g
        geo.attrs.indicatorEmissive.array[e + 2] = scratch.light.b
      }

      const vstate = pose.collided ? STATE_COLLIDED : pose.reachedGoal ? STATE_GOAL : STATE_NORMAL
      if (vstate === STATE_COLLIDED || perCar.lastState[id] !== vstate) {
        perCar.lastState[id] = vstate
        colorDirty = true
        const base = resources.baseColors[id]
        if (vstate === STATE_COLLIDED) {
          scratch.color.copy(base).lerp(stateColors.collided, 0.75)
          scratch.emissive.copy(stateColors.collided).multiplyScalar(0.35 + 0.45 * flash)
        } else if (vstate === STATE_GOAL) {
          scratch.color.copy(base)
          scratch.emissive.copy(stateColors.reachedEmissive)
        } else {
          scratch.color.copy(base)
          scratch.emissive.setRGB(0, 0, 0)
        }
        get('bodyNear')!.setColorAt(id, scratch.color)
        get('bodyFar')!.setColorAt(id, scratch.color)
        const e = id * 3
        for (const a of [geo.attrs.bodyEmissive, geo.attrs.farEmissive]) {
          a.array[e] = scratch.emissive.r
          a.array[e + 1] = scratch.emissive.g
          a.array[e + 2] = scratch.emissive.b
        }
      }

      if (id === followTarget) {
        hasHighlight = true
        highlightX = pose.x
        highlightY = pose.y
      }
    }

    if (!paused) spin.current -= (rollSpeed * delta) / WHEEL_RADIUS

    // 使っていない段は描かない（三角形も数えさせない）。台数の上限まで回さず、使っている最後の車までにする
    for (const p of VEHICLE_PARTS) {
      const mesh = get(p.key)!
      const last = p.tier === 'near' ? maxNear : p.tier === 'far' ? maxFar : maxAny
      mesh.count = (last + 1) * p.per
      mesh.visible = last >= 0
      mesh.instanceMatrix.needsUpdate = true
    }
    for (const p of VEHICLE_PARTS) {
      if (p.when === 'practical' && !practical) get(p.key)!.visible = false
      if (p.when === 'headlights' && !headOn) get(p.key)!.visible = false
    }

    const a = geo.attrs
    if (lightDirty) a.lampEmissive.needsUpdate = true
    if (indicatorDirty) a.indicatorEmissive.needsUpdate = true
    if (doorDirty) {
      a.doorBody.needsUpdate = true
      a.doorTrim.needsUpdate = true
      a.doorChrome.needsUpdate = true
      a.doorGlass.needsUpdate = true
      a.doorInterior.needsUpdate = true
    }
    if (signDirty) {
      a.signRows.needsUpdate = true
      a.signGlow.needsUpdate = true
    }
    if (colorDirty) {
      for (const key of ['bodyNear', 'bodyFar'] as const) {
        const c = get(key)!.instanceColor
        if (c) c.needsUpdate = true
      }
      a.bodyEmissive.needsUpdate = true
      a.farEmissive.needsUpdate = true
    }

    const ring = ringRef.current
    if (ring) {
      ring.visible = hasHighlight
      if (hasHighlight) {
        ring.position.set(highlightX, 0.06, -highlightY)
        ring.scale.setScalar(1 + 0.05 * Math.sin(nowMs * 0.004))
      }
    }
    const pin = pinRef.current
    if (pin) {
      pin.visible = hasHighlight
      if (hasHighlight) {
        const bob = 0.18 * Math.sin(nowMs * 0.0026)
        pin.position.set(highlightX, PIN_TIP_Y + bob, -highlightY)
        const target = resources.baseColors[followTarget]
        if (target) mats.pin.color.copy(target)
      }
    }
  }, MOTION_PRIORITY)

  return (
    <group>
      {VEHICLE_PARTS.map((p) => (
        <instancedMesh
          key={`${p.key}-${count}`}
          ref={(mesh: THREE.InstancedMesh | null) => {
            if (mesh) meshes.current[p.key] = mesh
            else delete meshes.current[p.key]
          }}
          args={[resources.geometries[p.key], resources.materials[p.material], count * p.per]}
          castShadow={castShadow && p.shadow === true}
          renderOrder={p.renderOrder ?? 0}
          frustumCulled={false}
        />
      ))}

      <mesh ref={ringRef} geometry={ringGeom} material={resources.materials.ring} visible={false} />

      <group ref={pinRef} visible={false}>
        <mesh geometry={pinConeGeom} material={resources.materials.pin} renderOrder={999} />
        <mesh geometry={pinBallGeom} material={resources.materials.pin} renderOrder={999} />
      </group>
    </group>
  )
}
