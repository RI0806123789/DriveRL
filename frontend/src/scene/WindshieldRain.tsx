/** フロントガラスの水滴。**運転席から見ている 1 台にだけ**、ガラスの面に貼った板へシェーダーで描く。 */

import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

import { useSimStore } from '../store/simStore'
import { computeAlpha, createPose, sampleVehicle } from './interpolation'
import { usePalette } from './usePalette'
import { composeBodyMatrix, composeVehicleMatrix, createTransformScratch } from './vehicleGeometry'
import { composeTiltMatrix, tiltOf } from './vehicleMotion'
import { displayedWeather } from './weatherView'
import {
  DROP_CELL_M,
  DROP_CHANCE,
  DROP_OPACITY,
  DROP_RADIUS,
  WINDSHIELD_LENGTH,
  WIPER_PIVOTS,
  WIPER_REACH,
  WIPER_SWEEP,
  dropFillSec,
  vehicleWipers,
  windshieldHalfWidth,
  windshieldPoint,
} from './wiper'

/** ガラスの内側へ沈める量 [m]。外側に貼るとガラスと重なって見えなくなる角度がある */
const INSET_M = 0.004

const VERTEX = /* glsl */ `
  attribute vec2 glass;
  varying vec2 vGlass;
  void main() {
    vGlass = glass;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`

/** 粒はセルに 1 つ。拭いてからの時間が、粒ごとの「現れるまでの時間」を超えたら描く */
const FRAGMENT = /* glsl */ `
  varying vec2 vGlass;
  uniform float uOpacity;
  uniform float uWet;
  uniform float uFill;
  uniform float uElapsed;
  uniform float uPeriod;
  uniform float uPrevElapsed;
  uniform float uPrevPeriod;
  uniform float uSweep;
  uniform vec2 uPivot0;
  uniform vec2 uPivot1;
  uniform vec2 uReach;
  uniform float uCell;
  uniform float uChance;
  uniform vec2 uRadius;
  uniform vec3 uLight;
  uniform vec3 uDark;

  const float PI = 3.14159265;
  const float NEVER = 1.0e4;

  float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

  float passAge(float phi) {
    if (phi < 0.0 || phi > uSweep) return NEVER;
    float f = phi / uSweep;
    if (uPeriod > 0.0 && uElapsed >= 0.0) {
      float up = uPeriod / (2.0 * PI) * acos(1.0 - 2.0 * f);
      float down = uPeriod - up;
      if (uElapsed >= down) return uElapsed - down;
      if (uElapsed >= up) return uElapsed - up;
    }
    if (uPrevPeriod > 0.0) {
      float up2 = uPrevPeriod / (2.0 * PI) * acos(1.0 - 2.0 * f);
      return uPrevElapsed - (uPrevPeriod - up2);
    }
    return NEVER;
  }

  float wipeAge(vec2 q, vec2 pivot) {
    vec2 d = q - pivot;
    float r = length(d);
    if (r < uReach.x || r > uReach.y) return NEVER;
    return passAge(atan(d.y, d.x));
  }

  void main() {
    vec2 g = vGlass / uCell;
    vec2 cell = floor(g);
    vec2 f = fract(g);
    float h1 = hash(cell);
    if (h1 > uChance) discard;
    float r = mix(uRadius.x, uRadius.y, hash(cell + 17.1));
    vec2 c = vec2(mix(r, 1.0 - r, hash(cell + 41.7)), mix(r, 1.0 - r, hash(cell + 7.3)));
    vec2 local = (f - c) / r;
    float d = length(local);
    if (d > 1.0) discard;
    vec2 centre = (cell + c) * uCell;
    // vGlass と付け根は (z, s)。ワイパーの角度は +z から +s へ測る（wiper.ts と同じ）
    float age = min(wipeAge(centre, uPivot0), wipeAge(centre, uPivot1));
    float delay = h1 / uChance * uFill;
    if (age < delay) discard;
    // 水の粒らしく：縁は暗く、中は向こうが透け、上側に空が映って小さく光る
    float edge = smoothstep(1.0, 0.82, d);
    float rim = smoothstep(0.5, 1.0, d);
    float lit = clamp(0.55 + 0.45 * local.y, 0.0, 1.0);
    vec3 col = mix(uLight, uDark, rim * 0.8 + (1.0 - lit) * 0.35);
    float spark = smoothstep(0.3, 0.0, length(local - vec2(-0.32, 0.42)));
    col = mix(col, vec3(1.0), spark * 0.85);
    gl_FragColor = vec4(col, edge * uOpacity * uWet * min(1.0, 0.45 + 0.55 * rim + spark));
  }
`

/** ガラスの面に沿った格子（属性 `glass` にガラスの座標 (z, s) を入れる） */
function makeDropletGeometry(): THREE.BufferGeometry {
  const rows = 8
  const cols = 12
  const pos: number[] = []
  const glass: number[] = []
  for (let r = 0; r <= rows; r++) {
    const s = (WINDSHIELD_LENGTH * r) / rows
    const half = windshieldHalfWidth(s)
    for (let c = 0; c <= cols; c++) {
      const z = -half + (2 * half * c) / cols
      const p = windshieldPoint(s, z, -INSET_M)
      pos.push(p[0], p[1], p[2])
      glass.push(z, s)
    }
  }
  const index: number[] = []
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const a = r * (cols + 1) + c
      const b = a + cols + 1
      index.push(a, b, a + 1, a + 1, b, b + 1)
    }
  }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  g.setAttribute('glass', new THREE.Float32BufferAttribute(glass, 2))
  g.setIndex(index)
  return g
}

export function WindshieldRain() {
  const palette = usePalette()
  const meshRef = useRef<THREE.Mesh>(null)
  const pose = useMemo(createPose, [])
  const resources = useMemo(() => {
    const geometry = makeDropletGeometry()
    const material = new THREE.ShaderMaterial({
      vertexShader: VERTEX,
      fragmentShader: FRAGMENT,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      uniforms: {
        uOpacity: { value: DROP_OPACITY },
        uWet: { value: 0 },
        uFill: { value: dropFillSec(1) },
        uElapsed: { value: -1 },
        uPeriod: { value: 0 },
        uPrevElapsed: { value: -1 },
        uPrevPeriod: { value: 0 },
        uSweep: { value: WIPER_SWEEP },
        uPivot0: { value: new THREE.Vector2(WIPER_PIVOTS[0][1], WIPER_PIVOTS[0][0]) },
        uPivot1: { value: new THREE.Vector2(WIPER_PIVOTS[1][1], WIPER_PIVOTS[1][0]) },
        uReach: { value: new THREE.Vector2(WIPER_REACH[0], WIPER_REACH[1]) },
        uCell: { value: DROP_CELL_M },
        uChance: { value: DROP_CHANCE },
        uRadius: { value: new THREE.Vector2(DROP_RADIUS[0], DROP_RADIUS[1]) },
        uLight: { value: new THREE.Color('#dfe8f0') },
        uDark: { value: new THREE.Color('#1c2530') },
      },
    })
    return { geometry, material }
  }, [])

  useEffect(() => {
    const r = resources
    return () => {
      r.geometry.dispose()
      r.material.dispose()
    }
  }, [resources])

  const scratch = useMemo(
    () => ({
      transform: createTransformScratch(),
      base: new THREE.Matrix4(),
      tilt: new THREE.Matrix4(),
      white: new THREE.Color('#ffffff'),
    }),
    [],
  )

  useFrame((state) => {
    const mesh = meshRef.current
    if (!mesh) return
    const store = useSimStore.getState()
    const wet = displayedWeather.wet
    const target =
      store.cameraMode === 'driver'
        ? store.followTarget
        : store.taxiCameraOn && store.mode === 'taxi'
          ? store.taxi.vehicleId
          : -1
    const alpha = computeAlpha(state.clock.oldTime, store.status.renderPaused)
    const ok = target >= 0 && wet > 0.01 && sampleVehicle(target, alpha, pose)
    mesh.visible = ok
    if (!ok) return

    composeVehicleMatrix(scratch.transform, pose.x, pose.y, pose.heading, scratch.base)
    const tilt = tiltOf(target)
    composeTiltMatrix(tilt.pitch, tilt.roll, scratch.tilt)
    composeBodyMatrix(scratch.base, scratch.tilt, mesh.matrix)
    mesh.matrixWorldNeedsUpdate = true

    const u = resources.material.uniforms
    const now = state.clock.oldTime / 1000
    const w = vehicleWipers.states[target]
    const finite = (v: number) => (Number.isFinite(v) ? v : -1)
    u.uElapsed.value = w ? finite(now - w.start) : -1
    u.uPeriod.value = w && Number.isFinite(w.start) ? w.period : 0
    u.uPrevElapsed.value = w ? finite(now - w.prevStart) : -1
    u.uPrevPeriod.value = w && Number.isFinite(w.prevStart) ? w.prevPeriod : 0
    u.uWet.value = Math.min(1, wet * 1.4)
    u.uFill.value = dropFillSec(Math.max(displayedWeather.rain, wet * 0.5))
    u.uLight.value.set(palette.rainDrop).lerp(scratch.white, 0.35)
  })

  return (
    <mesh
      ref={meshRef}
      geometry={resources.geometry}
      material={resources.material}
      matrixAutoUpdate={false}
      frustumCulled={false}
      renderOrder={3}
      visible={false}
    />
  )
}
