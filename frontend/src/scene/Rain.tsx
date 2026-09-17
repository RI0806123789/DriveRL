/** 雨。カメラの直前に貼った 1 枚の板へ、シェーダーで筋を降らせる。 */

import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { usePalette } from './usePalette'
import { displayedWeather, weatherLook } from './weatherView'

/**
 * ★ 3D 空間に雨粒を撒く形にしないこと。
 *
 * 俯瞰カメラは 300m 上空にいるので、カメラの周りに箱を置いて降らせても
 * 雨粒が小さすぎて 1 画素にも満たない（実際に作って見えなかった）。
 * 箱を俯瞰の高さに合わせて広げると、同じ密度を保つのに本数が 2 桁増える。
 * カメラに貼り付けた板なら、俯瞰でも運転席でも同じように見えて、板は 1 枚で済む。
 */
const VERTEX_SHADER = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`

const FRAGMENT_SHADER = /* glsl */ `
  varying vec2 vUv;
  uniform float uTime;
  uniform float uOpacity;
  uniform float uAspect;
  uniform float uStrength;
  uniform float uSlant;
  uniform vec3 uColor;

  float hash(float n) { return fract(sin(n * 91.3458) * 47453.5453); }

  /**
   * 1 層ぶんの雨。列ごとに速度・長さ・位相をずらす。
   * 手前の層ほど太く長く、ゆっくり流れて見える（視差）。
   */
  float sheet(vec2 uv, float columns, float speed, float seed, float width, float len) {
    vec2 p = uv;
    p.x += p.y * uSlant;
    p *= vec2(columns * uAspect, 2.0);

    float col = floor(p.x);
    float lane = hash(col + seed);
    float fall = fract(p.y - uTime * speed * (0.75 + lane * 0.5) + lane * 7.13);
    float across = abs(fract(p.x) - 0.5);

    float line = smoothstep(width, 0.0, across);
    float head = smoothstep(0.0, len * 0.35, fall);
    float tail = smoothstep(len, len * 0.35, fall);
    return line * head * tail * (0.45 + lane * 0.55);
  }

  void main() {
    // 近い層は必ず出す。強くなるほど奥の層が増えて密になる
    float a = sheet(vUv, 13.0, 0.95, 13.0, 0.15, 0.40) * 0.85;
    a += sheet(vUv, 22.0, 1.35, 7.0, 0.11, 0.27) * 0.62
       * smoothstep(0.18, 0.55, uStrength);
    a += sheet(vUv, 37.0, 1.85, 1.0, 0.07, 0.18) * 0.42
       * smoothstep(0.45, 0.9, uStrength);

    gl_FragColor = vec4(uColor, min(a, 1.0) * uOpacity);
  }
`

export function Rain() {
  const palette = usePalette()
  const mesh = useRef<THREE.Mesh>(null)
  const material = useRef<THREE.ShaderMaterial>(null)

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uOpacity: { value: 0 },
      uAspect: { value: 1 },
      uStrength: { value: 0 },
      uSlant: { value: 0.12 },
      uColor: { value: new THREE.Color(palette.rainDrop) },
    }),
    // 色は useFrame で差し替えるので作り直さない
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  )

  useFrame((state, delta) => {
    const plane = mesh.current
    const mat = material.current
    if (!plane || !mat) return

    const look = weatherLook(displayedWeather, palette.fogNear, palette.fogFar)
    plane.visible = look.dropOpacity > 0
    if (!plane.visible) return

    const camera = state.camera as THREE.PerspectiveCamera
    const dist = camera.near + 0.05
    plane.position.copy(camera.position)
    plane.quaternion.copy(camera.quaternion)
    plane.translateZ(-dist)

    const height = 2 * dist * Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5))
    plane.scale.set(height * camera.aspect, height, 1)

    mat.uniforms.uTime.value += delta
    mat.uniforms.uOpacity.value = look.dropOpacity
    mat.uniforms.uAspect.value = camera.aspect
    mat.uniforms.uStrength.value = displayedWeather.rain
    // 風向きをゆっくり振る。真下に降り続けると単調に見える
    mat.uniforms.uSlant.value =
      0.1 + Math.sin(mat.uniforms.uTime.value * 0.13) * 0.07
    mat.uniforms.uColor.value.set(palette.rainDrop)
  })

  return (
    <mesh ref={mesh} frustumCulled={false} renderOrder={999}>
      <planeGeometry args={[1, 1]} />
      <shaderMaterial
        ref={material}
        uniforms={uniforms}
        vertexShader={VERTEX_SHADER}
        fragmentShader={FRAGMENT_SHADER}
        transparent
        depthTest={false}
        depthWrite={false}
        fog={false}
      />
    </mesh>
  )
}
