/** 雨。カメラの直前に貼った 1 枚の板へ、シェーダーで筋を降らせる。 */

import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { frameBuffer } from '../store/frameBuffer'
import { usePalette } from './usePalette'
import { weatherLook } from './weatherView'

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
  uniform vec3 uColor;

  float hash(float n) { return fract(sin(n * 91.3458) * 47453.5453); }

  /** 1 層ぶんの雨。列ごとに速度と位相をずらす */
  float sheet(vec2 uv, float columns, float speed, float seed) {
    vec2 p = vec2(uv.x * columns * uAspect, uv.y * 2.0);
    float col = floor(p.x);
    float lane = hash(col + seed);
    float fall = fract(p.y - uTime * speed * (0.7 + lane * 0.8) + lane);
    float across = abs(fract(p.x) - 0.5);

    float line = smoothstep(0.16, 0.0, across);
    float drop = smoothstep(0.0, 0.05, fall) * smoothstep(0.34, 0.05, fall);
    return line * drop * (0.55 + lane * 0.45);
  }

  void main() {
    float a = sheet(vUv, 26.0, 1.5, 1.0) * 0.75;
    a += sheet(vUv, 16.0, 1.1, 7.0) * 0.55;
    gl_FragColor = vec4(uColor, a * uOpacity);
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

    const look = weatherLook(frameBuffer.weather, palette.fogNear, palette.fogFar)
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
