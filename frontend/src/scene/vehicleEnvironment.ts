/** 車体に映り込ませる環境マップ。**外部の写真（HDRI）は使わず、配色の空の色からコードで作る。** */

import * as THREE from 'three'
import type { ScenePalette } from './palette'

const VERTEX = /* glsl */ `
  varying vec3 vDir;
  void main() {
    vDir = normalize(position);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`

/** 空（天頂・地平線）・地面・街並みの帯・太陽のにじみ。 */
const FRAGMENT = /* glsl */ `
  varying vec3 vDir;
  uniform vec3 uZenith;
  uniform vec3 uHorizon;
  uniform vec3 uGround;
  uniform vec3 uCity;
  uniform vec3 uSun;
  uniform vec3 uSunDir;
  uniform float uSunStrength;

  float hash(float n) { return fract(sin(n * 12.9898) * 43758.5453); }

  void main() {
    vec3 d = normalize(vDir);
    float y = d.y;
    vec3 sky = mix(uHorizon, uZenith, pow(clamp(y, 0.0, 1.0), 0.55));
    vec3 ground = mix(uHorizon * 0.55 + uGround * 0.45, uGround, pow(clamp(-y, 0.0, 1.0), 0.35));
    vec3 col = y >= 0.0 ? sky : ground;

    // 地平線の少し上に、建物の並びらしい凸凹の帯を置く（車体の側面に街が映る）
    float az = atan(d.z, d.x);
    float slot = floor((az + 3.14159265) / 6.2831853 * 48.0);
    float h = 0.03 + 0.16 * hash(slot) * hash(slot + 17.0);
    if (y >= 0.0 && y < h) col = mix(col, uCity, 0.85);

    float sun = max(dot(d, normalize(uSunDir)), 0.0);
    col += uSun * (pow(sun, 380.0) * 6.0 + pow(sun, 12.0) * 0.35) * uSunStrength;
    gl_FragColor = vec4(col, 1.0);
  }
`

/** 配色から環境マップを作る。PMREM へ焼いたテクスチャと、捨てるための render target を返す */
export function createVehicleEnvironment(
  renderer: THREE.WebGLRenderer,
  palette: ScenePalette,
  night: boolean,
): { texture: THREE.Texture; dispose: () => void } {
  const scene = new THREE.Scene()
  const zenith = new THREE.Color(palette.hemiSky)
  const horizon = new THREE.Color(palette.sky)
  if (night) {
    zenith.multiplyScalar(0.18)
    horizon.lerp(new THREE.Color(palette.buildingHigh), 0.35)
  } else {
    horizon.lerp(new THREE.Color('#ffffff'), 0.25)
  }
  const material = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    uniforms: {
      uZenith: { value: zenith },
      uHorizon: { value: horizon },
      uGround: { value: new THREE.Color(palette.roadSurface).multiplyScalar(night ? 0.5 : 0.8) },
      uCity: { value: new THREE.Color(night ? palette.buildingLow : palette.buildingHigh).multiplyScalar(night ? 0.7 : 0.9) },
      uSun: { value: new THREE.Color(palette.sun) },
      uSunDir: { value: new THREE.Vector3(0.6, 1.1, 0.45) },
      uSunStrength: { value: night ? 0.0 : 1.0 },
    },
    vertexShader: VERTEX,
    fragmentShader: FRAGMENT,
  })
  const sphere = new THREE.Mesh(new THREE.SphereGeometry(10, 48, 24), material)
  scene.add(sphere)
  const pmrem = new THREE.PMREMGenerator(renderer)
  const target = pmrem.fromScene(scene, 0.02)
  pmrem.dispose()
  sphere.geometry.dispose()
  material.dispose()
  return {
    texture: target.texture,
    dispose: () => target.dispose(),
  }
}
