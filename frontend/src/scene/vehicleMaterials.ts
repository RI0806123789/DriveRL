/** 車両のマテリアルとシェーダーの差し込み。**マテリアルは作り直さず、色と値だけ差し替えること。** */

import * as THREE from 'three'
import type { ScenePalette } from './palette'
import { VEHICLE_LIGHT_OFF } from './palette'
import { SIGN_ROWS } from './taxiSign'
import { TAXI_DOOR_HINGE } from './vehicleGeometry'

/** 差し込み 1 か所。文字列なら見つけた行の後ろへ足し、関数なら見つけた行を置き換える */
type Splice = readonly [string, string | ((found: string) => string)]

/** シェーダーへの差し込み 1 つ。`key` はプログラムのキャッシュの区別に使う */
interface ShaderPatch {
  readonly key: string
  readonly vertex?: ReadonlyArray<Splice>
  readonly fragment?: ReadonlyArray<Splice>
}

function splice(src: string, [find, add]: Splice): string {
  return src.replace(find, typeof add === 'function' ? add(find) : `${find}\n${add}`)
}

/** 差し込みを順に当てる。`onBeforeCompile` は 1 つしか持てないので、ここでまとめる */
function applyPatches(material: THREE.Material, patches: ReadonlyArray<ShaderPatch>): void {
  const key = patches.map((p) => p.key).join('|')
  material.onBeforeCompile = (shader) => {
    for (const p of patches) {
      for (const s of p.vertex ?? []) shader.vertexShader = splice(shader.vertexShader, s)
      for (const s of p.fragment ?? []) shader.fragmentShader = splice(shader.fragmentShader, s)
    }
  }
  material.customProgramCacheKey = () => key
}

/** インスタンスごとの自己発光（灯火・メーターのウインカー表示・衝突時の点滅） */
const INSTANCE_EMISSIVE: ShaderPatch = {
  key: 'instanceEmissive',
  vertex: [
    ['#include <common>', 'attribute vec3 instanceEmissive;\nvarying vec3 vInstanceEmissive;'],
    ['#include <begin_vertex>', 'vInstanceEmissive = instanceEmissive;'],
  ],
  fragment: [
    ['#include <common>', 'varying vec3 vInstanceEmissive;'],
    ['#include <emissivemap_fragment>', 'totalEmissiveRadiance += vInstanceEmissive;'],
  ],
}

/** 左後席ドアの開閉。`doorPart` の頂点だけを蝶番（鉛直軸）まわりに回す。**法線は `defaultnormal` より前に回す** */
const DOOR: ShaderPatch = {
  key: 'door',
  vertex: [
    ['#include <common>', 'attribute float doorPart;\nattribute float instanceDoor;'],
    [
      '#include <beginnormal_vertex>',
      `float doorA = -instanceDoor * doorPart;
float doorC = cos(doorA);
float doorS = sin(doorA);
objectNormal = vec3(objectNormal.x * doorC + objectNormal.z * doorS, objectNormal.y, -objectNormal.x * doorS + objectNormal.z * doorC);`,
    ],
    [
      '#include <begin_vertex>',
      `vec2 doorRel = transformed.xz - vec2(${TAXI_DOOR_HINGE[0].toFixed(4)}, ${TAXI_DOOR_HINGE[1].toFixed(4)});
transformed.xz = vec2(${TAXI_DOOR_HINGE[0].toFixed(4)}, ${TAXI_DOOR_HINGE[1].toFixed(4)}) + vec2(doorRel.x * doorC + doorRel.y * doorS, -doorRel.x * doorS + doorRel.y * doorC);`,
    ],
  ],
}

/** 内側から見たガラスの明るさ（ガラスの色に掛ける）。光も映り込みも計算せず、色の付いた透ける板として出す */
const GLASS_INSIDE_SHADE = 0.55

/** 窓ガラスを内側（裏面）から見たときは光の計算を省く（外が見えることを優先し、画面を覆う面で重さも抑える） */
const GLASS_INSIDE: ShaderPatch = {
  key: 'glassInside',
  fragment: [
    [
      '#include <clipping_planes_fragment>',
      `if ( ! gl_FrontFacing ) {
  gl_FragColor = linearToOutputTexel( vec4( diffuse * ${GLASS_INSIDE_SHADE.toFixed(3)}, opacity ) );
  return;
}`,
    ],
  ],
}

/** アトラスの「段」を選んで貼る（ナンバープレート・メーター）。段はインスタンスの属性 `name` */
function atlasRows(name: string, rows: number): ShaderPatch {
  const total = Math.max(1, rows).toFixed(1)
  return {
    key: `atlas:${name}:${total}`,
    vertex: [
      ['#include <common>', `attribute float ${name};`],
      [
        '#include <uv_vertex>',
        `#ifdef USE_MAP
vMapUv = vec2( vMapUv.x, ( ${name} + vMapUv.y ) / ${total} );
#endif
#ifdef USE_EMISSIVEMAP
vEmissiveMapUv = vec2( vEmissiveMapUv.x, ( ${name} + vEmissiveMapUv.y ) / ${total} );
#endif`,
      ],
    ],
  }
}

/** タクシーの表示。頂点の `signRow` が負なら状態の段（インスタンス）、光り方は表示灯と表示板で分ける */
const TAXI_SIGN: ShaderPatch = {
  key: `taxiSign:${SIGN_ROWS}`,
  vertex: [
    [
      '#include <common>',
      'attribute float signRow;\nattribute float instanceSignRow;\nattribute vec2 instanceSignGlow;\nvarying float vSignGlow;',
    ],
    [
      '#include <uv_vertex>',
      `float signR = signRow < -0.5 ? instanceSignRow : signRow;
float signV = ${(SIGN_ROWS - 1).toFixed(1)} - signR;
#ifdef USE_MAP
vMapUv = vec2( vMapUv.x, ( signV + vMapUv.y ) / ${SIGN_ROWS.toFixed(1)} );
#endif
#ifdef USE_EMISSIVEMAP
vEmissiveMapUv = vec2( vEmissiveMapUv.x, ( signV + vEmissiveMapUv.y ) / ${SIGN_ROWS.toFixed(1)} );
#endif
vSignGlow = signRow < -0.5 ? instanceSignGlow.y : ( signRow < 1.5 ? instanceSignGlow.x : 0.0 );`,
    ],
  ],
  fragment: [
    ['#include <common>', 'varying float vSignGlow;'],
    ['#include <emissivemap_fragment>', 'totalEmissiveRadiance *= vSignGlow;'],
  ],
}

/** 塗装。クリア層（`clearcoat`）は描画の重さを測ったうえで使う（CLAUDE.md「車両の材質」） */
export const PAINT = { roughness: 0.34, metalness: 0.12, clearcoat: 1.0, clearcoatRoughness: 0.07 } as const
/** 濡れたときの行き先（路面と同じ `displayedWeather.wet` で寄せる） */
export const WET_PAINT = { roughness: 0.16, clearcoatRoughness: 0.02, darken: 0.88 } as const
const TRIM = { roughness: 0.55, wetRoughness: 0.28 } as const
const TYRE = { roughness: 0.92, wetRoughness: 0.55, wetDarken: 0.78 } as const
/** 環境マップの強さ（天候で暗くするときはこれに掛ける） */
const ENV = { paint: 0.85, glass: 1.1, chrome: 1.0, rim: 0.9, trim: 0.35, lamp: 0.8, far: 0.7 } as const

export interface VehicleMaterials {
  paint: THREE.MeshPhysicalMaterial
  farPaint: THREE.MeshStandardMaterial
  trim: THREE.MeshStandardMaterial
  chrome: THREE.MeshStandardMaterial
  glass: THREE.MeshStandardMaterial
  interior: THREE.MeshStandardMaterial
  controls: THREE.MeshStandardMaterial
  gauge: THREE.MeshStandardMaterial
  needle: THREE.MeshStandardMaterial
  indicator: THREE.MeshStandardMaterial
  tyre: THREE.MeshStandardMaterial
  rim: THREE.MeshStandardMaterial
  caliper: THREE.MeshStandardMaterial
  farWheel: THREE.MeshStandardMaterial
  lamp: THREE.MeshStandardMaterial
  plate: THREE.MeshStandardMaterial
  sign: THREE.MeshStandardMaterial
  pool: THREE.MeshBasicMaterial
  ring: THREE.MeshBasicMaterial
  pin: THREE.MeshBasicMaterial
}

/** マテリアルを 1 度だけ作る。配色は `applyVehiclePalette`、天候は `applyVehicleWeather` で差し替える */
export function createVehicleMaterials(palette: ScenePalette, count: number, gaugeRows: number): VehicleMaterials {
  const paint = new THREE.MeshPhysicalMaterial({ color: '#ffffff', ...PAINT })
  applyPatches(paint, [INSTANCE_EMISSIVE, DOOR])
  const farPaint = new THREE.MeshStandardMaterial({
    color: '#ffffff',
    roughness: PAINT.roughness,
    metalness: 0.18,
    vertexColors: true,
  })
  applyPatches(farPaint, [INSTANCE_EMISSIVE])
  const trim = new THREE.MeshStandardMaterial({ color: palette.vehicleTrim, roughness: TRIM.roughness, metalness: 0.05 })
  applyPatches(trim, [DOOR])
  const chrome = new THREE.MeshStandardMaterial({ color: '#d9dde2', roughness: 0.14, metalness: 1.0 })
  applyPatches(chrome, [DOOR])
  // ガラスは**両面**で描く。片面だと、内側から見たとき窓が消えて「屋根が無い車」になる
  const glass = new THREE.MeshStandardMaterial({
    color: palette.vehicleGlass,
    roughness: 0.06,
    metalness: 0.25,
    transparent: true,
    opacity: 0.42,
    side: THREE.DoubleSide,
    depthWrite: false,
  })
  applyPatches(glass, [DOOR, GLASS_INSIDE])
  const interior = new THREE.MeshStandardMaterial({
    color: palette.vehicleInterior,
    roughness: 0.85,
    metalness: 0.05,
    side: THREE.DoubleSide,
    vertexColors: true,
  })
  applyPatches(interior, [DOOR])
  const controls = new THREE.MeshStandardMaterial({
    color: palette.vehicleTrim,
    roughness: 0.6,
    metalness: 0.15,
    side: THREE.DoubleSide,
  })
  // 文字盤は目盛りをテクスチャで焼いてある。色を掛けると目盛りが沈むので白のまま。
  // 発光は emissiveMap（同じテクスチャ）に従わせる（emissive だけだと一様に光って白く飛ぶ）
  const gauge = new THREE.MeshStandardMaterial({
    color: '#ffffff',
    roughness: 0.4,
    metalness: 0.0,
    side: THREE.DoubleSide,
    emissive: new THREE.Color('#ffffff'),
    emissiveIntensity: 0.9,
    toneMapped: false,
  })
  applyPatches(gauge, [atlasRows('instanceGaugeRow', gaugeRows)])
  const needle = new THREE.MeshStandardMaterial({
    color: palette.vehicleNeedle,
    emissive: new THREE.Color(palette.vehicleNeedle),
    roughness: 0.5,
    toneMapped: false,
  })
  // 消灯時は色を沈ませる（明るい緑のまま置くと、点いていないのに点いて見える）
  const indicator = new THREE.MeshStandardMaterial({
    color: new THREE.Color(VEHICLE_LIGHT_OFF).lerp(new THREE.Color(palette.vehicleIndicator), 0.3),
    roughness: 0.5,
    metalness: 0.0,
    side: THREE.DoubleSide,
    toneMapped: false,
  })
  applyPatches(indicator, [INSTANCE_EMISSIVE])
  const tyre = new THREE.MeshStandardMaterial({ color: palette.vehicleWheel, roughness: TYRE.roughness, metalness: 0.0 })
  const rim = new THREE.MeshStandardMaterial({ color: '#c6ccd3', roughness: 0.3, metalness: 0.85, vertexColors: true })
  const caliper = new THREE.MeshStandardMaterial({ color: '#4a5058', roughness: 0.45, metalness: 0.55 })
  const farWheel = new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 0.8, metalness: 0.2, vertexColors: true })
  // 灯体そのものは消灯でも色が分かる程度に塗り、点灯は instanceEmissive で出す
  const lamp = new THREE.MeshStandardMaterial({ color: '#ffffff', roughness: 0.14, metalness: 0.35, toneMapped: false })
  applyPatches(lamp, [INSTANCE_EMISSIVE])
  const plate = new THREE.MeshStandardMaterial({
    roughness: 0.55,
    metalness: 0.05,
    // 夜でも読めるよう、わずかに自己発光させる（反射板の代わり）
    emissive: new THREE.Color('#2a2a26'),
  })
  applyPatches(plate, [atlasRows('instancePlateRow', count)])
  const sign = new THREE.MeshStandardMaterial({
    color: '#ffffff',
    roughness: 0.35,
    metalness: 0.0,
    emissive: new THREE.Color('#ffffff'),
    emissiveIntensity: 1.0,
    toneMapped: false,
  })
  applyPatches(sign, [TAXI_SIGN])
  const pool = new THREE.MeshBasicMaterial({
    color: '#fff0d2',
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    toneMapped: false,
    polygonOffset: true,
    polygonOffsetFactor: -7,
    polygonOffsetUnits: -14,
  })
  const ring = new THREE.MeshBasicMaterial({ color: palette.vehicleHighlight, transparent: true, opacity: 0.85 })
  const pin = new THREE.MeshBasicMaterial({
    color: '#ffffff',
    toneMapped: false,
    depthTest: false,
    depthWrite: false,
    transparent: true,
    opacity: 1,
  })
  return {
    paint,
    farPaint,
    trim,
    chrome,
    glass,
    interior,
    controls,
    gauge,
    needle,
    indicator,
    tyre,
    rim,
    caliper,
    farWheel,
    lamp,
    plate,
    sign,
    pool,
    ring,
    pin,
  }
}

/** 昼夜の配色を差し替える（マテリアルは作り直さない） */
export function applyVehiclePalette(m: VehicleMaterials, palette: ScenePalette): void {
  m.glass.color.set(palette.vehicleGlass)
  m.tyre.color.set(palette.vehicleWheel)
  m.ring.color.set(palette.vehicleHighlight)
  m.interior.color.set(palette.vehicleInterior)
  m.trim.color.set(palette.vehicleTrim)
  m.controls.color.set(palette.vehicleTrim)
  m.needle.color.set(palette.vehicleNeedle)
  m.needle.emissive.set(palette.vehicleNeedle)
  m.indicator.color.set(VEHICLE_LIGHT_OFF).lerp(new THREE.Color(palette.vehicleIndicator), 0.3)
}

/** 環境マップを付け替える（昼夜で作り直したときだけ） */
export function applyVehicleEnvironment(m: VehicleMaterials, env: THREE.Texture | null): void {
  for (const mat of [m.paint, m.farPaint, m.trim, m.chrome, m.glass, m.rim, m.caliper, m.lamp]) {
    if (mat.envMap === env) continue
    mat.envMap = env
    mat.needsUpdate = true
  }
}

/** 雨で濡らし、曇りで映り込みを暗くする。`wet` は `displayedWeather.wet`、`dim` は `weatherLook().dim` */
export function applyVehicleWeather(m: VehicleMaterials, wet: number, dim: number): void {
  const w = Math.min(1, Math.max(0, wet))
  const lerp = (a: number, b: number) => a + (b - a) * w
  m.paint.roughness = lerp(PAINT.roughness, WET_PAINT.roughness)
  m.paint.clearcoatRoughness = lerp(PAINT.clearcoatRoughness, WET_PAINT.clearcoatRoughness)
  m.paint.color.setScalar(lerp(1, WET_PAINT.darken))
  m.farPaint.roughness = lerp(PAINT.roughness, WET_PAINT.roughness)
  m.farPaint.color.setScalar(lerp(1, WET_PAINT.darken))
  m.trim.roughness = lerp(TRIM.roughness, TRIM.wetRoughness)
  m.tyre.roughness = lerp(TYRE.roughness, TYRE.wetRoughness)
  const boost = 1 + 0.35 * w
  m.paint.envMapIntensity = ENV.paint * dim * boost
  m.farPaint.envMapIntensity = ENV.far * dim * boost
  m.trim.envMapIntensity = ENV.trim * dim * boost
  m.glass.envMapIntensity = ENV.glass * dim
  m.chrome.envMapIntensity = ENV.chrome * dim
  m.rim.envMapIntensity = ENV.rim * dim
  m.caliper.envMapIntensity = ENV.rim * dim
  m.lamp.envMapIntensity = ENV.lamp * dim
}

/** タイヤの色は濡れると沈む（配色の色に掛ける係数） */
export function tyreWetDarken(wet: number): number {
  return 1 + (TYRE.wetDarken - 1) * Math.min(1, Math.max(0, wet))
}

export function disposeVehicleMaterials(m: VehicleMaterials): void {
  for (const mat of Object.values(m)) mat.dispose()
}

/** 前照灯が路面に落とす光の形（u が前後、v が左右）。明るさだけを持つ */
export function createHeadlightPoolTexture(): THREE.DataTexture {
  const W = 128
  const H = 64
  const data = new Uint8Array(W * H * 4)
  const smooth = (e0: number, e1: number, x: number) => {
    const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)
  }
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const u = (x + 0.5) / W
      const a = Math.abs(((y + 0.5) / H) * 2 - 1)
      const half = 0.22 + 0.78 * u
      const across = 1 - smooth(half * 0.55, half, a)
      const along = smooth(0, 0.07, u) * Math.pow(1 - smooth(0.18, 1, u), 1.3)
      const lobes = 0.8 + 0.2 * Math.cos((a / half) * Math.PI * 1.6)
      const v = Math.round(255 * Math.min(1, across * along * lobes))
      const o = (y * W + x) * 4
      data[o] = v
      data[o + 1] = v
      data[o + 2] = v
      data[o + 3] = 255
    }
  }
  const tex = new THREE.DataTexture(data, W, H, THREE.RGBAFormat)
  tex.magFilter = THREE.LinearFilter
  tex.minFilter = THREE.LinearFilter
  tex.needsUpdate = true
  return tex
}
