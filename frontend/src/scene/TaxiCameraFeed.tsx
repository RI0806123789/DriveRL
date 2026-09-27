/** タクシーの車載カメラ映像をスマホ画面へ転送する。**Canvas の中に置くこと。** */

import { useEffect, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

import { frameBuffer } from '../store/frameBuffer'
import { useSimStore } from '../store/simStore'
import { taxiCamera } from '../store/taxiCamera'
import { DET_LANE, type Detection } from '../types/protocol'
import { DRIVER_EYE_LOCAL, DRIVER_FOV_DEG, driverEye, driverLookAt, vehicleLocalToThree } from './cameraMath'
import { tiltLocalPoint, tiltOf } from './vehicleMotion'
import { detectionColor, detectionLabel } from './detectionLabels'
import { projectBox } from './detectionProjection'
import { computeAlpha, createPose, sampleVehicle } from './interpolation'
import type { VehiclePose } from './interpolation'
import { hiddenFromMirrors } from './mirrorHidden'

/** 転送するフレームレート。GPU から読み戻すので 60 では回さない */
export const FEED_FPS = 15

/** アンチエイリアスの標本数。メインの `antialias: true` と見え方を揃える */
const FEED_SAMPLES = 4

const LABEL_FLIP_THRESHOLD = 0.08
/** ラベルの文字の大きさ。表示の実寸に合わせて決める（高解像度で豆粒にしない） */
const LABEL_PX_AT_360 = 11

interface Rig {
  /** 場面をそのまま描く先（線形・半精度。トーンマッピングはまだ掛かっていない） */
  scene: THREE.WebGLRenderTarget
  /** トーンマッピングと sRGB への変換を済ませた、読み戻す先 */
  output: THREE.WebGLRenderTarget
  camera: THREE.PerspectiveCamera
  post: THREE.Mesh<THREE.PlaneGeometry, THREE.ShaderMaterial>
  /** ポストの板だけを置いた場面（render() は場面ごとに背景と霧を見る） */
  postScene: THREE.Scene
  pose: VehiclePose
  pixels: Uint8Array
  image: ImageData
  width: number
  height: number
  toneMapping: THREE.ToneMapping
}

/** three の関数名。レンダーターゲットへ描くときは three が掛けないので、ポストの板で掛ける */
const TONE_FUNCTION: Partial<Record<THREE.ToneMapping, string>> = {
  [THREE.LinearToneMapping]: 'LinearToneMapping',
  [THREE.ReinhardToneMapping]: 'ReinhardToneMapping',
  [THREE.CineonToneMapping]: 'CineonToneMapping',
  [THREE.ACESFilmicToneMapping]: 'ACESFilmicToneMapping',
  [THREE.AgXToneMapping]: 'AgXToneMapping',
  [THREE.NeutralToneMapping]: 'NeutralToneMapping',
}

const POST_ORTHO = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)

function makePost(map: THREE.Texture, toneMapping: THREE.ToneMapping): Rig['post'] {
  const fn = TONE_FUNCTION[toneMapping]
  const material = new THREE.ShaderMaterial({
    uniforms: { map: { value: map } },
    defines: fn ? { FEED_TONE: fn } : {},
    vertexShader: /* glsl */ `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = vec4(position.xy, 0.0, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      uniform sampler2D map;
      varying vec2 vUv;
      #include <tonemapping_pars_fragment>
      void main() {
        vec3 color = texture2D(map, vUv).rgb;
        #ifdef FEED_TONE
          color = FEED_TONE(color);
        #endif
        gl_FragColor = vec4(color, 1.0);
        #include <colorspace_fragment>
      }
    `,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
  })
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material)
  mesh.frustumCulled = false
  return mesh
}

function makeRig(width: number, height: number, toneMapping: THREE.ToneMapping): Rig {
  const scene = new THREE.WebGLRenderTarget(width, height, {
    samples: FEED_SAMPLES,
    type: THREE.HalfFloatType,
  })
  scene.texture.colorSpace = THREE.LinearSRGBColorSpace
  const output = new THREE.WebGLRenderTarget(width, height)
  output.texture.colorSpace = THREE.SRGBColorSpace
  const post = makePost(scene.texture, toneMapping)
  const postScene = new THREE.Scene()
  postScene.add(post)
  return {
    scene,
    output,
    camera: new THREE.PerspectiveCamera(DRIVER_FOV_DEG, width / height, 0.15, 8000),
    post,
    postScene,
    pose: createPose(),
    pixels: new Uint8Array(width * height * 4),
    image: new ImageData(width, height),
    width,
    height,
    toneMapping,
  }
}

function disposeRig(r: Rig): void {
  r.scene.dispose()
  r.output.dispose()
  r.post.geometry.dispose()
  r.post.material.dispose()
}

export function TaxiCameraFeed() {
  const vehicleId = useSimStore((s) => s.taxi.vehicleId)
  const rig = useRef<Rig | null>(null)
  const since = useRef(0)
  const hidden = useRef<THREE.Object3D[]>([])

  useEffect(
    () => () => {
      if (rig.current) disposeRig(rig.current)
      rig.current = null
      taxiCamera.live = false
    },
    [],
  )

  useFrame(({ gl, scene, clock }, delta) => {
    const canvas = taxiCamera.canvas
    if (!canvas || vehicleId < 0) {
      taxiCamera.live = false
      return
    }

    since.current += delta
    if (since.current < 1 / FEED_FPS) return
    since.current = 0

    const width = taxiCamera.width
    const height = taxiCamera.height
    let r = rig.current
    if (!r || r.width !== width || r.height !== height || r.toneMapping !== gl.toneMapping) {
      if (r) disposeRig(r)
      r = makeRig(width, height, gl.toneMapping)
      rig.current = r
    }
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width
      canvas.height = height
    }

    const paused = useSimStore.getState().status.renderPaused
    if (!sampleVehicle(vehicleId, computeAlpha(clock.oldTime, paused), r.pose)) {
      taxiCamera.live = false
      return
    }

    // 運転席カメラと同じく、目は車体と一緒に傾け、視線は水平に保つ
    const level = driverEye(r.pose.x, r.pose.y, r.pose.heading)
    const tilt = tiltOf(vehicleId)
    const eye = vehicleLocalToThree(
      tiltLocalPoint(DRIVER_EYE_LOCAL, tilt.pitch, tilt.roll),
      r.pose.x,
      r.pose.y,
      r.pose.heading,
    )
    const look = driverLookAt(r.pose.x, r.pose.y, r.pose.heading)
    r.camera.position.set(eye.x, eye.y, eye.z)
    r.camera.lookAt(look.x + eye.x - level.x, look.y + eye.y - level.y, look.z + eye.z - level.z)
    r.camera.updateMatrixWorld()

    // オフスクリーンへ描く。画面の一部を借りて描くと、そこを元の視点で
    //   描き直すためにシーンをもう一度走査することになる。
    //   影と行列は画面の描画で作ったものを使い回し（鏡と同じ）、画面のカメラの前に貼った板は隠す
    const prev = gl.getRenderTarget()
    const shadowAuto = gl.shadowMap.autoUpdate
    const sceneAuto = scene.matrixWorldAutoUpdate
    gl.shadowMap.autoUpdate = false
    scene.matrixWorldAutoUpdate = false
    const off = hidden.current
    off.length = 0
    for (const o of hiddenFromMirrors) {
      if (!o.visible) continue
      o.visible = false
      off.push(o)
    }
    gl.setRenderTarget(r.scene)
    gl.render(scene, r.camera)
    for (const o of off) o.visible = true
    scene.matrixWorldAutoUpdate = sceneAuto
    gl.shadowMap.autoUpdate = shadowAuto
    // three はレンダーターゲットへ描くときトーンマッピングを掛けないので、画面と揃えるためにここで掛ける
    gl.setRenderTarget(r.output)
    gl.render(r.postScene, POST_ORTHO)
    gl.setRenderTarget(prev)
    gl.readRenderTargetPixels(r.output, 0, 0, width, height, r.pixels)

    flipInto(r.image, r.pixels, width, height)
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.putImageData(r.image, 0, 0)
    drawDetections(ctx, vehicleId, width, height)
    taxiCamera.live = true
  })

  return null
}

/** WebGL の読み出しは下から上なので、行ごと入れ替えて `ImageData` へ移す。 */
function flipInto(
  image: ImageData,
  pixels: Uint8Array,
  width: number,
  height: number,
): void {
  const stride = width * 4
  const out = image.data
  for (let row = 0; row < height; row++) {
    const src = (height - 1 - row) * stride
    out.set(pixels.subarray(src, src + stride), row * stride)
  }
}

/** 認識結果の枠を重ねる。 */
function drawDetections(
  ctx: CanvasRenderingContext2D,
  vehicleId: number,
  width: number,
  height: number,
): void {
  const dets: Detection[] | undefined = frameBuffer.curr?.detections?.[String(vehicleId)]
  if (!dets || dets.length === 0) return

  const aspect = width / height
  const scale = height / 360
  const font = Math.max(9, Math.round(LABEL_PX_AT_360 * scale))
  const tagH = Math.round(font * 1.25)

  ctx.save()
  ctx.lineWidth = Math.max(1.25, 1.6 * scale)
  ctx.font = `600 ${font}px ui-sans-serif, system-ui, sans-serif`
  ctx.textBaseline = 'middle'

  for (const det of dets) {
    if (det.cls === DET_LANE) continue
    const box = projectBox(det.box, aspect)
    const w = box.width * width
    const h = box.height * height
    if (w <= 1 || h <= 1) continue
    const x = box.left * width
    const y = box.top * height
    const color = detectionColor(det)

    ctx.strokeStyle = color
    ctx.strokeRect(x, y, w, h)

    const label = detectionLabel(det)
    const textW = ctx.measureText(label).width
    const flip = box.top < LABEL_FLIP_THRESHOLD
    const tagY = flip ? y + 1 : y - tagH - 1
    ctx.fillStyle = color
    ctx.fillRect(x, tagY, textW + font * 0.7, tagH)
    ctx.fillStyle = '#0b0e11'
    ctx.fillText(label, x + font * 0.35, tagY + tagH / 2)
  }
  ctx.restore()
}
