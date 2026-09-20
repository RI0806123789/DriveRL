/** タクシーの車載カメラ映像をスマホ画面へ転送する。**Canvas の中に置くこと。** */

import { useEffect, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

import { frameBuffer } from '../store/frameBuffer'
import { useSimStore } from '../store/simStore'
import { taxiCamera } from '../store/taxiCamera'
import { DET_LANE, type Detection } from '../types/protocol'
import { DRIVER_FOV_DEG, driverEye, driverLookAt } from './cameraMath'
import { detectionColor, detectionLabel } from './detectionLabels'
import { projectBox } from './detectionProjection'
import { computeAlpha, createPose, sampleVehicle } from './interpolation'
import type { VehiclePose } from './interpolation'

/** 転送するフレームレート。GPU から読み戻すので 60 では回さない */
export const FEED_FPS = 15

/** アンチエイリアスの標本数。メインの `antialias: true` と見え方を揃える */
const FEED_SAMPLES = 4

const LABEL_FLIP_THRESHOLD = 0.08
/** ラベルの文字の大きさ。表示の実寸に合わせて決める（高解像度で豆粒にしない） */
const LABEL_PX_AT_360 = 11

interface Rig {
  target: THREE.WebGLRenderTarget
  camera: THREE.PerspectiveCamera
  pose: VehiclePose
  pixels: Uint8Array
  image: ImageData
  width: number
  height: number
}

function makeRig(width: number, height: number): Rig {
  // ★ 既定のレンダーターゲットは MSAA 無し・linear のままなので、
  //   メインの画（antialias: true / sRGB 出力）と揃える。揃えないと
  //   ギザギザのまま、色も暗く沈んで「荒い映像」になる
  const target = new THREE.WebGLRenderTarget(width, height, { samples: FEED_SAMPLES })
  target.texture.colorSpace = THREE.SRGBColorSpace
  target.texture.minFilter = THREE.LinearFilter
  target.texture.magFilter = THREE.LinearFilter
  return {
    target,
    camera: new THREE.PerspectiveCamera(DRIVER_FOV_DEG, width / height, 0.15, 8000),
    pose: createPose(),
    pixels: new Uint8Array(width * height * 4),
    image: new ImageData(width, height),
    width,
    height,
  }
}

export function TaxiCameraFeed() {
  const vehicleId = useSimStore((s) => s.taxi.vehicleId)
  const rig = useRef<Rig | null>(null)
  const since = useRef(0)

  useEffect(
    () => () => {
      rig.current?.target.dispose()
      rig.current = null
      taxiCamera.live = false
    },
    [],
  )

  useFrame(({ gl, scene }, delta) => {
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
    if (!r || r.width !== width || r.height !== height) {
      r?.target.dispose()
      r = makeRig(width, height)
      rig.current = r
    }
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width
      canvas.height = height
    }

    const paused = useSimStore.getState().status.renderPaused
    if (!sampleVehicle(vehicleId, computeAlpha(performance.now(), paused), r.pose)) {
      taxiCamera.live = false
      return
    }

    const eye = driverEye(r.pose.x, r.pose.y, r.pose.heading)
    const look = driverLookAt(r.pose.x, r.pose.y, r.pose.heading)
    r.camera.position.set(eye.x, eye.y, eye.z)
    r.camera.lookAt(look.x, look.y, look.z)

    // ★ オフスクリーンへ描く。画面の一部を借りて描くと、そこを元の視点で
    //   描き直すためにシーンをもう一度走査することになる
    const prev = gl.getRenderTarget()
    gl.setRenderTarget(r.target)
    gl.render(scene, r.camera)
    gl.setRenderTarget(prev)
    gl.readRenderTargetPixels(r.target, 0, 0, width, height, r.pixels)

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

/**
 * 認識結果の枠を重ねる。**投影は `DetectionOverlay` と同じ `projectBox()`** で、
 * 擬似カメラの正規化座標を表示面へ写す（カメラの位置には依らない）。
 */
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
