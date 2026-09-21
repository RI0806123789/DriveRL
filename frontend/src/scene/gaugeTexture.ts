/** メーターの文字盤（速度計とパワーメーター）を 1 枚のアトラスへ描く。 */

import * as THREE from 'three'

import {
  GAUGE_KINDS,
  GAUGE_SPEED_MAX_KMH,
  NEEDLE_START,
  NEEDLE_SWEEP,
} from './vehicleGeometry'

/** 1 段ぶんの解像度 [px]。文字盤は円なので正方形で描く */
const CELL = 256

/** 文字盤の地色と、目盛り・文字の色 */
const FACE_BG = '#0b0d10'
const TICK_MAJOR = '#e8ecf2'
const TICK_MINOR = '#7b828d'
const LABEL_INK = '#cfd6df'
/** 回生側（CHARGE）と出力側（POWER）の帯 */
const CHARGE_INK = '#3ad2a0'
const POWER_INK = '#ff9c3d'

/**
 * 割合 0..1 を文字盤上の角度 [rad] へ写す。
 * **針（`needleAngle`）と同じ式でなければ、目盛りと針がずれる。**
 */
function faceAngle(ratio: number): number {
  return NEEDLE_START + ratio * NEEDLE_SWEEP
}

/**
 * 文字盤の角度を、キャンバス上の向きへ直す。
 * 針先は `(0, L·cosθ, L·sinθ)` で、**運転者から見て +Z が右・+Y が上**。
 * 文字盤の UV は u が +Z・v が +Y に増えるので、キャンバスは鏡にならない。
 * キャンバスの y は下向きなので、θ から 90 度戻すだけで針と重なる。
 * ★ **ここで符号を反転すると、目盛りだけが左右逆**に焼かれて針と合わなくなる。
 */
function canvasAngle(ratio: number): number {
  return faceAngle(ratio) - Math.PI / 2
}

function drawTick(
  ctx: CanvasRenderingContext2D,
  ratio: number,
  inner: number,
  outer: number,
  width: number,
  color: string,
): void {
  const a = canvasAngle(ratio)
  const cx = CELL / 2
  const cy = CELL / 2
  ctx.strokeStyle = color
  ctx.lineWidth = width
  ctx.lineCap = 'round'
  ctx.beginPath()
  ctx.moveTo(cx + Math.cos(a) * inner, cy + Math.sin(a) * inner)
  ctx.lineTo(cx + Math.cos(a) * outer, cy + Math.sin(a) * outer)
  ctx.stroke()
}

/** 文字盤の共通部分（地色と外周のリング）。 */
function drawFace(ctx: CanvasRenderingContext2D): void {
  ctx.fillStyle = FACE_BG
  ctx.fillRect(0, 0, CELL, CELL)
  ctx.strokeStyle = '#2a2f37'
  ctx.lineWidth = 6
  ctx.beginPath()
  ctx.arc(CELL / 2, CELL / 2, CELL * 0.45, 0, Math.PI * 2)
  ctx.stroke()
}

/** 速度計。0 から `GAUGE_SPEED_MAX_KMH` まで 20km/h 刻みで数字を入れる。 */
function drawSpeedFace(ctx: CanvasRenderingContext2D): void {
  drawFace(ctx)
  const step = 20
  const majors = Math.round(GAUGE_SPEED_MAX_KMH / step)

  for (let i = 0; i <= majors * 2; i++) {
    const kmh = (i * step) / 2
    const ratio = kmh / GAUGE_SPEED_MAX_KMH
    const major = i % 2 === 0
    drawTick(
      ctx,
      ratio,
      CELL * (major ? 0.33 : 0.37),
      CELL * 0.43,
      major ? 5 : 2.5,
      major ? TICK_MAJOR : TICK_MINOR,
    )
  }

  ctx.fillStyle = LABEL_INK
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.font = `700 ${Math.round(CELL * 0.085)}px "Helvetica Neue", Arial, sans-serif`
  for (let i = 0; i <= majors; i++) {
    const kmh = i * step
    const a = canvasAngle(kmh / GAUGE_SPEED_MAX_KMH)
    const r = CELL * 0.255
    ctx.fillText(String(kmh), CELL / 2 + Math.cos(a) * r, CELL / 2 + Math.sin(a) * r)
  }

  ctx.fillStyle = TICK_MINOR
  ctx.font = `500 ${Math.round(CELL * 0.07)}px "Helvetica Neue", Arial, sans-serif`
  ctx.fillText('km/h', CELL / 2, CELL * 0.72)
}

/**
 * パワーメーター（EV）。中央が 0 で、左が回生（CHARGE）、右が出力（POWER）。
 * **数字は入れない**（実車の同種のメーターと同じ）。
 */
function drawPowerFace(ctx: CanvasRenderingContext2D): void {
  drawFace(ctx)

  // 回生側・出力側の帯
  const cx = CELL / 2
  const cy = CELL / 2
  for (const [from, to, color] of [
    [0, 0.5, CHARGE_INK],
    [0.5, 1, POWER_INK],
  ] as Array<[number, number, string]>) {
    ctx.strokeStyle = color
    ctx.lineWidth = 7
    ctx.beginPath()
    ctx.arc(cx, cy, CELL * 0.4, canvasAngle(from), canvasAngle(to), canvasAngle(to) < canvasAngle(from))
    ctx.stroke()
  }

  for (let i = 0; i <= 10; i++) {
    const ratio = i / 10
    const major = i % 5 === 0
    drawTick(
      ctx,
      ratio,
      CELL * (major ? 0.31 : 0.35),
      CELL * 0.375,
      major ? 5 : 2.5,
      major ? TICK_MAJOR : TICK_MINOR,
    )
  }

  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.font = `700 ${Math.round(CELL * 0.062)}px "Helvetica Neue", Arial, sans-serif`
  ctx.fillStyle = CHARGE_INK
  ctx.fillText('CHARGE', CELL * 0.29, CELL * 0.6)
  ctx.fillStyle = POWER_INK
  ctx.fillText('POWER', CELL * 0.71, CELL * 0.6)

  ctx.fillStyle = LABEL_INK
  ctx.font = `700 ${Math.round(CELL * 0.08)}px "Helvetica Neue", Arial, sans-serif`
  ctx.fillText('0', CELL / 2, CELL * 0.19)
}

/**
 * 文字盤のアトラスを作る。段の並びは `GAUGE_KINDS` と同じ順。
 * **呼んだ側が `dispose()` すること。**
 */
export function createGaugeAtlas(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = CELL
  canvas.height = CELL * GAUGE_KINDS.length
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('メーターの文字盤を描けませんでした')

  GAUGE_KINDS.forEach((kind, row) => {
    ctx.save()
    ctx.translate(0, row * CELL)
    if (kind === 'speed') drawSpeedFace(ctx)
    else drawPowerFace(ctx)
    ctx.restore()
  })

  const texture = new THREE.CanvasTexture(canvas)
  texture.colorSpace = THREE.SRGBColorSpace
  texture.anisotropy = 4
  texture.needsUpdate = true
  return texture
}
