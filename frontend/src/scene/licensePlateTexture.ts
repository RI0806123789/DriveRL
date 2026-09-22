/** ナンバープレートのテクスチャ（8 台分を 1 枚に並べたアトラス）を描く。 */

import * as THREE from 'three'

import {
  CELL_H,
  CELL_W,
  PLATE_BG,
  PLATE_INK,
  plateTextFor,
} from './licensePlate'

/** 縁の太さ。実物は 4mm ほどの縁取りが入る */
const BORDER = 10
/** ボルト穴の半径 */
const BOLT_R = 7

/** 一連指定番号を描く。 */
function drawSerial(ctx: CanvasRenderingContext2D, serial: string, top: number): void {
  const dots = serial.length - serial.replace(/・/g, '').length
  const rest = serial.slice(dots)
  const digitSize = Math.round(CELL_H * 0.46)
  const dotSize = Math.round(CELL_H * 0.2)
  const baseline = top + CELL_H * 0.67

  ctx.textAlign = 'right'
  ctx.font = `700 ${digitSize}px "Noto Sans JP", "Yu Gothic", sans-serif`
  const right = CELL_W * 0.9
  ctx.fillText(rest, right, baseline)
  const restWidth = ctx.measureText(rest).width

  ctx.font = `700 ${dotSize}px "Noto Sans JP", "Yu Gothic", sans-serif`
  const dotStep = dotSize * 0.62
  for (let i = 0; i < dots; i++) {
    ctx.fillText('・', right - restWidth - dotStep * i, baseline)
  }
  ctx.textAlign = 'center'
}

function drawPlate(
  ctx: CanvasRenderingContext2D,
  top: number,
  vehicleId: number,
  presetId: string | null | undefined,
): void {
  const text = plateTextFor(vehicleId, presetId)

  ctx.fillStyle = PLATE_BG
  ctx.fillRect(0, top, CELL_W, CELL_H)

  ctx.strokeStyle = PLATE_INK
  ctx.lineWidth = BORDER
  ctx.strokeRect(BORDER / 2, top + BORDER / 2, CELL_W - BORDER, CELL_H - BORDER)

  ctx.fillStyle = PLATE_INK
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'

  // 上段: 地名と分類番号。地名は左寄り、分類番号は右寄り
  ctx.font = `600 ${Math.round(CELL_H * 0.26)}px "Noto Sans JP", "Yu Gothic", sans-serif`
  ctx.fillText(text.region, CELL_W * 0.32, top + CELL_H * 0.26)
  ctx.font = `700 ${Math.round(CELL_H * 0.3)}px "Noto Sans JP", "Yu Gothic", sans-serif`
  ctx.fillText(text.classNumber, CELL_W * 0.68, top + CELL_H * 0.25)

  // 下段: ひらがなは小さく左、一連指定番号は大きく右
  ctx.font = `600 ${Math.round(CELL_H * 0.24)}px "Noto Sans JP", "Yu Gothic", sans-serif`
  ctx.fillText(text.kana, CELL_W * 0.13, top + CELL_H * 0.68)
  drawSerial(ctx, text.serial, top)

  // 封印（左上）と取り付けボルト（右上）
  ctx.fillStyle = '#c8c8c0'
  for (const x of [CELL_W * 0.22, CELL_W * 0.78]) {
    ctx.beginPath()
    ctx.arc(x, top + CELL_H * 0.12, BOLT_R, 0, Math.PI * 2)
    ctx.fill()
  }
}

/** 車両 `count` 台ぶんのプレートを縦に並べたテクスチャを作る。 */
export function createPlateAtlas(
  count: number,
  presetId: string | null | undefined,
): THREE.CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = CELL_W
  canvas.height = CELL_H * Math.max(1, count)
  const ctx = canvas.getContext('2d')
  if (ctx) {
    for (let i = 0; i < count; i++) drawPlate(ctx, i * CELL_H, i, presetId)
  }
  const texture = new THREE.CanvasTexture(canvas)
  texture.colorSpace = THREE.SRGBColorSpace
  texture.anisotropy = 4
  texture.needsUpdate = true
  return texture
}
