import * as THREE from 'three'
import type { MapSign, SignDirection, SignKind } from '../types/protocol'
import { SIGN_BOARD_THICKNESS, SIGN_FACE_OFFSET, SIGN_RADIUS } from './signGeometry.ts'
import { TRAFFIC_SIGN_COLORS } from './palette.ts'

export const TRAFFIC_SIGN_LABELS: Record<string, string> = {
  stop: '一時停止', crosswalk: '横断歩道', one_way: '一方通行',
  mandatory_direction: '指定方向外進行禁止', no_parking: '駐車禁止', no_stopping: '駐停車禁止',
}

export function trafficSignKey(sign: MapSign): string {
  return `${sign.kind}:${sign.kind === 'mandatory_direction' ? sign.direction ?? 'straight' : ''}`
}

export function trafficSignOutline(kind: SignKind): Array<[number, number]> | null {
  if (kind === 'stop') return [[-SIGN_RADIUS, SIGN_RADIUS], [SIGN_RADIUS, SIGN_RADIUS], [0, -SIGN_RADIUS]]
  const halfHeight = kind === 'one_way' ? SIGN_RADIUS / 2 : SIGN_RADIUS
  if (kind === 'crosswalk' || kind === 'one_way') {
    return [[-SIGN_RADIUS, -halfHeight], [SIGN_RADIUS, -halfHeight], [SIGN_RADIUS, halfHeight], [-SIGN_RADIUS, halfHeight]]
  }
  return null
}

export function createTrafficSignGeometry(kind: SignKind, face: boolean): THREE.BufferGeometry {
  const outline = trafficSignOutline(kind)
  const shape = new THREE.Shape()
  if (outline) {
    shape.moveTo(...outline[0])
    for (const point of outline.slice(1)) shape.lineTo(...point)
    shape.closePath()
  } else {
    shape.absarc(0, 0, SIGN_RADIUS, 0, Math.PI * 2, false)
  }
  const geometry = face
    ? new THREE.ShapeGeometry(shape, 24)
    : new THREE.ExtrudeGeometry(shape, { depth: SIGN_BOARD_THICKNESS, bevelEnabled: false, curveSegments: 24 })
  if (face) {
    const positions = geometry.getAttribute('position')
    const uv = geometry.getAttribute('uv')
    for (let i = 0; i < positions.count; i++) {
      uv.setXY(i, THREE.MathUtils.clamp((positions.getX(i) / SIGN_RADIUS + 1) / 2, 0, 1), THREE.MathUtils.clamp((positions.getY(i) / SIGN_RADIUS + 1) / 2, 0, 1))
    }
    geometry.translate(0, 0, SIGN_BOARD_THICKNESS / 2 + SIGN_FACE_OFFSET)
  } else geometry.translate(0, 0, -SIGN_BOARD_THICKNESS / 2)
  geometry.rotateY(Math.PI / 2)
  return geometry
}

function drawArrow(ctx: CanvasRenderingContext2D, angle: number): void {
  ctx.save()
  ctx.translate(128, 128)
  ctx.rotate(angle)
  ctx.beginPath()
  ctx.moveTo(-15, 75)
  ctx.lineTo(15, 75)
  ctx.lineTo(15, -12)
  ctx.lineTo(47, -12)
  ctx.lineTo(0, -75)
  ctx.lineTo(-47, -12)
  ctx.lineTo(-15, -12)
  ctx.closePath()
  ctx.fill()
  ctx.restore()
}

export function signArrowAngles(direction: SignDirection): number[] {
  switch (direction) {
    case 'left': return [-Math.PI / 2]
    case 'right': return [Math.PI / 2]
    case 'left_or_straight': return [-Math.PI / 2, 0]
    case 'right_or_straight': return [Math.PI / 2, 0]
    case 'left_or_right': return [-Math.PI / 2, Math.PI / 2]
    default: return [0]
  }
}

export function createTrafficSignTexture(kind: SignKind, direction: SignDirection): THREE.CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = 256
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('標識用キャンバスを作れません')
  ctx.fillStyle = kind === 'stop' ? TRAFFIC_SIGN_COLORS.red : TRAFFIC_SIGN_COLORS.blue
  ctx.fillRect(0, 0, 256, 256)
  ctx.strokeStyle = TRAFFIC_SIGN_COLORS.white
  ctx.fillStyle = TRAFFIC_SIGN_COLORS.white
  ctx.lineWidth = 10
  if (kind === 'stop') {
    ctx.beginPath()
    ctx.moveTo(13, 9)
    ctx.lineTo(243, 9)
    ctx.lineTo(128, 241)
    ctx.closePath()
    ctx.stroke()
    ctx.font = 'bold 46px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText('止まれ', 128, 93)
  } else if (kind === 'crosswalk') {
    ctx.beginPath()
    ctx.moveTo(128, 22)
    ctx.lineTo(232, 233)
    ctx.lineTo(24, 233)
    ctx.closePath()
    ctx.fill()
    ctx.fillStyle = TRAFFIC_SIGN_COLORS.blue
    ctx.fillRect(62, 198, 132, 9)
    ctx.fillRect(76, 218, 103, 9)
    ctx.beginPath()
    ctx.arc(127, 99, 13, 0, Math.PI * 2)
    ctx.fill()
    ctx.strokeStyle = TRAFFIC_SIGN_COLORS.blue
    ctx.lineWidth = 13
    ctx.beginPath()
    ctx.moveTo(127, 116); ctx.lineTo(125, 153)
    ctx.moveTo(94, 144); ctx.lineTo(126, 125); ctx.lineTo(155, 148)
    ctx.moveTo(125, 153); ctx.lineTo(99, 188)
    ctx.moveTo(125, 153); ctx.lineTo(155, 183)
    ctx.stroke()
  } else if (kind === 'one_way') {
    ctx.translate(0, 64)
    ctx.scale(1, 0.5)
    drawArrow(ctx, 0)
  } else if (kind === 'mandatory_direction') {
    for (const angle of signArrowAngles(direction)) drawArrow(ctx, angle)
  } else {
    ctx.strokeStyle = TRAFFIC_SIGN_COLORS.red
    ctx.lineWidth = 24
    ctx.beginPath()
    ctx.arc(128, 128, 114, 0, Math.PI * 2)
    ctx.moveTo(49, 49); ctx.lineTo(207, 207)
    if (kind === 'no_stopping') { ctx.moveTo(49, 207); ctx.lineTo(207, 49) }
    ctx.stroke()
  }
  const texture = new THREE.CanvasTexture(canvas)
  texture.colorSpace = THREE.SRGBColorSpace
  texture.anisotropy = 4
  return texture
}
