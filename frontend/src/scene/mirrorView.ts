/** ルームミラーとドアミラーの鏡面と、そこへ後ろの景色を映す視点（鏡に映した目）。**React から切り離した純粋モジュール。** */

import * as THREE from 'three'
import { DRIVER_EYE_LOCAL } from './cameraMath.ts'
import { vec, type Vec3 } from './meshBuilder.ts'
import { BACKLIGHT_BOTTOM_X, BACKLIGHT_TOP_X, BELT_Y, GLASS_TOP_Y, WINDSHIELD_TOP_X } from './vehicleGeometry.ts'

export type MirrorKey = 'room' | 'right' | 'left'

/** 鏡面 1 枚（車両ローカル）。`normal` は運転席の側を向き、`right` と `up` は運転席から鏡を見たときの右と上 */
export interface MirrorFace {
  readonly key: MirrorKey
  readonly center: Vec3
  readonly normal: Vec3
  readonly right: Vec3
  readonly up: Vec3
  readonly width: number
  readonly height: number
  /** 凸面鏡の広がり。映す範囲を中心のまわりに何倍へ広げるか（1 で平面鏡） */
  readonly spread: number
  /** 映像の画素数 [横, 縦] */
  readonly pixels: readonly [number, number]
}

function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

function scale(a: Vec3, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s]
}

/** 目の位置 `eye` から見て、反射した先が `look` の向きになるよう鏡を向ける（運転者が鏡を合わせるのと同じ） */
export function aimMirror(center: Vec3, eye: Vec3, look: Vec3): { normal: Vec3; right: Vec3; up: Vec3 } {
  const toEye = vec.normalize(vec.sub(eye, center), [-1, 0, 0])
  const normal = vec.normalize(add(toEye, vec.normalize(look, [-1, 0, 0])), [-1, 0, 0])
  const up = vec.normalize(vec.sub([0, 1, 0], scale(normal, normal[1])), [0, 1, 0])
  const right = vec.cross(up, normal)
  return { normal, right, up }
}

/** ルームミラー。リアガラス越しに真後ろをほぼ水平に映すよう合わせる（地平線が鏡の上寄りに入る） */
export const ROOM_MIRROR_CENTER: Vec3 = [WINDSHIELD_TOP_X + 0.07, 1.33, 0.02]
export const ROOM_MIRROR_SIZE: readonly [number, number] = [0.17, 0.048]
const ROOM_MIRROR_DOWN = THREE.MathUtils.degToRad(1)
/** リアガラスの真ん中（ルームミラーの映像がこの窓を通ることを verify が確かめる） */
export const REAR_GLASS_CENTER: Vec3 = [(BACKLIGHT_BOTTOM_X + BACKLIGHT_TOP_X) / 2, (BELT_Y + GLASS_TOP_Y) / 2, 0]

/** ドアミラーの鏡面の大きさ [幅, 高さ] と、筐体の縁の幅・奥行き・鏡の沈み込み [m] */
export const DOOR_MIRROR_GLASS: readonly [number, number] = [0.17, 0.1]
export const DOOR_MIRROR_RIM = 0.012
export const DOOR_MIRROR_DEPTH = 0.078
export const DOOR_MIRROR_RECESS = 0.014
/** 筐体の縁の中心（右側）と、運転席へ向ける角度 [rad]。左右の筐体は対称で、鏡だけを目に合わせて向ける */
export const DOOR_MIRROR_ORIGIN: Vec3 = [0.889, 1.06, 0.95]
export const DOOR_MIRROR_YAW = THREE.MathUtils.degToRad(25)
/** ドアミラーに映す向き（車の真後ろから外へ 6 度・下へ 2 度） */
const DOOR_MIRROR_OUTWARD = THREE.MathUtils.degToRad(6)
const DOOR_MIRROR_DOWN = THREE.MathUtils.degToRad(2)
/** ドアミラーは凸面鏡（平面鏡より広く映す）。目から遠い助手席側ほど曲率を強くする [右, 左] */
export const DOOR_MIRROR_SPREAD: readonly [number, number] = [1.3, 1.9]

/** 筐体の縁の半幅・半高 */
export const DOOR_MIRROR_HALF: readonly [number, number] = [
  DOOR_MIRROR_GLASS[0] / 2 + DOOR_MIRROR_RIM,
  DOOR_MIRROR_GLASS[1] / 2 + DOOR_MIRROR_RIM,
]

/** ドアミラーの筐体の座標系（`sign` は右 +1・左 -1）。`z` が鏡の面から運転席側へ向く軸、`x × y = z` */
export function doorMirrorFrame(sign: number): { origin: Vec3; x: Vec3; y: Vec3; z: Vec3 } {
  const s = Math.sign(sign) || 1
  const z: Vec3 = [-Math.cos(DOOR_MIRROR_YAW), 0, -s * Math.sin(DOOR_MIRROR_YAW)]
  const y: Vec3 = [0, 1, 0]
  const x: Vec3 = [z[2], 0, -z[0]]
  return { origin: [DOOR_MIRROR_ORIGIN[0], DOOR_MIRROR_ORIGIN[1], s * DOOR_MIRROR_ORIGIN[2]], x, y, z }
}

/** 筐体の座標（x, y, z）を車両ローカルへ */
export function doorMirrorPoint(sign: number, x: number, y: number, z: number): Vec3 {
  const f = doorMirrorFrame(sign)
  return add(f.origin, add(scale(f.x, x), add(scale(f.y, y), scale(f.z, z))))
}

/** 車両ローカルの点を筐体の座標へ */
export function doorMirrorLocal(sign: number, p: Vec3): Vec3 {
  const f = doorMirrorFrame(sign)
  const d = vec.sub(p, f.origin)
  return [vec.dot(d, f.x), vec.dot(d, f.y), vec.dot(d, f.z)]
}

function doorMirrorFace(sign: number): MirrorFace {
  const s = Math.sign(sign) || 1
  const center = doorMirrorPoint(s, 0, 0, -DOOR_MIRROR_RECESS)
  const look: Vec3 = [-1, -Math.tan(DOOR_MIRROR_DOWN), s * Math.tan(DOOR_MIRROR_OUTWARD)]
  return {
    key: s > 0 ? 'right' : 'left',
    center,
    ...aimMirror(center, DRIVER_EYE_LOCAL, look),
    width: DOOR_MIRROR_GLASS[0],
    height: DOOR_MIRROR_GLASS[1],
    spread: s > 0 ? DOOR_MIRROR_SPREAD[0] : DOOR_MIRROR_SPREAD[1],
    pixels: [224, 132],
  }
}

function roomMirrorFace(): MirrorFace {
  return {
    key: 'room',
    center: ROOM_MIRROR_CENTER,
    ...aimMirror(ROOM_MIRROR_CENTER, DRIVER_EYE_LOCAL, [-1, -Math.tan(ROOM_MIRROR_DOWN), 0]),
    width: ROOM_MIRROR_SIZE[0],
    height: ROOM_MIRROR_SIZE[1],
    spread: 1,
    pixels: [384, 107],
  }
}

/** 3 枚の鏡。並びは描き直す順番の既定にもなる */
export const MIRROR_FACES: ReadonlyArray<MirrorFace> = [roomMirrorFace(), doorMirrorFace(1), doorMirrorFace(-1)]

/** 鏡の面から運転席側へ、この厚みより手前は映さない [m]（鏡の縁や筐体を映さない） */
export const MIRROR_NEAR_PAD = 0.04
/** 映す距離の上限 [m] */
export const MIRROR_FAR_M = 350

/** 鏡に映す視点。`matrix` はカメラの姿勢（車両ローカル）、残りは視錐台（`makePerspective` の引数） */
export interface MirrorView {
  readonly matrix: THREE.Matrix4
  readonly eye: Vec3
  readonly left: number
  readonly right: number
  readonly top: number
  readonly bottom: number
  readonly near: number
  readonly far: number
}

/** 鏡の四隅（車両ローカル）。鏡に映した目から見て 左下・右下・左上・右上（運転席から見ると左右が入れ替わる） */
export function mirrorCorners(face: MirrorFace): [Vec3, Vec3, Vec3, Vec3] {
  const hw = scale(face.right, face.width / 2)
  const hh = scale(face.up, face.height / 2)
  const c = face.center
  return [
    add(c, vec.sub(hw, hh)),
    vec.sub(c, add(hw, hh)),
    add(c, add(hw, hh)),
    add(c, vec.sub(hh, hw)),
  ]
}

/** 目を鏡の面で折り返した位置から、鏡を窓として見る視錐台（鏡の面に垂直な軸の、中心を外した透視） */
export function mirrorView(face: MirrorFace, eye: Vec3 = DRIVER_EYE_LOCAL, far = MIRROR_FAR_M): MirrorView {
  const n = face.normal
  const dist = vec.dot(vec.sub(eye, face.center), n)
  const pe = vec.sub(eye, scale(n, 2 * dist))
  const vr = scale(face.right, -1)
  const vu = face.up
  const vn = scale(n, -1)
  const [pa, pb, pc] = mirrorCorners(face)
  const va = vec.sub(pa, pe)
  const vb = vec.sub(pb, pe)
  const vc = vec.sub(pc, pe)
  const d = -vec.dot(va, vn)
  const near = d + MIRROR_NEAR_PAD
  const k = near / d
  let l = vec.dot(vr, va) * k
  let r = vec.dot(vr, vb) * k
  let b = vec.dot(vu, va) * k
  let t = vec.dot(vu, vc) * k
  const cx = (l + r) / 2
  const cy = (b + t) / 2
  l = cx + (l - cx) * face.spread
  r = cx + (r - cx) * face.spread
  b = cy + (b - cy) * face.spread
  t = cy + (t - cy) * face.spread
  const matrix = new THREE.Matrix4().makeBasis(
    new THREE.Vector3(vr[0], vr[1], vr[2]),
    new THREE.Vector3(vu[0], vu[1], vu[2]),
    new THREE.Vector3(vn[0], vn[1], vn[2]),
  )
  matrix.setPosition(pe[0], pe[1], pe[2])
  return { matrix, eye: pe, left: l, right: r, top: t, bottom: b, near, far }
}

/** 映像を貼る板。鏡の面から `lift` だけ運転席側へ浮かせる。映像の左右は鏡に映した目の向きのまま（＝運転席から見て反転） */
export function makeMirrorQuadGeometry(face: MirrorFace, lift: number): THREE.BufferGeometry {
  const off = scale(face.normal, lift)
  const [pa, pb, pc, pd] = mirrorCorners(face).map((p) => add(p, off))
  // 表（反時計回り）が運転席側を向く並び
  const pos = [...pa, ...pd, ...pb, ...pa, ...pc, ...pd]
  const uv = [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2))
  g.setAttribute('normal', new THREE.Float32BufferAttribute([0, 1, 2, 3, 4, 5].flatMap(() => [...face.normal]), 3))
  g.computeBoundingSphere()
  return g
}

/** 1 枚の鏡を描き直す間隔 [s] と、1 フレームに描き直す枚数の上限（1 枚ずつ順に描いて、フレームごとの重さをならす） */
export const MIRROR_INTERVAL_SEC = 1 / 30
export const MIRRORS_PER_FRAME = 1

/** 描き直す鏡。画面に見えていて前回から間隔が空いたものを、古い順に上限まで */
export function mirrorsDue(
  ages: ReadonlyArray<number>,
  visible: ReadonlyArray<boolean>,
  interval = MIRROR_INTERVAL_SEC,
  perFrame = MIRRORS_PER_FRAME,
): number[] {
  const due: number[] = []
  for (let i = 0; i < ages.length; i++) if (visible[i] && ages[i] >= interval) due.push(i)
  due.sort((a, b) => ages[b] - ages[a])
  return due.slice(0, perFrame)
}
