/** 最高速度標識（規制標識 323「最高速度」）の 3D 配置を組み立てる純粋な計算。 */

import * as THREE from 'three'

/** 標示板の直径。一般道の規制標識は 60cm */
export const SIGN_DIAMETER = 0.6
export const SIGN_RADIUS = SIGN_DIAMETER / 2
/** 標示板下端の路面からの高さ */
export const SIGN_BOTTOM_HEIGHT = 1.8
/** 標示板上端の高さ。支柱はここまで伸ばす */
export const SIGN_TOP_HEIGHT = SIGN_BOTTOM_HEIGHT + SIGN_DIAMETER
export const SIGN_POLE_RADIUS = 0.05
/** 標示板の厚み（縁から見える板厚） */
export const SIGN_BOARD_THICKNESS = 0.04
/** 数字を描いた面を標示板の前面からどれだけ浮かせるか（深度の取り合いを避ける） */
export const SIGN_FACE_OFFSET = 0.002

/** この計算に必要な標識の情報だけを受け取る（protocol の型に依存しない）。 */
export interface SignPlacementInput {
  /** 支柱の位置（ENU）。既に進行方向左側の路端へ寄せてある */
  x: number
  y: number
  /** この標識が規制する側の進行方向 [rad]。標示板は heading + PI を向く */
  heading: number
  /** 規制速度 [m/s] */
  speedLimit: number
}

/** 標示板の中心（three 空間）。 */
export function signBoardCenter(sign: SignPlacementInput): [number, number, number] {
  return [sign.x, SIGN_BOTTOM_HEIGHT + SIGN_RADIUS, -sign.y]
}

/** 標示板の正面が向く方位 [rad]。インスタンスの rotation.y にそのまま入れる。 */
export function signBoardFacing(sign: SignPlacementInput): number {
  return sign.heading + Math.PI
}

/** 標示板の法線（three 空間の単位ベクトル）。検証用。 */
export function signBoardNormal(sign: SignPlacementInput): [number, number, number] {
  const facing = signBoardFacing(sign)
  return [Math.cos(facing), 0, -Math.sin(facing)]
}

/** 支柱の高さ。地面 y=0 から標示板の上端まで */
export function signPoleHeight(sign: SignPlacementInput): number {
  void sign
  return SIGN_TOP_HEIGHT
}

/** 支柱（円柱）の中心。円柱は中心基準なので高さの半分だけ持ち上げる */
export function signPoleCenter(sign: SignPlacementInput): [number, number, number] {
  return [sign.x, signPoleHeight(sign) / 2, -sign.y]
}

/** 標示板に描く数字。規制速度 [m/s] を km/h の整数に丸める */
export function signSpeedKph(sign: SignPlacementInput): number {
  return Math.round(sign.speedLimit * 3.6)
}

/** 支柱の円柱。中心が原点なので、インスタンスの位置に signPoleCenter を入れる */
export function createSignPoleGeometry(): THREE.BufferGeometry {
  return new THREE.CylinderGeometry(
    SIGN_POLE_RADIUS,
    SIGN_POLE_RADIUS * 1.2,
    SIGN_TOP_HEIGHT,
    8,
  )
}

/** 標示板の円板。円柱を寝かせて軸を +X に向けてある */
export function createSignBoardGeometry(): THREE.BufferGeometry {
  const g = new THREE.CylinderGeometry(SIGN_RADIUS, SIGN_RADIUS, SIGN_BOARD_THICKNESS, 24)
  g.rotateZ(-Math.PI / 2)
  return g
}

/** 数字を描いたテクスチャを貼る面。標示板の前面へ僅かに浮かせてある。 */
export function createSignFaceGeometry(): THREE.BufferGeometry {
  const g = new THREE.CircleGeometry(SIGN_RADIUS, 32)
  g.rotateY(Math.PI / 2)
  g.translate(SIGN_BOARD_THICKNESS / 2 + SIGN_FACE_OFFSET, 0, 0)
  return g
}
