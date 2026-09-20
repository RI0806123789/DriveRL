/** 車両の形と姿勢の計算。**React から切り離した純粋モジュール。** */

import * as THREE from 'three'

/** 車輪の半径 [m]。転がり角の計算にも使う */
export const WHEEL_RADIUS = 0.34

/** 車体ボックスの中心（車両ローカル） */
export const BODY_OFFSET: readonly [number, number, number] = [0, 0.62, 0]
/** ノーズの中心（前方が一目で分かるようにする出っ張り） */
export const NOSE_OFFSET: readonly [number, number, number] = [2.18, 0.72, 0]
/** キャビンの中心 */
export const CABIN_OFFSET: readonly [number, number, number] = [-0.25, 1.28, 0]

/** 車輪の取り付け位置と、舵角を効かせるかどうか。 */
export const WHEEL_OFFSETS: ReadonlyArray<{
  readonly position: readonly [number, number, number]
  readonly steered: boolean
}> = [
  { position: [1.45, WHEEL_RADIUS, 0.86], steered: true },
  { position: [1.45, WHEEL_RADIUS, -0.86], steered: true },
  { position: [-1.45, WHEEL_RADIUS, 0.86], steered: false },
  { position: [-1.45, WHEEL_RADIUS, -0.86], steered: false },
]

/** 車体。中心が地上 0.62m に来るよう平行移動を焼き込む */
export function makeBodyGeometry(): THREE.BufferGeometry {
  const g = new THREE.BoxGeometry(4.4, 0.72, 1.86)
  g.translate(BODY_OFFSET[0], BODY_OFFSET[1], BODY_OFFSET[2])
  return g
}

/** ノーズ */
export function makeNoseGeometry(): THREE.BufferGeometry {
  const g = new THREE.BoxGeometry(0.35, 0.28, 1.5)
  g.translate(NOSE_OFFSET[0], NOSE_OFFSET[1], NOSE_OFFSET[2])
  return g
}

/** キャビン */
export function makeCabinGeometry(): THREE.BufferGeometry {
  const g = new THREE.BoxGeometry(2.15, 0.62, 1.58)
  g.translate(CABIN_OFFSET[0], CABIN_OFFSET[1], CABIN_OFFSET[2])
  return g
}

/** 車輪。シリンダーの軸（+Y）を車体左右方向（+Z）へ向ける。 */
export function makeWheelGeometry(): THREE.BufferGeometry {
  const g = new THREE.CylinderGeometry(WHEEL_RADIUS, WHEEL_RADIUS, 0.24, 14)
  g.rotateX(Math.PI / 2)
  return g
}

/** ナンバープレート 1 枚。法線を +X に向ける（車両ローカルの前方） */
export function makePlateGeometry(width: number, height: number): THREE.BufferGeometry {
  const g = new THREE.PlaneGeometry(width, height)
  g.rotateY(Math.PI / 2)
  return g
}

/** プレート 1 枚のワールド行列。前は前方、後ろは真後ろを向く。 */
export function composePlateMatrix(
  scratch: TransformScratch,
  base: THREE.Matrix4,
  position: readonly [number, number, number],
  yaw: number,
  out: THREE.Matrix4,
): THREE.Matrix4 {
  const l = scratch.light
  l.position.set(position[0], position[1], position[2])
  l.rotation.set(0, yaw, 0)
  l.scale.setScalar(1)
  l.updateMatrix()
  return out.multiplyMatrices(base, l.matrix)
}

/** 灯体 1 個。位置は `scene/vehicleLights.ts` の `LIGHT_SLOTS` が決める */
export function makeLightGeometry(size: readonly [number, number, number]): THREE.BufferGeometry {
  return new THREE.BoxGeometry(size[0], size[1], size[2])
}

/** 行列を組み立てるときの作業用オブジェクト。毎フレームの new を避ける */
export interface TransformScratch {
  node: THREE.Object3D
  wheel: THREE.Object3D
  light: THREE.Object3D
}

export function createTransformScratch(): TransformScratch {
  return {
    node: new THREE.Object3D(),
    wheel: new THREE.Object3D(),
    light: new THREE.Object3D(),
  }
}

/** 車両そのものの姿勢（ENU の位置と方位 → three のワールド行列）。 */
export function composeVehicleMatrix(
  scratch: TransformScratch,
  enuX: number,
  enuY: number,
  heading: number,
  out: THREE.Matrix4,
): THREE.Matrix4 {
  const n = scratch.node
  n.position.set(enuX, 0, -enuY)
  n.rotation.set(0, heading, 0)
  n.scale.setScalar(1)
  n.updateMatrix()
  return out.copy(n.matrix)
}

/** 灯体 1 個のワールド行列。車体に固定なので回転は持たない。 */
export function composeLightMatrix(
  scratch: TransformScratch,
  base: THREE.Matrix4,
  position: readonly [number, number, number],
  out: THREE.Matrix4,
): THREE.Matrix4 {
  const l = scratch.light
  l.position.set(position[0], position[1], position[2])
  l.rotation.set(0, 0, 0)
  l.scale.setScalar(1)
  l.updateMatrix()
  return out.multiplyMatrices(base, l.matrix)
}

/** 車輪 1 本のワールド行列。 */
export function composeWheelMatrix(
  scratch: TransformScratch,
  base: THREE.Matrix4,
  index: number,
  steer: number,
  roll: number,
  out: THREE.Matrix4,
): THREE.Matrix4 {
  const spec = WHEEL_OFFSETS[index]
  const w = scratch.wheel
  w.position.set(spec.position[0], spec.position[1], spec.position[2])
  w.rotation.set(0, spec.steered ? steer : 0, roll)
  w.scale.setScalar(1)
  w.updateMatrix()
  return out.multiplyMatrices(base, w.matrix)
}
