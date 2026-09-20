/** 徒歩キャラクターの形と姿勢の計算。**React から切り離した純粋モジュール。** */

import * as THREE from 'three'

/** 身長と目線の高さ [m]。目線は運転席カメラと同じ流儀で「その高さに眼球がある」 */
export const PEDESTRIAN_HEIGHT = 1.68
export const EYE_HEIGHT = 1.58
/** 目を前へ出す量 [m]。頭のメッシュがニアクリップに被らない距離 */
export const EYE_FORWARD = 0.16

/** 歩く速さ [m/s] と、走る（Shift）ときの倍率 */
export const WALK_SPEED_MPS = 2.2
export const RUN_MULTIPLIER = 2.1

/** 手足の振れ幅 [rad] */
export const SWING_MAX_RAD = 0.62

/** 体の各部。位置は歩行者ローカル（前方 +X / 上 +Y / 左 -Z） */
export const HEAD_OFFSET: readonly [number, number, number] = [0.02, 1.5, 0]
export const TORSO_OFFSET: readonly [number, number, number] = [0, 1.12, 0]
export const HIP_OFFSET: readonly [number, number, number] = [0, 0.84, 0]

/** 振り子の支点（肩・股関節）。`steered` 相当の区別は無く、すべて Z 軸回りに振る */
export const LIMB_JOINTS: ReadonlyArray<{
  readonly position: readonly [number, number, number]
  readonly kind: 'arm' | 'leg'
  /** 左右。左が -1 */
  readonly side: -1 | 1
}> = [
  { position: [0, 1.38, -0.22], kind: 'arm', side: -1 },
  { position: [0, 1.38, 0.22], kind: 'arm', side: 1 },
  { position: [0, 0.84, -0.1], kind: 'leg', side: -1 },
  { position: [0, 0.84, 0.1], kind: 'leg', side: 1 },
]

export const ARM_LENGTH = 0.56
export const LEG_LENGTH = 0.82

/** 頭 */
export function makeHeadGeometry(): THREE.BufferGeometry {
  const g = new THREE.SphereGeometry(0.105, 18, 12)
  g.translate(HEAD_OFFSET[0], HEAD_OFFSET[1], HEAD_OFFSET[2])
  return g
}

/** 胴。厚み（前後）× 高さ × 幅（左右） */
export function makeTorsoGeometry(): THREE.BufferGeometry {
  const g = new THREE.BoxGeometry(0.22, 0.54, 0.38)
  g.translate(TORSO_OFFSET[0], TORSO_OFFSET[1], TORSO_OFFSET[2])
  return g
}

/** 腰 */
export function makeHipGeometry(): THREE.BufferGeometry {
  const g = new THREE.BoxGeometry(0.24, 0.16, 0.34)
  g.translate(HIP_OFFSET[0], HIP_OFFSET[1], HIP_OFFSET[2])
  return g
}

/** 腕。**上端（肩）が原点**になるよう下へずらす（支点で回すため） */
export function makeArmGeometry(): THREE.BufferGeometry {
  const g = new THREE.BoxGeometry(0.1, ARM_LENGTH, 0.1)
  g.translate(0, -ARM_LENGTH / 2, 0)
  return g
}

/** 脚。**上端（股関節）が原点** */
export function makeLegGeometry(): THREE.BufferGeometry {
  const g = new THREE.BoxGeometry(0.14, LEG_LENGTH, 0.14)
  g.translate(0, -LEG_LENGTH / 2, 0)
  return g
}

/** 手足の振り角と、歩くときの上下動 */
export interface LimbSwing {
  armLeft: number
  armRight: number
  legLeft: number
  legRight: number
  /** 体を持ち上げる量 [m] */
  bob: number
}

/** 歩行位相から手足の角度を作る。**腕は脚と逆位相**（対角の振り） */
export function limbSwing(stride: number, intensity: number): LimbSwing {
  const amp = SWING_MAX_RAD * Math.max(0, Math.min(1, intensity))
  const s = Math.sin(stride)
  return {
    legRight: s * amp,
    legLeft: -s * amp,
    armRight: -s * amp * 0.75,
    armLeft: s * amp * 0.75,
    bob: Math.abs(Math.cos(stride)) * 0.035 * Math.max(0, Math.min(1, intensity)),
  }
}

/**
 * NPC の服の色を散らす色相のずれ 0.0〜1.0。
 * 黄金角で回すので、隣り合うスロット番号でも色が似ない。
 */
export function npcHueOffset(id: number): number {
  return (id * 0.6180339887498949) % 1
}

/** 関節 1 つぶんの振り角を `LIMB_SWING` から引く */
export function swingFor(index: number, swing: LimbSwing): number {
  const joint = LIMB_JOINTS[index]
  if (joint.kind === 'arm') return joint.side < 0 ? swing.armLeft : swing.armRight
  return joint.side < 0 ? swing.legLeft : swing.legRight
}

export interface PedestrianScratch {
  node: THREE.Object3D
  limb: THREE.Object3D
}

export function createPedestrianScratch(): PedestrianScratch {
  return { node: new THREE.Object3D(), limb: new THREE.Object3D() }
}

/** 体そのものの姿勢（ENU の位置と方位 → three のワールド行列）。車両と同じ規約。 */
export function composePedestrianMatrix(
  scratch: PedestrianScratch,
  enuX: number,
  enuY: number,
  heading: number,
  bob: number,
  out: THREE.Matrix4,
): THREE.Matrix4 {
  const n = scratch.node
  n.position.set(enuX, bob, -enuY)
  n.rotation.set(0, heading, 0)
  n.scale.setScalar(1)
  n.updateMatrix()
  return out.copy(n.matrix)
}

/** 手足 1 本のワールド行列。支点で Z 軸回りに振る（+ が前） */
export function composeLimbMatrix(
  scratch: PedestrianScratch,
  base: THREE.Matrix4,
  index: number,
  swing: number,
  out: THREE.Matrix4,
): THREE.Matrix4 {
  const spec = LIMB_JOINTS[index]
  const l = scratch.limb
  l.position.set(spec.position[0], spec.position[1], spec.position[2])
  l.rotation.set(0, 0, swing)
  l.scale.setScalar(1)
  l.updateMatrix()
  return out.multiplyMatrices(base, l.matrix)
}

/** 目の位置（three 空間）。運転席カメラと同じく ENU → three へ写して返す */
export function eyePosition(
  enuX: number,
  enuY: number,
  heading: number,
  bob = 0,
): { x: number; y: number; z: number } {
  const cos = Math.cos(heading)
  const sin = Math.sin(heading)
  return {
    x: enuX + cos * EYE_FORWARD,
    y: EYE_HEIGHT + bob,
    z: -(enuY + sin * EYE_FORWARD),
  }
}

/** 視線の先（three 空間）。`pitch` は上が正 */
export function eyeLookAt(
  enuX: number,
  enuY: number,
  heading: number,
  pitch: number,
  distance = 12,
  bob = 0,
): { x: number; y: number; z: number } {
  const cos = Math.cos(heading)
  const sin = Math.sin(heading)
  const flat = Math.cos(pitch) * distance
  return {
    x: enuX + cos * (EYE_FORWARD + flat),
    y: EYE_HEIGHT + bob + Math.sin(pitch) * distance,
    z: -(enuY + sin * (EYE_FORWARD + flat)),
  }
}
