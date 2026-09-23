/** 車体の傾き（加減速のピッチ・旋回のロール）。**React から切り離した純粋モジュール。** */

import * as THREE from 'three'

/** 加速度 1 m/s² あたりの傾き [rad]。加速で後ろが沈み（正）、減速で前が沈む（負） */
export const PITCH_PER_ACCEL = THREE.MathUtils.degToRad(0.28)
/** 横向きの加速度 1 m/s² あたりのロール [rad]。旋回の外側へ傾く */
export const ROLL_PER_ACCEL = THREE.MathUtils.degToRad(0.45)
/** 傾きの上限 [rad]。**数度まで**（外接直方体からはみ出すと認識結果の枠とずれて見える） */
export const MAX_PITCH = THREE.MathUtils.degToRad(1.3)
export const MAX_ROLL = THREE.MathUtils.degToRad(2.0)

/** ばねの揺り戻しで上限を越えてよい割合（ここで止める） */
export const TILT_OVERSHOOT = 1.15

/** 傾きの支点の高さ [m]（重心のあたり） */
export const TILT_PIVOT_Y = 0.45

/** ホイールベース [m]（`backend/app/config.py` の WHEELBASE）。横加速度の見積もりに使う */
export const WHEELBASE_M = 2.6
/** 横加速度の上限 [m/s²]（`config.MAX_LATERAL_ACCEL`） */
export const MAX_LATERAL_ACCEL = 4.5
/** 前後の加速度の上限 [m/s²]（`config.MAX_ACCEL` / `MAX_DECEL`） */
const ACCEL_RANGE: readonly [number, number] = [-6, 3]

/** サスペンションのばね（固有角振動数 [rad/s] と減衰比）。減衰比 1 未満だと揺り戻しが出る */
export const SUSPENSION_OMEGA = 8.5
export const SUSPENSION_ZETA = 0.85

/** 速度の微分を均す時定数 [s]。補間済みの速度でも 20Hz の折れ目が残るため */
export const ACCEL_TAU_S = 0.12

/** 1 台ぶんの状態 */
export interface TiltState {
  pitch: number
  pitchRate: number
  roll: number
  rollRate: number
  /** 前のフレームの速度 [m/s] */
  lastSpeed: number
  /** 均した前後の加速度 [m/s²] */
  accel: number
  /** 前のフレームの値が使えるか（出現・瞬間移動の直後は false） */
  primed: boolean
}

export function createTiltState(): TiltState {
  return { pitch: 0, pitchRate: 0, roll: 0, rollRate: 0, lastSpeed: 0, accel: 0, primed: false }
}

/** 出現・瞬間移動したとき。傾きを水平へ戻し、次のフレームから測り直す */
export function resetTilt(s: TiltState, speed: number): void {
  s.pitch = 0
  s.pitchRate = 0
  s.roll = 0
  s.rollRate = 0
  s.lastSpeed = speed
  s.accel = 0
  s.primed = true
}

/** 横向きの加速度 [m/s²]。左へ曲がる（舵角が正）と正 */
export function lateralAccel(speed: number, steer: number): number {
  const a = (speed * speed * Math.tan(steer)) / WHEELBASE_M
  return Math.max(-MAX_LATERAL_ACCEL, Math.min(MAX_LATERAL_ACCEL, a))
}

/** 加速度から目標の傾き [rad]。 */
export function tiltTarget(accelLong: number, accelLat: number): { pitch: number; roll: number } {
  return {
    pitch: Math.max(-MAX_PITCH, Math.min(MAX_PITCH, accelLong * PITCH_PER_ACCEL)),
    roll: Math.max(-MAX_ROLL, Math.min(MAX_ROLL, accelLat * ROLL_PER_ACCEL)),
  }
}

/** 減衰振動で目標へ寄せる 1 ステップ（半陰的オイラー。dt を細かく刻んで発散させない） */
function spring(x: number, v: number, target: number, dt: number): [number, number] {
  const w = SUSPENSION_OMEGA
  const steps = Math.max(1, Math.ceil(dt / (1 / 240)))
  const h = dt / steps
  for (let k = 0; k < steps; k++) {
    const a = w * w * (target - x) - 2 * SUSPENSION_ZETA * w * v
    v += a * h
    x += v * h
  }
  return [x, v]
}

/** 補間済みの速度・舵角から、傾きを 1 フレーム進める。`dt` は描画の間隔 [s] */
export function stepTilt(s: TiltState, speed: number, steer: number, dt: number): void {
  if (!s.primed) {
    resetTilt(s, speed)
    return
  }
  const h = Math.min(Math.max(dt, 0), 0.1)
  if (h <= 0) return
  const raw = Math.max(ACCEL_RANGE[0], Math.min(ACCEL_RANGE[1], (speed - s.lastSpeed) / h))
  s.lastSpeed = speed
  s.accel += (raw - s.accel) * (1 - Math.exp(-h / ACCEL_TAU_S))
  const target = tiltTarget(s.accel, lateralAccel(speed, steer))
  ;[s.pitch, s.pitchRate] = spring(s.pitch, s.pitchRate, target.pitch, h)
  ;[s.roll, s.rollRate] = spring(s.roll, s.rollRate, target.roll, h)
  s.pitch = Math.max(-MAX_PITCH * TILT_OVERSHOOT, Math.min(MAX_PITCH * TILT_OVERSHOOT, s.pitch))
  s.roll = Math.max(-MAX_ROLL * TILT_OVERSHOOT, Math.min(MAX_ROLL * TILT_OVERSHOOT, s.roll))
}

const pivot = new THREE.Matrix4()
const unpivot = new THREE.Matrix4()
const rot = new THREE.Matrix4()
const euler = new THREE.Euler()

/** 車体の傾き（車両ローカル）。支点まわりにピッチ（Z 軸）とロール（X 軸）を掛ける */
export function composeTiltMatrix(pitch: number, roll: number, out: THREE.Matrix4): THREE.Matrix4 {
  pivot.makeTranslation(0, TILT_PIVOT_Y, 0)
  unpivot.makeTranslation(0, -TILT_PIVOT_Y, 0)
  euler.set(roll, 0, pitch, 'ZYX')
  rot.makeRotationFromEuler(euler)
  return out.copy(pivot).multiply(rot).multiply(unpivot)
}

const tiltTmp = new THREE.Matrix4()
const tiltVec = new THREE.Vector3()

/** 車両ローカルの点を、車体の傾きで動かした位置（車両ローカル）。運転席の目を車体と一緒に動かすのに使う */
export function tiltLocalPoint(
  p: readonly [number, number, number],
  pitch: number,
  roll: number,
): [number, number, number] {
  composeTiltMatrix(pitch, roll, tiltTmp)
  tiltVec.set(p[0], p[1], p[2]).applyMatrix4(tiltTmp)
  return [tiltVec.x, tiltVec.y, tiltVec.z]
}

/** 全車の傾き。**進めるのは `Vehicles` の 1 か所だけ**で、カメラ・ナビ・水滴は読むだけ */
export const vehicleTilt = {
  pitch: new Float32Array(0),
  roll: new Float32Array(0),
}

/** 台数に合わせて入れ物を用意する */
export function ensureTiltSlots(count: number): void {
  if (vehicleTilt.pitch.length === count) return
  vehicleTilt.pitch = new Float32Array(count)
  vehicleTilt.roll = new Float32Array(count)
}

/** 車両 id の傾き（入れ物が無ければ水平） */
export function tiltOf(id: number): { pitch: number; roll: number } {
  return {
    pitch: vehicleTilt.pitch[id] ?? 0,
    roll: vehicleTilt.roll[id] ?? 0,
  }
}
