/** 実用モードの徒歩キャラクター。**60fps で動くので zustand には入れない**（frameBuffer と同じ）。 */

/** WASD の押下状態 */
export interface PedestrianInput {
  forward: boolean
  back: boolean
  left: boolean
  right: boolean
}

export interface PedestrianState {
  /** ENU x [m]（東が正） */
  x: number
  /** ENU y [m]（北が正） */
  y: number
  /** 体の向き [rad]。ENU +x から反時計回り */
  heading: number
  /** 視線の上下 [rad]。上が正。**体の向きには影響しない** */
  pitch: number
  /** 乗車中か。true の間はカメラを運転席へ譲る */
  riding: boolean
  /** 乗っている車両スロット。-1 なら乗っていない（降車位置を出すのに使う） */
  ridingVehicle: number
  /** ポインタロック中か。false なら視線を動かさない */
  locked: boolean
  input: PedestrianInput
  /** 照準が合っている車両スロット。-1 なら無し */
  aimed: number
  /** 歩行の位相 [rad]。手足の振りに使う */
  stride: number
  /** いまの歩行速度 [m/s] */
  speed: number
  /** 歩くときの上下動 [m]。カメラもこれに合わせて揺れる */
  bob: number
  /** ワープするたびに増える。カメラの補間を切るのに使う */
  teleport: number
  /** 3D 側が初期化されているか */
  placed: boolean
}

export const pedestrian: PedestrianState = {
  x: 0,
  y: 0,
  heading: 0,
  pitch: 0,
  riding: false,
  ridingVehicle: -1,
  locked: false,
  input: { forward: false, back: false, left: false, right: false },
  aimed: -1,
  stride: 0,
  speed: 0,
  bob: 0,
  teleport: 0,
  placed: false,
}

/** 指定地点へワープさせる（乗車地点が決まったとき・降車したとき）。 */
export function placePedestrian(x: number, y: number, heading: number): void {
  pedestrian.x = x
  pedestrian.y = y
  pedestrian.heading = heading
  pedestrian.pitch = 0
  pedestrian.speed = 0
  pedestrian.stride = 0
  pedestrian.bob = 0
  pedestrian.teleport += 1
  pedestrian.placed = true
}

/** 押しっぱなしのキーを離した扱いにする（ポインタロックが外れたとき）。 */
export function releasePedestrianKeys(): void {
  pedestrian.input.forward = false
  pedestrian.input.back = false
  pedestrian.input.left = false
  pedestrian.input.right = false
}

/** 実用モードを抜けたときに素の状態へ戻す。 */
export function resetPedestrian(): void {
  releasePedestrianKeys()
  pedestrian.riding = false
  pedestrian.ridingVehicle = -1
  pedestrian.locked = false
  pedestrian.aimed = -1
  pedestrian.speed = 0
  pedestrian.stride = 0
  pedestrian.bob = 0
  pedestrian.placed = false
}
