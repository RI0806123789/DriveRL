/** 歩行者用信号機（縦 2 灯）の 3D 配置を組み立てる純粋な計算。 */

import * as THREE from 'three'

import type { NodePosition, SignalPlacementInput } from './signalGeometry'

/** 灯火径。歩行者用は車両用（300mm）より小さい 250mm */
export const PED_LAMP_DIAMETER = 0.25
export const PED_LAMP_RADIUS = PED_LAMP_DIAMETER / 2
/** 灯火の中心間隔（縦 2 灯） */
export const PED_LAMP_PITCH = 0.32
/** 灯器筐体 */
export const PED_HOUSING_W = 0.36
export const PED_HOUSING_H = 0.74
export const PED_HOUSING_D = 0.18
/** 灯器の中心高さ [m]。歩行者から見上げる位置 */
export const PED_MOUNT_HEIGHT = 2.55
export const PED_POLE_RADIUS = 0.055

/** 灯火の役割。**車両用と並びが違う**（上が赤・下が青） */
export const PED_ROLE_RED = 0
export const PED_ROLE_GREEN = 1

/**
 * 歩行者用信号は**車両信号の裏返し**。車が赤で止まっていれば渡ってよい。
 * `frame.signals` の現示（0=青 / 1=黄 / 2=赤）を受けて、渡れるかどうかを返す。
 */
export function pedestrianWalkable(vehiclePhase: number): boolean {
  return vehiclePhase === 2
}

/** 灯器を道路端からどれだけ外に置くか [m] */
export const PED_SIDE_MARGIN = 1.1
/** 停止線から横断歩道の中心までの距離 [m]（停止線は横断歩道の 5m 手前に引かれる） */
export const PED_CROSSWALK_AHEAD = 2.5

export interface PedestrianLampSlot {
  /** three 空間での灯火中心 */
  position: THREE.Vector3
  /** 灯器の正面が向く方位 [rad]（ENU） */
  facing: number
  /** 0=赤（上） / 1=青（下） */
  role: number
  /** 対応する**車両信号**の添字。色はこの現示の裏返しで決まる */
  signalIndex: number
}

export interface PedestrianSignalResult {
  /** 支柱と筐体を 1 つに統合するための部品 */
  parts: THREE.BufferGeometry[]
  lamps: PedestrianLampSlot[]
  /** 検証用: 灯器ごとの位置と向き */
  placements: Array<{ x: number; z: number; facing: number; signalIndex: number }>
}

/**
 * 車両信号 1 基につき、その停止線の先にある横断歩道の**両端**へ灯器を 1 基ずつ立てる。
 * 渡ってくる人に正対させるので、左端の灯器は右を、右端の灯器は左を向く。
 */
export function buildPedestrianSignalPlacement(
  signals: SignalPlacementInput[],
  nodes: NodePosition[],
): PedestrianSignalResult {
  const nodeById = new Map<number, NodePosition>()
  for (const n of nodes) nodeById.set(n.id, n)

  const parts: THREE.BufferGeometry[] = []
  const lamps: PedestrianLampSlot[] = []
  const placements: PedestrianSignalResult['placements'] = []

  for (let si = 0; si < signals.length; si++) {
    const sg = signals[si]
    const node = nodeById.get(sg.nodeId)
    const cos = Math.cos(sg.heading)
    const sin = Math.sin(sg.heading)
    // 停止線（信号の座標）から交差点側へ少し入ったところが横断歩道
    const baseX = sg.x + cos * PED_CROSSWALK_AHEAD
    const baseY = sg.y + sin * PED_CROSSWALK_AHEAD
    const leftX = -sin
    const leftY = cos
    const offset = sg.roadWidth / 2 + PED_SIDE_MARGIN

    for (const side of [-1, 1] as const) {
      const enuX = baseX + leftX * offset * side
      const enuY = baseY + leftY * offset * side
      const x = enuX
      const z = -enuY
      // 渡ってくる人に正対する。左端（side=+1）の灯器は右を向く
      const facing = sg.heading - (Math.PI / 2) * side

      placements.push({ x, z, facing, signalIndex: si })

      const poleHeight = PED_MOUNT_HEIGHT + PED_HOUSING_H / 2
      const pole = new THREE.CylinderGeometry(
        PED_POLE_RADIUS,
        PED_POLE_RADIUS * 1.25,
        poleHeight,
        7,
      )
      pole.translate(x, poleHeight / 2, z)
      parts.push(pole)

      const housing = new THREE.BoxGeometry(PED_HOUSING_D, PED_HOUSING_H, PED_HOUSING_W)
      housing.rotateY(facing)
      housing.translate(x, PED_MOUNT_HEIGHT, z)
      parts.push(housing)

      const frontX = Math.cos(facing)
      const frontZ = -Math.sin(facing)
      for (const role of [PED_ROLE_RED, PED_ROLE_GREEN]) {
        const dy = role === PED_ROLE_RED ? PED_LAMP_PITCH / 2 : -PED_LAMP_PITCH / 2
        lamps.push({
          position: new THREE.Vector3(
            x + frontX * (PED_HOUSING_D / 2),
            PED_MOUNT_HEIGHT + dy,
            z + frontZ * (PED_HOUSING_D / 2),
          ),
          facing,
          role,
          signalIndex: si,
        })
      }
    }
  }

  return { parts, lamps, placements }
}

/** 灯火に使う角板。法線を +X に向けてある（rotation.y にそのまま方位を入れられる） */
export function createPedestrianLampGeometry(): THREE.BufferGeometry {
  const g = new THREE.PlaneGeometry(PED_LAMP_DIAMETER, PED_LAMP_DIAMETER)
  g.rotateY(Math.PI / 2)
  const count = g.attributes.position.count
  g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(count * 3).fill(1), 3))
  return g
}
