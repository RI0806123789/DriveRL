/**
 * 交通信号機の 3D 配置を組み立てる純粋な計算。
 *
 * React に依存させず three だけに依存させてあるのは、
 * `frontend/scripts/verify-signal-geometry.ts` から Node で直接読み込んで
 * 向きの検証ができるようにするため。ブラウザを開かずに
 * 「灯器が運転者に正対しているか」「青が運転者から見て左か」を確かめられる。
 *
 * 座標変換の規約は docs/protocol.md 1.3 に従う:
 *   three.x = enu.x,  three.z = -enu.y,  前方 = +X
 */

import * as THREE from 'three'

// --- 寸法（メートル）。日本の車両用信号機の実寸に合わせている ---

/** 灯火径 300mm */
export const LAMP_DIAMETER = 0.3
export const LAMP_RADIUS = LAMP_DIAMETER / 2
/** 灯火の中心間隔 */
export const LAMP_PITCH = 0.35
/** 灯器筐体（3 灯 + 縁） */
export const HOUSING_W = 1.16
export const HOUSING_H = 0.44
export const HOUSING_D = 0.22
/** フード（庇）の突き出し */
export const HOOD_DEPTH = 0.2
/** 灯器下端の路面からの高さ。車道上に張り出す場合の基準は 5.0m */
export const MOUNT_HEIGHT = 5.0
export const POLE_RADIUS = 0.07
export const ARM_RADIUS = 0.05

/** 灯火の役割。docs/protocol.md の frame.signals と同じ並び */
export const ROLE_GREEN = 0
export const ROLE_YELLOW = 1
export const ROLE_RED = 2

/** この計算に必要な信号機の情報だけを受け取る（protocol の型に依存しない） */
export interface SignalPlacementInput {
  nodeId: number
  x: number
  y: number
  heading: number
  roadWidth: number
}

export interface NodePosition {
  id: number
  x: number
  y: number
}

export interface LampSlot {
  /** three 空間での灯火中心 */
  position: THREE.Vector3
  /** 灯器の正面が向く方位 [rad]（ENU）。インスタンスの rotation.y にそのまま入れる */
  facing: number
  /** 0=青 / 1=黄 / 2=赤 */
  role: number
  /** 対応する信号機の添字 */
  signalIndex: number
}

export interface SignalGeometryResult {
  /** 支柱・アーム・筐体・フードを 1 つに統合したもの（灯火は含まない） */
  parts: THREE.BufferGeometry[]
  lamps: LampSlot[]
  /** 検証用: 信号ごとの支柱・灯器の位置 */
  placements: Array<{
    headX: number
    headZ: number
    poleX: number
    poleZ: number
    facing: number
  }>
}

/**
 * 信号機の静的部品と灯火の配置を計算する。
 *
 * 日本の配置に合わせている点:
 *   - 灯器は**交差点の対面側**（進行方向の向こう側）に置く
 *   - 支柱は進行方向の**左側**（左側通行のため）
 *   - 灯火は運転者から見て**左から 青・黄・赤**
 */
export function buildSignalPlacement(
  signals: SignalPlacementInput[],
  nodes: NodePosition[],
): SignalGeometryResult {
  const nodeById = new Map<number, NodePosition>()
  for (const n of nodes) nodeById.set(n.id, n)

  const parts: THREE.BufferGeometry[] = []
  const lamps: LampSlot[] = []
  const placements: SignalGeometryResult['placements'] = []

  for (let si = 0; si < signals.length; si++) {
    const sg = signals[si]
    const node = nodeById.get(sg.nodeId)
    // 交差点中心が引けないときは停止線の位置で代用する
    const cx = node ? node.x : sg.x
    const cy = node ? node.y : sg.y

    const cos = Math.cos(sg.heading)
    const sin = Math.sin(sg.heading)
    // 進行方向の左（ENU では反時計回りに 90 度）
    const leftX = -sin
    const leftY = cos

    // 交差点を越えた先に灯器を置く
    const beyond = sg.roadWidth / 2 + 2.0
    const headEnuX = cx + cos * beyond + leftX * (sg.roadWidth * 0.25)
    const headEnuY = cy + sin * beyond + leftY * (sg.roadWidth * 0.25)
    // 支柱は同じ奥行きの、進行方向左側の路端
    const poleEnuX = cx + cos * beyond + leftX * (sg.roadWidth / 2 + 0.8)
    const poleEnuY = cy + sin * beyond + leftY * (sg.roadWidth / 2 + 0.8)

    // ENU -> three（z = -y）
    const headX = headEnuX
    const headZ = -headEnuY
    const poleX = poleEnuX
    const poleZ = -poleEnuY

    const headCenterY = MOUNT_HEIGHT + HOUSING_H / 2
    // 灯器は進入車両に正対する = 進行方向の逆を向く
    const facing = sg.heading + Math.PI

    placements.push({ headX, headZ, poleX, poleZ, facing })

    // --- 支柱 ---
    const poleHeight = MOUNT_HEIGHT + HOUSING_H + 0.4
    const pole = new THREE.CylinderGeometry(POLE_RADIUS, POLE_RADIUS * 1.2, poleHeight, 8)
    pole.translate(poleX, poleHeight / 2, poleZ)
    parts.push(pole)

    // --- アーム（支柱から灯器へ水平に伸ばす） ---
    const armDx = headX - poleX
    const armDz = headZ - poleZ
    const armLen = Math.hypot(armDx, armDz)
    if (armLen > 0.2) {
      const arm = new THREE.CylinderGeometry(ARM_RADIUS, ARM_RADIUS, armLen, 6)
      // 軸(+Y)を水平(±X)に倒してから、アーム方向へ向ける
      arm.rotateZ(Math.PI / 2)
      arm.rotateY(-Math.atan2(armDz, armDx))
      arm.translate((poleX + headX) / 2, poleHeight - 0.25, (poleZ + headZ) / 2)
      parts.push(arm)
    }

    // --- 灯器の筐体（背面板を兼ねる） ---
    // 前方 = +X の規約に合わせ、奥行きを X 軸に取る。これで rotateY(facing) だけで正面が向く。
    const housing = new THREE.BoxGeometry(HOUSING_D, HOUSING_H, HOUSING_W)
    housing.rotateY(facing)
    housing.translate(headX, headCenterY, headZ)
    parts.push(housing)

    // 灯器を吊るす金具
    const hanger = new THREE.CylinderGeometry(0.03, 0.03, 0.3, 6)
    hanger.translate(headX, headCenterY + HOUSING_H / 2 + 0.12, headZ)
    parts.push(hanger)

    // --- 3 灯 + フード ---
    // 「運転者から見て左」は ENU の heading + PI/2 側。three では (leftX, -leftY)。
    const acrossX = leftX
    const acrossZ = -leftY
    const frontX = Math.cos(facing)
    const frontZ = -Math.sin(facing)

    for (let k = 0; k < 3; k++) {
      // across は「運転者から見て左」の向き。青(k=0) を最も左に置きたいので
      // オフセットは +LAMP_PITCH から始めて減らしていく。
      // ここを (k - 1) にすると並びが左右反転して「左から赤・黄・青」になる。
      const t = (1 - k) * LAMP_PITCH
      const lx = headX + acrossX * t + frontX * (HOUSING_D / 2)
      const lz = headZ + acrossZ * t + frontZ * (HOUSING_D / 2)

      lamps.push({
        position: new THREE.Vector3(lx, headCenterY, lz),
        facing,
        role: k, // 0=青 / 1=黄 / 2=赤
        signalIndex: si,
      })

      // フード（庇）
      const hood = new THREE.CylinderGeometry(
        LAMP_RADIUS + 0.035,
        LAMP_RADIUS + 0.035,
        HOOD_DEPTH,
        10,
        1,
        true,
      )
      // 軸(+Y) -> rotateX(PI/2) で +Z -> さらに +PI/2 で +X（前方規約）-> facing へ
      hood.rotateX(Math.PI / 2)
      hood.rotateY(Math.PI / 2 + facing)
      hood.translate(
        lx + frontX * (HOOD_DEPTH / 2),
        headCenterY + 0.02,
        lz + frontZ * (HOOD_DEPTH / 2),
      )
      parts.push(hood)
    }
  }

  return { parts, lamps, placements }
}

/** 灯火に使う円板。法線を +X に向けてある（rotation.y にそのまま方位を入れられる） */
export function createLampGeometry(): THREE.BufferGeometry {
  const g = new THREE.CircleGeometry(LAMP_RADIUS, 16)
  // CircleGeometry の法線は +Z。前方 = +X の規約へ合わせる。
  g.rotateY(Math.PI / 2)
  // three のシェーダは USE_COLOR（= material.vertexColors）が立っていないと
  // instanceColor を最終色に掛けない。一方 vertexColors を立てると geometry の
  // color 属性が要求され、無いと未バインド属性 (0,0,0) 扱いで真っ黒になる。
  // 白で埋めた color 属性を持たせて両方を成立させる。
  const count = g.attributes.position.count
  g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(count * 3).fill(1), 3))
  return g
}
