/** 車両のライト（前照灯・制動灯・方向指示器）の配置と点灯条件。**純粋モジュール。** */

/** ライトの種類 */
export const LIGHT_HEAD = 0
export const LIGHT_TAIL = 1
export const LIGHT_TURN = 2

/**
 * 1 台ぶんのライト。位置は車両ローカル（前方 +X / 上 +Y / **右 +Z**）。
 * 並びは `vehicleGeometry.ts` の車体（4.4 × 0.72 × 1.86、中心 y=0.62）に合わせてある。
 */
export const LIGHT_SLOTS: ReadonlyArray<{
  readonly position: readonly [number, number, number]
  readonly kind: number
  /** 左右。左が -1 */
  readonly side: -1 | 1
}> = [
  { position: [2.28, 0.66, -0.62], kind: LIGHT_HEAD, side: -1 },
  { position: [2.28, 0.66, 0.62], kind: LIGHT_HEAD, side: 1 },
  { position: [-2.22, 0.72, -0.68], kind: LIGHT_TAIL, side: -1 },
  { position: [-2.22, 0.72, 0.68], kind: LIGHT_TAIL, side: 1 },
  { position: [2.22, 0.6, -0.88], kind: LIGHT_TURN, side: -1 },
  { position: [2.22, 0.6, 0.88], kind: LIGHT_TURN, side: 1 },
  { position: [-2.18, 0.6, -0.88], kind: LIGHT_TURN, side: -1 },
  { position: [-2.18, 0.6, 0.88], kind: LIGHT_TURN, side: 1 },
]

/** 1 台あたりのライトの数 */
export const LIGHTS_PER_VEHICLE = LIGHT_SLOTS.length

/** 灯体の大きさ [m]（幅 × 高さ × 奥行き相当） */
export const LIGHT_SIZE: readonly [number, number, number] = [0.12, 0.17, 0.2]

/**
 * 方向指示器の点滅周期 [Hz]。
 * 道路運送車両の保安基準では毎分 60〜120 回（＝ 1〜2Hz）と決まっている。
 */
export const BLINK_HZ = 1.5

/** 前照灯を点ける天候のしきい値。小雨（rain 0.35）から点く */
export const HEADLIGHT_RAIN = 0.25
export const HEADLIGHT_FOG = 0.1

/** 点滅の位相。true の間が点灯 */
export function blinkOn(nowMs: number): boolean {
  return ((nowMs * BLINK_HZ) / 1000) % 1 < 0.5
}

/**
 * 前照灯を点けるか。**夜と、視界が悪い天候で点ける**（道交法 52 条）。
 * 昼夜はフロント側にしか無い（`store/themeClock.ts` の日の出・日の入り）ので、
 * 天候とあわせてここで決める。
 */
export function headlightsOn(rain: number, fog: number, night: boolean): boolean {
  return night || rain >= HEADLIGHT_RAIN || fog >= HEADLIGHT_FOG
}

/** 1 台ぶんの点灯状態 */
export interface LightState {
  /** 前照灯 */
  head: boolean
  /** 尾灯（前照灯と連動する常時灯。制動灯とは別物） */
  tail: boolean
  /** 制動灯 */
  brake: boolean
  left: boolean
  right: boolean
}

export interface LightInput {
  braking: boolean
  /** -1=左 / 0=消灯 / +1=右 */
  turnSignal: number
  /** 乗降を待っている間は左右同時に点滅させる */
  hazard: boolean
  headlights: boolean
  /** `blinkOn()` の値 */
  blink: boolean
}

/** 入力から点灯状態を決める。**ハザードは方向指示器より優先する。** */
export function lightStateFor(input: LightInput): LightState {
  const blinking = input.blink
  return {
    head: input.headlights,
    tail: input.headlights,
    brake: input.braking,
    left: blinking && (input.hazard || input.turnSignal < 0),
    right: blinking && (input.hazard || input.turnSignal > 0),
  }
}

/** ライト 1 個の明るさ 0.0〜1.0。消灯でも尾灯はうっすら点く */
export function lightIntensity(slotIndex: number, state: LightState): number {
  const slot = LIGHT_SLOTS[slotIndex]
  if (slot.kind === LIGHT_HEAD) return state.head ? 1 : 0
  if (slot.kind === LIGHT_TAIL) {
    if (state.brake) return 1
    return state.tail ? 0.28 : 0
  }
  return (slot.side < 0 ? state.left : state.right) ? 1 : 0
}
