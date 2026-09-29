/** 階層型の方策の意図のダミー。実機と同じく 20 ステップ保ち、乱数は引かない。 */

import type { DriveOption, VehicleState } from '../../types/protocol.ts'

/** 実機の `config.HRL_OPTION_STEPS` と同じ */
export const MOCK_OPTION_STEPS = 20

/** 車の今の走りから、上位方策が選びそうな意図を決める */
export function guessOption(v: VehicleState): DriveOption {
  if (Math.abs(v.speed) < 0.5) return 'STOP'
  if (v.braking && Math.abs(v.speed) < 3) return 'YIELD'
  if (v.braking) return 'FOLLOW'
  return 'CRUISE'
}

/** 意図を 20 ステップごと（車ごとにずらす）に選び直し、その間は保つ */
export class MockOptions {
  private readonly held = new Map<number, DriveOption>()

  apply(vehicles: VehicleState[], tick: number): VehicleState[] {
    for (const v of vehicles) {
      if (!v.active) {
        this.held.delete(v.id)
        continue
      }
      let option = this.held.get(v.id)
      if (option === undefined || (tick + v.id * 3) % MOCK_OPTION_STEPS === 0) {
        option = guessOption(v)
        this.held.set(v.id, option)
      }
      v.currentOption = option
    }
    return vehicles
  }

  reset(): void {
    this.held.clear()
  }
}

/** 学習の進み具合から、意図の割合と加加速度のダミーを作る（進むほど停止が減り、滑らかになる） */
export function mockOptionMetrics(progress: number): { optionShares: number[]; jerkRms: number } {
  const p = Math.min(1, Math.max(0, progress))
  const raw = [0.42 + 0.18 * p, 0.2 + 0.05 * p, 0.14, 0.24 - 0.2 * p]
  const total = raw.reduce((a, b) => a + b, 0)
  return { optionShares: raw.map((v) => v / total), jerkRms: 9 - 6 * p }
}
