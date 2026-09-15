/** 車両の色。番号から色を決めるだけの純粋なモジュール。 */

/** 車両 64 台ぶんの色。 */
export const VEHICLE_COLORS = [
  '#9ecaff',
  '#7adcc8',
  '#d6bbfb',
  '#ffd08a',
  '#ff9ec0',
  '#a5e887',
  '#8181e4',
  '#ffc4a3',
  '#e1e481',
  '#e481e4',
  '#81e4a9',
  '#b588dd',
  '#a9d590',
  '#e481b6',
  '#a0e3b9',
  '#d590cc',
  '#d6eab8',
  '#8899dd',
  '#d5c790',
  '#a6aedd',
  '#e48b81',
  '#81dee4',
  '#e48195',
  '#81cae4',
  '#d5909c',
  '#bce6ce',
  '#efb3db',
  '#bcdfe6',
  '#d5a090',
  '#bcc9e6',
  '#e6d5bc',
  '#e6bcc7',
  '#55f155',
  '#c855f1',
  '#90f471',
  '#f155e2',
  '#55f18e',
  '#de71f4',
  '#c8f155',
  '#a171f4',
  '#a7e363',
  '#f155b8',
  '#55f1ae',
  '#f15589',
  '#55f1cd',
  '#f15555',
  '#55f1f1',
  '#f17455',
  '#5594f1',
  '#f1e755',
  '#55aef1',
  '#d2e363',
  '#e363a7',
  '#71d682',
  '#f8aaf8',
  '#f1c255',
  '#b094d1',
  '#d6c571',
  '#7ea8c8',
  '#f19455',
  '#e9d0f1',
  '#e36370',
  '#d6a771',
  '#e38563',
]

/** 黄金角。パレットを使い切ったときの予備の色をこの角度ずつ回して作る */
const GOLDEN_ANGLE = 137.508

/** 車両の色。番号ごとに違う色を返す。 */
export function vehicleColor(id: number): string {
  const index = Number.isFinite(id) ? Math.max(0, Math.floor(id)) : 0
  const palette = VEHICLE_COLORS[index]
  if (palette !== undefined) return palette

  const k = index - VEHICLE_COLORS.length
  const hue = (k * GOLDEN_ANGLE) % 360
  const lightness = 76 + ((k % 3) - 1) * 6
  return `hsl(${hue.toFixed(1)}, 55%, ${lightness}%)`
}
