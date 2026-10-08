/** 認識結果（Detection）から画面表示用の日本語ラベルと色を作る純粋関数。 */

import {
  DET_LANE,
  DET_OBSTACLE,
  DET_PEDESTRIAN,
  DET_SPEED_SIGN,
  DET_TRAFFIC_LIGHT,
  DET_VEHICLE,
  DET_STOP_SIGN,
  DET_CROSSWALK_SIGN,
  DET_ONE_WAY_SIGN,
  DET_MANDATORY_DIRECTION_SIGN,
  DET_NO_PARKING_SIGN,
  DET_NO_STOPPING_SIGN,
  type Detection,
} from '../types/protocol.ts'
import { SIGNAL_LAMP_COLORS } from './palette.ts'

/** 信号の灯色名。backend の `SIGNAL_PHASE_NAMES` と同じ並び（0=青 / 1=黄 / 2=赤） */
const SIGNAL_PHASE_NAMES = ['青', '黄', '赤'] as const

/** 灯色が無い・範囲外のときの枠色（グレー） */
const UNKNOWN_SIGNAL_COLOR = '#9aa5b1'

/** 画面に出す日本語ラベル。**backend の `Detection.label` と同じ規則。** */
export function detectionLabel(det: Detection): string {
  switch (det.cls) {
    case DET_TRAFFIC_LIGHT: {
      const phase = det.phase
      if (phase !== undefined && phase >= 0 && phase < SIGNAL_PHASE_NAMES.length) {
        return `信号機：${SIGNAL_PHASE_NAMES[phase]}`
      }
      return '信号機'
    }
    case DET_SPEED_SIGN: {
      if (det.speedLimit !== undefined) {
        return `速度標識：${Math.round(det.speedLimit * 3.6)}km/h`
      }
      return '速度標識'
    }
    case DET_LANE:
      return '車線'
    case DET_VEHICLE:
      return '車両'
    case DET_PEDESTRIAN:
      return '歩行者'
    case DET_OBSTACLE:
      return '障害物'
    case DET_STOP_SIGN:
      return '一時停止'
    case DET_CROSSWALK_SIGN:
      return '横断歩道'
    case DET_ONE_WAY_SIGN:
      return '一方通行'
    case DET_MANDATORY_DIRECTION_SIGN:
      return '指定方向外進行禁止'
    case DET_NO_PARKING_SIGN:
      return '駐車禁止'
    case DET_NO_STOPPING_SIGN:
      return '駐停車禁止'
    default:
      return '障害物'
  }
}

/** 安全ギミックが介入の根拠にしている検出の色。1 = 注意（黄）/ 2 = これで止めている（赤） */
export const HAZARD_COLORS = { 1: '#ffa000', 2: '#ff3b30' } as const

/** 枠の色。安全ギミックの根拠になっている検出はクラスの色より危険度の色を優先する */
export function detectionFrameColor(det: Detection): string {
  if (det.hazard === 1 || det.hazard === 2) return HAZARD_COLORS[det.hazard]
  return detectionColor(det)
}

/** ボックスの枠・ラベルに使う色。信号機は灯色で変える。 */
export function detectionColor(det: Detection): string {
  if (det.cls === DET_TRAFFIC_LIGHT) {
    const phase = det.phase
    if (phase !== undefined && phase >= 0 && phase < SIGNAL_LAMP_COLORS.length) {
      return SIGNAL_LAMP_COLORS[phase]
    }
    return UNKNOWN_SIGNAL_COLOR
  }
  switch (det.cls) {
    case DET_SPEED_SIGN:
    case DET_STOP_SIGN:
    case DET_CROSSWALK_SIGN:
    case DET_ONE_WAY_SIGN:
    case DET_MANDATORY_DIRECTION_SIGN:
    case DET_NO_PARKING_SIGN:
    case DET_NO_STOPPING_SIGN:
      return '#a479e8'
    case DET_VEHICLE:
      return '#4f9dff'
    case DET_LANE:
      return '#26c6da'
    case DET_PEDESTRIAN:
      return '#ffd24a'
    default:
      return '#ff7a3d'
  }
}
