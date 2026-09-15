/** 認識結果（Detection）から画面表示用の日本語ラベルと色を作る純粋関数。 */

import {
  DET_LANE,
  DET_OBSTACLE,
  DET_SPEED_SIGN,
  DET_TRAFFIC_LIGHT,
  DET_VEHICLE,
  type Detection,
} from '../types/protocol'
import { SIGNAL_LAMP_COLORS } from './palette'

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
    case DET_OBSTACLE:
      return '障害物'
    default:
      return '障害物'
  }
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
      return '#a479e8'
    case DET_VEHICLE:
      return '#4f9dff'
    case DET_LANE:
      return '#26c6da'
    default:
      return '#ff7a3d'
  }
}
