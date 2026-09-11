/**
 * 認識結果（Detection）から画面表示用の日本語ラベルと色を作る純粋関数。
 *
 * バックエンドは転送量を減らすため表示名の文字列を送らず、`cls` と属性だけを
 * 送ってくる（docs/protocol.md 2.3「detections」）。ここでの組み立て規則は
 * **`backend/app/percep/types.py` の `Detection.label` と同じにすること。**
 * 変えるときは両方直す（同じ検出結果に別の名前が付くと「画面の表示」と
 * 「学習の入力」が食い違って見える）。
 */

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

/**
 * 画面に出す日本語ラベル。**backend の `Detection.label` と同じ規則。**
 */
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
        // m/s -> km/h。backend は Python の round()（銀行丸め）を使うが、
        // 実在の規制速度が km/h でちょうど .5 に乗ることは無いため表示上は一致する
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
      // 未知のクラス。backend の DetClass は末尾追加のみの約束なので、
      // ここに来るのは未対応のクラスが増えたときだけ
      return '障害物'
  }
}

/**
 * ボックスの枠・ラベルに使う色。信号機は灯色で変える。
 *
 * 信号の灯色は `scene/palette.ts` の `SIGNAL_LAMP_COLORS` を使う
 * （灯器本体の描画と表示を同じ値で揃える、という palette.ts 側の約束）。
 * それ以外のクラスは昼夜で変えない固定色（各クラスの見分けが目的で、
 * 背景に対する視認性は `DetectionOverlay` 側の縁取りで確保する）。
 */
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
